"""Build, validate, and reuse encoded-corpus artifacts."""

from __future__ import annotations

# Orchestrates the encoded-corpus stage from the source root.

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator

from src.persistence.artifact_io import write_manifest
from src.persistence.artifact_validation import validate_encoded_corpus_directory
from src.records import ChunkRecord, PageRecord
from src.provenance import (
    encoded_corpus_identity,
    source_group_sha256,
)
from src.encoded_corpus_factory import (
    create_chunker,
    create_embedder,
    create_token_counter,
)
from src.persistence.encoded_corpus_writer import write_encoded_corpus


@dataclass
class _PageStats:
    num_pages: int = 0
    document_ids: set[str] = field(default_factory=set)
    dpr_documents: bool = False

    @property
    def num_documents(self) -> int:
        return self.num_pages if self.dpr_documents else len(self.document_ids)

    def observe(self, page: PageRecord) -> None:
        self.num_pages += 1
        if not self.dpr_documents:
            self.document_ids.add(page.doc_id)


@dataclass
class _ChunkStats:
    count: int = 0
    token_min: int | None = None
    token_max: int = 0
    token_total: int = 0
    overlap_values: list[int] = field(default_factory=list)
    previous: ChunkRecord | None = None

    def observe(
        self,
        chunk: ChunkRecord,
        token_counter: Any,
        token_budget: int | None,
    ) -> None:
        if chunk.vector_id != self.count:
            raise RuntimeError("Chunk vector ids must be the zero-based sequence")
        if chunk.token_count <= 0 or (
            token_budget is not None and chunk.token_count > token_budget
        ):
            raise RuntimeError(
                "Chunk token counts must be positive and within the configured budget"
            )
        self.count += 1
        self.token_min = (
            chunk.token_count
            if self.token_min is None
            else min(self.token_min, chunk.token_count)
        )
        self.token_max = max(self.token_max, chunk.token_count)
        self.token_total += chunk.token_count

        if self.previous is not None and self.previous.doc_id == chunk.doc_id:
            left = token_counter.token_sequence(self.previous.text)
            right = token_counter.token_sequence(chunk.text)
            limit = min(len(left), len(right))
            overlap = 0
            for size in range(1, limit + 1):
                if left[-size:] == right[:size]:
                    overlap = size
            self.overlap_values.append(overlap)
        self.previous = chunk

    def token_summary(self) -> dict[str, float | int]:
        if self.count <= 0 or self.token_min is None:
            raise RuntimeError("No chunks were produced from the corpus")
        return {
            "min": self.token_min,
            "mean": self.token_total / self.count,
            "max": self.token_max,
        }

    def overlap_summary(self) -> dict[str, float | int]:
        values = self.overlap_values
        if not values:
            return {
                "pairs": 0,
                "min": 0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0,
                "zero_count": 0,
            }
        return {
            "pairs": len(values),
            "min": min(values),
            "mean": mean(values),
            "median": median(values),
            "max": max(values),
            "zero_count": sum(value == 0 for value in values),
        }


def _coerce_page(value: PageRecord | dict[str, Any]) -> PageRecord:
    return value if isinstance(value, PageRecord) else PageRecord.from_mapping(value)


def _coerce_chunk(value: ChunkRecord | dict[str, Any]) -> ChunkRecord:
    return value if isinstance(value, ChunkRecord) else ChunkRecord.from_mapping(value)


def _tracked_pages(
    values: Iterable[PageRecord | dict[str, Any]],
    stats: _PageStats,
) -> Iterator[PageRecord]:
    for value in values:
        page = _coerce_page(value)
        stats.observe(page)
        yield page


def _tracked_chunks(
    values: Iterable[ChunkRecord | dict[str, Any]],
    stats: _ChunkStats,
    token_counter: Any,
    token_budget: int | None,
) -> Iterator[ChunkRecord]:
    for value in values:
        chunk = _coerce_chunk(value)
        stats.observe(chunk, token_counter, token_budget)
        yield chunk


def _encoded_corpus_inputs(
    config: dict[str, Any],
    loader: Any,
    corpus_path: Path,
) -> tuple[Any, Any, _PageStats, _ChunkStats, Iterator[ChunkRecord]]:
    token_counter = create_token_counter(config)
    chunker = create_chunker(config, token_counter)
    page_stats = _PageStats(dpr_documents=config["loader"]["type"] == "dpr_wikipedia")
    chunk_stats = _ChunkStats()

    if config["chunking"]["strategy"] == "presegmented":
        page_values = (
            loader.iter_pages(corpus_path)
            if callable(getattr(loader, "iter_pages", None))
            else iter(loader.load(corpus_path))
        )
        pages = _tracked_pages(page_values, page_stats)
        chunk_values = (
            chunker.iter_chunks(pages)
            if callable(getattr(chunker, "iter_chunks", None))
            else iter(chunker.chunk(list(pages)))
        )
    else:
        loaded_pages = [_coerce_page(value) for value in loader.load(corpus_path)]
        if not loaded_pages:
            raise RuntimeError("The corpus produced no extractable text records")
        for page in loaded_pages:
            page_stats.observe(page)
        chunk_values = iter(chunker.chunk(loaded_pages))

    chunks = _tracked_chunks(
        chunk_values,
        chunk_stats,
        token_counter,
        (
            None
            if config["chunking"]["strategy"] == "presegmented"
            else config["chunking"]["chunk_size_tokens"]
        ),
    )
    return token_counter, chunker, page_stats, chunk_stats, chunks


def build_or_reuse_encoded_corpus(
    config: dict[str, Any],
    loader: Any,
    corpus_path: Path,
    artifacts_root: Path,
    corpus: dict[str, Any],
    *,
    project_root: str | Path,
) -> tuple[Path, dict[str, Any], bool, float]:
    """Return one validated encoded corpus, building it atomically when absent."""

    source_sha = source_group_sha256(project_root, "encoded_corpus")
    encoded_corpus_id, spec_sha, spec = encoded_corpus_identity(
        config,
        corpus,
        source_sha,
    )
    cache_root = artifacts_root / "_encoded_corpora"
    encoded_corpus_dir = cache_root / encoded_corpus_id
    started = time.perf_counter()
    if encoded_corpus_dir.exists():
        return (
            encoded_corpus_dir,
            validate_encoded_corpus_directory(
                encoded_corpus_dir,
                encoded_corpus_id,
                spec_sha,
            ).manifest,
            True,
            (time.perf_counter() - started) * 1000,
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{encoded_corpus_id}-",
            dir=cache_root,
        )
    )
    try:
        token_counter, _, page_stats, chunk_stats, chunks = _encoded_corpus_inputs(
            config,
            loader,
            corpus_path,
        )
        embedder = create_embedder(config)
        if not isinstance(getattr(embedder, "dimension", None), int):
            configured_dimension = config["embedding"].get("dimension")
            if not isinstance(configured_dimension, int):
                raise RuntimeError("Embedding backend did not expose its dimension")
            setattr(embedder, "dimension", configured_dimension)
        artifact = write_encoded_corpus(
            chunks,
            embedder,
            staging,
            batch_size=config["embedding"].get("batch_size", 128),
        )
        if page_stats.num_pages <= 0:
            raise RuntimeError("The corpus produced no extractable text records")
        if chunk_stats.count != artifact.rows:
            raise RuntimeError(
                "Encoded-corpus rows do not match streamed chunk statistics"
            )
        manifest = {
            "status": "complete",
            "encoded_corpus_id": encoded_corpus_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "encoded_corpus_spec_sha256": spec_sha,
            "encoded_corpus_spec": spec,
            "corpus": {
                "num_pages": page_stats.num_pages,
                "num_documents": page_stats.num_documents,
            },
            "chunking": {
                "num_chunks": chunk_stats.count,
                "token_count": chunk_stats.token_summary(),
                "realized_overlap_tokens": chunk_stats.overlap_summary(),
            },
            "embedding": {
                "space": embedder.embedding_space("inner_product").to_dict()
            },
            "artifacts": artifact.artifact_descriptors(),
        }
        write_manifest(staging / "manifest.json", manifest)
        try:
            os.replace(staging, encoded_corpus_dir)
        except OSError:
            if not encoded_corpus_dir.exists():
                raise
            validated = validate_encoded_corpus_directory(
                encoded_corpus_dir,
                encoded_corpus_id,
                spec_sha,
            ).manifest
        else:
            validated = manifest
        return (
            encoded_corpus_dir,
            validated,
            False,
            (time.perf_counter() - started) * 1000,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
