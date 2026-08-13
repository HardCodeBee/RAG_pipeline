"""Build, validate, and reuse encoded-corpus artifacts."""

from __future__ import annotations

# Orchestrates the encoded-corpus stage from the source root.

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, Iterator

from src.persistence.artifact_io import write_manifest
from src.persistence.artifact_validation import validate_encoded_corpus_directory
from src.records import ChunkRecord, PageRecord
from src.embedders.text_embedder import create_embedder
from src.provenance import (
    encoded_corpus_identity,
    source_group_sha256,
)
from src.persistence.encoded_corpus_writer import write_encoded_corpus
from src.text.token_counters import RegexTokenCounter


@dataclass
class _ChunkStats:
    count: int = 0
    token_min: int | None = None
    token_max: int = 0
    token_total: int = 0

    def observe(self, chunk: ChunkRecord) -> None:
        if chunk.vector_id != self.count:
            raise RuntimeError("Chunk vector ids must be the zero-based sequence")
        if chunk.token_count <= 0:
            raise RuntimeError("Chunk token counts must be positive")
        self.count += 1
        self.token_min = (
            chunk.token_count
            if self.token_min is None
            else min(self.token_min, chunk.token_count)
        )
        self.token_max = max(self.token_max, chunk.token_count)
        self.token_total += chunk.token_count

    def token_summary(self) -> dict[str, float | int]:
        if self.count <= 0 or self.token_min is None:
            raise RuntimeError("No chunks were produced from the corpus")
        return {
            "min": self.token_min,
            "mean": self.token_total / self.count,
            "max": self.token_max,
        }

def _tracked_chunks(
    values: Iterable[ChunkRecord],
    stats: _ChunkStats,
) -> Iterator[ChunkRecord]:
    for chunk in values:
        stats.observe(chunk)
        yield chunk


def _presegmented_chunks(
    records: Iterable[PageRecord | Mapping[str, Any]],
    token_counter: Any,
) -> Iterator[ChunkRecord]:
    for vector_id, value in enumerate(records):
        record = value if isinstance(value, PageRecord) else PageRecord.from_mapping(value)
        token_count = token_counter.count(record.text)
        if token_count <= 0:
            raise ValueError(f"Pre-segmented chunk has no tokens: {record.doc_id}")
        yield ChunkRecord(
            chunk_id=record.doc_id,
            vector_id=vector_id,
            doc_id=record.doc_id,
            source=record.source,
            page_start=record.page,
            page_end=record.page,
            text=record.text,
            token_count=token_count,
        )


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
    # A stable partial directory lets individually committed embedding shards
    # survive a process restart. Its identity already includes corpus, model,
    # chunking, storage layout, environment, and source fingerprints.
    staging = cache_root / f".{encoded_corpus_id}.partial"
    if staging.exists() and not staging.is_dir():
        raise FileExistsError(f"Encoded-corpus recovery path is not a directory: {staging}")
    staging.mkdir(exist_ok=True)
    cleanup_staging = False

    # Recover the narrow crash window after the complete manifest was written
    # but before the partial directory was atomically renamed.
    if (staging / "manifest.json").is_file():
        recovered = validate_encoded_corpus_directory(
            staging,
            encoded_corpus_id,
            spec_sha,
        ).manifest
        try:
            os.replace(staging, encoded_corpus_dir)
        except OSError:
            if not encoded_corpus_dir.exists():
                raise
            recovered = validate_encoded_corpus_directory(
                encoded_corpus_dir,
                encoded_corpus_id,
                spec_sha,
            ).manifest
            cleanup_staging = True
        if cleanup_staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        return (
            encoded_corpus_dir,
            recovered,
            False,
            (time.perf_counter() - started) * 1000,
        )
    try:
        chunk_stats = _ChunkStats()
        chunks = _tracked_chunks(
            _presegmented_chunks(loader.iter_pages(corpus_path), RegexTokenCounter()),
            chunk_stats,
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
            encode_call_rows=config["embedding"].get(
                "encode_call_rows",
                config["embedding"].get("batch_size", 128),
            ),
            embedding_shard_rows=config["embedding"].get("shard_rows"),
        )
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
            "chunking": {
                "token_count": chunk_stats.token_summary(),
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
            cleanup_staging = True
        else:
            validated = manifest
        return (
            encoded_corpus_dir,
            validated,
            False,
            (time.perf_counter() - started) * 1000,
        )
    finally:
        # Preserve valid part checkpoints after failure. Only a concurrent
        # winner makes this process's partial directory redundant.
        if cleanup_staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
