"""Build, identify, validate, and resolve immutable BM25S artifacts."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import validate_config
from src.persistence.artifact_io import describe_artifact, iter_jsonl, write_manifest
from src.persistence.artifact_validation import (
    VerifiedBuild,
    VerifiedSparseIndex,
    validate_bm25_index_directory,
)
from src.persistence.beir_artifact_registry import resolve_pinned_beir_build
from src.provenance import (
    build_identity,
    json_sha256,
    producer_environment,
    resolved_roots,
    sha256_file,
    source_group_sha256,
)
from src.records import ChunkRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INDEX_FILES = {
    "data": "data.csc.index.npy",
    "indices": "indices.csc.index.npy",
    "indptr": "indptr.csc.index.npy",
    "index_vocab": "vocab.index.json",
    "params": "params.index.json",
    "tokenizer_vocab": "vocab.tokenizer.json",
    "tokenizer_stopwords": "stopwords.tokenizer.json",
}


def _bm25s_version() -> str:
    try:
        return importlib.metadata.version("bm25s")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "BM25 indexing requires bm25s; install requirements/experiment.txt"
        ) from exc


def bm25_index_spec(
    config: dict[str, Any],
    verified_build: VerifiedBuild,
    *,
    sparse_index_source_sha256: str,
    bm25s_version: str,
) -> dict[str, Any]:
    """Describe the BM25 arrays independently of query-only mmap selection."""

    bm25 = dict(config["bm25"])
    bm25.pop("mmap", None)
    chunks = verified_build.manifest["artifacts"]["chunks"]
    return {
        "backend": "bm25s",
        "bm25": bm25,
        "bm25s_version": bm25s_version,
        "source_build_id": verified_build.manifest["build_id"],
        "source_chunks": {
            key: chunks[key]
            for key in ("file", "size_bytes", "sha256", "rows")
        },
        "producer_environment": producer_environment("bm25s", "numpy"),
        "sparse_index_source_sha256": sparse_index_source_sha256,
    }


def bm25_index_identity(
    config: dict[str, Any],
    verified_build: VerifiedBuild,
) -> tuple[str, str, dict[str, Any]]:
    source_sha = source_group_sha256(PROJECT_ROOT, "sparse_index")
    spec = bm25_index_spec(
        config,
        verified_build,
        sparse_index_source_sha256=source_sha,
        bm25s_version=_bm25s_version(),
    )
    digest = json_sha256(spec)
    return f"bm25_{digest[:16]}", digest, spec


def expected_bm25_index_directory(
    config: dict[str, Any],
    verified_build: VerifiedBuild,
) -> tuple[Path, str, str, dict[str, Any]]:
    config = validate_config(config)
    if config["retrieval"]["method"] != "bm25":
        raise ValueError("BM25 index resolution requires retrieval.method=bm25")
    if not isinstance(verified_build, VerifiedBuild):
        raise TypeError("verified_build must be a VerifiedBuild")
    if config["bm25"]["backend"] == "sqlite":
        from src.retrievers.sqlite_bm25 import sqlite_bm25_identity

        sparse_id, spec_sha, spec = sqlite_bm25_identity(
            verified_build,
            k1=config["bm25"]["k1"],
            b=config["bm25"]["b"],
            analyzer=config["bm25"]["analyzer"],
        )
    else:
        sparse_id, spec_sha, spec = bm25_index_identity(config, verified_build)
    root = resolved_roots(config)["artifacts_root"] / "_sparse_indexes"
    return (root / sparse_id).resolve(), sparse_id, spec_sha, spec


def resolve_bm25_index(
    config: dict[str, Any],
    verified_build: VerifiedBuild,
) -> VerifiedSparseIndex:
    """Resolve an existing BM25 artifact without building during query startup."""

    directory, sparse_id, _, spec = expected_bm25_index_directory(
        config,
        verified_build,
    )
    if not directory.is_dir():
        raise FileNotFoundError(
            "BM25 index does not exist for the active chunks/config; run "
            "scripts/build_bm25_index.py first: "
            f"{directory}"
        )
    if config["bm25"]["backend"] == "sqlite":
        from src.retrievers.sqlite_bm25 import validate_sqlite_bm25_index

        return validate_sqlite_bm25_index(
            directory,
            expected_identity_sha256=json_sha256(spec),
        )
    return validate_bm25_index_directory(directory, sparse_id, expected_spec=spec)


def _chunk_texts(path: Path, expected_rows: int) -> Iterator[str]:
    count = 0
    for vector_id, row in enumerate(iter_jsonl(path)):
        record = ChunkRecord.from_mapping(row)
        if record.vector_id != vector_id:
            raise ValueError(
                f"Chunk vector id {record.vector_id} does not match row {vector_id}"
            )
        count += 1
        yield record.text
    if count != expected_rows:
        raise ValueError(
            f"Chunk row count {count} does not match manifest rows {expected_rows}"
        )


def _commit(
    staging: Path,
    destination: Path,
    sparse_id: str,
    spec_sha: str,
    spec: dict[str, Any],
) -> VerifiedSparseIndex:
    if destination.exists():
        existing = validate_bm25_index_directory(
            destination,
            sparse_id,
            expected_spec=spec,
        )
        if existing.manifest.get("sparse_index_spec_sha256") != spec_sha:
            raise RuntimeError("Concurrent BM25 build produced an incompatible artifact")
        return existing
    try:
        os.replace(staging, destination)
    except OSError:
        if not destination.exists():
            raise
        existing = validate_bm25_index_directory(
            destination,
            sparse_id,
            expected_spec=spec,
        )
        if existing.manifest.get("sparse_index_spec_sha256") != spec_sha:
            raise RuntimeError("Concurrent BM25 build produced an incompatible artifact")
        return existing
    return validate_bm25_index_directory(
        destination,
        sparse_id,
        expected_spec=spec,
    )


def build_bm25_index(
    config: dict[str, Any],
    verified_build: VerifiedBuild,
    *,
    show_progress: bool = True,
) -> VerifiedSparseIndex:
    """Build or reuse a BM25S artifact over a verified build's chunk rows."""

    config = validate_config(config)
    if config["bm25"]["backend"] == "sqlite":
        corpus = verified_build.manifest.get("build_spec", {}).get("corpus")
        if isinstance(corpus, dict):
            _, _, current_build_spec = build_identity(
                config,
                dict(corpus),
                source_group_sha256(PROJECT_ROOT, "build"),
            )
            pinned = resolve_pinned_beir_build(
                config=config,
                corpus=corpus,
                current_build_spec=current_build_spec,
                artifacts_root=resolved_roots(config)["artifacts_root"],
            )
            if (
                pinned is not None
                and pinned.verified_build.directory.resolve()
                == verified_build.directory.resolve()
            ):
                from src.retrievers.sqlite_bm25 import validate_sqlite_bm25_index

                return validate_sqlite_bm25_index(pinned.sqlite_bm25_directory)
    directory, sparse_id, spec_sha, spec = expected_bm25_index_directory(
        config,
        verified_build,
    )
    if directory.exists():
        if config["bm25"]["backend"] == "sqlite":
            from src.retrievers.sqlite_bm25 import (
                build_sqlite_bm25_index,
                validate_sqlite_bm25_index,
            )

            if (directory / "manifest.json").is_file():
                return validate_sqlite_bm25_index(
                    directory,
                    expected_identity_sha256=spec_sha,
                )
            return build_sqlite_bm25_index(
                verified_build,
                directory,
                k1=config["bm25"]["k1"],
                b=config["bm25"]["b"],
                analyzer=config["bm25"]["analyzer"],
                transaction_documents=config["bm25"]["transaction_documents"],
            )

        return validate_bm25_index_directory(directory, sparse_id, expected_spec=spec)

    if config["bm25"]["backend"] == "sqlite":
        from src.retrievers.sqlite_bm25 import build_sqlite_bm25_index

        return build_sqlite_bm25_index(
            verified_build,
            directory,
            k1=config["bm25"]["k1"],
            b=config["bm25"]["b"],
            analyzer=config["bm25"]["analyzer"],
            transaction_documents=config["bm25"]["transaction_documents"],
        )

    try:
        import bm25s
        from bm25s.tokenization import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BM25 indexing requires bm25s; install requirements/experiment.txt"
        ) from exc

    root = directory.parent
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{sparse_id}-", dir=root))
    started = time.perf_counter()
    try:
        rows = int(spec["source_chunks"]["rows"])
        chunks_path = verified_build.files["chunks"]
        tokenizer = Tokenizer(lower=True, stopwords="english")

        tokenization_started = time.perf_counter()
        corpus_tokens = tokenizer.tokenize(
            _chunk_texts(chunks_path, rows),
            length=rows,
            return_as="tuple",
            show_progress=show_progress,
        )
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000

        index_started = time.perf_counter()
        model = bm25s.BM25(
            method=config["bm25"]["method"],
            k1=config["bm25"]["k1"],
            b=config["bm25"]["b"],
            backend="numpy",
            csc_backend="numpy",
        )
        model.index(corpus_tokens, show_progress=show_progress)
        if model.scores.get("num_docs") != rows:
            raise ValueError("Built BM25 document count does not match source chunks")
        index_ms = (time.perf_counter() - index_started) * 1000

        save_started = time.perf_counter()
        model.save(staging, show_progress=show_progress)
        tokenizer.save_vocab(staging)
        tokenizer.save_stopwords(staging)
        save_ms = (time.perf_counter() - save_started) * 1000

        if sha256_file(chunks_path) != spec["source_chunks"]["sha256"]:
            raise RuntimeError("Source chunks changed while the BM25 index was being built")
        artifacts = {
            name: describe_artifact(staging / file_name)
            for name, file_name in _INDEX_FILES.items()
        }
        manifest = {
            "status": "complete",
            "sparse_index_id": sparse_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sparse_index_spec_sha256": spec_sha,
            "sparse_index_spec": spec,
            "source_build_id": verified_build.manifest["build_id"],
            "document_count": rows,
            "artifacts": artifacts,
            "timings_ms": {
                "tokenization": tokenization_ms,
                "index_build": index_ms,
                "save": save_ms,
                "total_before_commit": (time.perf_counter() - started) * 1000,
            },
        }
        write_manifest(staging / "manifest.json", manifest)
        validate_bm25_index_directory(staging, sparse_id, expected_spec=spec)
        return _commit(staging, directory, sparse_id, spec_sha, spec)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
