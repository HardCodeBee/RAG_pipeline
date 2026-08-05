"""Build immutable vector indexes from a reusable encoded corpus."""

from __future__ import annotations

# Structural orchestration for immutable index builds.

import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.artifact_io import close_numpy_memmap, describe_artifact, write_manifest
from src.persistence.artifact_validation import VerifiedBuild, validate_build_directory
from src.config import validate_config
from src.vector_index_factory import create_index
from src.provenance import (
    build_identity,
    corpus_inventory,
    environment_versions,
    git_state,
    resolved_roots,
    source_group_sha256,
    source_snapshot_sha256,
    zero_based_sequence_sha256,
)
from src.encoded_corpus_factory import create_loader, discover_corpus
from src.encoded_corpus_builder import build_or_reuse_encoded_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_encoded_corpus(
    encoded_corpus_dir: Path,
    encoded_corpus_manifest: dict[str, Any],
    staging: Path,
) -> tuple[Path, Path, Path]:
    output: list[Path] = []
    for name in ("chunks", "chunk_offsets", "embeddings"):
        descriptor = encoded_corpus_manifest["artifacts"][name]
        source = encoded_corpus_dir / descriptor["file"]
        destination = staging / descriptor["file"]
        _link_or_copy(source, destination)
        output.append(destination)
    return output[0], output[1], output[2]


def _training_sample(
    embeddings: np.ndarray,
    *,
    train_size: int,
    seed: int,
) -> np.ndarray:
    rows = embeddings.shape[0]
    sample_size = min(rows, train_size)
    if sample_size == rows:
        return np.ascontiguousarray(embeddings, dtype=np.float32)
    generator = np.random.default_rng(seed)
    positions = np.sort(generator.choice(rows, size=sample_size, replace=False))
    return np.ascontiguousarray(embeddings[positions], dtype=np.float32)


def _build_index_artifact(
    config: dict[str, Any],
    embeddings_path: Path,
    staging: Path,
) -> tuple[Path | None, float]:
    started = time.perf_counter()
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    try:
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] <= 0
            or embeddings.shape[1] <= 0
            or embeddings.dtype != np.dtype("float32")
        ):
            raise RuntimeError("Embedding artifact has an invalid shape or dtype")
        rows = int(embeddings.shape[0])
        index = create_index(config)
        if config["index"]["type"] in {"ivf_flat", "ivf_pq"}:
            training = _training_sample(
                embeddings,
                train_size=config["index"]["train_size"],
                seed=config["index"]["train_seed"],
            )
            index.train(training)
            del training
        else:
            index.train(np.ascontiguousarray(embeddings[0:1], dtype=np.float32))

        batch_size = config["index"]["build_batch_size"]
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            batch = np.ascontiguousarray(embeddings[start:end], dtype=np.float32)
            ids = np.arange(start, end, dtype=np.int64)
            index.add_with_ids(batch, ids)

        index_path = staging / "index.faiss" if index.backend == "faiss" else None
        if index_path is not None:
            index.save(index_path)
        elapsed_ms = (time.perf_counter() - started) * 1000
        del index
        return index_path, elapsed_ms
    finally:
        close_numpy_memmap(embeddings)


def _verify_exact_index(index: Any, embeddings: np.ndarray) -> None:
    query = np.ascontiguousarray(embeddings[0:1], dtype=np.float32)
    top_k = min(10, int(embeddings.shape[0]))
    expected_scores = np.asarray(embeddings @ query[0], dtype=np.float32)
    vector_ids = np.arange(embeddings.shape[0], dtype=np.int64)
    positions = np.lexsort((vector_ids, -expected_scores))[:top_k]
    hits = index.search_hits(query, top_k)
    if [hit.vector_id for hit in hits] != vector_ids[positions].tolist():
        raise RuntimeError("Exact index search does not match the NumPy reference")
    if not np.allclose(
        np.asarray([hit.score for hit in hits], dtype=np.float32),
        expected_scores[positions],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("Exact index scores do not match the NumPy reference")


def _verify_saved_index(
    config: dict[str, Any],
    index_path: Path | None,
    embeddings_path: Path,
) -> Any:
    verified = create_index(
        config,
        backend=config["index"]["backend"],
        index_type=config["index"]["type"],
    )
    expected_build_params = verified.build_params
    verified.load(index_path if index_path is not None else embeddings_path)
    if verified.build_params != expected_build_params:
        raise RuntimeError(
            "Reloaded index build parameters do not match the requested build specification"
        )
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    try:
        rows, dimension = map(int, embeddings.shape)
        if verified.count != rows or verified.dimension != dimension:
            raise RuntimeError("Reloaded index metadata does not match the encoded corpus")
        if verified.ids is None or not np.array_equal(
            verified.ids, np.arange(rows, dtype=np.int64)
        ):
            raise RuntimeError("Reloaded index vector ids do not match the encoded corpus")
        if config["index"]["type"] == "flat_ip":
            _verify_exact_index(verified, embeddings)
        else:
            hits = verified.search_hits(
                np.ascontiguousarray(embeddings[0:1], dtype=np.float32),
                min(10, rows),
            )
            if not hits or len({hit.vector_id for hit in hits}) != len(hits):
                raise RuntimeError("Reloaded ANN index returned invalid vector ids")
        return verified
    finally:
        close_numpy_memmap(embeddings)


def _ensure_corpus_unchanged(
    loader: Any,
    corpus_path: Path,
    expected_corpus: dict[str, Any],
) -> None:
    current_documents = loader.discover(corpus_path)
    current = corpus_inventory(current_documents, corpus_path)
    if current != expected_corpus:
        raise RuntimeError("Corpus changed while the encoded corpus or index was being built")


def _create_manifest(
    *,
    build_id: str,
    build_spec_sha: str,
    spec: dict[str, Any],
    source_snapshot_sha: str,
    documents: list[Path],
    encoded_corpus_manifest: dict[str, Any],
    index: Any,
    chunks_path: Path,
    offsets_path: Path,
    embeddings_path: Path,
    index_path: Path | None,
    timings: dict[str, float],
    started: float,
) -> dict[str, Any]:
    rows = int(encoded_corpus_manifest["artifacts"]["chunks"]["rows"])
    artifacts = {
        name: dict(encoded_corpus_manifest["artifacts"][name])
        for name in ("chunks", "chunk_offsets", "embeddings")
    }
    if index_path is not None:
        artifacts["index"] = describe_artifact(index_path)
    return {
        "status": "complete",
        "build_id": build_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_spec_sha256": build_spec_sha,
        "build_spec": spec,
        "source_snapshot_sha256": source_snapshot_sha,
        "corpus": {
            "num_files": len(documents),
            **encoded_corpus_manifest["corpus"],
        },
        "chunking": encoded_corpus_manifest["chunking"],
        "embedding": encoded_corpus_manifest["embedding"],
        "index": {
            "backend": index.backend,
            "type": index.index_type,
            "count": index.count,
            "dimension": index.dimension,
            "build_params": index.build_params,
        },
        "vector_id_sequence_sha256": zero_based_sequence_sha256(rows),
        "artifacts": artifacts,
        "git": git_state(PROJECT_ROOT),
        "environment": environment_versions(),
        "timings_ms": {
            **timings,
            "total_before_commit": (time.perf_counter() - started) * 1000,
        },
    }


def _commit_build(
    staging: Path,
    build_dir: Path,
    build_id: str,
    build_spec_sha: str,
    manifest: dict[str, Any],
) -> VerifiedBuild:
    if build_dir.exists():
        existing = validate_build_directory(build_dir, build_id)
        if existing.manifest.get("build_spec_sha256") != build_spec_sha:
            raise RuntimeError("Concurrent build produced an incompatible build directory")
        return existing
    try:
        os.replace(staging, build_dir)
    except OSError:
        if not build_dir.exists():
            raise
        existing = validate_build_directory(build_dir, build_id)
        if existing.manifest.get("build_spec_sha256") != build_spec_sha:
            raise RuntimeError("Concurrent build produced an incompatible build directory")
        return existing
    files = {
        name: (build_dir / descriptor["file"]).resolve()
        for name, descriptor in manifest["artifacts"].items()
    }
    return VerifiedBuild(
        directory=build_dir.resolve(),
        manifest=manifest,
        files=files,
    )


def build_index(config: dict[str, Any]) -> VerifiedBuild:
    """Build or reuse encoded-corpus data, then build one immutable vector index."""

    config = validate_config(config)
    if "_base_dir" not in config:
        raise ValueError("config must include _base_dir; use load_config()")
    roots = resolved_roots(config)
    loader = create_loader(config)
    documents, corpus = discover_corpus(loader, roots["corpus"])
    if not documents:
        raise RuntimeError(f"No corpus files found in corpus path: {roots['corpus']}")
    build_source_sha = source_group_sha256(PROJECT_ROOT, "build")
    source_snapshot_sha = source_snapshot_sha256(PROJECT_ROOT)
    build_id, build_spec_sha, spec = build_identity(config, corpus, build_source_sha)
    artifacts_root = roots["artifacts_root"]
    build_dir = artifacts_root / build_id
    if build_dir.exists():
        verified = validate_build_directory(build_dir, build_id)
        if verified.manifest.get("build_spec_sha256") != build_spec_sha or verified.manifest.get(
            "build_spec"
        ) != spec:
            raise ValueError("Existing build directory does not match the requested build spec")
        return verified

    artifacts_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoded_corpus_dir, encoded_corpus_manifest, _, encoded_corpus_ms = (
        build_or_reuse_encoded_corpus(
            config,
            loader,
            roots["corpus"],
            artifacts_root,
            corpus,
            project_root=PROJECT_ROOT,
        )
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}-", dir=artifacts_root))
    try:
        chunks_path, offsets_path, embeddings_path = _materialize_encoded_corpus(
            encoded_corpus_dir,
            encoded_corpus_manifest,
            staging,
        )
        index_path, index_ms = _build_index_artifact(
            config,
            embeddings_path,
            staging,
        )
        index = _verify_saved_index(config, index_path, embeddings_path)
        _ensure_corpus_unchanged(loader, roots["corpus"], corpus)
        manifest = _create_manifest(
            build_id=build_id,
            build_spec_sha=build_spec_sha,
            spec=spec,
            source_snapshot_sha=source_snapshot_sha,
            documents=documents,
            encoded_corpus_manifest=encoded_corpus_manifest,
            index=index,
            chunks_path=chunks_path,
            offsets_path=offsets_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
            timings={
                "encoded_corpus_build_or_validation": encoded_corpus_ms,
                "index_build_and_save": index_ms,
            },
            started=started,
        )
        write_manifest(staging / "manifest.json", manifest)
        return _commit_build(
            staging,
            build_dir,
            build_id,
            build_spec_sha,
            manifest,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
