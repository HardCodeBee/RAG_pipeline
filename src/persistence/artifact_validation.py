"""Single trust-boundary validator for persisted pipeline artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.artifact_io import close_numpy_memmap, read_json_object
from src.provenance import json_sha256, sha256_file
from src.records import EmbeddingSpaceSpec


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """An artifact path whose descriptor was checked in this process."""

    path: Path
    descriptor: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedBuild:
    """A complete build and its already-verified artifact paths."""

    directory: Path
    manifest: dict[str, Any]
    files: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class VerifiedEncodedCorpus:
    """A reusable encoded corpus verified at a disk trust boundary."""

    directory: Path
    manifest: dict[str, Any]
    files: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class VerifiedSparseIndex:
    """A complete BM25 index bound to one immutable chunk artifact."""

    directory: Path
    manifest: dict[str, Any]
    files: Mapping[str, Path]


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def verify_artifact_descriptor(
    root: str | Path,
    descriptor: Mapping[str, Any],
    *,
    label: str,
    expected_rows: int | None = None,
) -> VerifiedFile:
    """Verify path containment, size, digest, and an optional row contract."""

    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor must be a mapping")
    file_name = descriptor.get("file")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError(f"{label} descriptor has no file")
    relative = Path(file_name)
    if relative.is_absolute():
        raise ValueError(f"{label} artifact path must be relative")

    directory = Path(root).resolve()
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes its directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")

    size_bytes = descriptor.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or path.stat().st_size != size_bytes
    ):
        raise ValueError(f"{label} artifact size does not match its manifest")
    expected_sha = descriptor.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
        raise ValueError(f"{label} artifact descriptor has an invalid sha256")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label} artifact hash does not match its manifest")
    if expected_rows is not None:
        _positive_integer(expected_rows, label=f"{label} expected rows")
        if descriptor.get("rows") != expected_rows:
            raise ValueError(f"{label} row count does not match its manifest")
    return VerifiedFile(path=path, descriptor=descriptor)


def _validate_sharded_embedding_descriptor(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    rows: int,
    dimension: int,
    label: str,
) -> Path:
    """Validate one manifest plus every immutable ``part-*.npy`` artifact."""

    if descriptor.get("storage") != "sharded_npy":
        raise ValueError(f"{label} storage must be sharded_npy")
    if descriptor.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    if descriptor.get("dtype") != "float32" or descriptor.get("shape") != [
        rows,
        dimension,
    ]:
        raise ValueError(f"{label} shape or dtype is invalid")

    manifest_file = verify_artifact_descriptor(
        root,
        descriptor,
        label=f"{label} manifest",
    ).path
    manifest = read_json_object(manifest_file, label=f"{label} manifest payload")
    manifest_rows = manifest.get("total_rows", manifest.get("rows"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dtype") != "float32"
        or manifest.get("dimension") != dimension
        or manifest_rows != rows
    ):
        raise ValueError(f"{label} manifest metadata is inconsistent")
    descriptor_parts = descriptor.get("parts")
    manifest_parts = manifest.get("parts")
    if (
        not isinstance(descriptor_parts, list)
        or not descriptor_parts
        or manifest_parts != descriptor_parts
    ):
        raise ValueError(f"{label} part list differs from its manifest")

    expected_start = 0
    listed: set[str] = set()
    for position, part in enumerate(descriptor_parts):
        if not isinstance(part, Mapping):
            raise ValueError(f"{label} part {position} must be a mapping")
        file_name = part.get("file")
        if (
            not isinstance(file_name, str)
            or not re.fullmatch(r"part-\d{5,}\.npy", file_name)
            or Path(file_name).name != file_name
            or file_name in listed
        ):
            raise ValueError(f"{label} part {position} has an invalid file name")
        if file_name != f"part-{position:05d}.npy":
            raise ValueError(f"{label} parts must be numbered contiguously from zero")
        part_rows = _positive_integer(part.get("rows"), label=f"{label} part rows")
        if (
            part.get("start_row") != expected_start
            or part.get("shape") != [part_rows, dimension]
        ):
            raise ValueError(f"{label} part row range or shape is invalid")
        verified = verify_artifact_descriptor(
            manifest_file.parent,
            part,
            label=f"{label} {file_name}",
            expected_rows=part_rows,
        )
        values = np.load(verified.path, mmap_mode="r", allow_pickle=False)
        try:
            if values.shape != (part_rows, dimension) or values.dtype != np.dtype(
                "float32"
            ):
                raise ValueError(f"{label} {file_name} array metadata is invalid")
        finally:
            close_numpy_memmap(values)
        listed.add(file_name)
        expected_start += part_rows
    if expected_start != rows:
        raise ValueError(f"{label} part ranges do not cover all embedding rows")
    discovered = {
        path.name for path in manifest_file.parent.glob("part-*.npy") if path.is_file()
    }
    if discovered != listed:
        raise ValueError(f"{label} directory contains untracked embedding parts")
    return manifest_file


def validate_build_directory(
    build_dir: str | Path,
    expected_build_id: str | None = None,
) -> VerifiedBuild:
    """Validate one immutable build exactly when it enters the process."""

    directory = Path(build_dir).resolve()
    manifest = read_json_object(directory / "manifest.json", label="Build manifest")
    if manifest.get("status") != "complete":
        raise ValueError("Build manifest must have status=complete")
    if expected_build_id is not None and manifest.get("build_id") != expected_build_id:
        raise ValueError("Build directory identity does not match the expected build id")

    embedding = manifest.get("embedding")
    if not isinstance(embedding, dict) or set(embedding) != {"space"}:
        raise ValueError("Build manifest must contain one canonical embedding space")
    if not isinstance(embedding["space"], dict) or "query_prefix" in embedding["space"]:
        raise ValueError("Build manifest embedding space is invalid")
    try:
        embedding_space = EmbeddingSpaceSpec.from_mapping(embedding["space"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Build manifest embedding space is invalid") from exc

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Build manifest artifacts must be a mapping")
    index = manifest.get("index")
    if not isinstance(index, Mapping) or index.get("backend") not in {"faiss", "numpy"}:
        raise ValueError("Build manifest index metadata is invalid")
    required = ["chunks", "embeddings"]
    if "chunk_offsets" in artifacts:
        required.append("chunk_offsets")
    if index["backend"] == "faiss" and index.get("type") != "streaming_flat_ip":
        required.append("index")
    elif index["backend"] == "numpy" and "index" in artifacts:
        raise ValueError("NumPy builds must not contain a separate index artifact")
    elif index.get("type") == "streaming_flat_ip" and "index" in artifacts:
        raise ValueError("Streaming exact builds must not contain a persisted index artifact")

    files: dict[str, Path] = {}
    for name in required:
        descriptor = artifacts.get(name)
        verified = verify_artifact_descriptor(
            directory,
            descriptor if isinstance(descriptor, Mapping) else {},
            label=f"Build {name}",
        )
        files[name] = verified.path

    rows = _positive_integer(artifacts["chunks"].get("rows"), label="Chunk rows")
    if index.get("count") not in {None, rows} or index.get("dimension") not in {
        None,
        embedding_space.dimension,
    }:
        raise ValueError("Legacy index dimensions do not match build artifacts")
    shape = artifacts["embeddings"].get("shape")
    if shape is not None and (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != rows
        or shape[1] != embedding_space.dimension
    ):
        raise ValueError("Embedding artifact shape does not match chunks or index")
    if "chunk_offsets" in artifacts:
        offsets = artifacts["chunk_offsets"]
        if offsets.get("rows") != rows or offsets.get("dtype") != "uint64":
            raise ValueError("Chunk offset descriptor does not match chunks")
    embedding_storage = artifacts["embeddings"].get("storage")
    if embedding_storage is not None:
        if embedding_storage != "sharded_npy":
            raise ValueError("Build embedding storage is unsupported")
        if index.get("type") != "streaming_flat_ip":
            raise ValueError("Sharded embeddings require a streaming exact index")
        files["embeddings"] = _validate_sharded_embedding_descriptor(
            directory,
            artifacts["embeddings"],
            rows=rows,
            dimension=embedding_space.dimension,
            label="Build embeddings",
        )
    return VerifiedBuild(directory=directory, manifest=manifest, files=files)


_BM25S_ARTIFACT_FILES = {
    "data": "data.csc.index.npy",
    "indices": "indices.csc.index.npy",
    "indptr": "indptr.csc.index.npy",
    "index_vocab": "vocab.index.json",
    "params": "params.index.json",
    "tokenizer_vocab": "vocab.tokenizer.json",
    "tokenizer_stopwords": "stopwords.tokenizer.json",
}


def validate_bm25_index_directory(
    index_dir: str | Path,
    expected_id: str | None = None,
    *,
    expected_spec: Mapping[str, Any] | None = None,
) -> VerifiedSparseIndex:
    """Validate a persisted BM25S index and its source-chunk binding."""

    directory = Path(index_dir).resolve()
    manifest = read_json_object(directory / "manifest.json", label="BM25 index manifest")
    if manifest.get("status") != "complete":
        raise ValueError("BM25 index manifest must have status=complete")
    if expected_id is not None and manifest.get("sparse_index_id") != expected_id:
        raise ValueError("BM25 index identity does not match the expected sparse index id")

    spec = manifest.get("sparse_index_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("BM25 index manifest has no sparse index specification")
    spec_sha = json_sha256(spec)
    if manifest.get("sparse_index_spec_sha256") != spec_sha:
        raise ValueError("BM25 sparse index specification hash is invalid")
    if manifest.get("sparse_index_id") != f"bm25_{spec_sha[:16]}":
        raise ValueError("BM25 sparse index id is not derived from its specification")
    if expected_spec is not None and dict(spec) != dict(expected_spec):
        raise ValueError("BM25 sparse index specification does not match the active config")

    bm25 = spec.get("bm25")
    source_chunks = spec.get("source_chunks")
    if (
        not isinstance(bm25, Mapping)
        or bm25.get("backend") != "bm25s"
        or bm25.get("method") != "lucene"
        or not isinstance(source_chunks, Mapping)
    ):
        raise ValueError("BM25 sparse index specification is invalid")
    rows = _positive_integer(source_chunks.get("rows"), label="BM25 source chunk rows")
    source_build_id = spec.get("source_build_id")
    if not isinstance(source_build_id, str) or not source_build_id.startswith("build_"):
        raise ValueError("BM25 source build id is invalid")
    if manifest.get("source_build_id") != source_build_id:
        raise ValueError("BM25 manifest source build id does not match its specification")
    if (
        source_chunks.get("file") != "chunks.jsonl"
        or not isinstance(source_chunks.get("size_bytes"), int)
        or source_chunks["size_bytes"] <= 0
        or not isinstance(source_chunks.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(source_chunks["sha256"]) is None
    ):
        raise ValueError("BM25 source chunk descriptor is invalid")
    if manifest.get("document_count") != rows:
        raise ValueError("BM25 document count does not match its source chunks")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_BM25S_ARTIFACT_FILES):
        raise ValueError("BM25 index manifest has an invalid artifact set")
    files: dict[str, Path] = {}
    for name, file_name in _BM25S_ARTIFACT_FILES.items():
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, Mapping) or descriptor.get("file") != file_name:
            raise ValueError(f"BM25 {name} descriptor has an unexpected file name")
        files[name] = verify_artifact_descriptor(
            directory,
            descriptor,
            label=f"BM25 {name}",
        ).path

    params = read_json_object(files["params"], label="BM25 index parameters")
    if (
        params.get("num_docs") != rows
        or params.get("method") != bm25.get("method")
        or params.get("k1") != bm25.get("k1")
        or params.get("b") != bm25.get("b")
        or params.get("version") != spec.get("bm25s_version")
    ):
        raise ValueError("BM25 index parameters do not match the sparse index specification")
    read_json_object(files["index_vocab"], label="BM25 index vocabulary")
    read_json_object(files["tokenizer_vocab"], label="BM25 tokenizer vocabulary")
    with files["tokenizer_stopwords"].open("r", encoding="utf-8") as handle:
        stopwords = json.load(handle)
    if not isinstance(stopwords, list) or not all(
        isinstance(word, str) for word in stopwords
    ):
        raise ValueError("BM25 tokenizer stopwords must be a string list")

    data = np.load(files["data"], mmap_mode="r", allow_pickle=False)
    indices = np.load(files["indices"], mmap_mode="r", allow_pickle=False)
    indptr = np.load(files["indptr"], mmap_mode="r", allow_pickle=False)
    try:
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError("BM25 sparse arrays must be one-dimensional")
        if data.dtype != np.dtype("float32") or not np.issubdtype(
            indices.dtype, np.integer
        ) or not np.issubdtype(indptr.dtype, np.integer):
            raise ValueError("BM25 sparse array dtypes are invalid")
        if data.shape != indices.shape or indptr.shape[0] < 2:
            raise ValueError("BM25 sparse array shapes are inconsistent")
        if int(indptr[0]) != 0 or int(indptr[-1]) != data.shape[0]:
            raise ValueError("BM25 sparse matrix pointers are invalid")
    finally:
        close_numpy_memmap(data)
        close_numpy_memmap(indices)
        close_numpy_memmap(indptr)

    return VerifiedSparseIndex(directory=directory, manifest=manifest, files=files)


def validate_encoded_corpus_directory(
    directory: str | Path,
    expected_id: str,
    expected_spec_sha256: str,
) -> VerifiedEncodedCorpus:
    """Validate a reusable chunk/offset/embedding artifact set."""

    root = Path(directory).resolve()
    manifest = read_json_object(
        root / "manifest.json",
        label="Encoded-corpus manifest",
    )
    if manifest.get("status") != "complete":
        raise ValueError("Encoded-corpus manifest must have status=complete")
    if manifest.get("encoded_corpus_id") != expected_id:
        raise ValueError("Encoded-corpus directory identity does not match")
    if manifest.get("encoded_corpus_spec_sha256") != expected_spec_sha256:
        raise ValueError("Encoded-corpus specification does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Encoded-corpus manifest has no artifacts")

    files = {
        name: verify_artifact_descriptor(
            root,
            artifacts.get(name, {}),
            label=f"Encoded-corpus {name}",
        ).path
        for name in ("chunks", "chunk_offsets", "embeddings")
    }
    rows = _positive_integer(artifacts["chunks"].get("rows"), label="Encoded-corpus rows")
    dimension = _positive_integer(
        manifest.get("embedding", {}).get("space", {}).get("dimension"),
        label="Encoded-corpus embedding dimension",
    )
    legacy_num_chunks = manifest.get("chunking", {}).get("num_chunks")
    if artifacts["chunk_offsets"].get("rows") != rows or legacy_num_chunks not in {
        None,
        rows,
    }:
        raise ValueError("Encoded-corpus row counts are inconsistent")

    offsets = np.load(files["chunk_offsets"], mmap_mode="r", allow_pickle=False)
    embeddings = None
    try:
        if offsets.shape != (rows,) or offsets.dtype != np.dtype("uint64"):
            raise ValueError("Encoded-corpus chunk offsets are invalid")
        storage = artifacts["embeddings"].get("storage")
        if storage is None:
            embeddings = np.load(
                files["embeddings"], mmap_mode="r", allow_pickle=False
            )
            if (
                embeddings.shape != (rows, dimension)
                or embeddings.dtype != np.dtype("float32")
                or list(embeddings.shape) != artifacts["embeddings"].get("shape")
            ):
                raise ValueError("Encoded-corpus embeddings are invalid")
        elif storage == "sharded_npy":
            files["embeddings"] = _validate_sharded_embedding_descriptor(
                root,
                artifacts["embeddings"],
                rows=rows,
                dimension=dimension,
                label="Encoded-corpus embeddings",
            )
        else:
            raise ValueError("Encoded-corpus embedding storage is unsupported")
    finally:
        close_numpy_memmap(embeddings)
        close_numpy_memmap(offsets)
    if files["chunks"].stat().st_size <= 0:
        raise ValueError("Encoded-corpus chunks artifact is empty")
    return VerifiedEncodedCorpus(directory=root, manifest=manifest, files=files)
