"""Single trust-boundary validator for persisted pipeline artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.artifact_io import close_numpy_memmap, iter_jsonl, read_json_object
from src.provenance import json_sha256, sha256_file, zero_based_sequence_sha256
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


@dataclass(frozen=True, slots=True)
class VerifiedDprCorpus:
    """A DPR corpus manifest and its verified data files."""

    directory: Path
    manifest: dict[str, Any]
    files: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class VerifiedNQSplit:
    """A canonical NQ question split and its already-verified manifest chain."""

    questions_path: Path
    rows: list[dict[str, Any]]
    questions_manifest: dict[str, Any]
    questions_file_sha256: str
    questions_manifest_sha256: str


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
        EmbeddingSpaceSpec.from_mapping(embedding["space"])
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
    if index["backend"] == "faiss":
        required.append("index")
    elif "index" in artifacts:
        raise ValueError("NumPy builds must not contain a separate index artifact")

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
    shape = artifacts["embeddings"].get("shape")
    if shape is not None and (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != rows
        or shape[1] != index.get("dimension")
    ):
        raise ValueError("Embedding artifact shape does not match chunks or index")
    if "chunk_offsets" in artifacts:
        offsets = artifacts["chunk_offsets"]
        if offsets.get("rows") != rows or offsets.get("dtype") != "uint64":
            raise ValueError("Chunk offset descriptor does not match chunks")
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
    expected_vector_hash = zero_based_sequence_sha256(rows)
    if (
        spec.get("vector_id_sequence_sha256") != expected_vector_hash
        or manifest.get("vector_id_sequence_sha256") != expected_vector_hash
    ):
        raise ValueError("BM25 document ids do not match the zero-based chunk sequence")

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
    if (
        artifacts["chunk_offsets"].get("rows") != rows
        or manifest.get("chunking", {}).get("num_chunks") != rows
    ):
        raise ValueError("Encoded-corpus row counts are inconsistent")

    offsets = np.load(files["chunk_offsets"], mmap_mode="r", allow_pickle=False)
    embeddings = np.load(files["embeddings"], mmap_mode="r", allow_pickle=False)
    try:
        if offsets.shape != (rows,) or offsets.dtype != np.dtype("uint64"):
            raise ValueError("Encoded-corpus chunk offsets are invalid")
        if (
            embeddings.shape != (rows, dimension)
            or embeddings.dtype != np.dtype("float32")
            or list(embeddings.shape) != artifacts["embeddings"].get("shape")
        ):
            raise ValueError("Encoded-corpus embeddings are invalid")
    finally:
        close_numpy_memmap(embeddings)
        close_numpy_memmap(offsets)
    if files["chunks"].stat().st_size <= 0:
        raise ValueError("Encoded-corpus chunks artifact is empty")
    return VerifiedEncodedCorpus(directory=root, manifest=manifest, files=files)


def validate_dpr_corpus_directory(
    directory: str | Path,
    *,
    expected_protocol: str,
    text_format: str,
    require_canonical_counts: bool,
    canonical_passage_count: int,
    canonical_question_count: int,
) -> VerifiedDprCorpus:
    """Validate the persisted DPR corpus contract without parsing its rows."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DPR Wikipedia corpus directory does not exist: {root}")
    manifest = read_json_object(
        root / "manifest.json",
        label="DPR Wikipedia corpus manifest",
    )
    if manifest.get("status") != "complete":
        raise ValueError("DPR Wikipedia corpus manifest must have status=complete")
    if manifest.get("protocol") != expected_protocol:
        raise ValueError("DPR Wikipedia corpus protocol does not match the loader")
    if manifest.get("format") != "dpr_psgs_w100_jsonl_v1":
        raise ValueError("DPR Wikipedia corpus format is unsupported")
    if manifest.get("text_format") != text_format:
        raise ValueError("DPR Wikipedia text format does not match the loader")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("DPR Wikipedia corpus manifest has no artifacts")
    files = {
        name: verify_artifact_descriptor(
            root,
            artifacts.get(name, {}),
            label=f"DPR Wikipedia {name}",
        ).path
        for name in ("passages", "passage_ids")
    }
    expected_count = _positive_integer(
        manifest.get("counts", {}).get("final_passages"),
        label="DPR Wikipedia passage count",
    )
    if any(artifacts[name].get("rows") != expected_count for name in files):
        raise ValueError("DPR Wikipedia corpus row counts are inconsistent")
    if require_canonical_counts and (
        manifest.get("canonical_counts") is not True
        or expected_count != canonical_passage_count
        or manifest.get("counts", {}).get("selected_questions")
        != canonical_question_count
    ):
        raise ValueError(
            "DPR Wikipedia corpus is not the canonical passage/question dataset"
        )
    return VerifiedDprCorpus(directory=root, manifest=manifest, files=files)


def validate_nq_dataset_split(
    corpus_dir: str | Path,
    *,
    split: str,
    requested_path: str | Path | None,
    expected_protocol: str,
    canonical_split_counts: Mapping[str, int],
    canonical_passage_count: int,
    canonical_hard_negatives_per_question: int,
    canonical_seed: str,
) -> VerifiedNQSplit:
    """Validate the complete NQ root/corpus/questions/split manifest chain."""

    if split not in canonical_split_counts:
        raise ValueError(f"Unknown canonical NQ split: {split}")
    corpus_path = Path(corpus_dir).resolve()
    dataset_root = corpus_path.parent
    root_manifest = read_json_object(
        dataset_root / "manifest.json",
        label="NQ dataset manifest",
    )
    request = root_manifest.get("request")
    if (
        root_manifest.get("status") != "complete"
        or root_manifest.get("protocol") != expected_protocol
        or root_manifest.get("canonical_counts") is not True
        or not isinstance(request, Mapping)
        or request.get("target_passages") != canonical_passage_count
        or request.get("calibration_questions")
        != canonical_split_counts.get("calibration")
        or request.get("evaluation_questions")
        != canonical_split_counts.get("evaluation")
        or request.get("hard_negatives_per_question")
        != canonical_hard_negatives_per_question
        or request.get("seed") != canonical_seed
    ):
        raise ValueError("NQ evaluation requires the canonical complete dataset manifest")

    manifests = root_manifest.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("NQ dataset manifest has no child manifests")
    corpus_descriptor = manifests.get("corpus")
    questions_descriptor = manifests.get("questions")
    corpus_verified = verify_artifact_descriptor(
        dataset_root,
        corpus_descriptor if isinstance(corpus_descriptor, Mapping) else {},
        label="NQ corpus manifest",
    )
    if corpus_verified.path != (corpus_path / "manifest.json").resolve():
        raise ValueError("Dataset corpus descriptor does not match paths.corpus")
    corpus_manifest = read_json_object(
        corpus_verified.path,
        label="NQ corpus manifest",
    )
    corpus_sources = corpus_manifest.get("sources")
    corpus_counts = corpus_manifest.get("counts")
    if (
        corpus_manifest.get("status") != "complete"
        or corpus_manifest.get("protocol") != expected_protocol
        or corpus_manifest.get("canonical_counts") is not True
        or not isinstance(corpus_counts, Mapping)
        or corpus_counts.get("final_passages") != canonical_passage_count
        or not isinstance(corpus_sources, Mapping)
        or not isinstance(corpus_sources.get("wikipedia"), Mapping)
        or not isinstance(corpus_sources.get("questions"), Mapping)
        or corpus_sources["wikipedia"].get("sha256")
        != request.get("wikipedia_sha256")
        or corpus_sources["questions"].get("sha256")
        != request.get("questions_sha256")
    ):
        raise ValueError("Corpus manifest is not linked to the canonical root manifest")

    questions_verified = verify_artifact_descriptor(
        dataset_root,
        questions_descriptor if isinstance(questions_descriptor, Mapping) else {},
        label="NQ questions manifest",
    )
    if questions_verified.path != (dataset_root / "questions" / "manifest.json").resolve():
        raise ValueError("Dataset questions descriptor does not use the canonical path")
    questions_manifest = read_json_object(
        questions_verified.path,
        label="NQ questions manifest",
    )
    question_source = questions_manifest.get("source")
    if (
        questions_manifest.get("status") != "complete"
        or questions_manifest.get("protocol") != expected_protocol
        or questions_manifest.get("canonical_counts") is not True
        or questions_manifest.get("counts")
        != {
            "calibration": canonical_split_counts["calibration"],
            "evaluation": canonical_split_counts["evaluation"],
            "selected": sum(canonical_split_counts.values()),
        }
        or not isinstance(question_source, Mapping)
        or question_source.get("sha256") != request.get("questions_sha256")
    ):
        raise ValueError("Questions manifest is not the canonical selected split")

    artifacts = questions_manifest.get("artifacts")
    split_descriptor = artifacts.get(split) if isinstance(artifacts, Mapping) else None
    split_verified = verify_artifact_descriptor(
        questions_verified.path.parent,
        split_descriptor if isinstance(split_descriptor, Mapping) else {},
        label=f"NQ {split} questions",
        expected_rows=canonical_split_counts[split],
    )
    if requested_path is not None and Path(requested_path).resolve() != split_verified.path:
        raise ValueError(
            "--questions must resolve to the canonical file recorded for the selected split"
        )
    rows = list(iter_jsonl(split_verified.path))
    if len(rows) != canonical_split_counts[split]:
        raise ValueError(f"{split} question rows do not match their manifest")
    return VerifiedNQSplit(
        questions_path=split_verified.path,
        rows=rows,
        questions_manifest=questions_manifest,
        questions_file_sha256=str(split_verified.descriptor["sha256"]),
        questions_manifest_sha256=str(questions_verified.descriptor["sha256"]),
    )


def validate_prepared_dataset(
    output_dir: str | Path,
    *,
    expected_protocol: str,
    expected_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a prepared dataset root, child manifests, and every descriptor."""

    output = Path(output_dir).resolve()
    root = read_json_object(output / "manifest.json", label="Dataset manifest")
    if root.get("status") != "complete" or root.get("protocol") != expected_protocol:
        raise ValueError("Existing dataset manifest is incomplete or incompatible")
    if root.get("request") != expected_request:
        raise ValueError("Existing dataset directory was prepared with a different request")
    manifests = root.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("Existing dataset has no child manifest descriptors")
    for name in ("corpus", "questions"):
        child_path = verify_artifact_descriptor(
            output,
            manifests.get(name, {}),
            label=f"Dataset {name} manifest",
        ).path
        child = read_json_object(child_path, label=f"Dataset {name} manifest")
        if child.get("status") != "complete" or child.get("protocol") != expected_protocol:
            raise ValueError("Existing child manifest is incomplete or incompatible")
        artifacts = child.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"Existing {name} manifest has no artifacts")
        for artifact_name, descriptor in artifacts.items():
            verify_artifact_descriptor(
                child_path.parent,
                descriptor if isinstance(descriptor, Mapping) else {},
                label=f"Dataset {name} {artifact_name}",
            )
    return root
