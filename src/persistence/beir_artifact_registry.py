"""Resolve immutable BEIR artifacts across source-only cleanup revisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.persistence.artifact_io import read_json_object
from src.persistence.artifact_validation import (
    VerifiedBuild,
    validate_build_directory,
)
from src.provenance import encoded_corpus_spec, sha256_file


COMPATIBILITY_VERSION = 2
REGISTRY_RELATIVE_PATH = Path("_registries") / "beir_offline_v1.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
@dataclass(frozen=True, slots=True)
class PinnedBeirBuild:
    """One legacy build selected and verified through the compact registry."""

    verified_build: VerifiedBuild
    sqlite_bm25_directory: Path
    sqlite_bm25_id: str
    dataset: str
    unit: str
    compatibility_version: int = COMPATIBILITY_VERSION


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _directory(root: Path, relative: Path, *, label: str) -> Path:
    base = root.resolve()
    directory = (base / relative).resolve()
    try:
        directory.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the artifacts root") from exc
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} is missing: {directory}")
    return directory


def _manifest(directory: Path, pinned_sha256: str, *, label: str) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{label} manifest is missing: {path}")
    if _SHA256.fullmatch(pinned_sha256) is None:
        raise ValueError(f"{label} registry digest is invalid")
    if sha256_file(path) != pinned_sha256:
        raise ValueError(f"{label} manifest hash differs from the registry")
    return read_json_object(path, label=f"{label} manifest")


def _semantic_spec(spec: Mapping[str, Any], source_field: str, *, label: str) -> dict[str, Any]:
    value = dict(spec)
    if source_field not in value:
        raise ValueError(f"{label} has no {source_field}")
    value.pop(source_field)
    value.pop("chunking", None)
    value.pop("loader", None)
    corpus = value.get("corpus")
    if isinstance(corpus, Mapping):
        value["corpus"] = {"aggregate_sha256": corpus.get("aggregate_sha256")}
    environment = value.get("producer_environment")
    if isinstance(environment, Mapping):
        environment = dict(environment)
        packages = environment.get("packages")
        if isinstance(packages, Mapping):
            environment["packages"] = {
                key: item for key, item in packages.items() if key != "pyyaml"
            }
        value["producer_environment"] = environment
    return value


def _validate_registry_shape(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("compatibility_version") != COMPATIBILITY_VERSION:
        raise ValueError("BEIR registry compatibility version is unsupported")
    raw_entries = value.get("entries")
    if (
        not isinstance(raw_entries, list)
        or not raw_entries
        or any(not isinstance(entry, Mapping) for entry in raw_entries)
    ):
        raise ValueError("BEIR registry entries must be a non-empty list")
    return list(raw_entries)


def _validate_entry(
    *,
    artifacts_root: Path,
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    current_build_spec: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> PinnedBeirBuild | None:
    build_record = _mapping(entry["build"], label="BEIR build")
    build_id = str(build_record["artifact_id"])
    build_dir = _directory(artifacts_root, Path(build_id), label="BEIR build")
    build_manifest = _manifest(
        build_dir,
        str(build_record["manifest_sha256"]),
        label="BEIR build",
    )
    legacy_build_spec = _mapping(
        build_manifest.get("build_spec"),
        label="BEIR build spec",
    )
    legacy_embedding = _mapping(
        legacy_build_spec.get("embedding"),
        label="Legacy BEIR document embedding",
    )
    current_embedding = _mapping(
        current_build_spec.get("embedding"),
        label="Current BEIR document embedding",
    )
    # Registry entries are keyed by dataset/corpus so several dense baselines
    # can legitimately target the same prepared unit. A different document
    # embedding identifies another artifact, not a damaged legacy entry.
    if legacy_embedding != current_embedding:
        return None

    # Once the document embedding matches, keep the historical trust boundary
    # strict before accepting or semantically comparing the pinned build.
    verified_build = validate_build_directory(build_dir, build_id)
    if _semantic_spec(
        legacy_build_spec,
        "build_source_sha256",
        label="Legacy BEIR build spec",
    ) != _semantic_spec(
        current_build_spec,
        "build_source_sha256",
        label="Current BEIR build spec",
    ):
        raise ValueError("BEIR build semantic spec has changed")
    legacy_loader = _mapping(legacy_build_spec.get("loader"), label="BEIR loader")
    legacy_corpus = _mapping(legacy_build_spec.get("corpus"), label="BEIR corpus")
    if (
        legacy_loader.get("type") != "beir"
        or legacy_loader.get("expected_dataset") != entry.get("dataset")
        or legacy_corpus.get("aggregate_sha256")
        != entry.get("corpus_aggregate_sha256")
    ):
        raise ValueError("BEIR registry dataset binding differs from the build")

    encoded_record = _mapping(entry["encoded_corpus"], label="BEIR encoded corpus")
    encoded_id = str(encoded_record["artifact_id"])
    encoded_dir = _directory(
        artifacts_root,
        Path("_encoded_corpora") / encoded_id,
        label="BEIR encoded corpus",
    )
    encoded_manifest = _manifest(
        encoded_dir,
        str(encoded_record["manifest_sha256"]),
        label="BEIR encoded corpus",
    )
    encoded_spec = _mapping(
        encoded_manifest.get("encoded_corpus_spec"),
        label="BEIR encoded-corpus spec",
    )
    if (
        encoded_manifest.get("status") != "complete"
        or encoded_manifest.get("encoded_corpus_id") != encoded_id
    ):
        raise ValueError("BEIR encoded-corpus manifest is incomplete or misplaced")
    current_encoded_spec = encoded_corpus_spec(
        dict(config),
        dict(corpus),
        "compatibility-version-2",
    )
    if _semantic_spec(
        encoded_spec,
        "encoded_corpus_source_sha256",
        label="Legacy BEIR encoded-corpus spec",
    ) != _semantic_spec(
        current_encoded_spec,
        "encoded_corpus_source_sha256",
        label="Current BEIR encoded-corpus spec",
    ):
        raise ValueError("BEIR encoded-corpus semantic spec has changed")
    build_chunks = _mapping(
        build_manifest.get("artifacts", {}).get("chunks"),
        label="BEIR build chunks",
    )
    encoded_chunks = _mapping(
        encoded_manifest.get("artifacts", {}).get("chunks"),
        label="BEIR encoded-corpus chunks",
    )
    if encoded_chunks != build_chunks:
        raise ValueError("BEIR encoded corpus is not bound to the build chunks")

    sparse_record = _mapping(entry["sqlite_bm25"], label="BEIR SQLite BM25")
    sparse_id = str(sparse_record["artifact_id"])
    sparse_dir = _directory(
        artifacts_root,
        Path("_sparse_indexes") / sparse_id,
        label="BEIR SQLite BM25",
    )
    sparse_manifest = _manifest(
        sparse_dir,
        str(sparse_record["manifest_sha256"]),
        label="BEIR SQLite BM25",
    )
    sparse_identity = _mapping(
        sparse_manifest.get("identity"),
        label="BEIR SQLite BM25 identity",
    )
    sparse_chunks = _mapping(
        sparse_identity.get("source_chunks"),
        label="BEIR SQLite BM25 source chunks",
    )
    if (
        sparse_manifest.get("status") != "complete"
        or sparse_manifest.get("sparse_index_id") != sparse_id
        or sparse_identity.get("source_build_id") != build_id
        or any(
            sparse_chunks.get(key) != build_chunks.get(key)
            for key in ("file", "rows", "size_bytes", "sha256")
        )
    ):
        raise ValueError("BEIR SQLite BM25 is not bound to the build chunks")
    retrieval = _mapping(config.get("retrieval"), label="Active retrieval")
    if retrieval.get("method") == "bm25":
        bm25 = _mapping(config.get("bm25"), label="Active BM25 config")
    else:
        bm25 = None
    if bm25 is not None and bm25.get("backend") == "sqlite":
        sparse_bm25 = _mapping(
            sparse_identity.get("bm25"),
            label="BEIR SQLite BM25 parameters",
        )
        sparse_analyzer = _mapping(
            sparse_identity.get("analyzer"),
            label="BEIR SQLite BM25 analyzer",
        )
        if (
            sparse_bm25.get("method") != bm25.get("method")
            or sparse_bm25.get("k1") != bm25.get("k1")
            or sparse_bm25.get("b") != bm25.get("b")
            or sparse_analyzer.get("name") != bm25.get("analyzer")
        ):
            raise ValueError("BEIR SQLite BM25 semantic config has changed")
    return PinnedBeirBuild(
        verified_build=verified_build,
        sqlite_bm25_directory=sparse_dir,
        sqlite_bm25_id=sparse_id,
        dataset=str(entry["dataset"]),
        unit=str(entry["unit"]),
    )


def resolve_pinned_beir_build(
    *,
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    current_build_spec: Mapping[str, Any],
    artifacts_root: str | Path,
) -> PinnedBeirBuild | None:
    """Return one compatible legacy BEIR build when the compact registry exists."""

    loader = _mapping(config.get("loader"), label="Active loader")
    root = Path(artifacts_root).resolve()
    registry_path = (root / REGISTRY_RELATIVE_PATH).resolve()
    try:
        registry_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("BEIR registry path escapes the artifacts root") from exc
    if not registry_path.is_file():
        return None
    entries = _validate_registry_shape(
        read_json_object(registry_path, label="BEIR offline artifact registry")
    )
    dataset = _text(loader.get("expected_dataset"), label="BEIR expected_dataset")
    aggregate = corpus.get("aggregate_sha256")
    if not isinstance(aggregate, str) or _SHA256.fullmatch(aggregate) is None:
        raise ValueError("Active BEIR corpus aggregate is invalid")
    matches = [
        entry
        for entry in entries
        if entry.get("dataset") == dataset
        and entry.get("corpus_aggregate_sha256") == aggregate
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            "BEIR registry must contain exactly one entry for the active dataset/corpus"
        )
    entry = matches[0]
    return _validate_entry(
        artifacts_root=root,
        config=config,
        corpus=corpus,
        current_build_spec=current_build_spec,
        entry=entry,
    )
