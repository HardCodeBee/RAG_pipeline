"""Fail-closed resolution of pinned legacy BEIR offline artifacts.

The regular build identity remains the primary lookup path.  This module is a
narrow compatibility boundary for BEIR artifacts whose producer source hash
predates a source-only cleanup.  A versioned registry pins both sides of that
transition: the immutable producer manifests and the exact consumer source
hashes allowed to open them.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.persistence.artifact_io import read_json_object
from src.persistence.artifact_validation import VerifiedBuild, validate_build_directory
from src.provenance import (
    encoded_corpus_spec,
    json_sha256,
    sha256_file,
    source_group_sha256,
)


REGISTRY_KIND = "beir_offline_artifact_registry"
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_RELATIVE_PATH = Path("_registries") / "beir_offline_v1.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BUILD_ID = re.compile(r"build_[0-9a-f]{16}")
_ENCODED_ID = re.compile(r"encoded_corpus_[0-9a-f]{16}")
_SQLITE_ID = re.compile(r"sqlite_bm25_[0-9a-f]{16}")

_REGISTRY_KEYS = {
    "schema_version",
    "kind",
    "status",
    "consumer",
    "entries",
}
_CONSUMER_KEYS = {
    "build_source_sha256",
    "encoded_corpus_source_sha256",
    "sqlite_bm25_implementation_sha256",
}
_ENTRY_KEYS = {
    "dataset",
    "unit",
    "corpus_aggregate_sha256",
    "build",
    "encoded_corpus",
    "sqlite_bm25",
}
_BUILD_KEYS = {
    "artifact_id",
    "manifest_sha256",
    "spec_sha256",
    "producer_source_sha256",
    "chunks_sha256",
    "chunk_rows",
    "vector_id_sequence_sha256",
}
_ENCODED_KEYS = {
    "artifact_id",
    "manifest_sha256",
    "spec_sha256",
    "producer_source_sha256",
}
_SQLITE_KEYS = {
    "artifact_id",
    "manifest_sha256",
    "identity_sha256",
    "implementation_sha256",
}


@dataclass(frozen=True, slots=True)
class PinnedBeirBuild:
    """One verified legacy build selected by an exact BEIR registry entry."""

    verified_build: VerifiedBuild
    registry_path: Path
    dataset: str
    unit: str
    producer_build_source_sha256: str
    consumer_build_source_sha256: str


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} keys are not exact; missing={missing}, unknown={unknown}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _artifact_id(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    text = _text(value, label=label)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{label} has an invalid artifact id")
    return text


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _contained_directory(root: Path, relative: Path, *, label: str) -> Path:
    base = root.resolve()
    directory = (base / relative).resolve()
    try:
        directory.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the artifacts root") from exc
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {directory}")
    return directory


def _manifest(
    directory: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    path = directory / "manifest.json"
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} manifest hash differs from the pinned registry")
    return read_json_object(path, label=f"{label} manifest")


def _without_source(spec: Mapping[str, Any], field: str, *, label: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(spec))
    source = value.pop(field, None)
    _sha(source, label=f"{label} {field}")
    return value


def _validate_registry_shape(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    _exact_keys(value, _REGISTRY_KEYS, label="BEIR registry")
    if (
        value.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or value.get("kind") != REGISTRY_KIND
        or value.get("status") != "complete"
    ):
        raise ValueError("BEIR registry header is incomplete or incompatible")
    consumer = _mapping(value.get("consumer"), label="BEIR registry consumer")
    _exact_keys(consumer, _CONSUMER_KEYS, label="BEIR registry consumer")
    for key in sorted(_CONSUMER_KEYS):
        _sha(consumer.get(key), label=f"BEIR registry consumer {key}")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("BEIR registry entries must be a non-empty list")

    entries: list[Mapping[str, Any]] = []
    lookup_keys: set[tuple[str, str]] = set()
    units: set[tuple[str, str]] = set()
    artifact_ids: set[str] = set()
    for position, raw in enumerate(raw_entries):
        entry = _mapping(raw, label=f"BEIR registry entry {position}")
        _exact_keys(entry, _ENTRY_KEYS, label=f"BEIR registry entry {position}")
        dataset = _text(entry.get("dataset"), label=f"entry {position} dataset")
        unit = _text(entry.get("unit"), label=f"entry {position} unit")
        aggregate = _sha(
            entry.get("corpus_aggregate_sha256"),
            label=f"entry {position} corpus aggregate",
        )
        lookup_key = (dataset, aggregate)
        unit_key = (dataset, unit)
        if lookup_key in lookup_keys or unit_key in units:
            raise ValueError("BEIR registry contains a duplicate unit or corpus binding")
        lookup_keys.add(lookup_key)
        units.add(unit_key)

        records = (
            ("build", _BUILD_KEYS, _BUILD_ID),
            ("encoded_corpus", _ENCODED_KEYS, _ENCODED_ID),
            ("sqlite_bm25", _SQLITE_KEYS, _SQLITE_ID),
        )
        for name, keys, pattern in records:
            record = _mapping(entry.get(name), label=f"entry {position} {name}")
            _exact_keys(record, keys, label=f"entry {position} {name}")
            artifact_id = _artifact_id(
                record.get("artifact_id"),
                pattern,
                label=f"entry {position} {name} artifact_id",
            )
            if artifact_id in artifact_ids:
                raise ValueError("BEIR registry reuses an artifact id across entries")
            artifact_ids.add(artifact_id)
            for key in keys - {"artifact_id", "chunk_rows"}:
                _sha(record.get(key), label=f"entry {position} {name} {key}")
            if name == "build":
                _positive_int(record.get("chunk_rows"), label="build chunk_rows")
        entries.append(entry)
    return consumer, entries


def _validate_pinned_entry(
    *,
    artifacts_root: Path,
    project_root: Path,
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    entry: Mapping[str, Any],
    consumer: Mapping[str, Any],
    current_build_spec: Mapping[str, Any],
    current_build_source_sha256: str,
    require_sparse: bool,
) -> VerifiedBuild:
    build_record = _mapping(entry["build"], label="BEIR registry build")
    build_id = str(build_record["artifact_id"])
    build_dir = _contained_directory(
        artifacts_root,
        Path(build_id),
        label="Pinned BEIR build",
    )
    manifest = _manifest(
        build_dir,
        str(build_record["manifest_sha256"]),
        label="Pinned BEIR build",
    )
    spec = _mapping(manifest.get("build_spec"), label="Pinned BEIR build spec")
    spec_sha = json_sha256(spec)
    if (
        manifest.get("status") != "complete"
        or manifest.get("build_id") != build_id
        or manifest.get("build_spec_sha256") != spec_sha
        or build_record.get("spec_sha256") != spec_sha
        or build_id != f"build_{spec_sha[:16]}"
    ):
        raise ValueError("Pinned BEIR build identity is inconsistent")
    producer_source = _sha(
        spec.get("build_source_sha256"),
        label="Pinned BEIR producer build source",
    )
    if producer_source != build_record.get("producer_source_sha256"):
        raise ValueError("Pinned BEIR producer source differs from the registry")
    if current_build_source_sha256 != consumer.get("build_source_sha256"):
        raise ValueError("Current build source is not authorized by the BEIR registry")
    if _without_source(spec, "build_source_sha256", label="Pinned build") != _without_source(
        current_build_spec,
        "build_source_sha256",
        label="Current build",
    ):
        raise ValueError("Pinned BEIR build differs from the current semantic build spec")

    loader = _mapping(spec.get("loader"), label="Pinned BEIR loader")
    corpus = _mapping(spec.get("corpus"), label="Pinned BEIR corpus")
    if (
        loader.get("type") != "beir"
        or loader.get("expected_dataset") != entry.get("dataset")
        or corpus.get("aggregate_sha256") != entry.get("corpus_aggregate_sha256")
    ):
        raise ValueError("Pinned BEIR dataset or corpus binding is inconsistent")
    artifacts = _mapping(manifest.get("artifacts"), label="Pinned BEIR artifacts")
    chunks = _mapping(artifacts.get("chunks"), label="Pinned BEIR chunks")
    if (
        chunks.get("sha256") != build_record.get("chunks_sha256")
        or chunks.get("rows") != build_record.get("chunk_rows")
        or manifest.get("vector_id_sequence_sha256")
        != build_record.get("vector_id_sequence_sha256")
    ):
        raise ValueError("Pinned BEIR chunk identity differs from the registry")
    verified = validate_build_directory(build_dir, build_id)

    encoded_record = _mapping(entry["encoded_corpus"], label="BEIR encoded corpus")
    encoded_id = str(encoded_record["artifact_id"])
    encoded_dir = _contained_directory(
        artifacts_root,
        Path("_encoded_corpora") / encoded_id,
        label="Pinned BEIR encoded corpus",
    )
    encoded_manifest = _manifest(
        encoded_dir,
        str(encoded_record["manifest_sha256"]),
        label="Pinned BEIR encoded corpus",
    )
    encoded_spec = _mapping(
        encoded_manifest.get("encoded_corpus_spec"),
        label="Pinned BEIR encoded corpus spec",
    )
    encoded_spec_sha = json_sha256(encoded_spec)
    if (
        encoded_manifest.get("status") != "complete"
        or encoded_manifest.get("encoded_corpus_id") != encoded_id
        or encoded_manifest.get("encoded_corpus_spec_sha256") != encoded_spec_sha
        or encoded_record.get("spec_sha256") != encoded_spec_sha
        or encoded_id != f"encoded_corpus_{encoded_spec_sha[:16]}"
        or encoded_spec.get("encoded_corpus_source_sha256")
        != encoded_record.get("producer_source_sha256")
    ):
        raise ValueError("Pinned BEIR encoded-corpus identity is inconsistent")
    for key in ("loader", "chunking", "embedding", "corpus"):
        if encoded_spec.get(key) != spec.get(key):
            raise ValueError(f"Pinned encoded corpus differs from its build: {key}")
    encoded_artifacts = _mapping(
        encoded_manifest.get("artifacts"),
        label="Pinned encoded-corpus artifacts",
    )
    encoded_chunks = _mapping(
        encoded_artifacts.get("chunks"),
        label="Pinned encoded-corpus chunks",
    )
    if (
        encoded_chunks.get("sha256") != chunks.get("sha256")
        or encoded_chunks.get("rows") != chunks.get("rows")
    ):
        raise ValueError("Pinned encoded corpus is not bound to the build chunks")
    current_encoded_source = source_group_sha256(project_root, "encoded_corpus")
    if current_encoded_source != consumer.get("encoded_corpus_source_sha256"):
        raise ValueError("Current encoded-corpus source is not authorized by the registry")
    current_encoded_spec = encoded_corpus_spec(
        dict(config),
        dict(corpus),
        current_encoded_source,
    )
    if _without_source(
        encoded_spec,
        "encoded_corpus_source_sha256",
        label="Pinned encoded corpus",
    ) != _without_source(
        current_encoded_spec,
        "encoded_corpus_source_sha256",
        label="Current encoded corpus",
    ):
        raise ValueError(
            "Pinned BEIR encoded corpus differs from the current semantic spec"
        )

    sparse_record = _mapping(entry["sqlite_bm25"], label="BEIR SQLite BM25")
    sparse_id = str(sparse_record["artifact_id"])
    sparse_dir = _contained_directory(
        artifacts_root,
        Path("_sparse_indexes") / sparse_id,
        label="Pinned BEIR SQLite BM25",
    )
    sparse_manifest = _manifest(
        sparse_dir,
        str(sparse_record["manifest_sha256"]),
        label="Pinned BEIR SQLite BM25",
    )
    identity = _mapping(
        sparse_manifest.get("identity"),
        label="Pinned SQLite BM25 identity",
    )
    identity_sha = json_sha256(identity)
    if (
        sparse_manifest.get("status") != "complete"
        or sparse_manifest.get("sparse_index_id") != sparse_id
        or sparse_manifest.get("identity_sha256") != identity_sha
        or sparse_record.get("identity_sha256") != identity_sha
        or sparse_id != f"sqlite_bm25_{identity_sha[:16]}"
        or identity.get("source_build_id") != build_id
        or identity.get("implementation_sha256")
        != sparse_record.get("implementation_sha256")
    ):
        raise ValueError("Pinned SQLite BM25 identity is inconsistent")
    source_chunks = _mapping(
        identity.get("source_chunks"),
        label="Pinned SQLite BM25 source chunks",
    )
    if (
        source_chunks.get("sha256") != chunks.get("sha256")
        or source_chunks.get("rows") != chunks.get("rows")
    ):
        raise ValueError("Pinned SQLite BM25 is not bound to the build chunks")
    if require_sparse:
        current_implementation = sha256_file(
            project_root / "src" / "retrievers" / "sqlite_bm25.py"
        )
        if (
            current_implementation
            != consumer.get("sqlite_bm25_implementation_sha256")
            or current_implementation != identity.get("implementation_sha256")
        ):
            raise ValueError("Current SQLite BM25 implementation is not registry-compatible")
        from src.retrievers.sqlite_bm25 import validate_sqlite_bm25_index

        validate_sqlite_bm25_index(
            sparse_dir,
            expected_identity_sha256=identity_sha,
        )
    return verified


def resolve_pinned_beir_build(
    *,
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    current_build_spec: Mapping[str, Any],
    current_build_source_sha256: str,
    artifacts_root: str | Path,
    project_root: str | Path,
) -> PinnedBeirBuild | None:
    """Resolve one exact legacy BEIR build, or return ``None`` without a registry.

    A present registry is authoritative and fail-closed: malformed content, a
    missing entry, a semantic mismatch, or any pinned manifest mismatch raises.
    """

    loader = _mapping(config.get("loader"), label="Active loader")
    if loader.get("type") != "beir":
        return None
    root = Path(artifacts_root).resolve()
    registry_path = (root / REGISTRY_RELATIVE_PATH).resolve()
    try:
        registry_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("BEIR registry path escapes the artifacts root") from exc
    if not registry_path.is_file():
        return None
    registry = read_json_object(registry_path, label="BEIR offline artifact registry")
    consumer, entries = _validate_registry_shape(registry)
    dataset = _text(loader.get("expected_dataset"), label="BEIR expected_dataset")
    aggregate = _sha(
        corpus.get("aggregate_sha256"),
        label="Active BEIR corpus aggregate",
    )
    matches = [
        entry
        for entry in entries
        if entry.get("dataset") == dataset
        and entry.get("corpus_aggregate_sha256") == aggregate
    ]
    if len(matches) != 1:
        raise ValueError(
            "BEIR registry must contain exactly one entry for the active dataset/corpus; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    verified = _validate_pinned_entry(
        artifacts_root=root,
        project_root=Path(project_root).resolve(),
        config=config,
        corpus=corpus,
        entry=entry,
        consumer=consumer,
        current_build_spec=current_build_spec,
        current_build_source_sha256=_sha(
            current_build_source_sha256,
            label="Current build source",
        ),
        require_sparse=config.get("retrieval", {}).get("method") == "bm25",
    )
    return PinnedBeirBuild(
        verified_build=verified,
        registry_path=registry_path,
        dataset=dataset,
        unit=str(entry["unit"]),
        producer_build_source_sha256=str(
            entry["build"]["producer_source_sha256"]
        ),
        consumer_build_source_sha256=str(consumer["build_source_sha256"]),
    )
