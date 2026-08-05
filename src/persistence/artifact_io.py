"""Persistence primitives for manifests, JSONL, descriptors, and NumPy maps."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from src.provenance import canonical_json_bytes, sha256_file
from src.records import ChunkRecord


def read_json_object(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    """Read one JSON object without applying domain-specific validation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return value


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write one deterministic, human-readable JSON manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def encode_jsonl_row(value: Mapping[str, Any], *, canonical: bool = True) -> bytes:
    """Encode one JSONL object without its trailing newline."""

    if not isinstance(value, Mapping):
        raise TypeError("JSONL rows must be mappings")
    if canonical:
        return canonical_json_bytes(dict(value))
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL object rows in source order."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            yield value


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    canonical: bool = True,
) -> int:
    """Stream JSONL rows to disk and return the number written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(encode_jsonl_row(row, canonical=canonical).decode("utf-8"))
            handle.write("\n")
            count += 1
    return count


def describe_artifact(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
    rows: int | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one descriptor, reusing producer metadata when supplied."""

    artifact = Path(path)
    file_name = (
        artifact.relative_to(Path(relative_to)).as_posix()
        if relative_to is not None
        else artifact.name
    )
    descriptor: dict[str, Any] = {
        "file": file_name,
        "size_bytes": artifact.stat().st_size if size_bytes is None else size_bytes,
        "sha256": sha256_file(artifact) if sha256 is None else sha256,
    }
    if rows is not None:
        descriptor["rows"] = rows
    if extra:
        descriptor.update(extra)
    return descriptor


def decode_chunk_record_line(raw: bytes, expected_vector_id: int) -> ChunkRecord:
    """Decode one chunk row and enforce its row/vector-id alignment."""

    if not raw:
        raise RuntimeError(
            f"Chunk artifact ended before vector id {expected_vector_id}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Chunk artifact row {expected_vector_id} is invalid"
        ) from exc
    record = ChunkRecord.from_mapping(value)
    if record.vector_id != expected_vector_id:
        raise RuntimeError(
            f"Chunk artifact vector id {record.vector_id} does not match "
            f"row {expected_vector_id}"
        )
    return record


def close_numpy_memmap(array: Any, *, flush: bool = False) -> None:
    """Explicitly release a NumPy memory map, including on Windows."""

    if array is None:
        return
    if flush and callable(getattr(array, "flush", None)):
        array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()
