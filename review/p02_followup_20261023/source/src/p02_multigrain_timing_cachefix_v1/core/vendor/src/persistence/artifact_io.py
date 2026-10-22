"""Read JSON resources, decode chunk records, and close NumPy maps."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from p02_multigrain_timing_cachefix_v1.core.vendor.src.records import ChunkRecord


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
            f"Chunk artifact vector id {record.vector_id} does not match row {expected_vector_id}"
        )
    return record


def close_numpy_memmap(array: Any, *, flush: bool = False) -> None:
    """Explicitly release a NumPy memory map, including on Windows."""
    if array is None:
        return
    if flush and callable(getattr(array, "flush", None)):
        array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and (not mapping.closed):
        mapping.close()
