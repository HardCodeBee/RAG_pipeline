"""Query-time vector-id to ``ChunkRecord`` lookup implementations."""

from __future__ import annotations

# Chunk lookup is a retrieval component rather than a structural package.

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from src.persistence.artifact_io import close_numpy_memmap, decode_chunk_record_line
from src.records import ChunkRecord


class ChunkStore(Protocol):
    """Minimal retrieval-time chunk lookup contract."""

    def __len__(self) -> int: ...

    def get(self, vector_id: int) -> ChunkRecord: ...

    def get_many(self, vector_ids: Sequence[int]) -> list[ChunkRecord]: ...


def _validated_vector_id(vector_id: int, count: int) -> int:
    if isinstance(vector_id, bool) or not isinstance(vector_id, int):
        raise TypeError("vector_id must be an integer")
    if not 0 <= vector_id < count:
        raise KeyError(vector_id)
    return vector_id


class InMemoryChunkStore:
    """In-memory store used by small-corpus builds such as QASPER smoke tests."""

    def __init__(self, chunks: Iterable[ChunkRecord | dict]):
        records = [
            item if isinstance(item, ChunkRecord) else ChunkRecord.from_mapping(item)
            for item in chunks
        ]
        by_vector_id = {item.vector_id: item for item in records}
        if len(by_vector_id) != len(records):
            raise ValueError("Chunk vector ids must be unique")
        if set(by_vector_id) != set(range(len(records))):
            raise ValueError("Chunk vector ids must be the zero-based chunk sequence")
        self._records = by_vector_id

    def __len__(self) -> int:
        return len(self._records)

    def get(self, vector_id: int) -> ChunkRecord:
        return self._records[_validated_vector_id(vector_id, len(self))]

    def get_many(self, vector_ids: Sequence[int]) -> list[ChunkRecord]:
        return [self.get(vector_id) for vector_id in vector_ids]


class JsonlOffsetChunkStore:
    """Read only requested JSONL rows using a memory-mapped byte-offset table."""

    def __init__(
        self,
        chunks_path: str | Path,
        offsets_path: str | Path,
        *,
        expected_rows: int | None = None,
    ):
        self.chunks_path = Path(chunks_path).resolve()
        self.offsets_path = Path(offsets_path).resolve()
        if not self.chunks_path.is_file():
            raise FileNotFoundError(f"Chunk artifact is missing: {self.chunks_path}")
        if not self.offsets_path.is_file():
            raise FileNotFoundError(f"Chunk offset artifact is missing: {self.offsets_path}")

        offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
        if offsets.ndim != 1 or offsets.dtype != np.dtype("uint64"):
            raise ValueError("Chunk offsets must be a one-dimensional uint64 array")
        if expected_rows is not None and len(offsets) != expected_rows:
            raise ValueError("Chunk offset count does not match the manifest")
        if len(offsets):
            if int(offsets[0]) != 0:
                raise ValueError("The first chunk offset must be zero")
            if np.any(offsets[1:] <= offsets[:-1]):
                raise ValueError("Chunk offsets must be strictly increasing")
            if int(offsets[-1]) >= self.chunks_path.stat().st_size:
                raise ValueError("Chunk offsets point beyond the JSONL artifact")
        self._offsets = offsets

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def get(self, vector_id: int) -> ChunkRecord:
        return self.get_many([vector_id])[0]

    def get_many(self, vector_ids: Sequence[int]) -> list[ChunkRecord]:
        ids = [_validated_vector_id(value, len(self)) for value in vector_ids]
        records: list[ChunkRecord] = []
        with self.chunks_path.open("rb") as handle:
            for vector_id in ids:
                handle.seek(int(self._offsets[vector_id]))
                records.append(decode_chunk_record_line(handle.readline(), vector_id))
        return records

    def close(self) -> None:
        close_numpy_memmap(self._offsets)

    def __enter__(self) -> "JsonlOffsetChunkStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def as_chunk_store(chunks: ChunkStore | Iterable[ChunkRecord | dict]) -> ChunkStore:
    if all(hasattr(chunks, name) for name in ("__len__", "get", "get_many")):
        return chunks  # type: ignore[return-value]
    return InMemoryChunkStore(chunks)  # type: ignore[arg-type]
