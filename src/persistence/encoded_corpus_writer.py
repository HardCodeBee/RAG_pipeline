"""Memory-bounded construction of encoded-corpus artifacts."""

from __future__ import annotations

# Writes the encoded-corpus files without owning pipeline orchestration.

import os
import struct
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.persistence.artifact_io import (
    close_numpy_memmap,
    decode_chunk_record_line,
    describe_artifact,
    encode_jsonl_row,
)
from src.records import ChunkRecord
from src.provenance import sha256_file


class DocumentEmbedder(Protocol):
    """Minimal document-side embedding contract used by the builder."""

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EncodedCorpusArtifact:
    """Committed encoded-corpus files and their integrity metadata."""

    chunks_path: Path
    chunk_offsets_path: Path
    embeddings_path: Path
    rows: int
    dimension: int
    chunks_sha256: str
    chunk_offsets_sha256: str
    embeddings_sha256: str
    chunks_size_bytes: int
    chunk_offsets_size_bytes: int
    embeddings_size_bytes: int

    def artifact_descriptors(self) -> dict[str, dict[str, Any]]:
        """Return manifest-ready descriptors without adding record identities."""

        return {
            "chunks": describe_artifact(
                self.chunks_path,
                rows=self.rows,
                sha256=self.chunks_sha256,
                size_bytes=self.chunks_size_bytes,
            ),
            "chunk_offsets": describe_artifact(
                self.chunk_offsets_path,
                rows=self.rows,
                sha256=self.chunk_offsets_sha256,
                size_bytes=self.chunk_offsets_size_bytes,
                extra={"dtype": "uint64"},
            ),
            "embeddings": describe_artifact(
                self.embeddings_path,
                sha256=self.embeddings_sha256,
                size_bytes=self.embeddings_size_bytes,
                extra={
                    "shape": [self.rows, self.dimension],
                    "dtype": "float32",
                },
            ),
        }


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _temporary_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=suffix,
        dir=directory,
    )
    os.close(descriptor)
    return Path(raw_path)


def _flush_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    # Windows' file commit operation requires a writable descriptor.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _write_chunks_and_raw_offsets(
    chunks: Iterable[ChunkRecord],
    chunks_temp: Path,
    raw_offsets_temp: Path,
) -> int:
    rows = 0
    with chunks_temp.open("wb") as chunks_handle, raw_offsets_temp.open(
        "wb"
    ) as offsets_handle:
        for item in chunks:
            if not isinstance(item, ChunkRecord):
                raise TypeError("chunks must yield ChunkRecord values")
            if item.vector_id != rows:
                raise ValueError(
                    "Chunk vector ids must be the zero-based contiguous sequence; "
                    f"expected {rows}, found {item.vector_id}"
                )

            offset = chunks_handle.tell()
            offsets_handle.write(struct.pack("<Q", offset))
            encoded = encode_jsonl_row(item.to_dict(), canonical=False) + b"\n"
            chunks_handle.write(encoded)
            rows += 1

        _flush_file(chunks_handle)
        _flush_file(offsets_handle)

    if rows <= 0:
        raise ValueError("chunks must contain at least one ChunkRecord")
    return rows


def _write_offset_array(
    raw_offsets_temp: Path,
    offsets_temp: Path,
    *,
    rows: int,
) -> None:
    expected_size = rows * np.dtype("<u8").itemsize
    if raw_offsets_temp.stat().st_size != expected_size:
        raise RuntimeError("Raw chunk offset count does not match chunk rows")

    source: np.memmap | None = None
    destination: np.memmap | None = None
    try:
        source = np.memmap(
            raw_offsets_temp,
            dtype="<u8",
            mode="r",
            shape=(rows,),
        )
        destination = np.lib.format.open_memmap(
            offsets_temp,
            mode="w+",
            dtype=np.uint64,
            shape=(rows,),
        )
        copy_rows = 1_000_000
        for start in range(0, rows, copy_rows):
            end = min(start + copy_rows, rows)
            destination[start:end] = source[start:end]
    finally:
        close_numpy_memmap(destination, flush=True)
        close_numpy_memmap(source, flush=True)
    _fsync_path(offsets_temp)


def _encode_batch(
    embedder: DocumentEmbedder,
    texts: list[str],
    *,
    dimension: int,
) -> np.ndarray:
    embeddings = np.asarray(
        embedder.encode_documents(texts),
        dtype=np.float32,
    )
    expected_shape = (len(texts), dimension)
    if embeddings.ndim != 2 or embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding batch has shape {embeddings.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding batch contains non-finite values")
    return np.ascontiguousarray(embeddings)


def _write_embedding_array(
    chunks_temp: Path,
    embeddings_temp: Path,
    embedder: DocumentEmbedder,
    *,
    rows: int,
    dimension: int,
    batch_size: int,
) -> None:
    destination: np.memmap | None = None
    written = 0
    texts: list[str] = []
    try:
        destination = np.lib.format.open_memmap(
            embeddings_temp,
            mode="w+",
            dtype=np.float32,
            shape=(rows, dimension),
        )
        with chunks_temp.open("rb") as handle:
            for expected_vector_id, raw in enumerate(handle):
                record = decode_chunk_record_line(raw, expected_vector_id)
                texts.append(record.text)
                if len(texts) < batch_size:
                    continue
                embeddings = _encode_batch(
                    embedder,
                    texts,
                    dimension=dimension,
                )
                destination[written : written + len(texts)] = embeddings
                written += len(texts)
                texts.clear()

        if texts:
            embeddings = _encode_batch(
                embedder,
                texts,
                dimension=dimension,
            )
            destination[written : written + len(texts)] = embeddings
            written += len(texts)
            texts.clear()

        if written != rows:
            raise RuntimeError(
                f"Embedded {written} chunk rows; expected {rows}"
            )
    finally:
        close_numpy_memmap(destination, flush=True)
    _fsync_path(embeddings_temp)


def _validate_temporary_artifacts(
    chunks_temp: Path,
    offsets_temp: Path,
    embeddings_temp: Path,
    *,
    rows: int,
    dimension: int,
) -> None:
    offsets: np.memmap | None = None
    embeddings: np.memmap | None = None
    try:
        offsets = np.load(offsets_temp, mmap_mode="r", allow_pickle=False)
        if offsets.ndim != 1 or offsets.shape != (rows,):
            raise RuntimeError("Chunk offset artifact shape is invalid")
        if offsets.dtype != np.dtype("uint64"):
            raise RuntimeError("Chunk offset artifact dtype must be uint64")
        if int(offsets[0]) != 0:
            raise RuntimeError("The first chunk offset must be zero")
        if rows > 1 and np.any(offsets[1:] <= offsets[:-1]):
            raise RuntimeError("Chunk offsets must be strictly increasing")
        if int(offsets[-1]) >= chunks_temp.stat().st_size:
            raise RuntimeError("The final chunk offset is outside chunks.jsonl")

        embeddings = np.load(
            embeddings_temp,
            mmap_mode="r",
            allow_pickle=False,
        )
        if embeddings.shape != (rows, dimension):
            raise RuntimeError("Embedding artifact shape is invalid")
        if embeddings.dtype != np.dtype("float32"):
            raise RuntimeError("Embedding artifact dtype must be float32")
    finally:
        close_numpy_memmap(embeddings)
        close_numpy_memmap(offsets)


def write_encoded_corpus(
    chunks: Iterable[ChunkRecord],
    embedder: DocumentEmbedder,
    output_dir: str | Path,
    *,
    batch_size: int = 128,
) -> EncodedCorpusArtifact:
    """Build and commit reusable chunks, offsets, and embeddings.

    ``chunks`` is consumed exactly once. Chunk text is then read back from the
    temporary JSONL artifact in bounded batches for document encoding, avoiding
    an in-memory corpus-sized list. All file handles and memory maps are flushed
    and explicitly closed before the temporary files are renamed, which is
    required for reliable commits on Windows.
    """

    batch_size = _positive_integer("batch_size", batch_size)
    dimension = _positive_integer(
        "embedder.dimension",
        getattr(embedder, "dimension", None),
    )
    if not callable(getattr(embedder, "encode_documents", None)):
        raise TypeError("embedder must provide encode_documents(texts)")

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    chunks_path = directory / "chunks.jsonl"
    offsets_path = directory / "chunk_offsets.npy"
    embeddings_path = directory / "embeddings.npy"
    final_paths = (chunks_path, offsets_path, embeddings_path)
    existing = [path for path in final_paths if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Encoded-corpus artifact already exists: {names}")

    chunks_temp = _temporary_path(
        directory,
        prefix=".chunks-",
        suffix=".jsonl.tmp",
    )
    raw_offsets_temp = _temporary_path(
        directory,
        prefix=".chunk-offsets-",
        suffix=".u64.tmp",
    )
    offsets_temp = _temporary_path(
        directory,
        prefix=".chunk-offsets-",
        suffix=".npy.tmp",
    )
    embeddings_temp = _temporary_path(
        directory,
        prefix=".embeddings-",
        suffix=".npy.tmp",
    )
    temporary_paths = (
        chunks_temp,
        raw_offsets_temp,
        offsets_temp,
        embeddings_temp,
    )
    committed_paths: list[Path] = []

    try:
        rows = _write_chunks_and_raw_offsets(
            chunks,
            chunks_temp,
            raw_offsets_temp,
        )
        _write_offset_array(
            raw_offsets_temp,
            offsets_temp,
            rows=rows,
        )
        raw_offsets_temp.unlink()
        _write_embedding_array(
            chunks_temp,
            embeddings_temp,
            embedder,
            rows=rows,
            dimension=dimension,
            batch_size=batch_size,
        )
        _validate_temporary_artifacts(
            chunks_temp,
            offsets_temp,
            embeddings_temp,
            rows=rows,
            dimension=dimension,
        )

        for temporary, final in (
            (chunks_temp, chunks_path),
            (offsets_temp, offsets_path),
            (embeddings_temp, embeddings_path),
        ):
            os.replace(temporary, final)
            committed_paths.append(final)

        return EncodedCorpusArtifact(
            chunks_path=chunks_path,
            chunk_offsets_path=offsets_path,
            embeddings_path=embeddings_path,
            rows=rows,
            dimension=dimension,
            chunks_sha256=sha256_file(chunks_path),
            chunk_offsets_sha256=sha256_file(offsets_path),
            embeddings_sha256=sha256_file(embeddings_path),
            chunks_size_bytes=chunks_path.stat().st_size,
            chunk_offsets_size_bytes=offsets_path.stat().st_size,
            embeddings_size_bytes=embeddings_path.stat().st_size,
        )
    except Exception:
        for path in committed_paths:
            if path.is_file():
                path.unlink()
        raise
    finally:
        for path in temporary_paths:
            if path.is_file():
                path.unlink()
