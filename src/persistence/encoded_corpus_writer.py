"""Memory-bounded construction of encoded-corpus artifacts."""

from __future__ import annotations

# Writes the encoded-corpus files without owning pipeline orchestration.

import json
import os
import struct
import tempfile
from collections.abc import Iterable, Mapping, Sequence
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
    embedding_parts: tuple[dict[str, Any], ...] | None = None

    def artifact_descriptors(self) -> dict[str, dict[str, Any]]:
        """Return manifest-ready descriptors without adding record identities."""

        embeddings_descriptor = describe_artifact(
            self.embeddings_path,
            sha256=self.embeddings_sha256,
            size_bytes=self.embeddings_size_bytes,
            extra={
                "shape": [self.rows, self.dimension],
                "dtype": "float32",
            },
        )
        if self.embedding_parts is not None:
            embeddings_descriptor.update(
                {
                    "file": self.embeddings_path.relative_to(
                        self.chunks_path.parent
                    ).as_posix(),
                    "storage": "sharded_npy",
                    "schema_version": 1,
                    "parts": [dict(part) for part in self.embedding_parts],
                }
            )

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
            "embeddings": embeddings_descriptor,
        }


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_shard_rows(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "embedding_shard_rows must be a non-negative integer or None"
        )
    value = int(value)
    if value < 0:
        raise ValueError(
            "embedding_shard_rows must be a non-negative integer or None"
        )
    return value or None


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


def _write_json_object(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        _flush_file(handle)


def _atomic_write_json_object(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_path(
        path.parent,
        prefix=f".{path.name}-",
        suffix=".tmp",
    )
    try:
        _write_json_object(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


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
    encode_call_rows: int,
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
                if len(texts) < encode_call_rows:
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


def _validate_chunk_artifacts(
    chunks_temp: Path,
    offsets_temp: Path,
    *,
    rows: int,
) -> None:
    offsets: np.memmap | None = None
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
    finally:
        close_numpy_memmap(offsets)


def _validate_temporary_artifacts(
    chunks_temp: Path,
    offsets_temp: Path,
    embeddings_temp: Path,
    *,
    rows: int,
    dimension: int,
) -> None:
    embeddings: np.memmap | None = None
    _validate_chunk_artifacts(chunks_temp, offsets_temp, rows=rows)

    try:
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


def _part_file_name(index: int) -> str:
    return f"part-{index:05d}.npy"


def _part_geometry(
    index: int,
    *,
    rows: int,
    shard_rows: int,
) -> tuple[int, int]:
    start_row = index * shard_rows
    return start_row, min(shard_rows, rows - start_row)


def _validate_embedding_part(
    path: Path,
    *,
    file_name: str,
    start_row: int,
    part_rows: int,
    dimension: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Embedding shard is missing: {path}")

    embeddings: np.memmap | None = None
    try:
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (part_rows, dimension)
        if embeddings.ndim != 2 or embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Embedding shard {file_name} has shape {embeddings.shape}; "
                f"expected {expected_shape}"
            )
        if embeddings.dtype != np.dtype("float32"):
            raise RuntimeError(
                f"Embedding shard {file_name} dtype must be float32"
            )

        check_rows = max(1, 1_000_000 // dimension)
        for block_start in range(0, part_rows, check_rows):
            block_end = min(block_start + check_rows, part_rows)
            if not np.isfinite(embeddings[block_start:block_end]).all():
                raise RuntimeError(
                    f"Embedding shard {file_name} contains non-finite values"
                )
    finally:
        close_numpy_memmap(embeddings)

    return {
        "file": file_name,
        "start_row": start_row,
        "rows": part_rows,
        "shape": [part_rows, dimension],
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_embedding_part(
    chunks_path: Path,
    offsets_path: Path,
    destination_path: Path,
    embedder: DocumentEmbedder,
    *,
    start_row: int,
    part_rows: int,
    dimension: int,
    encode_call_rows: int,
) -> None:
    offsets: np.memmap | None = None
    destination: np.memmap | None = None
    texts: list[str] = []
    written = 0
    try:
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        destination = np.lib.format.open_memmap(
            destination_path,
            mode="w+",
            dtype=np.float32,
            shape=(part_rows, dimension),
        )
        with chunks_path.open("rb") as handle:
            handle.seek(int(offsets[start_row]))
            for expected_vector_id in range(
                start_row,
                start_row + part_rows,
            ):
                record = decode_chunk_record_line(
                    handle.readline(),
                    expected_vector_id,
                )
                texts.append(record.text)
                if len(texts) < encode_call_rows:
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

        if written != part_rows:
            raise RuntimeError(
                f"Embedded {written} shard rows; expected {part_rows}"
            )
    finally:
        close_numpy_memmap(destination, flush=True)
        close_numpy_memmap(offsets)
    _fsync_path(destination_path)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Embedding shard checkpoint is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Embedding shard checkpoint must be a JSON object")
    return value


def _checkpoint_state(
    *,
    chunks_sha256: str,
    rows: int,
    dimension: int,
    shard_rows: int,
    parts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "chunks_sha256": chunks_sha256,
        "dtype": "float32",
        "dimension": dimension,
        "rows": rows,
        "shard_rows": shard_rows,
        "parts": [dict(part) for part in parts],
    }


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _validate_checkpoint_identity(
    state: Mapping[str, Any],
    *,
    chunks_sha256: str,
    rows: int,
    dimension: int,
    shard_rows: int,
) -> list[dict[str, Any]]:
    expected = _checkpoint_state(
        chunks_sha256=chunks_sha256,
        rows=rows,
        dimension=dimension,
        shard_rows=shard_rows,
    )
    expected_keys = set(expected)
    if set(state) != expected_keys:
        raise ValueError("Embedding shard checkpoint fields are invalid")

    for name in expected_keys - {"parts"}:
        if not _strict_json_equal(state.get(name), expected[name]):
            raise ValueError(
                f"Embedding shard checkpoint {name} does not match the input"
            )

    parts = state.get("parts")
    if not isinstance(parts, list) or not all(
        isinstance(part, dict) for part in parts
    ):
        raise ValueError("Embedding shard checkpoint parts must be objects")
    return [dict(part) for part in parts]


def _part_index(path: Path) -> int | None:
    name = path.name
    prefix = "part-"
    suffix = ".npy"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    digits = name[len(prefix) : -len(suffix)]
    if len(digits) < 5 or not digits.isdigit():
        return None
    index = int(digits)
    if name != _part_file_name(index):
        return None
    return index


def _is_shard_temporary_file(path: Path) -> bool:
    return path.is_file() and path.name.startswith(
        (".part-", ".checkpoint.json-", ".manifest.json-")
    ) and path.name.endswith(".tmp")


def _load_or_initialize_checkpoint(
    embeddings_directory: Path,
    *,
    chunks_sha256: str,
    rows: int,
    dimension: int,
    shard_rows: int,
) -> tuple[Path, dict[str, Any]]:
    if embeddings_directory.exists() and not embeddings_directory.is_dir():
        raise FileExistsError(
            f"Embedding shard path is not a directory: {embeddings_directory}"
        )
    embeddings_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = embeddings_directory / "checkpoint.json"

    if checkpoint_path.exists():
        if not checkpoint_path.is_file():
            raise ValueError("Embedding shard checkpoint is not a file")
        state = _read_checkpoint(checkpoint_path)
        tracked_parts = _validate_checkpoint_identity(
            state,
            chunks_sha256=chunks_sha256,
            rows=rows,
            dimension=dimension,
            shard_rows=shard_rows,
        )
    else:
        existing = list(embeddings_directory.iterdir())
        if existing and all(_is_shard_temporary_file(path) for path in existing):
            for path in existing:
                path.unlink()
            existing = []
        if existing:
            raise FileExistsError(
                "Embedding shard directory is non-empty but has no checkpoint"
            )
        state = _checkpoint_state(
            chunks_sha256=chunks_sha256,
            rows=rows,
            dimension=dimension,
            shard_rows=shard_rows,
        )
        _atomic_write_json_object(checkpoint_path, state)
        tracked_parts = []

    for path in tuple(embeddings_directory.iterdir()):
        if _is_shard_temporary_file(path):
            path.unlink()

    indexed_parts: list[tuple[int, Path]] = []
    for path in embeddings_directory.iterdir():
        if path == checkpoint_path:
            continue
        index = _part_index(path)
        if index is None or not path.is_file():
            raise ValueError(
                f"Unexpected file in embedding shard directory: {path.name}"
            )
        indexed_parts.append((index, path))

    indexed_parts.sort(key=lambda item: item[0])
    part_count = (rows + shard_rows - 1) // shard_rows
    if len(indexed_parts) > part_count:
        raise ValueError("Embedding shard checkpoint contains too many parts")
    for expected_index, (index, _) in enumerate(indexed_parts):
        if index != expected_index:
            raise ValueError(
                "Embedding shard files must be a contiguous prefix starting at zero"
            )
    if len(tracked_parts) > len(indexed_parts):
        raise ValueError("Embedding shard checkpoint references a missing part")

    for index, claimed in enumerate(tracked_parts):
        start_row, part_rows = _part_geometry(
            index,
            rows=rows,
            shard_rows=shard_rows,
        )
        actual = _validate_embedding_part(
            indexed_parts[index][1],
            file_name=_part_file_name(index),
            start_row=start_row,
            part_rows=part_rows,
            dimension=dimension,
        )
        if not _strict_json_equal(claimed, actual):
            raise ValueError(
                f"Embedding shard checkpoint does not match {_part_file_name(index)}"
            )

    for index in range(len(tracked_parts), len(indexed_parts)):
        start_row, part_rows = _part_geometry(
            index,
            rows=rows,
            shard_rows=shard_rows,
        )
        actual = _validate_embedding_part(
            indexed_parts[index][1],
            file_name=_part_file_name(index),
            start_row=start_row,
            part_rows=part_rows,
            dimension=dimension,
        )
        tracked_parts.append(actual)
        state["parts"] = [dict(part) for part in tracked_parts]
        _atomic_write_json_object(checkpoint_path, state)

    state["parts"] = [dict(part) for part in tracked_parts]
    return checkpoint_path, state


def _write_sharded_embeddings(
    chunks_path: Path,
    offsets_path: Path,
    embeddings_directory: Path,
    embedder: DocumentEmbedder,
    *,
    chunks_sha256: str,
    rows: int,
    dimension: int,
    encode_call_rows: int,
    shard_rows: int,
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    checkpoint_path, state = _load_or_initialize_checkpoint(
        embeddings_directory,
        chunks_sha256=chunks_sha256,
        rows=rows,
        dimension=dimension,
        shard_rows=shard_rows,
    )
    parts = [dict(part) for part in state["parts"]]
    part_count = (rows + shard_rows - 1) // shard_rows

    for index in range(len(parts), part_count):
        file_name = _part_file_name(index)
        part_path = embeddings_directory / file_name
        start_row, part_rows = _part_geometry(
            index,
            rows=rows,
            shard_rows=shard_rows,
        )
        temporary = _temporary_path(
            embeddings_directory,
            prefix=f".{file_name}-",
            suffix=".tmp",
        )
        try:
            _write_embedding_part(
                chunks_path,
                offsets_path,
                temporary,
                embedder,
                start_row=start_row,
                part_rows=part_rows,
                dimension=dimension,
                encode_call_rows=encode_call_rows,
            )
            descriptor = _validate_embedding_part(
                temporary,
                file_name=file_name,
                start_row=start_row,
                part_rows=part_rows,
                dimension=dimension,
            )
            os.replace(temporary, part_path)
            parts.append(descriptor)
            state["parts"] = [dict(part) for part in parts]
            _atomic_write_json_object(checkpoint_path, state)
        finally:
            if temporary.is_file():
                temporary.unlink()

    return checkpoint_path, tuple(parts)


def _commit_or_validate_file(
    temporary: Path,
    final: Path,
    *,
    sha256: str,
    size_bytes: int,
) -> bool:
    if final.exists():
        if (
            not final.is_file()
            or final.stat().st_size != size_bytes
            or sha256_file(final) != sha256
        ):
            raise FileExistsError(
                f"Existing recovery artifact does not match: {final.name}"
            )
        temporary.unlink()
        return False
    os.replace(temporary, final)
    return True


def write_encoded_corpus(
    chunks: Iterable[ChunkRecord],
    embedder: DocumentEmbedder,
    output_dir: str | Path,
    *,
    batch_size: int = 128,
    encode_call_rows: int | None = None,
    embedding_shard_rows: int | None = None,
) -> EncodedCorpusArtifact:
    """Build and commit reusable chunks, offsets, and embeddings.

    ``chunks`` is consumed exactly once. Chunk text is then read back from the
    temporary JSONL artifact in groups of at most ``encode_call_rows`` for
    document encoding, avoiding an in-memory corpus-sized list. If omitted,
    ``encode_call_rows`` defaults to the legacy ``batch_size`` behavior. The
    embedder still owns its internal model batch size. All file handles and
    memory maps are flushed and explicitly closed before the temporary files
    are renamed, which is required for reliable commits on Windows. When
    ``embedding_shard_rows`` is positive, each completed embedding part is
    committed independently and can be reused after a failed process;
    ``embeddings/manifest.json`` is committed only after every part and the
    unchanged chunk artifacts are complete.
    """

    batch_size = _positive_integer("batch_size", batch_size)
    encode_call_rows = _positive_integer(
        "encode_call_rows",
        batch_size if encode_call_rows is None else encode_call_rows,
    )
    shard_rows = _optional_shard_rows(embedding_shard_rows)
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
    single_embeddings_path = directory / "embeddings.npy"
    embeddings_directory = directory / "embeddings"
    embeddings_manifest_path = embeddings_directory / "manifest.json"
    checkpoint_candidate = embeddings_directory / "checkpoint.json"

    if shard_rows is None:
        final_paths = (chunks_path, offsets_path, single_embeddings_path)
        existing = [path for path in final_paths if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Encoded-corpus artifact already exists: {names}"
            )
    else:
        if embeddings_directory.exists() and not embeddings_directory.is_dir():
            raise FileExistsError(
                f"Embedding shard path is not a directory: {embeddings_directory}"
            )
        complete_paths = (
            single_embeddings_path,
            embeddings_manifest_path,
        )
        existing = [path for path in complete_paths if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Encoded-corpus artifact already exists: {names}"
            )
        existing_chunk_paths = [
            path for path in (chunks_path, offsets_path) if path.exists()
        ]
        if existing_chunk_paths and not checkpoint_candidate.is_file():
            names = ", ".join(path.name for path in existing_chunk_paths)
            raise FileExistsError(
                "Encoded-corpus chunk artifact exists without a shard "
                f"checkpoint: {names}"
            )

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
    temporary_paths = [
        chunks_temp,
        raw_offsets_temp,
        offsets_temp,
    ]
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

        if shard_rows is None:
            embeddings_temp = _temporary_path(
                directory,
                prefix=".embeddings-",
                suffix=".npy.tmp",
            )
            temporary_paths.append(embeddings_temp)
            _write_embedding_array(
                chunks_temp,
                embeddings_temp,
                embedder,
                rows=rows,
                dimension=dimension,
                encode_call_rows=encode_call_rows,
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
                (embeddings_temp, single_embeddings_path),
            ):
                os.replace(temporary, final)
                committed_paths.append(final)

            return EncodedCorpusArtifact(
                chunks_path=chunks_path,
                chunk_offsets_path=offsets_path,
                embeddings_path=single_embeddings_path,
                rows=rows,
                dimension=dimension,
                chunks_sha256=sha256_file(chunks_path),
                chunk_offsets_sha256=sha256_file(offsets_path),
                embeddings_sha256=sha256_file(single_embeddings_path),
                chunks_size_bytes=chunks_path.stat().st_size,
                chunk_offsets_size_bytes=offsets_path.stat().st_size,
                embeddings_size_bytes=single_embeddings_path.stat().st_size,
            )

        _validate_chunk_artifacts(chunks_temp, offsets_temp, rows=rows)
        chunks_sha256 = sha256_file(chunks_temp)
        offsets_sha256 = sha256_file(offsets_temp)
        chunks_size_bytes = chunks_temp.stat().st_size
        offsets_size_bytes = offsets_temp.stat().st_size
        checkpoint_path, embedding_parts = _write_sharded_embeddings(
            chunks_temp,
            offsets_temp,
            embeddings_directory,
            embedder,
            chunks_sha256=chunks_sha256,
            rows=rows,
            dimension=dimension,
            encode_call_rows=encode_call_rows,
            shard_rows=shard_rows,
        )

        embeddings_manifest = {
            "schema_version": 1,
            "dtype": "float32",
            "dimension": dimension,
            "rows": rows,
            "parts": [dict(part) for part in embedding_parts],
        }
        manifest_temp = _temporary_path(
            embeddings_directory,
            prefix=".manifest.json-",
            suffix=".tmp",
        )
        temporary_paths.append(manifest_temp)
        _write_json_object(manifest_temp, embeddings_manifest)
        manifest_sha256 = sha256_file(manifest_temp)
        manifest_size_bytes = manifest_temp.stat().st_size

        artifact = EncodedCorpusArtifact(
            chunks_path=chunks_path,
            chunk_offsets_path=offsets_path,
            embeddings_path=embeddings_manifest_path,
            rows=rows,
            dimension=dimension,
            chunks_sha256=chunks_sha256,
            chunk_offsets_sha256=offsets_sha256,
            embeddings_sha256=manifest_sha256,
            chunks_size_bytes=chunks_size_bytes,
            chunk_offsets_size_bytes=offsets_size_bytes,
            embeddings_size_bytes=manifest_size_bytes,
            embedding_parts=tuple(dict(part) for part in embedding_parts),
        )

        for temporary, final, digest, size_bytes in (
            (
                chunks_temp,
                chunks_path,
                chunks_sha256,
                chunks_size_bytes,
            ),
            (
                offsets_temp,
                offsets_path,
                offsets_sha256,
                offsets_size_bytes,
            ),
        ):
            if _commit_or_validate_file(
                temporary,
                final,
                sha256=digest,
                size_bytes=size_bytes,
            ):
                committed_paths.append(final)

        if embeddings_manifest_path.exists():
            raise FileExistsError(
                "Encoded-corpus embedding manifest already exists"
            )
        os.replace(manifest_temp, embeddings_manifest_path)
        try:
            checkpoint_path.unlink()
        except OSError:
            pass
        return artifact
    except Exception:
        for path in committed_paths:
            if path.is_file():
                path.unlink()
        raise
    finally:
        for path in temporary_paths:
            if path.is_file():
                path.unlink()
