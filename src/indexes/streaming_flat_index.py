"""Memory-bounded exact inner-product search over disk-backed embeddings.

Unlike :mod:`src.indexes.vector_index`, this backend never constructs or loads
a corpus-sized FAISS index.  It memory-maps one embedding part at a time,
builds a temporary ``faiss.IndexFlatIP`` for a bounded row block, merges that
block's exact top-k results, and releases the temporary index before advancing.

The source may be either a legacy ``embeddings.npy`` file or a directory of
consecutively numbered ``part-*.npy`` files.  A directory-level
``manifest.json`` is optional for legacy compatibility; when present it is the
authoritative part list and is validated strictly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.records import VectorHit


_PART_NAME = re.compile(r"^part-(\d+)\.npy$")
_MANIFEST_NAME = "manifest.json"
_MANIFEST_SCHEMA_VERSION = 1
_FLOAT32 = np.dtype(np.float32)
_HASH_CHUNK_BYTES = 1024 * 1024
_FINITE_VALIDATION_ROWS = 8192


@dataclass(frozen=True, slots=True)
class EmbeddingShard:
    """One validated, contiguous range in the global embedding row space."""

    path: Path
    start_row: int
    rows: int
    dimension: int

    @property
    def stop_row(self) -> int:
        return self.start_row + self.rows


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid embedding manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Embedding manifest must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _close_memmap(values: np.ndarray) -> None:
    mmap = getattr(values, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _open_npy_memmap(path: Path) -> np.ndarray:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid NumPy embedding file: {path}") from exc
    if not isinstance(values, np.ndarray):
        raise ValueError(f"Embedding file must contain one NumPy array: {path}")
    return values


def _validate_array_structure(
    values: np.ndarray,
    *,
    path: Path,
    expected_rows: int | None = None,
    expected_dimension: int | None = None,
) -> tuple[int, int]:
    if values.ndim != 2:
        raise ValueError(f"Embedding array must be 2D: {path}")
    rows, dimension = map(int, values.shape)
    if rows <= 0 or dimension <= 0:
        raise ValueError(f"Embedding array must be non-empty: {path}")
    if values.dtype != _FLOAT32:
        raise ValueError(
            f"Embedding array must have dtype float32, found {values.dtype}: {path}"
        )
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"Embedding row count does not match manifest for {path}: "
            f"expected {expected_rows}, found {rows}"
        )
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Embedding dimension does not match for {path}: "
            f"expected {expected_dimension}, found {dimension}"
        )
    return rows, dimension


def _validate_finite(values: np.ndarray, *, path: Path) -> None:
    for start in range(0, int(values.shape[0]), _FINITE_VALIDATION_ROWS):
        stop = min(start + _FINITE_VALIDATION_ROWS, int(values.shape[0]))
        if not np.isfinite(values[start:stop]).all():
            raise ValueError(f"Embedding array must contain only finite values: {path}")


def _validated_file_shape(
    path: Path,
    *,
    expected_rows: int | None = None,
    expected_dimension: int | None = None,
) -> tuple[int, int]:
    values = _open_npy_memmap(path)
    try:
        shape = _validate_array_structure(
            values,
            path=path,
            expected_rows=expected_rows,
            expected_dimension=expected_dimension,
        )
        _validate_finite(values, path=path)
        return shape
    finally:
        _close_memmap(values)


def _part_number(filename: str) -> int:
    match = _PART_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Embedding part must be named part-<number>.npy: {filename}")
    return int(match.group(1))


def _manifest_parts(directory: Path, manifest_path: Path) -> tuple[EmbeddingShard, ...]:
    manifest = _load_json_mapping(manifest_path)
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Embedding manifest schema_version must be {_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("dtype") != "float32":
        raise ValueError("Embedding manifest dtype must be float32")

    dimension = _positive_integer("manifest.dimension", manifest.get("dimension"))
    declared_total_rows = manifest.get("total_rows")
    declared_rows = manifest.get("rows")
    if declared_total_rows is None and declared_rows is None:
        raise ValueError("Embedding manifest must define total_rows or rows")
    if declared_total_rows is not None and declared_rows is not None:
        total_rows = _positive_integer("manifest.total_rows", declared_total_rows)
        rows_alias = _positive_integer("manifest.rows", declared_rows)
        if rows_alias != total_rows:
            raise ValueError("Embedding manifest rows and total_rows must match")
    elif declared_total_rows is not None:
        total_rows = _positive_integer("manifest.total_rows", declared_total_rows)
    else:
        total_rows = _positive_integer("manifest.rows", declared_rows)
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Embedding manifest parts must be a non-empty list")

    directory_resolved = directory.resolve()
    expected_start = 0
    listed_files: list[str] = []
    shards: list[EmbeddingShard] = []

    for position, raw_part in enumerate(parts):
        label = f"manifest.parts[{position}]"
        if not isinstance(raw_part, Mapping):
            raise ValueError(f"{label} must be an object")

        filename = raw_part.get("file")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{label}.file must be a non-empty string")
        relative = Path(filename)
        if relative.is_absolute() or relative.name != filename:
            raise ValueError(f"{label}.file must name a file inside the embedding directory")
        if _part_number(filename) != position:
            raise ValueError("Embedding parts must be consecutively numbered from 0")
        if filename in listed_files:
            raise ValueError(f"Embedding manifest contains duplicate part: {filename}")

        path = (directory / relative).resolve()
        if path.parent != directory_resolved:
            raise ValueError(f"{label}.file escapes the embedding directory")
        if not path.is_file():
            raise FileNotFoundError(f"Embedding part does not exist: {path}")

        start_row = _non_negative_integer(f"{label}.start_row", raw_part.get("start_row"))
        if start_row != expected_start:
            raise ValueError(
                f"Embedding row ranges must be contiguous from 0: "
                f"expected start_row {expected_start}, found {start_row}"
            )
        rows = _positive_integer(f"{label}.rows", raw_part.get("rows"))
        shape = raw_part.get("shape")
        if not isinstance(shape, list) or shape != [rows, dimension]:
            raise ValueError(f"{label}.shape must equal [{rows}, {dimension}]")

        expected_hash = raw_part.get("sha256")
        if "sha256" in raw_part:
            if (
                not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash)
            ):
                raise ValueError(f"{label}.sha256 must be a 64-character hex digest")
            if _sha256(path) != expected_hash.lower():
                raise ValueError(f"Embedding SHA-256 mismatch: {path}")

        _validated_file_shape(
            path,
            expected_rows=rows,
            expected_dimension=dimension,
        )
        shards.append(
            EmbeddingShard(
                path=path,
                start_row=start_row,
                rows=rows,
                dimension=dimension,
            )
        )
        listed_files.append(filename)
        expected_start += rows

    if expected_start != total_rows:
        raise ValueError(
            "Embedding manifest total_rows does not match its contiguous part ranges"
        )

    discovered = {path.name for path in directory.glob("part-*.npy") if path.is_file()}
    if discovered != set(listed_files):
        raise ValueError("Embedding manifest part list does not match directory contents")
    return tuple(shards)


def _legacy_directory_parts(directory: Path) -> tuple[EmbeddingShard, ...]:
    numbered_paths: list[tuple[int, Path]] = []
    for path in directory.glob("part-*.npy"):
        if path.is_file():
            numbered_paths.append((_part_number(path.name), path.resolve()))
    numbered_paths.sort(key=lambda value: value[0])
    if not numbered_paths:
        raise FileNotFoundError(
            f"Embedding directory contains no part-*.npy files: {directory}"
        )
    numbers = [number for number, _ in numbered_paths]
    if numbers != list(range(len(numbered_paths))):
        raise ValueError("Embedding parts must be consecutively numbered from 0")

    expected_start = 0
    expected_dimension: int | None = None
    shards: list[EmbeddingShard] = []
    for _, path in numbered_paths:
        rows, dimension = _validated_file_shape(
            path,
            expected_dimension=expected_dimension,
        )
        if expected_dimension is None:
            expected_dimension = dimension
        shards.append(
            EmbeddingShard(
                path=path,
                start_row=expected_start,
                rows=rows,
                dimension=dimension,
            )
        )
        expected_start += rows
    return tuple(shards)


def _discover_shards(source: Path) -> tuple[EmbeddingShard, ...]:
    if source.is_file():
        if source.suffix.lower() != ".npy":
            raise ValueError("A legacy embedding source must be a .npy file")
        path = source.resolve()
        rows, dimension = _validated_file_shape(path)
        return (
            EmbeddingShard(
                path=path,
                start_row=0,
                rows=rows,
                dimension=dimension,
            ),
        )
    if not source.is_dir():
        raise FileNotFoundError(f"Embedding source does not exist: {source}")

    directory = source.resolve()
    manifest_path = directory / _MANIFEST_NAME
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise ValueError(f"Embedding manifest is not a file: {manifest_path}")
        return _manifest_parts(directory, manifest_path)
    return _legacy_directory_parts(directory)


def _require_faiss():
    try:
        import faiss
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "StreamingFlatIPIndex requires the optional faiss-cpu dependency"
        ) from exc
    return faiss


def _merge_top_k(
    best_scores: np.ndarray,
    best_ids: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    top_k: int,
) -> None:
    """Merge candidates in-place using score-descending/id-ascending order."""

    for row in range(best_scores.shape[0]):
        scores = np.concatenate((best_scores[row], candidate_scores[row]))
        ids = np.concatenate((best_ids[row], candidate_ids[row]))
        valid = ids >= 0
        scores = scores[valid]
        ids = ids[valid]
        order = np.lexsort((ids, -scores))[:top_k]
        selected = len(order)
        best_scores[row].fill(-np.inf)
        best_ids[row].fill(-1)
        best_scores[row, :selected] = scores[order]
        best_ids[row, :selected] = ids[order]


class StreamingFlatIPIndex:
    """Exact FAISS search whose resident corpus is bounded to one row block."""

    def __init__(
        self,
        embedding_source: str | Path | None = None,
        *,
        corpus_chunk_size: int = 50_000,
    ) -> None:
        self.embedding_source: Path | None = None
        self.requested_backend = "faiss"
        self.backend = "faiss"
        self.index_type = "streaming_flat_ip"
        self.index = None
        self.embeddings = None
        self.ids = None
        self.build_params: dict[str, int] = {}
        self.search_params: dict[str, int] = {}
        self.trained = True
        self.corpus_chunk_size = _positive_integer(
            "corpus_chunk_size",
            corpus_chunk_size,
        )
        self.shards: tuple[EmbeddingShard, ...] = ()
        self.dimension = 0
        self.count = 0
        if embedding_source is not None:
            self.load(embedding_source)

    @staticmethod
    def _validated_layout(
        shards: tuple[EmbeddingShard, ...],
    ) -> tuple[int, int]:
        dimensions = {shard.dimension for shard in shards}
        if len(dimensions) != 1:
            raise ValueError("All embedding shards must have the same dimension")
        dimension = dimensions.pop()
        count = sum(shard.rows for shard in shards)

        expected_start = 0
        for shard in shards:
            if shard.start_row != expected_start:
                raise ValueError("Embedding shard row ranges must be contiguous from 0")
            expected_start = shard.stop_row
        if expected_start != count:
            raise ValueError("Embedding shard row ranges do not match the total count")
        return dimension, count

    def load(self, path: str | Path) -> None:
        """Validate and attach a disk-backed embedding source without loading it."""

        source = Path(path)
        shards = _discover_shards(source)
        dimension, count = self._validated_layout(shards)
        self.embedding_source = source
        self.shards = shards
        self.dimension = dimension
        self.count = count

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "type": self.index_type,
            "count": self.count,
            "dimension": self.dimension,
            "trained": self.trained,
            "build_params": dict(self.build_params),
            "search_params": dict(self.search_params),
            "corpus_chunk_size": self.corpus_chunk_size,
            "shards": len(self.shards),
        }

    def _validate_queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("query_embeddings must be a 2D array")
        if queries.shape[0] <= 0:
            raise ValueError("query_embeddings must contain at least one vector")
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {queries.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        if not np.isfinite(queries).all():
            raise ValueError("query_embeddings must contain only finite values")
        return np.ascontiguousarray(queries)

    @staticmethod
    def _deterministic_local_results(
        *,
        block: np.ndarray,
        queries: np.ndarray,
        raw_scores: np.ndarray,
        raw_ids: np.ndarray,
        global_start: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve local cutoff ties deterministically without a Q-by-C matrix."""

        selected_scores = np.ascontiguousarray(raw_scores[:, :top_k])
        selected_ids = np.ascontiguousarray(raw_ids[:, :top_k], dtype=np.int64)
        rows = int(block.shape[0])

        # Asking FAISS for k+1 tells us whether the local cutoff intersects a
        # tie.  Only tied queries fall back to one bounded vector of exact
        # scores, avoiding a query-batch by corpus-block score matrix.
        if rows > top_k:
            tied_rows = np.flatnonzero(raw_scores[:, top_k - 1] == raw_scores[:, top_k])
            local_ids = np.arange(rows, dtype=np.int64)
            for query_row in tied_rows:
                full_scores = np.asarray(block @ queries[query_row], dtype=np.float32)
                order = np.lexsort((local_ids, -full_scores))[:top_k]
                selected_scores[query_row] = full_scores[order]
                selected_ids[query_row] = order

        selected_ids += global_start
        for row in range(selected_scores.shape[0]):
            order = np.lexsort((selected_ids[row], -selected_scores[row]))
            selected_scores[row] = selected_scores[row, order]
            selected_ids[row] = selected_ids[row, order]
        return selected_scores, selected_ids

    def search_many(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        query_batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search a fixed query set exactly while reading each corpus block once."""

        top_k = _positive_integer("top_k", top_k)
        query_batch_size = _positive_integer(
            "query_batch_size",
            query_batch_size,
        )
        if self.count <= 0 or self.dimension <= 0 or not self.shards:
            raise RuntimeError("Index has not been loaded")
        queries = self._validate_queries(query_embeddings)
        result_k = min(top_k, self.count)
        best_scores = np.full(
            (queries.shape[0], result_k),
            -np.inf,
            dtype=np.float32,
        )
        best_ids = np.full(
            (queries.shape[0], result_k),
            -1,
            dtype=np.int64,
        )
        faiss = _require_faiss()

        for shard in self.shards:
            values = _open_npy_memmap(shard.path)
            try:
                _validate_array_structure(
                    values,
                    path=shard.path,
                    expected_rows=shard.rows,
                    expected_dimension=self.dimension,
                )
                for local_start in range(0, shard.rows, self.corpus_chunk_size):
                    local_stop = min(
                        local_start + self.corpus_chunk_size,
                        shard.rows,
                    )
                    block = np.ascontiguousarray(values[local_start:local_stop])
                    if not np.isfinite(block).all():
                        raise ValueError(
                            "Embedding array must contain only finite values: "
                            f"{shard.path}"
                        )
                    index = faiss.IndexFlatIP(self.dimension)
                    try:
                        index.add(block)
                        local_k = min(result_k, int(block.shape[0]))
                        probe_k = min(int(block.shape[0]), local_k + 1)
                        global_start = shard.start_row + local_start
                        for query_start in range(
                            0,
                            int(queries.shape[0]),
                            query_batch_size,
                        ):
                            query_stop = min(
                                query_start + query_batch_size,
                                int(queries.shape[0]),
                            )
                            query_batch = queries[query_start:query_stop]
                            raw_scores, raw_ids = index.search(query_batch, probe_k)
                            local_scores, local_ids = self._deterministic_local_results(
                                block=block,
                                queries=query_batch,
                                raw_scores=raw_scores,
                                raw_ids=raw_ids,
                                global_start=global_start,
                                top_k=local_k,
                            )
                            _merge_top_k(
                                best_scores[query_start:query_stop],
                                best_ids[query_start:query_stop],
                                local_scores,
                                local_ids,
                                top_k=result_k,
                            )
                    finally:
                        index.reset()
                        del index
                    del block
            finally:
                _close_memmap(values)

        if np.any(best_ids < 0) or not np.isfinite(best_scores).all():
            raise RuntimeError("Streaming exact search did not produce a complete top-k")
        return best_scores, best_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-query compatibility wrapper around :meth:`search_many`."""

        self._reject_search_params(search_params)
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")
        scores, ids = self.search_many(query, top_k, query_batch_size=1)
        return scores[0], ids[0]

    def search_hits(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        scores, ids = self.search(
            query_embedding,
            top_k,
            search_params=search_params,
        )
        return [
            VectorHit(vector_id=int(vector_id), score=float(score))
            for score, vector_id in zip(scores, ids)
        ]

    @staticmethod
    def _reject_search_params(
        search_params: Mapping[str, Any] | None,
    ) -> None:
        if search_params is None:
            return
        if not isinstance(search_params, Mapping):
            raise ValueError("search_params must be a mapping")
        if search_params:
            raise ValueError(
                "streaming_flat_ip exact search does not accept search parameters"
            )

    def close(self) -> None:
        """No-op: all shard mappings and temporary indexes are scoped per search."""
