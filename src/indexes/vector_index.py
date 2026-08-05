"""Exact and approximate inner-product vector indexes.

The historical public name ``FlatIPIndex`` is retained for compatibility, but
the implementation now supports a small FAISS index family selected by
``index_type``:

* ``flat_ip``: exact inner-product search;
* ``hnsw_flat``: HNSW over uncompressed float vectors;
* ``ivf_flat``: inverted-file search over uncompressed float vectors;
* ``ivf_pq``: inverted-file search over product-quantized vectors.

Build-time parameters are fixed when an index is constructed. Search-time
parameters are supplied through FAISS ``SearchParameters`` on each call, so a
query can change ``nprobe`` or ``ef_search`` without mutating a shared index.
"""

from __future__ import annotations

# NumPy and FAISS vector-index implementations.

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.records import VectorHit


SUPPORTED_INDEX_TYPES = frozenset({"flat_ip", "hnsw_flat", "ivf_flat", "ivf_pq"})

_DEFAULT_BUILD_PARAMS: dict[str, dict[str, int]] = {
    "flat_ip": {},
    "hnsw_flat": {"m": 32, "ef_construction": 200},
    "ivf_flat": {"nlist": 1024},
    "ivf_pq": {"nlist": 1024, "m": 48, "nbits": 8},
}

_ALLOWED_SEARCH_PARAMS: dict[str, frozenset[str]] = {
    "flat_ip": frozenset(),
    "hnsw_flat": frozenset({"ef_search"}),
    "ivf_flat": frozenset({"nprobe", "max_codes"}),
    "ivf_pq": frozenset({"nprobe", "max_codes"}),
}


def _require_integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        comparator = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return value


def _mapping_copy(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _normalize_build_params(
    index_type: str,
    params: Mapping[str, Any] | None,
) -> dict[str, int]:
    supplied = _mapping_copy(params, label="build_params")
    allowed = set(_DEFAULT_BUILD_PARAMS[index_type])
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported build parameter(s) for {index_type}: {', '.join(unknown)}"
        )

    values = {**_DEFAULT_BUILD_PARAMS[index_type], **supplied}
    if index_type == "hnsw_flat":
        values["m"] = _require_integer("build_params.m", values["m"])
        values["ef_construction"] = _require_integer(
            "build_params.ef_construction",
            values["ef_construction"],
        )
    elif index_type == "ivf_flat":
        values["nlist"] = _require_integer("build_params.nlist", values["nlist"])
    elif index_type == "ivf_pq":
        values["nlist"] = _require_integer("build_params.nlist", values["nlist"])
        values["m"] = _require_integer("build_params.m", values["m"])
        values["nbits"] = _require_integer("build_params.nbits", values["nbits"])
        if values["nbits"] > 16:
            raise ValueError("build_params.nbits must be at most 16")
    return values


def _default_search_params(
    index_type: str,
    build_params: Mapping[str, int],
) -> dict[str, int]:
    if index_type == "hnsw_flat":
        return {"ef_search": 64}
    if index_type in {"ivf_flat", "ivf_pq"}:
        return {"nprobe": min(16, int(build_params["nlist"]))}
    return {}


def _normalize_search_params(
    index_type: str,
    params: Mapping[str, Any] | None,
    build_params: Mapping[str, int],
) -> dict[str, int]:
    supplied = _mapping_copy(params, label="search_params")
    allowed = _ALLOWED_SEARCH_PARAMS[index_type]
    unknown = sorted(set(supplied) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unsupported search parameter(s) for {index_type}: {', '.join(unknown)}"
        )

    values = {**_default_search_params(index_type, build_params), **supplied}
    if index_type == "hnsw_flat":
        values["ef_search"] = _require_integer(
            "search_params.ef_search",
            values["ef_search"],
        )
    elif index_type in {"ivf_flat", "ivf_pq"}:
        values["nprobe"] = _require_integer("search_params.nprobe", values["nprobe"])
        if values["nprobe"] > int(build_params["nlist"]):
            raise ValueError("search_params.nprobe cannot exceed build_params.nlist")
        if "max_codes" in values:
            values["max_codes"] = _require_integer(
                "search_params.max_codes",
                values["max_codes"],
                minimum=0,
            )
    return values


def _coerce_embeddings(embeddings: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError(f"{label} must contain at least one non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(values)


def _coerce_ids(ids: np.ndarray, *, expected_rows: int) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or values.shape[0] != expected_rows:
        raise ValueError("ids must be a 1D array with one id per embedding")
    if len(np.unique(values)) != len(values):
        raise ValueError("ids must be unique")
    return np.ascontiguousarray(values)


class FlatIPIndex:
    """A backward-compatible wrapper over exact and ANN index implementations."""

    def __init__(
        self,
        backend: str,
        index_type: str = "flat_ip",
        *,
        build_params: Mapping[str, Any] | None = None,
        search_params: Mapping[str, Any] | None = None,
    ):
        if backend not in {"faiss", "numpy"}:
            raise ValueError("backend must be one of: faiss, numpy")
        if index_type not in SUPPORTED_INDEX_TYPES:
            choices = ", ".join(sorted(SUPPORTED_INDEX_TYPES))
            raise ValueError(f"index_type must be one of: {choices}")
        if backend == "numpy" and index_type != "flat_ip":
            raise ValueError("NumPy backend supports only flat_ip")

        self.requested_backend = backend
        self.index_type = index_type
        self._build_params_explicit = build_params is not None
        self._build_params = _normalize_build_params(index_type, build_params)
        self._search_params_explicit = search_params is not None
        self._search_params = _normalize_search_params(
            index_type,
            search_params,
            self._build_params,
        )

        self.backend = ""
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self.dimension = 0
        self.count = 0

    @property
    def build_params(self) -> dict[str, int]:
        return dict(self._build_params)

    @property
    def search_params(self) -> dict[str, int]:
        return dict(self._search_params)

    def metadata(self) -> dict[str, Any]:
        """Return canonical index metadata suitable for a build/run manifest."""

        trained = False
        if self.backend == "faiss" and self.index is not None:
            trained = bool(self.index.is_trained)
        elif self.backend == "numpy" and self.embeddings is not None:
            trained = True
        return {
            "backend": self.backend or self.requested_backend,
            "type": self.index_type,
            "count": self.count,
            "dimension": self.dimension,
            "trained": trained,
            "build_params": self.build_params,
            "search_params": self.search_params,
        }

    def set_search_params(
        self,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Replace the default query-time parameters after strict validation."""

        values = _mapping_copy(params, label="search_params")
        duplicated = set(values).intersection(kwargs)
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(f"Duplicate search parameter(s): {names}")
        values.update(kwargs)
        self._search_params = _normalize_search_params(
            self.index_type,
            values,
            self._build_params,
        )
        self._search_params_explicit = True

    def _reset_runtime_state(self) -> None:
        self.backend = ""
        self.index = None
        self.embeddings = None
        self.ids = None
        self.dimension = 0
        self.count = 0

    def _initialize(self, dimension: int) -> None:
        dimension = _require_integer("dimension", dimension)
        if self.requested_backend == "numpy":
            self.backend = "numpy"
            self.dimension = dimension
            self.embeddings = np.empty((0, dimension), dtype=np.float32)
            self.ids = np.empty(0, dtype=np.int64)
            return

        import faiss

        if self.index_type == "flat_ip":
            base_index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "hnsw_flat":
            m = self._build_params["m"]
            base_index = faiss.IndexHNSWFlat(
                dimension,
                m,
                faiss.METRIC_INNER_PRODUCT,
            )
            base_index.hnsw.efConstruction = self._build_params["ef_construction"]
        elif self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(dimension)
            base_index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                self._build_params["nlist"],
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            m = self._build_params["m"]
            if dimension % m != 0:
                raise ValueError(
                    f"Embedding dimension {dimension} must be divisible by "
                    f"build_params.m={m} for ivf_pq"
                )
            quantizer = faiss.IndexFlatIP(dimension)
            base_index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                self._build_params["nlist"],
                m,
                self._build_params["nbits"],
                faiss.METRIC_INNER_PRODUCT,
            )

        self.index = faiss.IndexIDMap2(base_index)
        self.backend = "faiss"
        self.dimension = dimension
        self.ids = np.empty(0, dtype=np.int64)

    def _validate_dimension(self, embeddings: np.ndarray) -> None:
        if self.dimension and embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

    def _validate_training_rows(self, rows: int) -> None:
        if self.index_type not in {"ivf_flat", "ivf_pq"}:
            return
        nlist = self._build_params["nlist"]
        if rows < nlist:
            raise ValueError(
                f"{self.index_type} training requires at least nlist={nlist} vectors"
            )
        if self.index_type == "ivf_pq":
            codewords = 1 << self._build_params["nbits"]
            if rows < codewords:
                raise ValueError(
                    f"ivf_pq training requires at least 2**nbits={codewords} vectors"
                )

    def train(self, embeddings: np.ndarray) -> None:
        """Initialize and, when required, train the selected index family."""

        values = _coerce_embeddings(embeddings, label="training embeddings")
        if self.count:
            raise RuntimeError("Cannot train an index after vectors have been added")
        if self.index is None and self.embeddings is None:
            self._initialize(values.shape[1])
        self._validate_dimension(values)

        if self.requested_backend == "numpy":
            return
        if self.index is None:
            raise RuntimeError("FAISS index initialization failed")
        if self.index.is_trained:
            return
        self._validate_training_rows(values.shape[0])
        self.index.train(values)
        if not self.index.is_trained:
            raise RuntimeError("FAISS index training did not complete")

    def add_with_ids(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        """Add one vector batch while preserving caller-provided external IDs."""

        values = _coerce_embeddings(embeddings, label="embeddings")
        if self.index is None and self.embeddings is None:
            self._initialize(values.shape[1])
        self._validate_dimension(values)

        if ids is None:
            if self.count and (
                self.ids is None
                or not np.array_equal(self.ids, np.arange(self.count, dtype=np.int64))
            ):
                raise ValueError(
                    "ids are required when appending to an index with custom ids"
                )
            ids = np.arange(
                self.count,
                self.count + values.shape[0],
                dtype=np.int64,
            )
        id_values = _coerce_ids(ids, expected_rows=values.shape[0])
        if self.ids is not None and self.ids.size:
            overlap = np.intersect1d(self.ids, id_values, assume_unique=True)
            if overlap.size:
                raise ValueError("ids must be unique across all added batches")

        if self.requested_backend == "numpy":
            expected = np.arange(
                self.count,
                self.count + values.shape[0],
                dtype=np.int64,
            )
            if not np.array_equal(id_values, expected):
                raise ValueError(
                    "NumPy backend requires zero-based row-aligned vector ids"
                )
            if self.embeddings is None:
                raise RuntimeError("NumPy index initialization failed")
            self.embeddings = np.concatenate((self.embeddings, values), axis=0)
        else:
            if self.index is None:
                raise RuntimeError("FAISS index initialization failed")
            if not self.index.is_trained:
                raise RuntimeError(
                    f"{self.index_type} must be trained before vectors are added"
                )
            self.index.add_with_ids(values, id_values)

        existing_ids = (
            self.ids
            if self.ids is not None
            else np.empty(0, dtype=np.int64)
        )
        self.ids = np.ascontiguousarray(
            np.concatenate((existing_ids, id_values)),
            dtype=np.int64,
        )
        self.count = int(self.ids.shape[0])
        if self.index is not None and int(self.index.ntotal) != self.count:
            raise RuntimeError("FAISS vector count does not match tracked ids")

    def build(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        """Build a fresh index in one call, preserving the historical API."""

        values = _coerce_embeddings(embeddings, label="embeddings")
        self._reset_runtime_state()
        self.train(values)
        self.add_with_ids(values, ids)

    def _resolved_search_params(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, int]:
        if overrides is None:
            return self.search_params
        merged: dict[str, Any] = self.search_params
        merged.update(_mapping_copy(overrides, label="search_params"))
        return _normalize_search_params(
            self.index_type,
            merged,
            self._build_params,
        )

    def _faiss_search_parameters(self, values: Mapping[str, int]):
        import faiss

        if self.index_type == "hnsw_flat":
            return faiss.SearchParametersHNSW(
                efSearch=int(values["ef_search"]),
            )
        if self.index_type in {"ivf_flat", "ivf_pq"}:
            # SearchParametersIVF is accepted by both IndexIVFFlat and
            # IndexIVFPQ and is available across the supported faiss-cpu
            # versions. Some releases do not expose SearchParametersIVFPQ.
            params = faiss.SearchParametersIVF()
        else:
            return None

        params.nprobe = int(values["nprobe"])
        if "max_codes" in values:
            params.max_codes = int(values["max_codes"])
        return params

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self.backend or self.count <= 0 or self.dimension <= 0:
            raise RuntimeError("Index has not been built or loaded")

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        if not np.isfinite(query).all():
            raise ValueError("query_embedding must contain only finite values")
        query = np.ascontiguousarray(query)
        top_k = min(top_k, self.count)

        if self.backend == "faiss":
            values = self._resolved_search_params(search_params)
            params = self._faiss_search_parameters(values)
            if params is None:
                scores, indices = self.index.search(query, top_k)
            else:
                scores, indices = self.index.search(
                    query,
                    top_k,
                    params=params,
                )
            valid = indices[0] >= 0
            selected_scores = scores[0][valid]
            selected_ids = indices[0][valid]
            order = np.lexsort((selected_ids, -selected_scores))
            return selected_scores[order], selected_ids[order]

        if search_params:
            raise ValueError("NumPy flat_ip does not accept search parameters")
        if self.embeddings is None or self.ids is None:
            raise RuntimeError("NumPy index has not been built or loaded")
        scores = self.embeddings @ query[0]
        positions = np.lexsort((self.ids, -scores))[:top_k]
        return scores[positions], self.ids[positions]

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

    def _read_numpy_embeddings(
        self,
        source: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.load(source, allow_pickle=False).astype(np.float32)
        embeddings = _coerce_embeddings(
            embeddings,
            label=f"NumPy index at {source}",
        )
        ids = np.arange(embeddings.shape[0], dtype=np.int64)
        return embeddings, np.ascontiguousarray(ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.backend or self.count <= 0 or self.dimension <= 0:
            raise RuntimeError("Index has not been built or loaded")
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self.index, str(path))
            return
        raise RuntimeError(
            "NumPy search uses embeddings.npy directly and has no separate index artifact"
        )

    @staticmethod
    def _unwrap_loaded_index(loaded_index):
        import faiss

        if isinstance(loaded_index, (faiss.IndexIDMap2, faiss.IndexIDMap)):
            return faiss.downcast_index(loaded_index.index), True
        return loaded_index, False

    @staticmethod
    def _detect_faiss_index_type(base_index) -> str:
        import faiss

        if isinstance(base_index, faiss.IndexIVFPQ):
            return "ivf_pq"
        if isinstance(base_index, faiss.IndexIVFFlat):
            return "ivf_flat"
        if isinstance(base_index, faiss.IndexHNSWFlat):
            return "hnsw_flat"
        if isinstance(base_index, faiss.IndexFlatIP):
            return "flat_ip"
        raise ValueError(f"Unsupported FAISS index class: {type(base_index).__name__}")

    @staticmethod
    def _infer_build_params(index_type: str, base_index) -> dict[str, int]:
        if index_type == "hnsw_flat":
            return {
                "m": int(base_index.hnsw.nb_neighbors(1)),
                "ef_construction": int(base_index.hnsw.efConstruction),
            }
        if index_type == "ivf_flat":
            return {"nlist": int(base_index.nlist)}
        if index_type == "ivf_pq":
            return {
                "nlist": int(base_index.nlist),
                "m": int(base_index.pq.M),
                "nbits": int(base_index.pq.nbits),
            }
        return {}

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Index file does not exist: {path}")

        self._reset_runtime_state()
        if self.requested_backend == "faiss":
            try:
                import faiss

                loaded_index = faiss.read_index(str(path))
                base_index, has_external_id_map = self._unwrap_loaded_index(
                    loaded_index
                )
                detected_type = self._detect_faiss_index_type(base_index)
                if detected_type != self.index_type:
                    raise ValueError(
                        f"Expected FAISS {self.index_type}, found {detected_type}"
                    )
                if int(base_index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
                    raise ValueError("FAISS index must use inner-product similarity")
                if int(loaded_index.d) <= 0 or int(loaded_index.ntotal) <= 0:
                    raise ValueError(
                        "FAISS index must contain at least one non-empty vector"
                    )
                if not bool(loaded_index.is_trained):
                    raise ValueError("FAISS index must be trained")
                if not has_external_id_map and detected_type != "flat_ip":
                    raise ValueError(
                        "ANN indexes must preserve external ids with IndexIDMap"
                    )

                ids = (
                    faiss.vector_to_array(loaded_index.id_map).astype(np.int64)
                    if has_external_id_map
                    else np.arange(int(loaded_index.ntotal), dtype=np.int64)
                )
                if ids.shape[0] != int(loaded_index.ntotal):
                    raise ValueError("FAISS id map length does not match index count")
                if len(np.unique(ids)) != len(ids):
                    raise ValueError("FAISS index contains duplicate external ids")

                self.index = loaded_index
                self.backend = "faiss"
                self.dimension = int(loaded_index.d)
                self.count = int(loaded_index.ntotal)
                self.embeddings = None
                self.ids = np.ascontiguousarray(ids)
                loaded_build_params = self._infer_build_params(
                    detected_type,
                    base_index,
                )
                if (
                    self._build_params_explicit
                    and loaded_build_params != self._build_params
                ):
                    raise ValueError(
                        "FAISS index build parameters do not match the requested parameters"
                    )
                self._build_params = loaded_build_params
                if self._search_params_explicit:
                    self._search_params = _normalize_search_params(
                        self.index_type,
                        self._search_params,
                        self._build_params,
                    )
                else:
                    self._search_params = _normalize_search_params(
                        self.index_type,
                        None,
                        self._build_params,
                    )
                return
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load FAISS {self.index_type} index: {path}"
                ) from exc

        self.embeddings, self.ids = self._read_numpy_embeddings(path)
        self.backend = "numpy"
        self.index = None
        self.dimension = self.embeddings.shape[1]
        self.count = self.embeddings.shape[0]


# A clearer name for new call sites while preserving all existing imports.
FaissIndex = FlatIPIndex
