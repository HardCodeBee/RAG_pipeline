"""Construct vector indexes from validated configuration."""

from __future__ import annotations

# Index assembly is isolated from encoded-corpus and query assembly.

from typing import Any

from src.indexes.vector_index import FaissIndex


def create_index(
    config: dict[str, Any],
    *,
    backend: str | None = None,
    index_type: str | None = None,
    threads: int | None = None,
):
    index_config = config["index"]
    effective_backend = backend or index_config["backend"]
    effective_type = index_type or index_config["type"]
    if effective_backend == "faiss":
        thread_count = index_config["faiss_threads"] if threads is None else threads
        if thread_count:
            import faiss

            faiss.omp_set_num_threads(thread_count)

    build_params: dict[str, int] = {}
    if effective_type == "hnsw_flat":
        build_params = {
            "m": index_config["hnsw_m"],
            "ef_construction": index_config["ef_construction"],
        }
    elif effective_type == "ivf_flat":
        build_params = {"nlist": index_config["nlist"]}
    elif effective_type == "ivf_pq":
        build_params = {
            "nlist": index_config["nlist"],
            "m": index_config["pq_m"],
            "nbits": index_config["pq_nbits"],
        }

    retrieval = config["retrieval"]
    search_params = {
        key: retrieval[key]
        for key in ("nprobe", "ef_search", "max_codes")
        if key in retrieval
    }
    return FaissIndex(
        backend=effective_backend,
        index_type=effective_type,
        build_params=build_params,
        search_params=search_params,
    )
