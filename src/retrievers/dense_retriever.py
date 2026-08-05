"""Embed a query, search an index, and lazily map vector ids to chunks."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from src.retrievers.chunk_store import ChunkStore, as_chunk_store
from src.records import ChunkRecord, RetrievalTrace, SearchHit


class DenseRetriever:
    def __init__(
        self,
        chunks: ChunkStore | Iterable[ChunkRecord | dict],
        embedder,
        index,
        top_k: int = 5,
    ):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if index.count <= 0 or index.dimension <= 0:
            raise ValueError("index must be built or loaded before creating a retriever")

        store = as_chunk_store(chunks)
        if len(store) != index.count:
            raise ValueError(f"Chunk count {len(store)} does not match index count {index.count}")
        embedder_dimension = getattr(embedder, "dimension", None)
        if embedder_dimension is not None and embedder_dimension != index.dimension:
            raise ValueError(
                f"Embedder dimension {embedder_dimension} does not match index dimension {index.dimension}"
            )
        if getattr(index, "ids", None) is not None:
            index_ids = np.asarray(index.ids, dtype=np.int64)
            expected = np.arange(len(store), dtype=np.int64)
            if index_ids.shape != expected.shape or not np.array_equal(np.sort(index_ids), expected):
                raise ValueError("Index vector ids do not match chunk vector ids")

        self.chunk_store = store
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def retrieve_trace(
        self,
        query: str,
        top_k: int | None = None,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> RetrievalTrace:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        effective_top_k = self.top_k if top_k is None else top_k
        if (
            isinstance(effective_top_k, bool)
            or not isinstance(effective_top_k, int)
            or effective_top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")

        if effective_top_k == 0:
            return RetrievalTrace(
                top_k=0,
                results=(),
                timings_ms={
                    "query_embedding_ms": 0.0,
                    "index_search_ms": 0.0,
                    "chunk_mapping_ms": 0.0,
                    "total_ms": 0.0,
                },
            )

        started = time.perf_counter()
        embedding_started = time.perf_counter()
        query_embedding = self.embedder.encode_queries([query.strip()])
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000

        search_started = time.perf_counter()
        vector_hits = self.index.search_hits(
            query_embedding,
            effective_top_k,
            search_params=dict(search_params or {}),
        )
        search_latency_ms = (time.perf_counter() - search_started) * 1000

        mapping_started = time.perf_counter()
        try:
            records = self.chunk_store.get_many([hit.vector_id for hit in vector_hits])
        except KeyError as exc:
            raise ValueError(f"Index returned unknown vector id: {exc.args[0]}") from exc
        results = tuple(
            SearchHit(rank=rank, chunk=chunk, score=hit.score)
            for rank, (hit, chunk) in enumerate(zip(vector_hits, records), start=1)
        )
        mapping_latency_ms = (time.perf_counter() - mapping_started) * 1000

        return RetrievalTrace(
            top_k=effective_top_k,
            results=results,
            timings_ms={
                "query_embedding_ms": embedding_latency_ms,
                "index_search_ms": search_latency_ms,
                "chunk_mapping_ms": mapping_latency_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        )
