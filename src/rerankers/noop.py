"""A no-op reranker that preserves candidate order and score."""

from __future__ import annotations

import time
from collections.abc import Sequence

from src.records import SearchHit
from src.rerankers.reranker_contract import RerankResult, reranked_hits, validate_rerank_inputs


class NoOpReranker:
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        final_k: int | None = None,
    ) -> RerankResult:
        started = time.perf_counter()
        values, effective_final_k = validate_rerank_inputs(query, hits, final_k)
        ordered = [(hit, hit.score) for hit in values]
        results = reranked_hits(ordered, effective_final_k)
        return RerankResult(
            results=results,
            timing_ms=(time.perf_counter() - started) * 1000,
        )
