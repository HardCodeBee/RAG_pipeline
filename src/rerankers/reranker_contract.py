"""Small contracts shared by optional rerankers."""

from __future__ import annotations

# Shared reranker protocol and result contract.

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from src.records import SearchHit


@dataclass(frozen=True, slots=True)
class RerankTrace:
    """Compact two-stage ranking trace without duplicating passage text."""

    candidates: tuple[SearchHit, ...]
    scores: tuple[float, ...]
    order: tuple[int, ...]
    final_k: int
    score_kind: str
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(item, SearchHit) for item in self.candidates
        ):
            raise TypeError("RerankTrace.candidates must be a tuple of SearchHit values")
        if not isinstance(self.scores, tuple) or len(self.scores) != len(self.candidates):
            raise ValueError("RerankTrace.scores must align with candidates")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            for score in self.scores
        ):
            raise ValueError("RerankTrace.scores must contain only finite numbers")
        if not isinstance(self.order, tuple) or sorted(self.order) != list(
            range(len(self.candidates))
        ):
            raise ValueError("RerankTrace.order must be a permutation of candidate positions")
        if (
            isinstance(self.final_k, bool)
            or not isinstance(self.final_k, int)
            or not 0 <= self.final_k <= len(self.candidates)
        ):
            raise ValueError("RerankTrace.final_k must be between zero and candidate count")
        if not isinstance(self.score_kind, str) or not self.score_kind.strip():
            raise ValueError("RerankTrace.score_kind must be a non-empty string")
        if self.max_sequence_length is not None and (
            isinstance(self.max_sequence_length, bool)
            or not isinstance(self.max_sequence_length, int)
            or self.max_sequence_length <= 0
        ):
            raise ValueError("RerankTrace.max_sequence_length must be positive or None")

    def selected_results(self) -> tuple[SearchHit, ...]:
        ordered = (
            (self.candidates[position], self.scores[position])
            for position in self.order[: self.final_k]
        )
        return tuple(
            replace(hit, rank=rank, score=float(score))
            for rank, (hit, score) in enumerate(ordered, start=1)
        )

    def to_dict(self) -> dict[str, object]:
        rerank_ranks = [0] * len(self.candidates)
        for rank, position in enumerate(self.order, start=1):
            rerank_ranks[position] = rank
        return {
            "candidate_k": len(self.candidates),
            "final_k": self.final_k,
            "score_kind": self.score_kind,
            "max_sequence_length": self.max_sequence_length,
            "candidates": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "vector_id": hit.chunk.vector_id,
                    "retrieval_rank": hit.rank,
                    "retrieval_score": float(hit.score),
                    "rerank_rank": rerank_ranks[position],
                    "rerank_score": float(self.scores[position]),
                    "selected": rerank_ranks[position] <= self.final_k,
                }
                for position, hit in enumerate(self.candidates)
            ],
        }


@dataclass(frozen=True, slots=True)
class RerankResult:
    results: tuple[SearchHit, ...]
    timing_ms: float
    trace: RerankTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not all(
            isinstance(item, SearchHit) for item in self.results
        ):
            raise TypeError("RerankResult.results must be a tuple of SearchHit values")
        if [item.rank for item in self.results] != list(range(1, len(self.results) + 1)):
            raise ValueError("RerankResult ranks must be the consecutive sequence starting at one")
        if (
            isinstance(self.timing_ms, bool)
            or not isinstance(self.timing_ms, (int, float))
            or not math.isfinite(float(self.timing_ms))
            or self.timing_ms < 0
        ):
            raise ValueError("RerankResult.timing_ms must be a finite non-negative number")
        if self.trace is not None:
            if not isinstance(self.trace, RerankTrace):
                raise TypeError("RerankResult.trace must be a RerankTrace or None")
            if self.results != self.trace.selected_results():
                raise ValueError("RerankResult.results do not match the attached trace")

    @property
    def latency_ms(self) -> float:
        return float(self.timing_ms)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        final_k: int | None = None,
    ) -> RerankResult: ...


def validate_rerank_inputs(
    query: str,
    hits: Sequence[SearchHit],
    final_k: int | None,
) -> tuple[tuple[SearchHit, ...], int]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence):
        raise TypeError("hits must be a sequence of SearchHit values")

    values = tuple(hits)
    if not all(isinstance(item, SearchHit) for item in values):
        raise TypeError("hits must contain only SearchHit values")

    if final_k is None:
        effective_final_k = len(values)
    else:
        if isinstance(final_k, bool) or not isinstance(final_k, int):
            raise TypeError("final_k must be an integer or None")
        if final_k < 0:
            raise ValueError("final_k must be non-negative")
        effective_final_k = min(final_k, len(values))
    return values, effective_final_k


def reranked_hits(
    ordered: Sequence[tuple[SearchHit, float]],
    final_k: int,
) -> tuple[SearchHit, ...]:
    return tuple(
        replace(hit, rank=rank, score=float(score))
        for rank, (hit, score) in enumerate(ordered[:final_k], start=1)
    )
