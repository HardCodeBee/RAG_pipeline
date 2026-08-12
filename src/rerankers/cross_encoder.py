"""Sentence-Transformers cross-encoder reranking."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.records import SearchHit
from src.model_backends.huggingface_snapshot import resolve_hf_snapshot
from src.rerankers.reranker_contract import (
    RerankResult,
    RerankTrace,
    reranked_hits,
    validate_rerank_inputs,
)


_CROSS_ENCODER_MAX_LENGTH = 512
_CROSS_ENCODER_SCORE_KIND = "raw_logit"


def _load_cross_encoder(
    *,
    model_name: str,
    revision: str,
    device: str,
    local_files_only: bool,
):
    from sentence_transformers import CrossEncoder
    from torch import nn

    snapshot = resolve_hf_snapshot(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    return CrossEncoder(
        str(snapshot),
        device=device,
        local_files_only=True,
        max_length=_CROSS_ENCODER_MAX_LENGTH,
        # BGE documents its relevance score as the model logit.  Leaving this
        # unset makes Sentence-Transformers apply a sigmoid for one-label
        # models, which can collapse distinct large logits into equal float32
        # values and create artificial ties.
        activation_fn=nn.Identity(),
    )


def _validate_many_inputs(
    queries: Sequence[str],
    hits_by_query: Sequence[Sequence[SearchHit]],
    final_k: int | None,
) -> tuple[tuple[str, tuple[SearchHit, ...], int], ...]:
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise TypeError("queries must be a sequence of strings")
    if isinstance(hits_by_query, (str, bytes)) or not isinstance(
        hits_by_query,
        Sequence,
    ):
        raise TypeError("hits_by_query must be a sequence of hit sequences")

    query_values = tuple(queries)
    hit_groups = tuple(hits_by_query)
    if len(query_values) != len(hit_groups):
        raise ValueError("queries and hits_by_query must have the same length")
    if final_k is not None:
        if isinstance(final_k, bool) or not isinstance(final_k, int):
            raise TypeError("final_k must be an integer or None")
        if final_k < 0:
            raise ValueError("final_k must be non-negative")

    validated = []
    for query, hits in zip(query_values, hit_groups):
        values, effective_final_k = validate_rerank_inputs(query, hits, final_k)
        validated.append((query.strip(), values, effective_final_k))
    return tuple(validated)


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        device: str,
        batch_size: int = 32,
        local_files_only: bool = False,
    ):
        for value, name in (
            (model_name, "model_name"),
            (revision, "revision"),
            (device, "device"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")

        self.model_name = model_name.strip()
        self.revision = revision.strip()
        self.device = device.strip()
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = _load_cross_encoder(
            model_name=self.model_name,
            revision=self.revision,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        raw_scores: Any = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or scores.shape[0] != len(pairs):
            raise RuntimeError(
                f"CrossEncoder returned shape {scores.shape}; expected {(len(pairs),)}"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("CrossEncoder returned non-finite scores")
        return scores

    def rerank_many(
        self,
        queries: Sequence[str],
        hits_by_query: Sequence[Sequence[SearchHit]],
        final_k: int | None = None,
    ) -> tuple[RerankResult, ...]:
        """Rerank several questions with one shared model ``predict`` call.

        Candidate pairs are flattened in question order, scored together, and
        split back into independent stable rankings.  Each result's
        ``timing_ms`` is the total batch wall time divided equally across all
        input questions, so their sum represents the batch wall time rather
        than duplicated shared inference time.
        """

        started = time.perf_counter()
        validated = _validate_many_inputs(queries, hits_by_query, final_k)
        if not validated:
            return ()

        pairs: list[tuple[str, str]] = []
        score_ranges: list[tuple[int, int]] = []
        for query, hits, effective_final_k in validated:
            start = len(pairs)
            if hits and effective_final_k > 0:
                pairs.extend((query, hit.chunk.text) for hit in hits)
            score_ranges.append((start, len(pairs)))

        scores = (
            self._predict_scores(pairs)
            if pairs
            else np.empty((0,), dtype=np.float64)
        )
        prepared: list[tuple[tuple[SearchHit, ...], RerankTrace | None]] = []
        for (_, hits, effective_final_k), (start, end) in zip(
            validated,
            score_ranges,
        ):
            if not hits or effective_final_k == 0:
                prepared.append(((), None))
                continue

            query_scores = scores[start:end]
            # Python's stable sort retains candidate order for equal scores.
            order = sorted(
                range(len(hits)),
                key=lambda position: -float(query_scores[position]),
            )
            ordered = [
                (hits[position], float(query_scores[position]))
                for position in order
            ]
            results = reranked_hits(ordered, effective_final_k)
            trace = RerankTrace(
                candidates=hits,
                scores=tuple(float(score) for score in query_scores),
                order=tuple(order),
                final_k=effective_final_k,
                score_kind=_CROSS_ENCODER_SCORE_KIND,
                max_sequence_length=_CROSS_ENCODER_MAX_LENGTH,
            )
            prepared.append((results, trace))

        timing_ms = (time.perf_counter() - started) * 1000 / len(validated)
        return tuple(
            RerankResult(
                results=results,
                timing_ms=timing_ms,
                trace=trace,
            )
            for results, trace in prepared
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        final_k: int | None = None,
    ) -> RerankResult:
        return self.rerank_many((query,), (hits,), final_k)[0]
