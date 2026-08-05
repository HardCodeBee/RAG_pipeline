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

    def _scores(self, query: str, hits: tuple[SearchHit, ...]) -> np.ndarray:
        pairs = [(query.strip(), hit.chunk.text) for hit in hits]
        raw_scores: Any = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or scores.shape[0] != len(hits):
            raise RuntimeError(
                f"CrossEncoder returned shape {scores.shape}; expected {(len(hits),)}"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("CrossEncoder returned non-finite scores")
        return scores

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        final_k: int | None = None,
    ) -> RerankResult:
        started = time.perf_counter()
        values, effective_final_k = validate_rerank_inputs(query, hits, final_k)
        if not values or effective_final_k == 0:
            return RerankResult(
                results=(),
                timing_ms=(time.perf_counter() - started) * 1000,
            )

        scores = self._scores(query, values)
        # Python's stable sort retains the candidate order for equal scores.
        order = sorted(range(len(values)), key=lambda position: -float(scores[position]))
        ordered = [(values[position], float(scores[position])) for position in order]
        results = reranked_hits(ordered, effective_final_k)
        trace = RerankTrace(
            candidates=values,
            scores=tuple(float(score) for score in scores),
            order=tuple(order),
            final_k=effective_final_k,
            score_kind=_CROSS_ENCODER_SCORE_KIND,
            max_sequence_length=_CROSS_ENCODER_MAX_LENGTH,
        )
        return RerankResult(
            results=results,
            timing_ms=(time.perf_counter() - started) * 1000,
            trace=trace,
        )
