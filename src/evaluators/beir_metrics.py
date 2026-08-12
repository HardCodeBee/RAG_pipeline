"""Retrieval-only metrics for prepared BEIR query and qrels bundles."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any


METRICS_VERSION = "beir_retrieval_v2_trec_linear_ndcg_binary_recall"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_qrels(qrels: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_doc_id, raw_score in qrels.items():
        if not isinstance(raw_doc_id, str) or not raw_doc_id:
            raise ValueError("qrel document ids must be non-empty strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError("qrel relevance scores must be numeric")
        score = float(raw_score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("qrel relevance scores must be finite and non-negative")
        if score > 0:
            values[raw_doc_id] = score
    return values


def score_beir_query(
    retrieved_doc_ids: Sequence[str],
    qrels: Mapping[str, Any],
    *,
    k: int = 5,
) -> dict[str, float | int]:
    """Score one ranked list.

    NDCG uses the qrel relevance value itself as gain, matching BEIR's
    ``pytrec_eval``/``trec_eval ndcg_cut`` convention. Recall, precision,
    reciprocal rank, and average precision treat every qrel with relevance
    greater than zero as relevant. Duplicate retrieved ids are rejected instead
    of double-counted.
    """

    k = _positive_int(k, "k")
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in retrieved_doc_ids):
        raise ValueError("retrieved document ids must be non-empty strings")
    if len(retrieved_doc_ids) != len(set(retrieved_doc_ids)):
        raise ValueError("retrieved document ids must be unique")

    relevant = _validated_qrels(qrels)
    ranked = list(retrieved_doc_ids[:k])
    binary = [1 if doc_id in relevant else 0 for doc_id in ranked]
    hits = sum(binary)
    total_relevant = len(relevant)

    reciprocal_rank = 0.0
    precisions_at_hits: list[float] = []
    for rank, is_relevant in enumerate(binary, start=1):
        if not is_relevant:
            continue
        if reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / rank
        precisions_at_hits.append(sum(binary[:rank]) / rank)

    average_precision = (
        sum(precisions_at_hits) / min(total_relevant, k)
        if total_relevant
        else 0.0
    )
    dcg = sum(
        relevant.get(doc_id, 0.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked, start=1)
    )
    ideal_scores = sorted(relevant.values(), reverse=True)[:k]
    ideal_dcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    return {
        f"ndcg_at_{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
        f"map_at_{k}": average_precision,
        f"recall_at_{k}": hits / total_relevant if total_relevant else 0.0,
        f"precision_at_{k}": hits / k,
        f"mrr_at_{k}": reciprocal_rank,
        f"hit_at_{k}": int(hits > 0),
        "num_relevant": total_relevant,
        "num_retrieved": len(ranked),
    }


def summarize_beir_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Macro-average successful per-query metric dictionaries."""

    k = _positive_int(k, "k")
    values = list(rows)
    successful = [row for row in values if row.get("status") == "success"]
    failed = len(values) - len(successful)
    metric_names = (
        f"ndcg_at_{k}",
        f"map_at_{k}",
        f"recall_at_{k}",
        f"precision_at_{k}",
        f"mrr_at_{k}",
        f"hit_at_{k}",
    )
    summary: dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "effective_top_k": k,
        "num_questions": len(values),
        "num_successful_questions": len(successful),
        "num_failed_questions": failed,
    }
    for name in metric_names:
        samples = [
            float(row["metrics"][name])
            for row in successful
            if isinstance(row.get("metrics"), Mapping) and name in row["metrics"]
        ]
        summary[name] = mean(samples) if samples else 0.0
        summary[f"{name}_valid_count"] = len(samples)
    return summary
