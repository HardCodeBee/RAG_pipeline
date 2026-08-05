"""NQ protocol contracts and score summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.persistence.artifact_validation import VerifiedNQSplit, validate_nq_dataset_split
from src.evaluation_runner import validate_questions
from src.evaluators.nq_metrics import (
    positive_chunk_ids_from_evidence,
    summarize_nq_retrieval_scores,
    summarize_nq_scores,
)
from src.evaluators.operational_metrics import summarize_execution


EVALUATION_PROTOCOL = "nq_open_dpr_wiki_1m_gold_preserving_v1"
METRICS_VERSION = "nq_open_retrieval_and_short_answer_v2_nfkc_body_only"
RETRIEVAL_METRICS_VERSION = "nq_open_retrieval_only_v1_nfkc_body_only"
CANONICAL_SPLIT_COUNTS = {"calibration": 500, "evaluation": 1_500}
CANONICAL_PASSAGE_COUNT = 1_000_000
CANONICAL_HARD_NEGATIVES_PER_QUESTION = 50
CANONICAL_SEED = EVALUATION_PROTOCOL


def canonical_question_split(
    corpus_path: Path,
    *,
    split: str,
    requested_path: Path | None,
) -> VerifiedNQSplit:
    """Load a split after one centralized validation of its artifact chain."""

    return validate_nq_dataset_split(
        corpus_path,
        split=split,
        requested_path=requested_path,
        expected_protocol=EVALUATION_PROTOCOL,
        canonical_split_counts=CANONICAL_SPLIT_COUNTS,
        canonical_passage_count=CANONICAL_PASSAGE_COUNT,
        canonical_hard_negatives_per_question=CANONICAL_HARD_NEGATIVES_PER_QUESTION,
        canonical_seed=CANONICAL_SEED,
    )


def validate_nq_questions(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate common question fields plus NQ answers and positive evidence."""

    questions = validate_questions(values)
    for question in questions:
        answers = question.get("answers")
        evidence = question.get("evidence")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer.strip() for answer in answers)
        ):
            raise ValueError(f"NQ question {question['question_id']} has invalid answers")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"NQ question {question['question_id']} has invalid evidence")
        positive_chunk_ids_from_evidence(evidence)
    return questions


def validate_openai_generation(generation: Mapping[str, Any]) -> None:
    """Require provider-confirmed OpenAI metadata for a successful NQ row."""

    if generation.get("provider") != "openai":
        raise RuntimeError("A successful NQ row did not use the OpenAI provider")
    for field in ("model", "response_id"):
        if not isinstance(generation.get(field), str) or not generation[field].strip():
            raise RuntimeError(f"OpenAI generation has no non-empty {field}")
    usage = generation.get("token_usage")
    reported = usage.get("provider_reported") if isinstance(usage, Mapping) else None
    values: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = reported.get(field) if isinstance(reported, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"OpenAI generation has no provider-reported {field}")
        values[field] = value
    if values["total_tokens"] != values["input_tokens"] + values["output_tokens"]:
        raise RuntimeError("OpenAI provider token totals are inconsistent")


def summarize_nq_run(
    rows: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    split: str,
    effective_top_k: int,
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """Combine NQ scores with protocol-independent execution statistics."""

    execution = summarize_execution(rows)
    scores = [row["metrics"] for row in rows if row.get("status") == "success"]
    score_summary = (
        summarize_nq_retrieval_scores(scores)
        if retrieval_only
        else summarize_nq_scores(scores)
    )
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "metrics_version": (
            RETRIEVAL_METRICS_VERSION if retrieval_only else METRICS_VERSION
        ),
        "execution_mode": "retrieval_only" if retrieval_only else "rag",
        "question_split": split,
        "num_corpus_passages": manifest["index"]["count"],
        "effective_top_k": effective_top_k,
        "embedding_backend": manifest["embedding"]["space"]["backend"],
        "index_backend": manifest["index"]["backend"],
        "index_type": manifest["index"]["type"],
        "generation_provider": None if retrieval_only else ("openai" if scores else None),
        "openai_verified_success_count": 0 if retrieval_only else len(scores),
        **score_summary,
        "num_question_records": len(rows),
        "num_successful_questions": execution["num_successful_questions"],
        "num_failed_questions": execution["num_failed_questions"],
        "avg_retrieval_latency_ms": execution["avg_retrieval_latency_ms"],
        "p50_retrieval_latency_ms": execution["p50_retrieval_latency_ms"],
        "p95_retrieval_latency_ms": execution["p95_retrieval_latency_ms"],
        "avg_query_embedding_latency_ms": execution["avg_query_embedding_latency_ms"],
        "avg_index_search_latency_ms": execution["avg_index_search_latency_ms"],
        "p50_index_search_latency_ms": execution["p50_index_search_latency_ms"],
        "p95_index_search_latency_ms": execution["p95_index_search_latency_ms"],
        "avg_chunk_mapping_latency_ms": execution["avg_chunk_mapping_latency_ms"],
        "avg_rerank_latency_ms": execution["avg_rerank_latency_ms"],
        "p50_rerank_latency_ms": execution["p50_rerank_latency_ms"],
        "p95_rerank_latency_ms": execution["p95_rerank_latency_ms"],
        "avg_generation_latency_ms": execution["avg_generation_latency_ms"],
        "avg_total_latency_ms": execution["avg_total_latency_ms"],
        "p95_total_latency_ms": execution["p95_total_latency_ms"],
        "total_provider_input_tokens": execution["total_provider_input_tokens"],
        "total_provider_output_tokens": execution["total_provider_output_tokens"],
        "total_provider_total_tokens": execution["total_provider_total_tokens"],
    }
