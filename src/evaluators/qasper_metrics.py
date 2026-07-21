"""Small, dependency-free implementation of the official QASPER metrics."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")


def normalize_qasper_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text: str) -> str:
        return "".join(character for character in text if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(value.lower())).split())


def qasper_token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_qasper_answer(prediction).split()
    reference_tokens = normalize_qasper_answer(reference).split()
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_qasper_evidence(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def qasper_evidence_f1(prediction: Sequence[str], reference: Sequence[str]) -> float:
    predicted = [normalize_qasper_evidence(value) for value in prediction if normalize_qasper_evidence(value)]
    expected = [normalize_qasper_evidence(value) for value in reference if normalize_qasper_evidence(value)]
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(set(predicted) & set(expected))
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def qasper_reference_answer(reference: Mapping[str, Any]) -> tuple[str, str]:
    if reference.get("unanswerable", False):
        return "Unanswerable", "none"

    extractive = reference.get("extractive_spans") or []
    if extractive:
        return ", ".join(extractive), "extractive"

    free_form = reference.get("free_form_answer") or ""
    if free_form:
        return free_form, "abstractive"

    yes_no = reference.get("yes_no")
    if yes_no is not None:
        return ("Yes" if yes_no else "No"), "boolean"
    return "", "abstractive"


def score_qasper_question(
    predicted_answer: str,
    predicted_evidence: Sequence[str],
    references: Sequence[Mapping[str, Any]],
    *,
    text_evidence_only: bool = False,
) -> dict[str, Any]:
    if not references:
        raise ValueError("QASPER scoring requires at least one reference")

    answer_scores = []
    evidence_scores = []
    for reference in references:
        answer, answer_type = qasper_reference_answer(reference)
        answer_scores.append((qasper_token_f1(predicted_answer, answer), answer_type))

        gold_evidence = [] if reference.get("unanswerable", False) else list(reference.get("evidence") or [])
        if text_evidence_only:
            gold_evidence = [value for value in gold_evidence if "FLOAT SELECTED" not in value]
        evidence_scores.append(qasper_evidence_f1(predicted_evidence, gold_evidence))

    best_answer_position = max(range(len(answer_scores)), key=lambda position: answer_scores[position][0])
    return {
        "answer_f1": answer_scores[best_answer_position][0],
        "answer_type": answer_scores[best_answer_position][1],
        "evidence_f1": max(evidence_scores),
    }


def score_qasper_open_corpus(
    predicted_answer: str,
    hits: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    target_paper_id: str,
    evidence_units_by_paper: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Score question-only retrieval over a global QASPER paper collection.

    This is intentionally named as an open-corpus metric rather than the
    official paper-scoped QASPER evidence metric. Evidence units retrieved from
    a wrong paper remain in the prediction set and therefore reduce precision.
    """

    if not target_paper_id:
        raise ValueError("Open-corpus QASPER scoring requires a target paper id")
    if not references:
        raise ValueError("Open-corpus QASPER scoring requires at least one reference")

    target_rank = 0
    predicted_evidence: set[tuple[str, str]] = set()
    for position, hit in enumerate(hits, start=1):
        paper_id = str(hit.get("doc_id") or hit.get("source") or "").strip()
        if not paper_id:
            continue
        if paper_id == target_paper_id and target_rank == 0:
            target_rank = position
        units = evidence_units_by_paper.get(paper_id, ())
        page_start = int(hit.get("page_start", hit.get("page", 0)))
        page_end = int(hit.get("page_end", hit.get("page", page_start)))
        for page in range(page_start, page_end + 1):
            if not 1 <= page <= len(units):
                continue
            evidence = normalize_qasper_evidence(str(units[page - 1]))
            if evidence:
                predicted_evidence.add((paper_id, evidence))

    answer_scores: list[tuple[float, str]] = []
    evidence_recalls: list[float] = []
    evidence_f1_scores: list[float] = []
    for reference in references:
        answer, answer_type = qasper_reference_answer(reference)
        answer_scores.append((qasper_token_f1(predicted_answer, answer), answer_type))

        raw_gold = [] if reference.get("unanswerable", False) else list(reference.get("evidence") or [])
        gold_evidence = {
            (target_paper_id, normalized)
            for value in raw_gold
            if (normalized := normalize_qasper_evidence(str(value)))
        }
        overlap = len(predicted_evidence & gold_evidence)
        if gold_evidence:
            evidence_recalls.append(overlap / len(gold_evidence))
        if not predicted_evidence and not gold_evidence:
            evidence_f1_scores.append(1.0)
        elif not predicted_evidence or not gold_evidence or overlap == 0:
            evidence_f1_scores.append(0.0)
        else:
            precision = overlap / len(predicted_evidence)
            recall = overlap / len(gold_evidence)
            evidence_f1_scores.append(2 * precision * recall / (precision + recall))

    best_answer_position = max(range(len(answer_scores)), key=lambda position: answer_scores[position][0])
    return {
        "qasper_target_paper_hit_at_k": bool(target_rank),
        "qasper_target_paper_rr": 1.0 / target_rank if target_rank else 0.0,
        "qasper_answer_f1": answer_scores[best_answer_position][0],
        "qasper_target_evidence_recall_at_k": max(evidence_recalls) if evidence_recalls else None,
        "qasper_target_evidence_f1_at_k": max(evidence_f1_scores),
    }


def summarize_qasper_open_corpus(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the minimal question-only open-corpus evaluation metrics."""

    def values(name: str) -> list[float]:
        return [
            float(score[name])
            for score in scores
            if isinstance(score.get(name), (bool, int, float))
        ]

    paper_hits = values("qasper_target_paper_hit_at_k")
    paper_rr = values("qasper_target_paper_rr")
    answer_f1 = values("qasper_answer_f1")
    evidence_recall = values("qasper_target_evidence_recall_at_k")
    evidence_f1 = values("qasper_target_evidence_f1_at_k")
    return {
        "num_questions": len(scores),
        "qasper_target_paper_hit_rate_at_k": mean(paper_hits) if paper_hits else 0.0,
        "qasper_target_paper_mrr": mean(paper_rr) if paper_rr else 0.0,
        "qasper_answer_f1": mean(answer_f1) if answer_f1 else 0.0,
        "qasper_target_evidence_recall_at_k": mean(evidence_recall) if evidence_recall else 0.0,
        "qasper_target_evidence_recall_valid_count": len(evidence_recall),
        "qasper_target_evidence_f1_at_k": mean(evidence_f1) if evidence_f1 else 0.0,
    }


def summarize_qasper_scores(scores: Sequence[Mapping[str, Any]], missing_predictions: int = 0) -> dict[str, Any]:
    if missing_predictions < 0:
        raise ValueError("missing_predictions must be non-negative")
    answer_values = [float(score["answer_f1"]) for score in scores] + [0.0] * missing_predictions
    evidence_values = [float(score["evidence_f1"]) for score in scores] + [0.0] * missing_predictions
    by_type: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        answer_type = str(score["answer_type"])
        if answer_type not in ANSWER_TYPES:
            raise ValueError(f"Unknown QASPER answer type: {answer_type}")
        by_type[answer_type].append(float(score["answer_f1"]))

    return {
        "Answer F1": mean(answer_values) if answer_values else 0.0,
        "Answer F1 by type": {
            answer_type: mean(by_type[answer_type]) if by_type[answer_type] else 0.0
            for answer_type in ANSWER_TYPES
        },
        "Evidence F1": mean(evidence_values) if evidence_values else 0.0,
        "Missing predictions": missing_predictions,
    }
