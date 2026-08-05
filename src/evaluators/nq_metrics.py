"""Natural Questions Open retrieval and short-answer metrics."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_nq_answer(value: str) -> str:
    """Apply SQuAD-style normalization used for short-answer EM and F1."""

    if not isinstance(value, str):
        raise TypeError("NQ answer values must be strings")
    lowered = unicodedata.normalize("NFKC", value).lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if not unicodedata.category(character).startswith("P")
    )
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _answer_aliases(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("answers must be a sequence of strings")
    aliases = [value for value in values if isinstance(value, str) and value.strip()]
    if not aliases:
        raise ValueError("NQ scoring requires at least one non-empty answer alias")
    return aliases


def nq_answer_exact_match(prediction: str, answers: Sequence[str]) -> bool:
    normalized_prediction = normalize_nq_answer(prediction)
    return any(normalized_prediction == normalize_nq_answer(answer) for answer in _answer_aliases(answers))


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_nq_answer(prediction).split()
    reference_tokens = normalize_nq_answer(reference).split()
    if not reference_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def nq_answer_token_f1(prediction: str, answers: Sequence[str]) -> float:
    return max(_token_f1(prediction, answer) for answer in _answer_aliases(answers))


def _positive_ids(values: Sequence[str]) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("positive_chunk_ids must be a sequence")
    result = {str(value).strip() for value in values if str(value).strip()}
    if not result:
        raise ValueError("NQ retrieval scoring requires positive chunk ids")
    return result


def positive_chunk_ids_from_evidence(evidence: Sequence[Mapping[str, Any] | str]) -> list[str]:
    """Extract the fixed question-file alternatives into positive chunk ids."""

    if isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence")
    selected: list[str] = []
    seen: set[str] = set()

    def visit(value: Mapping[str, Any] | str) -> None:
        if isinstance(value, str):
            chunk_id = value.strip()
            if chunk_id and chunk_id not in seen:
                selected.append(chunk_id)
                seen.add(chunk_id)
            return
        if not isinstance(value, Mapping):
            raise TypeError("NQ evidence entries must be strings or mappings")
        alternatives = value.get("alternatives")
        if alternatives is not None:
            if not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes)):
                raise TypeError("evidence.alternatives must be a sequence")
            for alternative in alternatives:
                visit(alternative)
            return
        chunk_id = value.get("chunk_id")
        if chunk_id is not None:
            visit(str(chunk_id))

    for item in evidence:
        visit(item)
    if not selected:
        raise ValueError("NQ evidence contains no positive chunk ids")
    return selected


def nq_positive_passage_hit_at_k(
    results: Sequence[Mapping[str, Any]],
    positive_chunk_ids: Sequence[str],
    top_k: int | None = None,
) -> bool:
    positives = _positive_ids(positive_chunk_ids)
    selected = results[:top_k] if top_k is not None else results
    return any(str(result.get("chunk_id", "")).strip() in positives for result in selected)


def nq_positive_passage_mrr(
    results: Sequence[Mapping[str, Any]],
    positive_chunk_ids: Sequence[str],
    top_k: int | None = None,
) -> float:
    positives = _positive_ids(positive_chunk_ids)
    selected = results[:top_k] if top_k is not None else results
    for rank, result in enumerate(selected, start=1):
        if str(result.get("chunk_id", "")).strip() in positives:
            return 1.0 / rank
    return 0.0


def _contains_answer(text: str, answers: Sequence[str]) -> bool:
    text_tokens = normalize_nq_answer(text).split()
    for answer in _answer_aliases(answers):
        answer_tokens = normalize_nq_answer(answer).split()
        if not answer_tokens or len(answer_tokens) > len(text_tokens):
            continue
        width = len(answer_tokens)
        if any(text_tokens[index : index + width] == answer_tokens for index in range(len(text_tokens) - width + 1)):
            return True
    return False


def nq_answer_passage_hit_at_k(
    results: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    top_k: int | None = None,
) -> bool:
    selected = results[:top_k] if top_k is not None else results
    bodies = (
        text.partition("\n")[2] if "\n" in text else text
        for text in (str(result.get("text", "")) for result in selected)
    )
    return any(_contains_answer(body, answers) for body in bodies)


def score_nq_question(
    prediction: str,
    results: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    positive_chunk_ids: Sequence[str],
    top_k: int | None = None,
) -> dict[str, bool | float]:
    aliases = _answer_aliases(answers)
    return {
        **score_nq_retrieval(
            results,
            aliases,
            positive_chunk_ids,
            top_k=top_k,
        ),
        "nq_answer_exact_match": nq_answer_exact_match(prediction, aliases),
        "nq_answer_token_f1": nq_answer_token_f1(prediction, aliases),
    }


def score_nq_retrieval(
    results: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    positive_chunk_ids: Sequence[str],
    top_k: int | None = None,
) -> dict[str, bool | float]:
    """Score retrieval without requiring or fabricating a generated answer."""

    aliases = _answer_aliases(answers)
    return {
        "nq_positive_passage_hit_at_k": nq_positive_passage_hit_at_k(
            results,
            positive_chunk_ids,
            top_k=top_k,
        ),
        "nq_positive_passage_mrr": nq_positive_passage_mrr(
            results,
            positive_chunk_ids,
            top_k=top_k,
        ),
        "nq_answer_passage_hit_at_k": nq_answer_passage_hit_at_k(
            results,
            aliases,
            top_k=top_k,
        ),
    }


_RETRIEVAL_METRIC_NAMES = (
    "nq_positive_passage_hit_at_k",
    "nq_positive_passage_mrr",
    "nq_answer_passage_hit_at_k",
)


def _summarize_metrics(
    scores: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, int | float]:
    summary: dict[str, int | float] = {"num_questions": len(scores)}
    for name in metric_names:
        values = [
            float(score[name])
            for score in scores
            if isinstance(score.get(name), (bool, int, float))
        ]
        summary[name] = mean(values) if values else 0.0
        summary[f"{name}_valid_count"] = len(values)
    return summary


def summarize_nq_retrieval_scores(
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    return _summarize_metrics(scores, _RETRIEVAL_METRIC_NAMES)


def summarize_nq_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    return _summarize_metrics(
        scores,
        (
            *_RETRIEVAL_METRIC_NAMES,
            "nq_answer_exact_match",
            "nq_answer_token_f1",
        ),
    )
