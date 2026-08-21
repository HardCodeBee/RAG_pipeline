"""HotpotQA answer metrics and blind Answer Correctness judging helpers."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_LABEL_SCORES = {
    "incorrect": 0.0,
    "partially_correct": 0.5,
    "correct": 1.0,
}
_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(value: str) -> str:
    """Apply the official HotpotQA-style answer normalization."""

    if not isinstance(value, str):
        raise TypeError("answer must be a string")
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    prediction_normalized = " ".join(prediction_tokens)
    reference_normalized = " ".join(reference_tokens)

    if (
        prediction_normalized in _SPECIAL_ANSWERS
        or reference_normalized in _SPECIAL_ANSWERS
    ) and prediction_normalized != reference_normalized:
        return 0.0
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    reference_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = 0
    for token in prediction_tokens:
        available = reference_counts.get(token, 0)
        if available:
            overlap += 1
            reference_counts[token] = available - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Score against every reference and retain the best value per metric."""

    if isinstance(references, (str, bytes)) or not references:
        raise ValueError("references must contain at least one answer string")
    values = list(references)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("every reference answer must be a string")
    return {
        "normalized_exact_match": max(exact_match(prediction, value) for value in values),
        "normalized_token_f1": max(token_f1(prediction, value) for value in values),
    }


def answer_correctness_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(_LABEL_SCORES),
            }
        },
        "required": ["label"],
        "additionalProperties": False,
    }


def build_answer_correctness_prompt(
    *,
    question: str,
    reference_answers: Sequence[str],
    predicted_answer: str,
    rubric: Mapping[str, str],
) -> str:
    """Build a blind judge request containing no retrieval or action information."""

    expected_labels = set(_LABEL_SCORES)
    if set(rubric) != expected_labels:
        raise ValueError("rubric must define incorrect, partially_correct, and correct")
    payload = {
        "question": question,
        "reference_answers": list(reference_answers),
        "predicted_answer": predicted_answer,
    }
    rubric_lines = "\n".join(f"- {label}: {rubric[label]}" for label in _LABEL_SCORES)
    return "\n".join(
        [
            "Judge the predicted short answer against the reference answer(s).",
            "Use only the question, reference answer(s), and prediction below.",
            "Return exactly one rubric label through the required JSON schema.",
            "",
            "Rubric:",
            rubric_lines,
            "",
            "Inputs:",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]
    )


@dataclass(frozen=True, slots=True)
class AnswerCorrectnessResult:
    label: str
    score: float


def parse_answer_correctness(value: str | Mapping[str, Any]) -> AnswerCorrectnessResult:
    """Strictly parse one structured judge response."""

    try:
        payload = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Answer Correctness response is not a JSON object") from exc
    if set(payload) != {"label"}:
        raise ValueError("Answer Correctness response must contain only label")
    label = payload["label"]
    if label not in _LABEL_SCORES:
        raise ValueError(f"Unsupported Answer Correctness label: {label!r}")
    return AnswerCorrectnessResult(label=label, score=_LABEL_SCORES[label])


def linear_weighted_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Return linearly weighted Cohen kappa for the three ordered AC labels."""

    if len(first) != len(second):
        raise ValueError("label sequences must have the same length")
    if not first:
        return None
    labels = tuple(_LABEL_SCORES)
    positions = {label: position for position, label in enumerate(labels)}
    if any(label not in positions for label in [*first, *second]):
        raise ValueError("label sequences contain an unsupported AC label")

    count = float(len(first))
    observed_disagreement = sum(
        abs(positions[left] - positions[right]) / (len(labels) - 1)
        for left, right in zip(first, second)
    ) / count
    first_counts = {label: first.count(label) / count for label in labels}
    second_counts = {label: second.count(label) / count for label in labels}
    expected_disagreement = sum(
        first_counts[left]
        * second_counts[right]
        * abs(positions[left] - positions[right])
        / (len(labels) - 1)
        for left in labels
        for right in labels
    )
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else None
    return 1.0 - observed_disagreement / expected_disagreement


__all__ = [
    "AnswerCorrectnessResult",
    "answer_correctness_schema",
    "answer_metrics",
    "build_answer_correctness_prompt",
    "exact_match",
    "linear_weighted_kappa",
    "normalize_answer",
    "parse_answer_correctness",
    "token_f1",
]
