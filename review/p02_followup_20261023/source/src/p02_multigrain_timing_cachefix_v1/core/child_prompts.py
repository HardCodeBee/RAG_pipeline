"""Exact historical prompt strings plus pure rendering; no legacy controller imports."""

from collections.abc import Iterable, Mapping
from typing import Any

CHILD_INSTRUCTIONS = (
    "Answer the current subquestion using the retrieved passages and the relevant prior question-answer pairs.\n"
    "The original question supplies scope; answer only the current subquestion.\n"
    "Resolve each #N from its numbered prior question and answer. Prior answers are candidate intermediate results, not verified evidence. "
    "Correct them when a retrieved passage directly supports a different fact about the same entity and scope.\n"
    "Return one concise factual sentence preserving the entity, relation and necessary qualifiers. "
    "If a needed fact is missing, state what cannot be determined; lack of evidence does not mean no, none, an empty set or zero. "
    "Do not add reasoning or citations."
)


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(
            f"{name} must be {'a string' if allow_empty else 'a non-empty string'}"
        )
    return value


def _record(record: Mapping[str, Any]) -> tuple[int, str, str]:
    if not isinstance(record, Mapping):
        raise ValueError("question-answer records must be mappings")
    step_id = record.get("step_id", record.get("id"))
    if type(step_id) is not int or step_id < 1:
        raise ValueError("question-answer step_id must be a positive integer")
    return (
        step_id,
        _text(record.get("request"), "request"),
        _text(record.get("answer"), "answer"),
    )


def _ordered_records(
    records: Iterable[Mapping[str, Any]],
) -> list[tuple[int, str, str]]:
    ordered = sorted((_record(record) for record in records), key=lambda item: item[0])
    ids = [item[0] for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("question-answer records contain duplicate step IDs")
    return ordered


def _question_answers(records: list[tuple[int, str, str]]) -> str:
    return "\n\n".join(
        f"#{step_id} Question: {request}\nAnswer: {answer}"
        for step_id, request, answer in records
    )


def child_prompt(
    original_question: str,
    current_step_id: int,
    current_request: str,
    dependencies: Iterable[Mapping[str, Any]],
    context: str,
) -> str:
    """Answer a raw #N-bearing question with direct prerequisite QA only.

    ``dependencies`` uses the same input fields as :func:`render_helper`.
    The runner selects exactly the direct references; the renderer checks
    only basic input shape and that all supplied IDs precede the current ID.
    ``context`` is the current step's frozen retrieval packet. No upstream
    source passages, entire execution history or gold references are added.
    """
    _text(original_question, "original_question")
    _text(current_request, "current_request")
    _text(context, "context", allow_empty=True)
    if type(current_step_id) is not int or current_step_id < 1:
        raise ValueError("current_step_id must be a positive integer")
    records = _ordered_records(dependencies)
    if any(step_id >= current_step_id for step_id, _, _ in records):
        raise ValueError("dependency IDs must precede the current step")
    prior = (
        _question_answers(records)
        if records
        else "No prior question-answer pairs are required."
    )
    return "\n".join(
        [
            CHILD_INSTRUCTIONS,
            "",
            "Original question:",
            original_question,
            "",
            "Relevant prior question-answer pairs:",
            prior,
            "",
            f"Current subquestion #{current_step_id}:",
            current_request,
            "",
            "Retrieved passages:",
            context,
            "",
            "Answer:",
        ]
    )


GOAL_RULE = (
    "For a combination, make the conclusion consistent with the prior results you use. "
    "If direct passage evidence changes a prior result, state that correction in your factual sentence. "
    "Equal or unresolved quantities do not justify choosing a winner; counts from incomplete lists "
    "are not established real-world totals."
)


def child(question, sid, request, dependencies, context, examples, *, role="ordinary"):
    live = child_prompt(question, sid, request, dependencies, context)
    if role == "ordinary":
        prefix = CHILD_INSTRUCTIONS + "\n\n"
        text = "\n\n".join(
            "Example " + str(i + 1) + "\n" + r["text"] for i, r in enumerate(examples)
        )
        live = (
            prefix
            + "Worked examples:\n\n"
            + text
            + "\n\nActual task:\n"
            + live[len(prefix) :]
        )
    return live
