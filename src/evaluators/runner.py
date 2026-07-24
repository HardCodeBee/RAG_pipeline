"""Shared, evaluation-agnostic execution and checkpointing."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluators.logger import write_metadata_json, write_results, write_summary_csv
from src.io_utils import read_jsonl


_RESUME_COMPATIBILITY_FIELDS = (
    "questions_sha256",
    "build_id",
    "run_spec_sha256",
    "evaluation_spec_sha256",
    "effective_top_k",
)


def validate_resume_compatibility(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    mismatches = [
        field for field in _RESUME_COMPATIBILITY_FIELDS if previous.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError("Cannot resume an incompatible run; mismatched metadata: " + ", ".join(mismatches))


def validate_questions(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for position, item in enumerate(items, start=1):
        question = dict(item)
        question["question_id"] = question.get("question_id") or f"row_{position:06d}"
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"Question {question['question_id']} has no non-empty question text")
        questions.append(question)
    question_ids = [item["question_id"] for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question ids must be unique within an evaluation set")
    return questions


def _ordered_rows(
    questions: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        rows_by_id[question["question_id"]]
        for question in questions
        if question["question_id"] in rows_by_id
    ]


def run_evaluation(
    *,
    questions: Sequence[Mapping[str, Any]],
    run_dir: Path,
    metadata: dict[str, Any],
    resume: bool,
    evaluate_question: Callable[[Mapping[str, Any]], dict[str, Any]],
    summarize_rows: Callable[[list[dict[str, Any]]], dict[str, Any]],
    error_fields: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    process_started: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run questions in order, checkpoint atomically, and support exact resume."""

    process_started = process_started if process_started is not None else time.perf_counter()
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"
    questions_by_id = {question["question_id"]: question for question in questions}

    if resume:
        if not results_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume an incomplete or missing run directory: {run_dir}")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_resume_compatibility(previous_metadata, metadata)
        existing_rows = list(read_jsonl(results_path))
        metadata["started_at"] = previous_metadata.get("started_at", metadata["started_at"])
        metadata["resumed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["num_rows_written"] = len(existing_rows)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        existing_rows = []
        write_results(results_path, [], overwrite=False)
        write_metadata_json(metadata_path, metadata, overwrite=False)

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        question_id = row.get("question_id")
        if not question_id or question_id in rows_by_id:
            raise ValueError("Existing result rows must have unique, non-empty question_id values")
        if question_id not in questions_by_id:
            raise ValueError(f"Existing result row is not part of the current question set: {question_id}")
        if row.get("question") != questions_by_id[question_id]["question"]:
            raise ValueError(f"Existing result row has different question text: {question_id}")
        if row.get("identity", {}).get("run_spec_sha256") != metadata["run_spec_sha256"]:
            raise ValueError(f"Existing row has an incompatible run identity: {question_id}")
        rows_by_id[question_id] = row
    write_metadata_json(metadata_path, metadata)

    for question in questions:
        question_id = question["question_id"]
        existing = rows_by_id.get(question_id)
        if existing is not None and existing.get("status") != "error":
            continue
        try:
            row = evaluate_question(question)
            row["run_id"] = metadata["run_id"]
        except Exception as exc:
            identity = {
                field: metadata[field]
                for field in ("build_id", "run_spec_sha256")
                if field in metadata
            }
            row = {
                "question_id": question_id,
                "question": question["question"],
                "identity": identity,
                "run_id": metadata["run_id"],
                "status": "error",
                "error": {"type": exc.__class__.__name__, "message": str(exc)[:1000]},
                "metrics": {},
            }
            if error_fields is not None:
                row.update(error_fields(question))
        rows_by_id[question_id] = row
        rows = _ordered_rows(questions, rows_by_id)
        write_results(results_path, rows)
        metadata["num_rows_written"] = len(rows)
        metadata["num_failed_rows"] = sum(item.get("status") == "error" for item in rows)
        write_metadata_json(metadata_path, metadata)

    rows = _ordered_rows(questions, rows_by_id)
    summary = summarize_rows(rows)
    write_results(results_path, rows)
    write_summary_csv(summary_path, summary)
    metadata["status"] = "completed_with_errors" if any(
        row.get("status") == "error" for row in rows
    ) else "completed"
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["num_rows_written"] = len(rows)
    metadata["num_failed_rows"] = sum(row.get("status") == "error" for row in rows)
    metadata["process_end_to_end_latency_ms"] = (time.perf_counter() - process_started) * 1000
    metadata["summary"] = summary
    write_metadata_json(metadata_path, metadata)
    return rows, summary
