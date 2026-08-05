"""Shared, evaluation-agnostic execution and checkpointing."""

from __future__ import annotations

# Shared evaluation orchestration stays separate from protocol metrics.

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.persistence.artifact_io import iter_jsonl
from src.persistence.run_output_writer import (
    write_metadata_json,
    write_result_checkpoint,
    write_results,
    write_summary_csv,
)


_RESUME_COMPATIBILITY_FIELDS = (
    "questions_sha256",
    "questions_file_sha256",
    "build_id",
    "run_spec_sha256",
    "evaluation_spec_sha256",
    "evaluation_protocol",
    "question_split",
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


def _checkpoint_path(checkpoints_dir: Path, position: int) -> Path:
    """Use the stable question position instead of an untrusted question id."""

    return checkpoints_dir / f"{position:06d}.json"


def _validate_result_row(
    row: Any,
    *,
    question: Mapping[str, Any],
    run_spec_sha256: str,
    source: str,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source} must contain a JSON object")
    question_id = question["question_id"]
    if row.get("question_id") != question_id:
        raise ValueError(f"{source} has a different question_id: {question_id}")
    if row.get("question") != question["question"]:
        raise ValueError(f"{source} has different question text: {question_id}")
    identity = row.get("identity")
    if not isinstance(identity, Mapping) or identity.get("run_spec_sha256") != run_spec_sha256:
        raise ValueError(f"{source} has an incompatible run identity: {question_id}")
    return row


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read result checkpoint: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Result checkpoint must contain a JSON object: {path}")
    return value


def _load_resumable_rows(
    *,
    questions: Sequence[Mapping[str, Any]],
    results_path: Path,
    checkpoints_dir: Path,
    run_spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    questions_by_id = {question["question_id"]: question for question in questions}
    rows_by_id: dict[str, dict[str, Any]] = {}

    if results_path.is_file():
        for row in iter_jsonl(results_path):
            question_id = row.get("question_id")
            if not question_id or question_id in rows_by_id:
                raise ValueError("Existing result rows must have unique, non-empty question_id values")
            question = questions_by_id.get(question_id)
            if question is None:
                raise ValueError(
                    f"Existing result row is not part of the current question set: {question_id}"
                )
            rows_by_id[question_id] = _validate_result_row(
                row,
                question=question,
                run_spec_sha256=run_spec_sha256,
                source="Existing result row",
            )

    if not checkpoints_dir.is_dir():
        return rows_by_id

    expected_paths = {
        _checkpoint_path(checkpoints_dir, position).name
        for position in range(1, len(questions) + 1)
    }
    unexpected = sorted(
        path.name for path in checkpoints_dir.glob("*.json") if path.name not in expected_paths
    )
    if unexpected:
        raise ValueError("Unexpected result checkpoint files: " + ", ".join(unexpected))

    for position, question in enumerate(questions, start=1):
        checkpoint_path = _checkpoint_path(checkpoints_dir, position)
        if not checkpoint_path.is_file():
            continue
        row = _validate_result_row(
            _load_checkpoint(checkpoint_path),
            question=question,
            run_spec_sha256=run_spec_sha256,
            source=f"Result checkpoint {checkpoint_path.name}",
        )
        # A checkpoint records the latest attempt and therefore overrides the
        # same question's row from a previously merged results.jsonl.
        rows_by_id[question["question_id"]] = row
    return rows_by_id


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
    checkpoints_dir = run_dir / "checkpoints"

    if resume:
        if not run_dir.is_dir() or not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume an incomplete or missing run directory: {run_dir}")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_resume_compatibility(previous_metadata, metadata)
        metadata["started_at"] = previous_metadata.get("started_at", metadata["started_at"])
        metadata["resumed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoints_dir.mkdir()
        write_metadata_json(metadata_path, metadata, overwrite=False)

    rows_by_id = _load_resumable_rows(
        questions=questions,
        results_path=results_path,
        checkpoints_dir=checkpoints_dir,
        run_spec_sha256=metadata["run_spec_sha256"],
    )
    recovered_rows = _ordered_rows(questions, rows_by_id)
    num_rows_written = len(recovered_rows)
    num_failed_rows = sum(
        row.get("status") == "error" for row in recovered_rows
    )
    metadata["num_rows_written"] = num_rows_written
    metadata["num_failed_rows"] = num_failed_rows
    write_metadata_json(metadata_path, metadata)

    for position, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        existing = rows_by_id.get(question_id)
        if existing is not None and existing.get("status") != "error":
            continue
        try:
            row = evaluate_question(question)
            row["run_id"] = metadata["run_id"]
            row = _validate_result_row(
                row,
                question=question,
                run_spec_sha256=metadata["run_spec_sha256"],
                source="Evaluated result row",
            )
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
        if existing is None:
            num_rows_written += 1
        else:
            num_failed_rows -= int(existing.get("status") == "error")
        num_failed_rows += int(row.get("status") == "error")
        rows_by_id[question_id] = row
        write_result_checkpoint(_checkpoint_path(checkpoints_dir, position), row)
        metadata["num_rows_written"] = num_rows_written
        metadata["num_failed_rows"] = num_failed_rows
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
