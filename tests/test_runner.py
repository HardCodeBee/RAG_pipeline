from __future__ import annotations

import json
from datetime import datetime, timezone

from src.evaluators.runner import run_evaluation
from src.io_utils import read_jsonl


def _metadata(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "status": "running",
        "questions_sha256": "questions",
        "build_id": "build",
        "run_spec_sha256": "run",
        "evaluation_spec_sha256": "evaluation",
        "effective_top_k": 5,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_rows_written": 0,
    }


def test_runner_checkpoints_errors_and_retries_only_failed_rows(tmp_path) -> None:
    questions = [
        {"question_id": "q1", "question": "First?"},
        {"question_id": "q2", "question": "Second?"},
    ]
    run_dir = tmp_path / "run"
    first_calls = []

    def first_attempt(question):
        first_calls.append(question["question_id"])
        if question["question_id"] == "q2":
            raise ConnectionError("offline")
        return {
            "question_id": question["question_id"],
            "question": question["question"],
            "identity": {"run_spec_sha256": "run"},
            "status": "success",
            "metrics": {},
        }

    rows, _ = run_evaluation(
        questions=questions,
        run_dir=run_dir,
        metadata=_metadata("run"),
        resume=False,
        evaluate_question=first_attempt,
        summarize_rows=lambda values: {"num_questions": len(values)},
    )
    assert first_calls == ["q1", "q2"]
    assert [row["status"] for row in rows] == ["success", "error"]
    assert [row["question_id"] for row in read_jsonl(run_dir / "results.jsonl")] == ["q1", "q2"]

    resumed_calls = []

    def retry(question):
        resumed_calls.append(question["question_id"])
        return {
            "question_id": question["question_id"],
            "question": question["question"],
            "identity": {"run_spec_sha256": "run"},
            "status": "success",
            "metrics": {},
        }

    resumed_rows, _ = run_evaluation(
        questions=questions,
        run_dir=run_dir,
        metadata=_metadata("run"),
        resume=True,
        evaluate_question=retry,
        summarize_rows=lambda values: {"num_questions": len(values)},
    )
    assert resumed_calls == ["q2"]
    assert [row["status"] for row in resumed_rows] == ["success", "success"]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["num_failed_rows"] == 0
