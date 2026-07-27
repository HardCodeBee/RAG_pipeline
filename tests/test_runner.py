from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.run_eval import _concise_terminal_summary
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


def test_concise_terminal_summary_groups_only_demo_metrics() -> None:
    summary = {
        "num_questions": 24,
        "num_successful_questions": 24,
        "num_failed_questions": 0,
        "avg_total_latency_ms": 2541.7,
        "p95_total_latency_ms": 4632.1,
        "retrieval_expected_source_hit_rate": 0.9545,
        "retrieval_evidence_recall_at_k": 0.7273,
        "retrieval_evidence_mrr": 0.5076,
        "answer_token_f1": 0.2615,
        "answerability_decision_accuracy": 0.875,
        "answer_exact_match_rate": 0.0,
    }

    concise = _concise_terminal_summary(summary)

    assert concise == {
        "questions": {"total": 24, "successful": 24, "failed": 0},
        "latency_ms": {"avg_total": 2541.7, "p95_total": 4632.1},
        "retrieval": {
            "expected_source_hit_rate": 0.9545,
            "evidence_recall_at_k": 0.7273,
            "evidence_mrr": 0.5076,
        },
        "answer": {
            "token_f1": 0.2615,
            "answerability_decision_accuracy": 0.875,
        },
    }
    assert "answer_exact_match_rate" not in concise


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
