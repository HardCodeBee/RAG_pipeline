"""Evaluate the generic JSONL question schema against one immutable build."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import configure_utf8_output, positive_int, safe_run_id
from src.config import apply_cli_overrides, load_config, resolve_cli_path
from src.evaluators.metrics import evaluate_result, summarize_results
from src.evaluators.runner import run_evaluation, validate_questions
from src.io_utils import read_jsonl, sha256_file
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    source_group_sha256,
)


def _evaluate_generic_question(
    pipeline: NaiveRAGPipeline,
    question: Mapping[str, Any],
) -> dict[str, Any]:
    labels = {
        "gold_answer": question.get("gold_answer"),
        "expected_sources": question.get("expected_sources", []),
        "expected_evidence": question.get("evidence", []),
        "answerable": question.get("answerable"),
    }
    row = pipeline.query(
        question["question"],
        question_id=question["question_id"],
    )
    row.update(labels)
    row["question_type"] = question.get("question_type")
    row["metrics"] = evaluate_result(
        row["generation"]["answer"],
        row["retrieval"]["results"],
        **labels,
    )
    return row


def _generic_error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gold_answer": question.get("gold_answer"),
        "expected_sources": question.get("expected_sources", []),
        "expected_evidence": question.get("evidence", []),
        "answerable": question.get("answerable"),
        "question_type": question.get("question_type"),
        "metrics": {},
    }


def main(argv: Sequence[str] | None = None) -> Path:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate the baseline RAG pipeline on JSONL questions.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to a YAML config")
    parser.add_argument("--questions", default="data/questions_v1.jsonl", help="Question JSONL file")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = apply_cli_overrides(load_config(config_path), top_k=args.top_k)
    questions_path = resolve_cli_path(PROJECT_ROOT, args.questions)
    questions = validate_questions(read_jsonl(questions_path))
    questions_sha = sha256_file(questions_path)
    pipeline = NaiveRAGPipeline(config)
    evaluation_source_sha = source_group_sha256(PROJECT_ROOT, "evaluation")
    evaluation_value = evaluation_spec(questions_sha, evaluation_source_sha)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")
        + pipeline.runtime_metadata["run_spec_sha256"][:8]
    )
    run_dir = resolved_roots(config)["outputs_root"] / run_id
    metadata = {
        "run_id": run_id,
        "command": "run_eval",
        "status": "running",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "questions_path": str(questions_path),
        "questions_source": None,
        "questions_sha256": questions_sha,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha,
        **pipeline.runtime_metadata,
        "requested_top_k": args.top_k,
        "effective_top_k": config["retrieval"]["top_k"],
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
    }
    _, summary = run_evaluation(
        questions=questions,
        run_dir=run_dir,
        metadata=metadata,
        resume=args.resume,
        evaluate_question=lambda question: _evaluate_generic_question(pipeline, question),
        summarize_rows=summarize_results,
        error_fields=_generic_error_fields,
        process_started=process_started,
    )
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
