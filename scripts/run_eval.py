"""在 JSONL 问题集上评估一个不可变的构建、运行和评估身份。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import configure_utf8_output, positive_int, safe_run_id
from src.config import apply_cli_overrides, load_config, resolve_cli_path
from src.evaluators.logger import write_metadata_json, write_results, write_summary_csv
from src.evaluators.metrics import summarize_results
from src.io_utils import read_jsonl, sha256_file
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    PIPELINE_SCHEMA_VERSION,
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
)


_RESUME_COMPATIBILITY_FIELDS = (
    "schema_version",
    "questions_sha256",
    "build_id",
    "run_spec_sha256",
    "evaluation_spec_sha256",
    "source_sha256",
    "effective_top_k",
)


def validate_resume_compatibility(previous: dict, current: dict) -> None:
    mismatches = [
        field for field in _RESUME_COMPATIBILITY_FIELDS if previous.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError("Cannot resume an incompatible run; mismatched metadata: " + ", ".join(mismatches))


def _ordered_rows(questions: list[dict], rows_by_id: dict[str, dict]) -> list[dict]:
    ordered_ids = [question["question_id"] for question in questions]
    known = set(ordered_ids)
    rows = [rows_by_id[question_id] for question_id in ordered_ids if question_id in rows_by_id]
    rows.extend(row for question_id, row in rows_by_id.items() if question_id not in known)
    return rows


def main() -> None:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate the baseline RAG pipeline on JSONL questions.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to a YAML config")
    parser.add_argument("--questions", default="data/questions_v1.jsonl", help="Question JSONL file")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    questions_path = resolve_cli_path(PROJECT_ROOT, args.questions)
    config = apply_cli_overrides(load_config(config_path), top_k=args.top_k)
    questions = []
    for position, item in enumerate(read_jsonl(questions_path), start=1):
        question = dict(item)
        question["question_id"] = question.get("question_id") or f"row_{position:06d}"
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"Question {question['question_id']} has no non-empty question text")
        questions.append(question)
    question_ids = [item["question_id"] for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question ids must be unique within an evaluation set")

    started_at = datetime.now(timezone.utc).isoformat()
    questions_sha = sha256_file(questions_path)
    pipeline = NaiveRAGPipeline(config)
    evaluation_value = evaluation_spec(questions_sha, pipeline.runtime_metadata["source_sha256"])
    evaluation_sha = json_sha256(evaluation_value)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")
        + pipeline.runtime_metadata["run_spec_sha256"][:8]
    )
    run_dir = resolved_roots(config)["outputs_root"] / run_id
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"

    metadata = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "command": "run_eval",
        "status": "running",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "questions_path": str(questions_path),
        "questions_sha256": questions_sha,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": evaluation_sha,
        **pipeline.runtime_metadata,
        "requested_top_k": args.top_k,
        "effective_top_k": config["retrieval"]["top_k"],
        "resume": args.resume,
        "started_at": started_at,
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
    }

    if args.resume:
        if not results_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume an incomplete or missing run directory: {run_dir}")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_resume_compatibility(previous_metadata, metadata)
        existing_rows = list(read_jsonl(results_path))
        metadata["started_at"] = previous_metadata.get("started_at", started_at)
        metadata["resumed_at"] = started_at
        metadata["num_rows_written"] = len(existing_rows)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        existing_rows = []
        write_results(results_path, [], overwrite=False)
        write_metadata_json(metadata_path, metadata, overwrite=False)

    rows_by_id: dict[str, dict] = {}
    for row in existing_rows:
        question_id = row.get("question_id")
        if not question_id or question_id in rows_by_id:
            raise ValueError("Existing result rows must have unique, non-empty question_id values")
        if row.get("identity", {}).get("run_spec_sha256") != metadata["run_spec_sha256"]:
            raise ValueError(f"Existing row has an incompatible run identity: {question_id}")
        rows_by_id[question_id] = row
    write_metadata_json(metadata_path, metadata)

    for question in questions:
        existing = rows_by_id.get(question["question_id"])
        if existing is not None and existing.get("status") != "error":
            continue
        labels = {
            "gold_answer": question.get("gold_answer"),
            "expected_sources": question.get("expected_sources", []),
            "expected_evidence": question.get("evidence", []),
            "answerable": question.get("answerable"),
        }
        try:
            row = pipeline.answer_question(
                question["question"],
                question_id=question["question_id"],
                **labels,
            )
            row["run_id"] = run_id
            row["question_type"] = question.get("question_type")
        except Exception as exc:
            row = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "question_id": question["question_id"],
                "question": question["question"],
                "identity": {
                    "build_id": metadata["build_id"],
                    "run_spec_sha256": metadata["run_spec_sha256"],
                    "source_sha256": metadata["source_sha256"],
                },
                "run_id": run_id,
                "status": "error",
                "error": {"type": exc.__class__.__name__, "message": str(exc)[:1000]},
                **labels,
                "question_type": question.get("question_type"),
                "metrics": {},
            }
        rows_by_id[question["question_id"]] = row
        rows = _ordered_rows(questions, rows_by_id)
        write_results(results_path, rows)
        metadata["num_rows_written"] = len(rows)
        metadata["num_failed_rows"] = sum(item.get("status") == "error" for item in rows)
        write_metadata_json(metadata_path, metadata)
        if row["status"] == "error" and config["strict_backends"]:
            write_summary_csv(summary_path, summarize_results(rows))
            metadata["status"] = "failed"
            metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            metadata["error"] = row["error"]
            metadata["process_end_to_end_latency_ms"] = (time.perf_counter() - process_started) * 1000
            write_metadata_json(metadata_path, metadata)
            raise RuntimeError(f"Baseline run failed for question {question['question_id']}")

    rows = _ordered_rows(questions, rows_by_id)
    summary = summarize_results(rows)
    write_results(results_path, rows)
    write_summary_csv(summary_path, summary)
    metadata["status"] = "completed_with_errors" if summary["num_failed_questions"] else "completed"
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["num_rows_written"] = len(rows)
    metadata["num_failed_rows"] = summary["num_failed_questions"]
    metadata["process_end_to_end_latency_ms"] = (time.perf_counter() - process_started) * 1000
    write_metadata_json(metadata_path, metadata)

    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
