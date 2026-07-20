"""在不重复检索或生成的情况下创建带追踪信息的重新分析运行。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import safe_run_id
from src.evaluators.logger import write_metadata_json, write_results, write_summary_csv
from src.evaluators.metrics import evaluate_result, summarize_results
from src.io_utils import read_jsonl, sha256_file
from src.provenance import evaluation_spec, json_sha256, source_code_sha256


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Recompute metrics for a completed run without new model calls.")
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--questions", default="data/questions_v1.jsonl", type=Path)
    parser.add_argument("--run-id", required=True, type=safe_run_id)
    args = parser.parse_args()

    source_dir = args.source_run_dir.resolve()
    questions_path = (PROJECT_ROOT / args.questions).resolve() if not args.questions.is_absolute() else args.questions.resolve()
    target_dir = source_dir.parent / args.run_id
    target_dir.mkdir(parents=True, exist_ok=False)

    source_metadata_path = source_dir / "metadata.json"
    source_results_path = source_dir / "results.jsonl"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    questions = {row["question_id"]: row for row in read_jsonl(questions_path)}
    source_rows = list(read_jsonl(source_results_path))
    if sha256_file(questions_path) != source_metadata.get("questions_sha256"):
        raise ValueError("Question file differs from the one used by the source generation run")

    rows = []
    for source_row in source_rows:
        row = copy.deepcopy(source_row)
        question_id = row["question_id"]
        question = questions.get(question_id)
        if question is None or question.get("question") != row.get("question"):
            raise ValueError(f"Question text mismatch for source row: {question_id}")
        labels = {
            "gold_answer": question.get("gold_answer"),
            "expected_sources": question.get("expected_sources", []),
            "expected_evidence": question.get("evidence", []),
            "answerable": question.get("answerable"),
        }
        row.update(labels)
        row["metrics"] = evaluate_result(
            row["generation"]["answer"],
            row["retrieval"]["results"],
            **labels,
        )
        row["run_id"] = args.run_id
        row["reanalysis_source_run_id"] = source_metadata["run_id"]
        rows.append(row)

    questions_sha = sha256_file(questions_path)
    reanalysis_source_sha = source_code_sha256(PROJECT_ROOT)
    evaluation_value = evaluation_spec(questions_sha, reanalysis_source_sha)
    metadata = copy.deepcopy(source_metadata)
    metadata.update(
        {
            "run_id": args.run_id,
            "command": "recompute_metrics",
            "status": "completed",
            "questions_path": str(questions_path),
            "questions_sha256": questions_sha,
            "evaluation_spec": evaluation_value,
            "evaluation_spec_sha256": json_sha256(evaluation_value),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "num_rows_written": len(rows),
            "num_failed_rows": sum(row.get("status") == "error" for row in rows),
            "process_end_to_end_latency_ms": (time.perf_counter() - started) * 1000,
            "reanalysis": {
                "source_run_id": source_metadata["run_id"],
                "source_metadata_sha256": sha256_file(source_metadata_path),
                "source_results_sha256": sha256_file(source_results_path),
                "source_sha256": reanalysis_source_sha,
                "retrieval_or_generation_repeated": False,
            },
        }
    )
    write_results(target_dir / "results.jsonl", rows, overwrite=False)
    write_summary_csv(target_dir / "summary.csv", summarize_results(rows))
    write_metadata_json(target_dir / "metadata.json", metadata, overwrite=False)
    print(f"Saved reanalysis: {target_dir}")


if __name__ == "__main__":
    main()
