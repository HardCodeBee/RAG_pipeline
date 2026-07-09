from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.evaluators.logger import write_results, write_summary_csv
from src.evaluators.metrics import summarize_results
from src.io_utils import read_jsonl
from src.pipeline import NaiveRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Naive RAG v1 on a JSONL question set.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--questions", default="data/questions.jsonl", help="Question JSONL file")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    questions_path = resolve_path(config, args.questions)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    outputs_dir = resolve_path(config, config.get("logging", {}).get("outputs_dir", "outputs/runs"))
    results_path = outputs_dir / f"{run_id}_results.jsonl"
    summary_path = outputs_dir / f"{run_id}_summary.csv"

    pipeline = NaiveRAGPipeline(config)
    rows = []
    for question in read_jsonl(questions_path):
        rows.append(
            pipeline.answer_question(
                question["question"],
                question_id=question.get("question_id"),
                gold_answer=question.get("gold_answer"),
                expected_sources=question.get("expected_sources", []),
                top_k=args.top_k,
            )
        )

    summary = summarize_results(rows)
    write_results(results_path, rows)
    write_summary_csv(summary_path, summary)

    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

