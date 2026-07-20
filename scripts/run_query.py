"""运行一次查询，并可选地把追踪结果保存到不可变运行目录。"""

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
from src.evaluators.logger import write_metadata_json, write_results
from src.pipeline import NaiveRAGPipeline
from src.provenance import recorded_config, resolved_roots


def main() -> None:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run one query through the baseline RAG pipeline.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to a YAML config")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--question-id", default="manual_query")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    configure_utf8_output()

    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = apply_cli_overrides(load_config(config_path), top_k=args.top_k)
    started_at = datetime.now(timezone.utc).isoformat()
    pipeline = NaiveRAGPipeline(config)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("query_%Y%m%d_%H%M%S_")
        + pipeline.runtime_metadata["run_spec_sha256"][:8]
    )
    result = pipeline.query(args.query, question_id=args.question_id)
    result["run_id"] = run_id
    process_latency_ms = (time.perf_counter() - process_started) * 1000

    if not args.no_log:
        run_dir = resolved_roots(config)["outputs_root"] / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id,
            "command": "run_query",
            "status": "completed",
            "config_path": str(config_path),
            "effective_config": recorded_config(config),
            **pipeline.runtime_metadata,
            "effective_top_k": config["retrieval"]["top_k"],
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "process_end_to_end_latency_ms": process_latency_ms,
        }
        write_results(run_dir / "results.jsonl", [result], overwrite=False)
        write_metadata_json(run_dir / "metadata.json", metadata, overwrite=False)
        print(f"Saved run: {run_dir}")

    print("\nAnswer:\n")
    print(result["generation"]["answer"])
    print("\nTop retrieved chunks:\n")
    for item in result["retrieval"]["results"]:
        print(
            json.dumps(
                {
                    "rank": item["rank"],
                    "score": round(item["score"], 4),
                    "source": item["source"],
                    "pages": f"{item['page_start']}-{item['page_end']}",
                    "chunk_id": item["chunk_id"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
