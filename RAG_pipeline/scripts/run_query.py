from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.io_utils import write_jsonl
from src.pipeline import NaiveRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one query through Naive RAG v1.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--question-id", default="manual_query")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    pipeline = NaiveRAGPipeline(config)
    result = pipeline.answer_question(args.query, question_id=args.question_id, top_k=args.top_k)

    if not args.no_log:
        outputs_dir = resolve_path(config, config.get("logging", {}).get("outputs_dir", "outputs/runs"))
        output_path = outputs_dir / f"{args.question_id}_result.jsonl"
        write_jsonl(output_path, [result])
        print(f"Saved result: {output_path}")

    print("\nAnswer:\n")
    print(result["generation"]["answer"])
    print("\nTop retrieved chunks:\n")
    for item in result["retrieval"]["results"]:
        print(json.dumps(
            {
                "rank": item["rank"],
                "score": round(item["score"], 4),
                "source": item["source"],
                "pages": f"{item['page_start']}-{item['page_end']}",
                "chunk_id": item["chunk_id"],
            },
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()

