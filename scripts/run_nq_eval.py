"""Evaluate the frozen NQ Open split against the encoded DPR Wikipedia corpus."""

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

from scripts.cli_support import (
    configure_utf8_output,
    non_negative_int,
    positive_int,
    safe_run_id,
    temporary_openai_api_key,
)
from src.config import apply_cli_overrides, load_config, resolve_cli_path
from src.evaluators.nq_metrics import (
    positive_chunk_ids_from_evidence,
    score_nq_retrieval,
    score_nq_question,
)
from src.evaluators.nq_evaluation import (
    EVALUATION_PROTOCOL,
    METRICS_VERSION,
    RETRIEVAL_METRICS_VERSION,
    canonical_question_split,
    summarize_nq_run,
    validate_nq_questions,
    validate_openai_generation,
)
from src.evaluation_runner import run_evaluation
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    source_group_sha256,
)


def main(argv: Sequence[str] | None = None) -> Path:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Evaluate NQ Open questions on the frozen DPR Wikipedia subset."
    )
    parser.add_argument("--config", default="configs/nq_dpr_wiki.yaml")
    parser.add_argument(
        "--questions",
        default=None,
        help="Optional explicit path; it must match the canonical selected split",
    )
    parser.add_argument(
        "--split",
        choices=("calibration", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=non_negative_int, default=None)
    parser.add_argument("--max-questions", type=positive_int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval without calling a generator",
    )
    parser.add_argument(
        "--nprobe",
        type=positive_int,
        default=None,
        help="Override IVF nprobe without creating another config file",
    )
    parser.add_argument(
        "--ef-search",
        type=positive_int,
        default=None,
        help="Override HNSW efSearch without creating another config file",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="Local key file inside the project; never written to run metadata",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = apply_cli_overrides(load_config(config_path), top_k=args.top_k)
    index_type = config["index"]["type"]
    if args.nprobe is not None:
        if index_type not in {"ivf_flat", "ivf_pq"}:
            parser.error("--nprobe requires an IVF index config")
        config["retrieval"]["nprobe"] = args.nprobe
    if args.ef_search is not None:
        if index_type != "hnsw_flat":
            parser.error("--ef-search requires an HNSW index config")
        config["retrieval"]["ef_search"] = args.ef_search
    if args.retrieval_only:
        # The pipeline still assembles its configured components, but this local
        # generator is never called and avoids requiring remote credentials.
        config["generation"] = {
            "provider": "extractive",
            "max_output_tokens": 1,
        }
    if config["loader"]["type"] != "dpr_wikipedia":
        raise ValueError("NQ evaluation requires loader.type=dpr_wikipedia")
    if config["loader"]["expected_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("NQ evaluation protocol does not match the loader")
    if config["index"]["backend"] != "faiss":
        raise ValueError("NQ evaluation requires a real FAISS index")
    if not args.retrieval_only and config["generation"]["provider"] != "openai":
        raise ValueError("NQ evaluation requires the OpenAI generator")
    if not config["logging"]["save_retrieved_text"]:
        raise ValueError("NQ answer-passage metrics require logging.save_retrieved_text=true")

    corpus_path = resolved_roots(config)["corpus"]
    requested_questions_path = (
        resolve_cli_path(PROJECT_ROOT, args.questions)
        if args.questions is not None
        else None
    )
    verified_questions = canonical_question_split(
        corpus_path,
        split=args.split,
        requested_path=requested_questions_path,
    )
    questions_path = verified_questions.questions_path
    questions_manifest = verified_questions.questions_manifest
    all_questions = validate_nq_questions(verified_questions.rows)
    questions = (
        all_questions[: args.max_questions]
        if args.max_questions is not None
        else all_questions
    )
    if not questions:
        raise RuntimeError("The selected NQ question split is empty")

    with temporary_openai_api_key(args.api_key_file, allowed_root=PROJECT_ROOT):
        pipeline = NaiveRAGPipeline(config)
    manifest = pipeline.manifest

    selected_questions_sha = json_sha256(questions)
    questions_file_sha = verified_questions.questions_file_sha256
    evaluation_source_sha = source_group_sha256(PROJECT_ROOT, "evaluation")
    metrics_version = (
        RETRIEVAL_METRICS_VERSION if args.retrieval_only else METRICS_VERSION
    )
    evaluation_value = evaluation_spec(
        selected_questions_sha,
        evaluation_source_sha,
        metrics_version=metrics_version,
    )
    run_id = args.run_id or (
        ("nq_retrieval_" if args.retrieval_only else "nq_")
        + f"{args.split}_"
        + ("full_" if args.max_questions is None else f"n{args.max_questions}_")
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    run_dir = resolved_roots(config)["outputs_root"] / run_id

    def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
        positive_ids = positive_chunk_ids_from_evidence(question["evidence"])
        if args.retrieval_only:
            started = time.perf_counter()
            retrieval, rerank_trace = pipeline.retrieve_with_details(question["question"])
            retrieved_rows = [item.to_dict() for item in retrieval.results]
            row = {
                "status": "success",
                "question_id": question["question_id"],
                "question": question["question"],
                "identity": {
                    "build_id": pipeline.runtime_metadata["build_id"],
                    "run_spec_sha256": pipeline.runtime_metadata["run_spec_sha256"],
                },
                "retrieval": {
                    "candidate_k": config["retrieval"]["candidate_k"],
                    "top_k": retrieval.top_k,
                    "results": retrieved_rows,
                    "latency_ms": retrieval.latency_ms,
                    "timings_ms": dict(retrieval.timings_ms),
                },
                "total_latency_ms": (time.perf_counter() - started) * 1000,
            }
            if rerank_trace is not None:
                rerank_data = rerank_trace.to_dict()
                candidate_rows = [item.to_dict() for item in rerank_trace.candidates]
                rerank_data["dense_baseline_metrics"] = score_nq_retrieval(
                    candidate_rows[: retrieval.top_k],
                    question["answers"],
                    positive_ids,
                    top_k=retrieval.top_k,
                )
                rerank_data["candidate_metrics"] = score_nq_retrieval(
                    candidate_rows,
                    question["answers"],
                    positive_ids,
                    top_k=len(candidate_rows),
                )
                row["retrieval"]["rerank"] = rerank_data
            row["metrics"] = score_nq_retrieval(
                retrieved_rows,
                question["answers"],
                positive_ids,
                top_k=config["retrieval"]["final_k"],
            )
        else:
            row = pipeline.query(
                question["question"],
                question_id=question["question_id"],
            )
            validate_openai_generation(row["generation"])
            row["metrics"] = score_nq_question(
                row["generation"]["answer"],
                row["retrieval"]["results"],
                question["answers"],
                positive_ids,
                top_k=config["retrieval"]["final_k"],
            )
        row["answers"] = list(question["answers"])
        row["expected_evidence"] = list(question["evidence"])
        row["answerable"] = True
        row["question_type"] = question.get("question_type")
        row["evaluation"] = {
            "protocol": EVALUATION_PROTOCOL,
            "split": args.split,
        }
        return row

    def error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "answers": list(question["answers"]),
            "expected_evidence": list(question["evidence"]),
            "answerable": True,
            "question_type": question.get("question_type"),
            "metrics": {},
            "evaluation": {
                "protocol": EVALUATION_PROTOCOL,
                "split": args.split,
            },
        }

    metadata = {
        "run_id": run_id,
        "command": "run_nq_eval",
        "status": "running",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "questions_path": str(questions_path),
        "questions_file_sha256": questions_file_sha,
        "questions_manifest_sha256": verified_questions.questions_manifest_sha256,
        "questions_selection": questions_manifest["selection"],
        "questions_sha256": selected_questions_sha,
        "question_split": args.split,
        "max_questions": args.max_questions,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "metrics_version": metrics_version,
        "execution_mode": "retrieval_only" if args.retrieval_only else "rag",
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha,
        **pipeline.runtime_metadata,
        "requested_top_k": args.top_k,
        "effective_top_k": config["retrieval"]["final_k"],
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
    }
    try:
        rows, summary = run_evaluation(
            questions=questions,
            run_dir=run_dir,
            metadata=metadata,
            resume=args.resume,
            evaluate_question=evaluate_question,
            summarize_rows=lambda rows: summarize_nq_run(
                rows,
                manifest=manifest,
                split=args.split,
                effective_top_k=config["retrieval"]["final_k"],
                retrieval_only=args.retrieval_only,
            ),
            error_fields=error_fields,
            process_started=process_started,
        )
    finally:
        pipeline.close()
    failed_rows = [row for row in rows if row.get("status") != "success"]
    if failed_rows:
        raise RuntimeError(
            f"NQ evaluation saved {len(failed_rows)} failed row(s); use --resume after "
            "fixing the cause"
        )
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
