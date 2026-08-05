"""Run the fixed retrieval-focused QASPER protocol on the global paper corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scripts.cli_support import (
    configure_utf8_output,
    non_negative_int,
    positive_int,
    safe_run_id,
    temporary_openai_api_key,
)
from src.config import apply_cli_overrides, load_config, resolve_cli_path, validate_config
from src.evaluators.operational_metrics import summarize_execution
from src.evaluators.qasper_metrics import (
    score_qasper_open_corpus,
    score_qasper_open_corpus_retrieval,
    summarize_qasper_open_corpus,
    summarize_qasper_open_corpus_retrieval,
)
from src.evaluation_runner import run_evaluation, validate_questions
from src.loaders.qasper_loader import (
    QASPER_EVALUATION_SLICE,
    QASPER_SPLITS,
    load_qasper_dataset,
    qasper_evaluation_questions,
    qasper_evaluation_slice_stats,
    qasper_unit_records,
)
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    source_group_sha256,
)


EVALUATION_PROTOCOL = "qasper_open_corpus_text_extractive_single_evidence_v2"
RETRIEVAL_METRICS_VERSION = "qasper_open_corpus_retrieval_only_v1"


def load_qasper_eval_config(
    config_value: str | Path,
    *,
    retrieval_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Load and validate one explicit formal QASPER configuration."""

    config_path = resolve_cli_path(PROJECT_ROOT, config_value)
    config = load_config(config_path)

    required = {
        "QASPER all-split loader": config["loader"]
        == {"type": "qasper", "split": "all", "max_documents": None},
        "sentence_transformers": config["embedding"]["backend"] == "sentence_transformers",
        "faiss": config["index"]["backend"] == "faiss",
        "openai": retrieval_only or config["generation"]["provider"] == "openai",
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing:
        raise ValueError("QASPER full evaluation requires: " + ", ".join(missing))
    if not retrieval_only and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("QASPER full evaluation requires OPENAI_API_KEY")
    return config, config_path


def qasper_eval_inputs(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_questions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Derive validation questions and the global train/validation/test paper map."""

    articles_by_id: dict[str, Mapping[str, Any]] = {}
    for split in QASPER_SPLITS:
        if split not in dataset:
            raise ValueError(f"QASPER split is not available: {split}")
        for article in dataset[split]:
            paper_id = str(article.get("id") or "").strip()
            if not paper_id or paper_id in articles_by_id:
                raise ValueError("QASPER paper ids must be non-empty and globally unique")
            articles_by_id[paper_id] = article

    questions: list[dict[str, Any]] = []
    for article in dataset["validation"]:
        for question in qasper_evaluation_questions(article):
            questions.append({**question, "expected_sources": [question["paper_id"]]})
            if max_questions is not None and len(questions) >= max_questions:
                return questions, articles_by_id
    return questions, articles_by_id


def _failed_score(*, retrieval_only: bool) -> dict[str, Any]:
    result = {
        "qasper_target_paper_hit_at_k": False,
        "qasper_target_paper_rr": 0.0,
        "qasper_target_evidence_hit_at_k": False,
        "qasper_target_evidence_recall_at_k": 0.0,
        "qasper_target_evidence_f1_at_k": 0.0,
    }
    if not retrieval_only:
        result["qasper_answer_f1"] = 0.0
    return result


def _evaluation_fields(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "target_paper_id": question["paper_id"],
        "reference_count": len(question["references"]),
        "source_reference_count": question["source_reference_count"],
    }


def _qasper_summary(
    rows: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    effective_top_k: int,
    candidate_k: int,
    slice_stats: Mapping[str, Any],
    retrieval_only: bool,
) -> dict[str, Any]:
    scores = [
        row.get("metrics") or _failed_score(retrieval_only=retrieval_only)
        for row in rows
    ]
    execution = summarize_execution(rows)
    protocol_metrics = (
        summarize_qasper_open_corpus_retrieval(scores)
        if retrieval_only
        else summarize_qasper_open_corpus(scores)
    )
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "question_split": "validation",
        "num_candidate_questions_before_slice": slice_stats["candidate_questions"],
        "num_eligible_questions": slice_stats["selected_questions"],
        "num_candidate_references_before_slice": slice_stats["candidate_references"],
        "num_eligible_references": slice_stats["selected_references"],
        "corpus_splits": "+".join(QASPER_SPLITS),
        "num_corpus_papers": manifest["corpus"]["num_documents"],
        "num_index_chunks": manifest["index"]["count"],
        "candidate_k": candidate_k,
        "effective_top_k": effective_top_k,
        "embedding_backend": manifest["embedding"]["space"]["backend"],
        "index_backend": manifest["index"]["backend"],
        "generation_provider": "none" if retrieval_only else "openai",
        **protocol_metrics,
        "num_successful_questions": execution["num_successful_questions"],
        "num_failed_questions": execution["num_failed_questions"],
        "avg_retrieval_latency_ms": execution["avg_retrieval_latency_ms"],
        "p50_retrieval_latency_ms": execution["p50_retrieval_latency_ms"],
        "p95_retrieval_latency_ms": execution["p95_retrieval_latency_ms"],
        "avg_rerank_latency_ms": execution["avg_rerank_latency_ms"],
        "p50_rerank_latency_ms": execution["p50_rerank_latency_ms"],
        "p95_rerank_latency_ms": execution["p95_rerank_latency_ms"],
        "avg_generation_latency_ms": execution["avg_generation_latency_ms"],
        "avg_total_latency_ms": execution["avg_total_latency_ms"],
        "p95_total_latency_ms": execution["p95_total_latency_ms"],
        "total_provider_input_tokens": execution["total_provider_input_tokens"],
        "total_provider_output_tokens": execution["total_provider_output_tokens"],
        "total_provider_total_tokens": execution["total_provider_total_tokens"],
    }


def main(argv: Sequence[str] | None = None) -> Path:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Evaluate the retrieval-focused QASPER slice on the global paper corpus."
    )
    parser.add_argument("--config", default="configs/qasper_baseline.yaml")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=non_negative_int, default=None)
    parser.add_argument("--candidate-k", type=non_negative_int, default=None)
    parser.add_argument("--max-questions", type=positive_int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval without calling a generator",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Enable a cross-encoder reranker with this pinned model",
    )
    parser.add_argument("--reranker-revision", default=None)
    parser.add_argument("--reranker-batch-size", type=positive_int, default=32)
    parser.add_argument(
        "--reranker-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--reranker-local-files-only", action="store_true")
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
    if args.reranker_model is not None and not args.reranker_revision:
        parser.error("--reranker-model requires --reranker-revision")
    if args.reranker_model is None and args.reranker_revision is not None:
        parser.error("--reranker-revision requires --reranker-model")

    with temporary_openai_api_key(args.api_key_file, allowed_root=PROJECT_ROOT):
        config, config_path = load_qasper_eval_config(
            args.config,
            retrieval_only=args.retrieval_only,
        )
        config = apply_cli_overrides(config, top_k=args.top_k)
        if args.candidate_k is not None:
            config["retrieval"]["candidate_k"] = args.candidate_k
        if args.reranker_model is not None:
            config["retrieval"]["reranker"] = {
                "provider": "cross_encoder",
                "model_name": args.reranker_model,
                "revision": args.reranker_revision,
                "batch_size": args.reranker_batch_size,
                "device": args.reranker_device,
                "local_files_only": args.reranker_local_files_only,
            }
        if args.retrieval_only:
            # This component is assembled for the normal pipeline contract but
            # never called by the retrieval-only evaluation branch.
            config["generation"] = {
                "provider": "extractive",
                "max_output_tokens": 1,
            }
        config = validate_config(config)
    dataset_path = resolved_roots(config)["corpus"]
    dataset = load_qasper_dataset(dataset_path)
    slice_stats = qasper_evaluation_slice_stats(dataset["validation"])
    raw_questions, articles_by_id = qasper_eval_inputs(
        dataset,
        max_questions=args.max_questions,
    )
    questions = validate_questions(raw_questions)
    if not questions:
        raise RuntimeError(f"QASPER evaluation slice is empty: {QASPER_EVALUATION_SLICE}")

    with temporary_openai_api_key(args.api_key_file, allowed_root=PROJECT_ROOT):
        pipeline = NaiveRAGPipeline(config)
    manifest = pipeline.manifest
    if manifest["index"]["backend"] != "faiss":
        raise RuntimeError("QASPER index build did not use FAISS")
    questions_sha = json_sha256(questions)
    evaluation_source_sha = source_group_sha256(PROJECT_ROOT, "evaluation")
    evaluation_value = evaluation_spec(
        questions_sha,
        evaluation_source_sha,
        metrics_version=(
            RETRIEVAL_METRICS_VERSION if args.retrieval_only else EVALUATION_PROTOCOL
        ),
    )
    run_id = args.run_id or (
        (
            "qasper_retrieval_text_extractive_single_evidence_"
            if args.retrieval_only
            else "qasper_validation_text_extractive_single_evidence_"
        )
        + ("full_" if args.max_questions is None else f"n{args.max_questions}_")
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    run_dir = resolved_roots(config)["outputs_root"] / run_id
    evidence_units_by_paper = {
        paper_id: [unit.evidence for unit in qasper_unit_records(article)]
        for paper_id, article in articles_by_id.items()
    }

    def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
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
                # Score the original dense prefix from the exact same Top-50
                # call, so quality gains/losses are paired without a second
                # retrieval introducing any execution drift.
                rerank_data["dense_baseline_metrics"] = score_qasper_open_corpus_retrieval(
                    candidate_rows[: retrieval.top_k],
                    question["references"],
                    question["paper_id"],
                    evidence_units_by_paper,
                )
                rerank_data["candidate_metrics"] = score_qasper_open_corpus_retrieval(
                    candidate_rows,
                    question["references"],
                    question["paper_id"],
                    evidence_units_by_paper,
                )
                row["retrieval"]["rerank"] = rerank_data
            row["metrics"] = score_qasper_open_corpus_retrieval(
                retrieved_rows,
                question["references"],
                question["paper_id"],
                evidence_units_by_paper,
            )
        else:
            row = pipeline.query(question["question"], question_id=question["question_id"])
            generation = row["generation"]
            if generation["provider"] != "openai":
                raise RuntimeError("A successful QASPER row did not use the OpenAI provider")
            row["metrics"] = score_qasper_open_corpus(
                generation["answer"],
                row["retrieval"]["results"],
                question["references"],
                question["paper_id"],
                evidence_units_by_paper,
            )
        row["evaluation"] = _evaluation_fields(question)
        return row

    def error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "metrics": _failed_score(retrieval_only=args.retrieval_only),
            "evaluation": _evaluation_fields(question),
        }

    metadata = {
        "run_id": run_id,
        "command": "run_qasper_eval",
        "status": "running",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "questions_path": None,
        "questions_source": f"{dataset_path}#validation/qas?slice={QASPER_EVALUATION_SLICE}",
        "questions_sha256": questions_sha,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha,
        **pipeline.runtime_metadata,
        "execution_mode": "retrieval_only" if args.retrieval_only else "rag",
        "requested_top_k": args.top_k,
        "candidate_k": config["retrieval"]["candidate_k"],
        "effective_top_k": config["retrieval"]["final_k"],
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "question_selection": dict(slice_stats),
        "question_split": "validation",
        "corpus_splits": list(QASPER_SPLITS),
    }
    try:
        rows, summary = run_evaluation(
            questions=questions,
            run_dir=run_dir,
            metadata=metadata,
            resume=args.resume,
            evaluate_question=evaluate_question,
            summarize_rows=lambda rows: _qasper_summary(
                rows,
                manifest=manifest,
                effective_top_k=config["retrieval"]["final_k"],
                candidate_k=config["retrieval"]["candidate_k"],
                slice_stats=slice_stats,
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
            f"QASPER evaluation saved {len(failed_rows)} failed row(s); use --resume "
            "after fixing the cause"
        )
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
