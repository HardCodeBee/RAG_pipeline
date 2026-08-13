"""Run retrieval-only experiments on one prepared BEIR dataset unit."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output, positive_int, safe_run_id
from src.config import load_config, resolve_cli_path, validate_config
from src.evaluation_runner import run_evaluation, validate_resume_compatibility
from src.evaluators.beir_evaluation import (
    EVALUATION_PROTOCOL,
    METRICS_VERSION,
    FirstStageCandidateBatch,
    FirstStageCandidateStore,
    compute_streaming_first_stage,
    first_stage_identity,
    load_beir_question_split,
    score_beir_query,
    summarize_beir_evaluation,
)
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    source_files_sha256,
)
from src.records import SearchHit


def _evaluation_source_sha256() -> str:
    return source_files_sha256(PROJECT_ROOT, (
        "scripts/run_beir_eval.py",
        "src/evaluators/beir_evaluation.py",
        "src/evaluators/beir_metrics.py",
        "src/evaluation_runner.py",
        "src/loaders/beir_loader.py",
        "src/persistence/run_output_writer.py",
    ))


def _compact_hit(hit: SearchHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "chunk_id": hit.chunk.chunk_id,
        "doc_id": hit.chunk.doc_id,
        "vector_id": hit.chunk.vector_id,
        "score": float(hit.score),
    }


def _metrics_for_rankings(
    final_hits: Sequence[SearchHit],
    candidate_hits: Sequence[SearchHit],
    qrels: Mapping[str, Any],
    *,
    final_k: int,
    candidate_k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_doc_ids = [hit.chunk.doc_id for hit in candidate_hits]
    return (
        score_beir_query(
            [hit.chunk.doc_id for hit in final_hits],
            qrels,
            k=final_k,
        ),
        score_beir_query(candidate_doc_ids, qrels, k=final_k),
        score_beir_query(candidate_doc_ids, qrels, k=candidate_k),
    )


def _streaming_question_row(
    *,
    question: Mapping[str, Any],
    position: int,
    pipeline: NaiveRAGPipeline,
    batch: FirstStageCandidateBatch,
    candidate_k: int,
    final_k: int,
    save_text: bool,
    split: str,
    dataset: str,
    unit: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    mapping_started = time.perf_counter()
    vector_ids = [int(value) for value in batch.vector_ids[position]]
    chunks = pipeline.chunk_store.get_many(vector_ids)
    candidate_hits = tuple(
        SearchHit(rank=rank, chunk=chunk, score=float(score))
        for rank, (chunk, score) in enumerate(
            zip(chunks, batch.scores[position]), start=1
        )
    )
    mapping_ms = (time.perf_counter() - mapping_started) * 1000
    reranked = pipeline.reranker.rerank(
        str(question["question"]),
        candidate_hits,
        final_k=final_k,
    )
    final_hits = reranked.results
    metrics, first_stage_metrics, candidate_pool_metrics = _metrics_for_rankings(
        final_hits,
        candidate_hits,
        question["qrels"],
        final_k=final_k,
        candidate_k=candidate_k,
    )
    retrieval: dict[str, Any] = {
        "method": "dense",
        "candidate_k": candidate_k,
        "top_k": final_k,
        "results": [hit.to_dict(include_text=save_text) for hit in final_hits],
        "first_stage_results": [_compact_hit(hit) for hit in candidate_hits],
        "timings_ms": {
            "batch_query_embedding_ms": float(
                batch.timings_ms["query_embedding_ms"]
            ),
            "batch_index_search_ms": float(batch.timings_ms["index_search_ms"]),
            "batch_total_ms": float(batch.timings_ms["total_ms"]),
            "chunk_mapping_ms": mapping_ms,
            "rerank_ms": reranked.timing_ms,
            "post_batch_total_ms": (time.perf_counter() - started) * 1000,
        },
    }
    if reranked.trace is not None:
        retrieval["rerank"] = reranked.trace.to_dict()
    return {
        "status": "success",
        "question_id": question["question_id"],
        "question": question["question"],
        "identity": {
            "build_id": pipeline.runtime_metadata["build_id"],
            "run_spec_sha256": pipeline.runtime_metadata["run_spec_sha256"],
        },
        "qrels": dict(question["qrels"]),
        "retrieval": retrieval,
        "metrics": metrics,
        "first_stage_metrics": first_stage_metrics,
        "candidate_pool_metrics": candidate_pool_metrics,
        "total_latency_ms": (time.perf_counter() - started) * 1000,
        "evaluation": {
            "protocol": EVALUATION_PROTOCOL,
            "dataset": dataset,
            "unit": unit,
            "split": split,
        },
    }


def _pipeline_question_row(
    *,
    question: Mapping[str, Any],
    pipeline: NaiveRAGPipeline,
    candidate_k: int,
    final_k: int,
    save_text: bool,
    split: str,
    dataset: str,
    unit: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    retrieval_trace, rerank_trace = pipeline.retrieve_with_details(
        str(question["question"])
    )
    candidate_hits = (
        rerank_trace.candidates
        if rerank_trace is not None
        else retrieval_trace.results
    )
    if rerank_trace is None and candidate_k != final_k:
        raise RuntimeError(
            "The per-query pipeline did not expose its complete first-stage "
            "candidate pool"
        )
    final_hits = retrieval_trace.results
    metrics, first_stage_metrics, candidate_pool_metrics = _metrics_for_rankings(
        final_hits,
        candidate_hits,
        question["qrels"],
        final_k=final_k,
        candidate_k=candidate_k,
    )
    retrieval: dict[str, Any] = {
        "method": pipeline.config["retrieval"]["method"],
        "candidate_k": candidate_k,
        "top_k": final_k,
        "results": [hit.to_dict(include_text=save_text) for hit in final_hits],
        "first_stage_results": [_compact_hit(hit) for hit in candidate_hits],
        "timings_ms": dict(retrieval_trace.timings_ms),
    }
    if rerank_trace is not None:
        retrieval["rerank"] = rerank_trace.to_dict()
    return {
        "status": "success",
        "question_id": question["question_id"],
        "question": question["question"],
        "identity": {
            "build_id": pipeline.runtime_metadata["build_id"],
            "run_spec_sha256": pipeline.runtime_metadata["run_spec_sha256"],
        },
        "qrels": dict(question["qrels"]),
        "retrieval": retrieval,
        "metrics": metrics,
        "first_stage_metrics": first_stage_metrics,
        "candidate_pool_metrics": candidate_pool_metrics,
        "total_latency_ms": (time.perf_counter() - started) * 1000,
        "evaluation": {
            "protocol": EVALUATION_PROTOCOL,
            "dataset": dataset,
            "unit": unit,
            "split": split,
        },
    }


def _default_run_id(
    *,
    unit: str,
    method: str,
    split: str,
    max_questions: int | None,
) -> str:
    safe_unit = re.sub(r"[^A-Za-z0-9._-]+", "_", unit).strip("_")
    size = "full" if max_questions is None else f"n{max_questions}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"beir_{safe_unit}_{method}_{split}_{size}_{timestamp}"


def main(argv: Sequence[str] | None = None) -> Path:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run retrieval-only BM25/dense/reranking experiments on BEIR."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-questions", type=positive_int, default=None)
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    # This command is retrieval-only.  Keep generator construction local and
    # deterministic even when a shared config normally uses a remote model.
    config["generation"] = {"provider": "extractive", "max_output_tokens": 1}
    config = validate_config(config)
    candidate_k = config["retrieval"]["candidate_k"]
    final_k = config["retrieval"]["final_k"]
    if candidate_k <= 0 or final_k <= 0:
        raise ValueError("BEIR retrieval evaluation requires positive candidate_k/final_k")

    roots = resolved_roots(config)
    verified_questions = load_beir_question_split(
        roots["corpus"],
        split=args.split,
        max_questions=args.max_questions,
        expected_dataset=config["loader"]["expected_dataset"],
    )
    questions = list(verified_questions.questions)
    pipeline = NaiveRAGPipeline(config)
    method = config["retrieval"]["method"]
    index_type = config["index"]["type"]
    streaming_dense = method == "dense" and index_type == "streaming_flat_ip"

    questions_sha = json_sha256(questions)
    evaluation_source_sha = _evaluation_source_sha256()
    evaluation_value = evaluation_spec(
        questions_sha,
        evaluation_source_sha,
        metrics_version=METRICS_VERSION,
    )
    run_id = args.run_id or _default_run_id(
        unit=verified_questions.unit,
        method=method,
        split=args.split,
        max_questions=args.max_questions,
    )
    run_dir = roots["outputs_root"] / run_id
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"BEIR run directory already exists: {run_dir}")

    batch: FirstStageCandidateBatch | None = None
    candidate_store: FirstStageCandidateStore | None = None
    cache_identity: dict[str, Any] | None = None
    if streaming_dense:
        query_batch_size = config["retrieval"]["query_batch_size"]
        cache_identity = first_stage_identity(
            questions_sha256=questions_sha,
            question_ids=[question["question_id"] for question in questions],
            build_id=pipeline.runtime_metadata["build_id"],
            run_spec_sha256=pipeline.runtime_metadata["run_spec_sha256"],
            retrieval_method=method,
            index_type=index_type,
            candidate_k=candidate_k,
            query_batch_size=query_batch_size,
        )
        candidate_store = FirstStageCandidateStore(
            run_dir,
            identity=cache_identity,
            question_ids=[question["question_id"] for question in questions],
            corpus_count=len(pipeline.chunk_store),
        )

    metadata = {
        "run_id": run_id,
        "command": "run_beir_eval",
        "status": "running",
        "execution_mode": "retrieval_only",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "dataset": verified_questions.dataset,
        "unit": verified_questions.unit,
        "dataset_manifest_path": str(verified_questions.dataset_manifest_path),
        "dataset_manifest_sha256": verified_questions.dataset_manifest_sha256,
        "questions_path": str(verified_questions.queries_path),
        "questions_file_sha256": verified_questions.queries_file_sha256,
        "qrels_path": str(verified_questions.qrels_path),
        "qrels_file_sha256": verified_questions.qrels_file_sha256,
        "questions_sha256": questions_sha,
        "question_split": args.split,
        "question_selection_order": (
            "queries_jsonl_order_filtered_by_qrels_split_then_prefix_v1"
        ),
        "max_questions": args.max_questions,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha,
        **pipeline.runtime_metadata,
        "retrieval_method": method,
        "reranker_provider": config["retrieval"]["reranker"]["provider"],
        "candidate_k": candidate_k,
        "effective_top_k": final_k,
        "first_stage_mode": (
            "batch_streaming_flat_ip" if streaming_dense else f"per_query_{method}"
        ),
        "first_stage_identity": cache_identity,
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
    }

    try:
        if streaming_dense:
            if candidate_store is None:
                raise RuntimeError("Streaming candidate store was not initialized")
            if args.resume:
                metadata_path = run_dir / "metadata.json"
                if not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot resume a missing BEIR run: {run_dir}"
                    )
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                validate_resume_compatibility(previous, metadata)
                batch = candidate_store.load()
            if batch is None:
                print(
                    f"Batch-encoding {len(questions)} queries and scanning the corpus once...",
                    flush=True,
                )
                batch = compute_streaming_first_stage(
                    pipeline,
                    questions,
                    candidate_k=candidate_k,
                    query_batch_size=config["retrieval"]["query_batch_size"],
                )
                if args.resume:
                    batch = candidate_store.write(batch)
            metadata["first_stage_batch_timings_ms"] = dict(batch.timings_ms)
            metadata["first_stage_candidates_reused"] = batch.reused_cache

        position_by_id = {
            question["question_id"]: position
            for position, question in enumerate(questions)
        }
        cache_committed = batch is not None and batch.reused_cache

        def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal batch, cache_committed
            if streaming_dense:
                if batch is None or candidate_store is None:
                    raise RuntimeError("Streaming candidate batch is unavailable")
                if not cache_committed:
                    batch = candidate_store.write(batch)
                    cache_committed = True
                    metadata["first_stage_candidates_reused"] = False
                return _streaming_question_row(
                    question=question,
                    position=position_by_id[question["question_id"]],
                    pipeline=pipeline,
                    batch=batch,
                    candidate_k=candidate_k,
                    final_k=final_k,
                    save_text=config["logging"]["save_retrieved_text"],
                    split=args.split,
                    dataset=verified_questions.dataset,
                    unit=verified_questions.unit,
                )
            return _pipeline_question_row(
                question=question,
                pipeline=pipeline,
                candidate_k=candidate_k,
                final_k=final_k,
                save_text=config["logging"]["save_retrieved_text"],
                split=args.split,
                dataset=verified_questions.dataset,
                unit=verified_questions.unit,
            )

        def error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "qrels": dict(question["qrels"]),
                "metrics": {},
                "first_stage_metrics": {},
                "candidate_pool_metrics": {},
                "evaluation": {
                    "protocol": EVALUATION_PROTOCOL,
                    "dataset": verified_questions.dataset,
                    "unit": verified_questions.unit,
                    "split": args.split,
                },
            }

        rows, summary = run_evaluation(
            questions=questions,
            run_dir=run_dir,
            metadata=metadata,
            resume=args.resume,
            evaluate_question=evaluate_question,
            summarize_rows=lambda values: summarize_beir_evaluation(
                values,
                dataset=verified_questions.dataset,
                unit=verified_questions.unit,
                split=args.split,
                final_k=final_k,
                candidate_k=candidate_k,
                retrieval_method=method,
                reranker_provider=config["retrieval"]["reranker"]["provider"],
            ),
            error_fields=error_fields,
            process_started=process_started,
        )
    finally:
        pipeline.close()

    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise RuntimeError(
            f"BEIR evaluation saved {len(failed)} failed row(s); "
            "fix the cause and rerun with --resume"
        )
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
