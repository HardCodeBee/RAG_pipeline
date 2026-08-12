"""Recompute a completed four-condition BEIR suite at a deeper cutoff.

This command is deliberately offline: it validates and reuses the source
suite's shared first-stage candidate arrays and BGE logits.  It never opens a
retrieval index, an embedding model, a sparse retriever, or a reranker model.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output, positive_int, safe_run_id
from scripts.run_beir_suite import (
    FIRST_STAGE_SEARCH_K,
    METADATA_FLUSH_INTERVAL,
    _suite_summary_csv_rows,
    _summarize_suite_condition,
    _unit_key,
    _write_csv_atomic,
)
from src.evaluation_runner import run_evaluation
from src.evaluators.beir_evaluation import load_beir_question_split
from src.evaluators.beir_metrics import METRICS_VERSION, score_beir_query
from src.evaluators.beir_suite import (
    SUITE_EVALUATION_PROTOCOL,
    SharedCandidateBatch,
    SharedCandidateStore,
    SharedRerankScoreBatch,
    SharedRerankScoreStore,
    aggregate_suite_summaries,
    beir_row_from_shared_candidates,
)
from src.persistence.artifact_io import iter_jsonl, read_json_object
from src.persistence.artifact_validation import verify_artifact_descriptor
from src.persistence.run_output_writer import write_metadata_json
from src.provenance import evaluation_spec, json_sha256, sha256_file
from src.rerankers.noop import NoOpReranker
from src.retrievers.chunk_store import JsonlOffsetChunkStore


REANALYSIS_SCHEMA_VERSION = 1
SOURCE_CONDITIONS = {
    ("bm25", False): "bm25_top5",
    ("dense", False): "dense_top5",
    ("bm25", True): "bm25_top50_bge_top5",
    ("dense", True): "dense_top50_bge_top5",
}
SOURCE_FILES = (
    Path(__file__),
    PROJECT_ROOT / "src" / "evaluators" / "beir_metrics.py",
    PROJECT_ROOT / "src" / "evaluators" / "beir_evaluation.py",
    PROJECT_ROOT / "src" / "evaluators" / "beir_suite.py",
    PROJECT_ROOT / "src" / "evaluation_runner.py",
)


def _condition_name(method: str, *, rerank: bool, top_k: int) -> str:
    if method not in {"bm25", "dense"}:
        raise ValueError("method must be bm25 or dense")
    return (
        f"{method}_top50_bge_top{top_k}"
        if rerank
        else f"{method}_top{top_k}"
    )


def _source_snapshot() -> dict[str, Any]:
    files = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in SOURCE_FILES
    }
    return {
        "schema_version": REANALYSIS_SCHEMA_VERSION,
        "files": files,
        "sha256": json_sha256(files),
    }


def _validate_artifact(
    directory: Path,
    descriptor: Any,
    *,
    expected_file: str,
    label: str,
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor is missing")
    if descriptor.get("file") != expected_file:
        raise ValueError(f"{label} descriptor names the wrong file")
    path = directory / expected_file
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("size_bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise ValueError(f"{label} artifact is missing or corrupted")
    return path


def _load_source_rows(
    run_dir: Path,
    metadata: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if metadata.get("status") != "completed":
        raise ValueError(f"Source condition is not completed: {run_dir}")
    path = _validate_artifact(
        run_dir,
        metadata.get("results_artifact"),
        expected_file="results.jsonl",
        label="Source results",
    )
    _validate_artifact(
        run_dir,
        metadata.get("summary_artifact"),
        expected_file="summary.csv",
        label="Source summary",
    )
    rows = tuple(iter_jsonl(path))
    if len(rows) != len(questions):
        raise ValueError("Source result count differs from the verified question set")
    for question, row in zip(questions, rows):
        if (
            row.get("status") != "success"
            or row.get("question_id") != question["question_id"]
            or row.get("question") != question["question"]
            or row.get("qrels") != question["qrels"]
        ):
            raise ValueError(
                "Source result rows do not align with the verified question set: "
                f"{question['question_id']}"
            )
    return rows


def _open_chunk_store(source_run: Mapping[str, Any]) -> JsonlOffsetChunkStore:
    build_id = source_run.get("build_id")
    build_dir_value = source_run.get("build_dir")
    if not isinstance(build_id, str) or not isinstance(build_dir_value, str):
        raise ValueError("Source run does not identify its immutable build")
    build_dir = Path(build_dir_value).resolve()
    manifest = read_json_object(build_dir / "manifest.json", label="Build manifest")
    if manifest.get("status") != "complete" or manifest.get("build_id") != build_id:
        raise ValueError("Source build identity or completion status is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Source build has no artifact mapping")
    chunks = artifacts.get("chunks")
    offsets = artifacts.get("chunk_offsets")
    if not isinstance(chunks, Mapping) or not isinstance(offsets, Mapping):
        raise ValueError("Offline BEIR reanalysis requires chunks plus offset artifacts")
    rows = chunks.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("Source chunk row count is invalid")
    chunk_file = verify_artifact_descriptor(
        build_dir,
        chunks,
        label="Source build chunks",
        expected_rows=rows,
    ).path
    offset_file = verify_artifact_descriptor(
        build_dir,
        offsets,
        label="Source build chunk offsets",
        expected_rows=rows,
    ).path
    return JsonlOffsetChunkStore(chunk_file, offset_file, expected_rows=rows)


def _load_candidate_cache(
    source_unit_dir: Path,
    *,
    method: str,
    questions: Sequence[Mapping[str, Any]],
    corpus_count: int,
) -> tuple[SharedCandidateBatch, dict[str, Any], dict[str, Any]]:
    directory = source_unit_dir / "candidates" / method
    manifest = read_json_object(
        directory / "shared_first_stage_candidates.manifest.json",
        label="Shared first-stage candidate manifest",
    )
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Candidate cache identity is missing")
    store = SharedCandidateStore(
        directory,
        identity=identity,
        question_ids=[str(question["question_id"]) for question in questions],
        candidate_k=manifest.get("candidate_k"),
        corpus_count=corpus_count,
        require_full_width=manifest.get("require_full_width"),
    )
    batch = store.load()
    if batch is None:
        raise FileNotFoundError(f"Candidate cache is missing: {directory}")
    return batch, dict(identity), store.descriptor(unit_directory=source_unit_dir)


def _load_rerank_cache(
    source_unit_dir: Path,
    *,
    method: str,
    candidates: SharedCandidateBatch,
) -> tuple[SharedRerankScoreBatch, dict[str, Any], dict[str, Any]]:
    directory = source_unit_dir / "candidates" / method / "bge"
    manifest = read_json_object(
        directory / "shared_bge_scores.manifest.json",
        label="Shared BGE score manifest",
    )
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("BGE score cache identity is missing")
    store = SharedRerankScoreStore(
        directory,
        identity=identity,
        candidates=candidates,
    )
    batch = store.load()
    if batch is None:
        raise FileNotFoundError(f"BGE score cache is missing: {directory}")
    return batch, dict(identity), store.descriptor(unit_directory=source_unit_dir)


def _validate_source_cache_reference(
    source_run: Mapping[str, Any],
    *,
    candidate_identity: Mapping[str, Any],
    candidate_descriptor: Mapping[str, Any],
    rerank_identity: Mapping[str, Any] | None,
    rerank_descriptor: Mapping[str, Any] | None,
) -> None:
    if (
        source_run.get("shared_first_stage_identity") != dict(candidate_identity)
        or source_run.get("shared_first_stage_cache") != dict(candidate_descriptor)
        or source_run.get("shared_first_stage_cache_ref_sha256")
        != json_sha256(candidate_descriptor)
    ):
        raise ValueError("Source run does not reference the validated candidate cache")
    if rerank_identity is None:
        if (
            source_run.get("shared_bge_score_identity") is not None
            or source_run.get("shared_bge_score_cache") is not None
            or source_run.get("shared_bge_score_cache_ref_sha256") is not None
        ):
            raise ValueError("Baseline source run unexpectedly references BGE scores")
        return
    if rerank_descriptor is None:
        raise ValueError("Reranked source run requires a BGE cache descriptor")
    if (
        source_run.get("shared_bge_score_identity") != dict(rerank_identity)
        or source_run.get("shared_bge_score_cache") != dict(rerank_descriptor)
        or source_run.get("shared_bge_score_cache_ref_sha256")
        != json_sha256(rerank_descriptor)
    ):
        raise ValueError("Source run does not reference the validated BGE score cache")


def _derived_config(
    source_run: Mapping[str, Any],
    *,
    top_k: int,
    rerank: bool,
    physical_candidate_k: int,
) -> dict[str, Any]:
    config = copy.deepcopy(source_run.get("effective_config"))
    if not isinstance(config, dict) or not isinstance(config.get("retrieval"), dict):
        raise ValueError("Source run has no recorded effective retrieval configuration")
    retrieval = config["retrieval"]
    retrieval["candidate_k"] = physical_candidate_k if rerank else top_k
    retrieval["final_k"] = top_k
    retrieval["top_k"] = top_k
    if not rerank:
        retrieval["reranker"] = {"provider": "none"}
    return config


def _derived_run_spec(
    source_run: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    source_suite_id: str,
    source_condition: str,
    candidate_cache_ref_sha256: str,
    rerank_cache_ref_sha256: str | None,
    reanalysis_source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": REANALYSIS_SCHEMA_VERSION,
        "kind": "beir_offline_topk_reanalysis",
        "source_suite_id": source_suite_id,
        "source_condition": source_condition,
        "source_run_id": source_run.get("run_id"),
        "source_run_spec_sha256": source_run.get("run_spec_sha256"),
        "build_id": source_run.get("build_id"),
        "retrieval": copy.deepcopy(config["retrieval"]),
        "shared_first_stage_cache_ref_sha256": candidate_cache_ref_sha256,
        "shared_bge_score_cache_ref_sha256": rerank_cache_ref_sha256,
        "reanalysis_source_sha256": reanalysis_source_sha256,
    }


def _validate_topk_replay(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    source_top_k: int,
    candidate_cache_ref_sha256: str,
    rerank_cache_ref_sha256: str | None,
) -> None:
    if (
        row.get("question_id") != source_row.get("question_id")
        or row.get("question") != source_row.get("question")
        or row.get("qrels") != source_row.get("qrels")
    ):
        raise ValueError("Reanalysis row differs from its source question")
    retrieval = row.get("retrieval")
    source_retrieval = source_row.get("retrieval")
    if not isinstance(retrieval, Mapping) or not isinstance(source_retrieval, Mapping):
        raise ValueError("Source or reanalysis retrieval payload is missing")
    results = retrieval.get("results")
    source_results = source_retrieval.get("results")
    if not isinstance(results, list) or not isinstance(source_results, list):
        raise ValueError("Source or reanalysis ranked results are missing")
    if results[:source_top_k] != source_results:
        raise ValueError(
            "Cached reanalysis does not exactly replay the source ranked prefix: "
            f"{row.get('question_id')}"
        )
    replayed = score_beir_query(
        [str(result["doc_id"]) for result in results],
        row["qrels"],
        k=source_top_k,
    )
    if replayed != source_row.get("metrics"):
        raise ValueError(
            "Cached reanalysis does not exactly replay the source metrics: "
            f"{row.get('question_id')}"
        )
    shared = retrieval.get("shared_first_stage")
    if (
        not isinstance(shared, Mapping)
        or shared.get("cache_ref_sha256") != candidate_cache_ref_sha256
    ):
        raise ValueError("Reanalysis row is not bound to the validated candidate cache")
    rerank = retrieval.get("rerank")
    source_rerank = source_retrieval.get("rerank")
    if rerank_cache_ref_sha256 is None:
        if rerank is not None:
            raise ValueError("Baseline reanalysis unexpectedly contains rerank metadata")
    elif rerank is None:
        shared_candidate_count = shared.get("condition_candidate_count")
        if source_rerank is not None or shared_candidate_count != 0 or results:
            raise ValueError("Non-empty rerank row is missing its validated BGE cache")
    elif (
        not isinstance(rerank, Mapping)
        or rerank.get("cache_ref_sha256") != rerank_cache_ref_sha256
    ):
        raise ValueError("Reanalysis row is not bound to the validated BGE cache")


def _run_condition(
    *,
    destination_suite_id: str,
    destination_unit_dir: Path,
    source_suite_id: str,
    source_suite_dir: Path,
    source_unit_dir: Path,
    source_run_dir: Path,
    source_run: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    verified_questions: Any,
    questions: Sequence[Mapping[str, Any]],
    questions_sha256: str,
    chunk_store: JsonlOffsetChunkStore,
    batch: SharedCandidateBatch,
    candidate_identity: Mapping[str, Any],
    candidate_descriptor: Mapping[str, Any],
    rerank_score_batch: SharedRerankScoreBatch | None,
    rerank_identity: Mapping[str, Any] | None,
    rerank_descriptor: Mapping[str, Any] | None,
    method: str,
    rerank: bool,
    top_k: int,
    source_top_k: int,
    physical_candidate_k: int,
    reanalysis_snapshot: Mapping[str, Any],
    resume_suite: bool,
) -> tuple[str, dict[str, Any]]:
    condition = _condition_name(method, rerank=rerank, top_k=top_k)
    source_condition = SOURCE_CONDITIONS[(method, rerank)]
    run_dir = destination_unit_dir / "runs" / condition
    resume = resume_suite and run_dir.exists()
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Condition run already exists: {run_dir}")

    candidate_cache_ref = json_sha256(candidate_descriptor)
    rerank_cache_ref = (
        json_sha256(rerank_descriptor) if rerank_descriptor is not None else None
    )
    _validate_source_cache_reference(
        source_run,
        candidate_identity=candidate_identity,
        candidate_descriptor=candidate_descriptor,
        rerank_identity=rerank_identity,
        rerank_descriptor=rerank_descriptor,
    )
    config = _derived_config(
        source_run,
        top_k=top_k,
        rerank=rerank,
        physical_candidate_k=physical_candidate_k,
    )
    run_spec = _derived_run_spec(
        source_run,
        config=config,
        source_suite_id=source_suite_id,
        source_condition=source_condition,
        candidate_cache_ref_sha256=candidate_cache_ref,
        rerank_cache_ref_sha256=rerank_cache_ref,
        reanalysis_source_sha256=str(reanalysis_snapshot["sha256"]),
    )
    run_spec_sha256 = json_sha256(run_spec)
    evaluation_value = evaluation_spec(
        questions_sha256,
        str(reanalysis_snapshot["sha256"]),
        metrics_version=METRICS_VERSION,
    )
    condition_candidate_k = physical_candidate_k if rerank else top_k
    condition_run_id = (
        f"{destination_suite_id}__{_unit_key(verified_questions.unit)}__{condition}"
    )
    metadata = {
        "run_id": condition_run_id,
        "suite_id": destination_suite_id,
        "command": "recompute_beir_suite_topk",
        "condition": condition,
        "status": "running",
        "execution_mode": "offline_identity_bound_cache_reanalysis",
        "effective_config": config,
        "dataset": verified_questions.dataset,
        "unit": verified_questions.unit,
        "raw_beir_ids": True,
        "dataset_manifest_path": str(verified_questions.dataset_manifest_path),
        "dataset_manifest_sha256": verified_questions.dataset_manifest_sha256,
        "questions_path": str(verified_questions.queries_path),
        "questions_file_sha256": verified_questions.queries_file_sha256,
        "qrels_path": str(verified_questions.qrels_path),
        "qrels_file_sha256": verified_questions.qrels_file_sha256,
        "questions_sha256": questions_sha256,
        "question_split": verified_questions.split,
        "question_selection_order": (
            "queries_jsonl_order_filtered_by_qrels_split_then_prefix_v1"
        ),
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": reanalysis_snapshot["sha256"],
        "reanalysis_source_snapshot": dict(reanalysis_snapshot),
        "build_id": source_run["build_id"],
        "build_dir": source_run["build_dir"],
        "build_spec_sha256": source_run.get("build_spec_sha256"),
        "build_source_sha256": source_run.get("build_source_sha256"),
        "run_source_sha256": source_run.get("run_source_sha256"),
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "retrieval_method": method,
        "reranker_provider": (
            config["retrieval"]["reranker"].get("provider", "none")
        ),
        "candidate_k": condition_candidate_k,
        "effective_top_k": top_k,
        "shared_first_stage_candidate_k": physical_candidate_k,
        "shared_first_stage_requested_k": FIRST_STAGE_SEARCH_K,
        "ignore_identical_ids": True,
        "identical_id_policy": "leakage_safe_pre_candidate_filtering",
        "shared_first_stage_identity": dict(candidate_identity),
        "shared_first_stage_identity_sha256": json_sha256(candidate_identity),
        "shared_first_stage_cache": dict(candidate_descriptor),
        "shared_first_stage_cache_ref_sha256": candidate_cache_ref,
        "shared_first_stage_cache_source_unit_dir": str(source_unit_dir),
        "shared_first_stage_candidates_reused": True,
        "shared_first_stage_timings_ms": dict(batch.timings_ms),
        "shared_bge_scores_reused": rerank_score_batch is not None,
        "shared_bge_score_identity": (
            dict(rerank_identity) if rerank_identity is not None else None
        ),
        "shared_bge_score_identity_sha256": (
            json_sha256(rerank_identity) if rerank_identity is not None else None
        ),
        "shared_bge_score_cache": (
            dict(rerank_descriptor) if rerank_descriptor is not None else None
        ),
        "shared_bge_score_cache_ref_sha256": rerank_cache_ref,
        "source_suite_id": source_suite_id,
        "source_suite_dir": str(source_suite_dir),
        "source_condition": source_condition,
        "source_condition_run_dir": str(source_run_dir),
        "source_condition_run_id": source_run.get("run_id"),
        "source_run_spec_sha256": source_run.get("run_spec_sha256"),
        "source_top_k": source_top_k,
        "source_prefix_replay_validation": "exact_per_question",
        "latency_scope": (
            "source_first_stage_and_bge_timings_plus_offline_chunk_mapping; "
            "not_a_fresh_topk_latency_measurement"
        ),
        "retrieval_or_reranking_repeated": False,
        "resume": resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
        "metadata_flush_interval": METADATA_FLUSH_INTERVAL,
    }
    source_rows_by_id = {str(row["question_id"]): row for row in source_rows}
    position_by_id = {
        str(question["question_id"]): position
        for position, question in enumerate(questions)
    }
    no_op = NoOpReranker()

    def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
        question_id = str(question["question_id"])
        position = position_by_id[question_id]
        cached_scores: Sequence[float] | None = None
        cached_timing: float | None = None
        if rerank_score_batch is not None:
            cached_scores, cached_timing = rerank_score_batch.row(position, batch)
        row = beir_row_from_shared_candidates(
            question=question,
            position=position,
            batch=batch,
            chunk_store=chunk_store,
            reranker=no_op,
            method=method,
            physical_candidate_k=physical_candidate_k,
            condition_candidate_k=condition_candidate_k,
            final_k=top_k,
            build_id=str(source_run["build_id"]),
            run_spec_sha256=run_spec_sha256,
            save_text=bool(config["logging"]["save_retrieved_text"]),
            split=verified_questions.split,
            dataset=verified_questions.dataset,
            unit=verified_questions.unit,
            shared_candidate_cache_ref_sha256=candidate_cache_ref,
            rerank_scores=cached_scores,
            rerank_timing_ms=cached_timing,
            shared_rerank_cache_ref_sha256=rerank_cache_ref,
        )
        _validate_topk_replay(
            row,
            source_rows_by_id[question_id],
            source_top_k=source_top_k,
            candidate_cache_ref_sha256=candidate_cache_ref,
            rerank_cache_ref_sha256=rerank_cache_ref,
        )
        return row

    def error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "qrels": dict(question["qrels"]),
            "metrics": {},
            "first_stage_metrics": {},
            "candidate_pool_metrics": {},
            "evaluation": {
                "protocol": SUITE_EVALUATION_PROTOCOL,
                "dataset": verified_questions.dataset,
                "unit": verified_questions.unit,
                "split": verified_questions.split,
                "raw_beir_ids": True,
                "shared_first_stage_candidates": True,
                "identical_id_policy": "leakage_safe_pre_candidate_filtering",
            },
        }

    rows, summary = run_evaluation(
        questions=questions,
        run_dir=run_dir,
        metadata=metadata,
        resume=resume,
        evaluate_question=evaluate_question,
        summarize_rows=lambda values: _summarize_suite_condition(
            values,
            dataset=verified_questions.dataset,
            unit=verified_questions.unit,
            split=verified_questions.split,
            final_k=top_k,
            candidate_k=condition_candidate_k,
            retrieval_method=method,
            reranker_provider=metadata["reranker_provider"],
        ),
        error_fields=error_fields,
        metadata_flush_interval=METADATA_FLUSH_INTERVAL,
    )
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise RuntimeError(
            f"{verified_questions.unit}/{condition} saved {len(failed)} failed rows; "
            "fix the cause and resume the reanalysis"
        )
    return condition, summary


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=(
            "Offline BEIR Top-k reanalysis from a completed four-condition "
            "suite's shared Top-50 candidates and BGE logits."
        )
    )
    parser.add_argument("--source-suite-dir", required=True)
    parser.add_argument("--top-k", type=positive_int, required=True)
    parser.add_argument("--run-id", type=safe_run_id, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    configure_utf8_output()

    source_suite_dir = Path(args.source_suite_dir)
    if not source_suite_dir.is_absolute():
        source_suite_dir = PROJECT_ROOT / source_suite_dir
    source_suite_dir = source_suite_dir.resolve()
    source_metadata_path = source_suite_dir / "metadata.json"
    source_suite = read_json_object(
        source_metadata_path,
        label="Source BEIR suite metadata",
    )
    if source_suite.get("status") != "completed":
        raise ValueError("Source BEIR suite must have status=completed")
    source_identity = source_suite.get("suite_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("Source BEIR suite has no suite identity")
    if source_suite.get("suite_identity_sha256") != json_sha256(source_identity):
        raise ValueError("Source BEIR suite identity hash is invalid")
    source_suite_id = source_suite.get("run_id")
    source_top_k = source_identity.get("final_k")
    physical_candidate_k = source_identity.get("physical_candidate_k")
    if not isinstance(source_suite_id, str) or not source_suite_id:
        raise ValueError("Source BEIR suite id is invalid")
    if (
        isinstance(source_top_k, bool)
        or not isinstance(source_top_k, int)
        or source_top_k <= 0
        or isinstance(physical_candidate_k, bool)
        or not isinstance(physical_candidate_k, int)
        or physical_candidate_k <= 0
    ):
        raise ValueError("Source suite Top-k identity is invalid")
    if not source_top_k <= args.top_k <= physical_candidate_k:
        raise ValueError(
            f"--top-k must be between source Top-{source_top_k} and the shared "
            f"candidate depth {physical_candidate_k}"
        )
    source_conditions = source_identity.get("conditions")
    if set(source_conditions or ()) != set(SOURCE_CONDITIONS.values()):
        raise ValueError("Source suite is not the expected four-condition BEIR suite")

    destination_suite_dir = source_suite_dir.parent / args.run_id
    if destination_suite_dir == source_suite_dir:
        raise ValueError("Destination run id must not overwrite the source suite")
    reanalysis_snapshot = _source_snapshot()
    unit_descriptors = source_identity.get("units")
    if not isinstance(unit_descriptors, list) or not unit_descriptors:
        raise ValueError("Source BEIR suite has no unit descriptors")
    conditions = [
        _condition_name(method, rerank=rerank, top_k=args.top_k)
        for method in ("bm25", "dense")
        for rerank in (False, True)
    ]
    source_suite_descriptor = {
        "run_id": source_suite_id,
        "directory": str(source_suite_dir),
        "metadata_size_bytes": source_metadata_path.stat().st_size,
        "metadata_sha256": sha256_file(source_metadata_path),
        "suite_identity_sha256": source_suite["suite_identity_sha256"],
    }
    suite_identity = {
        "schema_version": REANALYSIS_SCHEMA_VERSION,
        "kind": "beir_offline_topk_reanalysis",
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "reanalysis_source_snapshot": reanalysis_snapshot,
        "source_suite": source_suite_descriptor,
        "source_top_k": source_top_k,
        "physical_candidate_k": physical_candidate_k,
        "final_k": args.top_k,
        "retrieval_or_reranking_repeated": False,
        "source_prefix_replay_validation": "exact_per_question",
        "units": copy.deepcopy(unit_descriptors),
        "conditions": conditions,
    }
    metadata_path = destination_suite_dir / "metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume a missing BEIR reanalysis: {destination_suite_dir}"
            )
        previous = read_json_object(metadata_path, label="BEIR reanalysis metadata")
        if previous.get("suite_identity") != suite_identity:
            raise ValueError("Cannot resume an incompatible BEIR reanalysis")
        suite_metadata = dict(previous)
        suite_metadata.update(
            {
                "status": "running",
                "resumed_at": now,
                "completed_at": None,
                "last_error": None,
            }
        )
        write_metadata_json(metadata_path, suite_metadata)
    else:
        if destination_suite_dir.exists():
            raise FileExistsError(
                f"BEIR reanalysis directory already exists: {destination_suite_dir}"
            )
        destination_suite_dir.mkdir(parents=True)
        suite_metadata = {
            "run_id": args.run_id,
            "command": "recompute_beir_suite_topk",
            "status": "running",
            "suite_identity": suite_identity,
            "suite_identity_sha256": json_sha256(suite_identity),
            "started_at": now,
            "completed_at": None,
            "completed_conditions": [],
            "last_error": None,
            "metadata_flush_interval": METADATA_FLUSH_INTERVAL,
        }
        write_metadata_json(metadata_path, suite_metadata, overwrite=False)

    summaries: list[dict[str, Any]] = []
    completed = set(suite_metadata.get("completed_conditions", ()))
    try:
        for unit_descriptor in unit_descriptors:
            if not isinstance(unit_descriptor, Mapping):
                raise ValueError("Source suite unit descriptor is invalid")
            unit_name = unit_descriptor.get("unit")
            dataset = unit_descriptor.get("dataset")
            source_data_dir = unit_descriptor.get("directory")
            if any(
                not isinstance(value, str) or not value
                for value in (unit_name, dataset, source_data_dir)
            ):
                raise ValueError("Source suite unit identity is invalid")
            source_unit_dir = source_suite_dir / "units" / _unit_key(unit_name)
            source_unit = read_json_object(
                source_unit_dir / "unit.json",
                label="Source suite unit descriptor",
            )
            verified = load_beir_question_split(
                Path(source_data_dir),
                split=str(source_unit["split"]),
                max_questions=None,
                expected_dataset=dataset,
            )
            questions = list(verified.questions)
            questions_sha256 = json_sha256(questions)
            if (
                source_unit.get("dataset") != verified.dataset
                or source_unit.get("unit") != verified.unit
                or source_unit.get("dataset_manifest_sha256")
                != verified.dataset_manifest_sha256
                or source_unit.get("questions_sha256") != questions_sha256
                or source_unit.get("num_questions") != len(questions)
            ):
                raise ValueError(f"Source suite unit changed: {unit_name}")

            destination_unit_dir = (
                destination_suite_dir / "units" / _unit_key(verified.unit)
            )
            destination_unit_dir.mkdir(parents=True, exist_ok=True)
            destination_unit_descriptor = {
                **source_unit,
                "source_suite_unit_dir": str(source_unit_dir),
            }
            destination_unit_path = destination_unit_dir / "unit.json"
            if destination_unit_path.exists():
                if read_json_object(
                    destination_unit_path,
                    label="Destination suite unit descriptor",
                ) != destination_unit_descriptor:
                    raise ValueError(f"Destination suite unit changed: {unit_name}")
            else:
                write_metadata_json(
                    destination_unit_path,
                    destination_unit_descriptor,
                    overwrite=False,
                )

            dense_source_run_dir = (
                source_unit_dir / "runs" / SOURCE_CONDITIONS[("dense", True)]
            )
            dense_source_run = read_json_object(
                dense_source_run_dir / "metadata.json",
                label="Dense source run metadata",
            )
            print(f"\n[{verified.unit}] validating immutable chunks...", flush=True)
            chunk_store = _open_chunk_store(dense_source_run)
            try:
                for method in ("bm25", "dense"):
                    print(
                        f"[{verified.unit}] validating {method} candidate cache...",
                        flush=True,
                    )
                    batch, candidate_identity, candidate_descriptor = (
                        _load_candidate_cache(
                            source_unit_dir,
                            method=method,
                            questions=questions,
                            corpus_count=len(chunk_store),
                        )
                    )
                    if int(candidate_descriptor.get("schema_version", 0)) <= 0:
                        raise ValueError("Candidate cache descriptor schema is invalid")
                    rerank_batch, rerank_identity, rerank_descriptor = (
                        _load_rerank_cache(
                            source_unit_dir,
                            method=method,
                            candidates=batch,
                        )
                    )
                    for rerank in (False, True):
                        source_condition = SOURCE_CONDITIONS[(method, rerank)]
                        source_run_dir = source_unit_dir / "runs" / source_condition
                        source_run = read_json_object(
                            source_run_dir / "metadata.json",
                            label="Source condition metadata",
                        )
                        if (
                            source_run.get("build_id")
                            != dense_source_run.get("build_id")
                            or Path(str(source_run.get("build_dir"))).resolve()
                            != Path(str(dense_source_run.get("build_dir"))).resolve()
                        ):
                            raise ValueError(
                                "Source conditions do not share one immutable chunk build"
                            )
                        source_rows = _load_source_rows(
                            source_run_dir,
                            source_run,
                            questions,
                        )
                        condition = _condition_name(
                            method,
                            rerank=rerank,
                            top_k=args.top_k,
                        )
                        print(
                            f"[{verified.unit}] offline recompute {condition}...",
                            flush=True,
                        )
                        condition, summary = _run_condition(
                            destination_suite_id=args.run_id,
                            destination_unit_dir=destination_unit_dir,
                            source_suite_id=source_suite_id,
                            source_suite_dir=source_suite_dir,
                            source_unit_dir=source_unit_dir,
                            source_run_dir=source_run_dir,
                            source_run=source_run,
                            source_rows=source_rows,
                            verified_questions=verified,
                            questions=questions,
                            questions_sha256=questions_sha256,
                            chunk_store=chunk_store,
                            batch=batch,
                            candidate_identity=candidate_identity,
                            candidate_descriptor=candidate_descriptor,
                            rerank_score_batch=rerank_batch if rerank else None,
                            rerank_identity=rerank_identity if rerank else None,
                            rerank_descriptor=rerank_descriptor if rerank else None,
                            method=method,
                            rerank=rerank,
                            top_k=args.top_k,
                            source_top_k=source_top_k,
                            physical_candidate_k=physical_candidate_k,
                            reanalysis_snapshot=reanalysis_snapshot,
                            resume_suite=args.resume,
                        )
                        summaries.append(
                            {
                                "dataset": verified.dataset,
                                "unit": verified.unit,
                                "condition": condition,
                                "summary": summary,
                            }
                        )
                        completed.add(f"{verified.unit}:{condition}")
                        suite_metadata["completed_conditions"] = sorted(completed)
                        write_metadata_json(metadata_path, suite_metadata)
            finally:
                chunk_store.close()

        aggregate = aggregate_suite_summaries(summaries)
        aggregate.update(
            {
                "suite_id": args.run_id,
                "source_suite_id": source_suite_id,
                "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
                "metrics_version": METRICS_VERSION,
                "physical_candidate_k": physical_candidate_k,
                "first_stage_search_k": FIRST_STAGE_SEARCH_K,
                "ignore_identical_ids": True,
                "identical_id_policy": "leakage_safe_pre_candidate_filtering",
                "final_k": args.top_k,
                "retrieval_or_reranking_repeated": False,
                "source_prefix_replay_validation": "exact_per_question",
            }
        )
        write_metadata_json(destination_suite_dir / "suite_summary.json", aggregate)
        _write_csv_atomic(
            destination_suite_dir / "suite_summary.csv",
            _suite_summary_csv_rows(aggregate),
        )
        suite_metadata.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "num_units": len(unit_descriptors),
                "num_conditions": len(summaries),
                "summary_json": str(
                    (destination_suite_dir / "suite_summary.json").resolve()
                ),
                "summary_csv": str(
                    (destination_suite_dir / "suite_summary.csv").resolve()
                ),
            }
        )
        write_metadata_json(metadata_path, suite_metadata)
    except Exception as exc:
        suite_metadata.update(
            {
                "status": "failed",
                "last_error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:1000],
                },
            }
        )
        write_metadata_json(metadata_path, suite_metadata)
        raise

    print(f"\nSaved offline BEIR Top-{args.top_k}: {destination_suite_dir}")
    print(json.dumps(aggregate["family_macro"], ensure_ascii=False, indent=2))
    return destination_suite_dir


if __name__ == "__main__":
    main()
