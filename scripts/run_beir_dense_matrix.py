"""Evaluate frozen dense baselines with the leakage-safe BEIR v2 protocol."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output, positive_int, safe_run_id
from scripts.run_beir_suite import (
    _unit_config,
    _unit_key,
    discover_prepared_units,
    select_prepared_units,
)
from src.config import load_config, resolve_cli_path, validate_config
from src.evaluators.beir_evaluation import (
    METRICS_VERSION,
    compute_streaming_first_stage,
    load_beir_question_split,
)
from src.evaluators.beir_metrics import score_beir_query, summarize_beir_rows
from src.evaluators.beir_suite import (
    SHARED_CANDIDATE_FILE,
    SHARED_CANDIDATE_MANIFEST,
    SUITE_EVALUATION_PROTOCOL,
    SharedCandidateBatch,
    SharedCandidateStore,
    aggregate_suite_summaries,
    shared_batch_from_dense,
)
from src.index_builder import build_index
from src.persistence.artifact_io import describe_artifact, iter_jsonl, read_json_object
from src.persistence.artifact_validation import verify_artifact_descriptor
from src.persistence.run_output_writer import write_metadata_json, write_results
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    json_sha256,
    recorded_config,
    resolved_roots,
    source_files_sha256,
)
from src.retrievers.chunk_store import JsonlOffsetChunkStore


CUTOFFS = (5, 10, 20, 50)
SEARCH_K = 51
RETAINED_K = 50
DEFAULT_BASELINES = (
    ("dpr", "configs/beir/nfcorpus_dense_dpr_top50.yaml"),
    ("contriever", "configs/beir/nfcorpus_dense_contriever_top50.yaml"),
)
BGE_REFERENCE = ("bge", "configs/beir/nfcorpus_dense_bge_top50_to5.yaml")
SELECTED_7 = ("nfcorpus", "fiqa", "arguana", "hotpotqa", "nq", "webis-touche2020", "cqadupstack")
_EXECUTION_KEYS = {"batch_size", "device", "encode_call_rows", "local_files_only", "shard_rows"}
_MATRIX_SCHEMA_VERSION = 1
_EVALUATION_SOURCE_FILES = (
    "scripts/cli_support.py",
    "scripts/run_beir_dense_matrix.py",
    "scripts/run_beir_suite.py",
    "src/evaluators/beir_evaluation.py",
    "src/evaluators/beir_metrics.py",
    "src/evaluators/beir_suite.py",
    "src/loaders/beir_loader.py",
    "src/persistence/artifact_io.py",
    "src/persistence/artifact_validation.py",
    "src/persistence/run_output_writer.py",
)
Baseline = tuple[str, Path, dict[str, Any]]
CandidateContext = tuple[SharedCandidateBatch, Any, dict[str, Any], Any]


def _evaluation_source_sha256() -> str:
    return source_files_sha256(PROJECT_ROOT, _EVALUATION_SOURCE_FILES)


def _scientific_embedding(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config["embedding"].items()
        if key not in _EXECUTION_KEYS
    }


def _load_baselines(include_bge: bool = False) -> tuple[Baseline, ...]:
    result = []
    pairs = (BGE_REFERENCE,) + DEFAULT_BASELINES if include_bge else DEFAULT_BASELINES
    for name, raw_path in pairs:
        path = resolve_cli_path(PROJECT_ROOT, raw_path)
        config = load_config(path)
        config["generation"] = {"provider": "extractive", "max_output_tokens": 1}
        config["retrieval"].update(
            method="dense",
            candidate_k=SEARCH_K,
            final_k=RETAINED_K,
            reranker={"provider": "none"},
        )
        config = validate_config(config)
        if config["index"].get("backend") != "faiss" or config["index"].get("type") != "streaming_flat_ip":
            raise ValueError(f"{name} must use faiss:streaming_flat_ip")
        result.append((name, path, config))
    return tuple(result)


def _matches(value: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(value.get(key) == item for key, item in expected.items())


def _source_chunks(run: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    build_dir = Path(run["build_dir"]).resolve()
    manifest_path = build_dir / "manifest.json"
    manifest = read_json_object(manifest_path, label="Reusable build manifest")
    if manifest.get("status") != "complete" or manifest.get("build_id") != run.get("build_id"):
        raise ValueError("Reusable build identity is invalid")
    artifacts = manifest["artifacts"]
    chunk_desc = artifacts["chunks"]
    chunks = verify_artifact_descriptor(build_dir, chunk_desc, label="Reusable chunks").path
    offset_desc = artifacts["chunk_offsets"]
    offsets = verify_artifact_descriptor(
        build_dir, offset_desc, label="Reusable chunk offsets", expected_rows=chunk_desc["rows"]
    ).path
    store = JsonlOffsetChunkStore(chunks, offsets, expected_rows=chunk_desc["rows"])
    return store, {"build_id": run["build_id"], "manifest": describe_artifact(manifest_path)}


def _reuse_v2(
    suites: tuple[Path, ...], config: dict[str, Any], unit: Any, split: str,
    question_ids: tuple[str, ...], dataset_sha: str,
) -> CandidateContext | None:
    unit_key = _unit_key(unit.unit)
    for suite in suites:
        metadata_path = suite / "metadata.json"
        if not metadata_path.is_file():
            continue
        suite_meta = read_json_object(metadata_path, label="Reusable suite metadata")
        identity = suite_meta.get("suite_identity", {})
        if not _matches(suite_meta, {"status": "completed", "command": "run_beir_suite"}) or not _matches(
            identity,
            {
                "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
                "physical_candidate_k": RETAINED_K,
                "first_stage_search_k": SEARCH_K,
                "split": split,
            },
        ):
            continue
        unit_dir = suite / "units" / unit_key
        run_path = unit_dir / "runs" / "dense_top5" / "metadata.json"
        cache_dir = unit_dir / "candidates" / "dense"
        if not run_path.is_file() or not (cache_dir / SHARED_CANDIDATE_MANIFEST).is_file():
            continue
        run = read_json_object(run_path, label="Reusable dense run metadata")
        effective = run.get("effective_config")
        if not _matches(
            run,
            {
                "status": "completed",
                "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
                "dataset": unit.dataset,
                "unit": unit.unit,
                "dataset_manifest_sha256": dataset_sha,
                "retrieval_method": "dense",
                "reranker_provider": "none",
                "shared_first_stage_candidate_k": RETAINED_K,
                "shared_first_stage_requested_k": SEARCH_K,
            },
        ) or not isinstance(effective, dict):
            continue
        if effective.get("index", {}).get("backend") != "faiss" or (
            effective.get("index", {}).get("type") != "streaming_flat_ip"
        ) or (
            _scientific_embedding(effective) != _scientific_embedding(config)
        ):
            continue
        manifest = read_json_object(
            cache_dir / SHARED_CANDIDATE_MANIFEST, label="Reusable candidate manifest"
        )
        cache_identity = manifest.get("identity", {})
        source_identity = run.get("shared_first_stage_identity")
        source_descriptor = run.get("shared_first_stage_cache")
        source_descriptor_sha = run.get("shared_first_stage_cache_ref_sha256")
        if not _matches(
            cache_identity,
            {
                "build_id": run.get("build_id"),
                "dataset_manifest_sha256": dataset_sha,
                "question_split": split,
                "retrieval_method": "dense",
                "index_type": "streaming_flat_ip",
                "requested_candidate_k": SEARCH_K,
                "retained_candidate_k": RETAINED_K,
            },
        ):
            continue
        if not isinstance(source_identity, Mapping) or cache_identity != source_identity:
            raise ValueError("Reusable candidate identity differs from its source run")
        if not isinstance(source_descriptor, Mapping) or (
            json_sha256(source_descriptor) != source_descriptor_sha
        ):
            raise ValueError("Reusable candidate descriptor reference is invalid")
        chunks, build = _source_chunks(run)
        try:
            data_path = cache_dir / SHARED_CANDIDATE_FILE
            with np.load(data_path, allow_pickle=False) as arrays:
                stored_ids = tuple(str(value) for value in arrays["question_ids"])
            store = SharedCandidateStore(
                cache_dir,
                identity=cache_identity,
                question_ids=stored_ids,
                candidate_k=RETAINED_K,
                corpus_count=len(chunks),
                require_full_width=True,
            )
            batch = store.load()
            if batch is None:
                raise RuntimeError("Reusable candidate cache disappeared")
            actual_descriptor = store.descriptor(unit_directory=unit_dir)
            if "identity_sha256" in source_descriptor:
                actual_descriptor["identity_sha256"] = json_sha256(cache_identity)
            if actual_descriptor != source_descriptor:
                raise ValueError("Reusable candidate descriptor differs from its source run")
            if batch.question_ids[: len(question_ids)] != question_ids:
                raise ValueError("Reusable candidates do not match the requested question prefix")
        except Exception:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
            raise
        return (
            batch,
            chunks,
            {
                "mode": "completed_v2_suite",
                "suite": str(suite),
                "candidate_cache": actual_descriptor,
                "build": build,
                "first_stage_timings_ms": dict(batch.timings_ms),
                "latency_scope": "reused_source_batch_not_fresh_matrix_latency",
            },
            chunks,
        )
    return None


def _fresh(
    config: dict[str, Any], questions: list[dict[str, Any]], dataset_sha: str,
    split: str, cache_dir: Path, unit_dir: Path, *, resume: bool = False,
) -> CandidateContext:
    pipeline = NaiveRAGPipeline(config)
    question_ids = tuple(str(row["question_id"]) for row in questions)
    identity = {
        "schema_version": 1,
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "questions_sha256": json_sha256(questions),
        "question_ids_sha256": json_sha256(list(question_ids)),
        "dataset_manifest_sha256": dataset_sha,
        "question_split": split,
        "build_id": pipeline.runtime_metadata["build_id"],
        "run_spec_sha256": pipeline.runtime_metadata["run_spec_sha256"],
        "retrieval_method": "dense",
        "index_type": "streaming_flat_ip",
        "requested_candidate_k": SEARCH_K,
        "retained_candidate_k": RETAINED_K,
        "query_embedding": _scientific_embedding(config),
    }
    store = SharedCandidateStore(
        cache_dir,
        identity=identity,
        question_ids=question_ids,
        candidate_k=RETAINED_K,
        corpus_count=len(pipeline.chunk_store),
        require_full_width=True,
    )
    try:
        batch = store.load() if resume else None
        if batch is None:
            dense = compute_streaming_first_stage(
                pipeline, questions, candidate_k=SEARCH_K,
                query_batch_size=config["retrieval"]["query_batch_size"],
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            batch = store.write(
                shared_batch_from_dense(
                    dense, questions=questions, chunk_store=pipeline.chunk_store,
                    retained_k=RETAINED_K,
                )
            )
        descriptor = store.descriptor(unit_directory=unit_dir)
        source = {
            "mode": (
                "resumed_matrix_candidate_cache"
                if batch.reused_cache else "fresh_exact_streaming_flat_ip"
            ),
            "candidate_cache": descriptor,
            "first_stage_timings_ms": dict(batch.timings_ms),
            "latency_scope": (
                "reused_matrix_cache_not_fresh_latency"
                if batch.reused_cache else "fresh_selected_question_batch"
            ),
        }
    except Exception:
        pipeline.close()
        raise
    return batch, pipeline.chunk_store, source, pipeline


def _score(
    batch: SharedCandidateBatch, chunks: Any, questions: list[dict[str, Any]],
    baseline: str, dataset: str, unit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if batch.question_ids[: len(questions)] != tuple(str(row["question_id"]) for row in questions):
        raise ValueError("Candidate/question order mismatch")
    results = []
    for position, question in enumerate(questions):
        _, vector_ids = batch.row(position)
        doc_ids = [chunk.doc_id for chunk in chunks.get_many(vector_ids.tolist())]
        metrics = {}
        for cutoff in CUTOFFS:
            value = score_beir_query(doc_ids, question["qrels"], k=cutoff)
            metrics[str(cutoff)] = value
        results.append({"question_id": question["question_id"], "metrics_by_cutoff": metrics})
    return results, _summarize_results(results, baseline, dataset, unit)


def _summarize_results(
    rows: list[dict[str, Any]], baseline: str, dataset: str, unit: str,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "unit": unit,
            "condition": f"{baseline}@{cutoff}",
            "summary": summarize_beir_rows(
                [
                    {"status": "success", "metrics": row["metrics_by_cutoff"][str(cutoff)]}
                    for row in rows
                ],
                k=cutoff,
            ),
        }
        for cutoff in CUTOFFS
    ]


def _corpus_count(unit: Any) -> int:
    manifest = read_json_object(unit.directory / "manifest.json", label="Prepared unit manifest")
    value = manifest.get("counts", {}).get("corpus")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Prepared unit has no positive corpus count: {unit.unit}")
    return value


def _build_selected(baselines: tuple[Baseline, ...], units: tuple[Any, ...]) -> None:
    for unit in sorted(units, key=lambda item: (_corpus_count(item), item.unit)):
        for name, _, template in baselines:
            if name == "bge":
                continue
            print(f"[{unit.unit}] {name} build START", flush=True)
            verified = build_index(_unit_config(template, unit))
            print(
                f"[{unit.unit}] {name} build DONE "
                f"build_id={verified.manifest['build_id']}",
                flush=True,
            )


def _suite_descriptor(path: Path) -> dict[str, Any]:
    metadata = path / "metadata.json"
    return {
        "path": str(path),
        "metadata": describe_artifact(metadata) if metadata.is_file() else None,
    }


def _matrix_identity(
    baselines: tuple[Baseline, ...], units: tuple[Any, ...], suites: tuple[Path, ...],
    *, split: str, max_questions: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": _MATRIX_SCHEMA_VERSION,
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "evaluation_source_sha256": _evaluation_source_sha256(),
        "search_k": SEARCH_K,
        "retained_k": RETAINED_K,
        "cutoffs": list(CUTOFFS),
        "split": split,
        "max_questions": max_questions,
        "baselines": [
            {
                "name": name,
                "template_path": str(path),
                "template": describe_artifact(path),
                "effective_template": recorded_config(config),
            }
            for name, path, config in baselines
        ],
        "units": [
            {
                "dataset": unit.dataset,
                "unit": unit.unit,
                "directory": str(unit.directory),
                "manifest_sha256": unit.manifest_sha256,
            }
            for unit in units
        ],
        "reuse_suites": [_suite_descriptor(path) for path in suites],
    }


def _open_matrix_run(
    run_dir: Path, run_id: str, identity: dict[str, Any], *, resume: bool,
) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume a missing dense matrix: {run_dir}")
        metadata = read_json_object(metadata_path, label="Dense matrix metadata")
        if metadata.get("matrix_identity") != identity or (
            metadata.get("matrix_identity_sha256") != json_sha256(identity)
        ):
            raise ValueError("Cannot resume an incompatible dense matrix")
        metadata.update(
            status="running", resumed_at=now, completed_at=None, last_error=None
        )
    else:
        if run_dir.exists():
            raise FileExistsError(f"Dense matrix directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        metadata = {
            "run_id": run_id,
            "command": "run_beir_dense_matrix",
            "status": "running",
            "matrix_identity": identity,
            "matrix_identity_sha256": json_sha256(identity),
            "started_at": now,
            "completed_at": None,
            "completed_baselines": [],
            "last_error": None,
        }
    write_metadata_json(metadata_path, metadata, overwrite=resume)
    return metadata


def _baseline_identity(
    matrix_identity_sha256: str, baseline: str, config: dict[str, Any], unit: Any,
    questions: list[dict[str, Any]], dataset_sha256: str, split: str,
) -> dict[str, Any]:
    return {
        "schema_version": _MATRIX_SCHEMA_VERSION,
        "matrix_identity_sha256": matrix_identity_sha256,
        "baseline": baseline,
        "dataset": unit.dataset,
        "unit": unit.unit,
        "split": split,
        "dataset_manifest_sha256": dataset_sha256,
        "questions_sha256": json_sha256(questions),
        "question_ids_sha256": json_sha256(
            [str(question["question_id"]) for question in questions]
        ),
        "num_questions": len(questions),
        "effective_config": recorded_config(config),
    }


def _load_completed_baseline(
    baseline_dir: Path, identity: dict[str, Any], question_ids: tuple[str, ...],
) -> list[dict[str, Any]] | None:
    metadata_path = baseline_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json_object(metadata_path, label="Dense baseline metadata")
    if metadata.get("status") != "completed":
        return None
    if metadata.get("baseline_identity") != identity or (
        metadata.get("baseline_identity_sha256") != json_sha256(identity)
    ):
        raise ValueError(f"Cannot resume incompatible completed baseline: {baseline_dir}")
    descriptor = metadata.get("results_artifact")
    if not isinstance(descriptor, Mapping) or descriptor.get("file") != "results.jsonl":
        raise ValueError("Completed dense baseline has no valid results descriptor")
    results_path = verify_artifact_descriptor(
        baseline_dir, descriptor, label="Completed dense baseline results",
        expected_rows=len(question_ids),
    ).path
    rows = list(iter_jsonl(results_path))
    if tuple(str(row.get("question_id")) for row in rows) != question_ids:
        raise ValueError("Completed dense baseline results do not match the question order")
    summaries = _summarize_results(
        rows, identity["baseline"], identity["dataset"], identity["unit"]
    )
    if metadata.get("summaries") != summaries:
        raise ValueError("Completed dense baseline summaries are invalid")
    return summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BGE/DPR/Contriever dense-only BEIR v2 matrix")
    parser.add_argument("--data-root", default="data/beir")
    parser.add_argument("--dataset", action="append", help="Default: nfcorpus; or use selected-7")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-questions", type=positive_int)
    parser.add_argument("--run-id", type=safe_run_id)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--build-missing", action="store_true",
        help="Build or idempotently reuse selected DPR/Contriever indexes before evaluation",
    )
    parser.add_argument("--include-bge", action="store_true", help="Add a reuse-only BGE v2 reference")
    parser.add_argument("--reuse-suite", action="append", help="Restrict v2 reuse search to this suite")
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _parser()
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")
    baselines = _load_baselines(args.include_bge)
    outputs = {resolved_roots(config)["outputs_root"] for _, _, config in baselines}
    if len(outputs) != 1:
        raise ValueError("All baselines must share one outputs_root")
    output_root = outputs.pop()
    selectors = args.dataset or ["nfcorpus"]
    if "selected-7" in selectors:
        if selectors != ["selected-7"]:
            raise ValueError("selected-7 cannot be combined with other dataset selectors")
        selectors = list(SELECTED_7)
    units = tuple(select_prepared_units(
        discover_prepared_units(resolve_cli_path(PROJECT_ROOT, args.data_root)), selectors
    ))
    if args.reuse_suite:
        suites = tuple(resolve_cli_path(PROJECT_ROOT, path) for path in args.reuse_suite)
    elif args.include_bge:
        suites = tuple(sorted(output_root.glob("beir_*four_condition_full_v2")))
    else:
        suites = ()
    if args.build_missing:
        _build_selected(baselines, units)
    run_id = args.run_id or "beir_dense_matrix_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    identity = _matrix_identity(
        baselines, units, suites, split=args.split, max_questions=args.max_questions
    )
    metadata = _open_matrix_run(run_dir, run_id, identity, resume=args.resume)
    identity_sha = json_sha256(identity)
    records: list[dict[str, Any]] = []
    completed: set[str] = set()
    try:
        for unit in units:
            verified = load_beir_question_split(
                unit.directory,
                split=args.split,
                max_questions=args.max_questions,
                expected_dataset=unit.dataset,
            )
            questions = list(verified.questions)
            question_ids = tuple(str(row["question_id"]) for row in questions)
            for baseline_name, _, template in baselines:
                config = _unit_config(template, unit)
                unit_dir = run_dir / "units" / _unit_key(unit.unit)
                baseline_dir = unit_dir / baseline_name
                baseline_identity = _baseline_identity(
                    identity_sha, baseline_name, config, unit, questions,
                    verified.dataset_manifest_sha256, args.split,
                )
                key = f"{unit.unit}:{baseline_name}"
                print(f"[{unit.unit}] {baseline_name} evaluation START", flush=True)
                summaries = (
                    _load_completed_baseline(baseline_dir, baseline_identity, question_ids)
                    if args.resume else None
                )
                if summaries is not None:
                    print(f"[{unit.unit}] resuming completed {baseline_name}", flush=True)
                    records.extend(summaries)
                else:
                    context = _reuse_v2(
                        suites, config, unit, args.split, question_ids,
                        verified.dataset_manifest_sha256,
                    )
                    if context is None:
                        if baseline_name == "bge":
                            raise FileNotFoundError(
                                "No compatible completed BGE v2 candidates were found "
                                f"for {unit.unit}"
                            )
                        context = _fresh(
                            config, questions, verified.dataset_manifest_sha256, args.split,
                            baseline_dir / "candidates", unit_dir, resume=args.resume,
                        )
                    batch, chunks, candidate_source, owner = context
                    try:
                        rows, summaries = _score(
                            batch, chunks, questions, baseline_name, unit.dataset, unit.unit
                        )
                        baseline_dir.mkdir(parents=True, exist_ok=True)
                        results_path = baseline_dir / "results.jsonl"
                        write_results(results_path, rows, overwrite=args.resume)
                        write_metadata_json(
                            baseline_dir / "metadata.json",
                            {
                                "status": "completed",
                                "baseline_identity": baseline_identity,
                                "baseline_identity_sha256": json_sha256(baseline_identity),
                                "results_artifact": describe_artifact(
                                    results_path, rows=len(rows)
                                ),
                                "candidate_source": candidate_source,
                                "summaries": summaries,
                            },
                            overwrite=args.resume,
                        )
                        records.extend(summaries)
                    finally:
                        close = getattr(owner, "close", None)
                        if callable(close):
                            close()
                completed.add(key)
                metadata["completed_baselines"] = sorted(completed)
                write_metadata_json(run_dir / "metadata.json", metadata)
        summary = {
            "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
            "metrics_version": METRICS_VERSION,
            "search_k": SEARCH_K,
            "retained_k": RETAINED_K,
            "cutoffs": list(CUTOFFS),
            **aggregate_suite_summaries(records),
        }
        summary_path = run_dir / "matrix_summary.json"
        write_metadata_json(summary_path, summary, overwrite=args.resume)
        metadata.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            last_error=None,
            matrix_summary=describe_artifact(summary_path),
        )
        write_metadata_json(run_dir / "metadata.json", metadata)
    except Exception as exc:
        metadata.update(
            status="failed",
            last_error={"type": exc.__class__.__name__, "message": str(exc)[:1000]},
        )
        write_metadata_json(run_dir / "metadata.json", metadata)
        raise
    print(f"Saved dense matrix: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
