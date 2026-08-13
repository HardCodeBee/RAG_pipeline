"""Run the four BEIR retrieval conditions from two shared Top-50 candidate sets."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output, positive_int, safe_run_id
from src.config import load_config, resolve_cli_path, validate_config
from src.evaluation_runner import run_evaluation
from src.evaluators.beir_evaluation import (
    METRICS_VERSION,
    compute_streaming_first_stage,
    load_beir_question_split,
    summarize_beir_evaluation,
)
from src.evaluators.beir_suite import (
    BM25FirstStageCheckpointStore,
    SharedCandidateBatch,
    SharedCandidateStore,
    SharedRerankScoreBatch,
    SharedRerankScoreStore,
    SUITE_EVALUATION_PROTOCOL,
    aggregate_suite_summaries,
    beir_row_from_shared_candidates,
    compute_bm25_first_stage,
    shared_batch_from_dense,
)
from src.persistence.artifact_io import read_json_object, replace_with_retry
from src.persistence.run_output_writer import write_metadata_json
from src.preparers.beir_dataset import (
    BEIR_DATASETS,
    CQADUPSTACK_FORUMS,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL,
    canonical_dataset_name,
)
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    run_spec,
    sha256_file,
    source_files_sha256,
)
from src.pipeline import NaiveRAGPipeline
from src.query_runtime_factory import create_reranker
from src.rerankers.reranker_contract import NoOpReranker


PHYSICAL_CANDIDATE_K = 50
FIRST_STAGE_SEARCH_K = 51
FINAL_K = 5
DENSE_BATCH_SIZE = 128
DENSE_QUERY_BATCH_SIZE = 1024
DEFAULT_RERANK_QUERY_GROUP_SIZE = 16
BM25_CHECKPOINT_QUERY_GROUP_SIZE = 128
METADATA_FLUSH_INTERVAL = 128
SUITE_SCHEMA_VERSION = 1
DEFAULT_DENSE_CONFIG = "configs/beir/nfcorpus_dense_bge_top50_to5.yaml"
DEFAULT_BM25_CONFIG = "configs/beir/nfcorpus_bm25_bge_top50_to5.yaml"
CONDITIONS = (
    "bm25_top5",
    "dense_top5",
    "bm25_top50_bge_top5",
    "dense_top50_bge_top5",
)


def _summarize_suite_condition(*args: Any, **kwargs: Any) -> dict[str, Any]:
    summary = summarize_beir_evaluation(*args, **kwargs)
    summary["evaluation_protocol"] = SUITE_EVALUATION_PROTOCOL
    return summary


@dataclass(frozen=True, slots=True)
class PreparedUnit:
    dataset: str
    unit: str
    directory: Path
    manifest_sha256: str


class _LazyReranker:
    """Avoid loading BGE when a resumed suite has no unfinished rerank rows."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = copy.deepcopy(dict(config))
        self._value: Any | None = None

    def _get(self) -> Any:
        if self._value is None:
            self._value = create_reranker(self._config)
        return self._value

    def rerank(self, query: str, hits: Sequence[Any], *, final_k: int | None = None):
        return self._get().rerank(query, hits, final_k=final_k)

    def rerank_many(
        self,
        queries: Sequence[str],
        hits_by_query: Sequence[Sequence[Any]],
        *,
        final_k: int,
    ):
        value = self._get()
        method = getattr(value, "rerank_many", None)
        if callable(method):
            return method(queries, hits_by_query, final_k=final_k)
        return tuple(
            value.rerank(query, hits, final_k=final_k)
            for query, hits in zip(queries, hits_by_query)
        )


def _read_unit_descriptor(path: Path) -> PreparedUnit:
    manifest_path = path / "manifest.json"
    manifest = read_json_object(manifest_path, label="Prepared BEIR unit manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "dataset_unit"
        or manifest.get("dataset") not in BEIR_DATASETS
        or not isinstance(manifest.get("unit"), str)
        or not manifest["unit"]
    ):
        raise ValueError(f"Invalid prepared BEIR unit manifest: {manifest_path}")
    return PreparedUnit(
        dataset=str(manifest["dataset"]),
        unit=str(manifest["unit"]),
        directory=path.resolve(),
        manifest_sha256=sha256_file(manifest_path),
    )


def discover_prepared_units(data_root: str | Path) -> tuple[PreparedUnit, ...]:
    """Discover only complete structured units; never download or prepare data."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BEIR data root does not exist: {root}")
    result: list[PreparedUnit] = []
    for dataset in BEIR_DATASETS:
        dataset_root = root / dataset
        if dataset == "cqadupstack":
            collection_path = dataset_root / "manifest.json"
            if not collection_path.is_file():
                continue
            collection = read_json_object(
                collection_path,
                label="CQADupStack collection manifest",
            )
            units = collection.get("units")
            if (
                collection.get("status") != "complete"
                or collection.get("protocol") != PROTOCOL
                or collection.get("schema_version") != MANIFEST_SCHEMA_VERSION
                or collection.get("kind") != "dataset_collection"
                or collection.get("dataset") != dataset
                or not isinstance(units, Mapping)
                or set(units) != set(CQADUPSTACK_FORUMS)
            ):
                raise ValueError(f"Invalid CQADupStack collection: {dataset_root}")
            for forum in CQADUPSTACK_FORUMS:
                descriptor = units[forum]
                if not isinstance(descriptor, Mapping) or descriptor.get("path") != forum:
                    raise ValueError(f"Invalid CQADupStack unit descriptor: {forum}")
                unit = _read_unit_descriptor(dataset_root / forum)
                if unit.unit != f"cqadupstack/{forum}":
                    raise ValueError(f"Unexpected CQADupStack unit identity: {unit.unit}")
                if descriptor.get("manifest_sha256") != unit.manifest_sha256:
                    raise ValueError(f"CQADupStack unit manifest changed: {forum}")
                result.append(unit)
            continue
        manifest_path = dataset_root / "manifest.json"
        if manifest_path.is_file():
            unit = _read_unit_descriptor(dataset_root)
            if unit.dataset != dataset or unit.unit != dataset:
                raise ValueError(f"Unexpected BEIR unit identity: {unit.unit}")
            result.append(unit)
    return tuple(result)


def select_prepared_units(
    prepared: Sequence[PreparedUnit],
    selectors: Sequence[str] | None,
) -> tuple[PreparedUnit, ...]:
    """Expand family selectors while retaining benchmark order."""

    values = [
        item.strip()
        for raw in (selectors or ("all-prepared",))
        for item in raw.split(",")
        if item.strip()
    ]
    if not values:
        raise ValueError("At least one dataset selector is required")
    by_unit = {item.unit: item for item in prepared}
    if values == ["all-prepared"]:
        if not prepared:
            raise FileNotFoundError("No complete prepared BEIR units were found")
        return tuple(prepared)
    if "all-prepared" in values:
        raise ValueError("all-prepared cannot be combined with other selectors")

    requested: set[str] = set()
    for raw in values:
        normalized = raw.strip().lower().replace("\\", "/")
        if normalized.startswith("cqadupstack/"):
            forum = normalized.split("/", 1)[1]
            if forum not in CQADUPSTACK_FORUMS:
                raise ValueError(f"Unknown CQADupStack forum: {forum}")
            requested.add(f"cqadupstack/{forum}")
            continue
        dataset = canonical_dataset_name(normalized)
        if dataset == "cqadupstack":
            requested.update(f"cqadupstack/{forum}" for forum in CQADUPSTACK_FORUMS)
        else:
            requested.add(dataset)
    missing = sorted(requested - set(by_unit))
    if missing:
        raise FileNotFoundError(
            "Requested BEIR units are not completely prepared: " + ", ".join(missing)
        )
    return tuple(item for item in prepared if item.unit in requested)


def _unit_key(unit: str) -> str:
    value = unit.replace("/", "__")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"BEIR unit cannot be represented safely in an output path: {unit}")
    return value


def _load_suite_template(path: Path, *, method: str) -> dict[str, Any]:
    config = load_config(path)
    config["generation"] = {"provider": "extractive", "max_output_tokens": 1}
    # The measured local optimum is shared by dense and BM25 configs because
    # embedding.batch_size is part of the immutable encoded-corpus identity.
    config["embedding"]["batch_size"] = DENSE_BATCH_SIZE
    if method == "dense":
        config["retrieval"]["query_batch_size"] = DENSE_QUERY_BATCH_SIZE
    config = validate_config(config)
    if config["retrieval"]["method"] != method:
        raise ValueError(f"Expected a {method} suite template")
    if config["retrieval"]["candidate_k"] != PHYSICAL_CANDIDATE_K:
        raise ValueError(f"Suite templates must use candidate_k={PHYSICAL_CANDIDATE_K}")
    if config["retrieval"]["final_k"] != FINAL_K:
        raise ValueError(f"Suite templates must use final_k={FINAL_K}")
    if config["retrieval"]["reranker"]["provider"] != "cross_encoder":
        raise ValueError("Suite templates must pin the BGE cross-encoder reranker")
    if method == "dense" and config["index"]["type"] != "streaming_flat_ip":
        raise ValueError("Dense BEIR suite requires streaming_flat_ip")
    if method == "bm25" and config["bm25"]["backend"] != "sqlite":
        raise ValueError("BEIR suite requires the disk-backed SQLite BM25 backend")
    return config


def _unit_config(template: Mapping[str, Any], unit: PreparedUnit) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    config["paths"]["corpus"] = str(unit.directory)
    config["loader"]["expected_dataset"] = unit.dataset
    return validate_config(config)


def _condition_config(
    template: Mapping[str, Any],
    *,
    rerank: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    retrieval = config["retrieval"]
    retrieval["candidate_k"] = PHYSICAL_CANDIDATE_K if rerank else FINAL_K
    retrieval["final_k"] = FINAL_K
    if not rerank:
        retrieval["reranker"] = {"provider": "none"}
    return validate_config(config)


def _physical_config(template: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    config["retrieval"]["candidate_k"] = FIRST_STAGE_SEARCH_K
    config["retrieval"]["final_k"] = FINAL_K
    config["retrieval"]["reranker"] = {"provider": "none"}
    return validate_config(config)


def _evaluation_source_sha256() -> str:
    return source_files_sha256(PROJECT_ROOT, (
        "scripts/run_beir_suite.py",
        "src/evaluators/beir_suite.py",
        "src/evaluators/beir_evaluation.py",
        "src/evaluators/beir_metrics.py",
        "src/evaluation_runner.py",
        "src/loaders/beir_loader.py",
        "src/persistence/run_output_writer.py",
    ))


def _first_stage_identity(
    *,
    config: Mapping[str, Any],
    pipeline: NaiveRAGPipeline,
    questions_sha256: str,
    question_ids: Sequence[str],
    dataset_manifest_sha256: str,
    split: str,
) -> dict[str, Any]:
    method = config["retrieval"]["method"]
    retrieval = dict(config["retrieval"])
    retrieval.pop("reranker", None)
    retrieval.pop("final_k", None)
    identity: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "questions_sha256": questions_sha256,
        "question_ids_sha256": json_sha256(list(question_ids)),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "question_split": split,
        "build_id": pipeline.runtime_metadata["build_id"],
        "run_source_sha256": pipeline.runtime_metadata["run_spec"]["run_source_sha256"],
        "retrieval_method": method,
        "retrieval": retrieval,
        "requested_candidate_k": FIRST_STAGE_SEARCH_K,
        "retained_candidate_k": PHYSICAL_CANDIDATE_K,
    }
    if method == "dense":
        embedding = config["embedding"]
        identity.update(
            {
                "index_type": config["index"]["type"],
                "query_embedding": {
                    key: embedding.get(key)
                    for key in (
                        "backend",
                        "model_name",
                        "revision",
                        "normalize",
                        "query_prefix",
                        "max_sequence_length",
                        "batch_size",
                    )
                },
            }
        )
    else:
        identity["bm25"] = dict(config["bm25"])
        identity["sparse_index_id"] = pipeline.runtime_metadata.get("sparse_index_id")
    return identity


def _condition_name(method: str, *, rerank: bool) -> str:
    return f"{method}_top50_bge_top5" if rerank else f"{method}_top5"


def _condition_runtime_metadata(
    pipeline: NaiveRAGPipeline,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(pipeline.runtime_metadata)
    spec = run_spec(
        dict(config),
        pipeline.runtime_metadata["build_id"],
        pipeline.runtime_metadata["run_spec"]["run_source_sha256"],
    )
    value["run_spec"] = spec
    value["run_spec_sha256"] = json_sha256(spec)
    return value


def _run_condition(
    *,
    suite_id: str,
    unit_dir: Path,
    verified_questions: Any,
    questions: Sequence[Mapping[str, Any]],
    questions_sha256: str,
    pipeline: NaiveRAGPipeline,
    batch: SharedCandidateBatch,
    cache_identity: Mapping[str, Any],
    candidate_cache_descriptor: Mapping[str, Any],
    config: dict[str, Any],
    reranker: Any,
    rerank_score_batch: SharedRerankScoreBatch | None,
    rerank_score_identity: Mapping[str, Any] | None,
    rerank_score_cache_descriptor: Mapping[str, Any] | None,
    split: str,
    evaluation_source_sha256: str,
    resume_suite: bool,
) -> tuple[str, dict[str, Any]]:
    method = config["retrieval"]["method"]
    rerank = config["retrieval"]["reranker"]["provider"] != "none"
    condition = _condition_name(method, rerank=rerank)
    candidate_cache_descriptor = dict(candidate_cache_descriptor)
    candidate_cache_ref_sha256 = json_sha256(candidate_cache_descriptor)
    if rerank:
        if rerank_score_identity is None or rerank_score_cache_descriptor is None:
            raise ValueError("Reranked conditions require a pinned BGE score cache")
        rerank_score_cache_descriptor = dict(rerank_score_cache_descriptor)
        rerank_cache_ref_sha256 = json_sha256(rerank_score_cache_descriptor)
    else:
        if rerank_score_identity is not None or rerank_score_cache_descriptor is not None:
            raise ValueError("Baseline conditions cannot reference a BGE score cache")
        rerank_cache_ref_sha256 = None
    run_dir = unit_dir / "runs" / condition
    resume = resume_suite and run_dir.exists()
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Condition run already exists: {run_dir}")
    runtime = _condition_runtime_metadata(pipeline, config)
    evaluation_value = evaluation_spec(
        questions_sha256,
        evaluation_source_sha256,
        metrics_version=METRICS_VERSION,
    )
    condition_candidate_k = config["retrieval"]["candidate_k"]
    condition_run_id = f"{suite_id}__{_unit_key(verified_questions.unit)}__{condition}"
    metadata = {
        "run_id": condition_run_id,
        "suite_id": suite_id,
        "command": "run_beir_suite",
        "condition": condition,
        "status": "running",
        "execution_mode": "retrieval_only_shared_first_stage",
        "effective_config": recorded_config(config),
        "dataset": verified_questions.dataset,
        "unit": verified_questions.unit,
        "dataset_manifest_path": str(verified_questions.dataset_manifest_path),
        "dataset_manifest_sha256": verified_questions.dataset_manifest_sha256,
        "questions_path": str(verified_questions.queries_path),
        "questions_file_sha256": verified_questions.queries_file_sha256,
        "qrels_path": str(verified_questions.qrels_path),
        "qrels_file_sha256": verified_questions.qrels_file_sha256,
        "questions_sha256": questions_sha256,
        "question_split": split,
        "question_selection_order": (
            "queries_jsonl_order_filtered_by_qrels_split_then_prefix_v1"
        ),
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha256,
        **runtime,
        "retrieval_method": method,
        "reranker_provider": config["retrieval"]["reranker"]["provider"],
        "candidate_k": condition_candidate_k,
        "effective_top_k": FINAL_K,
        "shared_first_stage_candidate_k": PHYSICAL_CANDIDATE_K,
        "shared_first_stage_requested_k": FIRST_STAGE_SEARCH_K,
        "shared_first_stage_identity": dict(cache_identity),
        "shared_first_stage_cache": candidate_cache_descriptor,
        "shared_first_stage_cache_ref_sha256": candidate_cache_ref_sha256,
        "shared_first_stage_candidates_reused": batch.reused_cache,
        "shared_first_stage_timings_ms": dict(batch.timings_ms),
        "shared_bge_scores_reused": (
            rerank_score_batch.reused_cache if rerank_score_batch is not None else None
        ),
        "shared_bge_score_identity": (
            dict(rerank_score_identity) if rerank_score_identity is not None else None
        ),
        "shared_bge_score_cache": rerank_score_cache_descriptor,
        "shared_bge_score_cache_ref_sha256": rerank_cache_ref_sha256,
        "resume": resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
    }
    position_by_id = {
        question["question_id"]: position
        for position, question in enumerate(questions)
    }

    def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
        position = position_by_id[question["question_id"]]
        cached_scores: Sequence[float] | None = None
        cached_timing: float | None = None
        if rerank_score_batch is not None:
            cached_scores, cached_timing = rerank_score_batch.row(position, batch)
        return beir_row_from_shared_candidates(
            question=question,
            position=position,
            batch=batch,
            chunk_store=pipeline.chunk_store,
            reranker=reranker,
            method=method,
            physical_candidate_k=PHYSICAL_CANDIDATE_K,
            condition_candidate_k=condition_candidate_k,
            final_k=FINAL_K,
            build_id=runtime["build_id"],
            run_spec_sha256=runtime["run_spec_sha256"],
            save_text=config["logging"]["save_retrieved_text"],
            split=split,
            dataset=verified_questions.dataset,
            unit=verified_questions.unit,
            shared_candidate_cache_ref_sha256=candidate_cache_ref_sha256,
            rerank_scores=cached_scores,
            rerank_timing_ms=cached_timing,
            shared_rerank_cache_ref_sha256=rerank_cache_ref_sha256,
        )

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
                "split": split,
                "shared_first_stage_candidates": True,
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
            split=split,
            final_k=FINAL_K,
            candidate_k=condition_candidate_k,
            retrieval_method=method,
            reranker_provider=config["retrieval"]["reranker"]["provider"],
        ),
        error_fields=error_fields,
        metadata_flush_interval=METADATA_FLUSH_INTERVAL,
    )
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise RuntimeError(
            f"{verified_questions.unit}/{condition} saved {len(failed)} failed rows; "
            "fix the cause and resume the suite"
        )
    return condition, summary


def _suite_summary_csv_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in aggregate["unit_results"]:
        rows.append(
            {
                "scope": "unit",
                "dataset": record["dataset"],
                "unit": record["unit"],
                "condition": record["condition"],
                **record["summary"],
            }
        )
    for record in aggregate["family_results"]:
        rows.append(
            {
                "scope": "family_macro",
                "dataset": record["dataset"],
                "unit": "",
                "condition": record["condition"],
                "num_units": record["num_units"],
                **record["metrics"],
            }
        )
    for condition, record in aggregate["unit_macro"].items():
        rows.append(
            {
                "scope": "suite_unit_macro",
                "dataset": "",
                "unit": "",
                "condition": condition,
                "num_units": record["num_units"],
                **record["metrics"],
            }
        )
    for condition, record in aggregate["family_macro"].items():
        rows.append(
            {
                "scope": "suite_family_macro",
                "dataset": "",
                "unit": "",
                "condition": condition,
                "num_families": record["num_families"],
                **record["metrics"],
            }
        )
    return rows


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as output:
            output.write(handle.getvalue())
            output.flush()
            os.fsync(output.fileno())
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_suite_id(max_questions: int | None) -> str:
    size = "full" if max_questions is None else f"n{max_questions}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"beir_four_condition_{size}_{timestamp}"


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=(
            "Run BM25 Top-5, Dense Top-5, BM25 Top-50->BGE->Top-5, and "
            "Dense Top-50->BGE->Top-5 from one shared Top-50 retrieval per method."
        )
    )
    parser.add_argument("--data-root", default="data/beir")
    parser.add_argument("--dense-config", default=DEFAULT_DENSE_CONFIG)
    parser.add_argument("--bm25-config", default=DEFAULT_BM25_CONFIG)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Repeat for families/units; default is all-prepared.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-questions", type=positive_int, default=None)
    parser.add_argument(
        "--rerank-query-group-size",
        type=positive_int,
        default=DEFAULT_RERANK_QUERY_GROUP_SIZE,
    )
    parser.add_argument("--list-prepared", action="store_true")
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    data_root = resolve_cli_path(PROJECT_ROOT, args.data_root)
    prepared = discover_prepared_units(data_root)
    if args.list_prepared:
        for unit in prepared:
            print(f"{unit.unit}\t{unit.directory}")
        return data_root
    selected = select_prepared_units(prepared, args.dataset)

    dense_path = resolve_cli_path(PROJECT_ROOT, args.dense_config)
    bm25_path = resolve_cli_path(PROJECT_ROOT, args.bm25_config)
    dense_template = _load_suite_template(dense_path, method="dense")
    bm25_template = _load_suite_template(bm25_path, method="bm25")
    if dense_template["retrieval"]["reranker"] != bm25_template["retrieval"]["reranker"]:
        raise ValueError("Dense and BM25 suite templates must pin the same BGE reranker")
    dense_roots = resolved_roots(dense_template)
    bm25_roots = resolved_roots(bm25_template)
    if dense_roots["artifacts_root"] != bm25_roots["artifacts_root"]:
        raise ValueError("Dense and BM25 templates must share artifacts_root")
    if dense_roots["outputs_root"] != bm25_roots["outputs_root"]:
        raise ValueError("Dense and BM25 templates must share outputs_root")

    suite_id = args.run_id or _default_suite_id(args.max_questions)
    suite_dir = dense_roots["outputs_root"] / suite_id
    evaluation_source_sha = _evaluation_source_sha256()
    suite_identity = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "evaluation_source_sha256": evaluation_source_sha,
        "dense_template": {
            "path": str(dense_path),
            "sha256": sha256_file(dense_path),
        },
        "bm25_template": {
            "path": str(bm25_path),
            "sha256": sha256_file(bm25_path),
        },
        "dense_embedding_batch_size": DENSE_BATCH_SIZE,
        "dense_query_batch_size": DENSE_QUERY_BATCH_SIZE,
        "physical_candidate_k": PHYSICAL_CANDIDATE_K,
        "first_stage_search_k": FIRST_STAGE_SEARCH_K,
        "rerank_query_group_size": args.rerank_query_group_size,
        "bm25_checkpoint_query_group_size": BM25_CHECKPOINT_QUERY_GROUP_SIZE,
        "final_k": FINAL_K,
        "split": args.split,
        "max_questions": args.max_questions,
        "units": [
            {
                "dataset": unit.dataset,
                "unit": unit.unit,
                "directory": str(unit.directory),
                "manifest_sha256": unit.manifest_sha256,
            }
            for unit in selected
        ],
        "conditions": list(CONDITIONS),
    }
    metadata_path = suite_dir / "metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume a missing BEIR suite: {suite_dir}")
        previous = read_json_object(metadata_path, label="BEIR suite metadata")
        previous_identity = dict(previous.get("suite_identity", {}))
        previous_identity.pop("metadata_flush_interval", None)
        if previous_identity != suite_identity:
            raise ValueError("Cannot resume an incompatible BEIR suite")
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
        if suite_dir.exists():
            raise FileExistsError(f"BEIR suite directory already exists: {suite_dir}")
        suite_dir.mkdir(parents=True)
        suite_metadata = {
            "run_id": suite_id,
            "command": "run_beir_suite",
            "status": "running",
            "suite_identity": suite_identity,
            "suite_identity_sha256": json_sha256(suite_identity),
            "started_at": now,
            "completed_at": None,
            "completed_conditions": [],
            "last_error": None,
        }
        write_metadata_json(metadata_path, suite_metadata, overwrite=False)

    baseline_reranker = NoOpReranker()
    bge_reranker = _LazyReranker(dense_template)
    summaries: list[dict[str, Any]] = []
    completed = set(suite_metadata.get("completed_conditions", ()))
    try:
        for unit in selected:
            print(f"\n[{unit.unit}] loading questions...", flush=True)
            verified = load_beir_question_split(
                unit.directory,
                split=args.split,
                max_questions=args.max_questions,
                expected_dataset=unit.dataset,
            )
            questions = list(verified.questions)
            questions_sha = json_sha256(questions)
            unit_dir = suite_dir / "units" / _unit_key(unit.unit)
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_descriptor = {
                "dataset": unit.dataset,
                "unit": unit.unit,
                "directory": str(unit.directory),
                "dataset_manifest_sha256": verified.dataset_manifest_sha256,
                "questions_sha256": questions_sha,
                "num_questions": len(questions),
                "split": args.split,
            }
            unit_path = unit_dir / "unit.json"
            if unit_path.exists():
                if read_json_object(unit_path, label="Suite unit descriptor") != unit_descriptor:
                    raise ValueError(f"Suite unit descriptor changed: {unit.unit}")
            else:
                write_metadata_json(unit_path, unit_descriptor, overwrite=False)

            method_build_ids: dict[str, str] = {}
            for method, raw_template in (
                ("dense", dense_template),
                ("bm25", bm25_template),
            ):
                unit_template = _unit_config(raw_template, unit)
                physical_config = _physical_config(unit_template)
                print(f"[{unit.unit}] opening {method} first stage...", flush=True)
                pipeline = NaiveRAGPipeline(physical_config)
                try:
                    method_build_ids[method] = pipeline.runtime_metadata["build_id"]
                    if len(set(method_build_ids.values())) > 1:
                        raise ValueError(
                            "Dense and BM25 conditions resolved to different source builds"
                        )
                    cache_identity = _first_stage_identity(
                        config=physical_config,
                        pipeline=pipeline,
                        questions_sha256=questions_sha,
                        question_ids=[question["question_id"] for question in questions],
                        dataset_manifest_sha256=verified.dataset_manifest_sha256,
                        split=args.split,
                    )
                    store = SharedCandidateStore(
                        unit_dir / "candidates" / method,
                        identity=cache_identity,
                        question_ids=[question["question_id"] for question in questions],
                        candidate_k=PHYSICAL_CANDIDATE_K,
                        corpus_count=len(pipeline.chunk_store),
                        require_full_width=method == "dense",
                    )
                    bm25_checkpoint_store = (
                        BM25FirstStageCheckpointStore(
                            unit_dir / "candidates" / method / "bm25_checkpoints",
                            identity=cache_identity,
                            question_ids=[
                                question["question_id"] for question in questions
                            ],
                            candidate_k=FIRST_STAGE_SEARCH_K,
                            retained_k=PHYSICAL_CANDIDATE_K,
                            corpus_count=len(pipeline.chunk_store),
                            query_group_size=BM25_CHECKPOINT_QUERY_GROUP_SIZE,
                        )
                        if method == "bm25"
                        else None
                    )
                    batch = store.load() if args.resume else None
                    if batch is None:
                        print(
                            f"[{unit.unit}] computing {method} Top-{PHYSICAL_CANDIDATE_K} once...",
                            flush=True,
                        )
                        if method == "dense":
                            dense_batch = compute_streaming_first_stage(
                                pipeline,
                                questions,
                                candidate_k=FIRST_STAGE_SEARCH_K,
                                query_batch_size=physical_config["retrieval"]["query_batch_size"],
                            )
                            batch = shared_batch_from_dense(
                                dense_batch,
                                questions=questions,
                                chunk_store=pipeline.chunk_store,
                                retained_k=PHYSICAL_CANDIDATE_K,
                            )
                        else:
                            batch = compute_bm25_first_stage(
                                pipeline,
                                questions,
                                candidate_k=FIRST_STAGE_SEARCH_K,
                                retained_k=PHYSICAL_CANDIDATE_K,
                                checkpoint_store=bm25_checkpoint_store,
                            )
                        batch = store.write(batch)
                    else:
                        print(f"[{unit.unit}] reusing {method} Top-50 cache", flush=True)
                    if bm25_checkpoint_store is not None:
                        # Only the final candidate manifest is authoritative;
                        # group checkpoints are dispensable after it validates.
                        bm25_checkpoint_store.cleanup()
                    candidate_cache_descriptor = store.descriptor(
                        unit_directory=unit_dir
                    )

                    for rerank in (False, True):
                        condition_config = _condition_config(unit_template, rerank=rerank)
                        rerank_score_batch: SharedRerankScoreBatch | None = None
                        rerank_score_identity: dict[str, Any] | None = None
                        rerank_score_cache_descriptor: dict[str, Any] | None = None
                        if rerank:
                            rerank_score_identity = {
                                "schema_version": SUITE_SCHEMA_VERSION,
                                "shared_first_stage_identity_sha256": json_sha256(
                                    cache_identity
                                ),
                                "shared_first_stage_cache_ref_sha256": json_sha256(
                                    candidate_cache_descriptor
                                ),
                                "question_ids_sha256": json_sha256(
                                    list(batch.question_ids)
                                ),
                                "candidate_indptr_sha256": json_sha256(
                                    [int(value) for value in batch.indptr]
                                ),
                                "reranker": dict(
                                    condition_config["retrieval"]["reranker"]
                                ),
                                "run_source_sha256": pipeline.runtime_metadata["run_spec"][
                                    "run_source_sha256"
                                ],
                                "final_k": FINAL_K,
                                "query_group_size": args.rerank_query_group_size,
                            }
                            rerank_store = SharedRerankScoreStore(
                                unit_dir / "candidates" / method / "bge",
                                identity=rerank_score_identity,
                                candidates=batch,
                            )
                            rerank_score_batch = (
                                rerank_store.load() if args.resume else None
                            )
                            if rerank_score_batch is None:
                                print(
                                    f"[{unit.unit}] computing {method} BGE scores in "
                                    f"query groups of {args.rerank_query_group_size}...",
                                    flush=True,
                                )
                                rerank_score_batch = rerank_store.compute(
                                    questions=questions,
                                    chunk_store=pipeline.chunk_store,
                                    reranker=bge_reranker,
                                    final_k=FINAL_K,
                                    query_group_size=args.rerank_query_group_size,
                                )
                                rerank_score_batch = rerank_store.write(
                                    rerank_score_batch
                                )
                            else:
                                print(
                                    f"[{unit.unit}] reusing {method} BGE score cache",
                                    flush=True,
                                )
                            rerank_score_cache_descriptor = rerank_store.descriptor(
                                unit_directory=unit_dir
                            )
                        condition, summary = _run_condition(
                            suite_id=suite_id,
                            unit_dir=unit_dir,
                            verified_questions=verified,
                            questions=questions,
                            questions_sha256=questions_sha,
                            pipeline=pipeline,
                            batch=batch,
                            cache_identity=cache_identity,
                            candidate_cache_descriptor=candidate_cache_descriptor,
                            config=condition_config,
                            reranker=bge_reranker if rerank else baseline_reranker,
                            rerank_score_batch=rerank_score_batch,
                            rerank_score_identity=rerank_score_identity,
                            rerank_score_cache_descriptor=(
                                rerank_score_cache_descriptor
                            ),
                            split=args.split,
                            evaluation_source_sha256=evaluation_source_sha,
                            resume_suite=args.resume,
                        )
                        summaries.append(
                            {
                                "dataset": unit.dataset,
                                "unit": unit.unit,
                                "condition": condition,
                                "summary": summary,
                            }
                        )
                        key = f"{unit.unit}:{condition}"
                        completed.add(key)
                        suite_metadata["completed_conditions"] = sorted(completed)
                        write_metadata_json(metadata_path, suite_metadata)
                finally:
                    pipeline.close()

        aggregate = aggregate_suite_summaries(summaries)
        aggregate.update(
            {
                "suite_id": suite_id,
                "evaluation_protocol": SUITE_EVALUATION_PROTOCOL,
                "metrics_version": METRICS_VERSION,
                "physical_candidate_k": PHYSICAL_CANDIDATE_K,
                "first_stage_search_k": FIRST_STAGE_SEARCH_K,
                "final_k": FINAL_K,
            }
        )
        write_metadata_json(suite_dir / "suite_summary.json", aggregate)
        _write_csv_atomic(
            suite_dir / "suite_summary.csv",
            _suite_summary_csv_rows(aggregate),
        )
        suite_metadata.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "num_units": len(selected),
                "num_conditions": len(summaries),
                "summary_json": str((suite_dir / "suite_summary.json").resolve()),
                "summary_csv": str((suite_dir / "suite_summary.csv").resolve()),
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

    print(f"\nSaved BEIR suite: {suite_dir}")
    print(json.dumps(aggregate["family_macro"], ensure_ascii=False, indent=2))
    return suite_dir


if __name__ == "__main__":
    main()
