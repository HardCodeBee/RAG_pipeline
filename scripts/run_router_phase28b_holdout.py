#!/usr/bin/env python3
"""Run the frozen Phase 2.8b independent router-candidate holdout selection."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_experiments.runtime import (  # noqa: E402
    connect as _connect,
    core_config as _core_config,
    dense_summary as _dense_summary,
    lexical_features as _lexical_features,
    load_examples as _load_examples,
    load_router_config as _load_router_config,
    sync_sample as _sync_sample,
)
from src.router_experiments.modeling import (  # noqa: E402
    RouterData,
    TIE_ATOL,
    candidate_by_id as _candidate_by_id,
    fit_affine_calibration as _fit_affine_calibration,
    fit_predict_candidate as _fit_predict_candidate,
    fit_predict_late_fusion as _fit_predict_late_fusion,
    load_frozen_data,
    make_group_stratified_folds,
    policy_metrics,
)
from src.router_experiments.phase28_runtime import (  # noqa: E402
    completed_labels as _completed_labels,
    generate as generate_phase28,
    prepare_bm25_incremental as _prepare_bm25_incremental,
    prepare_dense_incremental as _prepare_dense_incremental,
    spent_generation_cost as _spent_generation_cost,
)
from src.embedders.text_embedder import create_embedder  # noqa: E402
from src.retrievers.sqlite_bm25 import (  # noqa: E402
    analyze_sqlite_bm25_text,
    read_sqlite_bm25_term_stats,
)


DEFAULT_CONFIG = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase28b/config.yaml"
BASE_PHASE28_CONFIG = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase28/config.yaml"
ROUTER_CONFIG = PROJECT_ROOT / "outputs/router/hotpotqa_bd_router_v1/config.yaml"
CORPUS_PROTOTYPES = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1/"
    "features/corpus_prototypes.npy"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage",
        required=True,
        choices=("freeze", "retrieve", "fit", "generate", "evaluate", "status"),
    )
    parser.add_argument("--retrieval-action", choices=("both", "dense", "bm25"), default="both")
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _run_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(str(config["outputs"]["run_dir"]))


def _config_path(args: argparse.Namespace) -> Path:
    return _resolve(str(args.config))


def _validate_protocol(config: Mapping[str, Any], config_path: Path) -> None:
    protocol = config.get("protocol", {})
    if protocol.get("status") != "frozen_before_confirmation_open":
        raise ValueError("Phase 2.8b protocol is not frozen before confirmation open")
    if config.get("scope", {}).get("official_final_holdout_rows_allowed") != 0:
        raise ValueError("The official final holdout must remain sealed")
    confirmation = config["sealed_confirmation"]
    confirmation_path = _resolve(str(confirmation["path"]))
    if _sha256(confirmation_path) != str(confirmation["sha256"]).lower():
        raise ValueError("Sealed confirmation hash differs from the frozen protocol")
    for view in config["training_views"]:
        path = _resolve(str(view["config"]))
        if _sha256(path) != str(view["config_sha256"]).lower():
            raise ValueError(f"Training-view config hash differs: {view['id']}")
    router = _load_router_config(ROUTER_CONFIG)
    generation = config["generation"]
    if int(router["retrieval"]["candidate_k"]) != int(generation["retriever_candidate_k"]):
        raise ValueError("Retriever candidate_k differs from the frozen holdout protocol")
    if int(router["retrieval"]["final_k"]) != int(generation["generation_context_k"]):
        raise ValueError("Generation context_k differs from the frozen holdout protocol")
    if int(router["context"]["max_tokens"]) != int(generation["context_max_tokens"]):
        raise ValueError("Context token limit differs from the frozen holdout protocol")
    if str(router["prompt"]["version"]) != str(generation["prompt_id"]):
        raise ValueError("Prompt differs from the frozen holdout protocol")
    runtime_generation = _core_config(router, method="bm25")["generation"]
    for key in ("model", "temperature", "max_output_tokens"):
        if runtime_generation[key] != generation[key]:
            raise ValueError(f"Generation {key} differs from the frozen holdout protocol")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)


def freeze(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    run_dir = _run_dir(config)
    manifest_path = run_dir / "holdout_open_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("protocol_config_sha256") != _sha256(config_path):
            raise ValueError("Existing holdout run belongs to a different protocol config")
        return manifest

    confirmation_path = _resolve(str(config["sealed_confirmation"]["path"]))
    sealed = _read_json(confirmation_path)
    if sealed.get("sealed") is not True:
        raise ValueError("Confirmation sample was not marked sealed")
    rows = sealed.get("rows")
    expected = int(config["sealed_confirmation"]["expected_queries"])
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Expected {expected} sealed confirmation rows")
    query_ids = [str(row["query_id"]) for row in rows]
    group_ids = [str(row["group_id"]) for row in rows]
    if len(set(query_ids)) != expected or len(set(group_ids)) != expected:
        raise ValueError("Confirmation must contain unique query and group ids")
    if {str(row["partition"]) for row in rows} != {"train"}:
        raise ValueError("Confirmation rows must originate from HotpotQA train")

    phase28_run = _resolve("outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1")
    connection = sqlite3.connect(phase28_run / "state.sqlite3")
    try:
        acquisition_groups = {
            str(row[0]) for row in connection.execute("SELECT group_id FROM acquisition")
        }
    finally:
        connection.close()
    if set(group_ids) & acquisition_groups:
        raise RuntimeError("Confirmation groups overlap the Phase 2.8 acquisition pool")
    t2_features = _resolve(
        "outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1/"
        "training_views/T2_winner3000/snapshot/features.npz"
    )
    with np.load(t2_features, allow_pickle=False) as stored:
        training_groups = {str(value) for value in stored["group_ids"]}
    if set(group_ids) & training_groups:
        raise RuntimeError("Confirmation groups overlap a frozen training view")

    run_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "seed": int(sealed["seed"]),
        "selection": str(sealed["selection"]),
        "counts": {"train": expected},
        "rows": rows,
    }
    sample_path = run_dir / "sample.json"
    _write_json(sample_path, sample)
    shutil.copyfile(config_path, run_dir / "frozen_config.yaml")
    connection = _connect(run_dir)
    try:
        _sync_sample(connection, sample)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS acquisition (
                query_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                query_rank INTEGER NOT NULL,
                stratum TEXT,
                selected_lane TEXT,
                selected_at_unix REAL
            )
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO acquisition VALUES (?, ?, ?, ?, ?, ?)",
            [
                (query_id, group_id, rank, "sealed_confirmation", "holdout_selection", time.time())
                for rank, (query_id, group_id) in enumerate(zip(query_ids, group_ids))
            ],
        )
        connection.commit()
    finally:
        connection.close()
    manifest = {
        "protocol_id": config["protocol"]["id"],
        "status": "opened_for_frozen_candidate_selection",
        "protocol_config": _relative(config_path),
        "protocol_config_sha256": _sha256(config_path),
        "source_confirmation_sha256": _sha256(confirmation_path),
        "run_sample": _relative(sample_path),
        "run_sample_sha256": _sha256(sample_path),
        "queries": expected,
        "groups": len(set(group_ids)),
        "overlap_with_acquisition_groups": 0,
        "overlap_with_training_groups": 0,
        "training_outcomes_read": 0,
        "official_final_holdout_rows_read": 0,
        "opened_at_unix": time.time(),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _sample_and_examples(
    config: Mapping[str, Any], router: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    sample = _read_json(_run_dir(config) / "sample.json")
    examples, mapping = _load_examples(router, sample)
    return sample, examples, mapping


def retrieve(
    config: Mapping[str, Any], router: Mapping[str, Any], *, action: str
) -> dict[str, Any]:
    run_dir = _run_dir(config)
    _, examples, mapping = _sample_and_examples(config, router)
    repeats = int(config["generation"]["repeats_per_query_action"])
    connection = _connect(run_dir)
    try:
        if action in {"both", "bm25"}:
            _prepare_bm25_incremental(connection, router, examples, repeats=repeats)
        if action in {"both", "dense"}:
            _prepare_dense_incremental(
                connection,
                router,
                examples,
                repeats=repeats,
                query_batch_size=int(config["retrieval_execution"]["dense_query_batch_size"]),
                checkpoint_group_size=int(
                    config["retrieval_execution"]["dense_checkpoint_queries"]
                ),
            )
    finally:
        connection.close()
    result = {"mapping": mapping, "status": status(config)}
    _write_json(run_dir / "retrieval_status.json", result)
    return result


def _build_features(config: Mapping[str, Any], router: Mapping[str, Any]) -> Path:
    run_dir = _run_dir(config)
    path = run_dir / "holdout_features.npz"
    if path.is_file():
        return path
    _, examples, _ = _sample_and_examples(config, router)
    questions = [str(row["question"]) for row in examples]
    query_ids = [str(row["query_id"]) for row in examples]
    group_ids = [str(row["group_id"]) for row in examples]
    all_terms = {
        token for question in questions for token in analyze_sqlite_bm25_text(question)
    }
    bm25_database = _resolve(str(router["artifacts"]["bm25_index"])) / "index.sqlite3"
    stats = read_sqlite_bm25_term_stats(bm25_database, all_terms)
    lexical = np.asarray(
        [_lexical_features(question, stats)[0] for question in questions], dtype=np.float32
    )
    embedder = create_embedder(_core_config(router, method="dense"), role="query")
    embedding = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    prototypes = np.asarray(np.load(CORPUS_PROTOTYPES, allow_pickle=False), dtype=np.float32)
    corpus, _ = _dense_summary(embedding, prototypes)
    expected = int(config["sealed_confirmation"]["expected_queries"])
    if lexical.shape != (expected, 17) or corpus.shape != (expected, 13):
        raise ValueError("Holdout structured feature dimensions differ")
    if embedding.shape != (expected, 384):
        raise ValueError("Holdout embedding dimensions differ")
    if not all(np.isfinite(value).all() for value in (lexical, corpus, embedding)):
        raise ValueError("Holdout features contain non-finite values")
    np.savez_compressed(
        path,
        query_ids=np.asarray(query_ids, dtype=np.str_),
        group_ids=np.asarray(group_ids, dtype=np.str_),
        lexical=lexical,
        dense=corpus,
        embedding=embedding,
    )
    _write_json(
        run_dir / "holdout_features_manifest.json",
        {
            "protocol_id": config["protocol"]["id"],
            "query_count": expected,
            "features": _relative(path),
            "features_sha256": _sha256(path),
            "information_boundary": "query_and_corpus_static_only",
            "retrieval_or_generation_fields_used": [],
            "official_final_holdout_rows_read": 0,
        },
    )
    return path


def _combined_data(train: RouterData, feature_path: Path) -> tuple[RouterData, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as stored:
        hold_query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        hold_group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        hold_lexical = np.asarray(stored["lexical"], dtype=np.float64)
        hold_corpus = np.asarray(stored["dense"], dtype=np.float64)
        hold_embedding = np.asarray(stored["embedding"], dtype=np.float64)
    count = len(hold_query_ids)
    holdout_indices = np.arange(len(train.query_ids), len(train.query_ids) + count, dtype=np.int64)
    combined = RouterData(
        query_ids=np.concatenate([train.query_ids, hold_query_ids]),
        group_ids=np.concatenate([train.group_ids, hold_group_ids]),
        query_ranks=np.concatenate([train.query_ranks, np.arange(count, dtype=np.int64)]),
        bm25=np.concatenate([train.bm25, np.zeros(count)]),
        dense=np.concatenate([train.dense, np.zeros(count)]),
        gap=np.concatenate([train.gap, np.zeros(count)]),
        strata=np.concatenate([train.strata, np.full(count, "exact_tie", dtype=np.str_)]),
        repeat_values=np.concatenate([train.repeat_values, np.zeros((count, 2, 3))]),
        lexical=np.vstack([train.lexical, hold_lexical]),
        corpus=np.vstack([train.corpus, hold_corpus]),
        embedding=np.vstack([train.embedding, hold_embedding]),
    )
    return combined, holdout_indices


def _fit_predict_full(
    training_config: Mapping[str, Any],
    train: RouterData,
    feature_path: Path,
    candidate_id: str,
    replica_seed: int,
    model_seed: int,
    inner_folds_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = _candidate_by_id(candidate_id)
    combined, holdout_indices = _combined_data(train, feature_path)
    train_indices = np.arange(len(train.query_ids), dtype=np.int64)
    if spec.model_kind in {"late_fusion", "late_fusion_pca"}:
        final_raw, inner_raw, metadata = _fit_predict_late_fusion(
            training_config,
            combined,
            train_indices,
            holdout_indices,
            split_seed=replica_seed,
            outer_fold_id=0,
            model_seed=model_seed,
            use_pca_base=spec.model_kind == "late_fusion_pca",
        )
    else:
        inner_folds = make_group_stratified_folds(
            train,
            n_splits=inner_folds_count,
            seed=replica_seed + 100,
        )
        inner_raw = np.full(len(train.query_ids), np.nan, dtype=np.float64)
        inner_metadata: list[dict[str, Any]] = []
        for fold_id, (inner_train, inner_validation) in enumerate(inner_folds):
            predicted, metadata = _fit_predict_candidate(
                training_config,
                train,
                inner_train,
                inner_validation,
                spec,
                model_seed=model_seed,
                transform_seed=replica_seed + 10000 + fold_id,
            )
            inner_raw[inner_validation] = predicted
            inner_metadata.append(metadata)
        if not np.isfinite(inner_raw).all():
            raise RuntimeError("Full-fit calibration OOF predictions are incomplete")
        final_raw, final_metadata = _fit_predict_candidate(
            training_config,
            combined,
            train_indices,
            holdout_indices,
            spec,
            model_seed=model_seed,
            transform_seed=replica_seed + 20000,
        )
        metadata = {"inner_fits": inner_metadata, "full_fit": final_metadata}
    calibration = _fit_affine_calibration(inner_raw, train.gap)
    calibrated = calibration.predict(final_raw)
    if calibrated.shape != (len(holdout_indices),) or not np.isfinite(calibrated).all():
        raise RuntimeError("Full-fit holdout predictions are invalid")
    return calibrated, {
        "replica_seed": replica_seed,
        "model_seed": model_seed,
        "calibration": {"intercept": calibration.intercept, "slope": calibration.slope},
        "fit": metadata,
    }


def fit(config: Mapping[str, Any], router: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    run_dir = _run_dir(config)
    prediction_path = run_dir / "holdout_predictions.npz"
    freeze_path = run_dir / "prediction_freeze.json"
    if freeze_path.is_file():
        frozen = _read_json(freeze_path)
        if _sha256(prediction_path) != frozen["predictions_sha256"]:
            raise ValueError("Frozen holdout predictions changed")
        return frozen
    connection = _connect(run_dir)
    try:
        successful = int(
            connection.execute("SELECT COUNT(*) FROM outcomes WHERE status='success'").fetchone()[0]
        )
    finally:
        connection.close()
    if successful:
        raise RuntimeError("Full-fit prediction freeze must precede holdout generation labels")
    feature_path = _build_features(config, router)
    predictions: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    replicas = [int(value) for value in config["full_fit"]["replicas"]]
    model_seed = int(config["full_fit"]["model_seed"])
    inner_folds = int(config["full_fit"]["folds"])
    for view in config["training_views"]:
        view_id = str(view["id"])
        training_config = _load_yaml(_resolve(str(view["config"])))
        train, validation = load_frozen_data(training_config)
        if len(train.query_ids) != int(view["queries"]):
            raise ValueError(f"Training query count differs for {view_id}")
        for candidate_id in config["candidates"]:
            key = f"{view_id}__{candidate_id}"
            replica_predictions: list[np.ndarray] = []
            replica_metadata: list[dict[str, Any]] = []
            for replica_seed in replicas:
                print(f"full-fit {key} replica={replica_seed}", flush=True)
                predicted, fit_metadata = _fit_predict_full(
                    training_config,
                    train,
                    feature_path,
                    str(candidate_id),
                    replica_seed,
                    model_seed,
                    inner_folds,
                )
                replica_predictions.append(predicted)
                replica_metadata.append(fit_metadata)
            predictions[f"prediction__{key}"] = np.mean(
                np.stack(replica_predictions, axis=0), axis=0
            )
            metadata[key] = {
                "training_validation": validation,
                "replicas": replica_metadata,
            }
    with np.load(feature_path, allow_pickle=False) as stored:
        predictions["query_ids"] = np.asarray(stored["query_ids"], dtype=np.str_)
        predictions["group_ids"] = np.asarray(stored["group_ids"], dtype=np.str_)
    np.savez_compressed(prediction_path, **predictions)
    frozen = {
        "protocol_id": config["protocol"]["id"],
        "status": "frozen_before_generation_labels",
        "protocol_config_sha256": _sha256(config_path),
        "features_sha256": _sha256(feature_path),
        "predictions": _relative(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "strategies": sorted(metadata),
        "strategy_count": len(metadata),
        "query_count": int(config["sealed_confirmation"]["expected_queries"]),
        "fit_metadata": metadata,
        "holdout_outcome_rows_read_during_fit": 0,
        "official_final_holdout_rows_read": 0,
        "frozen_at_unix": time.time(),
    }
    _write_json(freeze_path, frozen)
    return frozen


def _runtime_phase28_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _load_yaml(BASE_PHASE28_CONFIG)
    runtime["protocol"] = copy.deepcopy(dict(config["protocol"]))
    runtime["generation"] = copy.deepcopy(dict(config["generation"]))
    runtime["outputs"] = copy.deepcopy(dict(config["outputs"]))
    return runtime


def generate(
    config: Mapping[str, Any], router: Mapping[str, Any], *, maximum: int | None, workers: int | None
) -> dict[str, Any]:
    run_dir = _run_dir(config)
    if not (run_dir / "prediction_freeze.json").is_file():
        raise FileNotFoundError("Freeze full-fit holdout predictions before generation")
    result = generate_phase28(
        _runtime_phase28_config(config),
        router,
        max_new_calls=maximum,
        workers=workers,
    )
    _write_json(run_dir / "generation_status.json", status(config))
    return result


def _bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    ordered = sorted(by_group)
    sums = np.asarray([np.sum(values[by_group[group]]) for group in ordered])
    counts = np.asarray([len(by_group[group]) for group in ordered], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        sampled = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        estimates[start:stop] = np.sum(sums[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _holdout_data(config: Mapping[str, Any]) -> tuple[RouterData, list[dict[str, Any]]]:
    run_dir = _run_dir(config)
    connection = _connect(run_dir)
    try:
        labels = _completed_labels(connection)
    finally:
        connection.close()
    expected = int(config["sealed_confirmation"]["expected_queries"])
    if len(labels) != expected:
        raise RuntimeError(f"Holdout labels are incomplete: {len(labels)}/{expected}")
    by_id = {str(row["query_id"]): row for row in labels}
    feature_path = run_dir / "holdout_features.npz"
    with np.load(feature_path, allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        lexical = np.asarray(stored["lexical"], dtype=np.float64)
        corpus = np.asarray(stored["dense"], dtype=np.float64)
        embedding = np.asarray(stored["embedding"], dtype=np.float64)
    ordered = [by_id[str(query_id)] for query_id in query_ids]
    bm25 = np.asarray([row["bm25_mean_f1"] for row in ordered], dtype=np.float64)
    dense = np.asarray([row["dense_mean_f1"] for row in ordered], dtype=np.float64)
    repeat_values = np.asarray(
        [[row["bm25_repeat_f1"], row["dense_repeat_f1"]] for row in ordered],
        dtype=np.float64,
    )
    gap = bm25 - dense
    strata = np.where(gap > TIE_ATOL, "bm25_winner", np.where(gap < -TIE_ATOL, "dense_winner", "exact_tie"))
    data = RouterData(
        query_ids=query_ids,
        group_ids=group_ids,
        query_ranks=np.arange(expected, dtype=np.int64),
        bm25=bm25,
        dense=dense,
        gap=gap,
        strata=np.asarray(strata, dtype=np.str_),
        repeat_values=repeat_values,
        lexical=lexical,
        corpus=corpus,
        embedding=embedding,
    )
    return data, ordered


def _strategy_parts(strategy: str) -> tuple[str, str]:
    view, candidate = strategy.split("__", 1)
    return view, candidate


def evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    prediction_freeze = _read_json(run_dir / "prediction_freeze.json")
    prediction_path = _resolve(str(prediction_freeze["predictions"]))
    if _sha256(prediction_path) != prediction_freeze["predictions_sha256"]:
        raise ValueError("Frozen predictions changed before evaluation")
    data, label_rows = _holdout_data(config)
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    bootstrap_resamples = int(config["evaluation"]["bootstrap_resamples"])
    best_fixed_action = "bm25" if float(np.mean(data.bm25)) > float(np.mean(data.dense)) else "dense"
    best_fixed = data.bm25 if best_fixed_action == "bm25" else data.dense
    complexity = {
        name: index for index, name in enumerate(config["selection"]["lower_complexity_order"])
    }
    results: list[dict[str, Any]] = []
    utilities: dict[str, np.ndarray] = {}
    score_arrays: dict[str, np.ndarray] = {}
    with np.load(prediction_path, allow_pickle=False) as stored:
        if not np.array_equal(np.asarray(stored["query_ids"], dtype=np.str_), data.query_ids):
            raise ValueError("Prediction and outcome query order differs")
        for strategy in prediction_freeze["strategies"]:
            scores = np.asarray(stored[f"prediction__{strategy}"], dtype=np.float64)
            score_arrays[strategy] = scores
            metrics = policy_metrics(
                data,
                scores,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            switches = scores > float(config["evaluation"]["fixed_threshold"])
            router_utility = np.where(switches, data.bm25, data.dense)
            utilities[strategy] = router_utility
            difference_best = router_utility - best_fixed
            ci_best = _bootstrap_ci(
                difference_best,
                data.group_ids,
                seed=bootstrap_seed + 101,
                resamples=bootstrap_resamples,
            )
            gain_best = float(np.mean(difference_best))
            ratio = metrics["harmful_to_beneficial_mass_ratio"]
            calibration = metrics["calibration"]
            deploy = config["selection"]["deployable_gate"]
            deploy_checks = {
                "practical_gain_over_best_fixed": gain_best
                >= float(deploy["practical_gain_over_best_fixed_minimum"]),
                "ci_lower_over_best_fixed": ci_best[0]
                > float(deploy["grouped_bootstrap_ci_lower_over_best_fixed_must_exceed"]),
                "harmful_to_beneficial_mass_ratio": ratio is not None
                and float(ratio) <= float(deploy["harmful_to_beneficial_mass_ratio_maximum"]),
                "switch_coverage": float(deploy["switch_coverage_minimum"])
                <= float(metrics["switch_coverage"])
                <= float(deploy["switch_coverage_maximum"]),
                "calibration_slope": float(deploy["calibration_slope_minimum"])
                <= float(calibration["slope"])
                <= float(deploy["calibration_slope_maximum"]),
                "top_decile_realized_gap": float(calibration["top_decile_realized_gap"]) > 0.0,
            }
            research = config["selection"]["research_shortlist_if_no_deployable_candidate"]
            research_checks = {
                "gain_over_best_fixed": gain_best >= float(research["gain_over_best_fixed_minimum"]),
                "ci_lower_over_best_fixed": ci_best[0]
                > float(research["grouped_bootstrap_ci_lower_over_best_fixed_must_exceed"]),
                "harmful_to_beneficial_mass_ratio": ratio is not None
                and float(ratio) <= float(research["harmful_to_beneficial_mass_ratio_maximum"]),
                "switch_coverage": float(metrics["switch_coverage"])
                <= float(research["switch_coverage_maximum"]),
                "non_tie_auc": metrics["non_tie_auc"] is not None
                and float(metrics["non_tie_auc"]) >= float(research["non_tie_auc_minimum"]),
            }
            view, candidate = _strategy_parts(strategy)
            results.append(
                {
                    "strategy": strategy,
                    "view": view,
                    "candidate_id": candidate,
                    "queries": len(data.query_ids),
                    "fixed_bm25_mean_f1": metrics["fixed_bm25_mean_f1"],
                    "fixed_dense_mean_f1": metrics["fixed_dense_mean_f1"],
                    "best_fixed_action": best_fixed_action,
                    "best_fixed_mean_f1": float(np.mean(best_fixed)),
                    "oracle_mean_f1": metrics["oracle_mean_f1"],
                    "router_mean_f1": metrics["router_mean_f1"],
                    "gain_over_fixed_dense": metrics["gain_over_fixed_dense"],
                    "gain_over_best_fixed": gain_best,
                    "gain_over_best_fixed_ci95_lower": ci_best[0],
                    "gain_over_best_fixed_ci95_upper": ci_best[1],
                    "switch_coverage": metrics["switch_coverage"],
                    "beneficial_switches": metrics["beneficial_switches"],
                    "neutral_switches": metrics["neutral_switches"],
                    "harmful_switches": metrics["harmful_switches"],
                    "harmful_to_beneficial_mass_ratio": ratio,
                    "oracle_recovery": metrics["oracle_recovery"],
                    "non_tie_auc": metrics["non_tie_auc"],
                    "gap_spearman": metrics["gap_spearman"],
                    "calibration_slope": calibration["slope"],
                    "top_decile_realized_gap": calibration["top_decile_realized_gap"],
                    "deployable_gate_passed": all(deploy_checks.values()),
                    "deployable_failed_checks": [name for name, ok in deploy_checks.items() if not ok],
                    "research_gate_passed": all(research_checks.values()),
                    "research_failed_checks": [name for name, ok in research_checks.items() if not ok],
                    "metrics": metrics,
                    "complexity_rank": complexity[candidate],
                }
            )
    results.sort(
        key=lambda row: (
            bool(row["deployable_gate_passed"]),
            float(row["gain_over_best_fixed_ci95_lower"]),
            -float(row["harmful_to_beneficial_mass_ratio"] or math.inf),
            -int(row["complexity_rank"]),
        ),
        reverse=True,
    )
    deployable = [row for row in results if row["deployable_gate_passed"]]
    research = [row for row in results if row["research_gate_passed"]]
    if deployable:
        selected = deployable
        recommendation = "ADVANCE_TO_OFFICIAL_FINAL_VALIDATION"
    elif research:
        selected = research
        recommendation = "RESEARCH_SHORTLIST_ONLY"
    else:
        selected = results[:2]
        recommendation = "NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY"
    top_strategy = str(selected[0]["strategy"])
    paired: list[dict[str, Any]] = []
    for row in results:
        strategy = str(row["strategy"])
        difference = utilities[top_strategy] - utilities[strategy]
        ci = _bootstrap_ci(
            difference,
            data.group_ids,
            seed=bootstrap_seed + 202,
            resamples=bootstrap_resamples,
        )
        paired.append(
            {
                "reference_strategy": top_strategy,
                "comparison_strategy": strategy,
                "mean_router_f1_difference": float(np.mean(difference)),
                "difference_ci95": ci,
            }
        )

    csv_path = run_dir / "holdout_candidate_comparison.csv"
    csv_fields = [
        key for key in results[0] if key not in {"metrics", "complexity_rank"}
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            projected = {key: row[key] for key in csv_fields}
            projected["deployable_failed_checks"] = ";".join(projected["deployable_failed_checks"])
            projected["research_failed_checks"] = ";".join(projected["research_failed_checks"])
            writer.writerow(projected)
    labels_path = run_dir / "holdout_query_metrics.jsonl.gz"
    with gzip.open(labels_path, "wt", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(label_rows):
            handle.write(
                json.dumps(
                    {
                        "query_id": str(data.query_ids[index]),
                        "group_id": str(data.group_ids[index]),
                        "bm25_repeat_f1": row["bm25_repeat_f1"],
                        "dense_repeat_f1": row["dense_repeat_f1"],
                        "bm25_mean_f1": float(data.bm25[index]),
                        "dense_mean_f1": float(data.dense[index]),
                        "gap": float(data.gap[index]),
                        "predicted_gaps": {
                            strategy: float(score_arrays[strategy][index])
                            for strategy in prediction_freeze["strategies"]
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    payload = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "holdout_role": "candidate_selection_not_final_test",
        "query_count": len(data.query_ids),
        "label_counts": {
            "bm25_winner": int(np.sum(data.gap > TIE_ATOL)),
            "dense_winner": int(np.sum(data.gap < -TIE_ATOL)),
            "exact_tie": int(np.sum(np.abs(data.gap) <= TIE_ATOL)),
        },
        "best_fixed_action": best_fixed_action,
        "fixed_bm25_mean_f1": float(np.mean(data.bm25)),
        "fixed_dense_mean_f1": float(np.mean(data.dense)),
        "best_fixed_mean_f1": float(np.mean(best_fixed)),
        "oracle_mean_f1": float(np.mean(np.maximum(data.bm25, data.dense))),
        "recommendation": recommendation,
        "selected_strategies": [str(row["strategy"]) for row in selected],
        "candidate_results": results,
        "paired_comparisons_to_top_selected": paired,
        "artifacts": {
            "comparison_csv": {"path": _relative(csv_path), "sha256": _sha256(csv_path)},
            "query_metrics": {"path": _relative(labels_path), "sha256": _sha256(labels_path)},
            "prediction_freeze": {
                "path": _relative(run_dir / "prediction_freeze.json"),
                "sha256": _sha256(run_dir / "prediction_freeze.json"),
            },
        },
        "estimated_generation_cost_usd": status(config)["estimated_generation_cost_usd"],
        "confirmation_rows_read": len(data.query_ids),
        "official_final_holdout_rows_read": 0,
        "holdout_reuse_for_retuning": "forbidden",
        "completed_at_unix": time.time(),
    }
    result_path = run_dir / "holdout_selection_results.json"
    _write_json(result_path, payload)
    return payload


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    database = run_dir / "state.sqlite3"
    if not database.is_file():
        return {"protocol_id": config["protocol"]["id"], "frozen": False}
    connection = _connect(run_dir)
    try:
        counts = {
            f"{action}:{state}": int(count)
            for action, state, count in connection.execute(
                "SELECT action, status, COUNT(*) FROM outcomes GROUP BY action, status"
            )
        }
        queries_by_action = {
            str(action): int(count)
            for action, count in connection.execute(
                "SELECT action, COUNT(DISTINCT query_id) FROM outcomes GROUP BY action"
            )
        }
        spent = _spent_generation_cost(connection)
    finally:
        connection.close()
    return {
        "protocol_id": config["protocol"]["id"],
        "frozen": (run_dir / "holdout_open_manifest.json").is_file(),
        "queries_by_action": queries_by_action,
        "outcome_counts": counts,
        "estimated_generation_cost_usd": spent,
        "features_complete": (run_dir / "holdout_features.npz").is_file(),
        "predictions_frozen": (run_dir / "prediction_freeze.json").is_file(),
        "evaluation_complete": (run_dir / "holdout_selection_results.json").is_file(),
        "official_final_holdout_rows_read": 0,
    }


def main() -> int:
    args = _arguments()
    config_path = _config_path(args)
    config = _load_yaml(config_path)
    _validate_protocol(config, config_path)
    router = _load_router_config(ROUTER_CONFIG)
    if args.stage == "freeze":
        result = freeze(config, config_path)
    elif args.stage == "retrieve":
        if not (_run_dir(config) / "holdout_open_manifest.json").is_file():
            raise FileNotFoundError("Run --stage freeze before retrieval")
        result = retrieve(config, router, action=args.retrieval_action)
    elif args.stage == "fit":
        result = fit(config, router, config_path)
    elif args.stage == "generate":
        result = generate(
            config,
            router,
            maximum=args.max_new_calls,
            workers=args.workers,
        )
    elif args.stage == "evaluate":
        result = evaluate(config)
    else:
        result = status(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
