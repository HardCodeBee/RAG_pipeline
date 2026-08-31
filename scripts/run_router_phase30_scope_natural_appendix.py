#!/usr/bin/env python3
"""Run the user-authorized Phase 2.10 natural-2000 evaluation appendix.

The prediction stage fits only T2 and freezes all natural predictions before
the evaluation stage parses any natural outcomes.  This appendix is explicitly
post-selection and cannot revise the formal Phase 2.10 internal stop decision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_experiments import modeling as phase27  # noqa: E402
from scripts import run_router_phase30_scope_incremental as phase30  # noqa: E402


PROTOCOL_ID = "hotpotqa_bd_router_phase30_scope_natural_appendix_v1"
RUN_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/"
    "phase30_scope_natural_appendix_v1"
)
CONFIG_PATH = PROJECT_ROOT / (
    "analysis/hotpotqa_router/phase30_scope_natural_appendix_config.yaml"
)
SCRIPT_PATH = Path(__file__).resolve()
NATURAL_HISTORICAL_FEATURES = phase30.PHASE28B_DIR / "holdout_features.npz"
NATURAL_SCOPE_FEATURES = phase30.PHASE29_DIR / "natural_features.npz"
NATURAL_OUTCOMES = phase30.PHASE28B_DIR / "holdout_query_metrics.jsonl.gz"
HISTORICAL_NATURAL_PREDICTIONS = phase30.PHASE28B_DIR / "holdout_predictions.npz"
FORMAL_DECISION = phase30.DECISION_PATH
PREFLIGHT_PATH = RUN_DIR / "appendix_preflight.json"
PREDICTIONS_PATH = RUN_DIR / "natural_predictions.npz"
PREDICTION_FREEZE_PATH = RUN_DIR / "natural_prediction_freeze.json"
EVALUATION_PATH = RUN_DIR / "natural_evaluation.json"
COMPARISON_PATH = RUN_DIR / "natural_comparison.csv"

CANDIDATES = (
    phase30.SCOPE_ONLY_ID,
    phase30.BASELINE_ID,
    phase30.AUGMENTED_ID,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "predict", "evaluate", "run", "status"),
    )
    return parser.parse_args()


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value["protocol"]["id"] != PROTOCOL_ID:
        raise ValueError("Appendix protocol id differs")
    if value["interpretation"]["official_final_holdout_rows_allowed"] != 0:
        raise ValueError("Official final holdout must remain sealed")
    if tuple(value["training"]["replica_seeds"]) != phase30.SPLIT_SEEDS:
        raise ValueError("Replica seeds differ from frozen Phase 2.10")
    if tuple(row["id"] for row in value["candidates"]) != CANDIDATES:
        raise ValueError("Candidate set or order differs")
    if float(value["policy"]["fixed_threshold"]) != phase30.FIXED_THRESHOLD:
        raise ValueError("Threshold differs from frozen Phase 2.10")
    return value


def _input_paths() -> dict[str, Path]:
    return {
        "formal_phase30_decision": FORMAL_DECISION,
        "T2_config": phase30.INPUTS["t2_config"][0],
        "T2_features": phase30.INPUTS["t2_features"][0],
        "T2_scope": phase30.FEATURE_PATH,
        "natural_historical_features": NATURAL_HISTORICAL_FEATURES,
        "natural_scope_features": NATURAL_SCOPE_FEATURES,
        "natural_historical_predictions": HISTORICAL_NATURAL_PREDICTIONS,
        "natural_outcomes": NATURAL_OUTCOMES,
    }


def _load_natural_features() -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(NATURAL_HISTORICAL_FEATURES, allow_pickle=False) as stored:
        historical = {
            "query_ids": np.asarray(stored["query_ids"], dtype=np.str_),
            "group_ids": np.asarray(stored["group_ids"], dtype=np.str_),
            "lexical": np.asarray(stored["lexical"], dtype=np.float64),
            "corpus": np.asarray(stored["dense"], dtype=np.float64),
            "embedding": np.asarray(stored["embedding"], dtype=np.float64),
        }
    with np.load(NATURAL_SCOPE_FEATURES, allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        names = np.asarray(stored["feature_names"], dtype=np.str_)
        matrix = np.asarray(stored["matrix"], dtype=np.float64)
    indices = np.flatnonzero(names == phase30.FEATURE_NAME)
    if len(indices) != 1:
        raise ValueError("Natural scope feature is missing or duplicated")
    if not np.array_equal(query_ids, historical["query_ids"]):
        raise ValueError("Natural feature query order differs")
    if not np.array_equal(group_ids, historical["group_ids"]):
        raise ValueError("Natural feature group order differs")
    scope = matrix[:, int(indices[0])]
    count = len(query_ids)
    if count != 2_000 or len(np.unique(group_ids)) != 2_000:
        raise ValueError("Natural feature row/group count differs")
    if historical["lexical"].shape != (count, 17):
        raise ValueError("Natural lexical feature dimensions differ")
    if historical["corpus"].shape != (count, 13):
        raise ValueError("Natural corpus feature dimensions differ")
    if historical["embedding"].shape != (count, 384):
        raise ValueError("Natural embedding dimensions differ")
    numeric_values = (
        historical["lexical"],
        historical["corpus"],
        historical["embedding"],
        scope,
    )
    if not all(np.isfinite(value).all() for value in numeric_values):
        raise ValueError("Natural features contain non-finite values")
    return historical, scope


def _combined_data(
    train: phase27.RouterData,
    natural: dict[str, np.ndarray],
) -> tuple[phase27.RouterData, np.ndarray]:
    count = len(natural["query_ids"])
    natural_indices = np.arange(len(train.query_ids), len(train.query_ids) + count)
    combined = phase27.RouterData(
        query_ids=np.concatenate([train.query_ids, natural["query_ids"]]),
        group_ids=np.concatenate([train.group_ids, natural["group_ids"]]),
        query_ranks=np.concatenate(
            [train.query_ranks, np.arange(count, dtype=np.int64)]
        ),
        bm25=np.concatenate([train.bm25, np.zeros(count)]),
        dense=np.concatenate([train.dense, np.zeros(count)]),
        gap=np.concatenate([train.gap, np.zeros(count)]),
        strata=np.concatenate(
            [train.strata, np.full(count, "exact_tie", dtype=np.str_)]
        ),
        repeat_values=np.concatenate(
            [train.repeat_values, np.zeros((count, 2, 3), dtype=np.float64)]
        ),
        lexical=np.vstack([train.lexical, natural["lexical"]]),
        corpus=np.vstack([train.corpus, natural["corpus"]]),
        embedding=np.vstack([train.embedding, natural["embedding"]]),
    )
    return combined, natural_indices.astype(np.int64)


def _candidate_data(
    train: phase27.RouterData,
    natural: dict[str, np.ndarray],
    train_scope: np.ndarray,
    natural_scope: np.ndarray,
    candidate_id: str,
) -> tuple[phase27.RouterData, dict[str, np.ndarray], phase27.CandidateSpec]:
    if candidate_id == phase30.SCOPE_ONLY_ID:
        train_data = replace(
            train,
            lexical=np.empty((len(train.query_ids), 0), dtype=np.float64),
            corpus=train_scope.reshape(-1, 1),
        )
        natural_data = {
            **natural,
            "lexical": np.empty((len(natural_scope), 0), dtype=np.float64),
            "corpus": natural_scope.reshape(-1, 1),
        }
        spec = phase27.CandidateSpec(candidate_id, "P30", "structured", "ridge")
    elif candidate_id == phase30.BASELINE_ID:
        train_data = train
        natural_data = natural
        spec = phase27.CandidateSpec(
            candidate_id, "M3", "pca_structured", "ridge", pca_dim=32
        )
    elif candidate_id == phase30.AUGMENTED_ID:
        train_data = replace(
            train,
            corpus=np.column_stack([train.corpus, train_scope]),
        )
        natural_data = {
            **natural,
            "corpus": np.column_stack([natural["corpus"], natural_scope]),
        }
        spec = phase27.CandidateSpec(
            candidate_id, "P30", "pca_structured", "ridge", pca_dim=32
        )
    else:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    return train_data, natural_data, spec


def _fit_predict_full(
    training_config: dict[str, Any],
    train: phase27.RouterData,
    natural: dict[str, np.ndarray],
    spec: phase27.CandidateSpec,
    *,
    replica_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    combined, natural_indices = _combined_data(train, natural)
    train_indices = np.arange(len(train.query_ids), dtype=np.int64)
    inner_folds = phase27.make_group_stratified_folds(
        train,
        n_splits=4,
        seed=replica_seed + 100,
    )
    inner_oof = np.full(len(train.query_ids), np.nan, dtype=np.float64)
    inner_metadata: list[dict[str, Any]] = []
    for fold_id, (inner_train, inner_validation) in enumerate(inner_folds):
        predicted, metadata = phase27.fit_predict_candidate(
            training_config,
            train,
            inner_train,
            inner_validation,
            spec,
            model_seed=phase30.MODEL_SEED,
            transform_seed=replica_seed + 10_000 + fold_id,
        )
        inner_oof[inner_validation] = predicted
        inner_metadata.append(metadata)
    if not np.isfinite(inner_oof).all():
        raise RuntimeError("Inner calibration OOF prediction is incomplete")
    raw_natural, full_metadata = phase27.fit_predict_candidate(
        training_config,
        combined,
        train_indices,
        natural_indices,
        spec,
        model_seed=phase30.MODEL_SEED,
        transform_seed=replica_seed + 20_000,
    )
    calibration = phase27.fit_affine_calibration(inner_oof, train.gap)
    calibrated = calibration.predict(raw_natural)
    if calibrated.shape != (2_000,) or not np.isfinite(calibrated).all():
        raise RuntimeError("Natural prediction is incomplete")
    return calibrated, {
        "replica_seed": replica_seed,
        "model_seed": phase30.MODEL_SEED,
        "calibration": {
            "intercept": calibration.intercept,
            "slope": calibration.slope,
        },
        "inner_fits": inner_metadata,
        "full_fit": full_metadata,
    }


def preflight() -> dict[str, Any]:
    config = _read_config()
    phase30._assert_freeze()
    formal = phase30._read_json(FORMAL_DECISION)
    if formal["decision"] != "STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL":
        raise ValueError("Unexpected formal Phase 2.10 decision")
    _, train, _ = phase30._load_training()
    train_scope = phase30._load_scope(train)
    natural, natural_scope = _load_natural_features()
    if len(train.query_ids) != 12_000 or len(np.unique(train.group_ids)) != 11_964:
        raise ValueError("T2 row/group count differs")
    paths = _input_paths()
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "authorization": "user_requested_eval_after_formal_internal_stop",
        "interpretation": config["interpretation"],
        "formal_phase30_decision": formal["decision"],
        "formal_phase30_decision_sha256": phase30._sha256(FORMAL_DECISION),
        "training": {
            "queries": len(train.query_ids),
            "groups": len(np.unique(train.group_ids)),
            "scope_values": len(train_scope),
        },
        "natural_features": {
            "queries": len(natural["query_ids"]),
            "groups": len(np.unique(natural["group_ids"])),
            "scope_values": len(natural_scope),
            "outcome_rows_parsed": 0,
        },
        "inputs": {
            name: {
                "path": phase30._relative(path),
                "bytes": path.stat().st_size,
                "sha256": phase30._sha256(path),
                "access": (
                    "hash_only_before_prediction_freeze"
                    if name == "natural_outcomes"
                    else "read_only"
                ),
            }
            for name, path in paths.items()
        },
        "implementation": {
            "config": phase30._sha256(CONFIG_PATH),
            "runner": phase30._sha256(SCRIPT_PATH),
            "phase27_model_runner": phase30._sha256(
                PROJECT_ROOT / "scripts/run_router_phase27_model_audit.py"
            ),
        },
        "official_final_holdout_rows": 0,
        "external_calls": 0,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    phase30._write_json(PREFLIGHT_PATH, result)
    return result


def _validate_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("Run preflight before prediction")
    value = phase30._read_json(PREFLIGHT_PATH)
    expected_implementation = {
        "config": phase30._sha256(CONFIG_PATH),
        "runner": phase30._sha256(SCRIPT_PATH),
        "phase27_model_runner": phase30._sha256(
            PROJECT_ROOT / "scripts/run_router_phase27_model_audit.py"
        ),
    }
    if value.get("implementation") != expected_implementation:
        raise RuntimeError("Appendix implementation changed after preflight")
    for name, path in _input_paths().items():
        if value["inputs"][name]["sha256"] != phase30._sha256(path):
            raise RuntimeError(f"Appendix input changed after preflight: {name}")
    return value


def predict() -> dict[str, Any]:
    _validate_preflight()
    if EVALUATION_PATH.is_file() and not PREDICTION_FREEZE_PATH.is_file():
        raise RuntimeError("Evaluation exists without a prediction freeze")
    if PREDICTION_FREEZE_PATH.is_file():
        value = phase30._read_json(PREDICTION_FREEZE_PATH)
        if value["predictions_sha256"] != phase30._sha256(PREDICTIONS_PATH):
            raise RuntimeError("Frozen appendix predictions changed")
        return value
    training_config, train, _ = phase30._load_training()
    train_scope = phase30._load_scope(train)
    natural, natural_scope = _load_natural_features()
    arrays: dict[str, np.ndarray] = {
        "query_ids": natural["query_ids"],
        "group_ids": natural["group_ids"],
    }
    fit_metadata: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for candidate_id in CANDIDATES:
        candidate_train, candidate_natural, spec = _candidate_data(
            train,
            natural,
            train_scope,
            natural_scope,
            candidate_id,
        )
        by_seed: list[np.ndarray] = []
        fit_metadata[candidate_id] = []
        for replica_seed in phase30.SPLIT_SEEDS:
            print(f"appendix full-fit {candidate_id} seed={replica_seed}", flush=True)
            prediction, metadata = _fit_predict_full(
                training_config,
                candidate_train,
                candidate_natural,
                spec,
                replica_seed=replica_seed,
            )
            arrays[f"prediction__{candidate_id}__{replica_seed}"] = prediction
            by_seed.append(prediction)
            fit_metadata[candidate_id].append(metadata)
        arrays[f"prediction_mean__{candidate_id}"] = np.mean(
            np.stack(by_seed, axis=0), axis=0
        )

    historical_key = f"prediction__T2_winner3000__{phase30.BASELINE_ID}"
    with np.load(HISTORICAL_NATURAL_PREDICTIONS, allow_pickle=False) as stored:
        historical_query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        historical_m3 = np.asarray(stored[historical_key], dtype=np.float64)
    current_m3 = arrays[f"prediction_mean__{phase30.BASELINE_ID}"]
    if not np.array_equal(historical_query_ids, arrays["query_ids"]):
        raise RuntimeError("Historical M3 natural query order differs")
    m3_maximum_difference = float(np.max(np.abs(current_m3 - historical_m3)))
    if not np.allclose(current_m3, historical_m3, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Fresh full-fit M3 does not reproduce historical natural prediction: "
            f"maximum absolute difference={m3_maximum_difference}"
        )
    phase30._write_npz(PREDICTIONS_PATH, **arrays)
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_natural_outcome_parsing",
        "predictions": phase30._relative(PREDICTIONS_PATH),
        "predictions_sha256": phase30._sha256(PREDICTIONS_PATH),
        "preflight_sha256": phase30._sha256(PREFLIGHT_PATH),
        "candidates": list(CANDIDATES),
        "replica_seeds": list(phase30.SPLIT_SEEDS),
        "queries": 2_000,
        "M3_historical_reproduction": {
            "allclose": True,
            "maximum_absolute_difference": m3_maximum_difference,
            "historical_predictions_sha256": phase30._sha256(
                HISTORICAL_NATURAL_PREDICTIONS
            ),
        },
        "fit_metadata": fit_metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "natural_outcome_rows_parsed_during_prediction": 0,
        "official_final_holdout_rows": 0,
        "external_calls": 0,
    }
    phase30._write_json(PREDICTION_FREEZE_PATH, result)
    return result


def _load_outcomes(natural: dict[str, np.ndarray]) -> phase27.RouterData:
    with gzip.open(NATURAL_OUTCOMES, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 2_000:
        raise RuntimeError("Natural outcome rows are incomplete")
    query_ids = np.asarray([str(row["query_id"]) for row in rows], dtype=np.str_)
    group_ids = np.asarray([str(row["group_id"]) for row in rows], dtype=np.str_)
    if not np.array_equal(query_ids, natural["query_ids"]):
        raise RuntimeError("Natural outcome query order differs from predictions")
    if not np.array_equal(group_ids, natural["group_ids"]):
        raise RuntimeError("Natural outcome group order differs from predictions")
    bm25 = np.asarray([row["bm25_mean_f1"] for row in rows], dtype=np.float64)
    dense = np.asarray([row["dense_mean_f1"] for row in rows], dtype=np.float64)
    gap = bm25 - dense
    repeats = np.asarray(
        [[row["bm25_repeat_f1"], row["dense_repeat_f1"]] for row in rows],
        dtype=np.float64,
    )
    strata = np.where(
        gap > phase27.TIE_ATOL,
        "bm25_winner",
        np.where(gap < -phase27.TIE_ATOL, "dense_winner", "exact_tie"),
    )
    return phase27.RouterData(
        query_ids=query_ids,
        group_ids=group_ids,
        query_ranks=np.arange(len(rows), dtype=np.int64),
        bm25=bm25,
        dense=dense,
        gap=gap,
        strata=np.asarray(strata, dtype=np.str_),
        repeat_values=repeats,
        lexical=natural["lexical"],
        corpus=natural["corpus"],
        embedding=natural["embedding"],
    )


def _candidate_evaluation(
    data: phase27.RouterData,
    scores: np.ndarray,
    *,
    best_action: str,
    best_fixed: np.ndarray,
) -> dict[str, Any]:
    return phase30._point_policy_metrics(
        data,
        scores,
        best_fixed_action=best_action,
        best_fixed=best_fixed,
    )


def _gate(metrics: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    calibration = metrics["calibration"]
    ratio = metrics["harmful_to_beneficial_mass_ratio"]
    checks = {
        "gain_over_best_fixed_at_least_0_01": (
            float(metrics["gain_over_best_fixed"]) >= 0.01
        ),
        "gain_ci95_lower_above_zero": (
            float(metrics["gain_over_best_fixed_ci95"][0]) > 0.0
        ),
        "paired_M3S_minus_M3_positive": (
            float(paired["point_estimate"]) > 0.0
        ),
        "paired_M3S_minus_M3_ci95_lower_above_zero": (
            float(paired["ci95"][0]) > 0.0
        ),
        "harmful_to_beneficial_mass_ratio_at_most_0_5": (
            ratio is not None and float(ratio) <= 0.5
        ),
        "switch_coverage_between_0_01_and_0_5": (
            0.01 <= float(metrics["switch_coverage"]) <= 0.5
        ),
        "calibration_slope_between_0_5_and_1_5": (
            0.5 <= float(calibration["slope"]) <= 1.5
        ),
        "top_decile_realized_gap_positive": (
            float(calibration["top_decile_realized_gap"]) > 0.0
        ),
        "non_tie_auc_at_least_0_55": (
            metrics["non_tie_auc"] is not None
            and float(metrics["non_tie_auc"]) >= 0.55
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def evaluate() -> dict[str, Any]:
    _validate_preflight()
    freeze = predict()
    if freeze["predictions_sha256"] != phase30._sha256(PREDICTIONS_PATH):
        raise RuntimeError("Predictions changed before natural outcome evaluation")
    natural, _ = _load_natural_features()
    data = _load_outcomes(natural)
    best_action, best_fixed = phase30._best_fixed(data)
    with np.load(PREDICTIONS_PATH, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}

    score_sets: dict[str, dict[str, np.ndarray]] = {}
    for candidate_id in CANDIDATES:
        score_sets[candidate_id] = {
            str(seed): arrays[f"prediction__{candidate_id}__{seed}"]
            for seed in phase30.SPLIT_SEEDS
        }
        score_sets[candidate_id]["consensus"] = arrays[
            f"prediction_mean__{candidate_id}"
        ]

    comparisons: dict[str, Any] = {}
    for label in (*[str(seed) for seed in phase30.SPLIT_SEEDS], "consensus"):
        scores = {
            candidate_id: score_sets[candidate_id][label]
            for candidate_id in CANDIDATES
        }
        utilities = {
            candidate_id: phase30._utility(data, prediction)
            for candidate_id, prediction in scores.items()
        }
        paired_values = (
            utilities[phase30.AUGMENTED_ID]
            - utilities[phase30.BASELINE_ID]
        )
        bootstrap_values = {
            f"gain__{candidate_id}": utility - best_fixed
            for candidate_id, utility in utilities.items()
        }
        bootstrap_values["paired"] = paired_values
        intervals = phase30._common_group_bootstrap_cis(
            bootstrap_values,
            data.group_ids,
        )
        candidate_metrics = {
            candidate_id: _candidate_evaluation(
                data,
                prediction,
                best_action=best_action,
                best_fixed=best_fixed,
            )
            for candidate_id, prediction in scores.items()
        }
        for candidate_id in CANDIDATES:
            candidate_metrics[candidate_id]["gain_over_best_fixed_ci95"] = intervals[
                f"gain__{candidate_id}"
            ]
        comparisons[label] = {
            "candidate_metrics": candidate_metrics,
            "M3S_minus_M3": {
                "point_estimate": float(np.mean(paired_values)),
                "ci95": intervals["paired"],
                "action_changes": int(
                    np.sum(
                        (scores[phase30.AUGMENTED_ID] > phase30.FIXED_THRESHOLD)
                        != (scores[phase30.BASELINE_ID] > phase30.FIXED_THRESHOLD)
                    )
                ),
            },
        }

    consensus = comparisons["consensus"]
    gate = _gate(
        consensus["candidate_metrics"][phase30.AUGMENTED_ID],
        consensus["M3S_minus_M3"],
    )
    decision = (
        "ELIGIBLE_FOR_SEPARATELY_FROZEN_OFFICIAL_FINAL_PROTOCOL"
        if gate["passed"]
        else "STOP_NO_POST_SELECTION_NATURAL_STABILITY"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "role": "consumed_post_selection_diagnostic_not_holdout_not_final_test",
        "formal_phase30_internal_decision_unchanged": True,
        "formal_phase30_internal_decision": "STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL",
        "queries": len(data.query_ids),
        "groups": len(np.unique(data.group_ids)),
        "label_counts": {
            "bm25_winner": int(np.sum(data.gap > phase27.TIE_ATOL)),
            "dense_winner": int(np.sum(data.gap < -phase27.TIE_ATOL)),
            "exact_tie": int(np.sum(np.abs(data.gap) <= phase27.TIE_ATOL)),
        },
        "fixed_bm25_mean_f1": float(np.mean(data.bm25)),
        "fixed_dense_mean_f1": float(np.mean(data.dense)),
        "best_fixed_action": best_action,
        "best_fixed_mean_f1": float(np.mean(best_fixed)),
        "oracle_mean_f1": float(np.mean(np.maximum(data.bm25, data.dense))),
        "split_and_consensus_results": comparisons,
        "continuation_gate": gate,
        "decision": decision,
        "prediction_freeze_sha256": phase30._sha256(PREDICTION_FREEZE_PATH),
        "predictions_sha256": phase30._sha256(PREDICTIONS_PATH),
        "natural_outcome_rows_parsed": len(data.query_ids),
        "official_final_holdout_rows": 0,
        "external_calls": 0,
    }
    phase30._write_json(EVALUATION_PATH, result)

    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "candidate_id",
            "router_mean_f1",
            "gain_over_best_fixed",
            "gain_ci95_lower",
            "gain_ci95_upper",
            "switch_coverage",
            "harmful_to_beneficial_mass_ratio",
            "non_tie_auc",
            "gap_spearman",
            "calibration_slope",
            "top_decile_realized_gap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate_id in CANDIDATES:
            metrics = consensus["candidate_metrics"][candidate_id]
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "router_mean_f1": metrics["router_mean_f1"],
                    "gain_over_best_fixed": metrics["gain_over_best_fixed"],
                    "gain_ci95_lower": metrics["gain_over_best_fixed_ci95"][0],
                    "gain_ci95_upper": metrics["gain_over_best_fixed_ci95"][1],
                    "switch_coverage": metrics["switch_coverage"],
                    "harmful_to_beneficial_mass_ratio": metrics[
                        "harmful_to_beneficial_mass_ratio"
                    ],
                    "non_tie_auc": metrics["non_tie_auc"],
                    "gap_spearman": metrics["gap_spearman"],
                    "calibration_slope": metrics["calibration"]["slope"],
                    "top_decile_realized_gap": metrics["calibration"][
                        "top_decile_realized_gap"
                    ],
                }
            )
    return result


def status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "preflight_complete": PREFLIGHT_PATH.is_file(),
        "predictions_frozen": PREDICTION_FREEZE_PATH.is_file(),
        "evaluation_complete": EVALUATION_PATH.is_file(),
        "official_final_holdout_rows": 0,
    }
    if EVALUATION_PATH.is_file():
        evaluation = phase30._read_json(EVALUATION_PATH)
        result["decision"] = evaluation["decision"]
        result["failed_checks"] = evaluation["continuation_gate"]["failed_checks"]
    return result


def main() -> int:
    arguments = _arguments()
    if arguments.stage == "preflight":
        result = preflight()
    elif arguments.stage == "predict":
        result = predict()
    elif arguments.stage == "evaluate":
        result = evaluate()
    elif arguments.stage == "run":
        preflight()
        predict()
        result = evaluate()
    else:
        result = status()
    print(json.dumps(phase30._json_ready(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
