#!/usr/bin/env python3
"""Complete non-gating Phase 2.10 fold and coefficient reporting.

This audit only consumes the frozen T2 training view and saved Phase 2.10 OOF
predictions.  It does not read or evaluate the natural or official holdouts, and
it never changes the formal Phase 2.10 decision.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_router_phase27_model_audit as phase27
from scripts import run_router_phase30_scope_incremental as phase30


DEFAULT_OUTPUT = phase30.RUN_DIR / "postrun_reporting_audit.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _slice_data(data: phase27.RouterData, indices: np.ndarray) -> phase27.RouterData:
    return replace(
        data,
        query_ids=data.query_ids[indices],
        group_ids=data.group_ids[indices],
        query_ranks=data.query_ranks[indices],
        bm25=data.bm25[indices],
        dense=data.dense[indices],
        gap=data.gap[indices],
        strata=data.strata[indices],
        repeat_values=data.repeat_values[indices],
        lexical=data.lexical[indices],
        corpus=data.corpus[indices],
        embedding=data.embedding[indices],
    )


def _fit_standardized_scope_coefficient(
    data: phase27.RouterData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    spec: phase27.CandidateSpec,
    *,
    transform_seed: int,
    calibration: dict[str, float],
    expected_prediction: np.ndarray,
) -> dict[str, Any]:
    train_matrix, validation_matrix, transform = phase27._transform_features(
        data,
        train_indices,
        validation_indices,
        spec,
        transform_seed=transform_seed,
    )
    model = Ridge(alpha=10.0)
    model.fit(train_matrix, data.gap[train_indices])
    raw_prediction = model.predict(validation_matrix)
    calibrated_prediction = (
        float(calibration["intercept"])
        + float(calibration["slope"]) * raw_prediction
    )
    maximum_difference = float(
        np.max(np.abs(calibrated_prediction - expected_prediction))
    )
    if not np.allclose(
        calibrated_prediction,
        expected_prediction,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Post-run outer refit does not reproduce saved OOF predictions: "
            f"maximum absolute difference={maximum_difference}"
        )
    raw_coefficient = float(model.coef_[-1 if spec.feature_kind == "structured" else 30])
    return {
        "standardized_scope_coefficient_raw_ridge": raw_coefficient,
        "standardized_scope_coefficient_after_affine_calibration": (
            raw_coefficient * float(calibration["slope"])
        ),
        "ridge_intercept_raw": float(model.intercept_),
        "calibration_intercept": float(calibration["intercept"]),
        "calibration_slope": float(calibration["slope"]),
        "outer_prediction_reproduction_max_abs_difference": maximum_difference,
        "transform": transform,
    }


def _coefficient_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "folds": int(len(values)),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_folds": int(np.sum(values > 0.0)),
        "negative_folds": int(np.sum(values < 0.0)),
        "zero_folds": int(np.sum(values == 0.0)),
    }


def main() -> int:
    arguments = _arguments()
    _, data, _ = phase30._load_training()
    scope = phase30._load_scope(data)
    scope_data = replace(
        data,
        lexical=np.empty((len(data.query_ids), 0), dtype=np.float64),
        corpus=scope.reshape(-1, 1),
    )
    augmented_data = replace(data, corpus=np.column_stack([data.corpus, scope]))
    if data.structured.shape[1] != 30 or augmented_data.structured.shape[1] != 31:
        raise RuntimeError("Unexpected structured feature dimensions")

    scope_spec = phase27.CandidateSpec(
        phase30.SCOPE_ONLY_ID,
        "P30",
        "structured",
        "ridge",
    )
    augmented_spec = phase27.CandidateSpec(
        phase30.AUGMENTED_ID,
        "P30",
        "pca_structured",
        "ridge",
        pca_dim=32,
    )

    metrics = phase30._read_json(phase30.OOF_RESULT_PATH)
    decision_sha256 = phase30._sha256(phase30.DECISION_PATH)
    seed_metrics = {int(row["split_seed"]): row for row in metrics["split_seeds"]}
    with np.load(phase30.OOF_PREDICTION_PATH, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}

    best_fixed_action, _ = phase30._best_fixed(data)
    fold_rows: list[dict[str, Any]] = []
    scope_coefficients: list[dict[str, Any]] = []
    augmented_coefficients: list[dict[str, Any]] = []
    candidate_ids = (
        phase30.BASELINE_ID,
        phase30.SCOPE_ONLY_ID,
        phase30.AUGMENTED_ID,
    )
    fit_key = {
        phase30.SCOPE_ONLY_ID: "scope_fit",
        phase30.AUGMENTED_ID: "augmented_fit",
    }

    for seed in phase30.SPLIT_SEEDS:
        fold_ids = arrays[f"fold_id__{seed}"].astype(np.int64)
        for outer_fold in sorted(int(value) for value in np.unique(fold_ids)):
            validation_indices = np.flatnonzero(fold_ids == outer_fold)
            train_indices = np.flatnonzero(fold_ids != outer_fold)
            fold_data = _slice_data(data, validation_indices)
            # Keep the globally frozen best action (BM25), even if a single fold's
            # finite-sample mean would select a different action.
            fold_best_fixed = (
                fold_data.bm25
                if best_fixed_action == "BM25"
                else fold_data.dense
            )
            scores = {
                candidate_id: arrays[
                    f"prediction__{candidate_id}__{seed}"
                ][validation_indices]
                for candidate_id in candidate_ids
            }
            candidate_metrics = {
                candidate_id: phase30._point_policy_metrics(
                    fold_data,
                    prediction,
                    best_fixed_action=best_fixed_action,
                    best_fixed=fold_best_fixed,
                )
                for candidate_id, prediction in scores.items()
            }
            paired_utility = (
                phase30._utility(fold_data, scores[phase30.AUGMENTED_ID])
                - phase30._utility(fold_data, scores[phase30.BASELINE_ID])
            )
            fold_rows.append(
                {
                    "split_seed": seed,
                    "outer_fold": outer_fold,
                    "train_queries": int(len(train_indices)),
                    "validation_queries": int(len(validation_indices)),
                    "validation_groups": int(
                        len(np.unique(fold_data.group_ids))
                    ),
                    "best_fixed_action_frozen_from_full_T2": best_fixed_action,
                    "candidate_metrics": candidate_metrics,
                    "M3S_minus_M3_policy_utility": float(
                        np.mean(paired_utility)
                    ),
                }
            )

            seed_row = seed_metrics[seed]
            for candidate_id, candidate_data, spec, destination in (
                (
                    phase30.SCOPE_ONLY_ID,
                    scope_data,
                    scope_spec,
                    scope_coefficients,
                ),
                (
                    phase30.AUGMENTED_ID,
                    augmented_data,
                    augmented_spec,
                    augmented_coefficients,
                ),
            ):
                fold_fit = seed_row[fit_key[candidate_id]]["folds"][outer_fold]
                calibration = fold_fit["calibration"]
                coefficient = _fit_standardized_scope_coefficient(
                    candidate_data,
                    train_indices,
                    validation_indices,
                    spec,
                    transform_seed=seed + 20_000 + outer_fold,
                    calibration=calibration,
                    expected_prediction=scores[candidate_id],
                )
                destination.append(
                    {
                        "split_seed": seed,
                        "outer_fold": outer_fold,
                        **coefficient,
                    }
                )

    raw_key = "standardized_scope_coefficient_raw_ridge"
    calibrated_key = "standardized_scope_coefficient_after_affine_calibration"
    result = {
        "protocol_id": phase30.PROTOCOL_ID,
        "audit_kind": "non_gating_postrun_reporting_completion",
        "status": "complete",
        "formal_decision_unchanged": True,
        "formal_decision": phase30._read_json(phase30.DECISION_PATH)["decision"],
        "formal_decision_sha256": decision_sha256,
        "data_boundary": {
            "training_view": "T2_winner3000",
            "queries": int(len(data.query_ids)),
            "groups": int(len(np.unique(data.group_ids))),
            "natural_outcome_rows_parsed": 0,
            "natural_outcome_values_read": 0,
            "official_final_holdout_rows": 0,
            "external_calls": 0,
        },
        "coefficient_definition": (
            "Ridge coefficient on fold-training-standardized scope; the calibrated "
            "coefficient multiplies it by that outer fold's non-negative affine slope"
        ),
        "outer_fold_policy_metrics": fold_rows,
        "scope_only_coefficients": scope_coefficients,
        "scope_only_coefficient_summary": {
            "raw": _coefficient_summary(scope_coefficients, raw_key),
            "calibrated": _coefficient_summary(scope_coefficients, calibrated_key),
        },
        "M3S_scope_coefficients": augmented_coefficients,
        "M3S_scope_coefficient_summary": {
            "raw": _coefficient_summary(augmented_coefficients, raw_key),
            "calibrated": _coefficient_summary(augmented_coefficients, calibrated_key),
        },
        "source_artifacts": {
            "internal_metrics": phase30._relative(phase30.OOF_RESULT_PATH),
            "internal_metrics_sha256": phase30._sha256(phase30.OOF_RESULT_PATH),
            "internal_oof_predictions": phase30._relative(
                phase30.OOF_PREDICTION_PATH
            ),
            "internal_oof_predictions_sha256": phase30._sha256(
                phase30.OOF_PREDICTION_PATH
            ),
            "scope_features": phase30._relative(phase30.FEATURE_PATH),
            "scope_features_sha256": phase30._sha256(phase30.FEATURE_PATH),
        },
    }
    phase30._write_json(arguments.output, result)
    if phase30._sha256(phase30.DECISION_PATH) != decision_sha256:
        raise RuntimeError("Formal Phase 2.10 decision changed during reporting audit")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
