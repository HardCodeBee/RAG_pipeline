"""Run the zero-call B/D Dense-default F1 structural gate on frozen Phase 3 data."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_QUERIES = 4_800
REPEATS = (0, 1, 2)
ACTIONS = ("bm25", "dense")
FOLD_SEED = 20260901
MODEL_SEEDS = (11, 23, 37, 53, 71)
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_RESAMPLES = 10_000
PRACTICAL_GAIN = 0.01
TIE_ATOL = 1e-12
CANDIDATES = (
    ("full_pairwise_xgboost", "full", "pairwise"),
    ("full_gain_xgboost", "full", "gain"),
    ("tier_a_pairwise_xgboost", "tier_a", "pairwise"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        default=(
            "outputs/router/hotpotqa_bd_router_v1/runs/"
            "phase3_f1_pairwise_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/router/hotpotqa_bd_router_v1/runs/"
            "phase26_bd_selective_f1_v1"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def _load_outcomes(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database = run_dir / "state.sqlite3"
    connection = _readonly_connection(database)
    try:
        partition_counts = {
            str(partition): int(count)
            for partition, count in connection.execute(
                "SELECT partition, COUNT(*) FROM sample_rows GROUP BY partition"
            )
        }
        outcome_partition_counts = {
            str(partition): int(count)
            for partition, count in connection.execute(
                """
                SELECT s.partition, COUNT(*)
                FROM sample_rows AS s
                JOIN outcomes AS o ON o.query_id = s.query_id
                GROUP BY s.partition
                """
            )
        }
        if set(outcome_partition_counts) - {"train"}:
            raise ValueError("The source run contains non-training outcome rows")
        rows = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id,
                   o.action, o.repeat_id, o.status, o.payload
            FROM sample_rows AS s
            JOIN outcomes AS o ON o.query_id = s.query_id
            WHERE s.partition = 'train'
              AND s.query_rank < ?
              AND o.action IN ('bm25', 'dense')
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'bm25' THEN 0 ELSE 1 END,
                     o.repeat_id
            """,
            (TRAIN_QUERIES,),
        ).fetchall()
    finally:
        connection.close()

    expected_rows = TRAIN_QUERIES * len(ACTIONS) * len(REPEATS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} B/D rows, found {len(rows)}")

    cells: dict[str, dict[str, Any]] = {}
    for rank, query_id, group_id, action, repeat_id, status, payload in rows:
        if status != "success":
            raise ValueError(f"Incomplete outcome for rank {rank}, action {action}")
        action = str(action)
        repeat_id = int(repeat_id)
        if action not in ACTIONS or repeat_id not in REPEATS:
            raise ValueError("Unexpected action or repeat")
        parsed = json.loads(payload)
        value = float(parsed["metrics"]["normalized_token_f1"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("F1 must be finite and in [0, 1]")
        query_id = str(query_id)
        cell = cells.setdefault(
            query_id,
            {
                "rank": int(rank),
                "query_id": query_id,
                "group_id": str(group_id),
                "values": {name: {} for name in ACTIONS},
            },
        )
        if cell["rank"] != int(rank) or cell["group_id"] != str(group_id):
            raise ValueError("Inconsistent query rank or group")
        if repeat_id in cell["values"][action]:
            raise ValueError("Duplicate query/action/repeat")
        cell["values"][action][repeat_id] = value

    records = sorted(cells.values(), key=lambda row: row["rank"])
    if len(records) != TRAIN_QUERIES:
        raise ValueError(f"Expected {TRAIN_QUERIES} queries, found {len(records)}")
    for expected_rank, record in enumerate(records):
        if record["rank"] != expected_rank:
            raise ValueError("Training query ranks must be contiguous from zero")
        for action in ACTIONS:
            if sorted(record["values"][action]) != list(REPEATS):
                raise ValueError("Each B/D action must contain repeats 0/1/2")
            record["values"][action] = [
                record["values"][action][repeat_id] for repeat_id in REPEATS
            ]

    validation = {
        "train_queries_read": len(records),
        "bd_repeat_rows_read": len(rows),
        "actions": list(ACTIONS),
        "repeat_ids": list(REPEATS),
        "all_selected_rows_success": True,
        "f1_finite_and_bounded": True,
        "sample_partition_counts": partition_counts,
        "outcome_partition_counts": outcome_partition_counts,
        "fresh_dev_outcome_rows_read": 0,
        "final_holdout_outcome_rows_read": 0,
        "partial_9600_rows_read": 0,
    }
    return records, validation


def _load_features(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = run_dir / "features" / "train.npz"
    with np.load(path, allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        lexical = np.asarray(stored["lexical"], dtype=np.float32)
        dense = np.asarray(stored["dense"], dtype=np.float32)
        embedding = np.asarray(stored["embedding"], dtype=np.float32)
    expected_query_ids = [str(record["query_id"]) for record in records]
    expected_group_ids = [str(record["group_id"]) for record in records]
    if list(query_ids[:TRAIN_QUERIES]) != expected_query_ids:
        raise ValueError("Feature and outcome query order differ")
    if list(group_ids[:TRAIN_QUERIES]) != expected_group_ids:
        raise ValueError("Feature and outcome group order differ")
    arrays = (lexical, dense, embedding)
    if any(array.shape[0] < TRAIN_QUERIES for array in arrays):
        raise ValueError("Feature matrix is shorter than the frozen structural gate")
    if any(not np.isfinite(array[:TRAIN_QUERIES]).all() for array in arrays):
        raise ValueError("Features must be finite")

    tier_a = lexical[:TRAIN_QUERIES]
    full = np.concatenate(
        [lexical[:TRAIN_QUERIES], dense[:TRAIN_QUERIES], embedding[:TRAIN_QUERIES]],
        axis=1,
    ).astype(np.float32)
    validation = {
        "feature_rows_available": int(len(query_ids)),
        "feature_rows_read": TRAIN_QUERIES,
        "query_order_matches": True,
        "group_order_matches": True,
        "features_finite": True,
        "tier_a_dimensions": int(tier_a.shape[1]),
        "full_dimensions": int(full.shape[1]),
    }
    return {"tier_a": tier_a, "full": full}, validation


def _utilities(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    bm25 = np.asarray(
        [np.mean(record["values"]["bm25"]) for record in records], dtype=np.float64
    )
    dense = np.asarray(
        [np.mean(record["values"]["dense"]) for record in records], dtype=np.float64
    )
    return bm25, dense


def _group_folds(
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = np.arange(len(groups))
    folds = [
        (np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64))
        for train, validation in splitter.split(positions, groups=groups)
    ]
    seen = np.concatenate([validation for _, validation in folds])
    if sorted(seen.tolist()) != list(range(len(groups))):
        raise RuntimeError("Validation folds must cover every row exactly once")
    for train, validation in folds:
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("A group crossed a fold boundary")
    return folds


def _new_model(kind: str, seed: int) -> Any:
    common = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": seed,
    }
    if kind == "pairwise":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    if kind == "gain":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    raise ValueError(f"Unknown model kind: {kind}")


def _fit_model(model: Any, kind: str, matrix: np.ndarray, gaps: np.ndarray) -> None:
    if kind == "pairwise":
        keep = np.abs(gaps) > TIE_ATOL
        labels = (gaps[keep] > 0.0).astype(np.int32)
        if len(np.unique(labels)) != 2:
            raise ValueError("Pairwise training fold must contain both preference classes")
        model.fit(matrix[keep], labels, sample_weight=np.abs(gaps[keep]))
    else:
        model.fit(matrix, gaps)


def _predict_scores(model: Any, kind: str, matrix: np.ndarray) -> np.ndarray:
    if kind == "pairwise":
        values = model.predict_proba(matrix)[:, 1]
    else:
        values = model.predict(matrix)
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
        raise RuntimeError("Model predictions must be a finite one-dimensional score")
    return scores


def choose_dense_default_threshold(scores: np.ndarray, gaps: np.ndarray) -> dict[str, Any]:
    """Choose score > threshold to maximize mean gain; ties prefer fewer switches."""
    scores = np.asarray(scores, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    if scores.shape != gaps.shape or scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Threshold scores and gaps must be aligned and non-empty")
    if not np.isfinite(scores).all() or not np.isfinite(gaps).all():
        raise ValueError("Threshold inputs must be finite")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    cumulative = np.cumsum(gaps[order])
    boundaries = np.flatnonzero(
        np.r_[ordered_scores[:-1] > ordered_scores[1:], True]
    ) + 1
    candidate_counts = np.r_[0, boundaries]

    best_count = 0
    best_total = 0.0
    for count in candidate_counts[1:]:
        total = float(cumulative[count - 1])
        if total > best_total + 1e-15:
            best_total = total
            best_count = int(count)
        elif abs(total - best_total) <= 1e-15 and int(count) < best_count:
            best_count = int(count)

    if best_count == 0:
        threshold = None
    elif best_count == len(scores):
        threshold = float(np.nextafter(ordered_scores[-1], -math.inf))
    else:
        threshold = float(ordered_scores[best_count])
    switched = (
        np.zeros(len(scores), dtype=bool)
        if threshold is None
        else scores > threshold
    )
    if int(np.sum(switched)) != best_count:
        raise RuntimeError("Threshold does not reproduce its selected switch count")
    return {
        "threshold": threshold,
        "inner_gain": float(np.mean(np.where(switched, gaps, 0.0))),
        "inner_switch_coverage": float(np.mean(switched)),
        "inner_switch_count": int(np.sum(switched)),
    }


def _nested_oof_for_seed(
    matrix: np.ndarray,
    gaps: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    model_seed: int,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    switches = np.zeros(len(matrix), dtype=bool)
    filled = np.zeros(len(matrix), dtype=bool)
    fold_results: list[dict[str, Any]] = []

    for outer_id, (outer_train, outer_validation) in enumerate(outer_folds):
        inner_groups = groups[outer_train]
        inner_folds = _group_folds(
            inner_groups,
            n_splits=INNER_FOLDS,
            seed=FOLD_SEED + 100 + outer_id,
        )
        inner_scores = np.full(len(outer_train), np.nan, dtype=np.float64)
        for inner_train_local, inner_validation_local in inner_folds:
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            model = _new_model(kind, model_seed)
            _fit_model(model, kind, matrix[inner_train], gaps[inner_train])
            inner_scores[inner_validation_local] = _predict_scores(
                model, kind, matrix[inner_validation]
            )
        if not np.isfinite(inner_scores).all():
            raise RuntimeError("Inner OOF scores were not completely filled")
        threshold = choose_dense_default_threshold(inner_scores, gaps[outer_train])

        model = _new_model(kind, model_seed)
        _fit_model(model, kind, matrix[outer_train], gaps[outer_train])
        outer_scores = _predict_scores(model, kind, matrix[outer_validation])
        fold_switches = (
            np.zeros(len(outer_scores), dtype=bool)
            if threshold["threshold"] is None
            else outer_scores > float(threshold["threshold"])
        )
        switches[outer_validation] = fold_switches
        filled[outer_validation] = True
        fold_results.append(
            {
                "outer_fold": outer_id,
                **threshold,
                "outer_queries": int(len(outer_validation)),
                "outer_gain": float(
                    np.mean(np.where(fold_switches, gaps[outer_validation], 0.0))
                ),
                "outer_switch_coverage": float(np.mean(fold_switches)),
            }
        )

    if not filled.all():
        raise RuntimeError("Outer OOF policy did not cover every query exactly once")
    return switches, fold_results


def _bootstrap_gain_ci(
    gains: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    ordered_groups = sorted(by_group)
    group_sums = np.asarray(
        [np.sum(gains[by_group[group]]) for group in ordered_groups],
        dtype=np.float64,
    )
    group_counts = np.asarray(
        [len(by_group[group]) for group in ordered_groups],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    batch_size = 128
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        sampled = rng.integers(
            0,
            len(ordered_groups),
            size=(stop - start, len(ordered_groups)),
        )
        estimates[start:stop] = (
            np.sum(group_sums[sampled], axis=1)
            / np.sum(group_counts[sampled], axis=1)
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _policy_metrics(
    bm25: np.ndarray,
    dense: np.ndarray,
    switches: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    with_ci: bool,
) -> dict[str, Any]:
    gaps = bm25 - dense
    paired_gains = np.where(switches, gaps, 0.0)
    switch_count = int(np.sum(switches))
    fixed_dense = float(np.mean(dense))
    router = fixed_dense + float(np.mean(paired_gains))
    oracle = float(np.mean(np.maximum(bm25, dense)))
    result = {
        "router_f1": router,
        "fixed_bm25_f1": float(np.mean(bm25)),
        "fixed_dense_f1": fixed_dense,
        "bd_oracle_f1": oracle,
        "bd_oracle_headroom": oracle - fixed_dense,
        "gain_over_fixed_dense": float(np.mean(paired_gains)),
        "switch_queries": switch_count,
        "switch_coverage": float(np.mean(switches)),
        "switch_precision": (
            float(np.mean(gaps[switches] > TIE_ATOL)) if switch_count else None
        ),
        "conditional_f1_gain": (
            float(np.mean(gaps[switches])) if switch_count else None
        ),
    }
    if with_ci:
        result["gain_ci95"] = _bootstrap_gain_ci(paired_gains, groups, seed=seed)
    return result


def run_gate(
    records: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    bm25, dense = _utilities(records)
    gaps = bm25 - dense
    groups = np.asarray([str(record["group_id"]) for record in records], dtype=np.str_)
    outer_folds = _group_folds(groups, n_splits=OUTER_FOLDS, seed=FOLD_SEED)
    candidates: dict[str, Any] = {}

    for candidate_id, (name, feature_tier, kind) in enumerate(CANDIDATES):
        print(f"candidate {candidate_id + 1}/{len(CANDIDATES)}: {name}", flush=True)
        matrix = features[feature_tier]
        seed_switches: list[np.ndarray] = []
        seed_results: list[dict[str, Any]] = []
        for seed_position, model_seed in enumerate(MODEL_SEEDS):
            print(
                f"  model seed {seed_position + 1}/{len(MODEL_SEEDS)}: {model_seed}",
                flush=True,
            )
            switches, fold_results = _nested_oof_for_seed(
                matrix,
                gaps,
                groups,
                kind=kind,
                model_seed=model_seed,
                outer_folds=outer_folds,
            )
            seed_switches.append(switches)
            metrics = _policy_metrics(
                bm25,
                dense,
                switches,
                groups,
                seed=BOOTSTRAP_SEED + candidate_id * 100 + seed_position,
                with_ci=False,
            )
            seed_results.append(
                {"model_seed": model_seed, **metrics, "folds": fold_results}
            )

        ensemble_switches = np.mean(np.asarray(seed_switches, dtype=np.float64), axis=0) > 0.5
        ensemble = _policy_metrics(
            bm25,
            dense,
            ensemble_switches,
            groups,
            seed=BOOTSTRAP_SEED + candidate_id * 100 + 50,
            with_ci=True,
        )
        seed_gains = [float(result["gain_over_fixed_dense"]) for result in seed_results]
        seed_median = float(np.median(seed_gains))
        gate = {
            "ensemble_gain_at_least_practical": (
                ensemble["gain_over_fixed_dense"] >= PRACTICAL_GAIN
            ),
            "ensemble_ci_lower_above_zero": ensemble["gain_ci95"][0] > 0.0,
            "all_model_seeds_positive": all(gain > 0.0 for gain in seed_gains),
            "seed_median_at_least_practical": seed_median >= PRACTICAL_GAIN,
            "nondegenerate_switching": ensemble["switch_queries"] > 0,
        }
        gate["passed"] = all(gate.values())
        candidates[name] = {
            "feature_tier": feature_tier,
            "model_kind": kind,
            "ensemble_rule": "majority_vote_dense_on_tie",
            "ensemble": ensemble,
            "seed_gain_median": seed_median,
            "seed_gain_min": float(min(seed_gains)),
            "seed_gain_max": float(max(seed_gains)),
            "positive_model_seeds": int(sum(gain > 0.0 for gain in seed_gains)),
            "model_seeds": seed_results,
            "gate": gate,
        }

    passing = [name for name, value in candidates.items() if value["gate"]["passed"]]
    strongest = max(
        candidates,
        key=lambda name: float(candidates[name]["ensemble"]["gain_over_fixed_dense"]),
    )
    return {
        "phase": "2.6",
        "status": "complete",
        "protocol": {
            "train_queries": TRAIN_QUERIES,
            "routing_target": "normalized_token_f1",
            "actions": list(ACTIONS),
            "default_action": "dense",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "fold_seed": FOLD_SEED,
            "model_seeds": list(MODEL_SEEDS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "practical_gain": PRACTICAL_GAIN,
            "external_calls": 0,
        },
        "candidates": candidates,
        "gate": {
            "passing_candidates": passing,
            "strongest_candidate": strongest,
            "decision": "GO_AC_LABEL_EXPANSION" if passing else "STOP_BEFORE_AC_LABELS",
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _arguments()
    run_dir = _resolve(args.source_run)
    records, outcome_validation = _load_outcomes(run_dir)
    features, feature_validation = _load_features(run_dir, records)
    validation = {**outcome_validation, **feature_validation}
    if args.dry_run:
        print(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    summary = run_gate(records, features)
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "validation.json", validation)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
