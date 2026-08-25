"""Audit whether F1 preferences are a valid surrogate for B/D AC routing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("bm25", "dense")
PARTITIONS = ("train", "dev")
EXPECTED_QUERIES = {"train": 300, "dev": 300}
EXPECTED_REPEATS = (0, 1, 2)
TIE_ATOL = 1e-12
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260831


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results",
        default=(
            "outputs/router/hotpotqa_bd_router_v1/runs/"
            "phase2_bd_headroom_repeated_v1/results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/router/hotpotqa_bd_router_v1/runs/"
            "phase25_f1_ac_audit_v1"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _metric(row: Mapping[str, Any], name: str, *, line_number: int) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or name not in metrics:
        raise ValueError(f"Line {line_number}: missing metric {name}")
    value = float(metrics[name])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Line {line_number}: {name} must be finite and in [0, 1]")
    return value


def load_phase2_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and strictly validate the frozen 600-query B/D repeat table."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    query_meta: dict[str, tuple[str, str]] = {}
    group_partition: dict[str, str] = {}
    source_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError(f"Line {line_number}: JSONL row must be an object")
            if row.get("status") != "success":
                raise ValueError(f"Line {line_number}: non-success source row")

            partition = str(row.get("split", ""))
            if partition == "final_holdout":
                raise ValueError("Phase 2.5 must not read final-holdout outcomes")
            if partition not in PARTITIONS:
                raise ValueError(f"Line {line_number}: unexpected partition {partition!r}")
            action = str(row.get("action", ""))
            if action not in ACTIONS:
                raise ValueError(f"Line {line_number}: unexpected action {action!r}")
            repeat_id = int(row.get("repeat_id", -1))
            if repeat_id not in EXPECTED_REPEATS:
                raise ValueError(f"Line {line_number}: unexpected repeat_id {repeat_id}")

            query_id = str(row.get("query_id", ""))
            group_id = str(row.get("group_id", ""))
            if not query_id or not group_id:
                raise ValueError(f"Line {line_number}: missing query_id or group_id")
            metadata = (partition, group_id)
            previous = query_meta.setdefault(query_id, metadata)
            if previous != metadata:
                raise ValueError(f"Line {line_number}: inconsistent query metadata")
            previous_partition = group_partition.setdefault(group_id, partition)
            if previous_partition != partition:
                raise ValueError(f"Line {line_number}: group crosses partitions")

            key = (query_id, action, repeat_id)
            if key in cells:
                raise ValueError(f"Line {line_number}: duplicate query/action/repeat")
            cells[key] = {
                "f1": _metric(row, "normalized_token_f1", line_number=line_number),
                "ac": _metric(row, "answer_correctness", line_number=line_number),
            }

    expected_rows = sum(EXPECTED_QUERIES.values()) * len(ACTIONS) * len(EXPECTED_REPEATS)
    if source_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows, found {source_rows}")

    records: list[dict[str, Any]] = []
    for query_id in sorted(query_meta):
        partition, group_id = query_meta[query_id]
        repeats: dict[str, dict[str, list[float]]] = {}
        for action in ACTIONS:
            missing = [
                repeat_id
                for repeat_id in EXPECTED_REPEATS
                if (query_id, action, repeat_id) not in cells
            ]
            if missing:
                raise ValueError(f"Query {query_id}, action {action}: missing repeats {missing}")
            repeats[action] = {
                metric: [cells[(query_id, action, repeat_id)][metric] for repeat_id in EXPECTED_REPEATS]
                for metric in ("f1", "ac")
            }
        records.append(
            {
                "query_id": query_id,
                "group_id": group_id,
                "partition": partition,
                "repeats": repeats,
            }
        )

    counts = {
        partition: sum(record["partition"] == partition for record in records)
        for partition in PARTITIONS
    }
    if counts != EXPECTED_QUERIES:
        raise ValueError(f"Expected query counts {EXPECTED_QUERIES}, found {counts}")
    if len(records) != sum(EXPECTED_QUERIES.values()):
        raise ValueError("Unexpected unique query count")

    validation = {
        "source_rows": source_rows,
        "queries": len(records),
        "partition_queries": counts,
        "actions": list(ACTIONS),
        "repeat_ids": list(EXPECTED_REPEATS),
        "all_rows_success": True,
        "metrics_finite_and_bounded": True,
        "group_partition_isolation": True,
        "final_holdout_rows": 0,
    }
    return records, validation


def _preference(value: float) -> str:
    if value > TIE_ATOL:
        return "bm25"
    if value < -TIE_ATOL:
        return "dense"
    return "tie"


def _bootstrap_mean_ci(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("Bootstrap values and groups must be non-empty and aligned")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    ordered_groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        sampled = rng.choice(ordered_groups, size=len(ordered_groups), replace=True)
        positions = [position for group in sampled for position in by_group[str(group)]]
        estimates[iteration] = float(np.mean(values[positions]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _selector_metrics(
    gains: np.ndarray,
    switched: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if gains.shape != switched.shape:
        raise ValueError("Selector gains and switches must align")
    switched_count = int(np.sum(switched))
    return {
        "ac_gain_over_dense": float(np.mean(gains)),
        "ac_gain_ci95": _bootstrap_mean_ci(
            gains,
            groups,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed,
        ),
        "switch_queries": switched_count,
        "switch_coverage": float(np.mean(switched)),
        "switch_precision": (
            float(np.mean(gains[switched] > TIE_ATOL)) if switched_count else None
        ),
        "conditional_ac_gain": (
            float(np.mean(gains[switched])) if switched_count else None
        ),
    }


def _cross_repeat(
    records: Sequence[Mapping[str, Any]],
    *,
    selector_metric: str,
    seed: int,
) -> dict[str, Any]:
    per_rotation: list[dict[str, Any]] = []
    query_gains = np.zeros(len(records), dtype=np.float64)
    query_switches = np.zeros(len(records), dtype=np.float64)
    groups = [str(record["group_id"]) for record in records]

    for held_out in EXPECTED_REPEATS:
        train_repeats = [repeat_id for repeat_id in EXPECTED_REPEATS if repeat_id != held_out]
        gains = np.zeros(len(records), dtype=np.float64)
        switched = np.zeros(len(records), dtype=bool)
        for index, record in enumerate(records):
            repeats = record["repeats"]
            selection_gap = float(
                np.mean([repeats["bm25"][selector_metric][repeat_id] for repeat_id in train_repeats])
                - np.mean([repeats["dense"][selector_metric][repeat_id] for repeat_id in train_repeats])
            )
            use_bm25 = selection_gap > TIE_ATOL
            switched[index] = use_bm25
            if use_bm25:
                gains[index] = float(
                    repeats["bm25"]["ac"][held_out] - repeats["dense"]["ac"][held_out]
                )
        query_gains += gains / len(EXPECTED_REPEATS)
        query_switches += switched.astype(np.float64) / len(EXPECTED_REPEATS)
        per_rotation.append(
            {
                "held_out_repeat": held_out,
                "ac_gain_over_dense": float(np.mean(gains)),
                "switch_queries": int(np.sum(switched)),
                "switch_coverage": float(np.mean(switched)),
            }
        )

    return {
        "selector_metric": selector_metric,
        "rotation_results": per_rotation,
        "mean_ac_gain_over_dense": float(np.mean(query_gains)),
        "mean_ac_gain_ci95": _bootstrap_mean_ci(
            query_gains,
            groups,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed,
        ),
        "mean_switch_coverage": float(np.mean(query_switches)),
    }


def _partition_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    partition_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = [str(record["group_id"]) for record in records]
    f1_gaps = np.asarray(
        [
            np.mean(record["repeats"]["bm25"]["f1"])
            - np.mean(record["repeats"]["dense"]["f1"])
            for record in records
        ],
        dtype=np.float64,
    )
    ac_gaps = np.asarray(
        [
            np.mean(record["repeats"]["bm25"]["ac"])
            - np.mean(record["repeats"]["dense"]["ac"])
            for record in records
        ],
        dtype=np.float64,
    )
    f1_preferences = [_preference(value) for value in f1_gaps]
    ac_preferences = [_preference(value) for value in ac_gaps]
    labels = ("bm25", "tie", "dense")
    confusion_rows = [
        {
            "partition": partition_name,
            "ac_preference": ac_label,
            "f1_bm25": sum(
                ac_value == ac_label and f1_value == "bm25"
                for ac_value, f1_value in zip(ac_preferences, f1_preferences)
            ),
            "f1_tie": sum(
                ac_value == ac_label and f1_value == "tie"
                for ac_value, f1_value in zip(ac_preferences, f1_preferences)
            ),
            "f1_dense": sum(
                ac_value == ac_label and f1_value == "dense"
                for ac_value, f1_value in zip(ac_preferences, f1_preferences)
            ),
        }
        for ac_label in labels
    ]

    ac_non_tie = np.abs(ac_gaps) > TIE_ATOL
    agreements = np.asarray(
        [f1_value == ac_value for f1_value, ac_value in zip(f1_preferences, ac_preferences)],
        dtype=bool,
    )
    weights = np.abs(ac_gaps)
    sign_agreement = float(np.mean(agreements[ac_non_tie])) if np.any(ac_non_tie) else None
    weighted_agreement = (
        float(np.sum(weights * agreements) / np.sum(weights)) if np.sum(weights) > 0 else None
    )

    mean_f1_switches = f1_gaps > TIE_ATOL
    mean_f1_gains = np.where(mean_f1_switches, ac_gaps, 0.0)
    mean_ac_switches = ac_gaps > TIE_ATOL
    mean_ac_gains = np.where(mean_ac_switches, ac_gaps, 0.0)

    stable_preferences: list[str] = []
    for record in records:
        repeat_preferences: list[str] = []
        for held_out in EXPECTED_REPEATS:
            training = [repeat_id for repeat_id in EXPECTED_REPEATS if repeat_id != held_out]
            gap = float(
                np.mean([record["repeats"]["bm25"]["ac"][repeat_id] for repeat_id in training])
                - np.mean([record["repeats"]["dense"]["ac"][repeat_id] for repeat_id in training])
            )
            repeat_preferences.append(_preference(gap))
        stable_preferences.append(
            repeat_preferences[0]
            if len(set(repeat_preferences)) == 1 and repeat_preferences[0] != "tie"
            else "unstable_or_tie"
        )

    result = {
        "queries": len(records),
        "groups": len(set(groups)),
        "mean_f1_gap_bm25_minus_dense": float(np.mean(f1_gaps)),
        "mean_ac_gap_bm25_minus_dense": float(np.mean(ac_gaps)),
        "preference_counts": {
            "f1": {label: f1_preferences.count(label) for label in labels},
            "ac": {label: ac_preferences.count(label) for label in labels},
        },
        "ac_non_tie_queries": int(np.sum(ac_non_tie)),
        "f1_ac_sign_agreement_on_ac_non_ties": sign_agreement,
        "ac_gap_weighted_sign_agreement": weighted_agreement,
        "f1_mean_selector_evaluated_in_ac": _selector_metrics(
            mean_f1_gains,
            mean_f1_switches,
            groups,
            seed=seed,
        ),
        "ac_mean_oracle": _selector_metrics(
            mean_ac_gains,
            mean_ac_switches,
            groups,
            seed=seed + 1,
        ),
        "cross_repeat_f1_selector": _cross_repeat(
            records,
            selector_metric="f1",
            seed=seed + 2,
        ),
        "cross_repeat_ac_selector": _cross_repeat(
            records,
            selector_metric="ac",
            seed=seed + 3,
        ),
        "leave_one_repeat_out_stable_preference": {
            "bm25": stable_preferences.count("bm25"),
            "dense": stable_preferences.count("dense"),
            "unstable_or_tie": stable_preferences.count("unstable_or_tie"),
        },
    }
    return result, confusion_rows


def run_audit(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    partitions: dict[str, Any] = {}
    confusion: list[dict[str, Any]] = []
    for offset, partition in enumerate(PARTITIONS):
        selected = [record for record in records if record["partition"] == partition]
        result, rows = _partition_audit(
            selected,
            partition_name=partition,
            seed=BOOTSTRAP_SEED + offset * 20,
        )
        partitions[partition] = result
        confusion.extend(rows)

    pooled, pooled_rows = _partition_audit(
        records,
        partition_name="pooled_descriptive",
        seed=BOOTSTRAP_SEED + 40,
    )
    confusion.extend(pooled_rows)

    primary = partitions["train"]
    ac_cross = primary["cross_repeat_ac_selector"]
    f1_cross = primary["cross_repeat_f1_selector"]
    ac_label_gate = (
        ac_cross["mean_ac_gain_over_dense"] > 0.0
        and ac_cross["mean_ac_gain_ci95"][0] > 0.0
    )
    f1_surrogate_gate = (
        f1_cross["mean_ac_gain_over_dense"] > 0.0
        and f1_cross["mean_ac_gain_ci95"][0] > 0.0
    )
    summary = {
        "phase": "2.5",
        "status": "complete",
        "protocol": {
            "actions": list(ACTIONS),
            "repeats_per_query_action": len(EXPECTED_REPEATS),
            "primary_gate_partition": "train",
            "historical_corroboration_partition": "dev",
            "pooled_role": "descriptive_only",
            "policy_tie_break": "dense",
            "bootstrap_unit": "information_need_group",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "cross_repeat_aggregation": "mean_three_rotations_within_query",
            "external_calls": 0,
        },
        "partitions": partitions,
        "pooled_descriptive": pooled,
        "gate": {
            "ac_preference_stable_for_training": ac_label_gate,
            "f1_retained_as_auxiliary": f1_surrogate_gate,
            "f1_role": "auxiliary_or_diagnostic" if f1_surrogate_gate else "reference_only",
            "decision": "GO_AC_LABEL_EXPANSION" if ac_label_gate else "STOP_UNSTABLE_AC_TARGET",
        },
    }
    return summary, confusion


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    confusion: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "preference_confusion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("partition", "ac_preference", "f1_bm25", "f1_tie", "f1_dense"),
        )
        writer.writeheader()
        writer.writerows(confusion)


def main() -> int:
    args = _arguments()
    source = _resolve(args.source_results)
    records, validation = load_phase2_records(source)
    if args.dry_run:
        print(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    summary, confusion = run_audit(records)
    _write_outputs(_resolve(args.output_dir), summary, validation, confusion)
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
