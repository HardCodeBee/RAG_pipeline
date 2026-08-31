#!/usr/bin/env python3
"""Summarize the frozen Phase 2.8 T0/T1/T2 formal router results."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_experiments.modeling import load_frozen_data, policy_metrics  # noqa: E402


RUN_DIR = PROJECT_ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1"
VIEWS = ("T0_tiecap", "T1_winner2000", "T2_winner3000")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    export_path = RUN_DIR / "export_manifest.json"
    export = _read_json(export_path)
    if not export.get("status", {}).get("all_quotas_met"):
        raise RuntimeError("Query-expansion quotas are not complete")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for view in VIEWS:
        view_dir = RUN_DIR / "training_views" / view
        config_path = view_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Expected YAML mapping: {config_path}")
        audit_dir = view_dir / "audit"
        formal_path = audit_dir / "formal_metrics.json"
        prediction_path = audit_dir / "formal_predictions.npz"
        formal = _read_json(formal_path)
        if formal.get("status") != "complete":
            raise RuntimeError(f"Formal audit is incomplete for {view}")
        data, validation = load_frozen_data(config)
        split_seeds = [int(value) for value in config["cross_validation"]["split_seeds"]]
        gate = config["formal_gate"]
        candidate_rows: dict[str, Any] = {}
        with np.load(prediction_path, allow_pickle=False) as stored:
            for candidate_id in formal["formal_candidates"]:
                scores = np.mean(
                    np.stack(
                        [
                            np.asarray(
                                stored[f"prediction__{candidate_id}__{split_seed}"],
                                dtype=np.float64,
                            )
                            for split_seed in split_seeds
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                metrics = policy_metrics(
                    data,
                    scores,
                    bootstrap_seed=int(config["evaluation"]["bootstrap"]["seed"]),
                    bootstrap_resamples=int(config["evaluation"]["bootstrap"]["resamples"]),
                )
                aggregate = formal["aggregate"][candidate_id]
                ratio = metrics["harmful_to_beneficial_mass_ratio"]
                slope = metrics["calibration"]["slope"]
                top_decile = metrics["calibration"]["top_decile_realized_gap"]
                checks = {
                    "practical_gain": float(aggregate["mean_gain_over_split_seeds"])
                    >= float(gate["practical_gain_minimum"]),
                    "grouped_bootstrap_ci_lower": float(metrics["gain_ci95"][0])
                    > float(gate["grouped_bootstrap_ci_lower_must_exceed"]),
                    "all_split_seed_gains_positive": bool(
                        aggregate["all_split_seed_gains_positive"]
                    ),
                    "harmful_to_beneficial_mass_ratio": ratio is not None
                    and float(ratio)
                    <= float(gate["harmful_to_beneficial_mass_ratio_maximum"]),
                    "switch_coverage": float(gate["switch_coverage_minimum"])
                    <= float(metrics["switch_coverage"])
                    <= float(gate["switch_coverage_maximum"]),
                    "calibration_slope": float(gate["calibration_slope_minimum"])
                    <= float(slope)
                    <= float(gate["calibration_slope_maximum"]),
                    "top_decile_realized_gap": float(top_decile) > 0.0,
                }
                passed = all(checks.values())
                seed_gains = [
                    float(item["gain_over_fixed_dense"])
                    for item in aggregate["split_seed_metrics"]
                ]
                row = {
                    "view": view,
                    "queries": int(len(data.query_ids)),
                    "bm25_winners": int(np.sum(data.gap > 1e-12)),
                    "dense_winners": int(np.sum(data.gap < -1e-12)),
                    "ties": int(np.sum(np.abs(data.gap) <= 1e-12)),
                    "candidate_id": candidate_id,
                    "fixed_bm25_mean_f1": metrics["fixed_bm25_mean_f1"],
                    "fixed_dense_mean_f1": metrics["fixed_dense_mean_f1"],
                    "oracle_mean_f1": metrics["oracle_mean_f1"],
                    "mean_gain_over_split_seeds": aggregate["mean_gain_over_split_seeds"],
                    "split_seed_gains": ";".join(f"{value:.12g}" for value in seed_gains),
                    "consensus_router_mean_f1": metrics["router_mean_f1"],
                    "consensus_gain_over_fixed_dense": metrics["gain_over_fixed_dense"],
                    "gain_ci95_lower": metrics["gain_ci95"][0],
                    "gain_ci95_upper": metrics["gain_ci95"][1],
                    "switch_coverage": metrics["switch_coverage"],
                    "beneficial_switches": metrics["beneficial_switches"],
                    "harmful_switches": metrics["harmful_switches"],
                    "harmful_to_beneficial_mass_ratio": ratio,
                    "high_margin_bm25_winner_recall": metrics[
                        "high_margin_bm25_winner_recall"
                    ],
                    "oracle_recovery": metrics["oracle_recovery"],
                    "gap_spearman": metrics["gap_spearman"],
                    "non_tie_auc": metrics["non_tie_auc"],
                    "calibration_slope": slope,
                    "top_decile_realized_gap": top_decile,
                    "gate_passed": passed,
                    "failed_checks": ";".join(name for name, ok in checks.items() if not ok),
                }
                rows.append(row)
                candidate_rows[candidate_id] = {
                    "passed": passed,
                    "checks": checks,
                    "mean_gain_over_split_seeds": aggregate["mean_gain_over_split_seeds"],
                    "split_seed_gains": seed_gains,
                    "consensus_metrics": metrics,
                }
        details[view] = {
            "query_count": int(len(data.query_ids)),
            "validation": validation,
            "config": {"path": _relative(config_path), "sha256": _sha256(config_path)},
            "formal_metrics": {"path": _relative(formal_path), "sha256": _sha256(formal_path)},
            "formal_predictions": {
                "path": _relative(prediction_path),
                "sha256": _sha256(prediction_path),
            },
            "candidates": candidate_rows,
        }

    csv_path = RUN_DIR / "training_scale_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passing = [row for row in rows if row["gate_passed"]]
    best_diagnostic = max(
        rows,
        key=lambda row: (
            float(row["gain_ci95_lower"]),
            float(row["consensus_gain_over_fixed_dense"]),
        ),
    )
    decision = {
        "protocol_id": export["status"]["protocol_id"],
        "status": "complete",
        "decision": (
            "OPEN_CONFIRMATION_POOL" if passing else "DO_NOT_OPEN_CONFIRMATION_POOL"
        ),
        "reason": (
            "At least one view/candidate passed every frozen internal gate."
            if passing
            else "No view/candidate passed every frozen internal gate."
        ),
        "passing_candidates": [
            {"view": row["view"], "candidate_id": row["candidate_id"]} for row in passing
        ],
        "best_diagnostic_by_ci_lower": {
            "view": best_diagnostic["view"],
            "candidate_id": best_diagnostic["candidate_id"],
            "consensus_gain_over_fixed_dense": best_diagnostic[
                "consensus_gain_over_fixed_dense"
            ],
            "gain_ci95": [
                best_diagnostic["gain_ci95_lower"],
                best_diagnostic["gain_ci95_upper"],
            ],
            "failed_checks": best_diagnostic["failed_checks"].split(";"),
        },
        "comparison_csv": {"path": _relative(csv_path), "sha256": _sha256(csv_path)},
        "views": details,
        "confirmation": {
            "planned_queries": 2000,
            "opened": bool(passing),
            "rows_read": 0,
        },
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "external_provider_calls": 0,
        "excluded_artifacts": {
            "T0_partial_legacy_screen": (
                "The Phase 2.7 screen entrypoint attempted the old 17-candidate screen; "
                "it was interrupted after one irrelevant baseline and is not used here."
            )
        },
        "completed_at_unix": time.time(),
    }
    decision_path = RUN_DIR / "training_decision.json"
    _write_json(decision_path, decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "passing_candidates": len(passing),
                "best_diagnostic_by_ci_lower": decision["best_diagnostic_by_ci_lower"],
                "comparison_csv": _relative(csv_path),
                "decision_json": _relative(decision_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
