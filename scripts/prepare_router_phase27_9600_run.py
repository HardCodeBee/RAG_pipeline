#!/usr/bin/env python3
"""Prepare the frozen-candidate 9,600-query Phase 2.7 confirmation run."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / (
    "analysis/hotpotqa_router/phases/phase27/config_screen_4800.yaml"
)
EXPANDED_CONFIG = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase27/config.yaml"
BASE_OUTPUT = (
    PROJECT_ROOT
    / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_v1"
)
EXPANDED_OUTPUT = (
    PROJECT_ROOT
    / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1"
)
SNAPSHOT_DIR = EXPANDED_OUTPUT / "snapshot"
SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "snapshot_manifest.json"
PROTOCOL_ID = "hotpotqa_bd_router_phase27_model_audit_9600_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def contract(path: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": relative(path),
        "sha256": sha256(path),
    }
    if expected_rows is not None:
        value["expected_rows"] = expected_rows
    return value


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("query_count") != 9600 or manifest.get("outcome_rows") != 57600:
        raise ValueError("The expanded snapshot is not the complete 9,600-query B/D pool")
    if manifest.get("fresh_dev_rows") != 0 or manifest.get("final_holdout_rows") != 0:
        raise ValueError("The expanded snapshot crossed a sealed evaluation boundary")
    if not all(manifest.get("prefix_validation", {}).values()):
        raise ValueError("The first 4,800-query prefix is not identical to the frozen pool")

    config["protocol"]["id"] = PROTOCOL_ID
    config["protocol"]["frozen_date"] = "2026-08-27"
    config["scope"]["query_count"] = 9600
    config["scope"]["excluded_from_v1"] = [
        value
        for value in config["scope"]["excluded_from_v1"]
        if value != "new_generation"
    ]
    config["scope"]["excluded_from_v1"].append("generation_beyond_authorized_9600")

    config["frozen_inputs"] = {
        "outcomes": contract(
            SNAPSHOT_DIR / "phase27_bd_9600_outcomes.jsonl.gz", expected_rows=57600
        ),
        "query_summary": contract(
            SNAPSHOT_DIR / "phase27_bd_9600_query_summary.csv.gz", expected_rows=9600
        ),
        "features": contract(
            SNAPSHOT_DIR / "phase27_features_9600.npz", expected_rows=9600
        ),
        "feature_schema": contract(SNAPSHOT_DIR / "feature_schema.json"),
        "snapshot_manifest": contract(SNAPSHOT_MANIFEST),
    }
    config["learning_curve"]["query_counts"] = [1200, 2400, 4800, 9600]
    config["learning_curve"].pop("may_trigger_9600_authorization_request_if", None)
    config["learning_curve"]["expanded_confirmation_rule"] = (
        "reuse the frozen 1200/2400/4800 rows and fit only the identical candidates at 9600"
    )
    config["authorization_gates"]["complete_9600_generation"] = (
        "authorized_by_user_and_completed_2026_08_27"
    )
    config["authorization_gates"]["generation_beyond_9600"] = (
        "not_authorized_by_this_confirmation"
    )
    config["planned_implementation"]["output_dir"] = relative(EXPANDED_OUTPUT)
    config["execution_strategy"]["state_model"]["results"] = relative(
        EXPANDED_OUTPUT / "execution_state.json"
    )
    config["execution_strategy"]["reuse_policy"] = {
        "candidate_selection": "reuse_exact_4800_candidate_freeze_without_reselection",
        "stages_2_to_6": "reuse_4800_diagnostic_evidence",
        "learning_curve_1200_2400_4800": "reuse_exact_completed_rows",
        "formal_9600": "fit_all_frozen_candidates_on_three_frozen_split_seeds",
    }
    for stage in config["execution_stages"]:
        if stage["stage"] == 0:
            stage["tasks"] = [
                task.replace(
                    "validate_4800_queries_and_28800_complete_bd_repeat_rows",
                    "validate_9600_queries_and_57600_complete_bd_repeat_rows",
                )
                for task in stage["tasks"]
            ]
        if stage["stage"] == 8:
            stage["tasks"] = [
                "reuse_identical_frozen_candidates_at_1200_2400_and_4800_queries",
                "run_identical_frozen_candidates_at_9600_queries",
                "keep_order_group_splits_features_targets_and_metrics_consistent",
                "do_not_request_additional_query_only_scaling_if_the_formal_gate_fails",
            ]

    EXPANDED_CONFIG.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )

    inherited_names = [
        "screen.json",
        "feature_block_report.json",
        "target_diagnostics.json",
        "stage4_report.json",
        "calibration.json",
        "router_evaluation.json",
    ]
    inherited: dict[str, Any] = {}
    for name in inherited_names:
        source = BASE_OUTPUT / name
        target = EXPANDED_OUTPUT / name
        shutil.copy2(source, target)
        inherited[name] = {
            "source": relative(source),
            "target": relative(target),
            "sha256": sha256(target),
            "role": "candidate_selection_or_diagnostic_evidence_from_frozen_4800_prefix",
        }

    base_freeze_path = BASE_OUTPUT / "candidate_freeze.json"
    base_freeze = json.loads(base_freeze_path.read_text(encoding="utf-8"))
    expanded_freeze = {
        "protocol_id": PROTOCOL_ID,
        "stage": "7.1_candidate_freeze_inherited_for_9600_confirmation",
        "status": "complete",
        "formal_candidates": base_freeze["formal_candidates"],
        "candidate_specs": base_freeze["candidate_specs"],
        "selection_rule": "no_reselection_after_4800_candidate_freeze",
        "source_candidate_freeze": contract(base_freeze_path),
        "source_frozen_at_unix": base_freeze["frozen_at_unix"],
        "stage_2_to_6_evidence": inherited,
        "snapshot_manifest": contract(SNAPSHOT_MANIFEST),
        "prefix_validation": manifest["prefix_validation"],
        "screen_is_not_a_formal_claim": True,
        "new_candidates_allowed": False,
        "external_calls": 0,
        "prepared_at_unix": time.time(),
    }
    (EXPANDED_OUTPUT / "candidate_freeze.json").write_text(
        json.dumps(expanded_freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    base_progress_path = BASE_OUTPUT / "learning_curve_progress.json"
    base_progress = json.loads(base_progress_path.read_text(encoding="utf-8"))
    if len(base_progress.get("rows", [])) != 36:
        raise ValueError("Expected 36 frozen learning-curve rows at 1,200/2,400/4,800")
    expanded_progress = {
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "formal_candidates": base_progress["formal_candidates"],
        "query_counts": [1200, 2400, 4800, 9600],
        "split_seeds": base_progress["split_seeds"],
        "rows": base_progress["rows"],
        "inherited_prefix_source": contract(base_progress_path),
        "inherited_row_count": 36,
        "external_calls": 0,
    }
    (EXPANDED_OUTPUT / "learning_curve_progress.json").write_text(
        json.dumps(expanded_progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    preparation = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "config": contract(EXPANDED_CONFIG),
        "snapshot_manifest": contract(SNAPSHOT_MANIFEST),
        "candidate_freeze": contract(EXPANDED_OUTPUT / "candidate_freeze.json"),
        "inherited_learning_curve_rows": 36,
        "formal_candidates": expanded_freeze["formal_candidates"],
        "query_counts": [1200, 2400, 4800, 9600],
        "split_seeds": base_progress["split_seeds"],
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "external_calls": 0,
        "prepared_at_unix": time.time(),
    }
    (EXPANDED_OUTPUT / "expanded_preparation.json").write_text(
        json.dumps(preparation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
