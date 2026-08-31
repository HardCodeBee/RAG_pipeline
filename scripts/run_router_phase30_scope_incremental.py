#!/usr/bin/env python3
"""Run the frozen Phase 2.10 scope-feature incremental Router diagnostic.

This runner is deliberately narrow.  It tests whether the only Phase 2.9
single feature that survived strict confirmation adds utility to the existing
T2 M3 model.  It never reads the previously consumed natural-confirmation
outcomes; a passing internal decision only authorizes a separately frozen,
retrospective natural diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts.router_phase29_trait_features import (  # noqa: E402
    CorpusCache,
    _question_features,
)
from src.router_experiments.modeling import (  # noqa: E402
    CandidateSpec,
    RouterData,
    TIE_ATOL,
    load_config as _load_config,
    load_frozen_data,
    policy_metrics,
    run_candidate_split,
)


PROTOCOL_ID = "hotpotqa_bd_router_phase30_scope_incremental_v1"
FEATURE_NAME = "c_scope_log_at_least_2_docs"
BASELINE_ID = "M3_pca32_structured_ridge"
SCOPE_ONLY_ID = "S0_scope_only_ridge"
AUGMENTED_ID = "M3S_pca32_structured_scope_ridge"
SPLIT_SEEDS = (20260901, 20260917, 20261003)
MODEL_SEED = 11
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_RESAMPLES = 10_000
FIXED_THRESHOLD = 0.0

RUN_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/"
    "phase30_scope_incremental_v1"
)
T2_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1/"
    "training_views/T2_winner3000"
)
PHASE29_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase29_single_trait_audit_v1"
)
PHASE28B_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase28_holdout_selection_v1"
)
CONFIG_PATH = PROJECT_ROOT / "analysis/hotpotqa_router/phase30_scope_incremental_config.yaml"
PLAN_PATH = PROJECT_ROOT / "analysis/hotpotqa_router/phase30_scope_incremental_plan.md"

INPUTS: dict[str, tuple[Path, str]] = {
    "t2_config": (
        T2_DIR / "config.yaml",
        "9fdab56caff8eff5f937da9f5556f5919899a4ea86c545713f2c6b94ca4f31ae",
    ),
    "t2_query_summary": (
        T2_DIR / "snapshot/query_summary.csv.gz",
        "4f487d70412bc51ede9e4a286f5fed9433f75ef8904ce296e39c250f2ef2c811",
    ),
    "t2_features": (
        T2_DIR / "snapshot/features.npz",
        "1ab63ff08457aa5a3235dc97d4678d890c94ceacd836a6a0a5c4d03eedd7718c",
    ),
    "frozen_m3_oof": (
        T2_DIR / "audit/formal_predictions.npz",
        "b6c9c20128cfbf2ac2f9224e06aa4a6abddf5a8a97c2485eeee081475d0f47f7",
    ),
    "frozen_m3_metrics": (
        T2_DIR / "audit/formal_metrics.json",
        "3fb5ea838eb00a411028db0c5b8fbb7d07dc2a776ecb991f4a1eefee5e9e759b",
    ),
    "phase29_cache": (
        PHASE29_DIR / "corpus_static_cache_v2.npz",
        "ee9077eee60184fa4d760f0f457fe693c47913162e3e3eda4e486dbd66bdfebd",
    ),
    "phase29_cache_manifest": (
        PHASE29_DIR / "corpus_static_cache_manifest.json",
        "0a9e26f31c425757072b624e226444216b62fda55d7db0cb976239823df0d52d",
    ),
    "phase29_discovery_features": (
        PHASE29_DIR / "discovery_features.npz",
        "ad3517430a8d54438461b60442801764ead282371f827b5513187cc336b1aec2",
    ),
    "phase29_natural_features": (
        PHASE29_DIR / "natural_features.npz",
        "983ebf61986a9c6841a389ca0964cb02c668adef7e105a5784beca32fde270f9",
    ),
    "phase29_catalog": (
        PHASE29_DIR / "feature_catalog.json",
        "c068bac560b1af63c6de6727d6d622cd898261dd12cc687df42c601ba38fa0f0",
    ),
    "phase29_decision": (
        PHASE29_DIR / "decision.json",
        "80eb1ebb113c43ec12e7783370f2907006e250714fe05f5517a583a9980914c2",
    ),
    "query_source": (
        PROJECT_ROOT / "data/beir/hotpotqa/queries/queries.jsonl",
        "26bd91dd9ad69592c02e07180e62096f572b6814f5f413695151485b4ab40a2c",
    ),
    # Hashing this sealed, already-consumed file is allowed.  No stage in this
    # runner parses its rows or outcome values.
    "sealed_natural_outcomes": (
        PHASE28B_DIR / "holdout_query_metrics.jsonl.gz",
        "1dbbbaa11b715ae006fc22b6dab73e8b270a87006cd2bd67ea99047a53c8f5a9",
    ),
}

IMPLEMENTATIONS = {
    "runner": Path(__file__).resolve(),
    "protocol_config": CONFIG_PATH,
    "protocol_plan": PLAN_PATH,
    "phase27_model_runner": PROJECT_ROOT / "scripts/run_router_phase27_model_audit.py",
    "phase29_feature_extractor": PROJECT_ROOT / "scripts/router_phase29_trait_features.py",
}

FEATURE_PATH = RUN_DIR / "scope_features_T2.npz"
FEATURE_MANIFEST_PATH = RUN_DIR / "scope_features_T2_manifest.json"
OOF_PREDICTION_PATH = RUN_DIR / "internal_oof_predictions.npz"
OOF_RESULT_PATH = RUN_DIR / "internal_metrics.json"
COMPARISON_PATH = RUN_DIR / "internal_comparison.csv"
DECISION_PATH = RUN_DIR / "internal_decision.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "extract", "oof", "decision", "run", "status"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_ready(value), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _implementation_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in IMPLEMENTATIONS.items()}


def _validate_protocol_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file() or not PLAN_PATH.is_file():
        raise FileNotFoundError("Phase 2.10 config and plan must exist before preflight")
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase 2.10 config must be a YAML mapping")
    if value.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected Phase 2.10 protocol id")
    scope = value.get("scope", {})
    if int(scope.get("training_queries", -1)) != 12_000:
        raise ValueError("Phase 2.10 config must freeze 12,000 T2 queries")
    if int(scope.get("training_groups", -1)) != 11_964:
        raise ValueError("Phase 2.10 config must freeze 11,964 T2 groups")
    if scope.get("candidates_exactly") != [
        SCOPE_ONLY_ID,
        BASELINE_ID,
        AUGMENTED_ID,
    ]:
        raise ValueError("Phase 2.10 candidate list differs from the runner")
    cross_validation = value.get("cross_validation", {})
    if tuple(int(seed) for seed in cross_validation.get("split_seeds", [])) != SPLIT_SEEDS:
        raise ValueError("Phase 2.10 split seeds differ from the runner")
    if int(cross_validation.get("outer_folds", -1)) != 5:
        raise ValueError("Phase 2.10 requires five outer folds")
    if int(cross_validation.get("inner_folds", -1)) != 4:
        raise ValueError("Phase 2.10 requires four inner folds")
    policy = value.get("policy", {})
    if float(policy.get("fixed_threshold", float("nan"))) != FIXED_THRESHOLD:
        raise ValueError("Phase 2.10 threshold must remain zero")
    bootstrap = value.get("evaluation", {}).get("bootstrap", {})
    if int(bootstrap.get("seed", -1)) != BOOTSTRAP_SEED:
        raise ValueError("Phase 2.10 bootstrap seed differs from the runner")
    if int(bootstrap.get("resamples", -1)) != BOOTSTRAP_RESAMPLES:
        raise ValueError("Phase 2.10 bootstrap resamples differ from the runner")
    return value


def _validate_input_hashes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, (path, expected) in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash changed for {name}: expected={expected} actual={actual}"
            )
        rows.append(
            {
                "name": name,
                "path": _relative(path),
                "sha256": actual,
                "passed": True,
            }
        )
    return rows


def _load_training() -> tuple[dict[str, Any], RouterData, dict[str, Any]]:
    config = _load_config(INPUTS["t2_config"][0])
    data, validation = load_frozen_data(config)
    return config, data, validation


def _baseline_arrays(data: RouterData) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(INPUTS["frozen_m3_oof"][0], allow_pickle=False) as stored:
        for seed in SPLIT_SEEDS:
            score_key = f"prediction__{BASELINE_ID}__{seed}"
            fold_key = f"fold_id__{BASELINE_ID}__{seed}"
            if score_key not in stored.files or fold_key not in stored.files:
                raise KeyError(f"Frozen M3 artifact lacks seed {seed}")
            scores = np.asarray(stored[score_key], dtype=np.float64)
            folds = np.asarray(stored[fold_key], dtype=np.int64)
            if scores.shape != data.gap.shape or folds.shape != data.gap.shape:
                raise ValueError(f"Frozen M3 shapes differ for seed {seed}")
            if not np.isfinite(scores).all() or set(np.unique(folds)) != set(range(5)):
                raise ValueError(f"Frozen M3 values are invalid for seed {seed}")
            for group in np.unique(data.group_ids):
                if len(np.unique(folds[data.group_ids == group])) != 1:
                    raise RuntimeError(f"Group {group} crosses frozen fold ids")
            result[seed] = (scores, folds)
    return result


def _assert_phase29_contract() -> dict[str, Any]:
    decision = _read_json(INPUTS["phase29_decision"][0])
    if decision.get("decision") != "SINGLE_FEATURE_GATE_PASS":
        raise ValueError("Phase 2.9 did not authorize a model diagnostic")
    if decision.get("supported_single_features") != [FEATURE_NAME]:
        raise ValueError("The Phase 2.9 supported-feature set changed")

    catalog = _read_json(INPUTS["phase29_catalog"][0])
    matches = [
        row for row in catalog.get("features", []) if row.get("name") == FEATURE_NAME
    ]
    if len(matches) != 1 or matches[0].get("kind") != "continuous":
        raise ValueError("The frozen scope feature is absent or changed")

    manifest = _read_json(INPUTS["phase29_cache_manifest"][0])
    extractor_hash = _sha256(IMPLEMENTATIONS["phase29_feature_extractor"])
    if int(manifest.get("cache_version", -1)) != 2:
        raise ValueError("Phase 2.9 corpus cache is not v2")
    if manifest.get("cache_sha256") != INPUTS["phase29_cache"][1]:
        raise ValueError("Phase 2.9 cache manifest does not match the frozen cache")
    if (
        manifest.get("implementation_hashes", {}).get("feature_extractor")
        != extractor_hash
    ):
        raise ValueError("Current extractor differs from the extractor frozen with cache v2")
    return {
        "decision": decision["decision"],
        "supported_features": decision["supported_single_features"],
        "feature_catalog_entry": matches[0],
        "cache_version": 2,
        "cache_sha256": manifest["cache_sha256"],
        "extractor_sha256": extractor_hash,
    }


def _write_preflight_companions(data: RouterData, baseline: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> None:
    preflight_path = RUN_DIR / "preflight.json"
    implementation_path = RUN_DIR / "implementation_freeze.json"
    split_path = RUN_DIR / "split_manifest.json"
    implementation = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_phase30_feature_extraction_and_oof",
        "preflight_sha256": _sha256(preflight_path),
        "implementation_hashes": _implementation_hashes(),
        "config_sha256": _sha256(CONFIG_PATH),
        "plan_sha256": _sha256(PLAN_PATH),
    }
    _write_json(implementation_path, implementation)
    split_rows: list[dict[str, Any]] = []
    for seed in SPLIT_SEEDS:
        folds = baseline[seed][1]
        fold_counts = {
            str(fold): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds))
        }
        group_counts = {
            str(fold): int(len(np.unique(data.group_ids[folds == fold])))
            for fold in sorted(np.unique(folds))
        }
        split_rows.append(
            {
                "split_seed": seed,
                "fold_counts": fold_counts,
                "group_counts": group_counts,
                "fold_ids_sha256": hashlib.sha256(
                    np.asarray(folds, dtype=np.int64).tobytes(order="C")
                ).hexdigest(),
                "group_crossings": 0,
            }
        )
    _write_json(
        split_path,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "reused_exact_historical_M3_outer_folds",
            "queries": len(data.query_ids),
            "groups": len(np.unique(data.group_ids)),
            "source_predictions_sha256": INPUTS["frozen_m3_oof"][1],
            "splits": split_rows,
        },
    )


def _validate_preflight_companions() -> None:
    implementation_path = RUN_DIR / "implementation_freeze.json"
    split_path = RUN_DIR / "split_manifest.json"
    if not implementation_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("Phase 2.10 implementation/split freeze is incomplete")
    implementation = _read_json(implementation_path)
    if implementation.get("preflight_sha256") != _sha256(RUN_DIR / "preflight.json"):
        raise RuntimeError("Phase 2.10 implementation freeze belongs to another preflight")
    if implementation.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Phase 2.10 implementation freeze drifted")
    split = _read_json(split_path)
    if split.get("source_predictions_sha256") != INPUTS["frozen_m3_oof"][1]:
        raise RuntimeError("Phase 2.10 split manifest source drifted")
    if int(split.get("queries", -1)) != 12_000 or int(split.get("groups", -1)) != 11_964:
        raise RuntimeError("Phase 2.10 split manifest counts drifted")


def preflight() -> dict[str, Any]:
    path = RUN_DIR / "preflight.json"
    if path.is_file():
        return _assert_freeze()
    if RUN_DIR.is_dir() and any(RUN_DIR.iterdir()):
        raise RuntimeError("Phase 2.10 run directory has artifacts but no preflight")

    _validate_protocol_config()
    input_rows = _validate_input_hashes()
    phase29 = _assert_phase29_contract()
    training_config, data, validation = _load_training()
    if len(data.query_ids) != 12_000 or len(np.unique(data.group_ids)) != 11_964:
        raise ValueError("T2 row/group counts differ from the frozen protocol")
    counts = {
        "bm25_winner": int(np.sum(data.gap > TIE_ATOL)),
        "exact_tie": int(np.sum(np.abs(data.gap) <= TIE_ATOL)),
        "dense_winner": int(np.sum(data.gap < -TIE_ATOL)),
    }
    if counts != {"bm25_winner": 3000, "exact_tie": 6000, "dense_winner": 3000}:
        raise ValueError(f"T2 target counts differ: {counts}")
    if data.structured.shape != (12_000, 30) or data.embedding.shape != (12_000, 384):
        raise ValueError("T2 feature dimensions differ")
    baseline = _baseline_arrays(data)

    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "passed_and_frozen",
        "created_at_unix": time.time(),
        "input_hashes": input_rows,
        "implementation_hashes": _implementation_hashes(),
        "protocol_config_sha256": _sha256(CONFIG_PATH),
        "protocol_plan_sha256": _sha256(PLAN_PATH),
        "phase29_contract": phase29,
        "training": {
            "view": "T2_winner3000",
            "queries": len(data.query_ids),
            "groups": len(np.unique(data.group_ids)),
            "target": "mean_f1_bm25_minus_mean_f1_dense",
            "counts": counts,
            "structured_dimensions": 30,
            "embedding_dimensions": 384,
            "validation": validation,
            "split_seeds": list(SPLIT_SEEDS),
            "outer_folds": int(training_config["cross_validation"]["outer_folds"]),
            "inner_folds": int(training_config["cross_validation"]["inner_folds"]),
        },
        "candidates": {
            SCOPE_ONLY_ID: "standardized scope feature -> Ridge(alpha=10)",
            BASELINE_ID: "frozen Phase 2.8 calibrated OOF predictions",
            AUGMENTED_ID: "31D standardized structured + fold-internal PCA32 -> Ridge(alpha=10)",
        },
        "primary_comparison": f"{AUGMENTED_ID}_minus_{BASELINE_ID}",
        "fixed_threshold": FIXED_THRESHOLD,
        "bootstrap": {
            "unit": "group_id",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "strict_internal_gate": {
            "all_three_seed_paired_gains_positive": True,
            "ensemble_paired_gain_positive": True,
            "ensemble_paired_ci95_lower_above_zero": True,
            "augmented_gain_over_best_fixed_minimum": 0.01,
            "augmented_gain_over_best_fixed_ci95_lower_above_zero": True,
            "all_three_seed_gains_over_best_fixed_positive": True,
            "harmful_to_beneficial_mass_ratio_maximum": 0.5,
            "switch_coverage_minimum": 0.01,
            "switch_coverage_maximum": 0.5,
            "calibration_slope_minimum": 0.5,
            "calibration_slope_maximum": 1.5,
            "top_decile_realized_gap_positive": True,
        },
        "natural_boundary": {
            "role": "previously_consumed_retrospective_confirmation_not_final_test",
            "outcome_file_hash_verified": True,
            "outcome_rows_parsed": 0,
            "outcome_values_read": 0,
            "prediction_or_evaluation_stage_implemented_here": False,
            "advance_requires": "strict_internal_gate_pass_and_separate_prediction_freeze",
        },
        "official_final_holdout_rows": 0,
        "external_calls": 0,
    }
    _write_json(path, payload)
    _write_preflight_companions(data, baseline)
    return payload


def _assert_freeze() -> dict[str, Any]:
    path = RUN_DIR / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("Run preflight first")
    payload = _read_json(path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Existing preflight belongs to another protocol")
    expected_inputs = {row["name"]: row["sha256"] for row in payload["input_hashes"]}
    for name, (input_path, frozen_constant) in INPUTS.items():
        actual = _sha256(input_path)
        if actual != frozen_constant or actual != expected_inputs.get(name):
            raise RuntimeError(f"Input drift after preflight: {name}")
    current_implementations = _implementation_hashes()
    if current_implementations != payload.get("implementation_hashes"):
        raise RuntimeError("Implementation drift after preflight; start a new run directory")
    _validate_protocol_config()
    _assert_phase29_contract()
    _validate_preflight_companions()
    return payload


def _t2_feature_identity() -> tuple[np.ndarray, np.ndarray]:
    with np.load(INPUTS["t2_features"][0], allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
    if query_ids.shape != (12_000,) or group_ids.shape != (12_000,):
        raise ValueError("T2 feature identity arrays have unexpected shapes")
    if len(set(query_ids.tolist())) != 12_000:
        raise ValueError("T2 feature identity contains duplicate query ids")
    return query_ids, group_ids


def _query_text_by_id(requested: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with INPUTS["query_source"][0].open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row.get("_id", row.get("id")))
            if query_id not in requested:
                continue
            if query_id in result:
                raise ValueError(f"Duplicate query-source id: {query_id}")
            result[query_id] = str(row.get("text", row.get("question", "")))
    if set(result) != requested:
        raise ValueError("Query source does not cover all requested T2 ids")
    return result


def _validate_extract_artifact() -> dict[str, Any]:
    feature_path = FEATURE_PATH
    manifest_path = FEATURE_MANIFEST_PATH
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Scope feature artifact is incomplete")
    manifest = _read_json(manifest_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "feature_sha256": _sha256(feature_path),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
        "cache_sha256": INPUTS["phase29_cache"][1],
        "extractor_sha256": _sha256(IMPLEMENTATIONS["phase29_feature_extractor"]),
        "discovery_source_sha256": INPUTS["phase29_discovery_features"][1],
        "natural_source_sha256": INPUTS["phase29_natural_features"][1],
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Refusing stale scope feature artifact: {actual}")
    return manifest


def extract() -> dict[str, Any]:
    _assert_freeze()
    feature_path = FEATURE_PATH
    manifest_path = FEATURE_MANIFEST_PATH
    if feature_path.is_file() or manifest_path.is_file():
        return _validate_extract_artifact()

    query_ids, group_ids = _t2_feature_identity()

    with np.load(
        INPUTS["phase29_discovery_features"][0], allow_pickle=False
    ) as stored:
        discovery_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        names = [str(value) for value in stored["feature_names"]]
        if names.count(FEATURE_NAME) != 1:
            raise ValueError("Phase 2.9 discovery matrix lacks the unique scope feature")
        discovery_values = np.asarray(
            stored["matrix"][:, names.index(FEATURE_NAME)], dtype=np.float64
        )
    if len(discovery_ids) != 6000 or len(set(discovery_ids.tolist())) != 6000:
        raise ValueError("Phase 2.9 discovery ids differ")
    if not set(map(str, discovery_ids)).issubset(set(map(str, query_ids))):
        raise ValueError("Phase 2.9 discovery ids are not a T2 subset")

    with np.load(
        INPUTS["phase29_natural_features"][0], allow_pickle=False
    ) as stored:
        natural_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        natural_names = [str(value) for value in stored["feature_names"]]
        if natural_names.count(FEATURE_NAME) != 1:
            raise ValueError("Phase 2.9 natural matrix lacks the unique scope feature")
        natural_frozen_values = np.asarray(
            stored["matrix"][:, natural_names.index(FEATURE_NAME)], dtype=np.float64
        )
    if len(natural_ids) != 2000 or len(set(natural_ids.tolist())) != 2000:
        raise ValueError("Phase 2.9 natural feature ids differ")
    if set(map(str, natural_ids)) & set(map(str, query_ids)):
        raise ValueError("Phase 2.9 natural queries overlap T2")

    requested_ids = set(map(str, query_ids)) | set(map(str, natural_ids))
    query_text = _query_text_by_id(requested_ids)

    position = {str(query_id): index for index, query_id in enumerate(query_ids)}
    values = np.full(len(query_ids), np.nan, dtype=np.float64)
    natural_values = np.full(len(natural_ids), np.nan, dtype=np.float64)
    cache = CorpusCache(INPUTS["phase29_cache"][0])
    started = time.perf_counter()
    try:
        for index, query_id in enumerate(query_ids):
            question = query_text[str(query_id)]
            values[index] = float(_question_features(question, cache)[FEATURE_NAME])
            if (index + 1) % 500 == 0:
                print(f"scope all T2: {index + 1}/{len(query_ids)}", flush=True)
        for index, query_id in enumerate(natural_ids):
            question = query_text[str(query_id)]
            natural_values[index] = float(
                _question_features(question, cache)[FEATURE_NAME]
            )
            if (index + 1) % 500 == 0:
                print(
                    f"scope natural parity: {index + 1}/{len(natural_ids)}",
                    flush=True,
                )
    finally:
        cache.close()
    if not np.isfinite(values).all():
        raise RuntimeError("Extracted scope feature contains non-finite values")

    parity = np.asarray(
        [values[position[str(query_id)]] for query_id in discovery_ids],
        dtype=np.float64,
    )
    if not np.array_equal(parity, discovery_values):
        raise RuntimeError("The 6000 Phase 2.9 discovery values changed")
    if not np.array_equal(natural_values, natural_frozen_values):
        maximum = float(np.max(np.abs(natural_values - natural_frozen_values)))
        raise RuntimeError(
            "The 2000 Phase 2.9 natural scope values changed: "
            f"max_abs={maximum}"
        )

    _write_npz(
        feature_path,
        query_ids=query_ids,
        group_ids=group_ids,
        values=values,
        feature_name=np.asarray([FEATURE_NAME], dtype=np.str_),
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "feature": FEATURE_NAME,
        "rows": len(values),
        "independently_extracted_T2_rows": len(values),
        "phase29_non_ties_independently_validated": len(discovery_ids),
        "phase29_natural_rows_independently_validated": len(natural_ids),
        "discovery_parity_exact": True,
        "discovery_max_absolute_difference": 0.0,
        "natural_parity_exact": True,
        "natural_max_absolute_difference": 0.0,
        "feature_path": _relative(feature_path),
        "feature_sha256": _sha256(feature_path),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
        "cache_sha256": INPUTS["phase29_cache"][1],
        "extractor_sha256": _sha256(IMPLEMENTATIONS["phase29_feature_extractor"]),
        "discovery_source_sha256": INPUTS["phase29_discovery_features"][1],
        "natural_source_sha256": INPUTS["phase29_natural_features"][1],
        "elapsed_seconds": time.perf_counter() - started,
        "natural_outcome_rows_parsed": 0,
        "official_final_holdout_rows": 0,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_scope(data: RouterData) -> np.ndarray:
    _validate_extract_artifact()
    with np.load(FEATURE_PATH, allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        values = np.asarray(stored["values"], dtype=np.float64)
    if not np.array_equal(query_ids, data.query_ids):
        raise ValueError("Scope feature query order differs from T2")
    if not np.array_equal(group_ids, data.group_ids):
        raise ValueError("Scope feature group order differs from T2")
    if values.shape != data.gap.shape or not np.isfinite(values).all():
        raise ValueError("Scope feature values are invalid")
    return values


def _utility(data: RouterData, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != data.gap.shape or not np.isfinite(values).all():
        raise ValueError("Policy scores are invalid")
    switches = values > FIXED_THRESHOLD
    return np.where(switches, data.bm25, data.dense)


def _common_group_bootstrap_cis(
    values: Mapping[str, np.ndarray], groups: np.ndarray
) -> dict[str, list[float]]:
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or len(np.unique(groups)) < 2:
        raise ValueError("Grouped bootstrap requires at least two aligned groups")
    names = list(values)
    arrays = [np.asarray(values[name], dtype=np.float64) for name in names]
    if any(array.shape != groups.shape or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Grouped bootstrap arrays must be aligned and finite")
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(str(group), []).append(index)
    ordered = sorted(by_group)
    group_sums = np.asarray(
        [
            [float(np.sum(array[by_group[group]])) for array in arrays]
            for group in ordered
        ],
        dtype=np.float64,
    )
    group_counts = np.asarray(
        [len(by_group[group]) for group in ordered], dtype=np.float64
    )
    estimates = np.empty((BOOTSTRAP_RESAMPLES, len(names)), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAP_RESAMPLES, 128):
        stop = min(start + 128, BOOTSTRAP_RESAMPLES)
        sampled = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        denominator = np.sum(group_counts[sampled], axis=1)
        estimates[start:stop] = np.sum(group_sums[sampled], axis=1) / denominator[:, None]
    return {
        name: [float(value) for value in np.quantile(estimates[:, index], [0.025, 0.975])]
        for index, name in enumerate(names)
    }


def _best_fixed(data: RouterData) -> tuple[str, np.ndarray]:
    if float(np.mean(data.bm25)) > float(np.mean(data.dense)):
        return "bm25", data.bm25
    return "dense", data.dense


def _point_policy_metrics(
    data: RouterData, scores: np.ndarray, *, best_fixed_action: str, best_fixed: np.ndarray
) -> dict[str, Any]:
    metrics = policy_metrics(
        data,
        scores,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=100,
        with_ci=False,
    )
    utility = _utility(data, scores)
    metrics.update(
        {
            "best_fixed_action": best_fixed_action,
            "best_fixed_mean_f1": float(np.mean(best_fixed)),
            "gain_over_best_fixed": float(np.mean(utility - best_fixed)),
        }
    )
    return metrics


def _compact_fit(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": result["candidate_id"],
        "split_seed": result["split_seed"],
        "folds": [
            {
                "outer_fold": row["outer_fold"],
                "train_queries": row["train_queries"],
                "validation_queries": row["validation_queries"],
                "calibration": row["calibration"],
                "fit_metadata": row["fit_metadata"],
            }
            for row in result["folds"]
        ],
    }


def _write_internal_comparison(
    path: Path,
    seed_rows: list[dict[str, Any]],
    ensemble: Mapping[str, Any],
) -> None:
    fields = [
        "level",
        "split_seed",
        "candidate_id",
        "router_mean_f1",
        "gain_over_fixed_dense",
        "gain_over_best_fixed",
        "gain_over_best_fixed_ci95_lower",
        "gain_over_best_fixed_ci95_upper",
        "switch_coverage",
        "harmful_to_beneficial_mass_ratio",
        "paired_M3S_minus_M3",
        "paired_M3S_minus_M3_ci95_lower",
        "paired_M3S_minus_M3_ci95_upper",
    ]
    rows: list[dict[str, Any]] = []
    for level, split_seed, metrics_by_candidate, paired, paired_ci in [
        *[
            (
                "split_seed",
                row["split_seed"],
                {
                    BASELINE_ID: row["baseline_metrics"],
                    SCOPE_ONLY_ID: row["scope_only_metrics"],
                    AUGMENTED_ID: row["augmented_metrics"],
                },
                row["paired_incremental_gain"],
                row["paired_incremental_gain_ci95"],
            )
            for row in seed_rows
        ],
        (
            "consensus",
            "mean_three_split_seeds",
            ensemble["metrics"],
            ensemble["paired_incremental_gain"],
            ensemble["paired_incremental_gain_ci95"],
        ),
    ]:
        for candidate_id, metrics in metrics_by_candidate.items():
            ci = metrics["gain_over_best_fixed_ci95"]
            rows.append(
                {
                    "level": level,
                    "split_seed": split_seed,
                    "candidate_id": candidate_id,
                    "router_mean_f1": metrics["router_mean_f1"],
                    "gain_over_fixed_dense": metrics["gain_over_fixed_dense"],
                    "gain_over_best_fixed": metrics["gain_over_best_fixed"],
                    "gain_over_best_fixed_ci95_lower": ci[0],
                    "gain_over_best_fixed_ci95_upper": ci[1],
                    "switch_coverage": metrics["switch_coverage"],
                    "harmful_to_beneficial_mass_ratio": metrics[
                        "harmful_to_beneficial_mass_ratio"
                    ],
                    "paired_M3S_minus_M3": (
                        paired if candidate_id == AUGMENTED_ID else ""
                    ),
                    "paired_M3S_minus_M3_ci95_lower": (
                        paired_ci[0] if candidate_id == AUGMENTED_ID else ""
                    ),
                    "paired_M3S_minus_M3_ci95_upper": (
                        paired_ci[1] if candidate_id == AUGMENTED_ID else ""
                    ),
                }
            )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate_oof_artifacts() -> dict[str, Any]:
    prediction_path = OOF_PREDICTION_PATH
    result_path = OOF_RESULT_PATH
    if (
        not prediction_path.is_file()
        or not result_path.is_file()
        or not COMPARISON_PATH.is_file()
    ):
        raise FileNotFoundError("Internal OOF artifacts are incomplete")
    result = _read_json(result_path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "predictions_sha256": _sha256(prediction_path),
        "scope_feature_sha256": _sha256(FEATURE_PATH),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
        "frozen_m3_oof_sha256": INPUTS["frozen_m3_oof"][1],
        "comparison_sha256": _sha256(COMPARISON_PATH),
    }
    actual = {key: result.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Refusing stale internal OOF artifacts: {actual}")
    return result


def oof() -> dict[str, Any]:
    _assert_freeze()
    _validate_extract_artifact()
    prediction_path = OOF_PREDICTION_PATH
    result_path = OOF_RESULT_PATH
    if prediction_path.is_file() or result_path.is_file() or COMPARISON_PATH.is_file():
        return _validate_oof_artifacts()

    training_config, data, _ = _load_training()
    scope = _load_scope(data)
    baseline = _baseline_arrays(data)
    scope_data = replace(
        data,
        lexical=np.empty((len(data.query_ids), 0), dtype=np.float64),
        corpus=scope.reshape(-1, 1),
    )
    augmented_data = replace(
        data,
        corpus=np.column_stack([data.corpus, scope]),
    )
    baseline_spec = CandidateSpec(
        BASELINE_ID, "M3", "pca_structured", "ridge", pca_dim=32
    )
    scope_spec = CandidateSpec(
        SCOPE_ONLY_ID, "P30", "structured", "ridge"
    )
    augmented_spec = CandidateSpec(
        AUGMENTED_ID, "P30", "pca_structured", "ridge", pca_dim=32
    )

    formal = _read_json(INPUTS["frozen_m3_metrics"][0])
    formal_splits = formal["candidate_splits"][BASELINE_ID]
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(data.query_ids, dtype=np.str_),
        "group_ids": np.asarray(data.group_ids, dtype=np.str_),
    }
    seed_rows: list[dict[str, Any]] = []
    all_scores: dict[str, list[np.ndarray]] = {
        BASELINE_ID: [],
        SCOPE_ONLY_ID: [],
        AUGMENTED_ID: [],
    }
    best_fixed_action, best_fixed = _best_fixed(data)
    started = time.perf_counter()
    for seed in SPLIT_SEEDS:
        print(f"Phase 2.10 OOF seed={seed}", flush=True)
        frozen_baseline_scores, frozen_baseline_folds = baseline[seed]
        baseline_fit, baseline_scores, baseline_folds = run_candidate_split(
            training_config,
            data,
            baseline_spec,
            split_seed=seed,
            model_seeds=[MODEL_SEED],
            bootstrap_resamples=100,
        )
        if not np.array_equal(baseline_folds, frozen_baseline_folds):
            raise RuntimeError(f"Rerun M3 fold ids differ for seed {seed}")
        if not np.allclose(
            baseline_scores,
            frozen_baseline_scores,
            rtol=1e-10,
            atol=1e-12,
        ):
            maximum = float(np.max(np.abs(baseline_scores - frozen_baseline_scores)))
            raise RuntimeError(
                f"Rerun M3 predictions differ for seed {seed}: max_abs={maximum}"
            )
        baseline_metrics = _point_policy_metrics(
            data,
            baseline_scores,
            best_fixed_action=best_fixed_action,
            best_fixed=best_fixed,
        )
        expected_gain = float(formal_splits[str(seed)]["metrics"]["gain_over_fixed_dense"])
        if not math.isclose(
            float(baseline_metrics["gain_over_fixed_dense"]),
            expected_gain,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise RuntimeError("Frozen M3 prediction order does not reproduce its metric")

        scope_fit, scope_scores, scope_folds = run_candidate_split(
            training_config,
            scope_data,
            scope_spec,
            split_seed=seed,
            model_seeds=[MODEL_SEED],
            bootstrap_resamples=100,
        )
        augmented_fit, augmented_scores, augmented_folds = run_candidate_split(
            training_config,
            augmented_data,
            augmented_spec,
            split_seed=seed,
            model_seeds=[MODEL_SEED],
            bootstrap_resamples=100,
        )
        if not np.array_equal(scope_folds, baseline_folds):
            raise RuntimeError(f"Scope-only fold ids differ for seed {seed}")
        if not np.array_equal(augmented_folds, baseline_folds):
            raise RuntimeError(f"Augmented M3 fold ids differ for seed {seed}")

        scope_metrics = _point_policy_metrics(
            data,
            scope_scores,
            best_fixed_action=best_fixed_action,
            best_fixed=best_fixed,
        )
        augmented_metrics = _point_policy_metrics(
            data,
            augmented_scores,
            best_fixed_action=best_fixed_action,
            best_fixed=best_fixed,
        )
        utilities = {
            BASELINE_ID: _utility(data, baseline_scores),
            SCOPE_ONLY_ID: _utility(data, scope_scores),
            AUGMENTED_ID: _utility(data, augmented_scores),
        }
        paired = utilities[AUGMENTED_ID] - utilities[BASELINE_ID]
        bootstrap_values = {
            f"gain__{candidate_id}": utility - best_fixed
            for candidate_id, utility in utilities.items()
        }
        bootstrap_values["paired"] = paired
        confidence_intervals = _common_group_bootstrap_cis(
            bootstrap_values, data.group_ids
        )
        for candidate_id, metrics in (
            (BASELINE_ID, baseline_metrics),
            (SCOPE_ONLY_ID, scope_metrics),
            (AUGMENTED_ID, augmented_metrics),
        ):
            metrics["gain_over_best_fixed_ci95"] = confidence_intervals[
                f"gain__{candidate_id}"
            ]
        seed_rows.append(
            {
                "split_seed": seed,
                "fixed_threshold": FIXED_THRESHOLD,
                "baseline_metrics": baseline_metrics,
                "scope_only_metrics": scope_metrics,
                "augmented_metrics": augmented_metrics,
                "paired_incremental_gain": float(np.mean(paired)),
                "paired_incremental_gain_ci95": confidence_intervals["paired"],
                "baseline_replication": {
                    "fold_ids_exact": True,
                    "predictions_allclose": True,
                    "rtol": 1e-10,
                    "atol": 1e-12,
                    "maximum_absolute_difference": float(
                        np.max(np.abs(baseline_scores - frozen_baseline_scores))
                    ),
                },
                "baseline_fit": _compact_fit(baseline_fit),
                "scope_fit": _compact_fit(scope_fit),
                "augmented_fit": _compact_fit(augmented_fit),
            }
        )
        for candidate_id, scores in (
            (BASELINE_ID, baseline_scores),
            (SCOPE_ONLY_ID, scope_scores),
            (AUGMENTED_ID, augmented_scores),
        ):
            arrays[f"prediction__{candidate_id}__{seed}"] = scores
            all_scores[candidate_id].append(scores)
        arrays[f"fold_id__{seed}"] = baseline_folds

    ensemble = {
        candidate_id: np.mean(np.stack(values, axis=0), axis=0)
        for candidate_id, values in all_scores.items()
    }
    for candidate_id, scores in ensemble.items():
        arrays[f"prediction_mean__{candidate_id}"] = scores
    ensemble_metrics = {
        candidate_id: _point_policy_metrics(
            data,
            scores,
            best_fixed_action=best_fixed_action,
            best_fixed=best_fixed,
        )
        for candidate_id, scores in ensemble.items()
    }
    ensemble_paired = (
        _utility(data, ensemble[AUGMENTED_ID])
        - _utility(data, ensemble[BASELINE_ID])
    )
    ensemble_utilities = {
        candidate_id: _utility(data, scores)
        for candidate_id, scores in ensemble.items()
    }
    ensemble_bootstrap_values = {
        f"gain__{candidate_id}": utility - best_fixed
        for candidate_id, utility in ensemble_utilities.items()
    }
    ensemble_bootstrap_values["paired"] = ensemble_paired
    ensemble_cis = _common_group_bootstrap_cis(
        ensemble_bootstrap_values, data.group_ids
    )
    for candidate_id, metrics in ensemble_metrics.items():
        metrics["gain_over_best_fixed_ci95"] = ensemble_cis[
            f"gain__{candidate_id}"
        ]
    ensemble_summary = {
        "fixed_threshold": FIXED_THRESHOLD,
        "metrics": ensemble_metrics,
        "paired_incremental_gain": float(np.mean(ensemble_paired)),
        "paired_incremental_gain_ci95": ensemble_cis["paired"],
        "augmented_gain_over_best_fixed": ensemble_metrics[AUGMENTED_ID][
            "gain_over_best_fixed"
        ],
        "augmented_gain_over_best_fixed_ci95": ensemble_metrics[AUGMENTED_ID][
            "gain_over_best_fixed_ci95"
        ],
    }

    _write_npz(prediction_path, **arrays)
    _write_internal_comparison(COMPARISON_PATH, seed_rows, ensemble_summary)
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "queries": len(data.query_ids),
        "groups": len(np.unique(data.group_ids)),
        "feature": FEATURE_NAME,
        "fixed_threshold": FIXED_THRESHOLD,
        "split_seeds": seed_rows,
        "ensemble": ensemble_summary,
        "predictions": _relative(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "comparison": _relative(COMPARISON_PATH),
        "comparison_sha256": _sha256(COMPARISON_PATH),
        "scope_feature_sha256": _sha256(FEATURE_PATH),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
        "frozen_m3_oof_sha256": INPUTS["frozen_m3_oof"][1],
        "bootstrap": {
            "unit": "group_id",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "natural_outcome_rows_parsed": 0,
        "official_final_holdout_rows": 0,
        "external_calls": 0,
    }
    _write_json(result_path, result)
    return result


def _validate_decision() -> dict[str, Any]:
    path = DECISION_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _read_json(path)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "internal_metrics_sha256": _sha256(OOF_RESULT_PATH),
        "internal_oof_predictions_sha256": _sha256(OOF_PREDICTION_PATH),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
    }
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Refusing stale Phase 2.10 decision: {actual}")
    return value


def decision() -> dict[str, Any]:
    _assert_freeze()
    results = _validate_oof_artifacts()
    path = DECISION_PATH
    if path.is_file():
        return _validate_decision()

    seeds = results["split_seeds"]
    ensemble = results["ensemble"]
    augmented = ensemble["metrics"][AUGMENTED_ID]
    calibration = augmented["calibration"]
    ratio = augmented["harmful_to_beneficial_mass_ratio"]
    checks = {
        "fixed_threshold_is_zero": float(results["fixed_threshold"]) == 0.0,
        "all_three_seed_paired_gains_positive": all(
            float(row["paired_incremental_gain"]) > 0.0 for row in seeds
        ),
        "ensemble_paired_gain_positive": float(ensemble["paired_incremental_gain"])
        > 0.0,
        "ensemble_paired_ci95_lower_above_zero": float(
            ensemble["paired_incremental_gain_ci95"][0]
        )
        > 0.0,
        "augmented_gain_over_best_fixed_at_least_0_01": float(
            ensemble["augmented_gain_over_best_fixed"]
        )
        >= 0.01,
        "augmented_gain_over_best_fixed_ci95_lower_above_zero": float(
            ensemble["augmented_gain_over_best_fixed_ci95"][0]
        )
        > 0.0,
        "all_three_seed_augmented_gains_over_best_fixed_positive": all(
            float(row["augmented_metrics"]["gain_over_best_fixed"]) > 0.0
            for row in seeds
        ),
        "harmful_to_beneficial_mass_ratio_at_most_0_5": ratio is not None
        and float(ratio) <= 0.5,
        "switch_coverage_between_0_01_and_0_5": 0.01
        <= float(augmented["switch_coverage"])
        <= 0.5,
        "calibration_slope_between_0_5_and_1_5": 0.5
        <= float(calibration["slope"])
        <= 1.5,
        "top_decile_realized_gap_positive": float(
            calibration["top_decile_realized_gap"]
        )
        > 0.0,
    }
    passed = all(checks.values())
    value = {
        "protocol_id": PROTOCOL_ID,
        "decision": (
            "ADVANCE_TO_SEPARATELY_FROZEN_RETROSPECTIVE_NATURAL_DIAGNOSTIC"
            if passed
            else "STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL"
        ),
        "strict_internal_gate_passed": passed,
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "primary_comparison": f"{AUGMENTED_ID}_minus_{BASELINE_ID}",
        "ensemble_paired_incremental_gain": ensemble["paired_incremental_gain"],
        "ensemble_paired_incremental_gain_ci95": ensemble[
            "paired_incremental_gain_ci95"
        ],
        "natural_confirmation_role": (
            "previously_consumed_retrospective_confirmation_not_final_test"
        ),
        "natural_outcomes_opened_by_phase30": False,
        "natural_outcome_rows_parsed": 0,
        "natural_outcome_values_read": 0,
        "natural_next_step": (
            "freeze full-fit predictions before reading consumed natural outcomes"
            if passed
            else "forbidden_by_internal_gate"
        ),
        "combination_search_beyond_frozen_scope_feature": False,
        "official_final_holdout_rows": 0,
        "external_calls": 0,
        "internal_metrics_sha256": _sha256(OOF_RESULT_PATH),
        "internal_oof_predictions_sha256": _sha256(OOF_PREDICTION_PATH),
        "preflight_sha256": _sha256(RUN_DIR / "preflight.json"),
        "created_at_unix": time.time(),
    }
    _write_json(path, value)
    return value


def status() -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for name, filename in (
        ("preflight", "preflight.json"),
        ("implementation_freeze", "implementation_freeze.json"),
        ("split_manifest", "split_manifest.json"),
        ("extract", FEATURE_MANIFEST_PATH.name),
        ("oof", OOF_RESULT_PATH.name),
        ("comparison", COMPARISON_PATH.name),
        ("decision", DECISION_PATH.name),
    ):
        path = RUN_DIR / filename
        stages[name] = {
            "complete": path.is_file(),
            "path": _relative(path),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "stages": stages,
        "natural_outcome_rows_parsed_by_design": 0,
        "official_final_holdout_rows": 0,
    }
    if DECISION_PATH.is_file():
        frozen = _validate_decision()
        result["decision"] = frozen["decision"]
        result["strict_internal_gate_passed"] = frozen[
            "strict_internal_gate_passed"
        ]
        result["failed_checks"] = frozen["failed_checks"]
    return result


def run_all() -> dict[str, Any]:
    preflight()
    extract()
    oof()
    return decision()


def main() -> int:
    args = _arguments()
    actions = {
        "preflight": preflight,
        "extract": extract,
        "oof": oof,
        "decision": decision,
        "run": run_all,
        "status": status,
    }
    result = actions[args.stage]()
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
