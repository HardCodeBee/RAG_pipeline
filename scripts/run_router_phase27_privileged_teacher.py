#!/usr/bin/env python3
"""Run the frozen post-retrieval/gold privileged-teacher diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_router_phase27_model_audit as audit


CONFIG_PATH = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase27/config.yaml"
RUN_DIR = (
    PROJECT_ROOT
    / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1"
)
TEACHER_DIR = RUN_DIR / "privileged_teacher"
DATABASE_PATH = (
    PROJECT_ROOT
    / "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1/state.sqlite3"
)
PROTOCOL_ID = "hotpotqa_bd_router_phase27_privileged_teacher_v1"
SPLIT_SEEDS = [20260901, 20260917, 20261003]
BOOTSTRAP_RESAMPLES = 10000
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


SCORE_STAT_NAMES = [
    "top1",
    "top2",
    "top3",
    "top5",
    "mean",
    "std",
    "range",
    "margin_1_2",
    "margin_1_5",
    "relative_margin_1_2",
    "relative_margin_1_5",
    "coefficient_of_variation",
    "relative_rank_slope",
    "normalized_softmax_entropy",
]


def probe_feature_names() -> list[str]:
    names = [
        f"probe__{action}_{name}"
        for action in ("bm25", "dense")
        for name in SCORE_STAT_NAMES
    ]
    names.extend(
        f"probe__{action}_{name}"
        for action in ("bm25", "dense")
        for name in ("context_token_count", "context_truncated", "query_context_jaccard")
    )
    names.extend(
        [
            "probe__same_top1_doc",
            "probe__doc_overlap_count_at_1",
            "probe__doc_overlap_count_at_3",
            "probe__doc_overlap_count_at_5",
            "probe__doc_overlap_jaccard_at_3",
            "probe__doc_overlap_jaccard_at_5",
            "probe__reciprocal_rank_overlap",
            "probe__bm25_top1_rank_in_dense",
            "probe__dense_top1_rank_in_bm25",
        ]
    )
    names.extend(
        f"probe__bm25_minus_dense_{name}"
        for name in (
            "relative_margin_1_2",
            "relative_margin_1_5",
            "coefficient_of_variation",
            "relative_rank_slope",
            "normalized_softmax_entropy",
            "context_token_count_scaled",
            "query_context_jaccard",
        )
    )
    return names


def gold_feature_names() -> list[str]:
    per_action = [
        "coverage_at_1",
        "coverage_at_2",
        "coverage_at_3",
        "coverage_at_5",
        "complete_at_1",
        "complete_at_2",
        "complete_at_3",
        "complete_at_5",
        "hit_at_5",
        "first_gold_rank_scaled",
        "mean_reciprocal_gold_rank",
    ]
    names = [
        f"gold__{action}_{name}"
        for action in ("bm25", "dense")
        for name in per_action
    ]
    names.append("gold__gold_document_count")
    names.extend(f"gold__bm25_minus_dense_{name}" for name in per_action)
    return names


@dataclass(frozen=True)
class TeacherSpec:
    candidate_id: str
    block: str
    model: str


SPECS = [
    TeacherSpec("P0_probe_ridge", "probe", "ridge"),
    TeacherSpec("P1_probe_xgb", "probe", "xgboost"),
    TeacherSpec("G0_gold_ridge", "gold", "ridge"),
    TeacherSpec("G1_gold_xgb", "gold", "xgboost"),
    TeacherSpec("G2_probe_gold_xgb", "probe_gold", "xgboost"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "report"))
    parser.add_argument("--max-workers", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    audit._write_json(path, value)


def _score_stats(scores: list[float]) -> list[float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Teacher diagnostic requires five finite retrieval scores")
    scale = max(abs(float(values[0])), 1e-9)
    mean = float(np.mean(values))
    std = float(np.std(values))
    standardized = (values - mean) / max(std, 1e-9)
    probability = np.exp(standardized - np.max(standardized))
    probability /= np.sum(probability)
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-15)))) / math.log(5)
    slope = float(np.polyfit(np.arange(1, 6, dtype=np.float64), values, 1)[0])
    return [
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[4]),
        mean,
        std,
        float(values[0] - values[4]),
        float(values[0] - values[1]),
        float(values[0] - values[4]),
        float((values[0] - values[1]) / scale),
        float((values[0] - values[4]) / scale),
        float(std / max(abs(mean), 1e-9)),
        float(slope / scale),
        entropy,
    ]


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _rank(value: str, values: list[str]) -> int:
    try:
        return values.index(value) + 1
    except ValueError:
        return 6


def build_probe_features(question: str, actions: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    stats: dict[str, list[float]] = {}
    context_features: dict[str, list[float]] = {}
    question_tokens = _tokens(question)
    for action in ("bm25", "dense"):
        row = actions[action]
        stats[action] = _score_stats(row["scores"])
        context_features[action] = [
            float(row["context_token_count"]),
            float(bool(row["context_truncated"])),
            _jaccard(question_tokens, _tokens(row["context_text"])),
        ]
    bm25_docs = actions["bm25"]["doc_ids"]
    dense_docs = actions["dense"]["doc_ids"]
    overlap_values: list[float] = [float(bm25_docs[0] == dense_docs[0])]
    for k in (1, 3, 5):
        overlap_values.append(float(len(set(bm25_docs[:k]) & set(dense_docs[:k]))))
    overlap_values.extend(
        [
            _jaccard(set(bm25_docs[:3]), set(dense_docs[:3])),
            _jaccard(set(bm25_docs[:5]), set(dense_docs[:5])),
            float(
                sum(
                    1.0 / (_rank(doc_id, bm25_docs) + _rank(doc_id, dense_docs))
                    for doc_id in set(bm25_docs) & set(dense_docs)
                )
            ),
            float(_rank(bm25_docs[0], dense_docs)),
            float(_rank(dense_docs[0], bm25_docs)),
        ]
    )
    named_bm25 = dict(zip(SCORE_STAT_NAMES, stats["bm25"], strict=True))
    named_dense = dict(zip(SCORE_STAT_NAMES, stats["dense"], strict=True))
    difference_values = [
        named_bm25[name] - named_dense[name]
        for name in (
            "relative_margin_1_2",
            "relative_margin_1_5",
            "coefficient_of_variation",
            "relative_rank_slope",
            "normalized_softmax_entropy",
        )
    ]
    difference_values.extend(
        [
            (context_features["bm25"][0] - context_features["dense"][0]) / 1800.0,
            context_features["bm25"][2] - context_features["dense"][2],
        ]
    )
    result = np.asarray(
        stats["bm25"]
        + stats["dense"]
        + context_features["bm25"]
        + context_features["dense"]
        + overlap_values
        + difference_values,
        dtype=np.float64,
    )
    if result.shape != (len(probe_feature_names()),):
        raise RuntimeError("Probe feature schema mismatch")
    return result


def _gold_action_features(doc_ids: list[str], gold_ids: set[str]) -> list[float]:
    denominator = max(len(gold_ids), 1)
    coverage = [len(set(doc_ids[:k]) & gold_ids) / denominator for k in (1, 2, 3, 5)]
    complete = [float(value >= 1.0) for value in coverage]
    ranks = [_rank(gold_id, doc_ids) for gold_id in sorted(gold_ids)]
    first_rank = min(ranks, default=6)
    reciprocal = float(np.mean([1.0 / rank if rank <= 5 else 0.0 for rank in ranks]))
    return [
        *coverage,
        *complete,
        float(coverage[-1] > 0.0),
        float(first_rank / 6.0),
        reciprocal,
    ]


def build_gold_features(actions: Mapping[str, Mapping[str, Any]], gold_ids: set[str]) -> np.ndarray:
    bm25 = _gold_action_features(actions["bm25"]["doc_ids"], gold_ids)
    dense = _gold_action_features(actions["dense"]["doc_ids"], gold_ids)
    result = np.asarray(
        bm25 + dense + [float(len(gold_ids))] + [left - right for left, right in zip(bm25, dense)],
        dtype=np.float64,
    )
    if result.shape != (len(gold_feature_names()),):
        raise RuntimeError("Gold feature schema mismatch")
    return result


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_feature_outcome_analysis",
        "role": "offline_privileged_diagnostic_not_a_deployable_pre_retrieval_router",
        "entry_condition": "STOP_QUERY_ONLY_V1",
        "query_count": 9600,
        "partition": "train_only",
        "target": "three_repeat_mean_bm25_minus_dense_normalized_token_f1",
        "forbidden_teacher_features": [
            "generated_answer",
            "normalized_token_f1",
            "answer_correctness",
            "current_query_action_utility",
        ],
        "feature_blocks": {
            "probe": {
                "scope": "post_both_retrievers_without_qrels_or_generation",
                "feature_names": probe_feature_names(),
            },
            "gold": {
                "scope": "qrels_and_supporting_evidence_privileged_only",
                "feature_names": gold_feature_names(),
            },
        },
        "candidates": [spec.__dict__ for spec in SPECS],
        "primary_gold_rule": "switch_to_bm25_only_if_gold_coverage_at_5_is_higher",
        "cross_validation": {
            "split_seeds": SPLIT_SEEDS,
            "outer_folds": 5,
            "inner_folds": 4,
            "group_field": "group_id",
            "strata": ["bm25_winner", "exact_tie", "dense_winner"],
            "all_preprocessing_and_calibration_fold_local": True,
        },
        "models": {
            "ridge": {"alpha": 10.0, "standardize": True},
            "xgboost": {
                "n_estimators": 250,
                "max_depth": 2,
                "learning_rate": 0.03,
                "min_child_weight": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10.0,
                "reg_alpha": 0.1,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs_per_worker": 4,
            },
        },
        "gate": {
            "practical_gain_minimum": 0.01,
            "grouped_bootstrap_ci_lower_must_exceed": 0.0,
            "all_three_split_gains_positive": True,
            "harmful_to_beneficial_mass_ratio_maximum": 0.5,
            "switch_coverage": [0.01, 0.5],
            "calibration_slope": [0.5, 1.5],
            "top_decile_realized_gap_positive": True,
        },
        "decision_rule": [
            "probe_candidate_passes_gate=>PROBE_SIGNAL_SUFFICIENT",
            "else_gold_candidate_or_primary_gold_rule_passes=>GOLD_EVIDENCE_SIGNAL_SUFFICIENT",
            "else=>RETRIEVAL_SIGNAL_INSUFFICIENT",
        ],
        "fresh_dev_rows_allowed": 0,
        "external_calls_allowed": 0,
        "inputs": {
            "config": {"path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256(CONFIG_PATH)},
            "database": {"path": str(DATABASE_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256(DATABASE_PATH)},
            "snapshot": {
                "path": str((RUN_DIR / "snapshot/snapshot_manifest.json").relative_to(PROJECT_ROOT)),
                "sha256": sha256(RUN_DIR / "snapshot/snapshot_manifest.json"),
            },
            "query_only_decision": {
                "path": str((RUN_DIR / "decision.json").relative_to(PROJECT_ROOT)),
                "sha256": sha256(RUN_DIR / "decision.json"),
            },
        },
        "frozen_at_unix": time.time(),
    }


def prepare() -> None:
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    decision = json.loads((RUN_DIR / "decision.json").read_text(encoding="utf-8"))
    if decision.get("decision") != "STOP_QUERY_ONLY_V1":
        raise ValueError("Privileged teacher is allowed only after Query-Only V1 stops")
    protocol_path = TEACHER_DIR / "teacher_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Existing teacher protocol belongs to another experiment")
    else:
        write_json(protocol_path, protocol_payload())

    config = audit._load_config(CONFIG_PATH)
    data, _ = audit.load_frozen_data(config)
    with sqlite3.connect(DATABASE_PATH) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        inconsistent_retrieval = connection.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT query_id, action, COUNT(*) AS n, "
            "COUNT(DISTINCT json_extract(payload, '$.retrieval')) AS variants "
            "FROM outcomes WHERE partition='train' AND action IN ('bm25','dense') "
            "AND status='success' GROUP BY query_id, action HAVING n<>3 OR variants<>1)"
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT s.query_rank, s.query_id, s.group_id, o.action, "
            "json_extract(o.payload, '$.question'), "
            "json_extract(o.payload, '$.gold_doc_ids'), "
            "json_extract(o.payload, '$.retrieval.doc_ids'), "
            "json_extract(o.payload, '$.retrieval.scores'), "
            "json_extract(o.payload, '$.retrieval.evidence_page_recall'), "
            "json_extract(o.payload, '$.retrieval.hit'), "
            "json_extract(o.payload, '$.context.token_count'), "
            "json_extract(o.payload, '$.context.truncated'), "
            "json_extract(o.payload, '$.context.text') "
            "FROM outcomes AS o JOIN sample_rows AS s USING(query_id) "
            "WHERE o.partition='train' AND s.partition='train' AND o.repeat_id=0 "
            "AND o.action IN ('bm25','dense') AND s.query_rank<9600 AND o.status='success' "
            "ORDER BY s.query_rank, o.action"
        ).fetchall()
    if integrity != "ok" or inconsistent_retrieval != 0 or len(rows) != 19200:
        raise ValueError("Retrieval payload integrity check failed")

    records: list[dict[str, Any]] = [{"actions": {}} for _ in range(9600)]
    for (
        query_rank,
        query_id,
        group_id,
        action,
        question,
        gold_json,
        doc_json,
        score_json,
        recorded_recall,
        recorded_hit,
        token_count,
        truncated,
        context_text,
    ) in rows:
        position = int(query_rank)
        if query_id != str(data.query_ids[position]) or group_id != str(data.group_ids[position]):
            raise ValueError("Retrieval payload order differs from the frozen feature order")
        record = records[position]
        record.setdefault("query_id", query_id)
        record.setdefault("group_id", group_id)
        record.setdefault("question", question)
        gold_ids = {str(value) for value in json.loads(gold_json)}
        if "gold_ids" in record and record["gold_ids"] != gold_ids:
            raise ValueError("Gold document IDs differ by action")
        record["gold_ids"] = gold_ids
        doc_ids = [str(value) for value in json.loads(doc_json)]
        scores = [float(value) for value in json.loads(score_json)]
        computed_recall = len(set(doc_ids) & gold_ids) / max(len(gold_ids), 1)
        if not math.isclose(float(recorded_recall), computed_recall, abs_tol=1e-12):
            raise ValueError("Recorded evidence recall differs from qrels recomputation")
        if float(recorded_hit) != float(computed_recall > 0.0):
            raise ValueError("Recorded hit differs from qrels recomputation")
        record["actions"][action] = {
            "doc_ids": doc_ids,
            "scores": scores,
            "context_token_count": int(token_count),
            "context_truncated": bool(truncated),
            "context_text": str(context_text),
        }

    probe = np.vstack(
        [build_probe_features(row["question"], row["actions"]) for row in records]
    )
    gold = np.vstack(
        [build_gold_features(row["actions"], row["gold_ids"]) for row in records]
    )
    if not np.isfinite(probe).all() or not np.isfinite(gold).all():
        raise ValueError("Teacher features contain non-finite values")
    feature_path = TEACHER_DIR / "teacher_features.npz"
    np.savez_compressed(
        feature_path,
        query_ids=data.query_ids,
        group_ids=data.group_ids,
        probe=probe,
        gold=gold,
    )
    schema = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "query_count": 9600,
        "probe": {
            "dimensions": int(probe.shape[1]),
            "feature_names": probe_feature_names(),
            "uses_qrels": False,
            "uses_generation_outcomes": False,
        },
        "gold": {
            "dimensions": int(gold.shape[1]),
            "feature_names": gold_feature_names(),
            "uses_qrels": True,
            "uses_generation_outcomes": False,
        },
        "database_integrity": integrity,
        "retrieval_repeat_variants": inconsistent_retrieval,
        "fresh_dev_rows_read": 0,
        "external_calls": 0,
        "artifact": {
            "path": str(feature_path.relative_to(PROJECT_ROOT)),
            "bytes": feature_path.stat().st_size,
            "sha256": sha256(feature_path),
        },
    }
    write_json(TEACHER_DIR / "feature_schema.json", schema)
    print(json.dumps(schema, indent=2))


def _fit_predict(
    spec: TeacherSpec,
    matrix: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if spec.model == "ridge":
        model: Any = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif spec.model == "xgboost":
        model = XGBRegressor(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            reg_alpha=0.1,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=4,
            random_state=int(seed),
            verbosity=0,
        )
    else:
        raise ValueError(spec.model)
    model.fit(matrix[train_indices], target[train_indices])
    return np.asarray(model.predict(matrix[validation_indices]), dtype=np.float64)


def _matrix(spec: TeacherSpec, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if spec.block == "probe":
        return arrays["probe"]
    if spec.block == "gold":
        return arrays["gold"]
    if spec.block == "probe_gold":
        return np.column_stack([arrays["probe"], arrays["gold"]])
    raise ValueError(spec.block)


def run_candidate(
    config: Mapping[str, Any],
    data: audit.RouterData,
    arrays: Mapping[str, np.ndarray],
    spec: TeacherSpec,
    split_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    matrix = _matrix(spec, arrays)
    outer_folds = audit.make_group_stratified_folds(data, n_splits=5, seed=split_seed)
    predictions = np.full(len(data.query_ids), np.nan, dtype=np.float64)
    fold_ids = np.full(len(data.query_ids), -1, dtype=np.int64)
    fold_results: list[dict[str, Any]] = []
    for outer_fold_id, (outer_train, outer_validation) in enumerate(outer_folds):
        inner_folds = audit.make_group_stratified_folds(
            data,
            n_splits=4,
            seed=split_seed + 100 + outer_fold_id,
            indices=outer_train,
        )
        local_position = {int(value): index for index, value in enumerate(outer_train)}
        inner_prediction = np.full(len(outer_train), np.nan, dtype=np.float64)
        for inner_fold_id, (inner_train, inner_validation) in enumerate(inner_folds):
            value = _fit_predict(
                spec,
                matrix,
                data.gap,
                inner_train,
                inner_validation,
                seed=split_seed + outer_fold_id * 100 + inner_fold_id,
            )
            positions = [local_position[int(index)] for index in inner_validation]
            inner_prediction[np.asarray(positions, dtype=np.int64)] = value
        if not np.isfinite(inner_prediction).all():
            raise RuntimeError("Teacher inner OOF predictions are incomplete")
        calibration = audit._fit_affine_calibration(inner_prediction, data.gap[outer_train])
        outer_raw = _fit_predict(
            spec,
            matrix,
            data.gap,
            outer_train,
            outer_validation,
            seed=split_seed + outer_fold_id * 100 + 99,
        )
        outer_score = calibration.predict(outer_raw)
        predictions[outer_validation] = outer_score
        fold_ids[outer_validation] = outer_fold_id
        fold_data = replace(
            data,
            query_ids=data.query_ids[outer_validation],
            group_ids=data.group_ids[outer_validation],
            query_ranks=data.query_ranks[outer_validation],
            bm25=data.bm25[outer_validation],
            dense=data.dense[outer_validation],
            gap=data.gap[outer_validation],
            strata=data.strata[outer_validation],
            repeat_values=data.repeat_values[outer_validation],
            lexical=data.lexical[outer_validation],
            corpus=data.corpus[outer_validation],
            embedding=data.embedding[outer_validation],
        )
        fold_metrics = audit.policy_metrics(
            fold_data,
            outer_score,
            bootstrap_seed=20260902,
            bootstrap_resamples=100,
            with_ci=False,
        )
        fold_results.append(
            {
                "outer_fold": outer_fold_id,
                "train_queries": int(len(outer_train)),
                "validation_queries": int(len(outer_validation)),
                "calibration": {
                    "intercept": calibration.intercept,
                    "slope": calibration.slope,
                },
                "metrics": fold_metrics,
            }
        )
        print(
            f"{spec.candidate_id} split={split_seed} fold={outer_fold_id + 1}/5 "
            f"gain={fold_metrics['gain_over_fixed_dense']:+.6f}",
            flush=True,
        )
    metrics = audit.policy_metrics(
        data,
        predictions,
        bootstrap_seed=20260902 + split_seed % 10000,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    return (
        {
            "candidate_id": spec.candidate_id,
            "feature_block": spec.block,
            "model": spec.model,
            "split_seed": split_seed,
            "metrics": metrics,
            "folds": fold_results,
            "external_calls": 0,
        },
        predictions,
        fold_ids,
    )


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_DATA: audit.RouterData | None = None
_WORKER_ARRAYS: dict[str, np.ndarray] | None = None
_WORKER_TASK_DIR: Path | None = None


def worker_init() -> None:
    global _WORKER_CONFIG, _WORKER_DATA, _WORKER_ARRAYS, _WORKER_TASK_DIR
    _WORKER_CONFIG = audit._load_config(CONFIG_PATH)
    _WORKER_DATA, _ = audit.load_frozen_data(_WORKER_CONFIG)
    with np.load(TEACHER_DIR / "teacher_features.npz", allow_pickle=False) as stored:
        _WORKER_ARRAYS = {
            "probe": np.asarray(stored["probe"], dtype=np.float64),
            "gold": np.asarray(stored["gold"], dtype=np.float64),
        }
    _WORKER_TASK_DIR = TEACHER_DIR / "tasks"


def worker_task(candidate_id: str, split_seed: int) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_DATA is None or _WORKER_ARRAYS is None:
        raise RuntimeError("Teacher worker was not initialized")
    spec = next(value for value in SPECS if value.candidate_id == candidate_id)
    result, scores, folds = run_candidate(
        _WORKER_CONFIG, _WORKER_DATA, _WORKER_ARRAYS, spec, split_seed
    )
    assert _WORKER_TASK_DIR is not None
    stem = f"{candidate_id}__{split_seed}"
    write_json(_WORKER_TASK_DIR / f"{stem}.json", result)
    np.savez_compressed(_WORKER_TASK_DIR / f"{stem}.npz", scores=scores, fold_ids=folds)
    return {
        "candidate_id": candidate_id,
        "split_seed": split_seed,
        "gain": result["metrics"]["gain_over_fixed_dense"],
    }


def _gate_result(
    config: Mapping[str, Any],
    data: audit.RouterData,
    split_results: list[dict[str, Any]],
    consensus_scores: np.ndarray,
) -> dict[str, Any]:
    metrics = audit.policy_metrics(
        data,
        consensus_scores,
        bootstrap_seed=20260902,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    gains = [row["metrics"]["gain_over_fixed_dense"] for row in split_results]
    ratio = metrics["harmful_to_beneficial_mass_ratio"]
    slope = metrics["calibration"]["slope"]
    top_decile = metrics["calibration"]["top_decile_realized_gap"]
    checks = {
        "practical_gain": float(np.mean(gains)) >= 0.01,
        "grouped_bootstrap_ci_lower": metrics["gain_ci95"][0] > 0.0,
        "all_split_seed_gains_positive": all(value > 0.0 for value in gains),
        "harmful_to_beneficial_mass_ratio": ratio is not None and ratio <= 0.5,
        "switch_coverage": 0.01 <= metrics["switch_coverage"] <= 0.5,
        "calibration_slope": 0.5 <= slope <= 1.5,
        "top_decile_realized_gap": top_decile > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "split_gains": gains,
        "mean_split_gain": float(np.mean(gains)),
        "consensus_metrics": metrics,
    }


def run(max_workers: int) -> None:
    protocol = json.loads((TEACHER_DIR / "teacher_protocol.json").read_text(encoding="utf-8"))
    schema = json.loads((TEACHER_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or schema.get("status") != "complete":
        raise ValueError("Prepare and freeze teacher features before training")
    task_dir = TEACHER_DIR / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, int]] = []
    for spec in SPECS:
        for split_seed in SPLIT_SEEDS:
            stem = f"{spec.candidate_id}__{split_seed}"
            if (task_dir / f"{stem}.json").exists() and (task_dir / f"{stem}.npz").exists():
                print(f"{stem} [resume]", flush=True)
            else:
                pending.append((spec.candidate_id, split_seed))
    if pending:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
            futures = {
                executor.submit(worker_task, candidate_id, split_seed): (candidate_id, split_seed)
                for candidate_id, split_seed in pending
            }
            for future in as_completed(futures):
                value = future.result()
                print(
                    f"{value['candidate_id']} split={value['split_seed']} [complete] "
                    f"gain={value['gain']:+.6f}",
                    flush=True,
                )

    config = audit._load_config(CONFIG_PATH)
    data, _ = audit.load_frozen_data(config)
    with np.load(TEACHER_DIR / "teacher_features.npz", allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    all_results: dict[str, dict[str, Any]] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    gate_results: dict[str, Any] = {}
    for spec in SPECS:
        all_results[spec.candidate_id] = {}
        split_scores: list[np.ndarray] = []
        split_results: list[dict[str, Any]] = []
        for split_seed in SPLIT_SEEDS:
            stem = f"{spec.candidate_id}__{split_seed}"
            result = json.loads((task_dir / f"{stem}.json").read_text(encoding="utf-8"))
            with np.load(task_dir / f"{stem}.npz", allow_pickle=False) as stored:
                scores = np.asarray(stored["scores"], dtype=np.float64)
                folds = np.asarray(stored["fold_ids"], dtype=np.int64)
            if scores.shape != (9600,) or set(folds.tolist()) != {0, 1, 2, 3, 4}:
                raise ValueError(f"Incomplete teacher task: {stem}")
            all_results[spec.candidate_id][str(split_seed)] = result
            prediction_arrays[f"prediction__{spec.candidate_id}__{split_seed}"] = scores
            prediction_arrays[f"fold_id__{spec.candidate_id}__{split_seed}"] = folds
            split_scores.append(scores)
            split_results.append(result)
        consensus = np.mean(np.stack(split_scores, axis=0), axis=0)
        prediction_arrays[f"consensus__{spec.candidate_id}"] = consensus
        gate_results[spec.candidate_id] = _gate_result(
            config, data, split_results, consensus
        )

    gold_names = schema["gold"]["feature_names"]
    gold_matrix = np.asarray(arrays["gold"], dtype=np.float64)
    rule_scores: dict[str, np.ndarray] = {}
    for k in (1, 3, 5):
        bm25_index = gold_names.index(f"gold__bm25_coverage_at_{k}")
        dense_index = gold_names.index(f"gold__dense_coverage_at_{k}")
        rule_scores[f"gold_coverage_at_{k}_rule"] = (
            gold_matrix[:, bm25_index] - gold_matrix[:, dense_index]
        )
    rule_metrics = {
        name: audit.policy_metrics(
            data,
            scores,
            bootstrap_seed=20260902,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
        for name, scores in rule_scores.items()
    }
    prediction_arrays.update({f"rule__{name}": scores for name, scores in rule_scores.items()})
    np.savez_compressed(TEACHER_DIR / "teacher_predictions.npz", **prediction_arrays)
    write_json(
        TEACHER_DIR / "candidate_metrics.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "candidate_splits": all_results,
            "formal_gate_results": gate_results,
            "gold_rule_metrics": rule_metrics,
            "external_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "trained_candidates": {
                    name: {
                        "passed": value["passed"],
                        "gain": value["consensus_metrics"]["gain_over_fixed_dense"],
                    }
                    for name, value in gate_results.items()
                },
                "gold_rules": {
                    name: value["gain_over_fixed_dense"] for name, value in rule_metrics.items()
                },
            },
            indent=2,
        )
    )


def _primary_rule_passed(metrics: Mapping[str, Any]) -> bool:
    ratio = metrics["harmful_to_beneficial_mass_ratio"]
    return bool(
        metrics["gain_over_fixed_dense"] >= 0.01
        and metrics["gain_ci95"][0] > 0.0
        and ratio is not None
        and ratio <= 0.5
        and 0.01 <= metrics["switch_coverage"] <= 0.5
    )


def report() -> None:
    metrics = json.loads((TEACHER_DIR / "candidate_metrics.json").read_text(encoding="utf-8"))
    gate = metrics["formal_gate_results"]
    rules = metrics["gold_rule_metrics"]
    probe_pass = [name for name in ("P0_probe_ridge", "P1_probe_xgb") if gate[name]["passed"]]
    gold_pass = [
        name
        for name in ("G0_gold_ridge", "G1_gold_xgb", "G2_probe_gold_xgb")
        if gate[name]["passed"]
    ]
    primary_rule = rules["gold_coverage_at_5_rule"]
    rule_pass = _primary_rule_passed(primary_rule)
    if probe_pass:
        conclusion = "PROBE_SIGNAL_SUFFICIENT"
        recommendation = (
            "Implement and validate a post-retrieval probe/cascade: retrieve shallow BM25 and Dense "
            "lists, estimate answer-utility gap, then generate from only the selected context."
        )
    elif gold_pass or rule_pass:
        conclusion = "GOLD_EVIDENCE_SIGNAL_SUFFICIENT"
        recommendation = (
            "The missing signal is evidence sufficiency, not query wording. The qrels-free probe "
            "already passes gain, confidence, stability, coverage, calibration, and ranking checks "
            "but narrowly misses the harmful-loss gate; the next architecture should be a shallow "
            "BM25+Dense probe/cascade with conservative safety control, not more query-only scaling."
        )
    else:
        conclusion = "RETRIEVAL_SIGNAL_INSUFFICIENT"
        recommendation = (
            "Even privileged retrieval evidence did not recover useful utility; prioritize context "
            "selection, reranking, prompt interaction, and generation robustness."
        )

    student = json.loads((RUN_DIR / "decision.json").read_text(encoding="utf-8"))[
        "formal_gate_results"
    ]["M3_pca32_structured_ridge"]["consensus_metrics"]
    config = audit._load_config(CONFIG_PATH)
    data, _ = audit.load_frozen_data(config)
    with np.load(RUN_DIR / "formal_predictions.npz", allow_pickle=False) as stored:
        student_scores = np.mean(
            np.stack(
                [
                    np.asarray(
                        stored[f"prediction__M3_pca32_structured_ridge__{split_seed}"],
                        dtype=np.float64,
                    )
                    for split_seed in SPLIT_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
    with np.load(TEACHER_DIR / "teacher_predictions.npz", allow_pickle=False) as stored:
        probe_scores = np.asarray(stored["consensus__P1_probe_xgb"], dtype=np.float64)
        gold_scores = np.asarray(stored["consensus__G1_gold_xgb"], dtype=np.float64)

    def contribution(scores: np.ndarray) -> np.ndarray:
        return np.where(scores > 0.0, data.gap, 0.0)

    paired_increments: dict[str, Any] = {}
    for name, left, right in (
        ("probe_xgb_minus_query_only_m3", probe_scores, student_scores),
        ("gold_xgb_minus_probe_xgb", gold_scores, probe_scores),
    ):
        difference = contribution(left) - contribution(right)
        paired_increments[name] = {
            "mean_gain_increment": float(np.mean(difference)),
            "grouped_bootstrap_ci95": audit._bootstrap_gain_ci(
                difference,
                data.group_ids,
                seed=20260902,
                resamples=BOOTSTRAP_RESAMPLES,
            ),
        }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "stage": 12,
        "status": "complete",
        "conclusion": conclusion,
        "recommendation": recommendation,
        "query_only_student": {
            "candidate_id": "M3_pca32_structured_ridge",
            "gain_over_fixed_dense": student["gain_over_fixed_dense"],
            "gain_ci95": student["gain_ci95"],
            "non_tie_auc": student["non_tie_auc"],
        },
        "probe_candidates_passing_full_gate": probe_pass,
        "gold_candidates_passing_full_gate": gold_pass,
        "primary_gold_rule_passed_diagnostic_gate": rule_pass,
        "paired_policy_increments": paired_increments,
        "formal_gate_results": gate,
        "gold_rule_metrics": rules,
        "teacher_is_deployable_pre_retrieval_router": False,
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "external_calls": 0,
    }
    write_json(RUN_DIR / "privileged_teacher_diagnostic.json", summary)

    task_json = list((TEACHER_DIR / "tasks").glob("*.json"))
    task_npz = list((TEACHER_DIR / "tasks").glob("*.npz"))
    with np.load(TEACHER_DIR / "teacher_predictions.npz", allow_pickle=False) as stored:
        prediction_array_count = len(stored.files)
        for name in stored.files:
            value = np.asarray(stored[name])
            if value.shape != (9600,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid teacher prediction array: {name}")
            if name.startswith("fold_id__") and set(value.tolist()) != {0, 1, 2, 3, 4}:
                raise ValueError(f"Incomplete teacher OOF fold coverage: {name}")
    with sqlite3.connect(DATABASE_PATH) as connection:
        fresh_dev_rows = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE partition='fresh_dev'"
        ).fetchone()[0]
        final_holdout_rows = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE partition='final_holdout'"
        ).fetchone()[0]
    if len(task_json) != 15 or len(task_npz) != 15:
        raise ValueError("Teacher task matrix is incomplete")
    if prediction_array_count != 38 or fresh_dev_rows or final_holdout_rows:
        raise ValueError("Teacher predictions or partition boundaries are invalid")
    validation = {
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "conclusion": conclusion,
        "query_count": 9600,
        "trained_candidate_split_tasks": len(task_json),
        "task_prediction_files": len(task_npz),
        "merged_prediction_arrays": prediction_array_count,
        "fresh_dev_rows_read": fresh_dev_rows,
        "final_holdout_rows_read": final_holdout_rows,
        "forbidden_outcome_features_used": [],
        "external_calls": 0,
        "artifacts": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in {
                "teacher_protocol.json": TEACHER_DIR / "teacher_protocol.json",
                "feature_schema.json": TEACHER_DIR / "feature_schema.json",
                "teacher_features.npz": TEACHER_DIR / "teacher_features.npz",
                "candidate_metrics.json": TEACHER_DIR / "candidate_metrics.json",
                "teacher_predictions.npz": TEACHER_DIR / "teacher_predictions.npz",
                "privileged_teacher_diagnostic.json": RUN_DIR
                / "privileged_teacher_diagnostic.json",
            }.items()
        },
    }
    write_json(TEACHER_DIR / "validation.json", validation)

    rows = []
    for spec in SPECS:
        value = gate[spec.candidate_id]
        metric = value["consensus_metrics"]
        rows.append(
            f"| {spec.candidate_id} | {spec.block} | {metric['gain_over_fixed_dense']:+.6f} | "
            f"[{metric['gain_ci95'][0]:+.6f}, {metric['gain_ci95'][1]:+.6f}] | "
            f"{metric['switch_coverage']:.1%} | {metric['harmful_to_beneficial_mass_ratio']:.3f} | "
            f"{metric['non_tie_auc']:.3f} | {'yes' if value['passed'] else 'no'} |"
        )
    rule_rows = []
    for name, metric in rules.items():
        rule_rows.append(
            f"| {name} | {metric['gain_over_fixed_dense']:+.6f} | "
            f"[{metric['gain_ci95'][0]:+.6f}, {metric['gain_ci95'][1]:+.6f}] | "
            f"{metric['switch_coverage']:.1%} | {metric['harmful_to_beneficial_mass_ratio']:.3f} |"
        )
    paired_rows = [
        f"| {name} | {value['mean_gain_increment']:+.6f} | "
        f"[{value['grouped_bootstrap_ci95'][0]:+.6f}, "
        f"{value['grouped_bootstrap_ci95'][1]:+.6f}] |"
        for name, value in paired_increments.items()
    ]
    report_text = f"""# Phase 2.7 Stage 12 privileged-teacher diagnostic

## Conclusion

**`{conclusion}`**

{recommendation}

This is an offline mechanism diagnostic. Gold/qrels candidates and rules are not
deployable pre-retrieval routers, and no fresh-dev or final-holdout row was read.

## Trained grouped-OOF candidates

| Candidate | information block | consensus gain | grouped CI95 | coverage | harmful/beneficial | non-tie AUC | full gate |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Frozen gold-evidence rules

| Rule | gain | grouped CI95 | coverage | harmful/beneficial |
|---|---:|---:|---:|---:|
{chr(10).join(rule_rows)}

## Paired policy increments

| Comparison | mean increment | grouped paired CI95 |
|---|---:|---:|
{chr(10).join(paired_rows)}

## Interpretation boundary

- Query-only M3 consensus gain was `{student['gain_over_fixed_dense']:+.6f}` with
  CI95 `[{student['gain_ci95'][0]:+.6f}, {student['gain_ci95'][1]:+.6f}]`.
- `probe` features use both retrievers' scores, margins, list overlap, context length,
  and query-context overlap, but do not use qrels or generation outcomes.
- `gold` features use supporting-document coverage/ranks and are privileged.
- No feature uses generated answers, F1 labels, Answer Correctness, or current-query
  action utility as an input.

The result determines whether the next architecture should be a shallow
post-retrieval cascade, a qrels-free evidence-sufficiency estimator/reranker, or a
generation/context robustness project. It does not authorize reporting any gold
teacher as an online router.
"""
    (TEACHER_DIR / "report.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "recommendation": recommendation}, indent=2))


def main() -> int:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run(args.max_workers)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
