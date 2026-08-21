"""Build compatibility features and train/evaluate the Phase 3 F1 pairwise router."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import _core_config, _load_router_config  # noqa: E402
from scripts.run_router_phase2 import _load_examples  # noqa: E402
from scripts.run_router_phase3 import (  # noqa: E402
    ACTIONS,
    _connect,
    _phase3,
    _read_json,
    _resolve,
    _run_dir,
    _selection,
    _write_json,
)
from src.embedders.text_embedder import create_embedder  # noqa: E402
from src.retrievers.sqlite_bm25 import (  # noqa: E402
    analyze_sqlite_bm25_text,
    read_sqlite_bm25_term_stats,
)


PAIRS = (("no_context", "bm25"), ("no_context", "dense"), ("bm25", "dense"))
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_YEAR_PATTERN = re.compile(r"^(?:1[5-9]\d{2}|20\d{2})$")
_QUOTED_PATTERN = re.compile(r'["“”]([^"“”]+)["“”]')


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
    parser.add_argument("--run-id", default="phase3_f1_pairwise_v1")
    parser.add_argument("--stage", choices=("features", "fit", "evaluate"), required=True)
    parser.add_argument("--partition", choices=("train", "fresh_dev"), default="train")
    parser.add_argument("--train-size", type=int, default=None)
    return parser.parse_args()


def _feature_path(run_dir: Path, partition: str) -> Path:
    return run_dir / "features" / f"{partition}.npz"


def _feature_schema_path(run_dir: Path) -> Path:
    return run_dir / "features" / "feature_schema.json"


def _centroid_path(run_dir: Path) -> Path:
    return run_dir / "features" / "corpus_prototypes.npy"


def _lexical_features(
    question: str,
    stats: Mapping[str, tuple[int, float]],
) -> tuple[list[float], list[str]]:
    tokens = list(analyze_sqlite_bm25_text(question))
    known = [(token, stats[token]) for token in tokens if token in stats]
    idf = np.asarray([value[1][1] for value in known], dtype=np.float64)
    df = np.asarray([value[1][0] for value in known], dtype=np.float64)
    denominator = max(1, len(tokens))

    def scalar(values: np.ndarray, operation: str) -> float:
        return float(getattr(values, operation)()) if len(values) else 0.0

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if len(values) else 0.0

    def max_idf(candidate_tokens: Sequence[str]) -> float:
        values = [stats[token][1] for token in candidate_tokens if token in stats]
        return float(max(values)) if values else 0.0

    words = _WORD_PATTERN.findall(question)
    number_tokens = [token.casefold() for token in words if token.isdigit()]
    year_tokens = [token.casefold() for token in words if _YEAR_PATTERN.fullmatch(token)]
    capitalized_tokens = [
        token.casefold() for token in words[1:] if token[:1].isupper() and not token.isupper()
    ]
    quoted_tokens = [
        token
        for phrase in _QUOTED_PATTERN.findall(question)
        for token in analyze_sqlite_bm25_text(phrase)
    ]
    names = [
        "corpus_vocabulary_coverage",
        "corpus_oov_ratio",
        "idf_mean",
        "idf_std",
        "idf_min",
        "idf_max",
        "idf_p90",
        "idf_max_share",
        "log_document_frequency_mean",
        "log_document_frequency_std",
        "df_at_most_10_ratio",
        "df_at_most_100_ratio",
        "df_at_most_1000_ratio",
        "numeric_anchor_max_idf",
        "year_anchor_max_idf",
        "capitalized_anchor_max_idf",
        "quoted_anchor_max_idf",
    ]
    values = [
        float(len(known) / denominator),
        float((len(tokens) - len(known)) / denominator),
        scalar(idf, "mean"),
        scalar(idf, "std"),
        scalar(idf, "min"),
        scalar(idf, "max"),
        percentile(idf, 90.0),
        float(idf.max() / idf.sum()) if len(idf) and idf.sum() > 0 else 0.0,
        scalar(np.log1p(df), "mean"),
        scalar(np.log1p(df), "std"),
        float(np.mean(df <= 10)) if len(df) else 0.0,
        float(np.mean(df <= 100)) if len(df) else 0.0,
        float(np.mean(df <= 1000)) if len(df) else 0.0,
        max_idf(number_tokens),
        max_idf(year_tokens),
        max_idf(capitalized_tokens),
        max_idf(quoted_tokens),
    ]
    return values, names


def _load_embedding_parts(router: Mapping[str, Any]) -> tuple[Path, list[Mapping[str, Any]]]:
    root = _resolve(str(router["artifacts"]["encoded_corpus"]))
    manifest = _read_json(root / "manifest.json")
    artifacts = manifest.get("artifacts")
    embeddings = artifacts.get("embeddings") if isinstance(artifacts, Mapping) else None
    parts = embeddings.get("parts") if isinstance(embeddings, Mapping) else None
    if not isinstance(parts, list) or not parts:
        raise ValueError("Encoded corpus manifest has no embedding parts")
    return root / "embeddings", parts


def _corpus_prototypes(
    router: Mapping[str, Any],
    run_dir: Path,
    *,
    clusters: int = 64,
    sample_rows: int = 100_000,
) -> np.ndarray:
    path = _centroid_path(run_dir)
    if path.is_file():
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != clusters:
            raise ValueError("Stored corpus prototypes have an unexpected shape")
        return values

    embedding_root, parts = _load_embedding_parts(router)
    rng = np.random.default_rng(int(_phase3(router)["sample_seed"]) + 1)
    samples: list[np.ndarray] = []
    remaining = sample_rows
    for index, part in enumerate(parts):
        if remaining <= 0:
            break
        remaining_parts = len(parts) - index
        requested = int(math.ceil(remaining / remaining_parts))
        rows = int(part["rows"])
        take = min(rows, requested)
        values = np.load(embedding_root / str(part["file"]), mmap_mode="r", allow_pickle=False)
        positions = rng.choice(rows, take, replace=False)
        samples.append(np.asarray(values[positions], dtype=np.float32))
        remaining -= take
        del values
    matrix = np.concatenate(samples, axis=0)
    if len(matrix) != sample_rows:
        raise RuntimeError(f"Prototype sample has {len(matrix)} rows, expected {sample_rows}")
    model = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=4096,
        max_iter=100,
        n_init=3,
        random_state=int(_phase3(router)["sample_seed"]) + 2,
    )
    model.fit(matrix)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centers = centers / norms
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centers, allow_pickle=False)
    return centers


def _dense_summary(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    similarities = np.asarray(embeddings @ prototypes.T, dtype=np.float32)
    ordered = np.sort(similarities, axis=1)[:, ::-1]
    top = ordered[:, :8]
    shifted = (similarities - similarities.max(axis=1, keepdims=True)) / 0.05
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    density = np.mean(similarities >= (ordered[:, :1] - 0.05), axis=1)
    summary = np.column_stack(
        [
            top,
            top[:, 0] - top[:, 1],
            similarities.mean(axis=1),
            similarities.std(axis=1),
            entropy,
            density,
        ]
    ).astype(np.float32)
    names = [f"prototype_similarity_top_{index}" for index in range(1, 9)] + [
        "prototype_similarity_margin",
        "prototype_similarity_mean",
        "prototype_similarity_std",
        "prototype_similarity_entropy",
        "prototype_local_density",
    ]
    return summary, names


def build_features(
    router: Mapping[str, Any],
    run_dir: Path,
    sample: Mapping[str, Any],
    partition: str,
) -> dict[str, Any]:
    if partition == "fresh_dev" and not (run_dir / "model_freeze.json").is_file():
        raise RuntimeError("Fresh dev features remain sealed until model freeze")
    path = _feature_path(run_dir, partition)
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            return {
                "partition": partition,
                "queries": int(stored["query_ids"].shape[0]),
                "reused": True,
            }

    selected = _selection(sample, partition)
    examples, _ = _load_examples(router, {"rows": selected})
    questions = [str(row["question"]) for row in examples]
    all_terms = {
        token for question in questions for token in analyze_sqlite_bm25_text(question)
    }
    database = _resolve(str(router["artifacts"]["bm25_index"])) / "index.sqlite3"
    started = time.perf_counter()
    stats = read_sqlite_bm25_term_stats(database, all_terms)
    lexical_rows = [_lexical_features(question, stats)[0] for question in questions]
    lexical_names = _lexical_features("", stats)[1]
    lexical_seconds = time.perf_counter() - started

    started = time.perf_counter()
    embedder = create_embedder(_core_config(router, method="dense"), role="query")
    embeddings = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    embedding_seconds = time.perf_counter() - started
    prototypes = _corpus_prototypes(router, run_dir)
    dense, dense_names = _dense_summary(embeddings, prototypes)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_ids=np.asarray([row["query_id"] for row in examples], dtype=np.str_),
        group_ids=np.asarray([row["group_id"] for row in examples], dtype=np.str_),
        lexical=np.asarray(lexical_rows, dtype=np.float32),
        dense=dense,
        embedding=embeddings,
    )
    schema = {
        "lexical": lexical_names,
        "dense_corpus": dense_names,
        "query_embedding": [f"query_embedding_{index}" for index in range(embeddings.shape[1])],
        "excluded_generic_surface_block": True,
        "online_information": "query_and_corpus_static_only",
        "cost": {
            "lexical_total_seconds": lexical_seconds,
            "embedding_total_seconds": embedding_seconds,
            "queries": len(examples),
        },
    }
    _write_json(_feature_schema_path(run_dir), schema)
    return {"partition": partition, "queries": len(examples), "reused": False}


def _load_feature_blocks(
    run_dir: Path,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_feature_path(run_dir, partition), allow_pickle=False) as stored:
        return (
            np.asarray(stored["query_ids"], dtype=np.str_),
            np.asarray(stored["group_ids"], dtype=np.str_),
            np.asarray(stored["lexical"], dtype=np.float32),
            np.asarray(stored["dense"], dtype=np.float32),
            np.asarray(stored["embedding"], dtype=np.float32),
        )


def _outcome_records(
    run_dir: Path,
    partition: str,
    query_limit: int,
) -> list[dict[str, Any]]:
    connection = _connect(run_dir)
    try:
        rows = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id, o.action, o.repeat_id,
                   o.status, o.payload
            FROM sample_rows AS s
            JOIN outcomes AS o ON o.query_id = s.query_id
            WHERE s.partition = ? AND s.query_rank < ?
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'no_context' THEN 0 WHEN 'bm25' THEN 1 ELSE 2 END,
                     o.repeat_id
            """,
            (partition, query_limit),
        )
        cells: dict[str, dict[str, Any]] = {}
        for rank, query_id, group_id, action, repeat_id, status, payload in rows:
            if status != "success":
                raise RuntimeError(
                    f"Incomplete outcome: {partition}/{query_id}/{action}/{repeat_id}/{status}"
                )
            row = json.loads(payload)
            metric = float(row["metrics"]["normalized_token_f1"])
            cell = cells.setdefault(
                str(query_id),
                {
                    "rank": int(rank),
                    "query_id": str(query_id),
                    "group_id": str(group_id),
                    "utilities": {name: [] for name in ACTIONS},
                },
            )
            cell["utilities"][str(action)].append(metric)
        records = sorted(cells.values(), key=lambda value: value["rank"])
        for record in records:
            for action in ACTIONS:
                values = record["utilities"][action]
                if len(values) != 3:
                    raise RuntimeError(f"Expected 3 repeats for {record['query_id']}/{action}")
                record["utilities"][action] = float(np.mean(values))
        if len(records) != query_limit:
            raise RuntimeError(f"Found {len(records)} records, expected {query_limit}")
        return records
    finally:
        connection.close()


def _pair_design(base: np.ndarray) -> np.ndarray:
    rows = base.shape[0]
    repeated = np.repeat(base, len(PAIRS), axis=0)
    pair_ids = np.tile(np.arange(len(PAIRS)), rows)
    one_hot = np.eye(len(PAIRS), dtype=np.float32)[pair_ids]
    interactions = np.zeros((len(repeated), base.shape[1] * len(PAIRS)), dtype=np.float32)
    for pair_id in range(len(PAIRS)):
        selected = pair_ids == pair_id
        start = pair_id * base.shape[1]
        interactions[selected, start : start + base.shape[1]] = repeated[selected]
    return np.concatenate([repeated, one_hot, interactions], axis=1)


def _utilities(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["utilities"][action]) for action in ACTIONS] for row in records],
        dtype=np.float32,
    )


def _pair_targets(utilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences = np.column_stack(
        [
            utilities[:, ACTIONS.index(left)] - utilities[:, ACTIONS.index(right)]
            for left, right in PAIRS
        ]
    ).reshape(-1)
    keep = np.abs(differences) > 1e-12
    labels = (differences[keep] > 0).astype(np.int32)
    weights = np.abs(differences[keep]).astype(np.float32)
    return keep, labels, weights


def _build_model(kind: str, seed: int) -> Any:
    if kind == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            n_jobs=8,
            random_state=seed,
        )
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model kind: {kind}")


def _fit(model: Any, kind: str, matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
    if kind == "logistic":
        model.fit(matrix, labels, model__sample_weight=weights)
    else:
        model.fit(matrix, labels, sample_weight=weights)


def _pair_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return values.reshape(-1, len(PAIRS))


def _actions_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    scores = np.zeros((probabilities.shape[0], len(ACTIONS)), dtype=np.float64)
    for pair_id, (left, right) in enumerate(PAIRS):
        left_index = ACTIONS.index(left)
        right_index = ACTIONS.index(right)
        scores[:, left_index] += probabilities[:, pair_id]
        scores[:, right_index] += 1.0 - probabilities[:, pair_id]
    # np.argmax provides the frozen N/B/D listed-order tie behavior only after
    # adding a tiny deterministic preference for Dense.
    scores[:, ACTIONS.index("dense")] += 1e-12
    return np.argmax(scores, axis=1)


def _policy_metrics(utilities: np.ndarray, chosen: np.ndarray) -> dict[str, Any]:
    positions = np.arange(len(utilities))
    router = float(np.mean(utilities[positions, chosen]))
    fixed = {action: float(np.mean(utilities[:, index])) for index, action in enumerate(ACTIONS)}
    oracle = float(np.mean(np.max(utilities, axis=1)))
    dense = fixed["dense"]
    headroom = oracle - dense
    return {
        "router_f1": router,
        "fixed": fixed,
        "oracle_f1": oracle,
        "gain_over_fixed_dense": router - dense,
        "oracle_headroom_over_fixed_dense": headroom,
        "oracle_recovery": None if headroom <= 0 else (router - dense) / headroom,
        "action_counts": dict(Counter(ACTIONS[int(index)] for index in chosen)),
    }


def _base_matrix(
    lexical: np.ndarray,
    dense: np.ndarray,
    embedding: np.ndarray,
    tier: str,
) -> np.ndarray:
    if tier == "tier_a":
        return np.asarray(lexical, dtype=np.float32)
    if tier == "full":
        return np.concatenate([lexical, dense, embedding], axis=1).astype(np.float32)
    raise ValueError(f"Unknown feature tier: {tier}")


def _cross_validate(
    records: Sequence[Mapping[str, Any]],
    base: np.ndarray,
    *,
    kind: str,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    utilities = _utilities(records)
    groups = np.asarray([str(row["group_id"]) for row in records], dtype=np.str_)
    design = _pair_design(base)
    differences = np.column_stack(
        [
            utilities[:, ACTIONS.index(left)] - utilities[:, ACTIONS.index(right)]
            for left, right in PAIRS
        ]
    ).reshape(-1)
    keep = np.abs(differences) > 1e-12
    full_labels = (differences > 0).astype(np.int32)
    full_weights = np.abs(differences).astype(np.float32)
    labels = full_labels[keep]
    weights = full_weights[keep]
    pair_query = np.repeat(np.arange(len(records)), len(PAIRS))
    oof = np.full((len(records), len(PAIRS)), np.nan, dtype=np.float64)
    pair_predictions = np.full(len(labels), np.nan, dtype=np.float64)
    kept_positions = np.flatnonzero(keep)

    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_queries, validation_queries in splitter.split(base, groups=groups):
        train_mask = keep & np.isin(pair_query, train_queries)
        validation_mask = np.isin(pair_query, validation_queries)
        model = _build_model(kind, seed)
        _fit(
            model,
            kind,
            design[train_mask],
            full_labels[train_mask],
            full_weights[train_mask],
        )
        probabilities = np.asarray(model.predict_proba(design[validation_mask])[:, 1])
        oof[validation_queries] = probabilities.reshape(-1, len(PAIRS))

        validation_kept_global = np.flatnonzero(keep & validation_mask)
        validation_kept_local = np.searchsorted(kept_positions, validation_kept_global)
        pair_predictions[validation_kept_local] = np.asarray(
            model.predict_proba(design[validation_kept_global])[:, 1]
        )

    if not np.isfinite(oof).all() or not np.isfinite(pair_predictions).all():
        raise RuntimeError("Cross-validation did not fill all predictions")
    chosen = _actions_from_probabilities(oof)
    metrics = _policy_metrics(utilities, chosen)
    metrics["pairwise_accuracy"] = float(np.mean((pair_predictions >= 0.5) == labels))
    metrics["pairwise_weighted_accuracy"] = float(
        np.average((pair_predictions >= 0.5) == labels, weights=weights)
    )
    metrics["non_tie_pairs"] = int(len(labels))
    metrics["tie_pairs"] = int(len(keep) - len(labels))
    return metrics


def fit_learning_point(
    router: Mapping[str, Any],
    run_dir: Path,
    train_size: int,
) -> dict[str, Any]:
    phase3 = _phase3(router)
    learning_curve = {int(value) for value in phase3["source_partitions"]["learning_curve"]}
    if train_size not in learning_curve:
        raise ValueError("train-size must be a frozen learning-curve size")
    records = _outcome_records(run_dir, "train", train_size)
    query_ids, group_ids, lexical, dense, embedding = _load_feature_blocks(run_dir, "train")
    if list(query_ids[:train_size]) != [row["query_id"] for row in records]:
        raise RuntimeError("Outcome and feature query order differ")
    if list(group_ids[:train_size]) != [row["group_id"] for row in records]:
        raise RuntimeError("Outcome and feature groups differ")
    seeds = [int(value) for value in phase3["training_seeds"]]
    folds = int(phase3["cv_folds"])
    candidates = (("full_xgboost", "full", "xgboost"), ("full_logistic", "full", "logistic"), ("tier_a_xgboost", "tier_a", "xgboost"))
    results: dict[str, Any] = {}
    for name, tier, kind in candidates:
        base = _base_matrix(lexical[:train_size], dense[:train_size], embedding[:train_size], tier)
        seed_results = [
            _cross_validate(records, base, kind=kind, seed=seed, folds=folds)
            for seed in seeds
        ]
        gains = [float(row["gain_over_fixed_dense"]) for row in seed_results]
        results[name] = {
            "feature_tier": tier,
            "model": kind,
            "seeds": seed_results,
            "median_gain_over_fixed_dense": float(np.median(gains)),
            "min_gain_over_fixed_dense": float(min(gains)),
            "max_gain_over_fixed_dense": float(max(gains)),
        }
    summary = {
        "train_queries": train_size,
        "split_unit": "phase0_group",
        "f1_is_screening_metric": True,
        "answer_correctness_calls": 0,
        "candidates": results,
    }
    _write_json(run_dir / "training" / f"learning_curve_{train_size}.json", summary)

    train_total = int(phase3["source_partitions"]["train_total"])
    if train_size == train_total:
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        base = _base_matrix(lexical[:train_size], dense[:train_size], embedding[:train_size], "full")
        design = _pair_design(base)
        utilities = _utilities(records)
        keep, labels, weights = _pair_targets(utilities)
        for seed in seeds:
            model = _build_model("xgboost", seed)
            _fit(model, "xgboost", design[keep], labels, weights)
            joblib.dump(model, models_dir / f"xgboost_seed_{seed}.joblib")
        freeze = {
            "train_queries": train_total,
            "model": "full_xgboost_pairwise",
            "training_seeds": seeds,
            "actions": list(ACTIONS),
            "pairs": [list(pair) for pair in PAIRS],
            "routing_target": "normalized_token_f1",
            "answer_correctness_calls": 0,
            "fresh_dev_outcomes_read": 0,
        }
        _write_json(run_dir / "model_freeze.json", freeze)
    return summary


def _group_bootstrap_gain(
    differences: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        values.append(float(np.mean(differences[indices])))
    return values


def evaluate_fresh_dev(router: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    phase3 = _phase3(router)
    if not (run_dir / "model_freeze.json").is_file():
        raise RuntimeError("Model is not frozen")
    dev_count = int(phase3["source_partitions"]["fresh_dev"])
    records = _outcome_records(run_dir, "fresh_dev", dev_count)
    query_ids, group_ids, lexical, dense, embedding = _load_feature_blocks(run_dir, "fresh_dev")
    if list(query_ids) != [row["query_id"] for row in records]:
        raise RuntimeError("Fresh-dev outcome and feature order differ")
    base = _base_matrix(lexical, dense, embedding, "full")
    design = _pair_design(base)
    utilities = _utilities(records)
    seeds = [int(value) for value in phase3["training_seeds"]]
    probability_rows: list[np.ndarray] = []
    seed_results: dict[str, Any] = {}
    for seed in seeds:
        model = joblib.load(run_dir / "models" / f"xgboost_seed_{seed}.joblib")
        probabilities = _pair_probabilities(model, design)
        probability_rows.append(probabilities)
        chosen = _actions_from_probabilities(probabilities)
        seed_results[str(seed)] = _policy_metrics(utilities, chosen)
    ensemble = np.mean(np.stack(probability_rows, axis=0), axis=0)
    chosen = _actions_from_probabilities(ensemble)
    official = _policy_metrics(utilities, chosen)
    positions = np.arange(dev_count)
    differences = (
        utilities[positions, chosen] - utilities[:, ACTIONS.index("dense")]
    ).astype(np.float64)
    bootstrap = _group_bootstrap_gain(
        differences,
        group_ids,
        resamples=int(phase3["bootstrap_resamples"]),
        seed=int(phase3["bootstrap_seed"]),
    )
    ci = [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]
    gains = [float(value["gain_over_fixed_dense"]) for value in seed_results.values()]
    gate = (
        official["gain_over_fixed_dense"] >= float(phase3["practical_f1_gain"])
        and ci[0] > float(phase3["f1_ci_lower_bound"])
        and all(value > 0.0 for value in gains)
        and float(np.median(gains)) >= float(phase3["practical_f1_gain"])
    )
    summary = {
        "phase": 3,
        "screening_metric": "normalized_token_f1",
        "answer_correctness_calls": 0,
        "sample": {
            "train": int(phase3["source_partitions"]["train_total"]),
            "fresh_dev": dev_count,
            "historical_dev_used_for_gate": 0,
            "final_holdout_outcomes": 0,
        },
        "official_ensemble": official,
        "seed_results": seed_results,
        "paired_group_bootstrap_ci95": ci,
        "gate": {
            "f1_screen_passed": bool(gate),
            "decision": "REPORT_F1_AND_WAIT_FOR_AC_AUTHORIZATION" if gate else "STOP_AFTER_F1",
            "automatic_ac_evaluation": "forbidden",
        },
    }
    _write_json(run_dir / "summary.json", summary)
    compact_path = run_dir / "fresh_dev_predictions.jsonl"
    with compact_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            value = {
                "query_id": record["query_id"],
                "group_id": record["group_id"],
                "chosen_action": ACTIONS[int(chosen[index])],
                "f1": {action: float(utilities[index, position]) for position, action in enumerate(ACTIONS)},
            }
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main() -> int:
    args = _arguments()
    config_path = _resolve(args.config)
    router = _load_router_config(config_path)
    _phase3(router)
    run_dir = _run_dir(config_path, args.run_id)
    sample = _read_json(run_dir / "sample.json")
    if args.stage == "features":
        result: Any = build_features(router, run_dir, sample, args.partition)
    elif args.stage == "fit":
        if args.partition != "train" or args.train_size is None:
            raise ValueError("fit requires --partition train and --train-size")
        result = fit_learning_point(router, run_dir, args.train_size)
    else:
        if args.partition != "fresh_dev":
            raise ValueError("evaluate requires --partition fresh_dev")
        result = evaluate_fresh_dev(router, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
