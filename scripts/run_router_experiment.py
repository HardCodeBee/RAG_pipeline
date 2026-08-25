"""Train and evaluate the frozen Phase 3 pre-retrieval BM25/Dense router baselines."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import _core_config, _load_router_config
from src.embedders.text_embedder import create_embedder
from src.retrievers.sqlite_bm25 import (
    analyze_sqlite_bm25_text,
    read_sqlite_bm25_term_stats,
)


ACTIONS = ("bm25", "dense")
PARTITIONS = ("train", "dev")
METRICS = ("f1", "em", "ac")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
_WH_WORDS = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
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
            "phase3_predictability_v1"
        ),
    )
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


def _aggregate_phase2_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int],
    expected_repeats: int,
) -> list[dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        if split == "final_holdout":
            raise ValueError("Phase 3 must not read final_holdout outcomes")
        if split not in PARTITIONS:
            continue
        if row.get("status") != "success":
            raise ValueError("Phase 3 source contains a non-success result")
        action = str(row.get("action", ""))
        if action not in ACTIONS:
            raise ValueError(f"Unexpected action: {action}")
        query_id = str(row["query_id"])
        cell = cells.setdefault(
            query_id,
            {
                "query_id": query_id,
                "group_id": str(row["group_id"]),
                "split": split,
                "question": str(row["question"]),
                "actions": {name: {} for name in ACTIONS},
            },
        )
        if (
            cell["group_id"] != str(row["group_id"])
            or cell["split"] != split
            or cell["question"] != str(row["question"])
        ):
            raise ValueError(f"Inconsistent metadata for query {query_id}")
        repeat_id = int(row["repeat_id"])
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Missing metrics for query {query_id}")
        action_rows = cell["actions"][action]
        if repeat_id in action_rows:
            raise ValueError(f"Duplicate repeat for query {query_id}, action {action}")
        action_rows[repeat_id] = {
            "f1": float(metrics["normalized_token_f1"]),
            "em": float(metrics["normalized_exact_match"]),
            "ac": float(metrics["answer_correctness"]),
        }

    records: list[dict[str, Any]] = []
    for query_id in sorted(cells):
        cell = cells[query_id]
        record = {key: cell[key] for key in ("query_id", "group_id", "split", "question")}
        record["repeats"] = {}
        for action in ACTIONS:
            action_rows = cell["actions"][action]
            if sorted(action_rows) != list(range(expected_repeats)):
                raise ValueError(f"Incomplete repeats for query {query_id}, action {action}")
            record["repeats"][action] = {
                metric: [action_rows[index][metric] for index in range(expected_repeats)]
                for metric in METRICS
            }
        records.append(record)

    counts = {partition: sum(row["split"] == partition for row in records) for partition in PARTITIONS}
    required = {partition: int(expected_counts[partition]) for partition in PARTITIONS}
    if counts != required:
        raise ValueError(f"Expected Phase 3 counts {required}, found {counts}")
    if len({row["query_id"] for row in records}) != len(records):
        raise ValueError("Query ids must be unique")
    return records


def _surface_features(question: str) -> tuple[list[float], list[str]]:
    words = _WORD_PATTERN.findall(question)
    folded = [word.casefold() for word in words]
    content = list(analyze_sqlite_bm25_text(question))
    lengths = np.asarray([len(word) for word in words], dtype=np.float64)
    uppercase_initial = sum(word[:1].isupper() for word in words)
    names = [
        "character_count",
        "word_count",
        "unique_word_ratio",
        "mean_word_length",
        "max_word_length",
        "content_word_ratio",
        "digit_token_count",
        "year_count",
        "uppercase_initial_ratio",
        "question_mark_count",
        "comma_count",
        "quote_count",
        "hyphen_count",
        "and_count",
        "or_count",
        "comparison_marker_count",
    ] + [f"starts_{word}" for word in _WH_WORDS]
    comparison = {"both", "between", "different", "same", "than", "versus", "vs"}
    first = folded[0] if folded else ""
    values = [
        float(len(question)),
        float(len(words)),
        float(len(set(folded)) / len(words)) if words else 0.0,
        float(lengths.mean()) if len(lengths) else 0.0,
        float(lengths.max()) if len(lengths) else 0.0,
        float(len(content) / len(words)) if words else 0.0,
        float(sum(word.isdigit() for word in words)),
        float(len(_YEAR_PATTERN.findall(question))),
        float(uppercase_initial / len(words)) if words else 0.0,
        float(question.count("?")),
        float(question.count(",")),
        float(sum(question.count(mark) for mark in ('"', "'", "“", "”"))),
        float(question.count("-")),
        float(folded.count("and")),
        float(folded.count("or")),
        float(sum(word in comparison for word in folded)),
    ] + [float(first == word) for word in _WH_WORDS]
    return values, names


def _lexical_features(
    question: str,
    stats: Mapping[str, tuple[int, float]],
) -> tuple[list[float], list[str]]:
    tokens = list(analyze_sqlite_bm25_text(question))
    known = [stats[token] for token in tokens if token in stats]
    df = np.asarray([value[0] for value in known], dtype=np.float64)
    idf = np.asarray([value[1] for value in known], dtype=np.float64)
    log_df = np.log1p(df)
    names = [
        "analyzed_token_count",
        "unique_analyzed_ratio",
        "known_token_count",
        "known_token_ratio",
        "oov_token_count",
        "idf_sum",
        "idf_mean",
        "idf_std",
        "idf_min",
        "idf_max",
        "log_df_mean",
        "log_df_std",
        "log_df_min",
        "log_df_max",
        "df_at_most_10_ratio",
        "df_at_most_100_ratio",
        "df_at_most_1000_ratio",
    ]

    def stat(values: np.ndarray, operation: str) -> float:
        if not len(values):
            return 0.0
        return float(getattr(values, operation)())

    denominator = len(tokens)
    values = [
        float(denominator),
        float(len(set(tokens)) / denominator) if denominator else 0.0,
        float(len(known)),
        float(len(known) / denominator) if denominator else 0.0,
        float(denominator - len(known)),
        float(idf.sum()) if len(idf) else 0.0,
        stat(idf, "mean"),
        stat(idf, "std"),
        stat(idf, "min"),
        stat(idf, "max"),
        stat(log_df, "mean"),
        stat(log_df, "std"),
        stat(log_df, "min"),
        stat(log_df, "max"),
        float(np.mean(df <= 10)) if len(df) else 0.0,
        float(np.mean(df <= 100)) if len(df) else 0.0,
        float(np.mean(df <= 1000)) if len(df) else 0.0,
    ]
    return values, names


def _extract_features(
    records: Sequence[Mapping[str, Any]],
    router: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, float]]:
    questions = [str(record["question"]) for record in records]
    started = time.perf_counter()
    surface_rows = [_surface_features(question)[0] for question in questions]
    surface_names = _surface_features("")[1]
    surface_seconds = time.perf_counter() - started

    database = _resolve(Path(str(router["artifacts"]["bm25_index"])) / "index.sqlite3")
    all_terms = {
        token
        for question in questions
        for token in analyze_sqlite_bm25_text(question)
    }
    started = time.perf_counter()
    stats = read_sqlite_bm25_term_stats(database, all_terms)
    lexical_rows = [_lexical_features(question, stats)[0] for question in questions]
    lexical_names = _lexical_features("", stats)[1]
    lexical_seconds = time.perf_counter() - started

    started = time.perf_counter()
    dense_config = _core_config(router, method="dense")
    embedder = create_embedder(dense_config, role="query")
    embeddings = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    embedding_seconds = time.perf_counter() - started
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Unexpected query embedding shape")

    surface = np.asarray(surface_rows, dtype=np.float32)
    lexical_only = np.asarray(lexical_rows, dtype=np.float32)
    lexical = np.concatenate([surface, lexical_only], axis=1)
    combined = np.concatenate([lexical, embeddings], axis=1)
    matrices = {
        "surface": surface,
        "lexical": lexical,
        "embedding": embeddings,
        "combined": combined,
    }
    embedding_names = [f"query_embedding_{index}" for index in range(embeddings.shape[1])]
    names = {
        "surface": surface_names,
        "lexical": surface_names + lexical_names,
        "embedding": embedding_names,
        "combined": surface_names + lexical_names + embedding_names,
    }
    count = max(1, len(records))
    costs = {
        "surface_total_seconds": surface_seconds,
        "surface_ms_per_query": 1000.0 * surface_seconds / count,
        "lexical_total_seconds": lexical_seconds,
        "lexical_ms_per_query": 1000.0 * lexical_seconds / count,
        "embedding_total_seconds": embedding_seconds,
        "embedding_ms_per_query": 1000.0 * embedding_seconds / count,
    }
    return matrices, names, costs


def _targets(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for metric in METRICS:
        result[metric] = np.asarray(
            [
                [np.mean(record["repeats"][action][metric]) for action in ACTIONS]
                for record in records
            ],
            dtype=np.float64,
        )
    return result


def _candidate_parts(candidate: str) -> tuple[str, str]:
    feature_set, model_kind = candidate.rsplit("_", 1)
    if feature_set not in {"surface", "lexical", "embedding", "combined"}:
        raise ValueError(f"Unknown feature set in candidate: {candidate}")
    if model_kind not in {"ridge", "tree", "mlp"}:
        raise ValueError(f"Unknown model kind in candidate: {candidate}")
    return feature_set, model_kind


def _build_model(model_kind: str, feature_set: str, seed: int) -> Any:
    if model_kind == "ridge":
        alpha = 1.0 if feature_set in {"surface", "lexical"} else 10.0
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    if model_kind == "tree":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=seed,
            n_jobs=1,
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=seed,
                ),
            ),
        ]
    )


def _seeds_for_model(model_kind: str, seeds: Sequence[int]) -> list[int]:
    return [int(seeds[0])] if model_kind == "ridge" else [int(seed) for seed in seeds]


def _fit_models(
    candidate: str,
    matrix: np.ndarray,
    target: np.ndarray,
    seeds: Sequence[int],
) -> list[Any]:
    feature_set, model_kind = _candidate_parts(candidate)
    models = []
    for seed in _seeds_for_model(model_kind, seeds):
        model = _build_model(model_kind, feature_set, seed)
        model.fit(matrix, target)
        models.append(model)
    return models


def _ensemble_predict(models: Sequence[Any], matrix: np.ndarray) -> np.ndarray:
    predictions = [np.asarray(model.predict(matrix), dtype=np.float64) for model in models]
    return np.mean(predictions, axis=0)


def _action_indices(predicted: np.ndarray) -> np.ndarray:
    if predicted.ndim != 2 or predicted.shape[1] != len(ACTIONS):
        raise ValueError("Predicted utility must have one column per action")
    return np.where(predicted[:, 0] > predicted[:, 1], 0, 1).astype(np.int64)


def _policy_metrics(
    predicted: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    high_margin: float,
) -> dict[str, Any]:
    actions = _action_indices(predicted)
    rows = np.arange(len(actions))
    f1 = targets["f1"]
    chosen = {metric: float(np.mean(values[rows, actions])) for metric, values in targets.items()}
    fixed = {
        action: {metric: float(np.mean(values[:, index])) for metric, values in targets.items()}
        for index, action in enumerate(ACTIONS)
    }
    fixed_action = max(ACTIONS, key=lambda action: (fixed[action]["f1"], action == "dense"))
    fixed_index = ACTIONS.index(fixed_action)
    oracle_f1 = float(np.mean(np.max(f1, axis=1)))
    gain = chosen["f1"] - fixed[fixed_action]["f1"]
    headroom = oracle_f1 - fixed[fixed_action]["f1"]
    gap = f1[:, 0] - f1[:, 1]
    non_ties = np.abs(gap) > 1e-12
    high = np.abs(gap) >= high_margin
    actual = np.where(gap > 0.0, 0, 1)
    return {
        "utility": chosen,
        "gain_over_best_fixed": {
            metric: chosen[metric] - fixed[fixed_action][metric] for metric in METRICS
        },
        "best_fixed": fixed_action,
        "fixed_utility": fixed[fixed_action],
        "fixed_actions": fixed,
        "random_expected_f1": float(np.mean(np.mean(f1, axis=1))),
        "oracle_f1": oracle_f1,
        "regret": oracle_f1 - chosen["f1"],
        "oracle_recovery": float(gain / headroom) if headroom > 0 else 0.0,
        "pairwise_accuracy": float(np.mean(actions[non_ties] == actual[non_ties])) if np.any(non_ties) else 0.0,
        "pairwise_queries": int(np.sum(non_ties)),
        "high_margin_accuracy": float(np.mean(actions[high] == actual[high])) if np.any(high) else 0.0,
        "high_margin_queries": int(np.sum(high)),
        "selection": {
            action: int(np.sum(actions == index)) for index, action in enumerate(ACTIONS)
        },
        "selection_fraction": {
            action: float(np.mean(actions == index)) for index, action in enumerate(ACTIONS)
        },
        "actions": actions,
        "fixed_index": fixed_index,
    }


def _serializable_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"actions", "fixed_index"}}


def _cross_validated_candidates(
    candidates: Sequence[str],
    matrices: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    cv_seed: int,
    seeds: Sequence[int],
    high_margin: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    fold_indices = list(splitter.split(train_indices, groups=groups))
    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    train_targets = {metric: value[train_indices] for metric, value in targets.items()}
    for candidate in candidates:
        feature_set, _ = _candidate_parts(candidate)
        matrix = matrices[feature_set][train_indices]
        oof = np.empty((len(train_indices), len(ACTIONS)), dtype=np.float64)
        started = time.perf_counter()
        for fit_positions, test_positions in fold_indices:
            models = _fit_models(
                candidate,
                matrix[fit_positions],
                train_targets["f1"][fit_positions],
                seeds,
            )
            oof[test_positions] = _ensemble_predict(models, matrix[test_positions])
        elapsed = time.perf_counter() - started
        policy = _policy_metrics(oof, train_targets, high_margin=high_margin)
        results[candidate] = {
            "feature_set": feature_set,
            "model": _candidate_parts(candidate)[1],
            "oof": _serializable_policy(policy),
            "training_seconds": elapsed,
        }
        predictions[candidate] = oof
    return results, predictions


def _nested_gain_bootstrap(
    records: Sequence[Mapping[str, Any]],
    actions: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    if len(records) != len(actions):
        raise ValueError("Bootstrap records and actions must align")
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_group[str(record["group_id"])].append(index)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    samples = {metric: np.empty(resamples, dtype=np.float64) for metric in METRICS}
    for position in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        differences = {metric: [] for metric in METRICS}
        for group in sampled_groups:
            for index in by_group[str(group)]:
                if int(actions[index]) == 1:
                    for metric in METRICS:
                        differences[metric].append(0.0)
                    continue
                bm25 = records[index]["repeats"]["bm25"]
                dense = records[index]["repeats"]["dense"]
                count = len(bm25["f1"])
                bm25_ids = rng.integers(0, count, size=count)
                dense_ids = rng.integers(0, count, size=count)
                for metric in METRICS:
                    selected_value = np.mean(np.asarray(bm25[metric])[bm25_ids])
                    fixed_value = np.mean(np.asarray(dense[metric])[dense_ids])
                    differences[metric].append(float(selected_value - fixed_value))
        for metric in METRICS:
            samples[metric][position] = float(np.mean(differences[metric]))
    return {
        metric: [float(value) for value in np.quantile(samples[metric], [0.025, 0.975])]
        for metric in METRICS
    }


def run_phase3(config_path: Path, source_path: Path, output_dir: Path) -> dict[str, Any]:
    router = _load_router_config(config_path)
    phase3 = router["phase3"]
    if phase3["final_holdout_outcomes"] != "forbidden":
        raise ValueError("Phase 3 final_holdout outcomes must remain forbidden")
    candidates = [str(value) for value in phase3["candidates"]]
    records = _aggregate_phase2_rows(
        _read_jsonl(source_path),
        expected_counts=phase3["source_partitions"],
        expected_repeats=int(phase3["repeats_per_query_action"]),
    )
    matrices, feature_names, feature_cost = _extract_features(records, router)
    targets = _targets(records)
    train_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "train"])
    dev_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "dev"])
    train_groups = np.asarray([records[i]["group_id"] for i in train_indices], dtype=object)
    seeds = [int(value) for value in phase3["training_seeds"]]
    high_margin = float(phase3["high_margin_f1_gap"])

    cv_results, oof_predictions = _cross_validated_candidates(
        candidates,
        matrices,
        targets,
        train_indices,
        train_groups,
        folds=int(phase3["cv_folds"]),
        cv_seed=int(phase3["cv_seed"]),
        seeds=seeds,
        high_margin=high_margin,
    )
    official = max(
        candidates,
        key=lambda candidate: (
            cv_results[candidate]["oof"]["utility"]["f1"],
            -candidates.index(candidate),
        ),
    )

    dev_targets = {metric: value[dev_indices] for metric, value in targets.items()}
    dev_results: dict[str, Any] = {}
    full_models: dict[str, list[Any]] = {}
    dev_predictions: dict[str, np.ndarray] = {}
    inference_cost: dict[str, float] = {}
    for candidate in candidates:
        feature_set, _ = _candidate_parts(candidate)
        models = _fit_models(
            candidate,
            matrices[feature_set][train_indices],
            targets["f1"][train_indices],
            seeds,
        )
        started = time.perf_counter()
        prediction = _ensemble_predict(models, matrices[feature_set][dev_indices])
        inference_cost[candidate] = 1000.0 * (time.perf_counter() - started) / len(dev_indices)
        policy = _policy_metrics(prediction, dev_targets, high_margin=high_margin)
        dev_results[candidate] = _serializable_policy(policy)
        full_models[candidate] = models
        dev_predictions[candidate] = prediction

    official_models = full_models[official]
    official_prediction = dev_predictions[official]
    official_policy = _policy_metrics(official_prediction, dev_targets, high_margin=high_margin)
    dev_records = [records[index] for index in dev_indices]
    intervals = _nested_gain_bootstrap(
        dev_records,
        official_policy["actions"],
        resamples=int(phase3["bootstrap_resamples"]),
        seed=int(phase3["bootstrap_seed"]),
    )

    individual_seed_gains: list[dict[str, Any]] = []
    _, official_model_kind = _candidate_parts(official)
    if official_model_kind == "ridge":
        seed_stable = True
        seed_mode = "deterministic"
    else:
        for seed, model in zip(_seeds_for_model(official_model_kind, seeds), official_models):
            prediction = np.asarray(
                model.predict(matrices[_candidate_parts(official)[0]][dev_indices]),
                dtype=np.float64,
            )
            policy = _policy_metrics(prediction, dev_targets, high_margin=high_margin)
            individual_seed_gains.append(
                {"seed": seed, "f1_gain": policy["gain_over_best_fixed"]["f1"]}
            )
        gains = [row["f1_gain"] for row in individual_seed_gains]
        seed_stable = all(gain > 0.0 for gain in gains) and float(np.median(gains)) >= float(
            phase3["practical_f1_gain"]
        )
        seed_mode = "five_seed_ensemble"

    official_gain = official_policy["gain_over_best_fixed"]
    f1_point_passed = official_gain["f1"] >= float(phase3["practical_f1_gain"])
    f1_interval_passed = intervals["f1"][0] > float(phase3["f1_ci_lower_bound"])
    ac_point_passed = official_gain["ac"] >= -float(phase3["ac_noninferiority_margin"])
    ac_interval_passed = intervals["ac"][0] >= -float(phase3["ac_noninferiority_margin"])
    complete = len(records) == sum(int(value) for value in phase3["source_partitions"].values())
    if not complete:
        decision = "REVISE"
    elif not (f1_point_passed and f1_interval_passed and seed_stable):
        decision = "STOP"
    elif not (ac_point_passed and ac_interval_passed):
        decision = "REVISE"
    else:
        decision = "GO"

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        feature_set: {
            "tier": "A" if feature_set in phase3["tier_a_feature_sets"] else "B",
            "dimension": len(names),
            "features": names,
        }
        for feature_set, names in feature_names.items()
    }
    _write_json(output_dir / "feature_schema.json", schema)
    np.savez_compressed(
        output_dir / "features.npz",
        query_ids=np.asarray([row["query_id"] for row in records]),
        partitions=np.asarray([row["split"] for row in records]),
        groups=np.asarray([row["group_id"] for row in records]),
        surface=matrices["surface"],
        lexical=matrices["lexical"],
        embedding=matrices["embedding"],
        combined=matrices["combined"],
        f1=targets["f1"],
        em=targets["em"],
        ac=targets["ac"],
    )
    joblib.dump(
        {
            "candidate": official,
            "feature_set": _candidate_parts(official)[0],
            "models": official_models,
        },
        output_dir / "official_model.joblib",
    )
    _write_json(
        output_dir / "cv_results.json",
        {
            "selection_source": "train_group_cv_only",
            "folds": int(phase3["cv_folds"]),
            "official_candidate": official,
            "candidates": cv_results,
        },
    )

    prediction_rows = []
    for position, record in enumerate(dev_records):
        action_index = int(official_policy["actions"][position])
        prediction_rows.append(
            {
                "query_id": record["query_id"],
                "group_id": record["group_id"],
                "split": "dev",
                "action": ACTIONS[action_index],
                "predicted_f1": {
                    action: float(official_prediction[position, index])
                    for index, action in enumerate(ACTIONS)
                },
                "observed": {
                    metric: {
                        action: float(dev_targets[metric][position, index])
                        for index, action in enumerate(ACTIONS)
                    }
                    for metric in METRICS
                },
            }
        )
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)

    summary = {
        "phase": 3,
        "status": "complete",
        "data": {
            "train_queries": int(len(train_indices)),
            "dev_queries": int(len(dev_indices)),
            "repeats_per_query_action": int(phase3["repeats_per_query_action"]),
            "final_holdout_outcomes": 0,
        },
        "protocol": {
            "selection": "train_group_cv_only",
            "dev": "one_time_gate_evaluation",
            "prediction": "two_action_f1_utility_then_argmax_with_dense_tie_break",
            "uncertainty": "group_plus_generation_repeat_bootstrap",
            "posthoc_candidate_switching": "forbidden",
        },
        "feature_cost": feature_cost,
        "model_inference_ms_per_query": inference_cost,
        "train_cv": {
            "official_candidate": official,
            "official_oof": cv_results[official]["oof"],
            "candidates": cv_results,
        },
        "dev": {
            "official_candidate": official,
            "official": _serializable_policy(official_policy),
            "gain_ci95": intervals,
            "candidates_descriptive_only": dev_results,
            "seed_stability": {
                "mode": seed_mode,
                "individual": individual_seed_gains,
                "passed": seed_stable,
            },
        },
        "gate": {
            "f1_point_at_least_practical": f1_point_passed,
            "f1_interval_lower_above_zero": f1_interval_passed,
            "ac_point_noninferior": ac_point_passed,
            "ac_interval_noninferior": ac_interval_passed,
            "cross_seed_stable": seed_stable,
            "complete_without_failures": complete,
            "decision": decision,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir.parents[1] / "summary.json", summary)
    return summary


def main() -> None:
    args = _arguments()
    summary = run_phase3(_resolve(args.config), _resolve(args.source_results), _resolve(args.output_dir))
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
