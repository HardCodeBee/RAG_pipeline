#!/usr/bin/env python3
"""Diagnose BM25-only/BGE-only query structure in a frozen DPR space.

The script consumes saved qrels-derived per-query retrieval metrics.  It does
not run retrieval, reranking, generation, or an online router.  Its primary
question is deliberately narrower: are exclusive BM25/BGE success labels
decodable from an independent frozen DPR question representation?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.embedders.text_embedder import create_embedder, l2_normalize
from src.retrievers.sqlite_bm25 import analyze_sqlite_bm25_text


SEED = 20260821
BALANCE_REPEATS = 20
BOOTSTRAP_REPEATS = 2000
PERMUTATION_REPEATS = 1000
K_VALUES = (10, 20, 50)
LABEL_TO_INT = {"bm25_only": 0, "bge_only": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.ASCII | re.IGNORECASE)
WH_WORDS = ("what", "who", "when", "where", "why", "how", "which", "whom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-csv",
        type=Path,
        default=Path(
            "docs/assets/history_aware_retriever_policy/analysis/"
            "query_policy_oracle.csv"
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/beir/nq/queries/queries.jsonl"),
    )
    parser.add_argument(
        "--dpr-config",
        type=Path,
        default=Path("configs/beir/nfcorpus_dense_dpr_top50.yaml"),
    )
    parser.add_argument(
        "--bm25-run-metadata",
        type=Path,
        default=Path(
            "outputs/beir_nq_four_condition_full_v2/units/nq/runs/"
            "bm25_top5/metadata.json"
        ),
    )
    parser.add_argument(
        "--sparse-root",
        type=Path,
        default=Path("artifacts/_sparse_indexes"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/representation_probe/nq_bm25_bge_dpr_v1"),
    )
    parser.add_argument("--unit", default="nq")
    parser.add_argument("--protocol", default="equal50")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_query(text: str) -> str:
    return " ".join(text.casefold().split())


def read_oracle_rows(path: Path, protocol: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw["protocol"] != protocol:
            continue
        bm25 = float(raw["bm25_ndcg_at_10"])
        bge = float(raw["dense_ndcg_at_10"])
        if not (math.isfinite(bm25) and math.isfinite(bge)):
            raise ValueError(f"non-finite nDCG for query {raw['query_id']}")
        rows.append(
            {
                "family": raw["family"],
                "unit": raw["unit"],
                "query_id": raw["query_id"],
                "bm25_ndcg_at_10": bm25,
                "bge_ndcg_at_10": bge,
            }
        )
    if not rows:
        raise RuntimeError(f"no rows found for protocol={protocol}")
    query_keys = [(row["unit"], row["query_id"]) for row in rows]
    if len(query_keys) != len(set(query_keys)):
        raise RuntimeError("duplicate unit/query pairs in oracle CSV")
    return rows


def exclusive_label(bm25: float, bge: float) -> str | None:
    bm25_success = bm25 > 0.0
    bge_success = bge > 0.0
    if bm25_success and not bge_success:
        return "bm25_only"
    if bge_success and not bm25_success:
        return "bge_only"
    return None


def selection_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit"])].append(row)
    result: list[dict[str, Any]] = []
    for unit, unit_rows in by_unit.items():
        labels = [
            exclusive_label(row["bm25_ndcg_at_10"], row["bge_ndcg_at_10"])
            for row in unit_rows
        ]
        counts = Counter(label for label in labels if label is not None)
        result.append(
            {
                "unit": unit,
                "num_queries": len(unit_rows),
                "bm25_only": counts["bm25_only"],
                "bge_only": counts["bge_only"],
                "exclusive_total": counts["bm25_only"] + counts["bge_only"],
            }
        )
    return sorted(result, key=lambda row: (-row["exclusive_total"], row["unit"]))


def read_questions(path: Path) -> tuple[dict[str, str], list[str]]:
    questions: dict[str, str] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query_id = str(row["_id"])
            text = str(row["text"]).strip()
            if not text:
                raise RuntimeError(f"empty query text at {path}:{line_number}")
            if query_id in questions:
                raise RuntimeError(f"duplicate query ID: {query_id}")
            questions[query_id] = text
            order.append(query_id)
    return questions, order


def build_labeled_rows(
    rows: Sequence[dict[str, Any]],
    questions: dict[str, str],
    query_order: Sequence[str],
    unit: str,
    margin: float,
) -> list[dict[str, Any]]:
    unit_rows = {row["query_id"]: row for row in rows if row["unit"] == unit}
    if set(unit_rows) != set(questions):
        missing_metrics = sorted(set(questions) - set(unit_rows))[:5]
        missing_questions = sorted(set(unit_rows) - set(questions))[:5]
        raise RuntimeError(
            "query/metric ID mismatch: "
            f"missing_metrics={missing_metrics}, missing_questions={missing_questions}"
        )
    result: list[dict[str, Any]] = []
    for query_id in query_order:
        row = unit_rows[query_id]
        label = exclusive_label(row["bm25_ndcg_at_10"], row["bge_ndcg_at_10"])
        if label is None:
            continue
        delta = row["bge_ndcg_at_10"] - row["bm25_ndcg_at_10"]
        result.append(
            {
                "query_id": query_id,
                "question": questions[query_id],
                "label": label,
                "label_int": LABEL_TO_INT[label],
                "bm25_ndcg_at_10": row["bm25_ndcg_at_10"],
                "bge_ndcg_at_10": row["bge_ndcg_at_10"],
                "bge_minus_bm25_ndcg_at_10": delta,
                "passes_abs_margin": abs(delta) >= margin,
            }
        )
    return result


def resolve_sparse_index(
    metadata_path: Path, sparse_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_build_id = str(metadata["build_id"])
    matches: list[tuple[Path, Path, dict[str, Any]]] = []
    for directory in sparse_root.iterdir():
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("backend") == "sqlite_bm25_v1"
            and manifest.get("identity", {}).get("source_build_id") == source_build_id
        ):
            database = directory / manifest["artifacts"]["database"]["file"]
            matches.append((database, manifest_path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one complete SQLite BM25 index for {source_build_id}, "
            f"found {len(matches)}"
        )
    database, manifest_path, manifest = matches[0]
    if not database.is_file():
        raise FileNotFoundError(database)
    return database, manifest_path, manifest


def read_idf(database: Path, terms: Iterable[str]) -> dict[str, float]:
    unique = sorted(set(terms))
    result: dict[str, float] = {}
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        for start in range(0, len(unique), 800):
            batch = unique[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT term, idf FROM term_stats WHERE term IN ({placeholders})"
            result.update(
                (str(term), float(idf))
                for term, idf in connection.execute(query, batch).fetchall()
            )
    finally:
        connection.close()
    return result


def surface_features(
    labeled_rows: Sequence[dict[str, Any]], idf_by_term: dict[str, float]
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    names = [
        "char_count",
        "token_count",
        "unique_token_ratio",
        "analyzed_token_count",
        "stopword_fraction",
        "digit_token_fraction",
        "punctuation_fraction",
        "question_mark",
        "mean_idf",
        "max_idf",
        "min_idf",
        "idf_oov_fraction",
    ] + [f"wh_{word}" for word in WH_WORDS]
    matrix: list[list[float]] = []
    feature_rows: list[dict[str, Any]] = []
    for row in labeled_rows:
        text = str(row["question"])
        raw_tokens = [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]
        analyzed = list(analyze_sqlite_bm25_text(text))
        token_count = len(raw_tokens)
        unique_ratio = len(set(raw_tokens)) / token_count if token_count else 0.0
        stopword_fraction = (
            1.0 - (len(analyzed) / token_count) if token_count else 0.0
        )
        digit_fraction = (
            sum(any(char.isdigit() for char in token) for token in raw_tokens)
            / token_count
            if token_count
            else 0.0
        )
        punctuation_fraction = (
            sum(not char.isalnum() and not char.isspace() for char in text) / len(text)
            if text
            else 0.0
        )
        idfs = [idf_by_term[token] for token in analyzed if token in idf_by_term]
        oov_fraction = (
            sum(token not in idf_by_term for token in analyzed) / len(analyzed)
            if analyzed
            else 0.0
        )
        wh_values = [float(word in raw_tokens) for word in WH_WORDS]
        values = [
            float(len(text)),
            float(token_count),
            float(unique_ratio),
            float(len(analyzed)),
            float(stopword_fraction),
            float(digit_fraction),
            float(punctuation_fraction),
            float("?" in text),
            float(np.mean(idfs)) if idfs else 0.0,
            float(max(idfs)) if idfs else 0.0,
            float(min(idfs)) if idfs else 0.0,
            float(oov_fraction),
            *wh_values,
        ]
        matrix.append(values)
        feature_rows.append(
            {"query_id": row["query_id"], **dict(zip(names, values, strict=True))}
        )
    result = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("surface features contain non-finite values")
    return result, names, feature_rows


def group_ids(labeled_rows: Sequence[dict[str, Any]]) -> np.ndarray:
    normalized_to_group: dict[str, int] = {}
    normalized_to_label: dict[str, int] = {}
    result: list[int] = []
    for row in labeled_rows:
        normalized = normalized_query(str(row["question"]))
        label = int(row["label_int"])
        if normalized in normalized_to_label and normalized_to_label[normalized] != label:
            raise RuntimeError("normalized duplicate query has conflicting labels")
        normalized_to_label[normalized] = label
        if normalized not in normalized_to_group:
            normalized_to_group[normalized] = len(normalized_to_group)
        result.append(normalized_to_group[normalized])
    return np.asarray(result, dtype=np.int64)


def metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
    }


def bootstrap_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    group_to_rows: dict[int, np.ndarray] = {
        int(group): np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    groups_by_label: dict[int, list[int]] = {0: [], 1: []}
    for group, indices in group_to_rows.items():
        labels = np.unique(y[indices])
        if len(labels) != 1:
            raise RuntimeError("one query group contains multiple labels")
        groups_by_label[int(labels[0])].append(group)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sampled_rows: list[int] = []
        for label in (0, 1):
            candidates = np.asarray(groups_by_label[label], dtype=np.int64)
            sampled_groups = rng.choice(candidates, size=len(candidates), replace=True)
            for group in sampled_groups:
                sampled_rows.extend(group_to_rows[int(group)].tolist())
        indices = np.asarray(sampled_rows, dtype=np.int64)
        metrics = metric_values(y[indices], probabilities[indices])
        for key, value in metrics.items():
            values[key].append(value)
    return dict(values)


def confidence_interval(values: Sequence[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    name: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
        raise ValueError(f"invalid probe inputs for {name}")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(x, y, groups), 1):
        inner = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=seed + fold
        )
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=seed + fold,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            pipeline,
            {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(x[train], y[train], groups=groups[train])
        fold_probabilities = search.predict_proba(x[test])[:, 1]
        probabilities[test] = fold_probabilities
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "best_c": float(search.best_params_["model__C"]),
                "inner_cv_roc_auc": float(search.best_score_),
                **metric_values(y[test], fold_probabilities),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"missing OOF probabilities for {name}")
    observed = metric_values(y, probabilities)
    bootstrap = bootstrap_metrics(
        y, probabilities, groups, BOOTSTRAP_REPEATS, seed + 1000
    )
    rng = np.random.default_rng(seed + 2000)
    null_auc = np.asarray(
        [roc_auc_score(rng.permutation(y), probabilities) for _ in range(PERMUTATION_REPEATS)]
    )
    result = {
        "name": name,
        "rows": len(y),
        "dimensions": int(x.shape[1]),
        "class_counts": {
            INT_TO_LABEL[label]: int(np.sum(y == label)) for label in (0, 1)
        },
        "oof": observed,
        "bootstrap_ci95": {
            key: confidence_interval(values) for key, values in bootstrap.items()
        },
        "permutation": {
            "repeats": PERMUTATION_REPEATS,
            "null_auc_mean": float(np.mean(null_auc)),
            "null_auc_ci95": confidence_interval(null_auc.tolist()),
            "p_greater_equal": float(
                (1 + np.sum(null_auc >= observed["roc_auc"]))
                / (PERMUTATION_REPEATS + 1)
            ),
        },
        "folds": fold_rows,
    }
    return result, probabilities


def paired_auc_difference(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_to_rows: dict[int, np.ndarray] = {
        int(group): np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    groups_by_label: dict[int, list[int]] = {0: [], 1: []}
    for group, indices in group_to_rows.items():
        label = int(np.unique(y[indices])[0])
        groups_by_label[label].append(group)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled_rows: list[int] = []
        for label in (0, 1):
            candidates = np.asarray(groups_by_label[label], dtype=np.int64)
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            for group in sampled:
                sampled_rows.extend(group_to_rows[int(group)].tolist())
        indices = np.asarray(sampled_rows, dtype=np.int64)
        values.append(
            float(
                roc_auc_score(y[indices], left[indices])
                - roc_auc_score(y[indices], right[indices])
            )
        )
    observed = float(roc_auc_score(y, left) - roc_auc_score(y, right))
    return {"observed": observed, "bootstrap_ci95": confidence_interval(values)}


def balanced_geometry(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    bm25_indices = np.flatnonzero(labels == LABEL_TO_INT["bm25_only"])
    bge_indices = np.flatnonzero(labels == LABEL_TO_INT["bge_only"])
    minority = min(len(bm25_indices), len(bge_indices))
    if minority <= max(K_VALUES):
        raise RuntimeError("too few minority rows for requested neighborhoods")
    rows: list[dict[str, Any]] = []
    fixed_indices: np.ndarray | None = None
    for repeat in range(BALANCE_REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        sampled_bm25 = rng.choice(bm25_indices, size=minority, replace=False)
        sampled_bge = rng.choice(bge_indices, size=minority, replace=False)
        indices = np.concatenate([sampled_bm25, sampled_bge])
        rng.shuffle(indices)
        if repeat == 0:
            fixed_indices = indices.copy()
        x = embeddings[indices]
        y = labels[indices]
        neighbors = NearestNeighbors(
            n_neighbors=max(K_VALUES) + 1,
            metric="cosine",
            algorithm="brute",
        ).fit(x)
        neighbor_indices = neighbors.kneighbors(x, return_distance=False)[:, 1:]
        row: dict[str, Any] = {
            "repeat": repeat,
            "seed": SEED + repeat,
            "rows_per_class": minority,
            "silhouette_cosine": float(silhouette_score(x, y, metric="cosine")),
        }
        means = [np.mean(x[y == label], axis=0) for label in (0, 1)]
        denominator = np.linalg.norm(means[0]) * np.linalg.norm(means[1])
        row["centroid_cosine_distance"] = float(
            1.0 - (np.dot(means[0], means[1]) / denominator)
        )
        for k in K_VALUES:
            local = neighbor_indices[:, :k]
            row[f"same_label_purity_at_{k}"] = float(
                np.mean(labels[indices][local] == y[:, None])
            )
        rows.append(row)
    if fixed_indices is None:
        raise AssertionError("balanced geometry did not execute")
    return rows, fixed_indices


def aggregate_geometry(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = ["silhouette_cosine", "centroid_cosine_distance"] + [
        f"same_label_purity_at_{k}" for k in K_VALUES
    ]
    return {
        key: {
            "median": float(np.median([float(row[key]) for row in rows])),
            "min": float(np.min([float(row[key]) for row in rows])),
            "max": float(np.max([float(row[key]) for row in rows])),
            "interval_95_across_balance_seeds": confidence_interval(
                [float(row[key]) for row in rows]
            ),
        }
        for key in keys
    }


def plot_projection(
    output: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    skip_tsne: bool,
) -> dict[str, Any]:
    x = embeddings[indices]
    y = labels[indices]
    pca = PCA(n_components=2, random_state=SEED)
    pca_values = pca.fit_transform(x)
    projections: list[tuple[str, np.ndarray]] = [("PCA", pca_values)]
    metadata: dict[str, Any] = {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tsne_executed": False,
        "umap_executed": False,
        "umap_reason": "umap-learn is not installed; qualitative fallback is t-SNE",
    }
    if not skip_tsne:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=SEED,
        )
        projections.append(("t-SNE (qualitative)", tsne.fit_transform(x)))
        metadata["tsne_executed"] = True
    fig, axes = plt.subplots(1, len(projections), figsize=(7 * len(projections), 6))
    if len(projections) == 1:
        axes = [axes]
    colors = {0: "#D55E00", 1: "#0072B2"}
    names = {0: "BM25-only", 1: "BGE-only"}
    for axis, (title, values) in zip(axes, projections, strict=True):
        for label in (0, 1):
            mask = y == label
            axis.scatter(
                values[mask, 0],
                values[mask, 1],
                s=20,
                alpha=0.62,
                c=colors[label],
                label=names[label],
                edgecolors="none",
            )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(frameon=False)
    fig.suptitle("Frozen DPR question space: balanced NQ exclusive-success queries")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metadata


def plot_neighborhood(output: Path, rows: Sequence[dict[str, Any]]) -> None:
    data = [
        [float(row[f"same_label_purity_at_{k}"]) for row in rows] for k in K_VALUES
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(data, tick_labels=[str(k) for k in K_VALUES], showmeans=True)
    axis.axhline(0.5, color="#666666", linestyle="--", label="balanced mixing baseline")
    axis.axhline(0.55, color="#D55E00", linestyle=":", label="pre-registered practical line")
    axis.set_xlabel("k nearest neighbors (cosine)")
    axis.set_ylabel("same-label neighbor fraction")
    axis.set_title("Neighborhood mixing across 20 balanced samples")
    axis.set_ylim(0.45, max(0.65, max(max(values) for values in data) + 0.02))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_probe_probabilities(
    output: Path, labels: np.ndarray, probabilities: dict[str, np.ndarray]
) -> None:
    names = [("surface", "Surface"), ("dpr_cosine", "DPR"), ("surface_plus_dpr", "Surface + DPR")]
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4.5), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, (key, title) in zip(axes, names, strict=True):
        for label, color, label_name in (
            (0, "#D55E00", "BM25-only"),
            (1, "#0072B2", "BGE-only"),
        ):
            axis.hist(
                probabilities[key][labels == label],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=label_name,
            )
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("OOF probability of BGE-only")
    axes[0].set_ylabel("density")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def class_summary(
    labels: np.ndarray, raw_embeddings: np.ndarray, surface: np.ndarray, names: Sequence[str]
) -> dict[str, Any]:
    norms = np.linalg.norm(raw_embeddings, axis=1)
    result: dict[str, Any] = {}
    for label in (0, 1):
        mask = labels == label
        result[INT_TO_LABEL[label]] = {
            "rows": int(np.sum(mask)),
            "raw_dpr_norm_mean": float(np.mean(norms[mask])),
            "raw_dpr_norm_std": float(np.std(norms[mask], ddof=1)),
            "surface_mean": {
                name: float(np.mean(surface[mask, index]))
                for index, name in enumerate(names)
            },
        }
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    counts = summary["dataset"]["class_counts"]
    geometry = summary["geometry"]["aggregate"]
    probes = summary["probes"]
    dpr = probes["dpr_cosine"]
    surface = probes["surface"]
    combined = probes["surface_plus_dpr"]
    sensitivity = probes["dpr_margin_sensitivity"]
    decision = summary["decision"]
    lines = [
        "# NQ BM25/BGE exclusive-success DPR 空间诊断",
        "",
        "## 冻结范围",
        "",
        "- 标签来自现有 qrels-based first-stage nDCG@10；未重跑检索、reranker 或生成。",
        "- BM25-only：BM25 nDCG@10 > 0 且 BGE nDCG@10 = 0。",
        "- BGE-only：BGE nDCG@10 > 0 且 BM25 nDCG@10 = 0。",
        f"- 样本：BM25-only {counts['bm25_only']}，BGE-only {counts['bge_only']}。",
        "- 表示：facebook/dpr-question_encoder-multiset-base 的 pooler_output；余弦分析使用 L2 副本。",
        "",
        "## 主要结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| DPR OOF ROC-AUC | {dpr['oof']['roc_auc']:.4f} |",
        f"| DPR ROC-AUC 95% CI | [{dpr['bootstrap_ci95']['roc_auc'][0]:.4f}, {dpr['bootstrap_ci95']['roc_auc'][1]:.4f}] |",
        f"| Surface OOF ROC-AUC | {surface['oof']['roc_auc']:.4f} |",
        f"| Surface + DPR OOF ROC-AUC | {combined['oof']['roc_auc']:.4f} |",
        f"| Margin sensitivity DPR ROC-AUC | {sensitivity['oof']['roc_auc']:.4f} |",
        f"| k=10 邻域同标签比例（20 次中位数） | {geometry['same_label_purity_at_10']['median']:.4f} |",
        f"| k=20 邻域同标签比例（20 次中位数） | {geometry['same_label_purity_at_20']['median']:.4f} |",
        f"| k=50 邻域同标签比例（20 次中位数） | {geometry['same_label_purity_at_50']['median']:.4f} |",
        f"| cosine silhouette（20 次中位数） | {geometry['silhouette_cosine']['median']:.4f} |",
        "",
        "## 预注册判定",
        "",
        f"**{decision['label']}**：{decision['explanation']}",
        "",
        "这个结论只表示：在 NQ 的冻结检索协议下，DPR query representation 是否包含与 BM25/BGE exclusive-success 标签相关的可解码信号。它不证明语言空间天然知道最佳 retriever，也不证明该信号可以带来端到端 router 增益。",
        "",
        "NQ 与 DPR 训练范式可能存在 encoder-dataset alignment；若为阳性，下一步应使用相同 DPR encoder 在 HotpotQA 上复现。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_paths = [
        args.oracle_csv,
        args.questions,
        args.dpr_config,
        args.bm25_run_metadata,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Reading frozen retrieval labels", flush=True)
    oracle_rows = read_oracle_rows(args.oracle_csv, args.protocol)
    selection = selection_rows(oracle_rows)
    write_csv(args.output_dir / "dataset_selection.csv", selection)
    if selection[0]["unit"] != args.unit:
        raise RuntimeError(
            f"selected unit {args.unit} is not the largest individual unit; "
            f"current largest is {selection[0]['unit']}"
        )
    questions, query_order = read_questions(args.questions)
    labeled = build_labeled_rows(
        oracle_rows, questions, query_order, args.unit, args.margin
    )
    label_counts = Counter(row["label"] for row in labeled)
    expected_counts = {"bm25_only": 260, "bge_only": 1049}
    if dict(label_counts) != expected_counts:
        raise RuntimeError(
            f"frozen class counts changed: expected {expected_counts}, got {dict(label_counts)}"
        )

    print("[2/6] Resolving corpus-static BM25 IDF and surface features", flush=True)
    database, sparse_manifest_path, sparse_manifest = resolve_sparse_index(
        args.bm25_run_metadata, args.sparse_root
    )
    all_terms = [
        term
        for row in labeled
        for term in analyze_sqlite_bm25_text(str(row["question"]))
    ]
    idf_by_term = read_idf(database, all_terms)
    surface, surface_names, surface_rows = surface_features(labeled, idf_by_term)
    write_csv(args.output_dir / "surface_features.csv", surface_rows)

    print("[3/6] Encoding 1,309 queries with the frozen DPR question encoder", flush=True)
    config = load_config(args.dpr_config)
    embedder = create_embedder(config, role="query")
    if (
        embedder.model_name != "facebook/dpr-question_encoder-multiset-base"
        or embedder.resolved_revision
        != "5325e4ee906435291d63046f535476cb3fc60d43"
        or embedder.pooling != "pooler_output"
        or embedder.normalize
    ):
        raise RuntimeError("DPR query encoder contract does not match the frozen experiment")
    encode_start = time.perf_counter()
    raw_embeddings = embedder.encode_queries([str(row["question"]) for row in labeled])
    encode_seconds = time.perf_counter() - encode_start
    normalized_embeddings = l2_normalize(raw_embeddings).astype(np.float32)
    if raw_embeddings.shape != (len(labeled), 768):
        raise RuntimeError(f"unexpected DPR embedding shape: {raw_embeddings.shape}")
    labels = np.asarray([int(row["label_int"]) for row in labeled], dtype=np.int64)
    groups = group_ids(labeled)
    np.savez_compressed(
        args.output_dir / "dpr_embeddings.npz",
        raw=raw_embeddings,
        l2_normalized=normalized_embeddings,
        labels=labels,
        query_ids=np.asarray([str(row["query_id"]) for row in labeled]),
    )
    with (args.output_dir / "labeled_queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("[4/6] Measuring balanced-space neighborhood overlap", flush=True)
    geometry_rows, fixed_indices = balanced_geometry(normalized_embeddings, labels)
    write_csv(args.output_dir / "balanced_geometry.csv", geometry_rows)
    geometry_aggregate = aggregate_geometry(geometry_rows)
    projection = plot_projection(
        figures_dir / "pca_tsne.png",
        normalized_embeddings,
        labels,
        fixed_indices,
        skip_tsne=args.skip_tsne,
    )
    plot_neighborhood(figures_dir / "neighborhood_purity.png", geometry_rows)

    print("[5/6] Running nested grouped OOF linear probes", flush=True)
    feature_sets = {
        "surface": surface,
        "dpr_cosine": normalized_embeddings.astype(np.float64),
        "surface_plus_dpr": np.concatenate(
            [surface, normalized_embeddings.astype(np.float64)], axis=1
        ),
    }
    probes: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for offset, (name, features) in enumerate(feature_sets.items()):
        print(f"      probe={name}", flush=True)
        probes[name], probabilities[name] = fit_probe(
            features, labels, groups, name=name, seed=SEED + 100 * offset
        )
    margin_mask = np.asarray([bool(row["passes_abs_margin"]) for row in labeled])
    probes["dpr_margin_sensitivity"], margin_probabilities = fit_probe(
        normalized_embeddings[margin_mask].astype(np.float64),
        labels[margin_mask],
        groups[margin_mask],
        name="dpr_margin_sensitivity",
        seed=SEED + 500,
    )
    probes["paired_auc_differences"] = {
        "dpr_minus_surface": paired_auc_difference(
            labels,
            probabilities["dpr_cosine"],
            probabilities["surface"],
            groups,
            SEED + 600,
        ),
        "combined_minus_surface": paired_auc_difference(
            labels,
            probabilities["surface_plus_dpr"],
            probabilities["surface"],
            groups,
            SEED + 700,
        ),
    }
    prediction_rows = []
    for index, row in enumerate(labeled):
        prediction_rows.append(
            {
                "query_id": row["query_id"],
                "label": row["label"],
                "surface_oof_probability_bge_only": probabilities["surface"][index],
                "dpr_oof_probability_bge_only": probabilities["dpr_cosine"][index],
                "combined_oof_probability_bge_only": probabilities["surface_plus_dpr"][index],
                "margin_sensitivity_included": bool(margin_mask[index]),
            }
        )
    write_csv(args.output_dir / "probe_predictions.csv", prediction_rows)
    plot_probe_probabilities(figures_dir / "probe_probabilities.png", labels, probabilities)

    print("[6/6] Freezing manifest and report", flush=True)
    dpr_probe = probes["dpr_cosine"]
    sensitivity_probe = probes["dpr_margin_sensitivity"]
    purity_pass = all(
        geometry_aggregate[f"same_label_purity_at_{k}"]["median"] >= 0.55
        for k in K_VALUES
    )
    decision_checks = {
        "dpr_auc_at_least_0_60": dpr_probe["oof"]["roc_auc"] >= 0.60,
        "dpr_auc_ci_lower_above_0_50": dpr_probe["bootstrap_ci95"]["roc_auc"][0]
        > 0.50,
        "all_knn_purity_medians_at_least_0_55": purity_pass,
        "margin_sensitivity_auc_at_least_0_55": sensitivity_probe["oof"]["roc_auc"]
        >= 0.55,
    }
    passed = all(decision_checks.values())
    decision = {
        "passed": passed,
        "label": "存在实用可解码结构" if passed else "未达到预注册的实用结构阈值",
        "checks": decision_checks,
        "explanation": (
            "DPR 空间同时通过线性可解码性、邻域纯度和 margin sensitivity 三类检查。"
            if passed
            else "至少一项线性可解码性、邻域纯度或 margin sensitivity 检查未通过；降维图不能改变该判定。"
        ),
    }
    summary = {
        "scope": {
            "kind": "offline_qrels_labeled_representation_diagnostic",
            "retrieval_executed": False,
            "reranking_executed": False,
            "generation_executed": False,
            "router_trained_for_deployment": False,
        },
        "dataset": {
            "unit": args.unit,
            "protocol": args.protocol,
            "total_queries": len(questions),
            "exclusive_queries": len(labeled),
            "class_counts": dict(label_counts),
            "margin": args.margin,
            "margin_rows": int(np.sum(margin_mask)),
        },
        "encoder": {
            "model_name": embedder.model_name,
            "revision": embedder.resolved_revision,
            "family": embedder.encoder_family,
            "pooling": embedder.pooling,
            "raw_normalized": embedder.normalize,
            "analysis_l2_copy": True,
            "max_sequence_length": embedder.max_sequence_length,
            "dimension": embedder.dimension,
            "device": embedder.device,
            "encode_seconds": encode_seconds,
        },
        "surface": {
            "feature_names": surface_names,
            "idf_terms_requested": len(set(all_terms)),
            "idf_terms_found": len(idf_by_term),
            "sparse_index_id": sparse_manifest["sparse_index_id"],
        },
        "class_summary": class_summary(
            labels, raw_embeddings, surface, surface_names
        ),
        "geometry": {
            "balance_repeats": BALANCE_REPEATS,
            "rows_per_class": int(min(label_counts.values())),
            "k_values": list(K_VALUES),
            "aggregate": geometry_aggregate,
            "projection": projection,
        },
        "probes": probes,
        "decision": decision,
    }
    json_dump(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "nq_bm25_bge_dpr_v1",
        "seed": SEED,
        "inputs": {
            str(path.as_posix()): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [*input_paths, sparse_manifest_path]
        },
        "sparse_database": {
            "path": database.as_posix(),
            "size_bytes": database.stat().st_size,
            "manifest_bound_sha256": sparse_manifest["artifacts"]["database"]["sha256"],
        },
        "label_rule": {
            "bm25_only": "bm25_ndcg_at_10 > 0 and bge_ndcg_at_10 == 0",
            "bge_only": "bge_ndcg_at_10 > 0 and bm25_ndcg_at_10 == 0",
            "both_success_and_both_failure": "excluded",
        },
        "analysis": {
            "balance_repeats": BALANCE_REPEATS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "permutation_repeats": PERMUTATION_REPEATS,
            "nested_grouped_outer_folds": 5,
            "nested_grouped_inner_folds": 4,
        },
        "artifacts": {
            str(path.relative_to(args.output_dir).as_posix()): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    json_dump(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "decision": decision["label"],
                "dpr_oof_auc": dpr_probe["oof"]["roc_auc"],
                "knn_purity_at_20": geometry_aggregate[
                    "same_label_purity_at_20"
                ]["median"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
