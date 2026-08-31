"""Execute the frozen zero-call HotpotQA B/D Router Phase 2.7 model audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / (
    "analysis/hotpotqa_router/phases/phase27/config_screen_4800.yaml"
)
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class RouterData:
    query_ids: np.ndarray
    group_ids: np.ndarray
    query_ranks: np.ndarray
    bm25: np.ndarray
    dense: np.ndarray
    gap: np.ndarray
    strata: np.ndarray
    repeat_values: np.ndarray
    lexical: np.ndarray
    corpus: np.ndarray
    embedding: np.ndarray

    @property
    def structured(self) -> np.ndarray:
        return np.column_stack([self.lexical, self.corpus])


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_kind: str
    model_kind: str
    loss: str = "mse"
    pca_dim: int | None = None
    xgb_preset: str = "regularized"


@dataclass(frozen=True)
class AffineCalibration:
    intercept: float
    slope: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(values, dtype=np.float64)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)),
        help="Frozen Phase 2.7 YAML configuration.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "screen",
            "stage2-report",
            "stage3-targets",
            "rotation",
            "stage4-report",
            "stage5-calibration",
            "stage6-evaluation",
            "freeze-candidates",
            "formal",
            "learning-curve",
            "decision",
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=None,
        help="Testing override; the frozen default is read from the config.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 2.7 config must be a mapping")
    if value.get("protocol", {}).get("external_calls_allowed") != 0:
        raise ValueError("Phase 2.7 must remain a zero-external-call audit")
    if value.get("protocol", {}).get("status") not in {
        "frozen_for_zero_call_implementation",
        "zero_call_execution_active",
    }:
        raise ValueError("Phase 2.7 config is not frozen or active for zero-call execution")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, contract in config["frozen_inputs"].items():
        path = _resolve(contract["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        observed = _sha256(path)
        expected = str(contract["sha256"])
        if observed != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {name}: expected {expected}, got {observed}"
            )
        verified[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": observed,
        }
    return verified


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _load_query_summary(
    path: Path,
    *,
    expected_rows: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "query_id",
            "group_id",
            "query_rank",
            "bm25_mean_f1",
            "dense_mean_f1",
            "f1_gap_bm25_minus_dense",
            "f1_winner",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Query summary is missing required Phase 2.7 fields")
        for row in reader:
            selected.append({field: str(row[field]) for field in required})
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} query-summary rows, found {len(selected)}"
        )

    query_ids = np.asarray([row["query_id"] for row in selected], dtype=np.str_)
    group_ids = np.asarray([row["group_id"] for row in selected], dtype=np.str_)
    ranks = np.asarray([int(row["query_rank"]) for row in selected], dtype=np.int64)
    bm25 = np.asarray(
        [_finite_float(row["bm25_mean_f1"], field="bm25_mean_f1") for row in selected]
    )
    dense = np.asarray(
        [_finite_float(row["dense_mean_f1"], field="dense_mean_f1") for row in selected]
    )
    recorded_gap = np.asarray(
        [
            _finite_float(row["f1_gap_bm25_minus_dense"], field="f1_gap")
            for row in selected
        ]
    )
    if len(set(query_ids.tolist())) != expected_rows:
        raise ValueError("Query IDs must be unique")
    if not np.array_equal(ranks, np.arange(expected_rows, dtype=np.int64)):
        raise ValueError(
            f"Query ranks must be the frozen contiguous 0..{expected_rows - 1} sequence"
        )
    if not np.allclose(bm25 - dense, recorded_gap, atol=1e-12, rtol=0.0):
        raise ValueError("Recorded F1 gap differs from BM25 minus Dense")
    if np.any((bm25 < 0.0) | (bm25 > 1.0) | (dense < 0.0) | (dense > 1.0)):
        raise ValueError("Mean F1 values must lie in [0, 1]")
    strata = np.where(
        recorded_gap > TIE_ATOL,
        "bm25_winner",
        np.where(recorded_gap < -TIE_ATOL, "dense_winner", "exact_tie"),
    )
    recorded_winners = np.asarray([row["f1_winner"] for row in selected], dtype=np.str_)
    expected_winners = np.where(
        strata == "bm25_winner",
        "bm25",
        np.where(strata == "dense_winner", "dense", "tie"),
    )
    if not np.array_equal(recorded_winners, expected_winners):
        raise ValueError("Recorded F1 winner differs from the frozen gap")
    return (
        {
            "query_ids": query_ids,
            "group_ids": group_ids,
            "query_ranks": ranks,
            "bm25": bm25,
            "dense": dense,
            "gap": recorded_gap,
            "strata": np.asarray(strata, dtype=np.str_),
        },
        selected,
    )


def _load_repeat_values(
    path: Path,
    *,
    query_ids: np.ndarray,
    group_ids: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    query_position = {str(query_id): index for index, query_id in enumerate(query_ids)}
    values = np.full((len(query_ids), 2, 3), np.nan, dtype=np.float64)
    actions = {"bm25": 0, "dense": 1}
    seen: set[tuple[str, str, int]] = set()
    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row_count += 1
            row = json.loads(line)
            query_id = str(row.get("query_id"))
            action = str(row.get("action"))
            repeat_id = int(row.get("repeat_id", -1))
            cell = (query_id, action, repeat_id)
            if cell in seen:
                raise ValueError(f"Duplicate B/D repeat cell: {cell}")
            seen.add(cell)
            if query_id not in query_position:
                raise ValueError(f"Outcome query is outside the frozen summary: {query_id}")
            if action not in actions or repeat_id not in (0, 1, 2):
                raise ValueError(f"Unexpected action/repeat cell: {cell}")
            if row.get("split") != "train":
                raise ValueError("Frozen Phase 2.7 outcomes must be train-only")
            if row.get("status") != "success":
                raise ValueError(f"Incomplete outcome cell: {cell}")
            position = query_position[query_id]
            if str(row.get("group_id")) != str(group_ids[position]):
                raise ValueError(f"Outcome group mismatch for {query_id}")
            if int(row.get("query_rank", -1)) != position:
                raise ValueError(f"Outcome query-rank mismatch for {query_id}")
            f1 = _finite_float(row.get("normalized_token_f1"), field="normalized_token_f1")
            if not 0.0 <= f1 <= 1.0:
                raise ValueError("Repeat F1 must lie in [0, 1]")
            values[position, actions[action], repeat_id] = f1
    if row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} outcome rows, found {row_count}")
    if not np.isfinite(values).all():
        raise ValueError("Frozen repeat tensor has missing or non-finite cells")
    return values


def load_frozen_data(
    config: Mapping[str, Any],
) -> tuple[RouterData, dict[str, Any]]:
    verified = verify_frozen_inputs(config)
    query_contract = config["frozen_inputs"]["query_summary"]
    outcome_contract = config["frozen_inputs"]["outcomes"]
    feature_contract = config["frozen_inputs"]["features"]
    summary, _ = _load_query_summary(
        _resolve(query_contract["path"]),
        expected_rows=int(query_contract["expected_rows"]),
    )
    repeat_values = _load_repeat_values(
        _resolve(outcome_contract["path"]),
        query_ids=summary["query_ids"],
        group_ids=summary["group_ids"],
        expected_rows=int(outcome_contract["expected_rows"]),
    )
    repeat_means = np.mean(repeat_values, axis=2)
    if not np.allclose(repeat_means[:, 0], summary["bm25"], atol=1e-12, rtol=0.0):
        raise ValueError("BM25 repeat means differ from the query summary")
    if not np.allclose(repeat_means[:, 1], summary["dense"], atol=1e-12, rtol=0.0):
        raise ValueError("Dense repeat means differ from the query summary")

    feature_path = _resolve(feature_contract["path"])
    with np.load(feature_path, allow_pickle=False) as stored:
        expected_keys = {"query_ids", "group_ids", "lexical", "dense", "embedding"}
        if set(stored.files) != expected_keys:
            raise ValueError(f"Unexpected feature arrays: {stored.files}")
        feature_query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        feature_group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        lexical = np.asarray(stored["lexical"], dtype=np.float64)
        corpus = np.asarray(stored["dense"], dtype=np.float64)
        embedding = np.asarray(stored["embedding"], dtype=np.float64)
    expected_queries = int(config["scope"]["query_count"])
    if feature_query_ids.shape != (expected_queries,) or not np.array_equal(
        feature_query_ids, summary["query_ids"]
    ):
        raise ValueError("Feature/query order mismatch")
    if feature_group_ids.shape != (expected_queries,) or not np.array_equal(
        feature_group_ids, summary["group_ids"]
    ):
        raise ValueError("Feature/group order mismatch")
    expected_shapes = {
        "lexical": (expected_queries, 17),
        "corpus": (expected_queries, 13),
        "embedding": (expected_queries, 384),
    }
    observed_shapes = {
        "lexical": lexical.shape,
        "corpus": corpus.shape,
        "embedding": embedding.shape,
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            f"Frozen feature dimensions differ: expected {expected_shapes}, got {observed_shapes}"
        )
    if not all(np.isfinite(value).all() for value in (lexical, corpus, embedding)):
        raise ValueError("Frozen model features must be finite")

    data = RouterData(
        query_ids=summary["query_ids"],
        group_ids=summary["group_ids"],
        query_ranks=summary["query_ranks"],
        bm25=summary["bm25"],
        dense=summary["dense"],
        gap=summary["gap"],
        strata=summary["strata"],
        repeat_values=repeat_values,
        lexical=lexical,
        corpus=corpus,
        embedding=embedding,
    )
    counts = {name: int(np.sum(data.strata == name)) for name in sorted(set(data.strata))}
    validation = {
        "status": "passed",
        "protocol_id": config["protocol"]["id"],
        "external_calls": 0,
        "frozen_inputs": verified,
        "queries_read": int(len(data.query_ids)),
        "outcome_rows_read": int(np.prod(repeat_values.shape)),
        "actions": ["bm25", "dense"],
        "repeat_ids": [0, 1, 2],
        "partition": "train_only",
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "partial_9600_rows_read": 0,
        "complete_9600_queries_read": int(len(data.query_ids)) if len(data.query_ids) == 9600 else 0,
        "feature_dimensions": {
            "lexical": int(lexical.shape[1]),
            "corpus_static": int(corpus.shape[1]),
            "embedding": int(embedding.shape[1]),
            "structured": int(data.structured.shape[1]),
            "raw_full": int(data.structured.shape[1] + embedding.shape[1]),
        },
        "strata_counts": counts,
        "fixed_bm25_mean_f1": float(np.mean(data.bm25)),
        "fixed_dense_mean_f1": float(np.mean(data.dense)),
        "bd_oracle_mean_f1": float(np.mean(np.maximum(data.bm25, data.dense))),
        "model_input_arrays": ["lexical", "corpus_static", "query_embedding"],
        "forbidden_online_fields_loaded_as_model_features": [],
    }
    return data, validation


def make_group_stratified_folds(
    data: RouterData,
    *,
    n_splits: int,
    seed: int,
    indices: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    selected = (
        np.arange(len(data.query_ids), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    local = np.arange(len(selected), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, validation_local in splitter.split(
        local,
        y=data.strata[selected],
        groups=data.group_ids[selected],
    ):
        train = selected[np.asarray(train_local, dtype=np.int64)]
        validation = selected[np.asarray(validation_local, dtype=np.int64)]
        if set(data.group_ids[train]) & set(data.group_ids[validation]):
            raise RuntimeError("A group crossed a fold boundary")
        folds.append((train, validation))
    seen = np.concatenate([validation for _, validation in folds])
    if sorted(seen.tolist()) != sorted(selected.tolist()):
        raise RuntimeError("Validation folds must cover every selected query exactly once")
    return folds


def _split_manifest(
    data: RouterData,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for split_seed in config["cross_validation"]["split_seeds"]:
        folds = make_group_stratified_folds(
            data,
            n_splits=int(config["cross_validation"]["outer_folds"]),
            seed=int(split_seed),
        )
        fold_rows: list[dict[str, Any]] = []
        for fold_id, (train, validation) in enumerate(folds):
            fold_rows.append(
                {
                    "outer_fold": fold_id,
                    "train_queries": int(len(train)),
                    "validation_queries": int(len(validation)),
                    "train_groups": int(len(set(data.group_ids[train]))),
                    "validation_groups": int(len(set(data.group_ids[validation]))),
                    "group_overlap": 0,
                    "validation_strata": {
                        name: int(np.sum(data.strata[validation] == name))
                        for name in ("bm25_winner", "exact_tie", "dense_winner")
                    },
                    "validation_query_ranks": data.query_ranks[validation].tolist(),
                }
            )
        manifests.append({"split_seed": int(split_seed), "folds": fold_rows})
    return {
        "status": "passed",
        "protocol_id": config["protocol"]["id"],
        "split_kind": config["cross_validation"]["split_kind"],
        "group_field": config["cross_validation"]["group_field"],
        "strata": config["cross_validation"]["strata"],
        "external_calls": 0,
        "splits": manifests,
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-c", "safe.directory=C:/Users/12442/Desktop/RAG_pipeine/RAG_pipeline", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _environment() -> dict[str, Any]:
    import scipy
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_preflight(
    config: Mapping[str, Any],
    data: RouterData,
    validation: Mapping[str, Any],
    output_dir: Path,
) -> None:
    result = {
        **validation,
        "config_status": config["protocol"]["status"],
        "evidence_code_head": config["protocol"]["evidence_code_head"],
        "current_git": _git_state(),
        "environment": _environment(),
        "completed_at_unix": time.time(),
    }
    manifest = _split_manifest(data, config)
    _write_json(output_dir / "preflight.json", result)
    _write_json(output_dir / "split_manifest.json", manifest)
    _write_execution_state(config, output_dir)
    print(json.dumps({"preflight": "passed", "output_dir": str(output_dir)}, indent=2))


def _fit_affine_calibration(scores: np.ndarray, gaps: np.ndarray) -> AffineCalibration:
    x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(gaps, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("Calibration inputs must be aligned, non-empty, and finite")
    model = LinearRegression(positive=True)
    model.fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if slope < 0.0 or not math.isfinite(slope + intercept):
        raise RuntimeError("Affine calibration must be finite and non-negative-slope")
    return AffineCalibration(intercept=intercept, slope=slope)


def _apply_rotation(embedding: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(embedding.shape[1], embedding.shape[1]))
    orthogonal, _ = np.linalg.qr(matrix)
    signs = np.sign(np.diag(orthogonal))
    signs[signs == 0.0] = 1.0
    orthogonal = orthogonal * signs
    rotated = np.asarray(embedding @ orthogonal, dtype=np.float64)
    if not np.allclose(
        embedding[:64] @ embedding[:64].T,
        rotated[:64] @ rotated[:64].T,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise RuntimeError("Orthogonal rotation did not preserve inner products")
    return rotated


def _transform_features(
    data: RouterData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    spec: CandidateSpec,
    *,
    transform_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    structured_scaler = StandardScaler()
    structured_train = structured_scaler.fit_transform(data.structured[train_indices])
    structured_validation = structured_scaler.transform(data.structured[validation_indices])
    embedding_mean = np.mean(data.embedding[train_indices], axis=0, keepdims=True)
    embedding_train = data.embedding[train_indices] - embedding_mean
    embedding_validation = data.embedding[validation_indices] - embedding_mean
    metadata: dict[str, Any] = {
        "structured_scaler_fit_queries": int(len(train_indices)),
        "embedding_center_fit_queries": int(len(train_indices)),
        "pca_fit_queries": 0,
    }
    if spec.feature_kind == "structured":
        return structured_train, structured_validation, metadata
    if spec.feature_kind == "embedding":
        return embedding_train, embedding_validation, metadata
    if spec.feature_kind == "raw_full":
        return (
            np.column_stack([structured_train, embedding_train]),
            np.column_stack([structured_validation, embedding_validation]),
            metadata,
        )
    if spec.feature_kind == "pca_structured":
        if spec.pca_dim is None:
            raise ValueError("PCA candidate requires pca_dim")
        pca = PCA(
            n_components=spec.pca_dim,
            whiten=False,
            svd_solver="randomized",
            random_state=transform_seed,
        )
        projected_train = pca.fit_transform(embedding_train)
        projected_validation = pca.transform(embedding_validation)
        metadata.update(
            {
                "pca_fit_queries": int(len(train_indices)),
                "pca_dimensions": int(spec.pca_dim),
                "pca_explained_variance_ratio_sum": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
            }
        )
        return (
            np.column_stack([structured_train, projected_train]),
            np.column_stack([structured_validation, projected_validation]),
            metadata,
        )
    raise ValueError(f"Unknown feature kind: {spec.feature_kind}")


def _xgb_parameters(
    config: Mapping[str, Any],
    spec: CandidateSpec,
    *,
    seed: int,
    n_estimators: int,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    search = config["search_spaces"]["xgboost"]
    parameters = {
        **search["presets"][spec.xgb_preset],
        "objective": search["objective_by_loss"][spec.loss],
        "eval_metric": search["eval_metric"],
        "n_estimators": int(n_estimators),
        "tree_method": search["tree_method"],
        "n_jobs": int(search["n_jobs"]),
        "random_state": int(seed),
        "verbosity": 0,
    }
    if early_stopping_rounds is not None:
        parameters["early_stopping_rounds"] = int(early_stopping_rounds)
    return parameters


def _early_stop_local_split(
    data: RouterData,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        folds = make_group_stratified_folds(
            data,
            n_splits=5,
            seed=seed,
            indices=np.asarray(train_indices, dtype=np.int64),
        )
    except ValueError:
        return None
    fit_global, stop_global = folds[0]
    position = {int(value): index for index, value in enumerate(train_indices)}
    return (
        np.asarray([position[int(value)] for value in fit_global], dtype=np.int64),
        np.asarray([position[int(value)] for value in stop_global], dtype=np.int64),
    )


class DualUtilityHuber:
    """Small deterministic linear two-head model with a Huber gap penalty."""

    def __init__(
        self,
        *,
        gap_lambda: float = 0.5,
        delta: float = 0.1,
        learning_rate: float = 0.01,
        alpha: float = 0.001,
        epochs: int = 1200,
    ) -> None:
        self.gap_lambda = gap_lambda
        self.delta = delta
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.weights: np.ndarray | None = None

    def _gradient(self, residual: np.ndarray) -> np.ndarray:
        return np.clip(residual / self.delta, -1.0, 1.0)

    def fit(self, matrix: np.ndarray, utilities: np.ndarray) -> "DualUtilityHuber":
        x = np.column_stack([np.ones(len(matrix)), np.asarray(matrix, dtype=np.float64)])
        y = np.asarray(utilities, dtype=np.float64)
        if y.shape != (len(x), 2):
            raise ValueError("Dual utility target must have shape (queries, 2)")
        weights = np.zeros((x.shape[1], 2), dtype=np.float64)
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        beta1, beta2 = 0.9, 0.999
        for epoch in range(1, self.epochs + 1):
            predicted = x @ weights
            gap_residual = (predicted[:, 0] - predicted[:, 1]) - (y[:, 0] - y[:, 1])
            derivative = np.column_stack(
                [
                    self._gradient(predicted[:, 0] - y[:, 0])
                    + self.gap_lambda * self._gradient(gap_residual),
                    self._gradient(predicted[:, 1] - y[:, 1])
                    - self.gap_lambda * self._gradient(gap_residual),
                ]
            )
            gradient = x.T @ derivative / len(x)
            gradient[1:] += self.alpha * weights[1:]
            first = beta1 * first + (1.0 - beta1) * gradient
            second = beta2 * second + (1.0 - beta2) * np.square(gradient)
            first_hat = first / (1.0 - beta1**epoch)
            second_hat = second / (1.0 - beta2**epoch)
            weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        self.weights = weights
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("DualUtilityHuber has not been fit")
        x = np.column_stack([np.ones(len(matrix)), np.asarray(matrix, dtype=np.float64)])
        return x @ self.weights


def _fit_predict_candidate(
    config: Mapping[str, Any],
    data: RouterData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    spec: CandidateSpec,
    *,
    model_seed: int,
    transform_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_matrix, validation_matrix, transform = _transform_features(
        data,
        train_indices,
        validation_indices,
        spec,
        transform_seed=transform_seed,
    )
    target = data.gap[train_indices]
    metadata: dict[str, Any] = {"transform": transform}
    if spec.model_kind == "ridge":
        model = Ridge(alpha=10.0)
        model.fit(train_matrix, target)
        prediction = model.predict(validation_matrix)
    elif spec.model_kind == "elastic_net":
        model = ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000, random_state=model_seed)
        model.fit(train_matrix, target)
        prediction = model.predict(validation_matrix)
    elif spec.model_kind == "huber":
        model = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=2000)
        model.fit(train_matrix, target)
        prediction = model.predict(validation_matrix)
    elif spec.model_kind == "dual_huber":
        model = DualUtilityHuber(gap_lambda=0.5)
        utilities = np.column_stack([data.bm25[train_indices], data.dense[train_indices]])
        model.fit(train_matrix, utilities)
        utility_prediction = model.predict(validation_matrix)
        prediction = utility_prediction[:, 0] - utility_prediction[:, 1]
    elif spec.model_kind == "xgb_sign":
        keep_local = np.flatnonzero(np.abs(target) > TIE_ATOL)
        labels = (target[keep_local] > 0.0).astype(np.int64)
        weights = np.abs(target[keep_local])
        if len(np.unique(labels)) != 2:
            raise ValueError("Weighted-sign training fold must contain both winner classes")
        search = config["search_spaces"]["xgboost"]
        maximum = int(search["max_estimators"])
        early_rounds = int(search["early_stopping_rounds"])
        common = {
            **search["presets"][spec.xgb_preset],
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": search["tree_method"],
            "n_jobs": int(search["n_jobs"]),
            "random_state": int(model_seed),
            "verbosity": 0,
        }
        winner_indices = np.asarray(train_indices, dtype=np.int64)[keep_local]
        stop_split = _early_stop_local_split(
            data,
            winner_indices,
            seed=transform_seed + model_seed,
        )
        selected_trees = min(300, maximum)
        if stop_split is not None:
            fit_winner_local, stop_winner_local = stop_split
            selector = XGBClassifier(
                **common,
                n_estimators=maximum,
                early_stopping_rounds=early_rounds,
            )
            selector.fit(
                train_matrix[keep_local[fit_winner_local]],
                labels[fit_winner_local],
                sample_weight=weights[fit_winner_local],
                eval_set=[
                    (
                        train_matrix[keep_local[stop_winner_local]],
                        labels[stop_winner_local],
                    )
                ],
                sample_weight_eval_set=[weights[stop_winner_local]],
                verbose=False,
            )
            selected_trees = int(getattr(selector, "best_iteration", selected_trees - 1)) + 1
        selected_trees = max(1, min(selected_trees, maximum))
        model = XGBClassifier(**common, n_estimators=selected_trees)
        model.fit(
            train_matrix[keep_local],
            labels,
            sample_weight=weights,
            verbose=False,
        )
        prediction = model.predict_proba(validation_matrix)[:, 1] - 0.5
        metadata.update(
            {
                "selected_trees": selected_trees,
                "training_non_ties": int(len(keep_local)),
                "training_exact_ties_skipped": int(len(target) - len(keep_local)),
            }
        )
    elif spec.model_kind == "xgboost":
        search = config["search_spaces"]["xgboost"]
        maximum = int(search["max_estimators"])
        early_rounds = int(search["early_stopping_rounds"])
        stop_split = _early_stop_local_split(
            data,
            np.asarray(train_indices, dtype=np.int64),
            seed=transform_seed + model_seed,
        )
        selected_trees = min(300, maximum)
        if stop_split is not None:
            fit_local, stop_local = stop_split
            selector = XGBRegressor(
                **_xgb_parameters(
                    config,
                    spec,
                    seed=model_seed,
                    n_estimators=maximum,
                    early_stopping_rounds=early_rounds,
                )
            )
            selector.fit(
                train_matrix[fit_local],
                target[fit_local],
                eval_set=[(train_matrix[stop_local], target[stop_local])],
                verbose=False,
            )
            selected_trees = int(getattr(selector, "best_iteration", selected_trees - 1)) + 1
        selected_trees = max(1, min(selected_trees, maximum))
        model = XGBRegressor(
            **_xgb_parameters(
                config,
                spec,
                seed=model_seed,
                n_estimators=selected_trees,
                early_stopping_rounds=None,
            )
        )
        model.fit(train_matrix, target, verbose=False)
        prediction = model.predict(validation_matrix)
        metadata["selected_trees"] = selected_trees
    else:
        raise ValueError(f"Unsupported model kind: {spec.model_kind}")
    result = np.asarray(prediction, dtype=np.float64)
    if result.shape != (len(validation_indices),) or not np.isfinite(result).all():
        raise RuntimeError(f"{spec.candidate_id} produced invalid predictions")
    return result, metadata


def _base_specs_for_late_fusion(
    *, use_pca_base: bool = False
) -> tuple[CandidateSpec, CandidateSpec]:
    embedding_base = (
        CandidateSpec(
            "M3_pca32_structured_ridge",
            "M3",
            "pca_structured",
            "ridge",
            pca_dim=32,
        )
        if use_pca_base
        else CandidateSpec("M1a_embedding_ridge", "M1", "embedding", "ridge")
    )
    return (
        CandidateSpec(
            "M0_structured_xgb_robust",
            "M0",
            "structured",
            "xgboost",
            loss="robust",
            xgb_preset="regularized",
        ),
        embedding_base,
    )


def _fit_predict_late_fusion(
    config: Mapping[str, Any],
    data: RouterData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    split_seed: int,
    outer_fold_id: int,
    model_seed: int,
    use_pca_base: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bases = _base_specs_for_late_fusion(use_pca_base=use_pca_base)
    inner_folds = make_group_stratified_folds(
        data,
        n_splits=int(config["cross_validation"]["inner_folds"]),
        seed=split_seed + 100 + outer_fold_id,
        indices=train_indices,
    )
    base_oof = np.full((len(train_indices), 2), np.nan, dtype=np.float64)
    local_position = {int(value): index for index, value in enumerate(train_indices)}
    for inner_train, inner_validation in inner_folds:
        validation_local = np.asarray(
            [local_position[int(value)] for value in inner_validation], dtype=np.int64
        )
        for base_id, base in enumerate(bases):
            predicted, _ = _fit_predict_candidate(
                config,
                data,
                inner_train,
                inner_validation,
                base,
                model_seed=model_seed,
                transform_seed=split_seed + 1000 + outer_fold_id * 10 + base_id,
            )
            base_oof[validation_local, base_id] = predicted
    if not np.isfinite(base_oof).all():
        raise RuntimeError("Late-fusion base OOF predictions are incomplete")
    meta_oof = np.full(len(train_indices), np.nan, dtype=np.float64)
    for meta_train, meta_validation in inner_folds:
        meta_train_local = np.asarray(
            [local_position[int(value)] for value in meta_train], dtype=np.int64
        )
        meta_validation_local = np.asarray(
            [local_position[int(value)] for value in meta_validation], dtype=np.int64
        )
        cross_fitted_meta = Ridge(alpha=1.0)
        cross_fitted_meta.fit(
            base_oof[meta_train_local], data.gap[meta_train]
        )
        meta_oof[meta_validation_local] = cross_fitted_meta.predict(
            base_oof[meta_validation_local]
        )
    if not np.isfinite(meta_oof).all():
        raise RuntimeError("Late-fusion meta OOF predictions are incomplete")
    meta = Ridge(alpha=1.0)
    meta.fit(base_oof, data.gap[train_indices])
    base_validation = np.column_stack(
        [
            _fit_predict_candidate(
                config,
                data,
                train_indices,
                validation_indices,
                base,
                model_seed=model_seed,
                transform_seed=split_seed + 2000 + outer_fold_id * 10 + base_id,
            )[0]
            for base_id, base in enumerate(bases)
        ]
    )
    return (
        np.asarray(meta.predict(base_validation), dtype=np.float64),
        meta_oof,
        {
            "base_candidate_ids": [base.candidate_id for base in bases],
            "meta_coefficients": meta.coef_.tolist(),
            "meta_intercept": float(meta.intercept_),
        },
    )


def _bootstrap_gain_ci(
    paired_gains: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    ordered = sorted(by_group)
    sums = np.asarray([np.sum(paired_gains[by_group[group]]) for group in ordered])
    counts = np.asarray([len(by_group[group]) for group in ordered], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        sampled = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        estimates[start:stop] = np.sum(sums[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _calibration_summary(scores: np.ndarray, gaps: np.ndarray) -> dict[str, Any]:
    diagnostic = LinearRegression().fit(scores.reshape(-1, 1), gaps)
    order = np.argsort(scores, kind="stable")
    bins: list[dict[str, Any]] = []
    populated_bins = [indices for indices in np.array_split(order, min(10, len(order))) if len(indices)]
    for bin_id, indices in enumerate(populated_bins):
        bins.append(
            {
                "decile": bin_id + 1,
                "queries": int(len(indices)),
                "predicted_gap_mean": float(np.mean(scores[indices])),
                "realized_gap_mean": float(np.mean(gaps[indices])),
            }
        )
    return {
        "intercept": float(diagnostic.intercept_),
        "slope": float(diagnostic.coef_[0]),
        "deciles": bins,
        "top_decile_realized_gap": bins[-1]["realized_gap_mean"],
    }


def policy_metrics(
    data: RouterData,
    calibrated_scores: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    with_ci: bool = True,
) -> dict[str, Any]:
    scores = np.asarray(calibrated_scores, dtype=np.float64)
    if scores.shape != data.gap.shape or not np.isfinite(scores).all():
        raise ValueError("Policy scores must align with the frozen query set")
    switches = scores > 0.0
    paired = np.where(switches, data.gap, 0.0)
    beneficial = switches & (data.gap > TIE_ATOL)
    neutral = switches & (np.abs(data.gap) <= TIE_ATOL)
    harmful = switches & (data.gap < -TIE_ATOL)
    missed = (~switches) & (data.gap > TIE_ATOL)
    beneficial_mass = float(np.mean(np.where(beneficial, data.gap, 0.0)))
    harmful_mass = float(np.mean(np.where(harmful, -data.gap, 0.0)))
    fixed_dense = float(np.mean(data.dense))
    oracle = float(np.mean(np.maximum(data.bm25, data.dense)))
    gain = float(np.mean(paired))
    non_tie = np.abs(data.gap) > TIE_ATOL
    auc = (
        float(roc_auc_score(data.gap[non_tie] > 0.0, scores[non_tie]))
        if len(np.unique(data.gap[non_tie] > 0.0)) == 2
        else None
    )
    correlation = (
        float("nan")
        if np.ptp(scores) <= 1e-15
        else float(spearmanr(scores, data.gap).statistic)
    )
    result: dict[str, Any] = {
        "router_mean_f1": fixed_dense + gain,
        "fixed_bm25_mean_f1": float(np.mean(data.bm25)),
        "fixed_dense_mean_f1": fixed_dense,
        "oracle_mean_f1": oracle,
        "oracle_headroom": oracle - fixed_dense,
        "gain_over_fixed_dense": gain,
        "policy_regret": oracle - (fixed_dense + gain),
        "oracle_recovery": gain / (oracle - fixed_dense) if oracle > fixed_dense else None,
        "switch_queries": int(np.sum(switches)),
        "switch_coverage": float(np.mean(switches)),
        "beneficial_switches": int(np.sum(beneficial)),
        "neutral_switches": int(np.sum(neutral)),
        "harmful_switches": int(np.sum(harmful)),
        "missed_bm25_beneficial_switches": int(np.sum(missed)),
        "beneficial_gain_mass": beneficial_mass,
        "harmful_loss_mass": harmful_mass,
        "harmful_to_beneficial_mass_ratio": (
            harmful_mass / beneficial_mass if beneficial_mass > 0.0 else None
        ),
        "conditional_gain_all_switches": (
            float(np.mean(data.gap[switches])) if np.any(switches) else None
        ),
        "conditional_gain_beneficial": (
            float(np.mean(data.gap[beneficial])) if np.any(beneficial) else None
        ),
        "conditional_loss_harmful": (
            float(np.mean(data.gap[harmful])) if np.any(harmful) else None
        ),
        "high_margin_bm25_winner_recall": (
            float(np.mean(switches[data.gap >= 0.1])) if np.any(data.gap >= 0.1) else None
        ),
        "gap_mae": float(mean_absolute_error(data.gap, scores)),
        "gap_rmse": float(math.sqrt(mean_squared_error(data.gap, scores))),
        "gap_spearman": float(correlation) if math.isfinite(float(correlation)) else None,
        "non_tie_auc": auc,
        "calibration": _calibration_summary(scores, data.gap),
    }
    if with_ci:
        result["gain_ci95"] = _bootstrap_gain_ci(
            paired,
            data.group_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
    return result


def _candidate_specs() -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            "M0_structured_xgb_mse", "M0", "structured", "xgboost", "mse", xgb_preset="balanced"
        ),
        CandidateSpec(
            "M0_structured_xgb_robust",
            "M0",
            "structured",
            "xgboost",
            "robust",
            xgb_preset="regularized",
        ),
        CandidateSpec("M1a_embedding_ridge", "M1", "embedding", "ridge"),
        CandidateSpec("M1b_embedding_elastic_net", "M1", "embedding", "elastic_net"),
        CandidateSpec(
            "M2_raw_full_xgb_mse", "M2", "raw_full", "xgboost", "mse", xgb_preset="legacy"
        ),
        CandidateSpec(
            "M2_raw_full_xgb_robust",
            "M2R",
            "raw_full",
            "xgboost",
            "robust",
            xgb_preset="balanced",
        ),
    ]
    for dimension in (16, 32, 64):
        specs.extend(
            [
                CandidateSpec(
                    f"M3_pca{dimension}_structured_ridge",
                    "M3",
                    "pca_structured",
                    "ridge",
                    pca_dim=dimension,
                ),
                CandidateSpec(
                    f"M3_pca{dimension}_structured_huber",
                    "M3",
                    "pca_structured",
                    "huber",
                    loss="robust",
                    pca_dim=dimension,
                ),
                CandidateSpec(
                    f"M3_pca{dimension}_structured_xgb_robust",
                    "M3",
                    "pca_structured",
                    "xgboost",
                    loss="robust",
                    pca_dim=dimension,
                    xgb_preset="regularized",
                ),
            ]
        )
    specs.extend(
        [
            CandidateSpec("M4_oof_late_fusion", "M4", "late_fusion", "late_fusion"),
            CandidateSpec(
                "M5_dual_utility_huber",
                "M5",
                "pca_structured",
                "dual_huber",
                loss="robust",
                pca_dim=32,
            ),
        ]
    )
    return specs


def _candidate_by_id(candidate_id: str) -> CandidateSpec:
    for candidate in _candidate_specs():
        if candidate.candidate_id == candidate_id:
            return candidate
    if candidate_id == "D4_raw_embedding_xgboost":
        return CandidateSpec(
            candidate_id,
            "D4",
            "embedding",
            "xgboost",
            loss="robust",
            xgb_preset="regularized",
        )
    if candidate_id == "D3_weighted_sign_raw_full_xgboost":
        return CandidateSpec(
            candidate_id,
            "D3",
            "raw_full",
            "xgb_sign",
            loss="mse",
            xgb_preset="legacy",
        )
    if candidate_id == "M4_pca32_oof_late_fusion":
        return CandidateSpec(
            candidate_id,
            "M4PCA",
            "late_fusion_pca",
            "late_fusion_pca",
        )
    raise KeyError(candidate_id)


def run_candidate_split(
    config: Mapping[str, Any],
    data: RouterData,
    spec: CandidateSpec,
    *,
    split_seed: int,
    model_seeds: Sequence[int],
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    outer_folds = make_group_stratified_folds(
        data,
        n_splits=int(config["cross_validation"]["outer_folds"]),
        seed=split_seed,
    )
    raw_oof = np.full(len(data.query_ids), np.nan, dtype=np.float64)
    calibrated_oof = np.full(len(data.query_ids), np.nan, dtype=np.float64)
    fold_ids = np.full(len(data.query_ids), -1, dtype=np.int64)
    fold_results: list[dict[str, Any]] = []
    for outer_fold_id, (outer_train, outer_validation) in enumerate(outer_folds):
        if spec.model_kind in {"late_fusion", "late_fusion_pca"}:
            final_raw, inner_raw, fusion_metadata = _fit_predict_late_fusion(
                config,
                data,
                outer_train,
                outer_validation,
                split_seed=split_seed,
                outer_fold_id=outer_fold_id,
                model_seed=int(model_seeds[0]),
                use_pca_base=spec.model_kind == "late_fusion_pca",
            )
            calibration = _fit_affine_calibration(inner_raw, data.gap[outer_train])
            final_calibrated = calibration.predict(final_raw)
            seed_metadata: list[dict[str, Any]] = [fusion_metadata]
        else:
            inner_folds = make_group_stratified_folds(
                data,
                n_splits=int(config["cross_validation"]["inner_folds"]),
                seed=split_seed + 100 + outer_fold_id,
                indices=outer_train,
            )
            local_position = {int(value): index for index, value in enumerate(outer_train)}
            inner_by_seed: list[np.ndarray] = []
            final_by_seed: list[np.ndarray] = []
            seed_metadata = []
            for model_seed in model_seeds:
                inner_raw = np.full(len(outer_train), np.nan, dtype=np.float64)
                fit_metadata: list[dict[str, Any]] = []
                for inner_fold_id, (inner_train, inner_validation) in enumerate(inner_folds):
                    predicted, metadata = _fit_predict_candidate(
                        config,
                        data,
                        inner_train,
                        inner_validation,
                        spec,
                        model_seed=int(model_seed),
                        transform_seed=split_seed
                        + 10000
                        + outer_fold_id * 100
                        + inner_fold_id,
                    )
                    validation_local = np.asarray(
                        [local_position[int(value)] for value in inner_validation], dtype=np.int64
                    )
                    inner_raw[validation_local] = predicted
                    fit_metadata.append(metadata)
                if not np.isfinite(inner_raw).all():
                    raise RuntimeError("Inner OOF predictions are incomplete")
                final_raw_seed, metadata = _fit_predict_candidate(
                    config,
                    data,
                    outer_train,
                    outer_validation,
                    spec,
                    model_seed=int(model_seed),
                    transform_seed=split_seed + 20000 + outer_fold_id,
                )
                inner_by_seed.append(inner_raw)
                final_by_seed.append(final_raw_seed)
                seed_metadata.append(
                    {
                        "model_seed": int(model_seed),
                        "inner_fits": fit_metadata,
                        "outer_fit": metadata,
                    }
                )
            inner_mean = np.mean(np.asarray(inner_by_seed), axis=0)
            final_raw = np.mean(np.asarray(final_by_seed), axis=0)
            calibration = _fit_affine_calibration(inner_mean, data.gap[outer_train])
            final_calibrated = calibration.predict(final_raw)
        raw_oof[outer_validation] = final_raw
        calibrated_oof[outer_validation] = final_calibrated
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
        fold_metrics = policy_metrics(
            fold_data,
            final_calibrated,
            bootstrap_seed=int(config["evaluation"]["bootstrap"]["seed"]),
            bootstrap_resamples=max(100, min(bootstrap_resamples, 500)),
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
                "fit_metadata": seed_metadata,
            }
        )
        print(
            f"    fold {outer_fold_id + 1}/{len(outer_folds)} "
            f"gain={fold_metrics['gain_over_fixed_dense']:+.6f} "
            f"coverage={fold_metrics['switch_coverage']:.3f}",
            flush=True,
        )
    if not np.isfinite(raw_oof).all() or not np.isfinite(calibrated_oof).all():
        raise RuntimeError("Outer OOF predictions are incomplete")
    metrics = policy_metrics(
        data,
        calibrated_oof,
        bootstrap_seed=int(config["evaluation"]["bootstrap"]["seed"])
        + int(split_seed) % 10000,
        bootstrap_resamples=bootstrap_resamples,
        with_ci=True,
    )
    fold_gains = np.asarray(
        [row["metrics"]["gain_over_fixed_dense"] for row in fold_results], dtype=np.float64
    )
    metrics["screen_selection_score"] = float(
        np.mean(fold_gains) - np.std(fold_gains, ddof=1) / math.sqrt(len(fold_gains))
    )
    return (
        {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "split_seed": int(split_seed),
            "model_seeds": [int(seed) for seed in model_seeds],
            "metrics": metrics,
            "folds": fold_results,
            "external_calls": 0,
        },
        calibrated_oof,
        fold_ids,
    )


def _screen_selection(candidate_results: Mapping[str, Any]) -> list[str]:
    baseline = "M2_raw_full_xgb_mse"
    best_by_family: dict[str, tuple[str, float]] = {}
    for candidate_id, result in candidate_results.items():
        if candidate_id == baseline:
            continue
        family = str(result["family"])
        score = float(result["metrics"]["screen_selection_score"])
        if family not in best_by_family or score > best_by_family[family][1]:
            best_by_family[family] = (candidate_id, score)
    strongest = sorted(best_by_family.values(), key=lambda item: item[1], reverse=True)[:3]
    return [baseline, *[candidate_id for candidate_id, _ in strongest]]


def run_screen(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    output_path = output_dir / "screen.json"
    prediction_path = output_dir / "screen_predictions.npz"
    existing: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if loaded.get("protocol_id") != config["protocol"]["id"]:
            raise ValueError("Existing screen output belongs to another protocol")
        existing = dict(loaded.get("candidates", {}))
    if prediction_path.exists():
        with np.load(prediction_path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    split_seed = int(config["screen"]["split_seeds"][0])
    model_seeds = [int(value) for value in config["screen"]["xgboost_model_seeds"]]
    for position, spec in enumerate(_candidate_specs(), start=1):
        if spec.candidate_id in existing:
            print(f"screen {position}/{len(_candidate_specs())}: {spec.candidate_id} [resume]", flush=True)
            continue
        print(f"screen {position}/{len(_candidate_specs())}: {spec.candidate_id}", flush=True)
        result, predictions, fold_ids = run_candidate_split(
            config,
            data,
            spec,
            split_seed=split_seed,
            model_seeds=model_seeds,
            bootstrap_resamples=bootstrap_resamples,
        )
        existing[spec.candidate_id] = result
        arrays[f"prediction__{spec.candidate_id}"] = predictions
        arrays[f"fold_id__{spec.candidate_id}"] = fold_ids
        selected = _screen_selection(existing) if "M2_raw_full_xgb_mse" in existing else []
        payload = {
            "protocol_id": config["protocol"]["id"],
            "stage": "7.1_quick_screen",
            "status": "running",
            "split_seed": split_seed,
            "model_seeds": model_seeds,
            "publishes_research_claim": False,
            "candidates": existing,
            "provisional_formal_candidates": selected,
            "external_calls": 0,
        }
        _write_json(output_path, payload)
        np.savez_compressed(prediction_path, **arrays)
    selected = _screen_selection(existing)
    payload = {
        "protocol_id": config["protocol"]["id"],
        "stage": "7.1_quick_screen",
        "status": "complete",
        "split_seed": split_seed,
        "model_seeds": model_seeds,
        "publishes_research_claim": False,
        "candidates": existing,
        "provisional_formal_candidates": selected,
        "candidate_list_status": "provisional_until_stage_2_to_6_reports_pass",
        "external_calls": 0,
    }
    _write_json(output_path, payload)
    _write_execution_state(config, output_dir)
    print(json.dumps({"screen": "complete", "formal_candidates": selected}, indent=2))


def run_rotation(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    screen_path = output_dir / "screen.json"
    if not screen_path.exists():
        raise FileNotFoundError("Run the quick screen before the rotation diagnostic")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    m3_ids = [
        candidate_id
        for candidate_id in screen["candidates"]
        if candidate_id.startswith("M3_")
    ]
    best_m3 = max(
        m3_ids,
        key=lambda candidate_id: screen["candidates"][candidate_id]["metrics"][
            "screen_selection_score"
        ],
    )
    diagnostic_ids = ["D4_raw_embedding_xgboost", "M1a_embedding_ridge", best_m3]
    split_seed = int(config["screen"]["split_seeds"][0])
    model_seeds = [int(value) for value in config["screen"]["xgboost_model_seeds"]]
    results: dict[str, Any] = {}
    for rotation_seed in [None, *config["rotation_diagnostic"]["rotation_seeds"]]:
        label = "unrotated" if rotation_seed is None else f"rotation_{rotation_seed}"
        rotated_data = (
            data
            if rotation_seed is None
            else replace(data, embedding=_apply_rotation(data.embedding, int(rotation_seed)))
        )
        results[label] = {}
        for candidate_id in diagnostic_ids:
            print(f"rotation {label}: {candidate_id}", flush=True)
            result, _, _ = run_candidate_split(
                config,
                rotated_data,
                _candidate_by_id(candidate_id),
                split_seed=split_seed,
                model_seeds=model_seeds,
                bootstrap_resamples=bootstrap_resamples,
            )
            results[label][candidate_id] = result
    _write_json(
        output_dir / "rotation_diagnostic.json",
        {
            "protocol_id": config["protocol"]["id"],
            "stage": 4,
            "status": "complete",
            "participates_in_model_selection": False,
            "diagnostic_candidates": diagnostic_ids,
            "results": results,
            "external_calls": 0,
        },
    )
    _write_execution_state(config, output_dir)
    print(json.dumps({"rotation": "complete", "candidates": diagnostic_ids}, indent=2))


def _load_completed_screen(
    config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    path = output_dir / "screen.json"
    if not path.exists():
        raise FileNotFoundError("The shared quick-screen compute has not completed")
    screen = json.loads(path.read_text(encoding="utf-8"))
    if screen.get("protocol_id") != config["protocol"]["id"]:
        raise ValueError("Screen artifact belongs to another protocol")
    if screen.get("status") != "complete":
        raise ValueError("Screen artifact is not complete")
    if len(screen.get("candidates", {})) != len(_candidate_specs()):
        raise ValueError("Screen artifact does not contain the closed 17-candidate matrix")
    return screen


def _metrics_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "gain_over_fixed_dense",
        "gain_ci95",
        "screen_selection_score",
        "router_mean_f1",
        "switch_queries",
        "switch_coverage",
        "beneficial_switches",
        "neutral_switches",
        "harmful_switches",
        "missed_bm25_beneficial_switches",
        "beneficial_gain_mass",
        "harmful_loss_mass",
        "harmful_to_beneficial_mass_ratio",
        "policy_regret",
        "oracle_recovery",
        "gap_mae",
        "gap_rmse",
        "gap_spearman",
        "non_tie_auc",
        "calibration",
    )
    return {key: metrics.get(key) for key in keys}


def _best_candidate(
    screen: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[str, Mapping[str, Any]]:
    matches = [
        (candidate_id, value)
        for candidate_id, value in screen["candidates"].items()
        if candidate_id.startswith(prefix)
    ]
    if not matches:
        raise ValueError(f"No screen candidate matches {prefix}")
    return max(
        matches,
        key=lambda item: float(item[1]["metrics"]["screen_selection_score"]),
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_execution_state(config: Mapping[str, Any], output_dir: Path) -> None:
    screen_complete = False
    screen_path = output_dir / "screen.json"
    if screen_path.exists():
        screen_complete = json.loads(screen_path.read_text(encoding="utf-8")).get("status") == "complete"
    formal_status = "not_started"
    formal_completed: dict[str, list[str]] = {}
    formal_path = output_dir / "formal_metrics.json"
    if formal_path.exists():
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        formal_status = str(formal.get("status", "unknown"))
        formal_completed = {
            candidate_id: sorted(value)
            for candidate_id, value in formal.get("candidate_splits", {}).items()
        }
        if not (output_dir / "candidate_freeze.json").exists():
            formal_status = "paused_before_stage_2_to_6_acceptance"
    stage_artifacts = {
        0: "preflight.json",
        1: "split_manifest.json",
        2: "feature_block_report.json",
        3: "target_diagnostics.json",
        4: "stage4_report.json",
        5: "calibration.json",
        6: "router_evaluation.json",
        8: "learning_curve.csv",
        9: "decision.json",
        10: "fresh_dev_metrics.json",
        11: "answer_correctness_validation.json",
        12: "privileged_teacher_diagnostic.json",
    }
    stages: dict[str, Any] = {}
    for stage in range(13):
        artifact = stage_artifacts.get(stage)
        stages[str(stage)] = {
            "spec_status": "frozen",
            "result_status": (
                "complete" if artifact and (output_dir / artifact).exists() else "pending"
            ),
            "artifact": artifact,
        }
    decision_path = output_dir / "decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        if decision == "STOP_QUERY_ONLY_V1":
            stages["10"]["result_status"] = "not_applicable_query_only_gate_failed"
            stages["11"]["result_status"] = "not_applicable_fresh_dev_not_entered"
    stages["7"] = {
        "spec_status": "frozen",
        "shared_screen_compute": "complete" if screen_complete else "pending",
        "candidate_freeze": (
            "complete" if (output_dir / "candidate_freeze.json").exists() else "pending"
        ),
        "formal_confirmation": formal_status,
        "completed_formal_splits": formal_completed,
    }
    _write_json(
        output_dir / "execution_state.json",
        {
            "protocol_id": config["protocol"]["id"],
            "status_model": "specification_and_results_are_separate",
            "stages": stages,
            "external_calls": 0,
            "updated_at_unix": time.time(),
        },
    )


def run_stage2_report(
    config: Mapping[str, Any], data: RouterData, output_dir: Path
) -> None:
    screen = _load_completed_screen(config, output_dir)
    best = {
        family: _best_candidate(screen, prefix=prefix)
        for family, prefix in {
            "M0_structured": "M0_",
            "M1_embedding": "M1",
            "M2_raw_full": "M2_",
            "M3_pca_structured": "M3_",
            "M4_late_fusion": "M4_",
            "M5_dual_utility": "M5_",
        }.items()
    }
    baseline_id = "M2_raw_full_xgb_mse"
    baseline_gain = float(
        screen["candidates"][baseline_id]["metrics"]["gain_over_fixed_dense"]
    )
    candidate_rows = {
        candidate_id: {
            "family": value["family"],
            "metrics": _metrics_snapshot(value["metrics"]),
        }
        for candidate_id, value in screen["candidates"].items()
    }
    best_rows = {
        family: {
            "candidate_id": candidate_id,
            "gain_over_fixed_dense": float(value["metrics"]["gain_over_fixed_dense"]),
            "delta_gain_vs_M2_baseline": float(
                value["metrics"]["gain_over_fixed_dense"] - baseline_gain
            ),
        }
        for family, (candidate_id, value) in best.items()
    }
    report = {
        "protocol_id": config["protocol"]["id"],
        "stage": 2,
        "status": "complete",
        "purpose": "separate_feature_block_signal_and_model_feature_fit",
        "evidence_scope": "one_split_quick_screen_not_a_formal_claim",
        "queries": int(len(data.query_ids)),
        "candidate_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
        "best_screen_candidate_by_feature_family": best_rows,
        "controlled_questions": {
            "structured_signal": ["M0_structured_xgb_mse", "M0_structured_xgb_robust"],
            "embedding_linear_signal": [
                "M1a_embedding_ridge",
                "M1b_embedding_elastic_net",
            ],
            "raw_concatenation_reproduction": [baseline_id, "M2_raw_full_xgb_robust"],
            "pca_dimension_and_model_fit": [
                candidate_id for candidate_id in candidate_rows if candidate_id.startswith("M3_")
            ],
            "late_fusion": ["M4_oof_late_fusion"],
            "dual_utility": ["M5_dual_utility_huber"],
        },
        "online_feature_boundary": {
            "used": ["lexical_17", "corpus_static_13", "query_embedding_384"],
            "forbidden_online_fields_used": [],
        },
        "external_calls": 0,
    }
    _write_json(output_dir / "feature_block_report.json", report)
    _write_execution_state(config, output_dir)
    print(json.dumps({"stage2": "complete", "best_by_family": best_rows}, indent=2))


def _repeat_diagnostics(data: RouterData) -> dict[str, Any]:
    variances = np.var(data.repeat_values, axis=2, ddof=1)
    standard_error_gap = np.sqrt(variances[:, 0] / 3.0 + variances[:, 1] / 3.0)
    pairwise_differences = (
        data.repeat_values[:, 0, :, None] - data.repeat_values[:, 1, None, :]
    )
    soft_preference = np.mean(pairwise_differences > TIE_ATOL, axis=(1, 2))
    paired_repeat_winners = np.sign(
        data.repeat_values[:, 0, :] - data.repeat_values[:, 1, :]
    )
    mean_sign = np.sign(data.gap)
    non_tie = mean_sign != 0.0
    matching_paired = np.mean(
        paired_repeat_winners[non_tie] == mean_sign[non_tie, None], axis=1
    )
    return {
        "bm25_repeat_variance_mean": float(np.mean(variances[:, 0])),
        "dense_repeat_variance_mean": float(np.mean(variances[:, 1])),
        "zero_variance_queries": {
            "bm25": int(np.sum(variances[:, 0] <= TIE_ATOL)),
            "dense": int(np.sum(variances[:, 1] <= TIE_ATOL)),
            "both": int(np.sum(np.max(variances, axis=1) <= TIE_ATOL)),
        },
        "gap_standard_error": {
            "mean": float(np.mean(standard_error_gap)),
            "median": float(np.median(standard_error_gap)),
            "p90": float(np.quantile(standard_error_gap, 0.9)),
            "maximum": float(np.max(standard_error_gap)),
        },
        "cross_repeat_soft_preference": {
            "mean": float(np.mean(soft_preference)),
            "non_tie_direction_accuracy": float(
                np.mean((soft_preference[non_tie] > 0.5) == (data.gap[non_tie] > 0.0))
            ),
            "ambiguous_4_over_9_to_5_over_9": int(
                np.sum((soft_preference >= 4.0 / 9.0) & (soft_preference <= 5.0 / 9.0))
            ),
        },
        "paired_repeat_winner_agreement_with_mean_winner": {
            "mean": float(np.mean(matching_paired)),
            "all_three_match": int(np.sum(matching_paired == 1.0)),
            "non_tie_queries": int(np.sum(non_tie)),
        },
        "inverse_variance_weighting_used_for_training": False,
    }


def run_stage3_targets(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    screen = _load_completed_screen(config, output_dir)
    print("stage 3 diagnostic: D3_weighted_sign_raw_full_xgboost", flush=True)
    sign_result, sign_predictions, sign_fold_ids = run_candidate_split(
        config,
        data,
        _candidate_by_id("D3_weighted_sign_raw_full_xgboost"),
        split_seed=int(config["screen"]["split_seeds"][0]),
        model_seeds=[int(config["screen"]["xgboost_model_seeds"][0])],
        bootstrap_resamples=bootstrap_resamples,
    )
    np.savez_compressed(
        output_dir / "target_diagnostic_predictions.npz",
        weighted_sign_calibrated_gap=sign_predictions,
        weighted_sign_fold_id=sign_fold_ids,
    )
    comparisons = {
        "M0_structured_xgb": {
            loss: _metrics_snapshot(
                screen["candidates"][f"M0_structured_xgb_{loss}"]["metrics"]
            )
            for loss in ("mse", "robust")
        },
        "M2_raw_full_xgb": {
            loss: _metrics_snapshot(
                screen["candidates"][f"M2_raw_full_xgb_{loss}"]["metrics"]
            )
            for loss in ("mse", "robust")
        },
        "M3_pca_huber": {
            dimension: _metrics_snapshot(
                screen["candidates"][f"M3_pca{dimension}_structured_huber"]["metrics"]
            )
            for dimension in (16, 32, 64)
        },
    }
    report = {
        "protocol_id": config["protocol"]["id"],
        "stage": 3,
        "status": "complete",
        "primary_target": "continuous_bm25_minus_dense_mean_f1_gap",
        "primary_loss_comparisons": comparisons,
        "weighted_sign_diagnostic": {
            "participates_in_model_selection": False,
            "exact_ties_skipped_for_classifier_loss": True,
            "sample_weight": "absolute_mean_f1_gap",
            "result": sign_result,
        },
        "repeat_diagnostics": _repeat_diagnostics(data),
        "repeat_aware_loss_used_for_model_selection": False,
        "evidence_scope": "one_split_diagnostic_not_a_formal_claim",
        "external_calls": 0,
    }
    _write_json(output_dir / "target_diagnostics.json", report)
    _write_execution_state(config, output_dir)
    print(
        json.dumps(
            {
                "stage3": "complete",
                "weighted_sign_gain": sign_result["metrics"]["gain_over_fixed_dense"],
            },
            indent=2,
        )
    )


def run_stage4_report(config: Mapping[str, Any], output_dir: Path) -> None:
    path = output_dir / "rotation_diagnostic.json"
    if not path.exists():
        raise FileNotFoundError("Run the rotation diagnostic before accepting stage 4")
    rotation = json.loads(path.read_text(encoding="utf-8"))
    labels = list(rotation["results"])
    candidates = list(rotation["diagnostic_candidates"])
    summaries: dict[str, Any] = {}
    for candidate_id in candidates:
        gains = {
            label: float(
                rotation["results"][label][candidate_id]["metrics"]["gain_over_fixed_dense"]
            )
            for label in labels
        }
        coverages = {
            label: float(
                rotation["results"][label][candidate_id]["metrics"]["switch_coverage"]
            )
            for label in labels
        }
        unrotated = gains["unrotated"]
        summaries[candidate_id] = {
            "gains": gains,
            "coverages": coverages,
            "maximum_absolute_gain_change_from_unrotated": float(
                max(abs(value - unrotated) for label, value in gains.items() if label != "unrotated")
            ),
            "gain_range": float(max(gains.values()) - min(gains.values())),
            "coverage_range": float(max(coverages.values()) - min(coverages.values())),
        }
    ridge_change = summaries["M1a_embedding_ridge"][
        "maximum_absolute_gain_change_from_unrotated"
    ]
    tree_change = summaries["D4_raw_embedding_xgboost"][
        "maximum_absolute_gain_change_from_unrotated"
    ]
    report = {
        "protocol_id": config["protocol"]["id"],
        "stage": 4,
        "status": "complete",
        "participates_in_model_selection": False,
        "preserved_geometry": ["inner_product", "cosine", "euclidean_distance"],
        "candidate_summaries": summaries,
        "diagnostic_flags": {
            "ridge_rotation_invariant_to_numerical_tolerance": ridge_change <= 1e-12,
            "raw_embedding_xgboost_more_sensitive_than_ridge": tree_change > ridge_change + 1e-6,
        },
        "source_artifact": _artifact_record(path),
        "external_calls": 0,
    }
    _write_json(output_dir / "stage4_report.json", report)
    _write_execution_state(config, output_dir)
    print(json.dumps({"stage4": "complete", "flags": report["diagnostic_flags"]}, indent=2))


def run_stage5_calibration(config: Mapping[str, Any], output_dir: Path) -> None:
    screen = _load_completed_screen(config, output_dir)
    candidates: dict[str, Any] = {}
    for candidate_id, result in screen["candidates"].items():
        slopes = np.asarray(
            [fold["calibration"]["slope"] for fold in result["folds"]], dtype=np.float64
        )
        coverages = np.asarray(
            [fold["metrics"]["switch_coverage"] for fold in result["folds"]],
            dtype=np.float64,
        )
        candidates[candidate_id] = {
            "fold_calibration_slopes": slopes.tolist(),
            "slope_minimum": float(np.min(slopes)),
            "slope_maximum": float(np.max(slopes)),
            "slope_standard_deviation": float(np.std(slopes)),
            "fold_switch_coverages": coverages.tolist(),
            "coverage_range": float(np.max(coverages) - np.min(coverages)),
            "oof_calibration": result["metrics"]["calibration"],
        }
    formal_multiseed: dict[str, Any] | None = None
    formal_path = output_dir / "formal_metrics.json"
    if formal_path.exists():
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        m2 = formal.get("candidate_splits", {}).get("M2_raw_full_xgb_mse", {})
        if "20260901" in m2:
            formal_multiseed = {
                "candidate_id": "M2_raw_full_xgb_mse",
                "split_seed": 20260901,
                "model_seeds": m2["20260901"]["model_seeds"],
                "metrics": _metrics_snapshot(m2["20260901"]["metrics"]),
            }
    report = {
        "protocol_id": config["protocol"]["id"],
        "stage": 5,
        "status": "complete",
        "ensemble_rule": "average_continuous_scores_before_calibration_and_decision",
        "calibrator": "nonnegative_affine_fit_on_inner_oof_only",
        "primary_threshold": 0.0,
        "zero_score_action": "dense",
        "free_threshold_scan_used_for_primary_policy": False,
        "one_standard_error_threshold": "secondary_report_only",
        "isotonic_used": False,
        "screen_candidate_calibration": candidates,
        "available_five_model_seed_evidence": formal_multiseed,
        "formal_multi_seed_validation_status": (
            "partial_M2_one_split_only" if formal_multiseed else "pending"
        ),
        "external_calls": 0,
    }
    _write_json(output_dir / "calibration.json", report)
    _write_execution_state(config, output_dir)
    print(json.dumps({"stage5": "complete", "candidates": len(candidates)}, indent=2))


def run_stage6_evaluation(config: Mapping[str, Any], output_dir: Path) -> None:
    screen = _load_completed_screen(config, output_dir)
    candidates: dict[str, Any] = {}
    identity_failures: list[str] = []
    for candidate_id, result in screen["candidates"].items():
        metrics = result["metrics"]
        if not math.isclose(
            float(metrics["gain_over_fixed_dense"]),
            float(metrics["beneficial_gain_mass"]) - float(metrics["harmful_loss_mass"]),
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            identity_failures.append(candidate_id)
        candidates[candidate_id] = _metrics_snapshot(metrics)
    if identity_failures:
        raise RuntimeError(f"Policy gain decomposition failed: {identity_failures}")
    report = {
        "protocol_id": config["protocol"]["id"],
        "stage": 6,
        "status": "complete",
        "primary_metric": "answer_f1_policy_gain_over_fixed_dense",
        "candidate_metrics": candidates,
        "policy_gain_decomposition_verified": True,
        "reported_failure_mechanisms": [
            "beneficial_neutral_harmful_switches",
            "missed_bm25_beneficial_switches",
            "policy_regret",
            "oracle_recovery",
            "predicted_gap_calibration",
            "auc_spearman_coverage_and_fold_variance",
        ],
        "retrieval_metrics_used_as_answer_utility": False,
        "evidence_scope": "one_split_quick_screen_not_a_formal_claim",
        "external_calls": 0,
    }
    _write_json(output_dir / "router_evaluation.json", report)
    _write_execution_state(config, output_dir)
    print(json.dumps({"stage6": "complete", "candidates": len(candidates)}, indent=2))


def freeze_formal_candidates(config: Mapping[str, Any], output_dir: Path) -> None:
    required = [
        "feature_block_report.json",
        "target_diagnostics.json",
        "stage4_report.json",
        "calibration.json",
        "router_evaluation.json",
    ]
    evidence = {name: _artifact_record(output_dir / name) for name in required}
    screen = _load_completed_screen(config, output_dir)
    candidates = screen.get("provisional_formal_candidates") or screen.get(
        "formal_candidates_frozen_from_screen"
    )
    if not isinstance(candidates, list) or not 3 <= len(candidates) <= 4:
        raise ValueError("Formal freeze must contain M2 plus two or three new candidates")
    if candidates[0] != "M2_raw_full_xgb_mse" or len(set(candidates)) != len(candidates):
        raise ValueError("Formal freeze must start with the unique M2 reproduction baseline")
    specs = {
        candidate_id: _candidate_by_id(candidate_id).__dict__ for candidate_id in candidates
    }
    formal_path = output_dir / "formal_metrics.json"
    reusable_formal: dict[str, Any] = {}
    if formal_path.exists():
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        if formal.get("formal_candidates") != candidates:
            raise ValueError("Paused formal result does not match the candidate freeze")
        reusable_formal = {
            candidate_id: sorted(value)
            for candidate_id, value in formal.get("candidate_splits", {}).items()
        }
    payload = {
        "protocol_id": config["protocol"]["id"],
        "stage": "7.1_candidate_freeze",
        "status": "complete",
        "formal_candidates": candidates,
        "candidate_specs": specs,
        "selection_rule": "M2_baseline_plus_top_three_distinct_new_families_by_screen_lower_one_se_score",
        "screen_is_not_a_formal_claim": True,
        "stage_2_to_6_evidence": evidence,
        "reusable_completed_formal_splits": reusable_formal,
        "incomplete_split_work_reused": False,
        "external_calls": 0,
        "frozen_at_unix": time.time(),
    }
    _write_json(output_dir / "candidate_freeze.json", payload)
    _write_execution_state(config, output_dir)
    print(json.dumps({"candidate_freeze": "complete", "candidates": candidates}, indent=2))


def run_formal(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    freeze_path = output_dir / "candidate_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(
            "Stage 2–6 reports and candidate_freeze.json are required before formal confirmation"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_id") != config["protocol"]["id"]:
        raise ValueError("Candidate freeze belongs to another protocol")
    candidate_ids = list(freeze["formal_candidates"])
    split_seeds = [int(value) for value in config["cross_validation"]["split_seeds"]]
    model_seeds = [int(value) for value in config["cross_validation"]["model_seeds"]]
    output_path = output_dir / "formal_metrics.json"
    completed: dict[str, Any] = {}
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if loaded.get("formal_candidates") != candidate_ids:
            raise ValueError("Existing formal output has a different frozen candidate list")
        completed = dict(loaded.get("candidate_splits", {}))
    predictions: dict[str, np.ndarray] = {}
    for candidate_id in candidate_ids:
        spec = _candidate_by_id(candidate_id)
        completed.setdefault(candidate_id, {})
        for split_seed in split_seeds:
            key = str(split_seed)
            if key in completed[candidate_id]:
                print(f"formal {candidate_id} split={split_seed} [resume]", flush=True)
                continue
            print(f"formal {candidate_id} split={split_seed}", flush=True)
            result, scores, fold_ids = run_candidate_split(
                config,
                data,
                spec,
                split_seed=split_seed,
                model_seeds=model_seeds if spec.model_kind == "xgboost" else [model_seeds[0]],
                bootstrap_resamples=bootstrap_resamples,
            )
            completed[candidate_id][key] = result
            predictions[f"prediction__{candidate_id}__{split_seed}"] = scores
            predictions[f"fold_id__{candidate_id}__{split_seed}"] = fold_ids
            _write_json(
                output_path,
                {
                    "protocol_id": config["protocol"]["id"],
                    "status": "running",
                    "formal_candidates": candidate_ids,
                    "candidate_splits": completed,
                    "external_calls": 0,
                },
            )
            existing_path = output_dir / "formal_predictions.npz"
            if existing_path.exists():
                with np.load(existing_path, allow_pickle=False) as stored:
                    prior = {name: np.asarray(stored[name]) for name in stored.files}
                prior.update(predictions)
                predictions = prior
            np.savez_compressed(existing_path, **predictions)
    aggregate: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        split_metrics = [
            completed[candidate_id][str(seed)]["metrics"] for seed in split_seeds
        ]
        aggregate[candidate_id] = {
            "mean_gain_over_split_seeds": float(
                np.mean([row["gain_over_fixed_dense"] for row in split_metrics])
            ),
            "all_split_seed_gains_positive": all(
                row["gain_over_fixed_dense"] > 0.0 for row in split_metrics
            ),
            "split_seed_metrics": split_metrics,
        }
    _write_json(
        output_path,
        {
            "protocol_id": config["protocol"]["id"],
            "status": "complete",
            "formal_candidates": candidate_ids,
            "candidate_splits": completed,
            "aggregate": aggregate,
            "external_calls": 0,
        },
    )
    _write_execution_state(config, output_dir)
    print(json.dumps({"formal": "complete", "aggregate": aggregate}, indent=2))


def run_learning_curve(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    formal_path = output_dir / "formal_metrics.json"
    if not formal_path.exists():
        raise FileNotFoundError("Run formal confirmation before the learning curve")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("status") != "complete":
        raise ValueError("Formal confirmation must be complete before the learning curve")
    candidate_ids = formal["formal_candidates"]
    split_seeds = [int(value) for value in config["cross_validation"]["split_seeds"]]
    formal_model_seeds = [int(value) for value in config["cross_validation"]["model_seeds"]]
    query_counts = [int(value) for value in config["learning_curve"]["query_counts"]]
    progress_path = output_dir / "learning_curve_progress.json"
    progress: dict[str, Any] = {
        "protocol_id": config["protocol"]["id"],
        "status": "running",
        "formal_candidates": candidate_ids,
        "query_counts": query_counts,
        "split_seeds": split_seeds,
        "rows": [],
        "external_calls": 0,
    }
    if progress_path.exists():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("protocol_id", config["protocol"]["id"]),
            ("formal_candidates", candidate_ids),
            ("query_counts", query_counts),
            ("split_seeds", split_seeds),
        ):
            if loaded.get(key) != expected:
                raise ValueError(f"Learning-curve progress has incompatible {key}")
        progress = loaded
    rows_by_key = {
        (int(row["queries"]), str(row["candidate_id"]), int(row["split_seed"])): row
        for row in progress.get("rows", [])
    }

    def save_progress() -> None:
        progress["rows"] = sorted(
            rows_by_key.values(),
            key=lambda row: (
                int(row["queries"]),
                candidate_ids.index(str(row["candidate_id"])),
                split_seeds.index(int(row["split_seed"])),
            ),
        )
        _write_json(progress_path, progress)

    for query_count in config["learning_curve"]["query_counts"]:
        count = int(query_count)
        subset = np.flatnonzero(data.query_ranks < count).astype(np.int64)
        if len(subset) != count:
            raise ValueError(
                f"Learning-curve subset rule expected {count} rows but selected {len(subset)}"
            )
        subset_data = replace(
            data,
            query_ids=data.query_ids[subset],
            group_ids=data.group_ids[subset],
            query_ranks=data.query_ranks[subset],
            bm25=data.bm25[subset],
            dense=data.dense[subset],
            gap=data.gap[subset],
            strata=data.strata[subset],
            repeat_values=data.repeat_values[subset],
            lexical=data.lexical[subset],
            corpus=data.corpus[subset],
            embedding=data.embedding[subset],
        )
        for candidate_id in candidate_ids:
            spec = _candidate_by_id(candidate_id)
            model_seeds = (
                formal_model_seeds if spec.model_kind == "xgboost" else [formal_model_seeds[0]]
            )
            for split_seed in split_seeds:
                row_key = (count, candidate_id, split_seed)
                if row_key in rows_by_key:
                    print(
                        f"learning curve n={count}: {candidate_id} split={split_seed} [resume]",
                        flush=True,
                    )
                    continue
                if count == len(data.query_ids):
                    result = formal["candidate_splits"][candidate_id][str(split_seed)]
                    source = "formal_metrics_reuse"
                    print(
                        f"learning curve n={count}: {candidate_id} split={split_seed} [formal reuse]",
                        flush=True,
                    )
                else:
                    print(
                        f"learning curve n={count}: {candidate_id} split={split_seed}",
                        flush=True,
                    )
                    result, _, _ = run_candidate_split(
                        config,
                        subset_data,
                        spec,
                        split_seed=split_seed,
                        model_seeds=model_seeds,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                    source = "stage8_fit"
                metrics = result["metrics"]
                calibration = metrics.get("calibration") or {}
                gain_ci = metrics.get("gain_ci95") or [None, None]
                rows_by_key[row_key] = {
                    "queries": count,
                    "candidate_id": candidate_id,
                    "split_seed": split_seed,
                    "model_seeds": ";".join(str(value) for value in model_seeds),
                    "source": source,
                    "gain_over_fixed_dense": metrics.get("gain_over_fixed_dense"),
                    "gain_ci95_lower": gain_ci[0],
                    "gain_ci95_upper": gain_ci[1],
                    "switch_coverage": metrics.get("switch_coverage"),
                    "beneficial_gain_mass": metrics.get("beneficial_gain_mass"),
                    "harmful_loss_mass": metrics.get("harmful_loss_mass"),
                    "harmful_to_beneficial_mass_ratio": metrics.get(
                        "harmful_to_beneficial_mass_ratio"
                    ),
                    "gap_spearman": metrics.get("gap_spearman"),
                    "non_tie_auc": metrics.get("non_tie_auc"),
                    "calibration_slope": calibration.get("slope"),
                    "top_decile_realized_gap": calibration.get("top_decile_realized_gap"),
                }
                save_progress()
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            int(row["queries"]),
            candidate_ids.index(str(row["candidate_id"])),
            split_seeds.index(int(row["split_seed"])),
        ),
    )
    expected_rows = len(query_counts) * len(candidate_ids) * len(split_seeds)
    if len(rows) != expected_rows:
        raise ValueError(f"Learning curve has {len(rows)} rows; expected {expected_rows}")
    path = output_dir / "learning_curve.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    progress["status"] = "complete"
    progress["rows"] = rows
    _write_json(progress_path, progress)
    _write_execution_state(config, output_dir)
    print(json.dumps({"learning_curve": "complete", "rows": len(rows)}, indent=2))


def run_decision(
    config: Mapping[str, Any],
    data: RouterData,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
) -> None:
    formal_path = output_dir / "formal_metrics.json"
    predictions_path = output_dir / "formal_predictions.npz"
    curve_path = output_dir / "learning_curve.csv"
    for path in (formal_path, predictions_path, curve_path):
        if not path.exists():
            raise FileNotFoundError(f"Stage 9 requires {path.name}")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("status") != "complete":
        raise ValueError("Formal confirmation is incomplete")
    candidate_ids = list(formal["formal_candidates"])
    split_seeds = [int(value) for value in config["cross_validation"]["split_seeds"]]
    with curve_path.open("r", encoding="utf-8", newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    expected_curve_rows = (
        len(candidate_ids) * len(split_seeds) * len(config["learning_curve"]["query_counts"])
    )
    if len(curve_rows) != expected_curve_rows:
        raise ValueError(
            f"Learning curve has {len(curve_rows)} rows; expected {expected_curve_rows}"
        )

    learning_summary: dict[str, Any] = {}
    query_counts = [int(value) for value in config["learning_curve"]["query_counts"]]
    for candidate_id in candidate_ids:
        means_by_count: dict[str, float] = {}
        for count in query_counts:
            matching = [
                row
                for row in curve_rows
                if row["candidate_id"] == candidate_id and int(row["queries"]) == count
            ]
            if len(matching) != len(split_seeds):
                raise ValueError(
                    f"Learning curve is incomplete for {candidate_id} at {count} queries"
                )
            means_by_count[str(count)] = float(
                np.mean([float(row["gain_over_fixed_dense"]) for row in matching])
            )
        max_query_count = max(query_counts)
        at_max_count = [
            row
            for row in curve_rows
            if row["candidate_id"] == candidate_id and int(row["queries"]) == max_query_count
        ]
        positive_at_max_count = sum(
            float(row["gain_over_fixed_dense"]) > 0.0 for row in at_max_count
        )
        slope = float(
            np.polyfit(
                np.asarray(query_counts, dtype=np.float64),
                np.asarray([means_by_count[str(count)] for count in query_counts]),
                1,
            )[0]
        )
        eligible = (
            means_by_count[str(max_query_count)] > 0.0
            and positive_at_max_count >= 2
            and slope > 0.0
        )
        learning_summary[candidate_id] = {
            "mean_gain_by_query_count": means_by_count,
            "fitted_gain_slope_per_query": slope,
            "max_query_count": int(max_query_count),
            "positive_split_seeds_at_max_query_count": int(positive_at_max_count),
            "positive_scaling_trend": bool(eligible),
            "request_9600_eligible": bool(eligible and max_query_count < 9600),
        }

    gate = config["formal_gate"]
    formal_gate_results: dict[str, Any] = {}
    passing_candidates: list[str] = []
    with np.load(predictions_path, allow_pickle=False) as stored:
        for candidate_id in candidate_ids:
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
            consensus_metrics = policy_metrics(
                data,
                scores,
                bootstrap_seed=int(config["evaluation"]["bootstrap"]["seed"]),
                bootstrap_resamples=bootstrap_resamples,
            )
            aggregate = formal["aggregate"][candidate_id]
            ratio = consensus_metrics["harmful_to_beneficial_mass_ratio"]
            slope = consensus_metrics["calibration"]["slope"]
            top_decile = consensus_metrics["calibration"]["top_decile_realized_gap"]
            checks = {
                "practical_gain": aggregate["mean_gain_over_split_seeds"]
                >= float(gate["practical_gain_minimum"]),
                "grouped_bootstrap_ci_lower": consensus_metrics["gain_ci95"][0]
                > float(gate["grouped_bootstrap_ci_lower_must_exceed"]),
                "all_split_seed_gains_positive": bool(
                    aggregate["all_split_seed_gains_positive"]
                ),
                "harmful_to_beneficial_mass_ratio": ratio is not None
                and ratio <= float(gate["harmful_to_beneficial_mass_ratio_maximum"]),
                "switch_coverage": float(gate["switch_coverage_minimum"])
                <= consensus_metrics["switch_coverage"]
                <= float(gate["switch_coverage_maximum"]),
                "calibration_slope": float(gate["calibration_slope_minimum"])
                <= slope
                <= float(gate["calibration_slope_maximum"]),
                "top_decile_realized_gap": top_decile > 0.0,
            }
            passed = all(checks.values())
            if passed:
                passing_candidates.append(candidate_id)
            formal_gate_results[candidate_id] = {
                "passed": passed,
                "checks": checks,
                "mean_gain_over_split_seeds": aggregate["mean_gain_over_split_seeds"],
                "consensus_metrics": consensus_metrics,
            }

    if passing_candidates:
        selected = max(
            passing_candidates,
            key=lambda candidate_id: formal_gate_results[candidate_id]["consensus_metrics"][
                "gain_ci95"
            ][0],
        )
        decision = "FREEZE_FOR_FRESH_DEV_REQUEST"
        best_diagnostic_candidate = selected
        rationale = "At least one candidate passed every frozen formal gate."
    else:
        eligible_9600 = [
            candidate_id
            for candidate_id in candidate_ids
            if learning_summary[candidate_id]["request_9600_eligible"]
        ]
        best_diagnostic_candidate = (
            max(
                candidate_ids,
                key=lambda candidate_id: learning_summary[candidate_id][
                    "mean_gain_by_query_count"
                ][str(max(query_counts))],
            )
            if candidate_ids
            else None
        )
        selected = None
        if max(query_counts) >= 9600:
            decision = "STOP_QUERY_ONLY_V1"
            rationale = (
                "No frozen candidate passed the formal gate after the authorized "
                "9,600-query confirmation. Additional query-only scaling is not authorized."
            )
        else:
            decision = "REQUEST_9600_AUTHORIZATION" if eligible_9600 else "STOP_QUERY_ONLY_V1"
            rationale = (
                "No candidate passed the formal gate, but at least one frozen candidate has "
                "positive 4,800-query mean gain, at least two positive split seeds, and a "
                "positive fitted learning-curve slope."
                if eligible_9600
                else "No candidate passed the formal gate or the frozen 9,600-request rule."
            )
    payload = {
        "protocol_id": config["protocol"]["id"],
        "stage": 9,
        "decision": decision,
        "selected_candidate": selected,
        "best_diagnostic_candidate": best_diagnostic_candidate,
        "rationale": rationale,
        "formal_gate_results": formal_gate_results,
        "learning_curve_summary": learning_summary,
        "does_not_authorize_paid_work": True,
        "external_calls": 0,
    }
    _write_json(output_dir / "decision.json", payload)
    _write_execution_state(config, output_dir)
    print(json.dumps({"decision": decision, "selected_candidate": selected}, indent=2))


def main() -> int:
    args = _arguments()
    config = _load_config(args.config)
    output_dir = _resolve(args.output_dir or config["planned_implementation"]["output_dir"])
    data, validation = load_frozen_data(config)
    bootstrap_resamples = int(
        args.bootstrap_resamples
        if args.bootstrap_resamples is not None
        else config["evaluation"]["bootstrap"]["resamples"]
    )
    if bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    if args.stage == "preflight":
        run_preflight(config, data, validation, output_dir)
    elif args.stage == "screen":
        run_screen(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    elif args.stage == "stage2-report":
        run_stage2_report(config, data, output_dir)
    elif args.stage == "stage3-targets":
        run_stage3_targets(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    elif args.stage == "rotation":
        run_rotation(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    elif args.stage == "stage4-report":
        run_stage4_report(config, output_dir)
    elif args.stage == "stage5-calibration":
        run_stage5_calibration(config, output_dir)
    elif args.stage == "stage6-evaluation":
        run_stage6_evaluation(config, output_dir)
    elif args.stage == "freeze-candidates":
        freeze_formal_candidates(config, output_dir)
    elif args.stage == "formal":
        run_formal(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    elif args.stage == "learning-curve":
        run_learning_curve(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    elif args.stage == "decision":
        run_decision(
            config,
            data,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    else:
        raise AssertionError(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
