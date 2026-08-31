"""Run the frozen Phase 2.9 original-query single-trait audit.

This runner intentionally has no model-fitting code. It stops before natural
confirmation when discovery yields no stable scalar feature, and it stops
before combinations/training when natural confirmation yields no supported
single feature.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.router_phase29_trait_features import (  # noqa: E402
    CorpusCache,
    FeatureSpec,
    build_corpus_cache,
    catalog_sha256,
    extract_feature_matrix,
    feature_catalog,
)


DEFAULT_CONFIG = "analysis/hotpotqa_router/phase29_query_trait_audit_config.yaml"
RUNNER_PATH = Path(__file__).resolve()
FEATURE_IMPLEMENTATION_PATH = REPO_ROOT / "scripts" / "router_phase29_trait_features.py"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "build-corpus-cache",
            "extract-discovery",
            "analyze-discovery",
            "extract-natural",
            "analyze-natural",
            "run",
            "status",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase 2.9 config must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _input(config: Mapping[str, Any], name: str) -> Path:
    return _resolve(str(config["inputs"][name]["path"]))


def _run_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(str(config["outputs"]["run_dir"]))


def _cache_path(config: Mapping[str, Any]) -> Path:
    version = int(config["feature_contract"]["stage_c"]["corpus_sample"]["cache_version"])
    return _run_dir(config) / f"corpus_static_cache_v{version}.npz"


def _implementation_hashes(config_path: Path, catalog_path: Path) -> dict[str, str]:
    return {
        "config": _sha256(config_path),
        "runner": _sha256(RUNNER_PATH),
        "feature_extractor": _sha256(FEATURE_IMPLEMENTATION_PATH),
        "feature_catalog_file": _sha256(catalog_path),
    }


def _assert_implementation_freeze(config: Mapping[str, Any]) -> None:
    preflight_path = _run_dir(config) / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Preflight freeze record is missing")
    frozen = json.loads(preflight_path.read_text(encoding="utf-8"))
    config_path = _resolve(str(frozen["config"]))
    catalog_path = _run_dir(config) / "feature_catalog.json"
    current = _implementation_hashes(config_path, catalog_path)
    if current != frozen["implementation_hashes"]:
        raise RuntimeError(
            f"Phase 2.9 implementation changed after preflight: frozen={frozen['implementation_hashes']} current={current}"
        )


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _schema_names(path: Path) -> tuple[list[str], list[str]]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    lexical = [str(value) for value in schema["lexical"]]
    dense = [str(value) for value in schema["dense_corpus"]]
    if len(lexical) != 17 or len(dense) != 13:
        raise ValueError("Historical feature schema must be 17D + 13D")
    return lexical, dense


def _discovery_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = _read_csv_gz(_input(config, "discovery_summary"))
    with np.load(_input(config, "discovery_features"), allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        lexical = np.asarray(stored["lexical"], dtype=np.float64)
        dense = np.asarray(stored["dense"], dtype=np.float64)
    row_ids = np.asarray([row["query_id"] for row in rows], dtype=np.str_)
    row_groups = np.asarray([row["group_id"] for row in rows], dtype=np.str_)
    if not np.array_equal(query_ids, row_ids) or not np.array_equal(group_ids, row_groups):
        raise ValueError("Discovery summary/features ID order differs")
    keep = np.asarray([row["f1_winner"] in {"bm25", "dense"} for row in rows], dtype=bool)
    selected = [row for row, retain in zip(rows, keep) if retain]
    old_rows = _read_csv_gz(_input(config, "old_cohort_summary"))
    old_ids = {row["query_id"] for row in old_rows}
    origins = np.asarray(["old" if row["query_id"] in old_ids else "new" for row in selected], dtype=np.str_)
    return {
        "rows_all": rows,
        "rows": selected,
        "query_ids": query_ids[keep],
        "group_ids": group_ids[keep],
        "questions": [row["question"] for row in selected],
        "legacy_lexical": lexical[keep],
        "legacy_dense": dense[keep],
        "labels": np.asarray([1 if row["f1_winner"] == "bm25" else 0 for row in selected], dtype=np.int8),
        "gaps": np.asarray([float(row["f1_gap_bm25_minus_dense"]) for row in selected], dtype=np.float64),
        "retrieval_gaps": np.asarray(
            [float(row["bm25_evidence_page_recall"]) - float(row["dense_evidence_page_recall"]) for row in selected],
            dtype=np.float64,
        ),
        "origins": origins,
    }


def _discovery_questions_without_outcomes(config: Mapping[str, Any]) -> list[str]:
    """Load discovery query text from query IDs without parsing any utility file."""

    with np.load(_input(config, "discovery_features"), allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
    texts = _query_texts(_input(config, "query_source"), set(query_ids.tolist()))
    return [texts[str(query_id)] for query_id in query_ids]


def _query_texts(path: Path, requested_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row.get("_id", row.get("id", "")))
            if query_id in requested_ids:
                text = row.get("text", row.get("query", row.get("question")))
                if text is None:
                    raise ValueError(f"Query {query_id} has no text field")
                found[query_id] = str(text)
                if len(found) == len(requested_ids):
                    break
    missing = sorted(requested_ids - set(found))
    if missing:
        raise ValueError(f"Missing natural query texts, first IDs: {missing[:5]}")
    return found


def _natural_membership_and_features(config: Mapping[str, Any]) -> dict[str, Any]:
    membership = json.loads(_input(config, "natural_membership").read_text(encoding="utf-8"))["rows"]
    member_ids = np.asarray([row["query_id"] for row in membership], dtype=np.str_)
    member_groups = np.asarray([row["group_id"] for row in membership], dtype=np.str_)
    with np.load(_input(config, "natural_features"), allow_pickle=False) as stored:
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
        lexical = np.asarray(stored["lexical"], dtype=np.float64)
        dense = np.asarray(stored["dense"], dtype=np.float64)
    if not np.array_equal(member_ids, query_ids) or not np.array_equal(member_groups, group_ids):
        raise ValueError("Natural membership/features ID order differs")
    texts = _query_texts(_input(config, "query_source"), set(query_ids.tolist()))
    return {
        "query_ids": query_ids,
        "group_ids": group_ids,
        "questions": [texts[str(query_id)] for query_id in query_ids],
        "legacy_lexical": lexical,
        "legacy_dense": dense,
    }


def _natural_metrics(config: Mapping[str, Any], query_ids: np.ndarray, group_ids: np.ndarray) -> dict[str, Any]:
    rows = _read_jsonl_gz(_input(config, "natural_metrics"))
    metric_ids = np.asarray([row["query_id"] for row in rows], dtype=np.str_)
    metric_groups = np.asarray([row["group_id"] for row in rows], dtype=np.str_)
    if not np.array_equal(metric_ids, query_ids) or not np.array_equal(metric_groups, group_ids):
        raise ValueError("Natural metrics/features ID order differs")
    gaps = np.asarray([float(row["gap"]) for row in rows], dtype=np.float64)
    labels = np.where(gaps > 0, 1, np.where(gaps < 0, 0, -1)).astype(np.int8)
    return {"rows": rows, "gaps": gaps, "labels": labels}


def _verify_input_hashes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, entry in config["inputs"].items():
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            continue
        path = _resolve(str(entry["path"]))
        actual = _sha256(path)
        expected = str(entry["sha256"])
        results.append({"input": name, "path": str(path.relative_to(REPO_ROOT)), "sha256": actual, "passed": actual == expected})
        if actual != expected:
            raise ValueError(f"Input hash mismatch for {name}: {actual} != {expected}")
    return results


def preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the catalog and implementation before parsing any discovery outcome.
    lexical_names, dense_names = _schema_names(_input(config, "discovery_feature_schema"))
    specs = feature_catalog(lexical_names, dense_names)
    catalog = {
        "protocol_id": config["protocol"]["id"],
        "frozen_before_outcome_association": True,
        "feature_count": len(specs),
        "catalog_sha256": catalog_sha256(specs),
        "features": [spec.__dict__ for spec in specs],
    }
    _write_json(run_dir / "feature_catalog.json", catalog)
    implementation_hashes = _implementation_hashes(config_path, run_dir / "feature_catalog.json")
    hashes = _verify_input_hashes(config)
    discovery = _discovery_inputs(config)
    natural = _natural_membership_and_features(config)

    labels = discovery["labels"]
    origins = discovery["origins"]
    group_counts = Counter(discovery["group_ids"].tolist())
    result = {
        "protocol_id": config["protocol"]["id"],
        "status": "passed",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_path),
        "implementation_hashes": implementation_hashes,
        "git_provenance": {
            "base_head": str(config["protocol"]["base_git_head"]),
            "phase29_files_tracked_by_base_head": False,
            "freeze_mechanism": "artifact_sha256",
        },
        "input_hashes": hashes,
        "feature_catalog": {
            "path": str((run_dir / "feature_catalog.json").relative_to(REPO_ROOT)),
            "sha256": _sha256(run_dir / "feature_catalog.json"),
            "semantic_catalog_sha256": catalog["catalog_sha256"],
            "features": len(specs),
            "families": dict(Counter(spec.family for spec in specs)),
        },
        "discovery": {
            "rows": len(labels),
            "bm25": int(np.sum(labels == 1)),
            "dense": int(np.sum(labels == 0)),
            "groups": len(group_counts),
            "groups_with_multiple_queries": int(sum(value > 1 for value in group_counts.values())),
            "old": {
                "bm25": int(np.sum((origins == "old") & (labels == 1))),
                "dense": int(np.sum((origins == "old") & (labels == 0))),
            },
            "new": {
                "bm25": int(np.sum((origins == "new") & (labels == 1))),
                "dense": int(np.sum((origins == "new") & (labels == 0))),
            },
        },
        "natural": {
            "rows": len(natural["query_ids"]),
            "groups": len(set(natural["group_ids"].tolist())),
            "role": "previously_consumed_natural_confirmation_not_final_holdout",
            "outcome_values_read_before_shortlist": 0,
            "expected_class_counts_not_revalidated_before_shortlist": dict(
                config["inputs"]["natural_metrics"]["expected_class_counts"]
            ),
        },
        "information_boundary": {
            "query_rewriting": 0,
            "current_ranked_results_as_features": 0,
            "qrels_or_gold_evidence_as_features": 0,
            "official_final_holdout_rows": 0,
            "external_provider_calls": 0,
        },
    }
    expected = config["inputs"]["discovery_summary"]["expected_class_counts"]
    if result["discovery"]["bm25"] != int(expected["bm25"]) or result["discovery"]["dense"] != int(expected["dense"]):
        raise ValueError("Discovery class counts differ from frozen config")
    expected_origins = config["inputs"]["discovery_summary"]["expected_origin_class_counts"]
    for origin in ("old", "new"):
        for action in ("bm25", "dense"):
            if result["discovery"][origin][action] != int(expected_origins[origin][action]):
                raise ValueError(f"Discovery origin count differs for {origin}/{action}")
    if result["natural"]["rows"] != int(config["inputs"]["natural_membership"]["expected_rows"]):
        raise ValueError("Natural membership row count differs")
    _write_json(run_dir / "preflight.json", result)
    return result


def build_cache_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Run preflight before building the corpus cache")
    _assert_implementation_freeze(config)
    cache_path = _cache_path(config)
    manifest_path = run_dir / "corpus_static_cache_manifest.json"
    frozen = json.loads(preflight_path.read_text(encoding="utf-8"))
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "protocol_id": config["protocol"]["id"],
            "cache": str(cache_path.relative_to(REPO_ROOT)),
            "cache_sha256": _sha256(cache_path),
            "implementation_hashes": frozen["implementation_hashes"],
            "cache_version": int(config["feature_contract"]["stage_c"]["corpus_sample"]["cache_version"]),
        }
        actual = {name: manifest.get(name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"Refusing stale corpus-cache reuse: expected={expected} actual={actual}")
        return manifest
    natural = _natural_membership_and_features(config)
    questions = _discovery_questions_without_outcomes(config) + list(natural["questions"])
    sample = config["feature_contract"]["stage_c"]["corpus_sample"]
    details = build_corpus_cache(
        questions=questions,
        index_path=_input(config, "bm25_index"),
        chunks_path=_input(config, "corpus_chunks"),
        offsets_path=_input(config, "corpus_offsets"),
        output_path=cache_path,
        sample_rows=int(sample["rows"]),
        sample_seed=int(sample["seed"]),
    )
    result = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "selection": sample["selection"],
        "uses_outcome_labels": False,
        "stores_ranked_results": False,
        "uses_qrels": False,
        "cache": str(cache_path.relative_to(REPO_ROOT)),
        "cache_sha256": _sha256(cache_path),
        "implementation_hashes": frozen["implementation_hashes"],
        "cache_version": int(sample["cache_version"]),
        "sample_seed": int(sample["seed"]),
        **details,
    }
    _write_json(manifest_path, result)
    return result


def extract_discovery(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    _assert_implementation_freeze(config)
    cache_path = _cache_path(config)
    if not cache_path.is_file():
        raise RuntimeError("Build corpus cache before discovery features")
    output = run_dir / "discovery_features.npz"
    manifest_path = run_dir / "discovery_features_manifest.json"
    if output.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        catalog = json.loads((run_dir / "feature_catalog.json").read_text(encoding="utf-8"))
        expected = {
            "protocol_id": config["protocol"]["id"],
            "matrix_sha256": _sha256(output),
            "corpus_cache_sha256": _sha256(cache_path),
            "catalog_sha256": catalog["catalog_sha256"],
            "implementation_hashes": frozen["implementation_hashes"],
        }
        actual = {name: manifest.get(name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"Refusing stale discovery-feature reuse: expected={expected} actual={actual}")
        return manifest
    data = _discovery_inputs(config)
    lexical_names, dense_names = _schema_names(_input(config, "discovery_feature_schema"))
    cache = CorpusCache(cache_path)
    started = time.perf_counter()
    try:
        matrix, specs = extract_feature_matrix(
            questions=data["questions"],
            legacy_lexical=data["legacy_lexical"],
            legacy_dense=data["legacy_dense"],
            legacy_lexical_names=lexical_names,
            dense_names=dense_names,
            cache=cache,
        )
    finally:
        cache.close()
    catalog = json.loads((run_dir / "feature_catalog.json").read_text(encoding="utf-8"))
    if catalog_sha256(specs) != catalog["catalog_sha256"]:
        raise ValueError("Extracted feature catalog differs from frozen preflight catalog")
    np.savez_compressed(
        output,
        query_ids=data["query_ids"],
        group_ids=data["group_ids"],
        labels=data["labels"],
        gaps=data["gaps"],
        retrieval_gaps=data["retrieval_gaps"],
        origins=data["origins"],
        matrix=matrix,
        feature_names=np.asarray([spec.name for spec in specs], dtype=np.str_),
    )
    result = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "finite_fraction": float(np.mean(np.isfinite(matrix))),
        "matrix": str(output.relative_to(REPO_ROOT)),
        "matrix_sha256": _sha256(output),
        "corpus_cache_sha256": _sha256(cache_path),
        "catalog_sha256": catalog["catalog_sha256"],
        "implementation_hashes": json.loads(
            (run_dir / "preflight.json").read_text(encoding="utf-8")
        )["implementation_hashes"],
        "elapsed_seconds": time.perf_counter() - started,
        "current_ranked_results_read": 0,
        "official_final_holdout_rows": 0,
    }
    _write_json(manifest_path, result)
    return result


def _feature_quality(values: np.ndarray, spec: FeatureSpec, quality: Mapping[str, Any], *, natural: bool) -> dict[str, Any]:
    finite = np.isfinite(values)
    missing_fraction = 1.0 - float(np.mean(finite))
    finite_values = values[finite]
    unique, counts = np.unique(finite_values, return_counts=True) if len(finite_values) else (np.asarray([]), np.asarray([]))
    result: dict[str, Any] = {
        "missing_fraction": missing_fraction,
        "unique_values": int(len(unique)),
        "maximum_single_value_fraction": float(np.max(counts) / len(finite_values)) if len(finite_values) else 1.0,
        "iqr": float(np.percentile(finite_values, 75) - np.percentile(finite_values, 25)) if len(finite_values) else 0.0,
        "passed": True,
        "failed_checks": [],
    }
    if missing_fraction > float(quality["maximum_missing_fraction"]):
        result["failed_checks"].append("missing_fraction")
    if spec.kind == "binary":
        minimum = int(quality["binary_natural_minimum_per_level"] if natural else quality["binary_discovery_minimum_per_level"])
        if len(unique) != 2 or int(np.min(counts)) < minimum:
            result["failed_checks"].append("binary_level_count")
    else:
        if len(unique) < int(quality["continuous_minimum_unique_values"]):
            result["failed_checks"].append("unique_values")
        if result["iqr"] <= float(quality["continuous_iqr_must_exceed"]):
            result["failed_checks"].append("iqr")
        if result["maximum_single_value_fraction"] > float(quality["continuous_maximum_single_value_fraction"]):
            result["failed_checks"].append("dominant_value")
    result["passed"] = not result["failed_checks"]
    return result


def _auc(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(labels) & np.isin(labels, (0, 1))
    x = values[mask]
    y = labels[mask]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(x, method="average")
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _odds_ratio(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(labels) & np.isin(labels, (0, 1))
    finite_values = values[mask]
    finite_labels = labels[mask]
    if not len(finite_values):
        return float("nan")
    low, high = np.min(finite_values), np.max(finite_values)
    if low == high:
        return float("nan")
    x = finite_values == high
    a = float(np.sum(x & (finite_labels == 1))) + 0.5
    b = float(np.sum(x & (finite_labels == 0))) + 0.5
    c = float(np.sum((~x) & (finite_labels == 1))) + 0.5
    d = float(np.sum((~x) & (finite_labels == 0))) + 0.5
    return (a * d) / (b * c)


def _orient_values(values: np.ndarray, spec: FeatureSpec, direction: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if direction >= 0:
        return values
    if spec.kind == "binary":
        finite = values[np.isfinite(values)]
        if not len(finite):
            return values
        return float(np.min(finite) + np.max(finite)) - values
    return -values


def _gap_effect(values: np.ndarray, gaps: np.ndarray, spec: FeatureSpec, direction: int) -> float:
    oriented = _orient_values(values, spec, direction)
    finite = np.isfinite(oriented) & np.isfinite(gaps)
    x = oriented[finite]
    g = gaps[finite]
    if spec.kind == "binary":
        high = x > np.min(x)
        return float(np.mean(g[high]) - np.mean(g[~high])) if np.any(high) and np.any(~high) else float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    low = x <= q1
    high = x >= q3
    return float(np.mean(g[high]) - np.mean(g[low])) if np.any(high) and np.any(low) else float("nan")


def _point_metrics(values: np.ndarray, labels: np.ndarray, gaps: np.ndarray, spec: FeatureSpec, direction: int | None = None) -> dict[str, float | int]:
    complete = np.isfinite(values) & np.isfinite(gaps) & np.isfinite(labels) & np.isin(labels, (0, 1))
    values_complete = values[complete]
    labels_complete = labels[complete]
    gaps_complete = gaps[complete]
    raw_auc = _auc(values_complete, labels_complete)
    raw_or = _odds_ratio(values_complete, labels_complete) if spec.kind == "binary" else float("nan")
    if direction is None:
        direction = 1 if (raw_or >= 1.0 if spec.kind == "binary" else raw_auc >= 0.5) else -1
    oriented_values = _orient_values(values_complete, spec, direction)
    oriented_auc = _auc(oriented_values, labels_complete)
    oriented_or = _odds_ratio(oriented_values, labels_complete) if spec.kind == "binary" else float("nan")
    raw_rho = float(spearmanr(values_complete, gaps_complete, nan_policy="omit").statistic)
    oriented_rho = direction * raw_rho
    gap_effect = _gap_effect(values_complete, gaps_complete, spec, direction)
    return {
        "direction": int(direction),
        "raw_auc": raw_auc,
        "directional_auc": oriented_auc,
        "raw_odds_ratio": raw_or,
        "directional_odds_ratio": oriented_or,
        "raw_spearman": raw_rho,
        "directional_spearman": oriented_rho,
        "directional_gap_effect": gap_effect,
    }


def _group_blocks(group_ids: np.ndarray) -> dict[int, list[np.ndarray]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[str(group_id)].append(index)
    blocks: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for indices in grouped.values():
        blocks[len(indices)].append(np.asarray(indices, dtype=np.int64))
    return dict(blocks)


def _permuted_row_indices(
    blocks: Mapping[int, Sequence[np.ndarray]],
    rows: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty((resamples, rows), dtype=np.int32)
    identity = np.arange(rows, dtype=np.int32)
    block_arrays = {
        size: np.vstack(values).astype(np.int64, copy=False)
        for size, values in blocks.items()
    }
    for sample in range(resamples):
        mapping = identity.copy()
        for values in block_arrays.values():
            order = rng.permutation(len(values))
            mapping[values.reshape(-1)] = values[order].reshape(-1)
        output[sample] = mapping
    return output


def _permutation_pvalues(
    *,
    matrix: np.ndarray,
    feature_indices: Sequence[int],
    specs: Sequence[FeatureSpec],
    directions: Sequence[int],
    labels: np.ndarray,
    gaps: np.ndarray,
    group_ids: np.ndarray,
    resamples: int,
    seed: int,
    one_sided_frozen_direction: bool,
) -> tuple[dict[int, float], dict[int, float]]:
    if not feature_indices:
        return {}, {}
    selected = np.asarray(feature_indices, dtype=np.int64)
    x = matrix[:, selected]
    if not np.isfinite(x).all() or not np.isfinite(gaps).all():
        raise ValueError("Permutation inputs must be fully finite under the v1 zero-missing contract")
    oriented = np.column_stack([
        _orient_values(x[:, position], specs[int(feature_index)], int(directions[position]))
        for position, feature_index in enumerate(selected)
    ])
    label_ranks = np.column_stack([rankdata(oriented[:, j], method="average") for j in range(oriented.shape[1])])
    x_rank_z = np.column_stack([
        (rankdata(oriented[:, j], method="average") - np.mean(rankdata(oriented[:, j], method="average")))
        / max(np.std(rankdata(oriented[:, j], method="average")), 1e-12)
        for j in range(oriented.shape[1])
    ])
    gap_ranks = rankdata(gaps, method="average")
    gap_rank_z = (gap_ranks - np.mean(gap_ranks)) / max(np.std(gap_ranks), 1e-12)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    observed_auc = np.asarray([_auc(oriented[:, j], labels) for j in range(oriented.shape[1])])
    observed_rho = np.asarray([float(spearmanr(oriented[:, j], gaps).statistic) for j in range(oriented.shape[1])])
    observed_binary_gap = np.asarray([
        _gap_effect(x[:, j], gaps, specs[index], int(directions[position]))
        for position, (j, index) in enumerate(zip(range(oriented.shape[1]), selected))
    ])
    winner_extreme = np.zeros(len(selected), dtype=np.int64)
    gap_extreme = np.zeros(len(selected), dtype=np.int64)
    blocks = _group_blocks(group_ids)
    rng = np.random.default_rng(seed)
    batch = 100
    for start in range(0, resamples, batch):
        take = min(batch, resamples - start)
        indices = _permuted_row_indices(blocks, len(labels), take, rng)
        permuted_labels = labels[indices]
        rank_sums = np.einsum("bn,nk->bk", permuted_labels, label_ranks, optimize=True)
        aucs = (rank_sums - positives * (positives + 1) / 2.0) / (positives * negatives)
        permuted_gap_z = gap_rank_z[indices]
        rhos = (permuted_gap_z @ x_rank_z) / len(labels)
        for position, feature_index in enumerate(selected):
            if one_sided_frozen_direction:
                winner_extreme[position] += int(np.sum(aucs[:, position] >= observed_auc[position] - 1e-15))
            else:
                winner_extreme[position] += int(np.sum(np.abs(aucs[:, position] - 0.5) >= abs(observed_auc[position] - 0.5) - 1e-15))
            if specs[int(feature_index)].kind == "binary":
                binary = oriented[:, position] > np.min(oriented[:, position])
                high_n = int(np.sum(binary))
                low_n = len(binary) - high_n
                centered = binary.astype(np.float64) / max(high_n, 1) - (~binary).astype(np.float64) / max(low_n, 1)
                differences = gaps[indices] @ centered
                if one_sided_frozen_direction:
                    gap_extreme[position] += int(np.sum(differences >= observed_binary_gap[position] - 1e-15))
                else:
                    gap_extreme[position] += int(np.sum(np.abs(differences) >= abs(observed_binary_gap[position]) - 1e-15))
            else:
                if one_sided_frozen_direction:
                    gap_extreme[position] += int(np.sum(rhos[:, position] >= observed_rho[position] - 1e-15))
                else:
                    gap_extreme[position] += int(np.sum(np.abs(rhos[:, position]) >= abs(observed_rho[position]) - 1e-15))
        if (start + take) % 1000 == 0:
            print(f"permutations: {start + take}/{resamples}", flush=True)
    winner = {int(index): float((winner_extreme[position] + 1) / (resamples + 1)) for position, index in enumerate(selected)}
    gap = {int(index): float((gap_extreme[position] + 1) / (resamples + 1)) for position, index in enumerate(selected)}
    return winner, gap


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _old_new_stability(
    values: np.ndarray,
    labels: np.ndarray,
    gaps: np.ndarray,
    origins: np.ndarray,
    spec: FeatureSpec,
    direction: int,
    full: Mapping[str, Any],
    gate: Mapping[str, Any],
    quality_config: Mapping[str, Any],
) -> dict[str, Any]:
    subgroups: dict[str, Any] = {}
    passed = True
    failed: list[str] = []
    preserve = float(gate["preserve_full_effect_fraction"])
    for origin in ("old", "new"):
        mask = origins == origin
        subgroup_quality_config = dict(quality_config)
        subgroup_quality_config["binary_discovery_minimum_per_level"] = int(
            gate["minimum_binary_rows_per_level"]
        )
        subgroup_quality = _feature_quality(
            values[mask], spec, subgroup_quality_config, natural=False
        )
        metrics = _point_metrics(values[mask], labels[mask], gaps[mask], spec, direction=direction)
        subgroups[origin] = {"quality": subgroup_quality, "metrics": metrics}
        if not subgroup_quality["passed"]:
            failed.extend(f"{origin}_quality_{value}" for value in subgroup_quality["failed_checks"])
            continue
        required = ["directional_gap_effect"]
        required.append("directional_odds_ratio" if spec.kind == "binary" else "directional_auc")
        if spec.kind != "binary":
            required.append("directional_spearman")
        if not all(np.isfinite(float(metrics[name])) for name in required):
            failed.append(f"{origin}_nonfinite_metric")
            continue
        if spec.kind == "binary":
            full_log = max(0.0, math.log(max(float(full["directional_odds_ratio"]), 1e-12)))
            sub_log = math.log(max(float(metrics["directional_odds_ratio"]), 1e-12))
            if float(metrics["directional_odds_ratio"]) <= float(gate["binary_or_must_exceed"]):
                failed.append(f"{origin}_or")
            if sub_log < preserve * full_log:
                failed.append(f"{origin}_or_retention")
        else:
            full_auc_excess = max(0.0, float(full["directional_auc"]) - 0.5)
            sub_auc_excess = float(metrics["directional_auc"]) - 0.5
            if float(metrics["directional_auc"]) < float(gate["continuous_auc_minimum"]):
                failed.append(f"{origin}_auc")
            if float(metrics["directional_spearman"]) < float(gate["continuous_spearman_minimum"]):
                failed.append(f"{origin}_spearman")
            if sub_auc_excess < preserve * full_auc_excess:
                failed.append(f"{origin}_auc_retention")
            if float(metrics["directional_spearman"]) < preserve * float(full["directional_spearman"]):
                failed.append(f"{origin}_spearman_retention")
        if float(metrics["directional_gap_effect"]) <= 0:
            failed.append(f"{origin}_gap_direction")
        if float(metrics["directional_gap_effect"]) < preserve * float(full["directional_gap_effect"]):
            failed.append(f"{origin}_gap_retention")
    passed = not failed
    return {"passed": passed, "failed_checks": failed, "subgroups": subgroups}


def _bootstrap_candidate(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    gaps: np.ndarray,
    group_ids: np.ndarray,
    spec: FeatureSpec,
    direction: int,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    if not np.isfinite(values).all() or not np.isfinite(gaps).all():
        raise ValueError("Bootstrap inputs must be fully finite under the v1 zero-missing contract")
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[str(group_id)].append(index)
    blocks = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
    block_sizes = np.asarray([len(block) for block in blocks], dtype=np.int64)
    singleton_rows = np.asarray([block[0] if len(block) == 1 else -1 for block in blocks], dtype=np.int64)
    rng = np.random.default_rng(seed)
    aucs = np.empty(resamples, dtype=np.float64)
    rhos = np.empty(resamples, dtype=np.float64)
    gap_effects = np.empty(resamples, dtype=np.float64)
    odds = np.empty(resamples, dtype=np.float64)
    oriented = _orient_values(values, spec, direction)
    for sample in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        chosen_sizes = block_sizes[chosen]
        singleton = chosen[chosen_sizes == 1]
        nonsingleton = chosen[chosen_sizes > 1]
        parts = [singleton_rows[singleton]]
        if len(nonsingleton):
            parts.extend(blocks[int(position)] for position in nonsingleton)
        indices = np.concatenate(parts)
        x = oriented[indices]
        y = labels[indices]
        g = gaps[indices]
        aucs[sample] = _auc(x, y)
        rhos[sample] = float(spearmanr(x, g, nan_policy="omit").statistic)
        gap_effects[sample] = _gap_effect(values[indices], g, spec, direction)
        odds[sample] = _odds_ratio(x, y) if spec.kind == "binary" else np.nan
        if (sample + 1) % 2000 == 0:
            print(f"bootstrap {spec.name}: {sample + 1}/{resamples}", flush=True)

    def interval(array: np.ndarray) -> list[float]:
        finite = array[np.isfinite(array)]
        return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]

    return {
        "auc_ci95": interval(aucs),
        "spearman_ci95": interval(rhos),
        "gap_effect_ci95": interval(gap_effects),
        "odds_ratio_ci95": interval(odds) if spec.kind == "binary" else [float("nan"), float("nan")],
    }


def _discovery_point_gate(metrics: Mapping[str, Any], spec: FeatureSpec, gate: Mapping[str, Any]) -> bool:
    if spec.kind == "binary":
        binary = gate["binary_gate"]
        if not all(np.isfinite(float(metrics[name])) for name in ("directional_odds_ratio", "directional_gap_effect")):
            return False
        return (
            float(metrics["directional_odds_ratio"]) >= float(binary["directional_odds_ratio_minimum"])
            and float(metrics["directional_gap_effect"]) >= float(binary["directional_gap_difference_minimum"])
        )
    continuous = gate["continuous_gate"]
    if not all(np.isfinite(float(metrics[name])) for name in ("directional_auc", "directional_spearman", "directional_gap_effect")):
        return False
    return (
        float(metrics["directional_auc"]) >= float(continuous["directional_auc_minimum"])
        and float(metrics["directional_spearman"]) >= float(continuous["directional_spearman_minimum"])
        and float(metrics["directional_gap_effect"]) >= float(continuous["directional_q4_minus_q1_gap_minimum"])
    )


def _ci_gate(metrics: Mapping[str, Any], spec: FeatureSpec, gate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    scalar_names = ["directional_gap_effect", "winner_p_holm", "gap_p_holm"]
    scalar_names.append("directional_odds_ratio" if spec.kind == "binary" else "directional_auc")
    if spec.kind != "binary":
        scalar_names.append("directional_spearman")
    interval_names = ["gap_effect_ci95", "odds_ratio_ci95" if spec.kind == "binary" else "auc_ci95"]
    if spec.kind != "binary":
        interval_names.append("spearman_ci95")
    if not all(np.isfinite(float(metrics[name])) for name in scalar_names) or not all(
        np.isfinite(np.asarray(metrics[name], dtype=np.float64)).all() for name in interval_names
    ):
        return False, ["nonfinite_metric_or_interval"]
    if spec.kind == "binary":
        binary = gate["binary_gate"]
        if float(metrics["directional_odds_ratio"]) < float(binary["directional_odds_ratio_minimum"]):
            failed.append("odds_ratio_effect")
        if float(metrics["odds_ratio_ci95"][0]) <= float(binary["odds_ratio_ci95_lower_must_exceed"]):
            failed.append("odds_ratio_ci")
        if float(metrics["directional_gap_effect"]) < float(binary["directional_gap_difference_minimum"]):
            failed.append("gap_effect")
        if float(metrics["gap_effect_ci95"][0]) <= float(binary["gap_difference_ci95_lower_must_exceed"]):
            failed.append("gap_effect_ci")
    else:
        continuous = gate["continuous_gate"]
        if float(metrics["directional_auc"]) < float(continuous["directional_auc_minimum"]):
            failed.append("auc_effect")
        if float(metrics["auc_ci95"][0]) <= float(continuous["auc_ci95_lower_must_exceed"]):
            failed.append("auc_ci")
        if float(metrics["directional_spearman"]) < float(continuous["directional_spearman_minimum"]):
            failed.append("spearman_effect")
        rho_low, rho_high = metrics["spearman_ci95"]
        if float(rho_low) <= 0 <= float(rho_high):
            failed.append("spearman_ci")
        if float(metrics["directional_gap_effect"]) < float(continuous["directional_q4_minus_q1_gap_minimum"]):
            failed.append("gap_effect")
        if float(metrics["gap_effect_ci95"][0]) <= float(continuous["q4_minus_q1_ci95_lower_must_exceed"]):
            failed.append("gap_effect_ci")
    if float(metrics["winner_p_holm"]) > 0.05:
        failed.append("winner_holm")
    if float(metrics["gap_p_holm"]) > 0.05:
        failed.append("gap_holm")
    return not failed, failed


def _flat_row(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    quality = result["quality"]
    stability = result.get("old_new_stability", {})
    return {
        "feature": result["feature"],
        "family": result["family"],
        "kind": result["kind"],
        "status": result["status"],
        "direction": metrics.get("direction"),
        "directional_auc": metrics.get("directional_auc"),
        "directional_odds_ratio": metrics.get("directional_odds_ratio"),
        "directional_spearman": metrics.get("directional_spearman"),
        "directional_gap_effect": metrics.get("directional_gap_effect"),
        "winner_p_raw": metrics.get("winner_p_raw"),
        "winner_p_holm": metrics.get("winner_p_holm"),
        "gap_p_raw": metrics.get("gap_p_raw"),
        "gap_p_holm": metrics.get("gap_p_holm"),
        "auc_ci95_lower": metrics.get("auc_ci95", [None, None])[0],
        "auc_ci95_upper": metrics.get("auc_ci95", [None, None])[1],
        "spearman_ci95_lower": metrics.get("spearman_ci95", [None, None])[0],
        "spearman_ci95_upper": metrics.get("spearman_ci95", [None, None])[1],
        "gap_ci95_lower": metrics.get("gap_effect_ci95", [None, None])[0],
        "gap_ci95_upper": metrics.get("gap_effect_ci95", [None, None])[1],
        "or_ci95_lower": metrics.get("odds_ratio_ci95", [None, None])[0],
        "or_ci95_upper": metrics.get("odds_ratio_ci95", [None, None])[1],
        "retrieval_gap_spearman": metrics.get("retrieval_gap_spearman"),
        "quality_pass": quality.get("passed"),
        "missing_fraction": quality.get("missing_fraction"),
        "unique_values": quality.get("unique_values"),
        "old_new_stable": stability.get("passed"),
        "failed_checks": "|".join(result.get("failed_checks", [])),
    }


def _write_report(
    config: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    discovery_results: Sequence[Mapping[str, Any]],
    natural_results: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    run_dir = _run_dir(config)
    discovery_passed = [row for row in discovery_results if row["status"] == "advance_to_natural"]
    natural_supported = [row for row in (natural_results or []) if row["status"] == "supported_single_feature"]
    top = sorted(
        discovery_results,
        key=lambda row: float(row["metrics"].get("directional_auc", float("-inf"))),
        reverse=True,
    )[:15]
    lines = [
        "# Phase 2.9 HotpotQA BM25/BGE 原始 Query 单特质审计",
        "",
        f"- 决策：`{decision['decision']}`",
        f"- 冻结候选特质：{len(discovery_results)}",
        f"- 发现集通过：{len(discovery_passed)}",
        f"- 自然复现支持：{len(natural_supported)}",
        "- Query rewrite：0",
        "- 特征组合/交互：0",
        "- Router 训练：0",
        "- 官方 final holdout 读取：0",
        "",
        "## 发现集最高 directional AUC（仅用于审计，不代表通过）",
        "",
        "| Feature | Family | AUC | Spearman | Gap effect | Holm winner | Holm gap | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in top:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['feature']}` | {row['family']} | {float(metrics.get('directional_auc', float('nan'))):.4f} "
            f"| {float(metrics.get('directional_spearman', float('nan'))):.4f} "
            f"| {float(metrics.get('directional_gap_effect', float('nan'))):.4f} "
            f"| {float(metrics.get('winner_p_holm', 1.0)):.4g} "
            f"| {float(metrics.get('gap_p_holm', 1.0)):.4g} | {row['status']} |"
        )
    lines.extend([
        "",
        "## 结论边界",
        "",
        str(decision["conclusion"]),
        "",
        "检索 evidence-page recall 的关联只作为机制描述；单特质是否通过完全由 answer-F1 winner/gap 门控决定。",
        "自然 2,000 已经用于此前候选选择，只能作为预注册自然分布复现，不是新的 final holdout。",
        "",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_discovery(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    _assert_implementation_freeze(config)
    path = run_dir / "discovery_features.npz"
    if not path.is_file():
        raise RuntimeError("Extract discovery features before analysis")
    with np.load(path, allow_pickle=False) as stored:
        matrix = np.asarray(stored["matrix"], dtype=np.float64)
        feature_names = np.asarray(stored["feature_names"], dtype=np.str_)
        labels = np.asarray(stored["labels"], dtype=np.int8)
        gaps = np.asarray(stored["gaps"], dtype=np.float64)
        retrieval_gaps = np.asarray(stored["retrieval_gaps"], dtype=np.float64)
        origins = np.asarray(stored["origins"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
    catalog_data = json.loads((run_dir / "feature_catalog.json").read_text(encoding="utf-8"))
    specs = [FeatureSpec(**value) for value in catalog_data["features"]]
    if [spec.name for spec in specs] != feature_names.tolist():
        raise ValueError("Discovery matrix columns differ from frozen catalog")
    stats_config = config["statistics"]
    quality_config = stats_config["quality"]
    gate = stats_config["discovery"]
    results: list[dict[str, Any]] = []
    inference_indices: list[int] = []
    directions: list[int] = []
    for index, spec in enumerate(specs):
        values = matrix[:, index]
        quality = _feature_quality(values, spec, quality_config, natural=False)
        metrics = _point_metrics(values, labels, gaps, spec)
        metrics["retrieval_gap_spearman"] = float(spearmanr(values, retrieval_gaps, nan_policy="omit").statistic)
        point_pass = quality["passed"] and _discovery_point_gate(metrics, spec, gate)
        result = {
            "feature": spec.name,
            "family": spec.family,
            "kind": spec.kind,
            "quality": quality,
            "metrics": metrics,
            "point_gate_pass": point_pass,
            "status": "pending_inference" if point_pass else ("insufficient_variation" if not quality["passed"] else "practically_too_small"),
            "failed_checks": list(quality["failed_checks"]),
        }
        results.append(result)
        if point_pass:
            inference_indices.append(index)
            directions.append(int(metrics["direction"]))
    inference = stats_config["inference"]
    winner_p, gap_p = _permutation_pvalues(
        matrix=matrix,
        feature_indices=inference_indices,
        specs=specs,
        directions=directions,
        labels=labels,
        gaps=gaps,
        group_ids=group_ids,
        resamples=int(inference["permutation_resamples"]),
        seed=int(inference["seed"]),
        one_sided_frozen_direction=False,
    )
    all_winner = np.ones(len(specs), dtype=np.float64)
    all_gap = np.ones(len(specs), dtype=np.float64)
    for index, value in winner_p.items():
        all_winner[index] = value
    for index, value in gap_p.items():
        all_gap[index] = value
    winner_holm = _holm_adjust(all_winner)
    gap_holm = _holm_adjust(all_gap)
    bootstrap_indices: list[int] = []
    for index, result in enumerate(results):
        metrics = result["metrics"]
        metrics["winner_p_raw"] = float(all_winner[index])
        metrics["winner_p_holm"] = float(winner_holm[index])
        metrics["gap_p_raw"] = float(all_gap[index])
        metrics["gap_p_holm"] = float(gap_holm[index])
        metrics.update({
            "auc_ci95": [float("nan"), float("nan")],
            "spearman_ci95": [float("nan"), float("nan")],
            "gap_effect_ci95": [float("nan"), float("nan")],
            "odds_ratio_ci95": [float("nan"), float("nan")],
        })
        if not result["point_gate_pass"]:
            continue
        stability = _old_new_stability(
            matrix[:, index], labels, gaps, origins, specs[index], int(metrics["direction"]), metrics,
            gate["old_new_stability"], quality_config,
        )
        result["old_new_stability"] = stability
        if not stability["passed"]:
            result["status"] = "old_new_unstable"
            result["failed_checks"].extend(stability["failed_checks"])
            continue
        if metrics["winner_p_holm"] > 0.05 or metrics["gap_p_holm"] > 0.05:
            result["status"] = "multiplicity_not_significant"
            if metrics["winner_p_holm"] > 0.05:
                result["failed_checks"].append("winner_holm")
            if metrics["gap_p_holm"] > 0.05:
                result["failed_checks"].append("gap_holm")
            continue
        bootstrap_indices.append(index)
    for index in bootstrap_indices:
        result = results[index]
        metrics = result["metrics"]
        intervals = _bootstrap_candidate(
            values=matrix[:, index], labels=labels, gaps=gaps, group_ids=group_ids,
            spec=specs[index], direction=int(metrics["direction"]),
            resamples=int(inference["bootstrap_resamples"]), seed=int(inference["seed"]) + 1000 + index,
        )
        metrics.update(intervals)
        passed, failed = _ci_gate(metrics, specs[index], gate)
        result["status"] = "advance_to_natural" if passed else "confidence_interval_gate_failed"
        result["failed_checks"].extend(failed)

    advanced = [result for result in results if result["status"] == "advance_to_natural"]
    report = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "rows": int(len(labels)),
        "groups": int(len(set(group_ids.tolist()))),
        "features_tested": len(specs),
        "features_quality_passed": int(sum(result["quality"]["passed"] for result in results)),
        "features_point_gate_passed": int(sum(result["point_gate_pass"] for result in results)),
        "features_advanced_to_natural": len(advanced),
        "multiple_testing": "Holm separately over all frozen winner and gap tests",
        "results": results,
    }
    _write_json(run_dir / "discovery_single_feature_results.json", report)
    _write_csv(run_dir / "discovery_single_feature_results.csv", [_flat_row(result) for result in results])
    shortlist = {
        "protocol_id": config["protocol"]["id"],
        "frozen_before_natural_feature_outcome_analysis": True,
        "discovery_results_sha256": _sha256(run_dir / "discovery_single_feature_results.json"),
        "catalog_sha256": catalog_data["catalog_sha256"],
        "features": [
            {
                "feature": result["feature"],
                "direction": int(result["metrics"]["direction"]),
                "family": result["family"],
                "kind": result["kind"],
            }
            for result in advanced
        ],
    }
    _write_json(run_dir / "single_feature_shortlist.json", shortlist)
    if not advanced:
        decision = {
            "protocol_id": config["protocol"]["id"],
            "decision": "STOP_NO_DISCOVERY_SINGLE_FEATURE",
            "supported_single_features": [],
            "natural_confirmation_run": False,
            "combination_analysis_run": False,
            "interaction_analysis_run": False,
            "model_training_run": False,
            "official_final_holdout_rows": 0,
            "forbidden_next_actions": config["decision"]["zero_discovery_candidates"]["forbid"],
            "conclusion": "No frozen single original-query feature passed the full discovery and old/new stability gate; by protocol, natural confirmation, combinations, interactions, and model training stop here.",
        }
        _write_json(run_dir / "decision.json", decision)
        _write_report(config, decision=decision, discovery_results=results)
    return report


def extract_natural(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    _assert_implementation_freeze(config)
    shortlist_path = run_dir / "single_feature_shortlist.json"
    if not shortlist_path.is_file():
        raise RuntimeError("Freeze the discovery shortlist first")
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    if not shortlist["features"]:
        return {"status": "skipped_by_zero_discovery_gate", "rows": 0}
    output = run_dir / "natural_features.npz"
    manifest_path = run_dir / "natural_features_manifest.json"
    if output.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        expected = {
            "protocol_id": config["protocol"]["id"],
            "matrix_sha256": _sha256(output),
            "shortlist_sha256": _sha256(shortlist_path),
            "corpus_cache_sha256": _sha256(_cache_path(config)),
            "implementation_hashes": frozen["implementation_hashes"],
        }
        actual = {name: manifest.get(name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"Refusing stale natural-feature reuse: expected={expected} actual={actual}")
        return manifest
    data = _natural_membership_and_features(config)
    lexical_names, dense_names = _schema_names(_input(config, "discovery_feature_schema"))
    _assert_implementation_freeze(config)
    cache = CorpusCache(_cache_path(config))
    started = time.perf_counter()
    try:
        matrix, specs = extract_feature_matrix(
            questions=data["questions"],
            legacy_lexical=data["legacy_lexical"],
            legacy_dense=data["legacy_dense"],
            legacy_lexical_names=lexical_names,
            dense_names=dense_names,
            cache=cache,
        )
    finally:
        cache.close()
    catalog = json.loads((run_dir / "feature_catalog.json").read_text(encoding="utf-8"))
    if catalog_sha256(specs) != catalog["catalog_sha256"]:
        raise ValueError("Natural feature catalog differs from frozen catalog")
    np.savez_compressed(
        output,
        query_ids=data["query_ids"],
        group_ids=data["group_ids"],
        matrix=matrix,
        feature_names=np.asarray([spec.name for spec in specs], dtype=np.str_),
    )
    result = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "rows": int(matrix.shape[0]),
        "features_extracted": int(matrix.shape[1]),
        "features_allowed_for_confirmation": len(shortlist["features"]),
        "shortlist_sha256": _sha256(shortlist_path),
        "matrix": str(output.relative_to(REPO_ROOT)),
        "matrix_sha256": _sha256(output),
        "elapsed_seconds": time.perf_counter() - started,
        "corpus_cache_sha256": _sha256(_cache_path(config)),
        "implementation_hashes": json.loads(
            (run_dir / "preflight.json").read_text(encoding="utf-8")
        )["implementation_hashes"],
        "official_final_holdout_rows": 0,
    }
    _write_json(manifest_path, result)
    return result


def analyze_natural(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    _assert_implementation_freeze(config)
    shortlist = json.loads((run_dir / "single_feature_shortlist.json").read_text(encoding="utf-8"))
    if not shortlist["features"]:
        return {"status": "skipped_by_zero_discovery_gate"}
    with np.load(run_dir / "natural_features.npz", allow_pickle=False) as stored:
        matrix = np.asarray(stored["matrix"], dtype=np.float64)
        feature_names = np.asarray(stored["feature_names"], dtype=np.str_)
        query_ids = np.asarray(stored["query_ids"], dtype=np.str_)
        group_ids = np.asarray(stored["group_ids"], dtype=np.str_)
    metrics_data = _natural_metrics(config, query_ids, group_ids)
    gaps = metrics_data["gaps"]
    labels_all = metrics_data["labels"]
    non_tie = labels_all >= 0
    labels = labels_all[non_tie]
    expected_counts = config["inputs"]["natural_metrics"]["expected_class_counts"]
    actual_counts = {
        "bm25": int(np.sum(labels_all == 1)),
        "dense": int(np.sum(labels_all == 0)),
        "tie": int(np.sum(labels_all == -1)),
    }
    if any(actual_counts[name] != int(expected_counts[name]) for name in actual_counts):
        raise ValueError(f"Natural class counts differ: {actual_counts} vs {dict(expected_counts)}")
    if int(np.sum(non_tie)) != int(config["statistics"]["natural_confirmation"]["expected_non_tie"]):
        raise ValueError("Natural non-tie row count differs")
    stats_config = config["statistics"]
    quality_config = stats_config["quality"]
    gate = stats_config["natural_confirmation"]
    inference = stats_config["inference"]
    catalog_data = json.loads((run_dir / "feature_catalog.json").read_text(encoding="utf-8"))
    specs = [FeatureSpec(**value) for value in catalog_data["features"]]
    name_to_index = {name: index for index, name in enumerate(feature_names.tolist())}
    results: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    directions: list[int] = []
    class_minimum = int(gate["minimum_winner_rows_per_class"])
    winner_class_quality = {
        "bm25": int(np.sum(labels == 1)),
        "dense": int(np.sum(labels == 0)),
        "minimum_required": class_minimum,
        "passed": int(np.sum(labels == 1)) >= class_minimum and int(np.sum(labels == 0)) >= class_minimum,
    }
    for frozen in shortlist["features"]:
        index = name_to_index[frozen["feature"]]
        spec = specs[index]
        gap_quality = _feature_quality(matrix[:, index], spec, quality_config, natural=True)
        winner_quality = _feature_quality(matrix[non_tie, index], spec, quality_config, natural=True)
        quality = {
            "passed": bool(gap_quality["passed"] and winner_quality["passed"] and winner_class_quality["passed"]),
            "failed_checks": (
                [f"gap_endpoint_{value}" for value in gap_quality["failed_checks"]]
                + [f"winner_endpoint_{value}" for value in winner_quality["failed_checks"]]
                + ([] if winner_class_quality["passed"] else ["winner_class_rows"])
            ),
            "missing_fraction": max(gap_quality["missing_fraction"], winner_quality["missing_fraction"]),
            "unique_values": min(gap_quality["unique_values"], winner_quality["unique_values"]),
            "maximum_single_value_fraction": max(
                gap_quality["maximum_single_value_fraction"], winner_quality["maximum_single_value_fraction"]
            ),
            "iqr": min(gap_quality["iqr"], winner_quality["iqr"]),
            "gap_endpoint": gap_quality,
            "winner_endpoint": winner_quality,
            "winner_classes": winner_class_quality,
        }
        metrics = _point_metrics(
            matrix[non_tie, index], labels, gaps[non_tie], spec, direction=int(frozen["direction"])
        )
        # Continuous gap endpoint uses all 2,000, including ties.
        all_gap_metrics = _point_metrics(
            matrix[:, index], np.where(gaps > 0, 1, 0).astype(np.int8), gaps, spec,
            direction=int(frozen["direction"]),
        )
        metrics["raw_spearman"] = all_gap_metrics["raw_spearman"]
        metrics["directional_spearman"] = all_gap_metrics["directional_spearman"]
        metrics["directional_gap_effect"] = all_gap_metrics["directional_gap_effect"]
        result = {
            "feature": spec.name,
            "family": spec.family,
            "kind": spec.kind,
            "quality": quality,
            "metrics": metrics,
            "status": "pending_inference" if quality["passed"] else "natural_invalid",
            "failed_checks": list(quality["failed_checks"]),
        }
        results.append(result)
        if quality["passed"]:
            selected_indices.append(index)
            directions.append(int(frozen["direction"]))
    # Winner and gap permutations use their appropriate populations.
    winner_p, _ = _permutation_pvalues(
        matrix=matrix[non_tie], feature_indices=selected_indices, specs=specs,
        directions=directions, labels=labels, gaps=gaps[non_tie], group_ids=group_ids[non_tie],
        resamples=int(inference["permutation_resamples"]), seed=int(inference["seed"]) + 20000,
        one_sided_frozen_direction=True,
    )
    dummy_labels = np.where(gaps > 0, 1, 0).astype(np.int8)
    _, gap_p = _permutation_pvalues(
        matrix=matrix, feature_indices=selected_indices, specs=specs,
        directions=directions, labels=dummy_labels, gaps=gaps, group_ids=group_ids,
        resamples=int(inference["permutation_resamples"]), seed=int(inference["seed"]) + 30000,
        one_sided_frozen_direction=True,
    )
    total_features = len(specs)
    all_winner = np.ones(total_features, dtype=np.float64)
    all_gap = np.ones(total_features, dtype=np.float64)
    for index, value in winner_p.items():
        all_winner[index] = value
    for index, value in gap_p.items():
        all_gap[index] = value
    winner_holm = _holm_adjust(all_winner)
    gap_holm = _holm_adjust(all_gap)
    for position, frozen in enumerate(shortlist["features"]):
        result = results[position]
        index = name_to_index[frozen["feature"]]
        metrics = result["metrics"]
        metrics.update({
            "winner_p_raw": float(all_winner[index]),
            "winner_p_holm": float(winner_holm[index]),
            "gap_p_raw": float(all_gap[index]),
            "gap_p_holm": float(gap_holm[index]),
            "auc_ci95": [float("nan"), float("nan")],
            "spearman_ci95": [float("nan"), float("nan")],
            "gap_effect_ci95": [float("nan"), float("nan")],
            "odds_ratio_ci95": [float("nan"), float("nan")],
        })
        if not result["quality"]["passed"]:
            continue
        intervals_winner = _bootstrap_candidate(
            values=matrix[non_tie, index], labels=labels, gaps=gaps[non_tie], group_ids=group_ids[non_tie],
            spec=specs[index], direction=int(frozen["direction"]),
            resamples=int(inference["bootstrap_resamples"]), seed=int(inference["seed"]) + 40000 + index,
        )
        intervals_gap = _bootstrap_candidate(
            values=matrix[:, index], labels=dummy_labels, gaps=gaps, group_ids=group_ids,
            spec=specs[index], direction=int(frozen["direction"]),
            resamples=int(inference["bootstrap_resamples"]), seed=int(inference["seed"]) + 50000 + index,
        )
        metrics["auc_ci95"] = intervals_winner["auc_ci95"]
        metrics["odds_ratio_ci95"] = intervals_winner["odds_ratio_ci95"]
        metrics["spearman_ci95"] = intervals_gap["spearman_ci95"]
        metrics["gap_effect_ci95"] = intervals_gap["gap_effect_ci95"]
        passed, failed = _ci_gate(metrics, specs[index], gate)
        result["status"] = "supported_single_feature" if passed else "natural_not_replicated"
        result["failed_checks"].extend(failed)

    supported = [result for result in results if result["status"] == "supported_single_feature"]
    report = {
        "protocol_id": config["protocol"]["id"],
        "status": "complete",
        "rows": int(len(gaps)),
        "non_tie_rows": int(np.sum(non_tie)),
        "bm25_winners": int(np.sum(labels == 1)),
        "dense_winners": int(np.sum(labels == 0)),
        "winner_class_quality": winner_class_quality,
        "frozen_features_tested": len(results),
        "supported_single_features": len(supported),
        "results": results,
    }
    _write_json(run_dir / "natural_single_feature_results.json", report)
    _write_csv(run_dir / "natural_single_feature_results.csv", [_flat_row(result) for result in results])
    discovery_report = json.loads((run_dir / "discovery_single_feature_results.json").read_text(encoding="utf-8"))
    if supported:
        decision_name = "SINGLE_FEATURE_GATE_PASS"
        conclusion = (
            f"{len(supported)} frozen single original-query feature(s) passed discovery, old/new stability, "
            "and natural confirmation. This permits a separately frozen combination/model diagnostic, but does not itself prove policy gain."
        )
        forbidden: list[str] = []
    else:
        decision_name = "STOP_NO_REPLICATED_SINGLE_FEATURE"
        conclusion = (
            "No frozen single original-query feature passed the complete discovery-to-natural-confirmation chain; "
            "by protocol, combinations, interactions, threshold retuning, model training, and later stages stop."
        )
        forbidden = list(config["decision"]["zero_natural_confirmed_features"]["forbid"])
    decision = {
        "protocol_id": config["protocol"]["id"],
        "decision": decision_name,
        "supported_single_features": [result["feature"] for result in supported],
        "natural_confirmation_run": True,
        "combination_analysis_run": False,
        "interaction_analysis_run": False,
        "model_training_run": False,
        "official_final_holdout_rows": 0,
        "forbidden_next_actions": forbidden,
        "conclusion": conclusion,
    }
    _write_json(run_dir / "decision.json", decision)
    _write_report(
        config, decision=decision, discovery_results=discovery_report["results"], natural_results=results
    )
    return report


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    files = [
        "preflight.json", "feature_catalog.json", "corpus_static_cache_manifest.json",
        "discovery_features_manifest.json", "discovery_single_feature_results.json",
        "single_feature_shortlist.json", "natural_features_manifest.json",
        "natural_single_feature_results.json", "decision.json", "REPORT.md",
    ]
    result: dict[str, Any] = {
        "protocol_id": config["protocol"]["id"],
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "artifacts": {name: (run_dir / name).is_file() for name in files},
    }
    if (run_dir / "decision.json").is_file():
        result["decision"] = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    return result


def run_all(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    preflight(config, config_path)
    build_cache_stage(config)
    extract_discovery(config)
    discovery = analyze_discovery(config)
    if discovery["features_advanced_to_natural"] == 0:
        return status(config)
    extract_natural(config)
    analyze_natural(config)
    return status(config)


def main() -> int:
    args = _arguments()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.stage == "preflight":
        result = preflight(config, config_path)
    elif args.stage == "build-corpus-cache":
        result = build_cache_stage(config)
    elif args.stage == "extract-discovery":
        result = extract_discovery(config)
    elif args.stage == "analyze-discovery":
        result = analyze_discovery(config)
    elif args.stage == "extract-natural":
        result = extract_natural(config)
    elif args.stage == "analyze-natural":
        result = analyze_natural(config)
    elif args.stage == "run":
        result = run_all(config, config_path)
    else:
        result = status(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
