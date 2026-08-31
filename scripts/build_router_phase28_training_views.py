#!/usr/bin/env python3
"""Build the frozen T0/T1/T2 Phase 2.8 router-training views.

This is deliberately a zero-provider-call step.  It combines the immutable
Phase 2.7 snapshot with completed Phase 2.8 BM25/Dense answer-F1 labels,
extracts only deployment-available query/static-corpus features for the new
queries, and emits Phase-2.7-runner-compatible frozen inputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHASE28_CONFIG = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase28/config.yaml"
PHASE27_TEMPLATE = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase27/config.yaml"
ROUTER_CONFIG = PROJECT_ROOT / "outputs/router/hotpotqa_bd_router_v1/config.yaml"
SOURCE_RUN_DIR = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1"
)
RUN_DIR = SOURCE_RUN_DIR
OLD_SNAPSHOT = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/"
    "phase27_model_audit_9600_v1/snapshot"
)
OLD_FEATURE_RUN = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1"
)
ACTIONS = ("bm25", "dense")
REPEATS = (0, 1, 2)
SUMMARY_FIELDS = [
    "query_id",
    "group_id",
    "question",
    "query_rank",
    "bm25_mean_f1",
    "bm25_mean_em",
    "bm25_evidence_page_recall",
    "bm25_retrieval_hit",
    "dense_mean_f1",
    "dense_mean_em",
    "dense_evidence_page_recall",
    "dense_retrieval_hit",
    "f1_winner",
    "f1_oracle",
    "f1_gap_bm25_minus_dense",
]
CANDIDATES = [
    "M3_pca32_structured_ridge",
    "M4_oof_late_fusion",
    "M4_pca32_oof_late_fusion",
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase28-config", default=str(PHASE28_CONFIG))
    parser.add_argument("--phase27-template", default=str(PHASE27_TEMPLATE))
    parser.add_argument("--output-dir", default=str(SOURCE_RUN_DIR))
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _winner(gap: float, tolerance: float) -> str:
    if gap > tolerance:
        return "bm25"
    if gap < -tolerance:
        return "dense"
    return "tie"


def _load_old_summary() -> list[dict[str, Any]]:
    path = OLD_SNAPSHOT / "phase27_bd_9600_query_summary.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 9600:
        raise ValueError("The immutable Phase 2.7 summary is not 9,600 rows")
    return rows


def _load_old_outcomes() -> dict[str, list[dict[str, Any]]]:
    path = OLD_SNAPSHOT / "phase27_bd_9600_outcomes.jsonl.gz"
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_query[str(row["query_id"])].append(row)
    if len(by_query) != 9600 or any(len(rows) != 6 for rows in by_query.values()):
        raise ValueError("The immutable Phase 2.7 outcomes are incomplete")
    return dict(by_query)


def _load_new_payloads() -> dict[str, list[dict[str, Any]]]:
    path = SOURCE_RUN_DIR / "expanded_outcomes.jsonl.gz"
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") == "success":
                by_query[str(payload["query_id"])].append(payload)
    return {
        query_id: values
        for query_id, values in by_query.items()
        if len(values) == len(ACTIONS) * len(REPEATS)
    }


def _load_new_ranks() -> dict[str, int]:
    path = SOURCE_RUN_DIR / "expanded_query_labels.jsonl.gz"
    result: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["query_id"])] = int(row["query_rank"])
    return result


def _project_outcome(row: Mapping[str, Any], *, query_rank: int) -> dict[str, Any]:
    metrics = row.get("metrics")
    retrieval = row.get("retrieval")
    context = row.get("context")
    generation = row.get("generation")
    if not all(isinstance(value, Mapping) for value in (metrics, retrieval, context, generation)):
        raise ValueError(f"Incomplete success payload for {row.get('query_id')}")
    usage = generation.get("token_usage")
    provider = usage.get("provider_reported") if isinstance(usage, Mapping) else None
    provider = provider if isinstance(provider, Mapping) else {}
    return {
        "query_id": str(row["query_id"]),
        "group_id": str(row["group_id"]),
        "split": "train",
        "action": str(row["action"]),
        "repeat_id": int(row["repeat_id"]),
        "status": "success",
        "question": str(row["question"]),
        "reference_answers": list(row["reference_answers"]),
        "supporting_titles": list(row["supporting_titles"]),
        "gold_doc_ids": list(row["gold_doc_ids"]),
        "prediction": str(row["prediction"]),
        "normalized_token_f1": float(metrics["normalized_token_f1"]),
        "normalized_exact_match": float(metrics["normalized_exact_match"]),
        "answer_correctness": row.get("answer_correctness"),
        "retrieved_doc_ids": list(retrieval["doc_ids"]),
        "retrieval_scores": [float(value) for value in retrieval["scores"]],
        "evidence_page_recall": float(retrieval["evidence_page_recall"]),
        "retrieval_hit": float(retrieval["hit"]),
        "retrieval_latency_ms": float(retrieval["latency_ms"]),
        "context_token_count": int(context["token_count"]),
        "context_truncated": bool(context["truncated"]),
        "generation_latency_ms": float(generation["latency_ms"]),
        "generation_input_tokens": (
            int(provider["input_tokens"])
            if isinstance(provider.get("input_tokens"), int)
            else None
        ),
        "generation_output_tokens": (
            int(provider["output_tokens"])
            if isinstance(provider.get("output_tokens"), int)
            else None
        ),
        "query_rank": int(query_rank),
    }


def _normalize_old_outcome(row: Mapping[str, Any], *, query_rank: int) -> dict[str, Any]:
    result = dict(row)
    result["query_rank"] = int(query_rank)
    result["split"] = "train"
    return result


def _summary(rows: Sequence[Mapping[str, Any]], *, tolerance: float) -> dict[str, Any]:
    if len(rows) != 6:
        raise ValueError("Every training query must have six outcome rows")
    by_action = {
        action: sorted(
            [row for row in rows if row["action"] == action],
            key=lambda row: int(row["repeat_id"]),
        )
        for action in ACTIONS
    }
    if any([int(row["repeat_id"]) for row in values] != list(REPEATS) for values in by_action.values()):
        raise ValueError("Every action must have repeat ids 0/1/2")
    means: dict[str, dict[str, float]] = {}
    for action, values in by_action.items():
        means[action] = {
            "f1": float(np.mean([float(row["normalized_token_f1"]) for row in values])),
            "em": float(np.mean([float(row["normalized_exact_match"]) for row in values])),
            "recall": float(np.mean([float(row["evidence_page_recall"]) for row in values])),
            "hit": float(np.mean([float(row["retrieval_hit"]) for row in values])),
        }
    first = rows[0]
    gap = means["bm25"]["f1"] - means["dense"]["f1"]
    return {
        "query_id": str(first["query_id"]),
        "group_id": str(first["group_id"]),
        "question": str(first["question"]),
        "query_rank": int(first["query_rank"]),
        "bm25_mean_f1": means["bm25"]["f1"],
        "bm25_mean_em": means["bm25"]["em"],
        "bm25_evidence_page_recall": means["bm25"]["recall"],
        "bm25_retrieval_hit": means["bm25"]["hit"],
        "dense_mean_f1": means["dense"]["f1"],
        "dense_mean_em": means["dense"]["em"],
        "dense_evidence_page_recall": means["dense"]["recall"],
        "dense_retrieval_hit": means["dense"]["hit"],
        "f1_winner": _winner(gap, tolerance),
        "f1_oracle": max(means["bm25"]["f1"], means["dense"]["f1"]),
        "f1_gap_bm25_minus_dense": gap,
    }


def _stable_order(query_ids: Sequence[str], seed: int) -> list[str]:
    def key(query_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
        return digest, query_id

    return sorted(query_ids, key=key)


def _select_views(
    config: Mapping[str, Any],
    old_summary: Sequence[Mapping[str, Any]],
    new_payloads: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[str]]:
    tolerance = float(config["labels"]["tie_tolerance"])
    old_by_winner: dict[str, list[str]] = defaultdict(list)
    for row in old_summary:
        old_by_winner[str(row["f1_winner"])].append(str(row["query_id"]))
    new_ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for query_id, payloads in new_payloads.items():
        projected = [_project_outcome(row, query_rank=0) for row in payloads]
        row = _summary(projected, tolerance=tolerance)
        new_rank = int(payloads[0].get("query_rank", 0))
        # The acquisition rank is not included in provider payloads.  Query ids
        # remain the deterministic secondary key; the primary rank is loaded below.
        new_ranked[row["f1_winner"]].append((new_rank, query_id))
    acquisition_rank = _load_new_ranks()
    new_by_winner = {
        winner: sorted(
            [query_id for _, query_id in values],
            key=lambda query_id: (acquisition_rank[query_id], query_id),
        )
        for winner, values in new_ranked.items()
    }
    tie_count = int(config["training_views"]["tie_count"])
    tie_seed = int(config["training_views"]["tie_seed"])
    tie_pool = np.asarray(old_by_winner["tie"], dtype=np.str_)
    if len(tie_pool) < tie_count:
        raise ValueError("The old snapshot does not contain 6,000 ties")
    rng = np.random.default_rng(tie_seed)
    tie_ids = sorted(rng.choice(tie_pool, tie_count, replace=False).tolist())

    result: dict[str, list[str]] = {}
    for checkpoint in config["training_views"]["checkpoints"]:
        selected: list[str] = list(tie_ids)
        for winner, field in (("bm25", "bm25_winner"), ("dense", "dense_winner")):
            target = int(checkpoint[field])
            existing = list(old_by_winner[winner])
            needed = target - len(existing)
            if needed < 0:
                existing = existing[:target]
                needed = 0
            available = new_by_winner.get(winner, [])
            if len(available) < needed:
                raise RuntimeError(
                    f"{checkpoint['id']} needs {needed} new {winner} winners; "
                    f"only {len(available)} are complete"
                )
            selected.extend(existing)
            selected.extend(available[:needed])
        expected = int(checkpoint["bm25_winner"]) + int(checkpoint["dense_winner"]) + tie_count
        if len(selected) != expected or len(set(selected)) != expected:
            raise RuntimeError(f"Invalid deterministic selection for {checkpoint['id']}")
        result[str(checkpoint["id"])] = _stable_order(selected, tie_seed)
    ordered_ids = [str(row["id"]) for row in config["training_views"]["checkpoints"]]
    for smaller, larger in zip(ordered_ids, ordered_ids[1:]):
        if not set(result[smaller]).issubset(result[larger]):
            raise RuntimeError("T0/T1/T2 query sets must be nested")
    return result


def _new_features(
    selected_new_ids: Sequence[str],
    new_payloads: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, np.ndarray]:
    path = SOURCE_RUN_DIR / "training_views/new_features_t2.npz"
    selected_new_ids = list(selected_new_ids)
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        if arrays["query_ids"].tolist() != selected_new_ids:
            raise ValueError("Stored new-feature query order differs from frozen T2 selection")
        return arrays

    from src.router_experiments.runtime import core_config, load_router_config
    from scripts.run_router_phase3_train import _dense_summary, _lexical_features
    from src.embedders.text_embedder import create_embedder
    from src.retrievers.sqlite_bm25 import (
        analyze_sqlite_bm25_text,
        read_sqlite_bm25_term_stats,
    )

    router = load_router_config(ROUTER_CONFIG)
    questions = [str(new_payloads[query_id][0]["question"]) for query_id in selected_new_ids]
    group_ids = [str(new_payloads[query_id][0]["group_id"]) for query_id in selected_new_ids]
    all_terms = {
        token for question in questions for token in analyze_sqlite_bm25_text(question)
    }
    bm25_database = _resolve(str(router["artifacts"]["bm25_index"])) / "index.sqlite3"
    stats = read_sqlite_bm25_term_stats(bm25_database, all_terms)
    lexical = np.asarray(
        [_lexical_features(question, stats)[0] for question in questions],
        dtype=np.float32,
    )
    embedder = create_embedder(core_config(router, method="dense"), role="query")
    embedding = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    prototypes = np.asarray(
        np.load(OLD_FEATURE_RUN / "features/corpus_prototypes.npy", allow_pickle=False),
        dtype=np.float32,
    )
    dense, _ = _dense_summary(embedding, prototypes)
    arrays = {
        "query_ids": np.asarray(selected_new_ids, dtype=np.str_),
        "group_ids": np.asarray(group_ids, dtype=np.str_),
        "lexical": lexical,
        "dense": dense,
        "embedding": embedding,
    }
    if lexical.shape != (len(selected_new_ids), 17) or dense.shape != (len(selected_new_ids), 13):
        raise ValueError("New structured features have unexpected dimensions")
    if embedding.shape != (len(selected_new_ids), 384):
        raise ValueError("New BGE query embeddings have unexpected dimensions")
    if not all(np.isfinite(value).all() for value in (lexical, dense, embedding)):
        raise ValueError("New features contain non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    _write_json(
        path.with_suffix(".manifest.json"),
        {
            "query_count": len(selected_new_ids),
            "query_selection": "T2_new_winners_only_deterministic_acquisition_order",
            "information_boundary": "query_and_corpus_static_only",
            "features_sha256": _sha256(path),
            "fresh_dev_rows_read": 0,
            "final_holdout_rows_read": 0,
            "external_provider_calls": 0,
        },
    )
    return arrays


def _feature_maps() -> tuple[dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    path = OLD_SNAPSHOT / "phase27_features_9600.npz"
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    mapping = {
        str(query_id): (
            str(arrays["group_ids"][index]),
            np.asarray(arrays["lexical"][index]),
            np.asarray(arrays["dense"][index]),
            np.asarray(arrays["embedding"][index]),
        )
        for index, query_id in enumerate(arrays["query_ids"])
    }
    return mapping, arrays


def _snapshot_config(
    template: Mapping[str, Any],
    *,
    view_id: str,
    query_count: int,
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    protocol_id = f"hotpotqa_bd_router_phase28_{view_id.lower()}_retrain_v1"
    config["protocol"].update(
        {
            "id": protocol_id,
            "status": "zero_call_execution_active",
            "frozen_date": "2026-08-28",
            "external_calls_allowed": 0,
        }
    )
    config["scope"].update(
        {
            "query_count": int(query_count),
            "research_question": (
                "Do M3, M4, or M4-PCA improve when BM25/Dense answer-F1 "
                "winner counts are expanded without sample weights?"
            ),
        }
    )
    config["frozen_inputs"] = {
        "outcomes": {
            "path": _relative(artifacts["outcomes"]),
            "sha256": _sha256(artifacts["outcomes"]),
            "expected_rows": int(query_count) * 6,
        },
        "query_summary": {
            "path": _relative(artifacts["summary"]),
            "sha256": _sha256(artifacts["summary"]),
            "expected_rows": int(query_count),
        },
        "features": {
            "path": _relative(artifacts["features"]),
            "sha256": _sha256(artifacts["features"]),
            "expected_rows": int(query_count),
        },
        "feature_schema": {
            "path": _relative(artifacts["schema"]),
            "sha256": _sha256(artifacts["schema"]),
        },
        "snapshot_manifest": {
            "path": _relative(artifacts["manifest"]),
            "sha256": _sha256(artifacts["manifest"]),
        },
    }
    config["candidates"] = [
        {"id": "M3_pca32_structured_ridge"},
        {"id": "M4_oof_late_fusion"},
        {"id": "M4_pca32_oof_late_fusion"},
    ]
    config["learning_curve"]["query_counts"] = [int(query_count)]
    audit_dir = RUN_DIR / f"training_views/{view_id}/audit"
    config["planned_implementation"]["output_dir"] = _relative(audit_dir)
    config["planned_implementation"]["required_outputs"] = [
        "preflight.json",
        "split_manifest.json",
        "candidate_freeze.json",
        "formal_metrics.json",
        "formal_predictions.npz",
    ]
    return config


def main() -> int:
    global RUN_DIR
    args = _arguments()
    RUN_DIR = _resolve(args.output_dir)
    phase28 = _load_yaml(_resolve(args.phase28_config))
    template = _load_yaml(_resolve(args.phase27_template))
    status_path = SOURCE_RUN_DIR / "export_manifest.json"
    if not status_path.is_file():
        raise FileNotFoundError("Run Phase 2.8 --stage export after meeting the label quotas")
    export = _read_json(status_path)
    if not export.get("status", {}).get("all_quotas_met"):
        raise RuntimeError("Phase 2.8 winner/high-margin quotas are not complete")

    old_summary = _load_old_summary()
    old_outcomes = _load_old_outcomes()
    new_payloads = _load_new_payloads()
    views = _select_views(phase28, old_summary, new_payloads)
    old_query_ids = {str(row["query_id"]) for row in old_summary}
    t2_id = str(phase28["training_views"]["checkpoints"][-1]["id"])
    selected_new_ids = [query_id for query_id in views[t2_id] if query_id not in old_query_ids]
    new_arrays = _new_features(selected_new_ids, new_payloads)
    old_feature_map, _ = _feature_maps()
    new_feature_map = {
        str(query_id): (
            str(new_arrays["group_ids"][index]),
            np.asarray(new_arrays["lexical"][index]),
            np.asarray(new_arrays["dense"][index]),
            np.asarray(new_arrays["embedding"][index]),
        )
        for index, query_id in enumerate(new_arrays["query_ids"])
    }
    feature_map = {**old_feature_map, **new_feature_map}

    tolerance = float(phase28["labels"]["tie_tolerance"])
    view_manifests: dict[str, Any] = {}
    checkpoint_by_id = {
        str(row["id"]): row for row in phase28["training_views"]["checkpoints"]
    }
    for view_id, query_ids in views.items():
        snapshot_dir = RUN_DIR / f"training_views/{view_id}/snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        outcomes_path = snapshot_dir / "outcomes.jsonl.gz"
        summary_path = snapshot_dir / "query_summary.csv.gz"
        features_path = snapshot_dir / "features.npz"
        schema_path = snapshot_dir / "feature_schema.json"
        manifest_path = snapshot_dir / "snapshot_manifest.json"

        summaries: list[dict[str, Any]] = []
        projected_by_query: dict[str, list[dict[str, Any]]] = {}
        for rank, query_id in enumerate(query_ids):
            if query_id in old_outcomes:
                projected = [
                    _normalize_old_outcome(row, query_rank=rank)
                    for row in old_outcomes[query_id]
                ]
            else:
                projected = [
                    _project_outcome(row, query_rank=rank)
                    for row in new_payloads[query_id]
                ]
            projected.sort(key=lambda row: (ACTIONS.index(str(row["action"])), int(row["repeat_id"])))
            projected_by_query[query_id] = projected
            summaries.append(_summary(projected, tolerance=tolerance))

        with gzip.open(outcomes_path, "wt", encoding="utf-8", newline="\n") as handle:
            for query_id in query_ids:
                for row in projected_by_query[query_id]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with gzip.open(summary_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summaries)

        missing_features = [query_id for query_id in query_ids if query_id not in feature_map]
        if missing_features:
            raise KeyError(f"Missing features for {len(missing_features)} selected queries")
        np.savez_compressed(
            features_path,
            query_ids=np.asarray(query_ids, dtype=np.str_),
            group_ids=np.asarray([feature_map[q][0] for q in query_ids], dtype=np.str_),
            lexical=np.asarray([feature_map[q][1] for q in query_ids], dtype=np.float32),
            dense=np.asarray([feature_map[q][2] for q in query_ids], dtype=np.float32),
            embedding=np.asarray([feature_map[q][3] for q in query_ids], dtype=np.float32),
        )
        shutil.copyfile(OLD_SNAPSHOT / "feature_schema.json", schema_path)

        observed_counts = {
            "bm25_winner": sum(row["f1_winner"] == "bm25" for row in summaries),
            "dense_winner": sum(row["f1_winner"] == "dense" for row in summaries),
            "exact_tie": sum(row["f1_winner"] == "tie" for row in summaries),
        }
        checkpoint = checkpoint_by_id[view_id]
        expected_counts = {
            "bm25_winner": int(checkpoint["bm25_winner"]),
            "dense_winner": int(checkpoint["dense_winner"]),
            "exact_tie": int(phase28["training_views"]["tie_count"]),
        }
        if observed_counts != expected_counts:
            raise RuntimeError(
                f"{view_id} class counts differ: {observed_counts} != {expected_counts}"
            )
        artifacts = [outcomes_path, summary_path, features_path, schema_path]
        manifest = {
            "protocol_id": f"hotpotqa_bd_router_phase28_{view_id.lower()}_snapshot_v1",
            "status": "complete",
            "view_id": view_id,
            "query_count": len(query_ids),
            "outcome_rows": len(query_ids) * 6,
            "class_counts": observed_counts,
            "sample_weight": "none",
            "tie_source": "phase27_existing_9600_only",
            "selection_rule": (
                "all_old_winners_then_new_winners_by_frozen_acquisition_rank; "
                "old_ties_seeded_without_replacement"
            ),
            "nested_view_query_ids": True,
            "fresh_dev_rows_read": 0,
            "final_holdout_rows_read": 0,
            "external_provider_calls": 0,
            "artifacts": {
                path.name: {
                    "path": _relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifacts
            },
            "completed_at_unix": time.time(),
        }
        _write_json(manifest_path, manifest)
        paths = {
            "outcomes": outcomes_path,
            "summary": summary_path,
            "features": features_path,
            "schema": schema_path,
            "manifest": manifest_path,
        }
        training_config = _snapshot_config(
            template,
            view_id=view_id,
            query_count=len(query_ids),
            artifacts=paths,
        )
        config_path = RUN_DIR / f"training_views/{view_id}/config.yaml"
        config_path.write_text(
            yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        from src.router_experiments.modeling import candidate_by_id

        audit_dir = RUN_DIR / f"training_views/{view_id}/audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            audit_dir / "candidate_freeze.json",
            {
                "protocol_id": training_config["protocol"]["id"],
                "status": "complete",
                "formal_candidates": CANDIDATES,
                "candidate_specs": {
                candidate: candidate_by_id(candidate).__dict__
                    for candidate in CANDIDATES
                },
                "selection_rule": "three_candidates_frozen_before_expanded_outcomes_were_generated",
                "sample_weight": "none",
                "external_calls": 0,
                "frozen_at_unix": time.time(),
            },
        )
        view_manifests[view_id] = {
            "config": _relative(config_path),
            "config_sha256": _sha256(config_path),
            "snapshot": _relative(manifest_path),
            "snapshot_sha256": _sha256(manifest_path),
            "query_count": len(query_ids),
            "class_counts": observed_counts,
        }

    result = {
        "protocol_id": phase28["protocol"]["id"],
        "status": "complete",
        "views": view_manifests,
        "new_feature_queries": len(selected_new_ids),
        "candidates": CANDIDATES,
        "sample_weight": "none",
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "external_provider_calls": 0,
    }
    _write_json(RUN_DIR / "training_views/manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
