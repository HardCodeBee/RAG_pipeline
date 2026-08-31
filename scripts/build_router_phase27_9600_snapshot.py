"""Build and validate the immutable 9,600-query B/D snapshot after authorized completion."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_LIMIT = 9600
ACTIONS = ("bm25", "dense")
REPEATS = (0, 1, 2)
SOURCE_RUN = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1"
)
PRIOR_ROOT = PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase00_26/results"
OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/snapshot"
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_outcome(
    rank: int, query_id: str, group_id: str, payload_text: str
) -> dict[str, Any]:
    row = json.loads(payload_text)
    metrics = row.get("metrics")
    retrieval = row.get("retrieval")
    context = row.get("context")
    generation = row.get("generation")
    if not all(isinstance(value, Mapping) for value in (metrics, retrieval, context, generation)):
        raise ValueError(f"Incomplete success payload at query rank {rank}")
    token_usage = generation.get("token_usage")
    provider = token_usage.get("provider_reported") if isinstance(token_usage, Mapping) else None
    provider = provider if isinstance(provider, Mapping) else {}
    projected = {
        "query_id": str(query_id),
        "group_id": str(group_id),
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
        "answer_correctness": None,
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
        "query_rank": int(rank),
    }
    if not 0.0 <= projected["normalized_token_f1"] <= 1.0:
        raise ValueError("F1 is outside [0, 1]")
    if not projected["prediction"].strip():
        raise ValueError("A success payload has an empty prediction")
    return projected


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    result: list[dict[str, Any]] = []
    for query_rows in grouped.values():
        query_rows.sort(key=lambda row: (ACTIONS.index(row["action"]), row["repeat_id"]))
        if len(query_rows) != len(ACTIONS) * len(REPEATS):
            raise ValueError("A query does not have six B/D repeat rows")
        by_action = {
            action: [row for row in query_rows if row["action"] == action]
            for action in ACTIONS
        }
        if any(sorted(row["repeat_id"] for row in values) != list(REPEATS) for values in by_action.values()):
            raise ValueError("A query/action does not have repeats 0/1/2")
        mean_f1 = {
            action: float(np.mean([row["normalized_token_f1"] for row in values]))
            for action, values in by_action.items()
        }
        mean_em = {
            action: float(np.mean([row["normalized_exact_match"] for row in values]))
            for action, values in by_action.items()
        }
        mean_recall = {
            action: float(np.mean([row["evidence_page_recall"] for row in values]))
            for action, values in by_action.items()
        }
        mean_hit = {
            action: float(np.mean([row["retrieval_hit"] for row in values]))
            for action, values in by_action.items()
        }
        gap = mean_f1["bm25"] - mean_f1["dense"]
        winner = "tie" if math.isclose(gap, 0.0, abs_tol=1e-12) else ("bm25" if gap > 0 else "dense")
        first = query_rows[0]
        result.append(
            {
                "query_id": first["query_id"],
                "group_id": first["group_id"],
                "question": first["question"],
                "query_rank": first["query_rank"],
                "bm25_mean_f1": mean_f1["bm25"],
                "bm25_mean_em": mean_em["bm25"],
                "bm25_evidence_page_recall": mean_recall["bm25"],
                "bm25_retrieval_hit": mean_hit["bm25"],
                "dense_mean_f1": mean_f1["dense"],
                "dense_mean_em": mean_em["dense"],
                "dense_evidence_page_recall": mean_recall["dense"],
                "dense_retrieval_hit": mean_hit["dense"],
                "f1_winner": winner,
                "f1_oracle": max(mean_f1.values()),
                "f1_gap_bm25_minus_dense": gap,
            }
        )
    result.sort(key=lambda row: int(row["query_rank"]))
    if [int(row["query_rank"]) for row in result] != list(range(QUERY_LIMIT)):
        raise ValueError("Query ranks are not contiguous from zero")
    return result


def _prior_outcomes() -> list[dict[str, Any]]:
    path = PRIOR_ROOT / "phase26/phase26_bd_4800_outcomes.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prior_summary() -> list[dict[str, str]]:
    path = PRIOR_ROOT / "phase26/phase26_bd_4800_query_summary.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    database = SOURCE_RUN / "state.sqlite3"
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    try:
        source_rows = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id, o.status, o.payload
            FROM sample_rows AS s
            JOIN outcomes AS o ON o.query_id = s.query_id
            WHERE s.partition = 'train'
              AND s.query_rank < ?
              AND o.action IN ('bm25', 'dense')
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'bm25' THEN 0 ELSE 1 END,
                     o.repeat_id
            """,
            (QUERY_LIMIT,),
        ).fetchall()
        non_train = int(
            connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE partition != 'train'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    expected = QUERY_LIMIT * len(ACTIONS) * len(REPEATS)
    if len(source_rows) != expected or non_train != 0:
        raise ValueError("The 9,600 source is incomplete or has opened a sealed partition")
    outcomes: list[dict[str, Any]] = []
    for rank, query_id, group_id, status_name, payload in source_rows:
        if status_name != "success":
            raise ValueError("The 9,600 source contains a non-success outcome")
        outcomes.append(_project_outcome(int(rank), str(query_id), str(group_id), str(payload)))
    summaries = _summary(outcomes)

    prior_outcomes = _prior_outcomes()
    prefix_outcomes = [row for row in outcomes if int(row["query_rank"]) < 4800]
    outcomes_prefix_equal = prefix_outcomes == prior_outcomes
    if not outcomes_prefix_equal:
        raise ValueError("The 9,600 outcome prefix differs from the frozen 4,800 snapshot")
    prior_summary = _prior_summary()
    summary_prefix_equal = all(
        {key: str(value) for key, value in row.items()} == prior
        for row, prior in zip(summaries[:4800], prior_summary, strict=True)
    )
    if not summary_prefix_equal:
        raise ValueError("The 9,600 summary prefix differs from the frozen 4,800 snapshot")

    source_features = SOURCE_RUN / "features/train.npz"
    prior_features = PRIOR_ROOT / "phase26/features/phase26_features_4800.npz"
    with np.load(source_features, allow_pickle=False) as source, np.load(
        prior_features, allow_pickle=False
    ) as prior:
        arrays = {name: np.asarray(source[name]) for name in source.files}
        features_prefix_equal = all(
            name in prior.files and np.array_equal(arrays[name][:4800], prior[name])
            for name in arrays
        )
    if not features_prefix_equal or any(len(value) != QUERY_LIMIT for value in arrays.values()):
        raise ValueError("The 9,600 features are incomplete or differ in the frozen prefix")
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("The 9,600 features contain non-finite values")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes_path = OUTPUT_ROOT / "phase27_bd_9600_outcomes.jsonl.gz"
    with gzip.open(outcomes_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in outcomes:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path = OUTPUT_ROOT / "phase27_bd_9600_query_summary.csv.gz"
    with gzip.open(summary_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    features_path = OUTPUT_ROOT / "phase27_features_9600.npz"
    np.savez_compressed(features_path, **arrays)
    schema_path = OUTPUT_ROOT / "feature_schema.json"
    shutil.copyfile(SOURCE_RUN / "features/feature_schema.json", schema_path)

    paths = [outcomes_path, summary_path, features_path, schema_path]
    manifest = {
        "protocol_id": "hotpotqa_bd_router_phase27_9600_snapshot_v1",
        "status": "complete",
        "query_count": QUERY_LIMIT,
        "outcome_rows": len(outcomes),
        "actions": list(ACTIONS),
        "repeat_ids": list(REPEATS),
        "partition": "train_only",
        "fresh_dev_rows": 0,
        "final_holdout_rows": 0,
        "prefix_validation": {
            "outcomes_first_4800_equal": outcomes_prefix_equal,
            "summary_first_4800_equal": summary_prefix_equal,
            "features_first_4800_equal": features_prefix_equal,
        },
        "artifacts": {
            path.name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        },
        "source_database": {
            "path": str(database.relative_to(PROJECT_ROOT)),
            "bytes": database.stat().st_size,
            "sha256": _sha256(database),
        },
        "external_calls_during_snapshot_build": 0,
        "completed_at_unix": time.time(),
    }
    manifest_path = OUTPUT_ROOT / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
