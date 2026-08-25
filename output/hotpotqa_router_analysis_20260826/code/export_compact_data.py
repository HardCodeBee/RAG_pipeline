from __future__ import annotations

import csv
import gzip
import io
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EPS = 1e-12


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gzip_text_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, compressed, io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    raw, compressed, text = _gzip_text_writer(path)
    count = 0
    try:
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            text.write("\n")
            count += 1
    finally:
        text.close()
        compressed.close()
        raw.close()
    return count


def _write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise ValueError(f"No rows for {path}")
    raw, compressed, text = _gzip_text_writer(path)
    try:
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        text.close()
        compressed.close()
        raw.close()
    return len(rows)


def _token_count(generation: dict[str, Any], field: str) -> int | None:
    usage = generation.get("token_usage") or {}
    provider = usage.get("provider_reported") or {}
    estimated = usage.get("estimated") or {}
    value = provider.get(field, estimated.get(field))
    return None if value is None else int(value)


def _compact_payload(
    payload: dict[str, Any],
    *,
    query_rank: int | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    retrieval = payload.get("retrieval") or {}
    context = payload.get("context") or {}
    metrics = payload.get("metrics") or {}
    generation = payload.get("generation") or {}
    row = {
        "query_id": str(payload["query_id"]),
        "group_id": str(group_id or payload.get("group_id", "")),
        "split": str(payload.get("split", "")),
        "action": str(payload["action"]),
        "repeat_id": int(payload.get("repeat_id", 0)),
        "status": str(payload.get("status", "")),
        "question": str(payload.get("question", "")),
        "reference_answers": payload.get("reference_answers") or [],
        "supporting_titles": payload.get("supporting_titles") or [],
        "gold_doc_ids": payload.get("gold_doc_ids") or [],
        "prediction": str(payload.get("prediction", "")),
        "normalized_token_f1": float(metrics["normalized_token_f1"]),
        "normalized_exact_match": float(metrics["normalized_exact_match"]),
        "answer_correctness": (
            None if metrics.get("answer_correctness") is None else float(metrics["answer_correctness"])
        ),
        "retrieved_doc_ids": retrieval.get("doc_ids") or [],
        "retrieval_scores": retrieval.get("scores") or [],
        "evidence_page_recall": float(retrieval.get("evidence_page_recall", 0.0)),
        "retrieval_hit": float(retrieval.get("hit", 0.0)),
        "retrieval_latency_ms": float(retrieval.get("latency_ms", 0.0)),
        "context_token_count": int(context.get("token_count", 0)),
        "context_truncated": bool(context.get("truncated", False)),
        "generation_latency_ms": float(generation.get("latency_ms", 0.0)),
        "generation_input_tokens": _token_count(generation, "input_tokens"),
        "generation_output_tokens": _token_count(generation, "output_tokens"),
    }
    if query_rank is not None:
        row["query_rank"] = int(query_rank)
    return row


def _winner(means: dict[str, float]) -> str:
    best = max(means.values())
    winners = [name for name, value in means.items() if abs(value - best) <= EPS]
    return winners[0] if len(winners) == 1 else "tie"


def _summarize_queries(
    rows: list[dict[str, Any]],
    *,
    actions: tuple[str, ...],
    include_split: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split", ""), row["query_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (split, query_id), items in grouped.items():
        by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_action[item["action"]].append(item)
        if set(by_action) != set(actions):
            raise ValueError(f"Incomplete actions for {query_id}: {sorted(by_action)}")
        for action in actions:
            if sorted(row["repeat_id"] for row in by_action[action]) != [0, 1, 2]:
                raise ValueError(f"Incomplete repeats for {query_id}/{action}")

        first = items[0]
        record: dict[str, Any] = {
            "query_id": query_id,
            "group_id": first["group_id"],
            "question": first["question"],
        }
        if "query_rank" in first:
            record["query_rank"] = min(int(item["query_rank"]) for item in items)
        if include_split:
            record["split"] = split

        f1_means: dict[str, float] = {}
        em_means: dict[str, float] = {}
        ac_means: dict[str, float] = {}
        for action in actions:
            action_rows = by_action[action]
            f1_means[action] = float(np.mean([row["normalized_token_f1"] for row in action_rows]))
            em_means[action] = float(np.mean([row["normalized_exact_match"] for row in action_rows]))
            ac_values = [row["answer_correctness"] for row in action_rows]
            if all(value is not None for value in ac_values):
                ac_means[action] = float(np.mean(ac_values))
            record[f"{action}_mean_f1"] = f1_means[action]
            record[f"{action}_mean_em"] = em_means[action]
            record[f"{action}_evidence_page_recall"] = float(
                np.mean([row["evidence_page_recall"] for row in action_rows])
            )
            record[f"{action}_retrieval_hit"] = float(
                np.mean([row["retrieval_hit"] for row in action_rows])
            )
            if action in ac_means:
                record[f"{action}_mean_ac"] = ac_means[action]

        record["f1_winner"] = _winner(f1_means)
        record["f1_oracle"] = max(f1_means.values())
        if len(actions) == 2:
            record[f"f1_gap_{actions[0]}_minus_{actions[1]}"] = (
                f1_means[actions[0]] - f1_means[actions[1]]
            )
        if ac_means:
            record["ac_winner"] = _winner(ac_means)
            record["ac_oracle"] = max(ac_means.values())
        summaries.append(record)

    summaries.sort(key=lambda row: (row.get("split", ""), row.get("query_rank", 0), row["query_id"]))
    aggregate: dict[str, Any] = {
        "queries": len(summaries),
        "actions": list(actions),
        "fixed_mean_f1": {
            action: float(np.mean([row[f"{action}_mean_f1"] for row in summaries]))
            for action in actions
        },
        "query_oracle_mean_f1": float(np.mean([row["f1_oracle"] for row in summaries])),
        "f1_winner_counts": {
            label: sum(row["f1_winner"] == label for row in summaries)
            for label in (*actions, "tie")
        },
    }
    if all(f"{action}_mean_ac" in summaries[0] for action in actions):
        aggregate["fixed_mean_ac"] = {
            action: float(np.mean([row[f"{action}_mean_ac"] for row in summaries]))
            for action in actions
        }
        aggregate["query_oracle_mean_ac"] = float(np.mean([row["ac_oracle"] for row in summaries]))
        aggregate["ac_winner_counts"] = {
            label: sum(row["ac_winner"] == label for row in summaries)
            for label in (*actions, "tie")
        }
    return summaries, aggregate


def export_phase2() -> dict[str, Any]:
    source = REPO_ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase2_bd_headroom_repeated_v1/results.jsonl"
    rows = [_compact_payload(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 3600:
        raise ValueError(f"Expected 3,600 Phase 2 rows, found {len(rows)}")
    output = PACKAGE_ROOT / "data/phase2/phase2_bd_600_outcomes.jsonl.gz"
    written = _write_jsonl_gz(output, rows)
    summaries, aggregate = _summarize_queries(rows, actions=("bm25", "dense"), include_split=True)
    _write_csv_gz(PACKAGE_ROOT / "data/phase2/phase2_bd_600_query_summary.csv.gz", summaries)
    return {"outcome_rows": written, "query_rows": len(summaries), "aggregate": aggregate}


def _read_phase26_rows() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    source = REPO_ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1/state.sqlite3"
    connection = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    try:
        samples = connection.execute(
            """
            SELECT query_rank, query_id, group_id
            FROM sample_rows
            WHERE partition='train'
            ORDER BY query_rank
            LIMIT 4800
            """
        ).fetchall()
        if len(samples) != 4800:
            raise ValueError(f"Expected 4,800 frozen samples, found {len(samples)}")
        min_rank = int(samples[0][0])
        max_rank = int(samples[-1][0])
        records = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id, o.action, o.repeat_id, o.status, o.payload
            FROM sample_rows AS s
            JOIN outcomes AS o ON o.query_id=s.query_id AND o.partition=s.partition
            WHERE s.partition='train' AND s.query_rank BETWEEN ? AND ?
              AND o.action IN ('bm25','dense')
            ORDER BY s.query_rank, o.action, o.repeat_id
            """,
            (min_rank, max_rank),
        ).fetchall()
    finally:
        connection.close()

    rows: list[dict[str, Any]] = []
    for rank, query_id, group_id, action, repeat_id, status, payload_text in records:
        if status != "success":
            raise ValueError(f"Frozen Phase 2.6 row is not successful: {query_id}/{action}/{repeat_id}")
        payload = json.loads(payload_text)
        payload["action"] = action
        payload["repeat_id"] = repeat_id
        rows.append(_compact_payload(payload, query_rank=int(rank), group_id=str(group_id)))
    if len(rows) != 28800:
        raise ValueError(f"Expected 28,800 Phase 2.6 rows, found {len(rows)}")
    query_ids = [str(row[1]) for row in samples]
    group_ids = [str(row[2]) for row in samples]
    return rows, query_ids, group_ids


def export_phase26() -> dict[str, Any]:
    rows, query_ids, group_ids = _read_phase26_rows()
    written = _write_jsonl_gz(
        PACKAGE_ROOT / "data/phase26/phase26_bd_4800_outcomes.jsonl.gz", rows
    )
    summaries, aggregate = _summarize_queries(rows, actions=("bm25", "dense"), include_split=False)
    _write_csv_gz(PACKAGE_ROOT / "data/phase26/phase26_bd_4800_query_summary.csv.gz", summaries)

    feature_source = REPO_ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1/features/train.npz"
    with np.load(feature_source, allow_pickle=False) as source:
        if source["query_ids"][:4800].tolist() != query_ids:
            raise ValueError("Phase 2.6 feature/query order mismatch")
        if source["group_ids"][:4800].tolist() != group_ids:
            raise ValueError("Phase 2.6 feature/group order mismatch")
        np.savez_compressed(
            PACKAGE_ROOT / "features/phase26/phase26_features_4800.npz",
            query_ids=source["query_ids"][:4800],
            group_ids=source["group_ids"][:4800],
            lexical=source["lexical"][:4800],
            dense=source["dense"][:4800],
            embedding=source["embedding"][:4800],
        )
    return {"outcome_rows": written, "query_rows": len(summaries), "aggregate": aggregate}


def export_contriever() -> dict[str, Any]:
    source = REPO_ROOT / "outputs/router/hotpotqa_bc_router_v1/runs/phase3_contriever_f1_v1/state.sqlite3"
    connection = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    try:
        samples = connection.execute(
            "SELECT query_rank, query_id, group_id FROM sample_rows ORDER BY query_rank"
        ).fetchall()
        records = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id, o.action, o.repeat_id, o.status, o.payload
            FROM sample_rows AS s JOIN outcomes AS o ON o.query_id=s.query_id
            ORDER BY s.query_rank, o.action, o.repeat_id
            """
        ).fetchall()
    finally:
        connection.close()
    if len(samples) != 1200 or len(records) != 10800:
        raise ValueError(f"Unexpected Contriever counts: samples={len(samples)}, outcomes={len(records)}")

    rows: list[dict[str, Any]] = []
    for rank, query_id, group_id, action, repeat_id, status, payload_text in records:
        if status != "success":
            raise ValueError(f"Contriever row is not successful: {query_id}/{action}/{repeat_id}")
        payload = json.loads(payload_text)
        payload["action"] = action
        payload["repeat_id"] = repeat_id
        rows.append(_compact_payload(payload, query_rank=int(rank), group_id=str(group_id)))
    written = _write_jsonl_gz(
        PACKAGE_ROOT / "data/contriever/phase3_contriever_1200_outcomes.jsonl.gz", rows
    )
    summaries, aggregate = _summarize_queries(
        rows, actions=("bm25", "bge", "contriever"), include_split=False
    )
    _write_csv_gz(
        PACKAGE_ROOT / "data/contriever/phase3_contriever_1200_query_summary.csv.gz", summaries
    )

    feature_source = REPO_ROOT / "outputs/router/hotpotqa_bc_router_v1/runs/phase3_contriever_f1_v1/features.npz"
    query_ids = np.asarray([str(row[1]) for row in samples])
    group_ids = np.asarray([str(row[2]) for row in samples])
    with np.load(feature_source, allow_pickle=False) as source_features:
        if any(source_features[name].shape[0] != 1200 for name in source_features.files):
            raise ValueError("Contriever feature row count mismatch")
        np.savez_compressed(
            PACKAGE_ROOT / "features/contriever/contriever_features_1200.npz",
            query_ids=query_ids,
            group_ids=group_ids,
            lexical=source_features["lexical"],
            dense=source_features["dense"],
            embedding=source_features["embedding"],
        )
    return {"outcome_rows": written, "query_rows": len(summaries), "aggregate": aggregate}


def main() -> None:
    derived = {
        "phase2_repeated": export_phase2(),
        "phase26_bd": export_phase26(),
        "phase3_contriever": export_contriever(),
    }
    _json_dump(PACKAGE_ROOT / "data/derived_metrics.json", derived)
    validation = {
        "status": "passed",
        "phase2_outcome_rows": derived["phase2_repeated"]["outcome_rows"],
        "phase26_outcome_rows": derived["phase26_bd"]["outcome_rows"],
        "contriever_outcome_rows": derived["phase3_contriever"]["outcome_rows"],
        "fresh_dev_rows_exported": 0,
        "final_holdout_rows_exported": 0,
        "context_text_exported": False,
        "provider_payloads_exported": False,
        "api_credentials_exported": False,
    }
    _json_dump(PACKAGE_ROOT / "data/export_validation.json", validation)


if __name__ == "__main__":
    main()
