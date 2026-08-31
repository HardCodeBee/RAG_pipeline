#!/usr/bin/env python3
"""Acquire additional HotpotQA BM25/Dense winners for the Phase 2.8 audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_experiments.runtime import (  # noqa: E402
    GENERATOR_PRICES,
    base_without_dataset as _base_without_dataset,
    connect as _connect,
    core_config as _core_config,
    deduplicated_hits as _deduplicated_hits,
    existing_query_ids as _existing_query_ids,
    insert_repeats as _insert_repeats,
    load_examples as _load_examples,
    load_router_config as _load_router_config,
    prompt_for_row as _prompt_for_row,
    spent_generation_cost as _spent_generation_cost,
    sync_sample as _sync_sample,
    usage_cost as _usage_cost,
)
from src.evaluators.beir_evaluation import compute_streaming_first_stage  # noqa: E402
from src.evaluators.hotpot_answer import answer_metrics  # noqa: E402
from src.generators.answer_generator import LLMGenerator  # noqa: E402
from src.pipeline import NaiveRAGPipeline  # noqa: E402
from src.retrievers.chunk_store import JsonlOffsetChunkStore  # noqa: E402
from src.retrievers.sqlite_bm25 import analyze_sqlite_bm25_text  # noqa: E402
from src.text.token_counters import RegexTokenCounter  # noqa: E402


ACTIONS = ("bm25", "dense")
DEFAULT_PHASE28_CONFIG = "analysis/hotpotqa_router/phases/phase28/config.yaml"
DEFAULT_ROUTER_CONFIG = "outputs/router/hotpotqa_bd_router_v1/config.yaml"
DEFAULT_EXISTING_SAMPLE = (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1/sample.json"
)
BM25_CACHE_DIR = PROJECT_ROOT / "outputs/router/hotpotqa_bd_router_v1/cache"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase28-config", default=DEFAULT_PHASE28_CONFIG)
    parser.add_argument("--router-config", default=DEFAULT_ROUTER_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("freeze", "retrieve", "select", "generate", "status", "export"),
    )
    parser.add_argument("--max-new-retrieval-queries", type=int, default=None)
    parser.add_argument(
        "--retrieval-action",
        choices=("both", "dense", "bm25"),
        default="both",
        help="Execution-only checkpoint split; retrieval semantics are unchanged.",
    )
    parser.add_argument("--bm25-plus", type=int, default=0)
    parser.add_argument("--dense-plus", type=int, default=0)
    parser.add_argument("--bm25-fallback", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
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


def _load_phase28(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase 2.8 config must be a mapping")
    if value.get("protocol", {}).get("id") != "hotpotqa_bd_router_phase28_query_expansion_v1":
        raise ValueError("Unexpected Phase 2.8 protocol id")
    generation = value.get("generation", {})
    if generation.get("prompt_id") != "hotpot_short_answer_v1":
        raise ValueError("Expansion must preserve hotpot_short_answer_v1")
    if int(generation.get("repeats_per_query_action", -1)) != 3:
        raise ValueError("Expansion requires three repeats per query/action")
    if value.get("training_views", {}).get("sample_weight") != "none":
        raise ValueError("Phase 2.8 training must remain unweighted")
    return value


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    run_dir = _resolve(str(config["outputs"]["run_dir"]))
    return {
        "run_dir": run_dir,
        "sample": run_dir / "sample.json",
        "confirmation": run_dir / "confirmation_sample.json",
        "pool_manifest": run_dir / "pool_manifest.json",
        "database": run_dir / "state.sqlite3",
    }


def _group_representatives(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, str]]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(
            {
                "query_id": str(row["query_id"]),
                "group_id": str(row["group_id"]),
                "partition": "train",
            }
        )
    rng = np.random.default_rng(seed)
    representatives: list[dict[str, str]] = []
    for group_id in sorted(by_group):
        values = by_group[group_id]
        representatives.append(values[int(rng.integers(0, len(values)))])
    order = rng.permutation(len(representatives))
    return [representatives[int(position)] for position in order]


def freeze_pool(
    config: Mapping[str, Any], router: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    paths = _paths(config)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    if paths["sample"].is_file():
        return _read_json(paths["pool_manifest"])

    split_path = _resolve(str(router["split"]["path"]))
    split = _read_json(split_path)
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("split.json has no assignments")
    existing_sample = _read_json(_resolve(DEFAULT_EXISTING_SAMPLE))
    existing_train = [
        row for row in existing_sample.get("rows", []) if row.get("partition") == "train"
    ]
    expected_existing = int(config["scope"]["existing_query_count"])
    if len(existing_train) != expected_existing:
        raise ValueError("Existing Phase 2.7 train sample count changed")
    existing_groups = {str(row["group_id"]) for row in existing_train}
    eligible = [
        row
        for row in assignments
        if row.get("partition") == "train"
        and str(row["group_id"]) not in existing_groups
    ]
    representatives = _group_representatives(
        eligible, seed=int(config["acquisition"]["pool_seed"])
    )
    confirmation_count = int(config["acquisition"]["confirmation_queries"])
    if len(representatives) <= confirmation_count:
        raise ValueError("Eligible pool is too small")
    confirmation_rng = np.random.default_rng(
        int(config["acquisition"]["confirmation_seed"])
    )
    confirmation_positions = set(
        int(value)
        for value in confirmation_rng.choice(
            len(representatives), confirmation_count, replace=False
        )
    )
    confirmation = [
        row for position, row in enumerate(representatives)
        if position in confirmation_positions
    ]
    acquisition = [
        row for position, row in enumerate(representatives)
        if position not in confirmation_positions
    ]
    stratum_rng = np.random.default_rng(int(config["acquisition"]["stratum_seed"]))
    acquisition = [
        acquisition[int(position)]
        for position in stratum_rng.permutation(len(acquisition))
    ]
    confirmation_groups = {row["group_id"] for row in confirmation}
    acquisition_groups = {row["group_id"] for row in acquisition}
    if confirmation_groups & acquisition_groups or existing_groups & acquisition_groups:
        raise RuntimeError("Query acquisition group boundary is invalid")

    sample = {
        "seed": int(config["acquisition"]["pool_seed"]),
        "selection": "one_query_per_new_train_group_after_existing_group_exclusion",
        "counts": {"train": len(acquisition), "fresh_dev": 0},
        "rows": acquisition,
    }
    confirmation_sample = {
        "seed": int(config["acquisition"]["confirmation_seed"]),
        "selection": "sealed_uniform_group_confirmation_before_retrieval",
        "sealed": True,
        "rows": confirmation,
    }
    _write_json(paths["sample"], sample)
    _write_json(paths["confirmation"], confirmation_sample)

    connection = _connect(paths["run_dir"])
    try:
        _sync_sample(connection, sample)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS acquisition (
                query_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                query_rank INTEGER NOT NULL,
                stratum TEXT,
                selected_lane TEXT,
                selected_at_unix REAL
            )
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO acquisition VALUES (?, ?, ?, NULL, NULL, NULL)",
            [
                (row["query_id"], row["group_id"], rank)
                for rank, row in enumerate(acquisition)
            ],
        )
        connection.commit()
    finally:
        connection.close()

    manifest = {
        "protocol_id": config["protocol"]["id"],
        "status": "frozen",
        "eligible_train_rows_before_group_representative": len(eligible),
        "eligible_new_groups": len(representatives),
        "acquisition_queries": len(acquisition),
        "confirmation_queries": len(confirmation),
        "existing_query_count": len(existing_train),
        "existing_group_count": len(existing_groups),
        "split_sha256": _sha256(split_path),
        "phase28_config_sha256": _sha256(config_path),
        "sample_sha256": _sha256(paths["sample"]),
        "confirmation_sha256": _sha256(paths["confirmation"]),
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
        "created_at_unix": time.time(),
    }
    _write_json(paths["pool_manifest"], manifest)
    return manifest


def _pending_retrieval_rows(
    run_dir: Path,
    sample: Mapping[str, Any],
    *,
    limit: int,
    retrieval_action: str = "both",
    claim_id: str | None = None,
) -> list[dict[str, str]]:
    if retrieval_action == "dense":
        missing_clause = "d.query_id IS NULL"
    elif retrieval_action == "bm25":
        missing_clause = "b.query_id IS NULL AND d.query_id IS NOT NULL"
    elif retrieval_action == "both":
        missing_clause = "b.query_id IS NULL OR d.query_id IS NULL"
    else:
        raise ValueError(f"Unknown retrieval action: {retrieval_action}")
    connection = _connect(run_dir)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_claims (
                query_id TEXT NOT NULL,
                action TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                claimed_at_unix REAL NOT NULL,
                PRIMARY KEY (query_id, action)
            )
            """
        )
        connection.commit()
        if claim_id is not None:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM retrieval_claims WHERE claimed_at_unix < ?",
                (time.time() - 6 * 60 * 60,),
            )
        rows = connection.execute(
            f"""
            SELECT s.query_id
            FROM sample_rows AS s
            LEFT JOIN outcomes AS b
              ON b.query_id=s.query_id AND b.action='bm25' AND b.repeat_id=0
            LEFT JOIN outcomes AS d
              ON d.query_id=s.query_id AND d.action='dense' AND d.repeat_id=0
            LEFT JOIN retrieval_claims AS c
              ON c.query_id=s.query_id
             AND (c.action=? OR ?='both')
            WHERE s.partition='train' AND ({missing_clause}) AND c.query_id IS NULL
            ORDER BY s.query_rank
            LIMIT ?
            """,
            (retrieval_action, retrieval_action, limit),
        ).fetchall()
        if claim_id is not None and rows:
            connection.executemany(
                """
                INSERT INTO retrieval_claims
                (query_id, action, claim_id, claimed_at_unix)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (str(query_id), retrieval_action, claim_id, time.time())
                    for (query_id,) in rows
                ],
            )
        connection.commit()
    finally:
        connection.close()
    by_id = {str(row["query_id"]): dict(row) for row in sample["rows"]}
    return [by_id[str(query_id)] for (query_id,) in rows]


def _release_retrieval_claims(run_dir: Path, claim_id: str) -> None:
    connection = _connect(run_dir)
    try:
        connection.execute(
            "DELETE FROM retrieval_claims WHERE claim_id=?", (claim_id,)
        )
        connection.commit()
    finally:
        connection.close()


def _jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_set = {str(value) for value in left}
    right_set = {str(value) for value in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _refresh_strata(run_dir: Path, query_ids: Sequence[str]) -> dict[str, int]:
    connection = _connect(run_dir)
    counts: dict[str, int] = defaultdict(int)
    try:
        for query_id in query_ids:
            payloads = {
                str(action): json.loads(payload)
                for action, payload in connection.execute(
                    """
                    SELECT action, payload FROM outcomes
                    WHERE query_id=? AND repeat_id=0 AND action IN ('bm25','dense')
                    """,
                    (query_id,),
                )
            }
            if set(payloads) != set(ACTIONS):
                raise RuntimeError(f"Incomplete retrieval payload for {query_id}")
            bm25 = payloads["bm25"]["retrieval"]
            dense = payloads["dense"]["retrieval"]
            bm25_recall = float(bm25["evidence_page_recall"])
            dense_recall = float(dense["evidence_page_recall"])
            if bm25_recall > dense_recall:
                stratum = "bm25_plus"
            elif dense_recall > bm25_recall:
                stratum = "dense_plus"
            else:
                overlap = _jaccard(bm25["doc_ids"], dense["doc_ids"])
                if overlap == 0.0:
                    stratum = "equal_zero_overlap"
                elif overlap <= 0.25:
                    stratum = "equal_low_overlap"
                else:
                    stratum = "equal_other"
            connection.execute(
                "UPDATE acquisition SET stratum=? WHERE query_id=?",
                (stratum, query_id),
            )
            counts[stratum] += 1
        connection.commit()
    finally:
        connection.close()
    return dict(counts)


def _reconcile_completed_strata(run_dir: Path) -> dict[str, int]:
    connection = _connect(run_dir)
    try:
        query_ids = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT a.query_id
                FROM acquisition AS a
                JOIN outcomes AS b
                  ON b.query_id=a.query_id AND b.action='bm25' AND b.repeat_id=0
                JOIN outcomes AS d
                  ON d.query_id=a.query_id AND d.action='dense' AND d.repeat_id=0
                WHERE (a.stratum IS NULL OR a.stratum='unretrieved')
                ORDER BY a.query_rank
                """
            )
        ]
    finally:
        connection.close()
    return _refresh_strata(run_dir, query_ids) if query_ids else {}


class _ExactNumpySQLiteBM25:
    """Exact scorer for the frozen SQLite postings, without SQL GROUP BY."""

    def __init__(self, index_dir: Path, cache_dir: Path) -> None:
        manifest = _read_json(index_dir / "manifest.json")
        if manifest.get("backend") != "sqlite_bm25_v1":
            raise ValueError("Phase 2.8 requires sqlite_bm25_v1")
        self.document_count = int(manifest["document_count"])
        self.average_document_length = float(manifest["average_document_length"])
        bm25 = manifest["identity"]["bm25"]
        self.k1 = float(bm25["k1"])
        self.b = float(bm25["b"])
        database = index_dir / str(manifest["artifacts"]["database"]["file"])
        self.connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True
        )
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.execute("PRAGMA mmap_size=536870912")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.lengths_path = cache_dir / "bm25_doc_lengths.npy"
        if not self.lengths_path.is_file():
            self._build_lengths()
        self.lengths = np.load(self.lengths_path, mmap_mode="r")
        if self.lengths.shape != (self.document_count,):
            raise ValueError("BM25 document-length cache has the wrong shape")
        self.scores = np.zeros(self.document_count, dtype=np.float64)
        self.postings_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self.postings_cache_bytes = 0
        self.postings_cache_max_bytes = 512 * 1024 * 1024

    def _build_lengths(self) -> None:
        target = np.zeros(self.document_count, dtype=np.int32)
        expected = 0
        cursor = self.connection.execute(
            "SELECT vector_id, length FROM docs ORDER BY vector_id"
        )
        while True:
            rows = cursor.fetchmany(100000)
            if not rows:
                break
            ids = np.fromiter((int(row[0]) for row in rows), dtype=np.int64)
            values = np.fromiter((int(row[1]) for row in rows), dtype=np.int32)
            if ids[0] != expected or not np.array_equal(
                ids, np.arange(expected, expected + len(ids), dtype=np.int64)
            ):
                raise ValueError("BM25 docs vector ids are not contiguous")
            target[ids] = values
            expected += len(ids)
        if expected != self.document_count:
            raise ValueError("BM25 document-length cache row count mismatch")
        np.save(self.lengths_path, target, allow_pickle=False)

    def close(self) -> None:
        self.connection.close()

    def _postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self.postings_cache.pop(term, None)
        if cached is not None:
            self.postings_cache[term] = cached
            return cached
        id_parts: list[np.ndarray] = []
        tf_parts: list[np.ndarray] = []
        cursor = self.connection.execute(
            "SELECT vector_id, tf FROM postings WHERE term=? ORDER BY vector_id",
            (term,),
        )
        while True:
            rows = cursor.fetchmany(100000)
            if not rows:
                break
            id_parts.append(
                np.fromiter((int(row[0]) for row in rows), dtype=np.int64)
            )
            tf_parts.append(
                np.fromiter((float(row[1]) for row in rows), dtype=np.float64)
            )
        if not id_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        vector_ids = id_parts[0] if len(id_parts) == 1 else np.concatenate(id_parts)
        tf = tf_parts[0] if len(tf_parts) == 1 else np.concatenate(tf_parts)
        cache_bytes = int(vector_ids.nbytes + tf.nbytes)
        if cache_bytes <= self.postings_cache_max_bytes:
            while (
                self.postings_cache
                and self.postings_cache_bytes + cache_bytes
                > self.postings_cache_max_bytes
            ):
                _, evicted = self.postings_cache.popitem(last=False)
                self.postings_cache_bytes -= int(
                    evicted[0].nbytes + evicted[1].nbytes
                )
            self.postings_cache[term] = (vector_ids, tf)
            self.postings_cache_bytes += cache_bytes
        return vector_ids, tf

    def retrieve(self, query: str, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self.scores.fill(0.0)
        terms = sorted(set(analyze_sqlite_bm25_text(query)))
        if not terms:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        placeholders = ",".join("?" for _ in terms)
        stats = {
            str(term): float(idf)
            for term, idf in self.connection.execute(
                f"SELECT term, idf FROM term_stats WHERE term IN ({placeholders})",
                terms,
            )
        }
        for term in terms:
            idf = stats.get(term)
            if idf is None:
                continue
            vector_ids, tf = self._postings(term)
            if not len(vector_ids):
                continue
            lengths = np.asarray(self.lengths[vector_ids], dtype=np.float64)
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * lengths / self.average_document_length
            )
            self.scores[vector_ids] += idf * (tf * (self.k1 + 1.0) / denominator)
        nonzero = np.flatnonzero(self.scores > 0.0)
        if not len(nonzero):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        if len(nonzero) > top_k:
            values = self.scores[nonzero]
            provisional = np.argpartition(values, -top_k)[-top_k:]
            threshold = float(np.min(values[provisional]))
            higher = nonzero[values > threshold]
            tied = nonzero[values == threshold]
            remaining = top_k - len(higher)
            nonzero = np.concatenate((higher, np.sort(tied)[:remaining]))
        order = np.lexsort((nonzero, -self.scores[nonzero]))
        vector_ids = nonzero[order].astype(np.int64, copy=False)
        return vector_ids, self.scores[vector_ids].astype(np.float64, copy=True)


def _prepare_dense_incremental(
    connection: sqlite3.Connection,
    router: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    query_batch_size: int,
    checkpoint_group_size: int,
) -> None:
    """Run the unchanged exact dense search in committed query groups."""

    partition = str(examples[0]["split"])
    existing = _existing_query_ids(connection, partition, "dense", repeats)
    pending_examples = [row for row in examples if str(row["query_id"]) not in existing]
    if not pending_examples:
        return
    candidate_k = int(router["retrieval"]["candidate_k"])
    final_k = int(router["retrieval"]["final_k"])
    token_counter = RegexTokenCounter()
    context_max_tokens = int(router["context"]["max_tokens"])
    dense_config = _core_config(router, method="dense")
    with NaiveRAGPipeline(dense_config) as pipeline:
        for start in range(0, len(pending_examples), checkpoint_group_size):
            group = pending_examples[start : start + checkpoint_group_size]
            questions = [
                {
                    "question_id": row["query_id"],
                    "question": row["question"],
                    "qrels": {},
                }
                for row in group
            ]
            started = time.perf_counter()
            batch = compute_streaming_first_stage(
                pipeline,
                questions,
                candidate_k=candidate_k,
                query_batch_size=query_batch_size,
            )
            total_ms = (time.perf_counter() - started) * 1000.0
            for position, example in enumerate(group):
                hits = _deduplicated_hits(
                    pipeline.chunk_store,
                    batch.vector_ids[position],
                    batch.scores[position],
                    final_k=final_k,
                )
                base = _base_without_dataset(
                    example,
                    action="dense",
                    hits=hits,
                    retrieval_latency_ms=total_ms / len(group),
                    token_counter=token_counter,
                    context_max_tokens=context_max_tokens,
                )
                _insert_repeats(connection, base, repeats)
            connection.commit()
            print(
                json.dumps(
                    {
                        "dense_rows_prepared": min(start + len(group), len(pending_examples)),
                        "dense_group_queries": len(group),
                        "dense_group_total_ms": total_ms,
                    }
                ),
                flush=True,
            )


def _prepare_bm25_incremental(
    connection: sqlite3.Connection,
    router: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> None:
    """Run the unchanged BM25 SQL while committing each completed query."""

    partition = str(examples[0]["split"])
    existing = _existing_query_ids(connection, partition, "bm25", repeats)
    pending_examples = [row for row in examples if str(row["query_id"]) not in existing]
    if not pending_examples:
        return
    candidate_k = int(router["retrieval"]["candidate_k"])
    final_k = int(router["retrieval"]["final_k"])
    token_counter = RegexTokenCounter()
    context_max_tokens = int(router["context"]["max_tokens"])
    build_dir = _resolve(str(router["artifacts"]["build"]))
    build_manifest = _read_json(build_dir / "manifest.json")
    artifacts = build_manifest["artifacts"]
    chunk_store = JsonlOffsetChunkStore(
        build_dir / str(artifacts["chunks"]["file"]),
        build_dir / str(artifacts["chunk_offsets"]["file"]),
        expected_rows=int(artifacts["chunks"]["rows"]),
    )
    scorer = _ExactNumpySQLiteBM25(
        _resolve(str(router["artifacts"]["bm25_index"])),
        BM25_CACHE_DIR,
    )
    try:
        for position, example in enumerate(pending_examples, start=1):
            started = time.perf_counter()
            vector_ids, scores = scorer.retrieve(
                str(example["question"]), top_k=candidate_k
            )
            latency = (time.perf_counter() - started) * 1000.0
            hits = _deduplicated_hits(
                chunk_store,
                vector_ids,
                scores,
                final_k=final_k,
            )
            base = _base_without_dataset(
                example,
                action="bm25",
                hits=hits,
                retrieval_latency_ms=latency,
                token_counter=token_counter,
                context_max_tokens=context_max_tokens,
            )
            _insert_repeats(connection, base, repeats)
            connection.commit()
            if position % 25 == 0 or position == len(pending_examples):
                print(
                    json.dumps(
                        {
                            "bm25_rows_prepared": position,
                            "bm25_query_latency_ms": latency,
                        }
                    ),
                    flush=True,
                )
    finally:
        try:
            scorer.close()
        finally:
            chunk_store.close()


def retrieve(
    config: Mapping[str, Any],
    router: Mapping[str, Any],
    *,
    maximum: int | None,
    retrieval_action: str = "both",
) -> dict[str, Any]:
    paths = _paths(config)
    reconciled_before = _reconcile_completed_strata(paths["run_dir"])
    sample = _read_json(paths["sample"])
    limit = int(config["acquisition"]["retrieval_block_size"])
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("max-new-retrieval-queries must be positive")
        limit = maximum
    claim_id = f"{os.getpid()}-{uuid.uuid4().hex}"
    try:
        pending = _pending_retrieval_rows(
            paths["run_dir"],
            sample,
            limit=limit,
            retrieval_action=retrieval_action,
            claim_id=claim_id,
        )
        if not pending:
            return status(config)
        examples, mapping = _load_examples(router, {"rows": pending})
        repeats = int(config["generation"]["repeats_per_query_action"])
        connection = _connect(paths["run_dir"])
        try:
            if retrieval_action in {"both", "dense"}:
                _prepare_dense_incremental(
                    connection,
                    router,
                    examples,
                    repeats=repeats,
                    query_batch_size=int(config["acquisition"]["dense_query_batch_size"]),
                    checkpoint_group_size=int(
                        config["acquisition"]["retrieval_block_size"]
                    ),
                )
            if retrieval_action in {"both", "bm25"}:
                _prepare_bm25_incremental(connection, router, examples, repeats=repeats)
        finally:
            connection.close()
    finally:
        _release_retrieval_claims(paths["run_dir"], claim_id)
    strata = _reconcile_completed_strata(paths["run_dir"])
    result = {
        "retrieved_queries": len(pending),
        "mapped_queries": int(mapping["mapped"]),
        "retrieval_action": retrieval_action,
        "reconciled_before": reconciled_before,
        "new_strata": strata,
    }
    _write_json(paths["run_dir"] / "last_retrieval.json", result)
    return result


def _select_from_stratum(
    connection: sqlite3.Connection,
    *,
    strata: Sequence[str],
    count: int,
    lane: str,
) -> list[str]:
    if count <= 0:
        return []
    placeholders = ",".join("?" for _ in strata)
    rows = connection.execute(
        f"""
        SELECT query_id FROM acquisition
        WHERE stratum IN ({placeholders}) AND selected_lane IS NULL
        ORDER BY query_rank
        LIMIT ?
        """,
        [*strata, count],
    ).fetchall()
    query_ids = [str(row[0]) for row in rows]
    now = time.time()
    connection.executemany(
        "UPDATE acquisition SET selected_lane=?, selected_at_unix=? WHERE query_id=?",
        [(lane, now, query_id) for query_id in query_ids],
    )
    return query_ids


def select_for_generation(
    config: Mapping[str, Any], *, bm25_plus: int, dense_plus: int, bm25_fallback: int
) -> dict[str, Any]:
    paths = _paths(config)
    connection = _connect(paths["run_dir"])
    try:
        selected = {
            "bm25_plus": _select_from_stratum(
                connection,
                strata=("bm25_plus",),
                count=bm25_plus,
                lane="bm25_plus",
            ),
            "dense_plus": _select_from_stratum(
                connection,
                strata=("dense_plus",),
                count=dense_plus,
                lane="dense_plus",
            ),
            "bm25_fallback": _select_from_stratum(
                connection,
                strata=("equal_zero_overlap", "equal_low_overlap"),
                count=bm25_fallback,
                lane="bm25_fallback",
            ),
        }
        connection.commit()
    finally:
        connection.close()
    result = {
        "selected": {name: len(values) for name, values in selected.items()},
        "selected_query_ids": selected,
    }
    _write_json(paths["run_dir"] / "last_selection.json", result)
    return result


def _generation_view(result: Any) -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "success",
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
    }


def generate(
    config: Mapping[str, Any],
    router: Mapping[str, Any],
    *,
    max_new_calls: int | None,
    workers: int | None,
) -> dict[str, Any]:
    paths = _paths(config)
    connection = _connect(paths["run_dir"])
    local = threading.local()
    created: list[LLMGenerator] = []
    created_lock = threading.Lock()
    try:
        pending = connection.execute(
            """
            SELECT o.query_id, o.action, o.repeat_id, o.payload
            FROM outcomes AS o
            JOIN acquisition AS a ON a.query_id=o.query_id
            WHERE a.selected_lane IS NOT NULL
              AND o.action IN ('bm25','dense')
              AND o.status IN ('pending_generation','generation_failure')
            ORDER BY a.selected_at_unix, a.query_rank,
                     CASE o.action WHEN 'bm25' THEN 0 ELSE 1 END, o.repeat_id
            """
        ).fetchall()
        if max_new_calls is not None:
            if max_new_calls <= 0:
                raise ValueError("max-new-calls must be positive")
            pending = pending[:max_new_calls]
        if not pending:
            return status(config)
        generation = _core_config(router, method="bm25")["generation"]
        expected_model = str(config["generation"]["model"])
        if generation["model"] != expected_model:
            raise ValueError("Router generation model differs from frozen Phase 2.8 model")
        worker_count = int(
            config["generation"]["provider_workers"] if workers is None else workers
        )
        if worker_count <= 0:
            raise ValueError("workers must be positive")
        spent = _spent_generation_cost(connection)
        budget = float(config["generation"]["hard_budget_usd"])
        if spent >= budget:
            raise RuntimeError("Phase 2.8 generation budget is exhausted")

        def worker(item: Sequence[Any]) -> tuple[str, str, int, dict[str, Any]]:
            query_id, action, repeat_id, payload_text = item
            generator = getattr(local, "generator", None)
            if generator is None:
                generator = LLMGenerator(
                    provider=generation["provider"],
                    model=generation["model"],
                    temperature=generation["temperature"],
                    max_output_tokens=generation["max_output_tokens"],
                    timeout_seconds=generation["timeout_seconds"],
                    max_retries=generation["max_retries"],
                )
                local.generator = generator
                with created_lock:
                    created.append(generator)
            row = json.loads(str(payload_text))
            try:
                result = generator.generate_from_prompt(
                    _prompt_for_row(row, str(router["prompt"]["version"])),
                    row["question"],
                    [],
                )
                prediction = result.answer.strip()
                metrics = answer_metrics(prediction, row["reference_answers"])
                metrics.pop("answer_correctness", None)
                row.update(
                    {
                        "status": "success",
                        "prediction": prediction,
                        "metrics": metrics,
                        "generation": _generation_view(result),
                        "completion_protocol": config["protocol"]["id"],
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "generation_failure",
                        "generation": {
                            "attempted": True,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        "completion_protocol": config["protocol"]["id"],
                    }
                )
            return str(query_id), str(action), int(repeat_id), row

        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(worker, item) for item in pending]
                for future in as_completed(futures):
                    query_id, action, repeat_id, row = future.result()
                    connection.execute(
                        """
                        UPDATE outcomes SET status=?, payload=?
                        WHERE query_id=? AND action=? AND repeat_id=?
                        """,
                        (
                            str(row["status"]),
                            json.dumps(row, ensure_ascii=False, allow_nan=False),
                            query_id,
                            action,
                            repeat_id,
                        ),
                    )
                    completed += 1
                    if completed % 100 == 0:
                        connection.commit()
                        print(
                            json.dumps(
                                {"generation_new_calls": completed, "selected_calls": len(pending)}
                            ),
                            flush=True,
                        )
        finally:
            connection.commit()
            for generator in created:
                close = getattr(getattr(generator, "_client", None), "close", None)
                if callable(close):
                    close()
        if _spent_generation_cost(connection) > budget:
            raise RuntimeError("Phase 2.8 generation budget exceeded")
    finally:
        connection.close()
    return status(config)


def _completed_labels(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT a.query_id, a.group_id, a.query_rank, a.stratum, a.selected_lane,
               o.action, o.repeat_id, o.status, o.payload
        FROM acquisition AS a
        JOIN outcomes AS o ON o.query_id=a.query_id
        WHERE a.selected_lane IS NOT NULL AND o.action IN ('bm25','dense')
        ORDER BY a.query_rank, o.action, o.repeat_id
        """
    )
    by_query: dict[str, dict[str, Any]] = {}
    for query_id, group_id, rank, stratum, lane, action, repeat_id, state, payload in rows:
        query_id = str(query_id)
        record = by_query.setdefault(
            query_id,
            {
                "query_id": query_id,
                "group_id": str(group_id),
                "query_rank": int(rank),
                "acquisition_stratum": str(stratum),
                "selected_lane": str(lane),
                "values": defaultdict(dict),
            },
        )
        if state == "success":
            value = json.loads(str(payload))["metrics"]["normalized_token_f1"]
            record["values"][str(action)][int(repeat_id)] = float(value)
    result: list[dict[str, Any]] = []
    for record in by_query.values():
        values = record.pop("values")
        if any(set(values.get(action, {})) != {0, 1, 2} for action in ACTIONS):
            continue
        bm25_values = [values["bm25"][index] for index in range(3)]
        dense_values = [values["dense"][index] for index in range(3)]
        gap = float(np.mean(bm25_values) - np.mean(dense_values))
        record.update(
            {
                "bm25_repeat_f1": bm25_values,
                "dense_repeat_f1": dense_values,
                "bm25_mean_f1": float(np.mean(bm25_values)),
                "dense_mean_f1": float(np.mean(dense_values)),
                "f1_gap_bm25_minus_dense": gap,
            }
        )
        result.append(record)
    return result


def _label_counts(config: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tolerance = float(config["labels"]["tie_tolerance"])
    margin = float(config["labels"]["high_margin_gap"])
    counts = {
        "bm25_winner": 0,
        "dense_winner": 0,
        "exact_tie": 0,
        "high_margin_bm25_winner": 0,
        "high_margin_dense_winner": 0,
    }
    for row in labels:
        gap = float(row["f1_gap_bm25_minus_dense"])
        if gap > tolerance:
            counts["bm25_winner"] += 1
            counts["high_margin_bm25_winner"] += gap >= margin
        elif gap < -tolerance:
            counts["dense_winner"] += 1
            counts["high_margin_dense_winner"] += gap <= -margin
        else:
            counts["exact_tie"] += 1
    return {key: int(value) for key, value in counts.items()}


def status(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    if not paths["database"].is_file():
        return {"protocol_id": config["protocol"]["id"], "frozen": False}
    connection = _connect(paths["run_dir"])
    try:
        retrieval_progress_row = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN b.query_id IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN d.query_id IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN b.query_id IS NOT NULL AND d.query_id IS NOT NULL
                            THEN 1 ELSE 0 END)
            FROM acquisition AS a
            LEFT JOIN outcomes AS b
              ON b.query_id=a.query_id AND b.action='bm25' AND b.repeat_id=0
            LEFT JOIN outcomes AS d
              ON d.query_id=a.query_id AND d.action='dense' AND d.repeat_id=0
            """
        ).fetchone()
        if retrieval_progress_row is None:
            raise RuntimeError("Failed to read Phase 2.8 retrieval progress")
        acquisition_total, bm25_retrieved, dense_retrieved, paired_retrieved = (
            int(value or 0) for value in retrieval_progress_row
        )
        retrieval_progress = {
            "acquisition_total": acquisition_total,
            "bm25_retrieved": bm25_retrieved,
            "dense_retrieved": dense_retrieved,
            "paired_retrieved": paired_retrieved,
            "dense_only_in_progress": dense_retrieved - paired_retrieved,
            "bm25_only_in_progress": bm25_retrieved - paired_retrieved,
        }
        acquisition_counts = {
            str(name): int(count)
            for name, count in connection.execute(
                "SELECT COALESCE(stratum,'unretrieved'), COUNT(*) FROM acquisition GROUP BY stratum"
            )
        }
        selected_counts = {
            str(name): int(count)
            for name, count in connection.execute(
                """
                SELECT COALESCE(selected_lane,'unselected'), COUNT(*)
                FROM acquisition GROUP BY selected_lane
                """
            )
        }
        outcome_counts = {
            f"{action}:{state}": int(count)
            for action, state, count in connection.execute(
                """
                SELECT o.action, o.status, COUNT(*) FROM outcomes AS o
                JOIN acquisition AS a ON a.query_id=o.query_id
                WHERE a.selected_lane IS NOT NULL AND o.action IN ('bm25','dense')
                GROUP BY o.action, o.status
                """
            )
        }
        labels = _completed_labels(connection)
        new_counts = _label_counts(config, labels)
        existing = {key: int(value) for key, value in config["labels"]["existing"].items()}
        combined = {
            key: int(existing.get(key, 0) + new_counts.get(key, 0))
            for key in set(existing) | set(new_counts)
        }
        quota = {key: int(value) for key, value in config["labels"]["quota"].items()}
        quota_checks = {key: combined.get(key, 0) >= value for key, value in quota.items()}
        lane_labels: dict[str, dict[str, int]] = {}
        for lane in sorted({str(row["selected_lane"]) for row in labels}):
            lane_labels[lane] = _label_counts(
                config, [row for row in labels if row["selected_lane"] == lane]
            )
        spent = _spent_generation_cost(connection)
    finally:
        connection.close()
    return {
        "protocol_id": config["protocol"]["id"],
        "frozen": True,
        "retrieval_progress": retrieval_progress,
        "acquisition_counts": acquisition_counts,
        "selected_counts": selected_counts,
        "selected_outcome_counts": outcome_counts,
        "completed_selected_queries": len(labels),
        "new_label_counts": new_counts,
        "combined_label_counts": combined,
        "quota": quota,
        "quota_checks": quota_checks,
        "all_quotas_met": all(quota_checks.values()),
        "lane_label_counts": lane_labels,
        "estimated_generation_cost_usd": spent,
        "fresh_dev_rows_read": 0,
        "final_holdout_rows_read": 0,
    }


def export(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(config)
    connection = _connect(paths["run_dir"])
    try:
        labels = _completed_labels(connection)
        label_path = paths["run_dir"] / "expanded_query_labels.jsonl.gz"
        with gzip.open(label_path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in labels:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        outcome_path = paths["run_dir"] / "expanded_outcomes.jsonl.gz"
        with gzip.open(outcome_path, "wt", encoding="utf-8", newline="\n") as handle:
            for (payload,) in connection.execute(
                """
                SELECT o.payload FROM outcomes AS o JOIN acquisition AS a USING(query_id)
                WHERE a.selected_lane IS NOT NULL AND o.action IN ('bm25','dense')
                ORDER BY a.query_rank, o.action, o.repeat_id
                """
            ):
                handle.write(str(payload) + "\n")
    finally:
        connection.close()
    result = {
        "labels": str(label_path),
        "label_rows": len(labels),
        "labels_sha256": _sha256(label_path),
        "outcomes": str(outcome_path),
        "outcome_rows": len(labels) * 6,
        "outcomes_sha256": _sha256(outcome_path),
        "status": status(config),
    }
    _write_json(paths["run_dir"] / "export_manifest.json", result)
    return result


def main() -> int:
    args = _arguments()
    phase28_path = _resolve(args.phase28_config)
    phase28 = _load_phase28(phase28_path)
    router = _load_router_config(_resolve(args.router_config))
    if args.stage == "freeze":
        result = freeze_pool(phase28, router, phase28_path)
    else:
        paths = _paths(phase28)
        if not paths["sample"].is_file():
            raise RuntimeError("Run --stage freeze first")
        if args.stage == "retrieve":
            result = retrieve(
                phase28,
                router,
                maximum=args.max_new_retrieval_queries,
                retrieval_action=args.retrieval_action,
            )
        elif args.stage == "select":
            result = select_for_generation(
                phase28,
                bm25_plus=args.bm25_plus,
                dense_plus=args.dense_plus,
                bm25_fallback=args.bm25_fallback,
            )
        elif args.stage == "generate":
            result = generate(
                phase28,
                router,
                max_new_calls=args.max_new_calls,
                workers=args.workers,
            )
        elif args.stage == "export":
            result = export(phase28)
        else:
            result = status(phase28)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
