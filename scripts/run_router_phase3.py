"""Run the F1-screened Phase 3 no-context/BM25/Dense router experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import (  # noqa: E402
    GENERATOR_PRICES,
    _base_row,
    _core_config,
    _deduplicated_hits,
    _load_router_config,
    _prompt_for_row,
    _usage_cost,
)
from scripts.run_router_phase2 import _load_examples  # noqa: E402
from src.evaluators.beir_evaluation import compute_streaming_first_stage  # noqa: E402
from src.evaluators.beir_suite import compute_bm25_first_stage  # noqa: E402
from src.evaluators.hotpot_answer import answer_metrics  # noqa: E402
from src.pipeline import NaiveRAGPipeline  # noqa: E402
from src.text.token_counters import RegexTokenCounter  # noqa: E402


ACTIONS = ("no_context", "bm25", "dense")
PARTITIONS = ("train", "fresh_dev")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
    parser.add_argument(
        "--run-id",
        default="phase3_f1_pairwise_v1",
    )
    parser.add_argument(
        "--stage",
        choices=("sample", "prepare", "generate", "export", "status"),
        required=True,
    )
    parser.add_argument("--partition", choices=PARTITIONS, default="train")
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Restrict generation/export to the first nested train queries.",
    )
    parser.add_argument("--max-new-calls", type=int, default=None)
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _phase3(router: Mapping[str, Any]) -> Mapping[str, Any]:
    value = router.get("phase3")
    if not isinstance(value, Mapping):
        raise ValueError("Router config has no phase3 mapping")
    if value.get("routing_target") != "normalized_token_f1":
        raise ValueError("Phase 3 screen must use normalized_token_f1")
    if int(value.get("answer_correctness_calls_in_screen", -1)) != 0:
        raise ValueError("Phase 3 F1 screen must forbid answer-correctness calls")
    if value.get("automatic_ac_evaluation") != "forbidden":
        raise ValueError("Phase 3 must not automatically run answer correctness")
    if tuple(value.get("actions", ())) != ACTIONS:
        raise ValueError(f"Phase 3 actions must be {ACTIONS}")
    return value


def _run_dir(config_path: Path, run_id: str) -> Path:
    return config_path.parent / "runs" / run_id


def _sample_path(run_dir: Path) -> Path:
    return run_dir / "sample.json"


def _database_path(run_dir: Path) -> Path:
    return run_dir / "state.sqlite3"


def _source_run_dir(config_path: Path, router: Mapping[str, Any]) -> Path:
    return config_path.parent / "runs" / str(_phase3(router)["source_run"])


def freeze_sample(
    router: Mapping[str, Any],
    config_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    path = _sample_path(run_dir)
    if path.is_file():
        return _read_json(path)

    phase3 = _phase3(router)
    source_dir = _source_run_dir(config_path, router)
    source_sample = _read_json(source_dir / "sample.json")
    source_rows = source_sample.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError("Phase 2 source sample has no rows")
    old_train = [row for row in source_rows if row.get("partition") == "train"]
    old_dev = [row for row in source_rows if row.get("partition") == "dev"]
    expected_reused = int(phase3["source_partitions"]["reused_train"])
    if len(old_train) != expected_reused or not old_dev:
        raise ValueError("Unexpected Phase 2 source sample counts")

    split = _read_json(_resolve(str(router["split"]["path"])))
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("split.json has no assignments")
    by_id = {str(row["query_id"]): row for row in assignments}
    old_train_ids = {str(row["query_id"]) for row in old_train}
    old_dev_ids = {str(row["query_id"]) for row in old_dev}
    old_dev_groups = {str(row["group_id"]) for row in old_dev}

    train_total = int(phase3["source_partitions"]["train_total"])
    fresh_dev_total = int(phase3["source_partitions"]["fresh_dev"])
    rng = np.random.default_rng(int(phase3["sample_seed"]))

    train_candidates = [
        row
        for row in assignments
        if row.get("partition") == "train" and str(row["query_id"]) not in old_train_ids
    ]
    needed_train = train_total - len(old_train)
    train_positions = rng.choice(len(train_candidates), needed_train, replace=False)
    additional_train = [train_candidates[int(position)] for position in train_positions]

    dev_candidates = [
        row
        for row in assignments
        if row.get("partition") == "dev"
        and str(row["query_id"]) not in old_dev_ids
        and str(row["group_id"]) not in old_dev_groups
    ]
    dev_positions = rng.choice(len(dev_candidates), fresh_dev_total, replace=False)
    fresh_dev = [dev_candidates[int(position)] for position in dev_positions]

    rows: list[dict[str, str]] = []
    for partition, selected in (
        ("train", old_train + additional_train),
        ("fresh_dev", fresh_dev),
    ):
        for row in selected:
            query_id = str(row["query_id"])
            assignment = by_id[query_id]
            rows.append(
                {
                    "query_id": query_id,
                    "group_id": str(assignment["group_id"]),
                    "partition": partition,
                }
            )

    train_rows = [row for row in rows if row["partition"] == "train"]
    dev_rows = [row for row in rows if row["partition"] == "fresh_dev"]
    if len(train_rows) != train_total or len(dev_rows) != fresh_dev_total:
        raise RuntimeError("Frozen Phase 3 sample count mismatch")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("Frozen Phase 3 query ids are not unique")
    if old_dev_ids & {row["query_id"] for row in dev_rows}:
        raise RuntimeError("Fresh dev overlaps the historical dev queries")
    if old_dev_groups & {row["group_id"] for row in dev_rows}:
        raise RuntimeError("Fresh dev overlaps the historical dev groups")
    if any(by_id[row["query_id"]]["partition"] == "final_holdout" for row in rows):
        raise RuntimeError("Phase 3 sample contains final holdout")

    sample = {
        "seed": int(phase3["sample_seed"]),
        "selection": "nested_train_and_fresh_dev_without_replacement",
        "learning_curve": list(phase3["source_partitions"]["learning_curve"]),
        "counts": {"train": len(train_rows), "fresh_dev": len(dev_rows)},
        "reused_train": len(old_train),
        "historical_dev_excluded": len(old_dev),
        "final_holdout_outcomes": 0,
        "rows": rows,
    }
    _write_json(path, sample)
    return sample


def _connect(run_dir: Path) -> sqlite3.Connection:
    path = _database_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_rows (
            partition TEXT NOT NULL,
            query_rank INTEGER NOT NULL,
            query_id TEXT NOT NULL PRIMARY KEY,
            group_id TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            partition TEXT NOT NULL,
            query_id TEXT NOT NULL,
            action TEXT NOT NULL,
            repeat_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            reused INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (query_id, action, repeat_id)
        )
        """
    )
    connection.commit()
    return connection


def _sync_sample(connection: sqlite3.Connection, sample: Mapping[str, Any]) -> None:
    ranks = {partition: 0 for partition in PARTITIONS}
    rows = sample.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Sample has no rows")
    for row in rows:
        partition = str(row["partition"])
        connection.execute(
            "INSERT OR IGNORE INTO sample_rows VALUES (?, ?, ?, ?)",
            (
                partition,
                ranks[partition],
                str(row["query_id"]),
                str(row["group_id"]),
            ),
        )
        ranks[partition] += 1
    connection.commit()


def _selection(sample: Mapping[str, Any], partition: str) -> list[dict[str, str]]:
    return [
        dict(row)
        for row in sample["rows"]
        if isinstance(row, Mapping) and row.get("partition") == partition
    ]


def _insert_row(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    reused: bool,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO outcomes
        (partition, query_id, action, repeat_id, status, reused, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["split"]),
            str(row["query_id"]),
            str(row["action"]),
            int(row["repeat_id"]),
            str(row["status"]),
            int(reused),
            json.dumps(row, ensure_ascii=False, allow_nan=False),
        ),
    )


def _project_reused_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Reused Phase 2 row has no metrics")
    projected = copy.deepcopy(dict(row))
    projected.pop("dataset_id", None)
    projected.pop("answer_correctness", None)
    projected["metrics"] = {
        "normalized_exact_match": float(metrics["normalized_exact_match"]),
        "normalized_token_f1": float(metrics["normalized_token_f1"]),
    }
    projected["status"] = "success"
    return projected


def _prepare_reused_train(
    connection: sqlite3.Connection,
    source_dir: Path,
    reused_ids: set[str],
) -> int:
    inserted_before = connection.total_changes
    with (source_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("query_id")) not in reused_ids or row.get("split") != "train":
                continue
            if row.get("status") != "success" or row.get("action") not in ACTIONS:
                raise ValueError("Phase 2 reusable row is incomplete")
            _insert_row(connection, _project_reused_row(row), reused=True)
    connection.commit()
    return connection.total_changes - inserted_before


def _existing_query_ids(
    connection: sqlite3.Connection,
    partition: str,
    action: str,
    repeats: int,
) -> set[str]:
    rows = connection.execute(
        """
        SELECT query_id
        FROM outcomes
        WHERE partition = ? AND action = ?
        GROUP BY query_id
        HAVING COUNT(*) = ?
        """,
        (partition, action, repeats),
    )
    return {str(row[0]) for row in rows}


def _base_without_dataset(
    example: Mapping[str, Any],
    *,
    action: str,
    hits: Sequence[Any],
    retrieval_latency_ms: float,
    token_counter: RegexTokenCounter,
    context_max_tokens: int,
) -> dict[str, Any]:
    row = _base_row(
        example,
        action=action,
        hits=hits,
        retrieval_latency_ms=retrieval_latency_ms,
        token_counter=token_counter,
        context_max_tokens=context_max_tokens,
    )
    row.pop("dataset_id", None)
    return row


def _insert_repeats(
    connection: sqlite3.Connection,
    base: Mapping[str, Any],
    repeats: int,
) -> None:
    for repeat_id in range(repeats):
        row = copy.deepcopy(dict(base))
        row["repeat_id"] = repeat_id
        _insert_row(connection, row, reused=False)


def _prepare_no_context(
    connection: sqlite3.Connection,
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    context_max_tokens: int,
) -> None:
    existing = _existing_query_ids(connection, str(examples[0]["split"]), "no_context", repeats)
    token_counter = RegexTokenCounter()
    for position, example in enumerate(examples, start=1):
        if str(example["query_id"]) in existing:
            continue
        base = _base_without_dataset(
            example,
            action="no_context",
            hits=(),
            retrieval_latency_ms=0.0,
            token_counter=token_counter,
            context_max_tokens=context_max_tokens,
        )
        _insert_repeats(connection, base, repeats)
        if position % 250 == 0:
            connection.commit()
    connection.commit()


def _prepare_dense(
    connection: sqlite3.Connection,
    router: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> None:
    partition = str(examples[0]["split"])
    existing = _existing_query_ids(connection, partition, "dense", repeats)
    pending_examples = [row for row in examples if str(row["query_id"]) not in existing]
    if not pending_examples:
        return
    questions = [
        {"question_id": row["query_id"], "question": row["question"], "qrels": {}}
        for row in pending_examples
    ]
    phase3 = _phase3(router)
    candidate_k = int(router["retrieval"]["candidate_k"])
    final_k = int(router["retrieval"]["final_k"])
    token_counter = RegexTokenCounter()
    context_max_tokens = int(router["context"]["max_tokens"])
    dense_config = _core_config(router, method="dense")
    with NaiveRAGPipeline(dense_config) as pipeline:
        started = time.perf_counter()
        batch = compute_streaming_first_stage(
            pipeline,
            questions,
            candidate_k=candidate_k,
            query_batch_size=int(phase3["dense_query_batch_size"]),
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        for position, example in enumerate(pending_examples, start=1):
            hits = _deduplicated_hits(
                pipeline.chunk_store,
                batch.vector_ids[position - 1],
                batch.scores[position - 1],
                final_k=final_k,
            )
            base = _base_without_dataset(
                example,
                action="dense",
                hits=hits,
                retrieval_latency_ms=total_ms / len(pending_examples),
                token_counter=token_counter,
                context_max_tokens=context_max_tokens,
            )
            _insert_repeats(connection, base, repeats)
            if position % 100 == 0:
                connection.commit()
                print(json.dumps({"dense_rows_prepared": position}), flush=True)
    connection.commit()


def _prepare_bm25(
    connection: sqlite3.Connection,
    router: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> None:
    partition = str(examples[0]["split"])
    existing = _existing_query_ids(connection, partition, "bm25", repeats)
    pending_examples = [row for row in examples if str(row["query_id"]) not in existing]
    if not pending_examples:
        return
    questions = [
        {"question_id": row["query_id"], "question": row["question"], "qrels": {}}
        for row in pending_examples
    ]
    candidate_k = int(router["retrieval"]["candidate_k"])
    final_k = int(router["retrieval"]["final_k"])
    token_counter = RegexTokenCounter()
    context_max_tokens = int(router["context"]["max_tokens"])
    bm25_config = _core_config(router, method="bm25")
    already_prepared = len(examples) - len(pending_examples)
    with NaiveRAGPipeline(bm25_config) as pipeline:
        # BM25 latency varies sharply with posting-list frequency.  Commit in
        # small groups so a pathological query cannot hide too much completed
        # work behind one transaction.
        group_size = 10
        for start in range(0, len(pending_examples), group_size):
            stop = min(start + group_size, len(pending_examples))
            group_examples = pending_examples[start:stop]
            group_questions = questions[start:stop]
            batch = compute_bm25_first_stage(
                pipeline,
                group_questions,
                candidate_k=candidate_k,
                retained_k=candidate_k,
            )
            for local_position, example in enumerate(group_examples):
                scores, vector_ids = batch.row(local_position)
                hits = _deduplicated_hits(
                    pipeline.chunk_store,
                    vector_ids,
                    scores,
                    final_k=final_k,
                )
                latency = float(
                    batch.per_question_timings_ms[local_position].get("total_ms", 0.0)
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
            print(
                json.dumps({"bm25_rows_prepared": already_prepared + stop}),
                flush=True,
            )
    connection.commit()


def prepare_partition(
    router: Mapping[str, Any],
    config_path: Path,
    run_dir: Path,
    sample: Mapping[str, Any],
    partition: str,
    *,
    train_size: int | None,
) -> dict[str, int]:
    if partition == "fresh_dev" and not (run_dir / "model_freeze.json").is_file():
        raise RuntimeError("Fresh dev remains sealed until model_freeze.json exists")
    selected = _selection(sample, partition)
    if partition == "train":
        phase3 = _phase3(router)
        limit = int(
            phase3["source_partitions"]["train_total"]
            if train_size is None
            else train_size
        )
        if limit not in {
            int(value) for value in phase3["source_partitions"]["learning_curve"]
        }:
            raise ValueError("train-size must be a frozen learning-curve size")
        selected = selected[:limit]
    elif train_size is not None:
        raise ValueError("train-size only applies to the train partition")
    examples, mapping = _load_examples(router, {"rows": selected})
    phase3 = _phase3(router)
    repeats = int(phase3["repeats_per_query_action"])
    connection = _connect(run_dir)
    try:
        _sync_sample(connection, sample)
        if partition == "train":
            reused_count = int(phase3["source_partitions"]["reused_train"])
            reused_ids = {str(row["query_id"]) for row in selected[:reused_count]}
            _prepare_reused_train(
                connection,
                _source_run_dir(config_path, router),
                reused_ids,
            )
        _prepare_no_context(
            connection,
            examples,
            repeats=repeats,
            context_max_tokens=int(router["context"]["max_tokens"]),
        )
        _prepare_dense(connection, router, examples, repeats=repeats)
        _prepare_bm25(connection, router, examples, repeats=repeats)
        expected = len(selected) * len(ACTIONS) * repeats
        query_ids = {str(row["query_id"]) for row in selected}
        actual = sum(
            1
            for (query_id,) in connection.execute(
                "SELECT query_id FROM outcomes WHERE partition = ?", (partition,)
            )
            if str(query_id) in query_ids
        )
        if actual != expected:
            raise RuntimeError(f"Prepared {actual} {partition} rows, expected {expected}")
        result = {
            "sampled_queries": len(selected),
            "mapped_queries": int(mapping["mapped"]),
            "prepared_rows": actual,
        }
        suffix = str(len(selected)) if partition == "train" else "all"
        _write_json(run_dir / f"{partition}_prepare_{suffix}.json", result)
        return result
    finally:
        connection.close()


def _generator_view(result: Any) -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "success",
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
    }


def _spent_generation_cost(connection: sqlite3.Connection) -> float:
    total = 0.0
    rows = connection.execute(
        "SELECT payload FROM outcomes WHERE reused = 0 AND status = 'success'"
    )
    for (payload,) in rows:
        row = json.loads(payload)
        generation = row.get("generation")
        if isinstance(generation, Mapping):
            total += _usage_cost(generation.get("token_usage", {}), GENERATOR_PRICES)
    return total


def generate_partition(
    router: Mapping[str, Any],
    run_dir: Path,
    sample: Mapping[str, Any],
    partition: str,
    *,
    train_size: int | None,
    max_new_calls: int | None,
) -> dict[str, Any]:
    from src.generators.answer_generator import LLMGenerator

    phase3 = _phase3(router)
    if partition == "train":
        limit = int(phase3["source_partitions"]["train_total"] if train_size is None else train_size)
        if limit not in {int(value) for value in phase3["source_partitions"]["learning_curve"]}:
            raise ValueError("train-size must be a frozen learning-curve size")
    else:
        if train_size is not None:
            raise ValueError("train-size only applies to the train partition")
        if not (run_dir / "model_freeze.json").is_file():
            raise RuntimeError("Fresh dev remains sealed until model_freeze.json exists")
        limit = int(phase3["source_partitions"]["fresh_dev"])

    connection = _connect(run_dir)
    try:
        _sync_sample(connection, sample)
        pending_rows = connection.execute(
            """
            SELECT o.query_id, o.action, o.repeat_id, o.payload
            FROM outcomes AS o
            JOIN sample_rows AS s ON s.query_id = o.query_id
            WHERE o.partition = ? AND s.query_rank < ? AND o.status = 'pending_generation'
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'no_context' THEN 0 WHEN 'bm25' THEN 1 ELSE 2 END,
                     o.repeat_id
            """,
            (partition, limit),
        ).fetchall()
        attempted = int(
            connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE reused = 0 AND status != 'pending_generation'"
            ).fetchone()[0]
        )
        remaining = max(0, int(phase3["generation_call_limit"]) - attempted)
        if max_new_calls is not None:
            remaining = min(remaining, int(max_new_calls))
        pending_rows = pending_rows[:remaining]
        if not pending_rows:
            return status(run_dir, partition=partition, query_limit=limit)

        generation = _core_config(router, method="bm25")["generation"]
        local = threading.local()
        created: list[LLMGenerator] = []
        created_lock = threading.Lock()

        def worker(item: Sequence[Any]) -> tuple[str, str, int, dict[str, Any]]:
            query_id, action, repeat_id, payload = item
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
            row = json.loads(payload)
            try:
                result = generator.generate_from_prompt(_prompt_for_row(row), row["question"], [])
                prediction = result.answer.strip()
                metrics = answer_metrics(prediction, row["reference_answers"])
                metrics.pop("answer_correctness", None)
                row.update(
                    {
                        "status": "success",
                        "prediction": prediction,
                        "metrics": metrics,
                        "generation": _generator_view(result),
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
                    }
                )
            return str(query_id), str(action), int(repeat_id), row

        spent = _spent_generation_cost(connection)
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=int(phase3["provider_workers"])) as pool:
                futures = [pool.submit(worker, item) for item in pending_rows]
                for future in as_completed(futures):
                    query_id, action, repeat_id, row = future.result()
                    connection.execute(
                        """
                        UPDATE outcomes SET status = ?, payload = ?
                        WHERE query_id = ? AND action = ? AND repeat_id = ?
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
                    generation_view = row.get("generation")
                    if row["status"] == "success" and isinstance(generation_view, Mapping):
                        spent += _usage_cost(
                            generation_view.get("token_usage", {}), GENERATOR_PRICES
                        )
                    if completed % 100 == 0:
                        connection.commit()
                        print(
                            json.dumps(
                                {
                                    "generation_new_calls": completed,
                                    "partition": partition,
                                    "query_limit": limit,
                                    "estimated_cost_usd": spent,
                                }
                            ),
                            flush=True,
                        )
        finally:
            connection.commit()
            for generator in created:
                close = getattr(getattr(generator, "_client", None), "close", None)
                if callable(close):
                    close()
        if spent > float(phase3["generation_hard_budget_usd"]):
            raise RuntimeError(
                f"Generation budget exceeded: {spent:.6f} > "
                f"{float(phase3['generation_hard_budget_usd']):.6f}"
            )
        return status(run_dir, partition=partition, query_limit=limit)
    finally:
        connection.close()


def status(run_dir: Path, *, partition: str, query_limit: int | None = None) -> dict[str, Any]:
    connection = _connect(run_dir)
    try:
        if query_limit is None:
            rows = connection.execute(
                """
                SELECT o.status, COUNT(*)
                FROM outcomes AS o
                WHERE o.partition = ?
                GROUP BY o.status
                """,
                (partition,),
            )
        else:
            rows = connection.execute(
                """
                SELECT o.status, COUNT(*)
                FROM outcomes AS o
                JOIN sample_rows AS s ON s.query_id = o.query_id
                WHERE o.partition = ? AND s.query_rank < ?
                GROUP BY o.status
                """,
                (partition, query_limit),
            )
        counts = {str(name): int(count) for name, count in rows}
        return {"partition": partition, "query_limit": query_limit, "status_counts": counts}
    finally:
        connection.close()


def export_results(
    run_dir: Path,
    partition: str,
    *,
    query_limit: int | None,
) -> dict[str, Any]:
    connection = _connect(run_dir)
    try:
        parameters: list[Any] = [partition]
        limit_clause = ""
        if query_limit is not None:
            limit_clause = "AND s.query_rank < ?"
            parameters.append(query_limit)
        rows = connection.execute(
            f"""
            SELECT o.payload
            FROM outcomes AS o
            JOIN sample_rows AS s ON s.query_id = o.query_id
            WHERE o.partition = ? {limit_clause}
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'no_context' THEN 0 WHEN 'bm25' THEN 1 ELSE 2 END,
                     o.repeat_id
            """,
            parameters,
        )
        suffix = "all" if query_limit is None else str(query_limit)
        path = run_dir / f"{partition}_results_{suffix}.jsonl"
        count = 0
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for (payload,) in rows:
                handle.write(str(payload) + "\n")
                count += 1
        result = {"path": str(path), "rows": count}
        _write_json(run_dir / f"{partition}_export_{suffix}.json", result)
        return result
    finally:
        connection.close()


def main() -> int:
    args = _arguments()
    config_path = _resolve(args.config)
    router = _load_router_config(config_path)
    _phase3(router)
    run_dir = _run_dir(config_path, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample = freeze_sample(router, config_path, run_dir)

    if args.stage == "sample":
        result: Any = {
            key: value for key, value in sample.items() if key != "rows"
        }
    elif args.stage == "prepare":
        result = prepare_partition(
            router,
            config_path,
            run_dir,
            sample,
            args.partition,
            train_size=args.train_size,
        )
    elif args.stage == "generate":
        result = generate_partition(
            router,
            run_dir,
            sample,
            args.partition,
            train_size=args.train_size,
            max_new_calls=args.max_new_calls,
        )
    elif args.stage == "export":
        result = export_results(
            run_dir,
            args.partition,
            query_limit=args.train_size if args.partition == "train" else None,
        )
    else:
        result = status(
            run_dir,
            partition=args.partition,
            query_limit=args.train_size if args.partition == "train" else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
