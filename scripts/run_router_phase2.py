"""Run the frozen Phase 2 retrieval-policy answer-utility headroom experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import (
    GENERATOR_PRICES,
    JUDGE_PRICES,
    _base_row,
    _checkpoint_path,
    _core_config,
    _deduplicated_hits,
    _judge_call,
    _load_checkpoint,
    _load_router_config,
    _prompt_for_row,
    _read_json,
    _resolve_project_path,
    _usage_cost,
    _write_all_rows,
)
from src.evaluators.beir_evaluation import compute_streaming_first_stage
from src.evaluators.beir_suite import compute_bm25_first_stage
from src.evaluators.hotpot_answer import answer_metrics, normalize_answer
from src.persistence.run_output_writer import (
    write_metadata_json,
    write_result_checkpoint,
    write_results,
)
from src.pipeline import NaiveRAGPipeline
from src.text.token_counters import RegexTokenCounter


ACTIONS = ("bm25", "dense")
AC_PRIMARY_ACTIONS = ("no_context", "bm25", "dense")
PARTITIONS = ("train", "dev")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
    parser.add_argument("--run-id", default="phase2_bd_headroom_v1")
    parser.add_argument(
        "--revision",
        action="store_true",
        help="Use the frozen all-query repeat protocol in phase2_revision.",
    )
    parser.add_argument(
        "--ac-primary",
        action="store_true",
        help="Reuse repeated B/D outcomes, add no-context, and use AC as primary utility.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "judge", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-new-calls",
        type=int,
        default=None,
        help="Limit new provider calls in this invocation for a resumable smoke test.",
    )
    return parser.parse_args()


def _sample_path(run_dir: Path) -> Path:
    return run_dir / "sample.json"


def freeze_sample(router: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    path = _sample_path(run_dir)
    if path.is_file():
        return _read_json(path)
    split = _read_json(_resolve_project_path(str(router["split"]["path"])))
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("split.json has no assignments list")
    phase2 = router["phase2"]
    rng = np.random.default_rng(int(phase2["sample_seed"]))
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for partition in PARTITIONS:
        candidates = [row for row in assignments if row.get("partition") == partition]
        requested = int(phase2["sample_without_replacement"][partition])
        if requested > len(candidates):
            raise ValueError(f"Phase 2 {partition} sample exceeds the frozen partition")
        positions = set(int(value) for value in rng.choice(len(candidates), requested, replace=False))
        rows = [
            {
                "query_id": str(row["query_id"]),
                "group_id": str(row["group_id"]),
                "partition": partition,
            }
            for position, row in enumerate(candidates)
            if position in positions
        ]
        selected.extend(rows)
        counts[partition] = len(rows)
        group_counts[partition] = len({row["group_id"] for row in rows})
    if any(row["partition"] == "final_holdout" for row in selected):
        raise RuntimeError("Phase 2 sample must not contain final_holdout")
    sample = {
        "seed": int(phase2["sample_seed"]),
        "selection": "uniform_query_without_replacement_within_partition",
        "counts": counts,
        "group_counts": group_counts,
        "rows": selected,
    }
    write_metadata_json(path, sample, overwrite=False)
    return sample


def _load_examples(
    router: Mapping[str, Any], sample: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_rows = sample["rows"]
    selected = {str(row["query_id"]): row for row in selected_rows}
    if len(selected) != len(selected_rows):
        raise ValueError("Phase 2 sampled query ids must be unique")
    dataset_root = _resolve_project_path(str(router["dataset"]["root"]))
    examples: dict[str, dict[str, Any]] = {}
    with (dataset_root / "queries" / "queries.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row.get("_id", ""))
            if query_id not in selected:
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            answer = metadata.get("answer")
            references = answer if isinstance(answer, list) else [answer]
            supporting_facts = metadata.get("supporting_facts")
            if (
                not isinstance(row.get("text"), str)
                or not references
                or any(not isinstance(value, str) or not value for value in references)
                or not isinstance(supporting_facts, list)
            ):
                continue
            titles: list[str] = []
            for fact in supporting_facts:
                if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str):
                    raise ValueError(f"Malformed supporting fact for {query_id}")
                if fact[0] not in titles:
                    titles.append(fact[0])
            selection = selected[query_id]
            examples[query_id] = {
                "dataset_id": str(router["dataset"]["dataset_id"]),
                "split": str(selection["partition"]),
                "query_id": query_id,
                "group_id": str(selection["group_id"]),
                "question": row["text"],
                "reference_answers": list(references),
                "supporting_titles": titles,
                "gold_doc_ids": [],
            }
    with (dataset_root / "qrels" / "train.tsv").open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError("Unexpected train qrels header")
        for line in handle:
            query_id, corpus_id, score = line.rstrip("\r\n").split("\t")
            if query_id in examples and float(score) > 0.0:
                examples[query_id]["gold_doc_ids"].append(corpus_id)
    missing = sorted(set(selected) - set(examples))
    missing_evidence = sorted(
        query_id for query_id, row in examples.items() if not row["gold_doc_ids"]
    )
    if missing or missing_evidence:
        raise ValueError(
            f"Phase 2 mapping incomplete: missing={len(missing)}, "
            f"missing_evidence={len(missing_evidence)}"
        )
    ordered = [examples[str(row["query_id"])] for row in selected_rows]
    return ordered, {
        "sampled": len(selected_rows),
        "mapped": len(ordered),
        "mapping_failures": 0,
        "excluded": 0,
    }


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "checkpoints").glob("*.json"))
    if not paths:
        raise FileNotFoundError("No Phase 2 checkpoints exist")
    return [_load_checkpoint(path) for path in paths]


def _revision_router(router: Mapping[str, Any]) -> dict[str, Any]:
    revision = router.get("phase2_revision")
    if not isinstance(revision, Mapping):
        raise ValueError("config has no phase2_revision mapping")
    merged = copy.deepcopy(dict(router))
    merged["phase2"] = {**dict(router["phase2"]), **dict(revision)}
    return merged


def _ac_primary_router(router: Mapping[str, Any]) -> dict[str, Any]:
    revision = router.get("phase2_ac_primary")
    if not isinstance(revision, Mapping):
        raise ValueError("config has no phase2_ac_primary mapping")
    merged = copy.deepcopy(dict(router))
    merged["phase2"] = {**dict(router["phase2"]), **dict(revision)}
    return merged


def _pending_repeat(template: Mapping[str, Any], repeat_id: int) -> dict[str, Any]:
    row = copy.deepcopy(dict(template))
    row["repeat_id"] = repeat_id
    row["status"] = "pending_generation"
    for field in ("prediction", "metrics", "generation", "answer_correctness"):
        row.pop(field, None)
    return row


def prepare_revision(
    router: Mapping[str, Any], output_root: Path, run_dir: Path
) -> list[dict[str, Any]]:
    revision = router["phase2_revision"]
    repeat_count = int(revision["repeats_per_query_action"])
    expected_queries = sum(
        int(router["phase2"]["sample_without_replacement"][partition])
        for partition in PARTITIONS
    )
    expected_rows = expected_queries * len(ACTIONS) * repeat_count
    existing = sorted((run_dir / "checkpoints").glob("*.json"))
    if len(existing) == expected_rows:
        return [_load_checkpoint(path) for path in existing]
    if existing:
        raise RuntimeError("Partial Phase 2 revision checkpoints exist")

    sample = freeze_sample(router, run_dir)
    source_dir = output_root / "runs" / str(revision["source_run"])
    source_sample = _read_json(_sample_path(source_dir))
    if sample.get("rows") != source_sample.get("rows"):
        raise ValueError("Phase 2 revision sample differs from its source run")
    source_rows = _load_rows(source_dir)
    source_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in source_rows:
        key = (
            str(row["split"]),
            str(row["query_id"]),
            str(row["action"]),
            int(row.get("repeat_id", 0)),
        )
        if key in source_by_key:
            raise ValueError(f"Duplicate source repeat row: {key}")
        source_by_key[key] = row

    rows: list[dict[str, Any]] = []
    for sampled in sample["rows"]:
        partition = str(sampled["partition"])
        query_id = str(sampled["query_id"])
        for action in ACTIONS:
            template = source_by_key.get((partition, query_id, action, 0))
            if template is None or template.get("status") != "success":
                raise ValueError(
                    f"Missing successful source row for {(partition, query_id, action)}"
                )
            for repeat_id in range(repeat_count):
                source = source_by_key.get(
                    (partition, query_id, action, repeat_id)
                )
                if source is None:
                    rows.append(_pending_repeat(template, repeat_id))
                else:
                    value = copy.deepcopy(source)
                    value["repeat_id"] = repeat_id
                    rows.append(value)
    if len(rows) != expected_rows:
        raise ValueError(f"Prepared {len(rows)} rows, expected {expected_rows}")
    if any(row["split"] == "final_holdout" for row in rows):
        raise RuntimeError("Phase 2 revision must not contain final_holdout")
    for position, row in enumerate(rows):
        write_result_checkpoint(
            _checkpoint_path(run_dir / "checkpoints", position), row
        )
    write_metadata_json(
        run_dir / "mapping_counts.json",
        {
            "sampled": expected_queries,
            "mapped": expected_queries,
            "mapping_failures": 0,
            "excluded": 0,
        },
    )
    _write_all_rows(run_dir, rows)
    return rows


def prepare_ac_primary(
    router: Mapping[str, Any], output_root: Path, run_dir: Path
) -> list[dict[str, Any]]:
    revision = router["phase2_ac_primary"]
    repeat_count = int(revision["repeats_per_query_action"])
    expected_queries = sum(
        int(router["phase2"]["sample_without_replacement"][partition])
        for partition in PARTITIONS
    )
    expected_rows = expected_queries * len(AC_PRIMARY_ACTIONS) * repeat_count
    existing = sorted((run_dir / "checkpoints").glob("*.json"))
    if len(existing) == expected_rows:
        return [_load_checkpoint(path) for path in existing]
    if len(existing) > expected_rows:
        raise RuntimeError("AC-primary Phase 2 has more checkpoints than expected")

    sample = freeze_sample(router, run_dir)
    source_dir = output_root / "runs" / str(revision["source_run"])
    source_sample = _read_json(_sample_path(source_dir))
    if sample.get("rows") != source_sample.get("rows"):
        raise ValueError("AC-primary Phase 2 sample differs from its source run")
    source_rows = _load_rows(source_dir)
    source_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in source_rows:
        key = (
            str(row["split"]),
            str(row["query_id"]),
            str(row["action"]),
            int(row.get("repeat_id", 0)),
        )
        if key in source_by_key:
            raise ValueError(f"Duplicate source repeat row: {key}")
        source_by_key[key] = row

    token_counter = RegexTokenCounter()
    context_max_tokens = int(router["context"]["max_tokens"])
    rows: list[dict[str, Any]] = []
    for sampled in sample["rows"]:
        partition = str(sampled["partition"])
        query_id = str(sampled["query_id"])
        template = source_by_key.get((partition, query_id, "bm25", 0))
        if template is None or template.get("status") != "success":
            raise ValueError(f"Missing source template for {(partition, query_id)}")
        example = {
            "dataset_id": template["dataset_id"],
            "split": template["split"],
            "query_id": template["query_id"],
            "group_id": template["group_id"],
            "question": template["question"],
            "reference_answers": template["reference_answers"],
            "supporting_titles": template["supporting_titles"],
            "gold_doc_ids": template["gold_doc_ids"],
        }
        no_context = _base_row(
            example,
            action="no_context",
            hits=(),
            retrieval_latency_ms=0.0,
            token_counter=token_counter,
            context_max_tokens=context_max_tokens,
        )
        for repeat_id in range(repeat_count):
            rows.append(_pending_repeat(no_context, repeat_id))
        for action in ACTIONS:
            for repeat_id in range(repeat_count):
                source = source_by_key.get(
                    (partition, query_id, action, repeat_id)
                )
                if source is None or source.get("status") != "success":
                    raise ValueError(
                        f"Missing successful source row for "
                        f"{(partition, query_id, action, repeat_id)}"
                    )
                rows.append(copy.deepcopy(source))

    if len(rows) != expected_rows:
        raise ValueError(f"Prepared {len(rows)} rows, expected {expected_rows}")
    for position, row in enumerate(rows):
        path = _checkpoint_path(run_dir / "checkpoints", position)
        if path.is_file():
            existing_row = _load_checkpoint(path)
            expected_key = (
                str(row["split"]),
                str(row["query_id"]),
                str(row["action"]),
                int(row.get("repeat_id", 0)),
            )
            existing_key = (
                str(existing_row["split"]),
                str(existing_row["query_id"]),
                str(existing_row["action"]),
                int(existing_row.get("repeat_id", 0)),
            )
            if existing_key != expected_key:
                raise ValueError(
                    f"Partial checkpoint key mismatch at {position}: "
                    f"{existing_key} != {expected_key}"
                )
            continue
        write_result_checkpoint(path, row)
    _write_all_rows(run_dir, rows)
    return rows


def prepare(router: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    phase2 = router["phase2"]
    expected_main = sum(int(phase2["sample_without_replacement"][p]) for p in PARTITIONS) * 2
    expected_repeats = (
        int(phase2["sensitivity_repeat_queries_per_partition"])
        * len(PARTITIONS)
        * len(ACTIONS)
    )
    expected_rows = expected_main + expected_repeats
    existing = sorted((run_dir / "checkpoints").glob("*.json"))
    if len(existing) == expected_rows:
        return [_load_checkpoint(path) for path in existing]
    if existing:
        raise RuntimeError("Partial Phase 2 prepare checkpoints exist")

    sample = freeze_sample(router, run_dir)
    examples, mapping = _load_examples(router, sample)
    questions = [
        {"question_id": row["query_id"], "question": row["question"], "qrels": {}}
        for row in examples
    ]
    final_k = int(router["retrieval"]["final_k"])
    candidate_k = int(router["retrieval"]["candidate_k"])
    context_max_tokens = int(router["context"]["max_tokens"])
    token_counter = RegexTokenCounter()
    build_path = _resolve_project_path(str(router["artifacts"]["build"]))

    dense_config = _core_config(router, method="dense")
    with NaiveRAGPipeline(dense_config) as pipeline:
        if pipeline.build_dir != build_path:
            raise ValueError("Dense registry did not resolve the frozen build")
        started = time.perf_counter()
        batch = compute_streaming_first_stage(
            pipeline,
            questions,
            candidate_k=candidate_k,
            query_batch_size=min(100, len(questions)),
        )
        dense_total_ms = (time.perf_counter() - started) * 1000.0
        dense_hits = [
            _deduplicated_hits(
                pipeline.chunk_store,
                batch.vector_ids[position],
                batch.scores[position],
                final_k=final_k,
            )
            for position in range(len(examples))
        ]

    bm25_config = _core_config(router, method="bm25")
    with NaiveRAGPipeline(bm25_config) as pipeline:
        if pipeline.build_dir != build_path:
            raise ValueError("BM25 registry did not resolve the frozen build")
        batch = compute_bm25_first_stage(
            pipeline,
            questions,
            candidate_k=candidate_k,
            retained_k=candidate_k,
        )
        bm25_hits = []
        for position in range(len(examples)):
            scores, vector_ids = batch.row(position)
            bm25_hits.append(
                _deduplicated_hits(
                    pipeline.chunk_store,
                    vector_ids,
                    scores,
                    final_k=final_k,
                )
            )

    rows: list[dict[str, Any]] = []
    for position, example in enumerate(examples):
        for action, hits, latency in (
            (
                "bm25",
                bm25_hits[position],
                float(batch.per_question_timings_ms[position].get("total_ms", 0.0)),
            ),
            ("dense", dense_hits[position], dense_total_ms / len(examples)),
        ):
            rows.append(
                _base_row(
                    example,
                    action=action,
                    hits=hits,
                    retrieval_latency_ms=latency,
                    token_counter=token_counter,
                    context_max_tokens=context_max_tokens,
                )
            )

    repeat_rng = np.random.default_rng(int(phase2["sample_seed"]) + 1)
    repeat_ids: set[str] = set()
    repeat_count = int(phase2["sensitivity_repeat_queries_per_partition"])
    for partition in PARTITIONS:
        candidates = [row["query_id"] for row in examples if row["split"] == partition]
        repeat_ids.update(
            str(value)
            for value in repeat_rng.choice(candidates, repeat_count, replace=False)
        )
    repeats: list[dict[str, Any]] = []
    for row in rows:
        if row["query_id"] in repeat_ids:
            row["repeat_id"] = 0
            duplicate = copy.deepcopy(row)
            duplicate["repeat_id"] = 1
            repeats.append(duplicate)
    rows.extend(repeats)
    if len(rows) != expected_rows:
        raise ValueError(f"Prepared {len(rows)} rows, expected {expected_rows}")
    for position, row in enumerate(rows):
        write_result_checkpoint(_checkpoint_path(run_dir / "checkpoints", position), row)
    write_metadata_json(run_dir / "mapping_counts.json", mapping)
    _write_all_rows(run_dir, rows)
    return rows


def _generator_view(result: Any) -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "success",
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
    }


def generate_parallel(
    router: Mapping[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_calls: int | None,
) -> list[dict[str, Any]]:
    from src.generators.answer_generator import LLMGenerator

    phase2 = router["phase2"]
    generation = _core_config(router, method="bm25")["generation"]
    call_limit = int(phase2["generation_call_limit"])
    budget = float(phase2["generation_hard_budget_usd"])
    attempted = sum(
        bool(row.get("generation", {}).get("attempted"))
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    spent = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    pending = [
        position
        for position, row in enumerate(rows)
        if not isinstance(row.get("generation"), Mapping)
        or not row["generation"].get("attempted")
    ]
    remaining = max(0, call_limit - attempted)
    if max_new_calls is not None:
        remaining = min(remaining, max_new_calls)
    pending = pending[:remaining]
    if not pending:
        return rows

    local = threading.local()
    created: list[LLMGenerator] = []
    created_lock = threading.Lock()

    def worker(position: int) -> tuple[int, dict[str, Any]]:
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
        row = rows[position]
        try:
            result = generator.generate_from_prompt(
                _prompt_for_row(row, str(router["prompt"]["version"])),
                row["question"],
                [],
            )
            prediction = result.answer.strip()
            return position, {
                "status": "pending_answer_correctness",
                "prediction": prediction,
                "metrics": answer_metrics(prediction, row["reference_answers"]),
                "generation": _generator_view(result),
            }
        except Exception as exc:
            return position, {
                "status": "generation_failure",
                "generation": {
                    "attempted": True,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=int(phase2["provider_workers"])) as pool:
            futures = [pool.submit(worker, position) for position in pending]
            for future in as_completed(futures):
                position, update = future.result()
                rows[position].update(update)
                completed += 1
                if update["generation"]["status"] == "success":
                    spent += _usage_cost(
                        update["generation"]["token_usage"], GENERATOR_PRICES
                    )
                write_result_checkpoint(
                    _checkpoint_path(run_dir / "checkpoints", position), rows[position]
                )
                if completed % 50 == 0:
                    print(
                        json.dumps(
                            {"generation_new_calls": completed, "estimated_cost_usd": spent}
                        ),
                        flush=True,
                    )
    finally:
        for generator in created:
            close = getattr(getattr(generator, "_client", None), "close", None)
            if callable(close):
                close()
    if spent > budget:
        raise RuntimeError(f"Generation budget exceeded: {spent:.6f} > {budget:.6f}")
    _write_all_rows(run_dir, rows)
    return rows


def _run_judge_jobs(
    router: Mapping[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    positions: Sequence[int],
    *,
    append_repeat: bool,
) -> tuple[int, float]:
    from openai import OpenAI

    phase2 = router["phase2"]
    local = threading.local()
    clients: list[Any] = []
    clients_lock = threading.Lock()

    def worker(position: int) -> tuple[int, dict[str, Any], float]:
        client = getattr(local, "client", None)
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=2)
            local.client = client
            with clients_lock:
                clients.append(client)
        try:
            result, cost = _judge_call(client, router, rows[position])
            return position, result, cost
        except Exception as exc:
            return position, {
                "attempted": True,
                "status": "failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, 0.0

    completed = 0
    spent = 0.0
    try:
        with ThreadPoolExecutor(max_workers=int(phase2["provider_workers"])) as pool:
            futures = [pool.submit(worker, position) for position in positions]
            for future in as_completed(futures):
                position, result, cost = future.result()
                values = rows[position].setdefault("answer_correctness", [])
                values.append(result)
                if not append_repeat:
                    if result["status"] == "success":
                        rows[position]["metrics"]["answer_correctness"] = result["score"]
                        rows[position]["status"] = "success"
                    else:
                        rows[position]["status"] = "answer_correctness_failure"
                completed += 1
                spent += cost
                write_result_checkpoint(
                    _checkpoint_path(run_dir / "checkpoints", position), rows[position]
                )
                if completed % 50 == 0:
                    print(
                        json.dumps(
                            {"judge_new_calls": completed, "estimated_cost_usd": spent}
                        ),
                        flush=True,
                    )
    finally:
        for client in clients:
            client.close()
    return completed, spent


def judge_parallel(
    router: Mapping[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_calls: int | None,
) -> list[dict[str, Any]]:
    phase2 = router["phase2"]
    limit = int(phase2["answer_correctness_call_limit"])
    budget = float(phase2["answer_correctness_hard_budget_usd"])
    attempted = sum(len(row.get("answer_correctness", [])) for row in rows)
    spent = sum(
        _usage_cost(value.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for value in row.get("answer_correctness", [])
        if isinstance(value, Mapping)
    )
    allowance = max(0, limit - attempted)
    if max_new_calls is not None:
        allowance = min(allowance, max_new_calls)
    primary = [
        position
        for position, row in enumerate(rows)
        if row.get("generation", {}).get("status") == "success"
        and not row.get("answer_correctness")
    ][:allowance]
    completed, added_cost = _run_judge_jobs(
        router, run_dir, rows, primary, append_repeat=False
    )
    allowance -= completed
    spent += added_cost

    if allowance > 0 and not any(
        row.get("generation", {}).get("status") == "success"
        and not row.get("answer_correctness")
        for row in rows
    ):
        repeat_candidates: list[int] = []
        for partition in PARTITIONS:
            query_ids: list[str] = []
            for row in rows:
                if (
                    row["split"] == partition
                    and row.get("repeat_id") == 0
                    and row["query_id"] not in query_ids
                ):
                    query_ids.append(row["query_id"])
                if len(query_ids) >= 10:
                    break
            selected = set(query_ids)
            repeat_candidates.extend(
                position
                for position, row in enumerate(rows)
                if row["split"] == partition
                and row.get("repeat_id") == 0
                and row["query_id"] in selected
                and len(row.get("answer_correctness", [])) == 1
            )
        repeat_candidates = repeat_candidates[:allowance]
        _, added_cost = _run_judge_jobs(
            router,
            run_dir,
            rows,
            repeat_candidates,
            append_repeat=True,
        )
        spent += added_cost
    if spent > budget:
        raise RuntimeError(f"AC budget exceeded: {spent:.6f} > {budget:.6f}")
    _write_all_rows(run_dir, rows)
    return rows


def select_best_fixed(
    records: Sequence[Mapping[str, Any]],
    tie_break: str,
    *,
    actions: Sequence[str] = ACTIONS,
    primary_metric: str = "f1",
) -> str:
    if not records:
        raise ValueError("Cannot select a fixed action from no records")
    action_order = tuple(str(action) for action in actions)
    if tie_break not in action_order:
        raise ValueError("tie_break must be one of the candidate actions")
    means = {
        action: float(
            np.mean([row[f"{primary_metric}_{action}"] for row in records])
        )
        for action in action_order
    }
    best_value = max(means.values())
    tied = tuple(action for action in action_order if means[action] == best_value)
    if tie_break in tied:
        return tie_break
    return tied[0]


def compute_headroom_point(
    records: Sequence[Mapping[str, Any]],
    tie_break: str,
    *,
    actions: Sequence[str] = ACTIONS,
    primary_metric: str = "f1",
) -> dict[str, Any]:
    action_order = tuple(str(action) for action in actions)
    fixed = select_best_fixed(
        records,
        tie_break,
        actions=action_order,
        primary_metric=primary_metric,
    )
    oracle_tie_order = (tie_break, *[a for a in action_order if a != tie_break])
    chosen: list[str] = []
    for row in records:
        best_value = max(row[f"{primary_metric}_{action}"] for action in action_order)
        chosen.append(
            next(
                action
                for action in oracle_tie_order
                if row[f"{primary_metric}_{action}"] == best_value
            )
        )
    result: dict[str, Any] = {
        "best_fixed": fixed,
        "primary_metric": primary_metric,
        "chosen_actions": chosen,
    }
    for metric in ("ac", "f1", "em"):
        fixed_value = float(np.mean([row[f"{metric}_{fixed}"] for row in records]))
        oracle_value = float(
            np.mean(
                [row[f"{metric}_{action}"] for row, action in zip(records, chosen)]
            )
        )
        delta = float(
            np.mean(
                [
                    row[f"{metric}_{action}"] - row[f"{metric}_{fixed}"]
                    for row, action in zip(records, chosen)
                ]
            )
        )
        result[f"fixed_{metric}"] = fixed_value
        result[f"{metric}_for_{primary_metric}_oracle"] = oracle_value
        if metric == primary_metric:
            result[f"oracle_{metric}"] = oracle_value
            result[f"{metric}_headroom"] = delta
        else:
            result[f"{metric}_delta_for_{primary_metric}_oracle"] = delta
            independent = float(
                np.mean(
                    [
                        max(row[f"{metric}_{action}"] for action in action_order)
                        - row[f"{metric}_{fixed}"]
                        for row in records
                    ]
                )
            )
            result[f"independent_{metric}_headroom"] = independent
    return result


def _utility_records(
    rows: Sequence[Mapping[str, Any]],
    partition: str,
    *,
    actions: Sequence[str] = ACTIONS,
) -> list[dict[str, Any]]:
    action_order = tuple(str(action) for action in actions)
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == partition and row["action"] in action_order:
            grouped[(str(row["query_id"]), str(row["action"]))].append(row)
    query_ids = sorted({query_id for query_id, _ in grouped})
    records: list[dict[str, Any]] = []
    for query_id in query_ids:
        action_rows = {action: grouped[(query_id, action)] for action in action_order}
        if any(any(row.get("status") != "success" for row in values) for values in action_rows.values()):
            continue
        record: dict[str, Any] = {
            "query_id": query_id,
            "group_id": str(action_rows[action_order[0]][0]["group_id"]),
        }
        for action, values in action_rows.items():
            record[f"f1_{action}"] = float(
                np.mean([row["metrics"]["normalized_token_f1"] for row in values])
            )
            record[f"em_{action}"] = float(
                np.mean([row["metrics"]["normalized_exact_match"] for row in values])
            )
            record[f"ac_{action}"] = float(
                np.mean([row["metrics"]["answer_correctness"] for row in values])
            )
        records.append(record)
    return records


def _bootstrap_headroom(
    records: Sequence[Mapping[str, Any]],
    *,
    tie_break: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    by_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row["group_id"])].append(row)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    f1 = np.empty(resamples, dtype=np.float64)
    ac = np.empty(resamples, dtype=np.float64)
    em = np.empty(resamples, dtype=np.float64)
    fixed_counts = {action: 0 for action in ACTIONS}
    for position in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = [row for group in sampled_groups for row in by_group[str(group)]]
        point = compute_headroom_point(sampled, tie_break)
        f1[position] = point["f1_headroom"]
        ac[position] = point["ac_delta_for_f1_oracle"]
        em[position] = point["em_delta_for_f1_oracle"]
        fixed_counts[str(point["best_fixed"])] += 1
    return {
        "f1_ci95": [float(value) for value in np.quantile(f1, [0.025, 0.975])],
        "ac_ci95": [float(value) for value in np.quantile(ac, [0.025, 0.975])],
        "em_ci95": [float(value) for value in np.quantile(em, [0.025, 0.975])],
        "best_fixed_selection_fraction": {
            action: count / resamples for action, count in fixed_counts.items()
        },
    }


def _repeat_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["split"], row["query_id"], row["action"], row.get("repeat_id", 0)): row
        for row in rows
    }
    pairs = []
    for key, row in by_key.items():
        if key[3] != 0:
            continue
        pair = by_key.get((key[0], key[1], key[2], 1))
        if pair is not None and row.get("status") == pair.get("status") == "success":
            pairs.append((row, pair))
    judge_pairs = [
        row["answer_correctness"][:2]
        for row in rows
        if len(row.get("answer_correctness", [])) >= 2
        and all(item.get("status") == "success" for item in row["answer_correctness"][:2])
    ]
    return {
        "generation_repeat_pairs": len(pairs),
        "normalized_answer_agreement": float(
            np.mean(
                [
                    normalize_answer(left["prediction"])
                    == normalize_answer(right["prediction"])
                    for left, right in pairs
                ]
            )
        ),
        "f1_agreement": float(
            np.mean(
                [
                    left["metrics"]["normalized_token_f1"]
                    == right["metrics"]["normalized_token_f1"]
                    for left, right in pairs
                ]
            )
        ),
        "em_agreement": float(
            np.mean(
                [
                    left["metrics"]["normalized_exact_match"]
                    == right["metrics"]["normalized_exact_match"]
                    for left, right in pairs
                ]
            )
        ),
        "ac_score_agreement": float(
            np.mean(
                [
                    left["metrics"]["answer_correctness"]
                    == right["metrics"]["answer_correctness"]
                    for left, right in pairs
                ]
            )
        ),
        "judge_repeat_pairs": len(judge_pairs),
        "judge_label_agreement": float(
            np.mean([left["label"] == right["label"] for left, right in judge_pairs])
        ),
    }


def _row_ac_score(row: Mapping[str, Any]) -> float:
    value = row.get("metrics", {}).get("answer_correctness")
    if value is None:
        raise ValueError("A successful generation row has no successful AC score")
    return float(value)


def _repeat_cells(
    rows: Sequence[Mapping[str, Any]],
    partition: str,
    *,
    actions: Sequence[str] = ACTIONS,
) -> list[dict[str, Any]]:
    action_order = tuple(str(action) for action in actions)
    grouped: defaultdict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["split"] != partition or row["action"] not in action_order:
            continue
        key = (str(row["query_id"]), str(row["action"]))
        repeat_id = int(row.get("repeat_id", 0))
        if repeat_id in grouped[key]:
            raise ValueError(f"Duplicate repeat_id for {key}: {repeat_id}")
        grouped[key][repeat_id] = row

    records: list[dict[str, Any]] = []
    for query_id in sorted({query_id for query_id, _ in grouped}):
        action_rows = {action: grouped[(query_id, action)] for action in action_order}
        repeat_ids = tuple(sorted(action_rows[action_order[0]]))
        if not repeat_ids or any(
            tuple(sorted(action_rows[action])) != repeat_ids
            for action in action_order[1:]
        ):
            continue
        if any(
            row.get("status") != "success"
            for values in action_rows.values()
            for row in values.values()
        ):
            continue
        record: dict[str, Any] = {
            "query_id": query_id,
            "group_id": str(
                action_rows[action_order[0]][repeat_ids[0]]["group_id"]
            ),
            "repeat_ids": repeat_ids,
        }
        for action, values in action_rows.items():
            record[f"f1_{action}"] = {
                repeat_id: float(values[repeat_id]["metrics"]["normalized_token_f1"])
                for repeat_id in repeat_ids
            }
            record[f"em_{action}"] = {
                repeat_id: float(values[repeat_id]["metrics"]["normalized_exact_match"])
                for repeat_id in repeat_ids
            }
            record[f"ac_{action}"] = {
                repeat_id: _row_ac_score(values[repeat_id])
                for repeat_id in repeat_ids
            }
        records.append(record)
    return records


def _mean_repeat_records(
    cells: Sequence[Mapping[str, Any]],
    allowed_repeat_ids: Sequence[int] | None = None,
    *,
    actions: Sequence[str] = ACTIONS,
) -> list[dict[str, Any]]:
    action_order = tuple(str(action) for action in actions)
    records: list[dict[str, Any]] = []
    for cell in cells:
        repeat_ids = (
            tuple(int(value) for value in allowed_repeat_ids)
            if allowed_repeat_ids is not None
            else tuple(int(value) for value in cell["repeat_ids"])
        )
        if not repeat_ids or any(value not in cell["repeat_ids"] for value in repeat_ids):
            raise ValueError("Requested repeat ids are unavailable")
        record: dict[str, Any] = {
            "query_id": cell["query_id"],
            "group_id": cell["group_id"],
        }
        for action in action_order:
            for metric in ("f1", "em", "ac"):
                values = cell[f"{metric}_{action}"]
                record[f"{metric}_{action}"] = float(
                    np.mean([values[repeat_id] for repeat_id in repeat_ids])
                )
        records.append(record)
    return records


def _nested_bootstrap_headroom(
    cells: Sequence[Mapping[str, Any]],
    *,
    tie_break: str,
    resamples: int,
    seed: int,
    allowed_repeat_ids: Sequence[int] | None = None,
    actions: Sequence[str] = ACTIONS,
    primary_metric: str = "f1",
) -> dict[str, Any]:
    action_order = tuple(str(action) for action in actions)
    by_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_group[str(cell["group_id"])].append(cell)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("No complete repeated utility cells exist")
    rng = np.random.default_rng(seed)
    if all(len(by_group[group]) == 1 for group in groups):
        ordered_cells = [by_group[group][0] for group in groups]
        repeat_ids = (
            tuple(int(value) for value in allowed_repeat_ids)
            if allowed_repeat_ids is not None
            else tuple(int(value) for value in ordered_cells[0]["repeat_ids"])
        )
        if not repeat_ids or any(
            any(repeat_id not in cell["repeat_ids"] for repeat_id in repeat_ids)
            for cell in ordered_cells
        ):
            raise ValueError("Bootstrap repeat ids are unavailable")
        metric_values: dict[str, np.ndarray] = {}
        for metric in ("f1", "em", "ac"):
            metric_values[metric] = np.asarray(
                [
                    [
                        [
                            float(cell[f"{metric}_{action}"][repeat_id])
                            for repeat_id in repeat_ids
                        ]
                        for action in action_order
                    ]
                    for cell in ordered_cells
                ],
                dtype=np.float64,
            )
        query_count = len(ordered_cells)
        action_count = len(action_order)
        repeat_count = len(repeat_ids)
        sampled_queries = rng.integers(
            0, query_count, size=(resamples, query_count)
        )
        sampled_repeats = rng.integers(
            0,
            repeat_count,
            size=(resamples, query_count, action_count, repeat_count),
        )
        action_indices = np.arange(action_count)[None, None, :, None]
        query_indices = sampled_queries[:, :, None, None]
        sampled_means = {
            metric: values[
                query_indices,
                action_indices,
                sampled_repeats,
            ].mean(axis=3)
            for metric, values in metric_values.items()
        }
        priority_actions = (
            tie_break,
            *[action for action in action_order if action != tie_break],
        )
        priority_indices = np.asarray(
            [action_order.index(action) for action in priority_actions],
            dtype=np.int64,
        )
        primary = sampled_means[primary_metric]
        fixed_priority = np.argmax(
            primary[:, :, priority_indices].mean(axis=1), axis=1
        )
        fixed_indices = priority_indices[fixed_priority]
        oracle_priority = np.argmax(primary[:, :, priority_indices], axis=2)
        oracle_indices = priority_indices[oracle_priority]
        deltas: dict[str, np.ndarray] = {}
        for metric, values in sampled_means.items():
            oracle_values = np.take_along_axis(
                values, oracle_indices[:, :, None], axis=2
            )[:, :, 0]
            fixed_lookup = np.broadcast_to(
                fixed_indices[:, None, None], (resamples, query_count, 1)
            )
            fixed_values = np.take_along_axis(values, fixed_lookup, axis=2)[:, :, 0]
            deltas[metric] = (oracle_values - fixed_values).mean(axis=1)
        fixed_counts_array = np.bincount(
            fixed_indices, minlength=action_count
        )
        return {
            "f1_ci95": [
                float(value) for value in np.quantile(deltas["f1"], [0.025, 0.975])
            ],
            "ac_ci95": [
                float(value) for value in np.quantile(deltas["ac"], [0.025, 0.975])
            ],
            "em_ci95": [
                float(value) for value in np.quantile(deltas["em"], [0.025, 0.975])
            ],
            "best_fixed_selection_fraction": {
                action: float(fixed_counts_array[index] / resamples)
                for index, action in enumerate(action_order)
            },
        }
    f1 = np.empty(resamples, dtype=np.float64)
    ac = np.empty(resamples, dtype=np.float64)
    em = np.empty(resamples, dtype=np.float64)
    fixed_counts = {action: 0 for action in action_order}
    for position in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled: list[dict[str, Any]] = []
        for group in sampled_groups:
            for cell in by_group[str(group)]:
                repeat_ids = (
                    tuple(int(value) for value in allowed_repeat_ids)
                    if allowed_repeat_ids is not None
                    else tuple(int(value) for value in cell["repeat_ids"])
                )
                value: dict[str, Any] = {
                    "query_id": cell["query_id"],
                    "group_id": cell["group_id"],
                }
                for action in action_order:
                    sampled_ids = rng.choice(
                        repeat_ids, size=len(repeat_ids), replace=True
                    )
                    for metric in ("f1", "em", "ac"):
                        outcomes = cell[f"{metric}_{action}"]
                        value[f"{metric}_{action}"] = float(
                            np.mean([outcomes[int(item)] for item in sampled_ids])
                        )
                sampled.append(value)
        point = compute_headroom_point(
            sampled,
            tie_break,
            actions=action_order,
            primary_metric=primary_metric,
        )
        f1[position] = (
            point["f1_headroom"]
            if primary_metric == "f1"
            else point[f"f1_delta_for_{primary_metric}_oracle"]
        )
        ac[position] = (
            point["ac_headroom"]
            if primary_metric == "ac"
            else point[f"ac_delta_for_{primary_metric}_oracle"]
        )
        em[position] = (
            point["em_headroom"]
            if primary_metric == "em"
            else point[f"em_delta_for_{primary_metric}_oracle"]
        )
        fixed_counts[str(point["best_fixed"])] += 1
    return {
        "f1_ci95": [float(value) for value in np.quantile(f1, [0.025, 0.975])],
        "ac_ci95": [float(value) for value in np.quantile(ac, [0.025, 0.975])],
        "em_ci95": [float(value) for value in np.quantile(em, [0.025, 0.975])],
        "best_fixed_selection_fraction": {
            action: count / resamples for action, count in fixed_counts.items()
        },
    }


def _all_repeat_diagnostics(
    rows: Sequence[Mapping[str, Any]], repeat_count: int
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["query_id"]), str(row["action"]))].append(row)
    complete = [
        sorted(values, key=lambda row: int(row.get("repeat_id", 0)))
        for values in grouped.values()
        if len(values) == repeat_count
        and all(row.get("status") == "success" for row in values)
    ]
    pairs = [pair for values in complete for pair in combinations(values, 2)]
    judge_pairs = [
        row["answer_correctness"][:2]
        for row in rows
        if len(row.get("answer_correctness", [])) >= 2
        and all(item.get("status") == "success" for item in row["answer_correctness"][:2])
    ]
    return {
        "complete_query_action_cells": len(complete),
        "generation_repeat_pairs": len(pairs),
        "normalized_answer_pair_agreement": float(
            np.mean(
                [
                    normalize_answer(left["prediction"])
                    == normalize_answer(right["prediction"])
                    for left, right in pairs
                ]
            )
        ),
        "f1_pair_agreement": float(
            np.mean(
                [
                    left["metrics"]["normalized_token_f1"]
                    == right["metrics"]["normalized_token_f1"]
                    for left, right in pairs
                ]
            )
        ),
        "em_pair_agreement": float(
            np.mean(
                [
                    left["metrics"]["normalized_exact_match"]
                    == right["metrics"]["normalized_exact_match"]
                    for left, right in pairs
                ]
            )
        ),
        "ac_pair_agreement": float(
            np.mean([_row_ac_score(left) == _row_ac_score(right) for left, right in pairs])
        ),
        "judge_repeat_pairs": len(judge_pairs),
        "judge_label_agreement": float(
            np.mean([left["label"] == right["label"] for left, right in judge_pairs])
        ),
        "role": "diagnostic_after_nondeterminism_trigger",
    }


def _write_action_table(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cell_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        cell_counts[(str(row["split"]), str(row["query_id"]), str(row["action"]))] += 1
    phase2_rows = []
    for row in rows:
        value = {
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "query_id": row["query_id"],
            "action": row["action"],
            "status": row["status"],
            "normalized_token_f1": row.get("metrics", {}).get("normalized_token_f1"),
            "normalized_exact_match": row.get("metrics", {}).get(
                "normalized_exact_match"
            ),
            "answer_correctness": row.get("metrics", {}).get("answer_correctness"),
            "evidence_page_recall": row["retrieval"]["evidence_page_recall"],
            "retrieval_hit": row["retrieval"]["hit"],
            "retrieval_latency_ms": row["retrieval"]["latency_ms"],
            "generation_latency_ms": row.get("generation", {}).get("latency_ms"),
            "token_usage": row.get("generation", {}).get("token_usage"),
        }
        key = (str(row["split"]), str(row["query_id"]), str(row["action"]))
        if cell_counts[key] > 1:
            value["repeat_id"] = int(row.get("repeat_id", 0))
        phase2_rows.append(value)
    action_path = output_root / "action_table.jsonl"
    pilot_rows = []
    if action_path.is_file():
        with action_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if value.get("split") == "pilot":
                    pilot_rows.append(value)
    write_results(action_path, [*pilot_rows, *phase2_rows])


def summarize(
    router: Mapping[str, Any], output_root: Path, run_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    phase2 = router["phase2"]
    tie_break = str(phase2["best_fixed_tie_break"])
    dev = _utility_records(rows, "dev")
    train = _utility_records(rows, "train")
    expected = {p: int(phase2["sample_without_replacement"][p]) for p in PARTITIONS}
    failures = {
        "generation": sum(row.get("status") == "generation_failure" for row in rows),
        "answer_correctness": sum(
            row.get("status") == "answer_correctness_failure" for row in rows
        ),
        "pending": sum(str(row.get("status", "")).startswith("pending") for row in rows),
    }
    point = compute_headroom_point(dev, tie_break)
    bootstrap = _bootstrap_headroom(
        dev,
        tie_break=tie_break,
        resamples=int(router["evaluation"]["bootstrap_resamples"]),
        seed=int(phase2["sample_seed"]),
    )
    margin = float(phase2["high_margin_f1_gap"])
    gaps = np.asarray([row["f1_dense"] - row["f1_bm25"] for row in dev])
    high_margin = {
        "threshold": margin,
        "bm25": int(np.sum(gaps <= -margin)),
        "dense": int(np.sum(gaps >= margin)),
        "ties_or_low_margin": int(np.sum(np.abs(gaps) < margin)),
    }
    fixed = point["best_fixed"]
    gains = sorted(
        [max(row["f1_bm25"], row["f1_dense"]) - row[f"f1_{fixed}"] for row in dev],
        reverse=True,
    )
    positive = [value for value in gains if value > 0.0]
    total_gain = sum(positive)
    concentration = {
        "positive_gain_queries": len(positive),
        "top1_share": None if total_gain == 0 else positive[0] / total_gain,
        "top5_share": None if total_gain == 0 else sum(positive[:5]) / total_gain,
    }
    stability = _repeat_stability(rows)
    action_summary: dict[str, Any] = {}
    for partition, records in (("train", train), ("dev", dev)):
        action_summary[partition] = {
            action: {
                "queries": len(records),
                "normalized_token_f1": float(
                    np.mean([row[f"f1_{action}"] for row in records])
                ),
                "normalized_exact_match": float(
                    np.mean([row[f"em_{action}"] for row in records])
                ),
                "answer_correctness": float(
                    np.mean([row[f"ac_{action}"] for row in records])
                ),
            }
            for action in ACTIONS
        }

    generation_cost = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    judge_cost = sum(
        _usage_cost(item.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for item in row.get("answer_correctness", [])
        if isinstance(item, Mapping)
    )
    practical = float(phase2["practical_f1_headroom"])
    noninferiority = float(router["evaluation"]["practical_effect"]["ac_noninferiority_margin"])
    support_minimum = int(phase2["minimum_high_margin_queries_per_action"])
    repeat_minimum = float(phase2["minimum_repeat_metric_agreement"])
    f1_pass = (
        point["f1_headroom"] >= practical
        and bootstrap["f1_ci95"][0] >= practical
    )
    ac_pass = (
        point["ac_delta_for_f1_oracle"] >= 0.0
        and bootstrap["ac_ci95"][0] >= -noninferiority
    )
    support_pass = (
        high_margin["bm25"] >= support_minimum
        and high_margin["dense"] >= support_minimum
    )
    stability_pass = (
        stability["f1_agreement"] >= repeat_minimum
        and stability["ac_score_agreement"] >= repeat_minimum
        and stability["judge_label_agreement"] >= repeat_minimum
    )
    complete = len(train) == expected["train"] and len(dev) == expected["dev"]
    no_failures = all(value == 0 for value in failures.values())
    if f1_pass and ac_pass and support_pass and stability_pass and complete and no_failures:
        decision = "GO"
    elif not f1_pass and complete and no_failures:
        decision = "STOP"
    else:
        decision = "REVISE"
    summary = {
        "phase": 2,
        "run_id": run_dir.name,
        "status": "complete" if complete and no_failures else "incomplete",
        "sample": {
            "train": len(train),
            "dev": len(dev),
            "final_holdout_outcomes": 0,
        },
        "actions": action_summary,
        "dev_headroom": {
            **{key: value for key, value in point.items() if key != "chosen_actions"},
            **bootstrap,
        },
        "high_margin_support": high_margin,
        "headroom_concentration": concentration,
        "stability": stability,
        "failures": failures,
        "estimated_cost_usd": {
            "generation": generation_cost,
            "answer_correctness": judge_cost,
            "total": generation_cost + judge_cost,
        },
        "gate": {
            "f1_headroom_passed": f1_pass,
            "ac_confirmatory_passed": ac_pass,
            "two_action_high_margin_support_passed": support_pass,
            "repeat_stability_passed": stability_pass,
            "complete_without_failures": complete and no_failures,
            "decision": decision,
        },
    }
    write_metadata_json(run_dir / "summary.json", summary)
    write_metadata_json(output_root / "summary.json", summary)
    _write_action_table(output_root, rows)
    return summary


def summarize_revision(
    router: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase2 = router["phase2"]
    revision = router["phase2_revision"]
    repeat_count = int(revision["repeats_per_query_action"])
    repeat_ids = tuple(range(repeat_count))
    tie_break = str(phase2["best_fixed_tie_break"])
    resamples = int(revision["bootstrap_resamples"])
    bootstrap_seed = int(revision["bootstrap_seed"])
    train_cells = _repeat_cells(rows, "train")
    dev_cells = _repeat_cells(rows, "dev")
    train = _mean_repeat_records(train_cells)
    dev = _mean_repeat_records(dev_cells)
    expected = {
        partition: int(phase2["sample_without_replacement"][partition])
        for partition in PARTITIONS
    }
    failures = {
        "generation": sum(row.get("status") == "generation_failure" for row in rows),
        "answer_correctness": sum(
            row.get("status") == "answer_correctness_failure" for row in rows
        ),
        "pending": sum(str(row.get("status", "")).startswith("pending") for row in rows),
    }
    point = compute_headroom_point(dev, tie_break)
    bootstrap = _nested_bootstrap_headroom(
        dev_cells,
        tie_break=tie_break,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    practical = float(phase2["practical_f1_headroom"])
    noninferiority = float(
        router["evaluation"]["practical_effect"]["ac_noninferiority_margin"]
    )

    leave_one_out: dict[str, Any] = {}
    sensitivity_pass = True
    for omitted in repeat_ids:
        allowed = tuple(value for value in repeat_ids if value != omitted)
        records = _mean_repeat_records(dev_cells, allowed)
        sensitivity_point = compute_headroom_point(records, tie_break)
        sensitivity_bootstrap = _nested_bootstrap_headroom(
            dev_cells,
            tie_break=tie_break,
            resamples=resamples,
            seed=bootstrap_seed + omitted + 1,
            allowed_repeat_ids=allowed,
        )
        passed = (
            sensitivity_point["best_fixed"] == point["best_fixed"]
            and sensitivity_point["f1_headroom"] >= practical
            and sensitivity_bootstrap["f1_ci95"][0] >= practical
            and sensitivity_point["ac_delta_for_f1_oracle"] >= 0.0
            and sensitivity_bootstrap["ac_ci95"][0] >= -noninferiority
        )
        sensitivity_pass = sensitivity_pass and passed
        leave_one_out[f"omit_{omitted}"] = {
            **{
                key: value
                for key, value in sensitivity_point.items()
                if key != "chosen_actions"
            },
            **sensitivity_bootstrap,
            "passed": passed,
        }

    margin = float(phase2["high_margin_f1_gap"])
    gaps = np.asarray([row["f1_dense"] - row["f1_bm25"] for row in dev])
    high_margin = {
        "threshold": margin,
        "bm25": int(np.sum(gaps <= -margin)),
        "dense": int(np.sum(gaps >= margin)),
        "ties_or_low_margin": int(np.sum(np.abs(gaps) < margin)),
    }
    fixed = point["best_fixed"]
    gains = sorted(
        [max(row["f1_bm25"], row["f1_dense"]) - row[f"f1_{fixed}"] for row in dev],
        reverse=True,
    )
    positive = [value for value in gains if value > 0.0]
    total_gain = sum(positive)
    concentration = {
        "positive_gain_queries": len(positive),
        "top1_share": None if total_gain == 0 else positive[0] / total_gain,
        "top5_share": None if total_gain == 0 else sum(positive[:5]) / total_gain,
    }
    action_summary: dict[str, Any] = {}
    for partition, records in (("train", train), ("dev", dev)):
        action_summary[partition] = {
            action: {
                "queries": len(records),
                "normalized_token_f1": float(
                    np.mean([row[f"f1_{action}"] for row in records])
                ),
                "normalized_exact_match": float(
                    np.mean([row[f"em_{action}"] for row in records])
                ),
                "answer_correctness": float(
                    np.mean([row[f"ac_{action}"] for row in records])
                ),
            }
            for action in ACTIONS
        }

    generation_cost = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    judge_cost = sum(
        _usage_cost(item.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for item in row.get("answer_correctness", [])
        if isinstance(item, Mapping)
    )
    support_minimum = int(phase2["minimum_high_margin_queries_per_action"])
    f1_pass = (
        point["f1_headroom"] >= practical
        and bootstrap["f1_ci95"][0] >= practical
    )
    ac_pass = (
        point["ac_delta_for_f1_oracle"] >= 0.0
        and bootstrap["ac_ci95"][0] >= -noninferiority
    )
    support_pass = (
        high_margin["bm25"] >= support_minimum
        and high_margin["dense"] >= support_minimum
    )
    row_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        row_counts[(str(row["split"]), str(row["query_id"]), str(row["action"]))] += 1
    expected_cells = sum(expected.values()) * len(ACTIONS)
    equal_repeat_protocol = (
        len(rows) == expected_cells * repeat_count
        and len(row_counts) == expected_cells
        and all(value == repeat_count for value in row_counts.values())
        and all(row["split"] in PARTITIONS for row in rows)
    )
    complete = (
        len(train) == expected["train"]
        and len(dev) == expected["dev"]
        and equal_repeat_protocol
    )
    no_failures = all(value == 0 for value in failures.values())
    if (
        f1_pass
        and ac_pass
        and support_pass
        and sensitivity_pass
        and complete
        and no_failures
    ):
        decision = "GO"
    elif not f1_pass and complete and no_failures:
        decision = "STOP"
    else:
        decision = "REVISE"
    summary = {
        "phase": 2,
        "revision": 1,
        "run_id": run_dir.name,
        "status": "complete" if complete and no_failures else "incomplete",
        "sample": {
            "train": len(train),
            "dev": len(dev),
            "final_holdout_outcomes": 0,
        },
        "generation_protocol": {
            "repeats_per_query_action": repeat_count,
            "aggregation": revision["aggregation"],
            "uncertainty": revision["uncertainty"],
            "exact_repeat_agreement_role": revision["exact_repeat_agreement_role"],
        },
        "actions": action_summary,
        "dev_headroom": {
            **{key: value for key, value in point.items() if key != "chosen_actions"},
            **bootstrap,
        },
        "leave_one_repeat_out": leave_one_out,
        "high_margin_support": high_margin,
        "headroom_concentration": concentration,
        "repeat_diagnostics": _all_repeat_diagnostics(rows, repeat_count),
        "failures": failures,
        "estimated_cost_usd": {
            "generation": generation_cost,
            "answer_correctness": judge_cost,
            "total": generation_cost + judge_cost,
        },
        "gate": {
            "f1_headroom_with_nested_uncertainty_passed": f1_pass,
            "ac_confirmatory_with_nested_uncertainty_passed": ac_pass,
            "two_action_high_margin_support_passed": support_pass,
            "leave_one_repeat_out_passed": sensitivity_pass,
            "equal_repeat_protocol_passed": equal_repeat_protocol,
            "complete_without_failures": complete and no_failures,
            "decision": decision,
        },
    }
    write_metadata_json(run_dir / "summary.json", summary)
    write_metadata_json(output_root / "summary.json", summary)
    _write_action_table(output_root, rows)
    return summary


def _winner_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[str],
    metric: str,
) -> dict[str, Any]:
    action_order = tuple(str(action) for action in actions)
    unique = {action: 0 for action in action_order}
    tied: defaultdict[str, int] = defaultdict(int)
    for row in records:
        best = max(row[f"{metric}_{action}"] for action in action_order)
        winners = tuple(
            action for action in action_order if row[f"{metric}_{action}"] == best
        )
        if len(winners) == 1:
            unique[winners[0]] += 1
        else:
            tied["+".join(winners)] += 1

    pairwise: dict[str, Any] = {}
    for left, right in combinations(action_order, 2):
        left_count = sum(
            row[f"{metric}_{left}"] > row[f"{metric}_{right}"] for row in records
        )
        right_count = sum(
            row[f"{metric}_{right}"] > row[f"{metric}_{left}"] for row in records
        )
        pairwise[f"{left}_vs_{right}"] = {
            left: int(left_count),
            right: int(right_count),
            "tie": len(records) - int(left_count) - int(right_count),
        }
    return {
        "metric": metric,
        "unique_winner": unique,
        "tied_best": dict(sorted(tied.items())),
        "pairwise": pairwise,
    }


def summarize_ac_primary(
    router: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    phase2 = router["phase2"]
    revision = router["phase2_ac_primary"]
    actions = AC_PRIMARY_ACTIONS
    repeat_count = int(revision["repeats_per_query_action"])
    repeat_ids = tuple(range(repeat_count))
    tie_break = str(revision["best_fixed_tie_break"])
    resamples = int(revision["bootstrap_resamples"])
    bootstrap_seed = int(revision["bootstrap_seed"])
    train_cells = _repeat_cells(rows, "train", actions=actions)
    dev_cells = _repeat_cells(rows, "dev", actions=actions)
    train = _mean_repeat_records(train_cells, actions=actions)
    dev = _mean_repeat_records(dev_cells, actions=actions)
    expected = {
        partition: int(phase2["sample_without_replacement"][partition])
        for partition in PARTITIONS
    }
    failures = {
        "generation": sum(row.get("status") == "generation_failure" for row in rows),
        "answer_correctness": sum(
            row.get("status") == "answer_correctness_failure" for row in rows
        ),
        "pending": sum(str(row.get("status", "")).startswith("pending") for row in rows),
    }
    point = compute_headroom_point(
        dev,
        tie_break,
        actions=actions,
        primary_metric="ac",
    )
    bootstrap = _nested_bootstrap_headroom(
        dev_cells,
        tie_break=tie_break,
        resamples=resamples,
        seed=bootstrap_seed,
        actions=actions,
        primary_metric="ac",
    )
    practical = float(revision["practical_ac_headroom"])

    leave_one_out: dict[str, Any] = {}
    sensitivity_pass = True
    for omitted in repeat_ids:
        allowed = tuple(value for value in repeat_ids if value != omitted)
        records = _mean_repeat_records(
            dev_cells,
            allowed,
            actions=actions,
        )
        sensitivity_point = compute_headroom_point(
            records,
            tie_break,
            actions=actions,
            primary_metric="ac",
        )
        sensitivity_bootstrap = _nested_bootstrap_headroom(
            dev_cells,
            tie_break=tie_break,
            resamples=resamples,
            seed=bootstrap_seed + omitted + 1,
            allowed_repeat_ids=allowed,
            actions=actions,
            primary_metric="ac",
        )
        passed = (
            sensitivity_point["best_fixed"] == point["best_fixed"]
            and sensitivity_point["ac_headroom"] >= practical
            and sensitivity_bootstrap["ac_ci95"][0] >= practical
        )
        sensitivity_pass = sensitivity_pass and passed
        leave_one_out[f"omit_{omitted}"] = {
            **{
                key: value
                for key, value in sensitivity_point.items()
                if key != "chosen_actions"
            },
            **sensitivity_bootstrap,
            "passed": passed,
        }

    fixed = str(point["best_fixed"])
    gains = sorted(
        [
            max(row[f"ac_{action}"] for action in actions) - row[f"ac_{fixed}"]
            for row in dev
        ],
        reverse=True,
    )
    positive = [value for value in gains if value > 0.0]
    total_gain = sum(positive)
    concentration = {
        "positive_gain_queries": len(positive),
        "top1_share": None if total_gain == 0 else positive[0] / total_gain,
        "top5_share": None if total_gain == 0 else sum(positive[:5]) / total_gain,
    }
    action_summary: dict[str, Any] = {}
    for partition, records in (("train", train), ("dev", dev)):
        action_summary[partition] = {
            action: {
                "queries": len(records),
                "answer_correctness": float(
                    np.mean([row[f"ac_{action}"] for row in records])
                ),
                "normalized_token_f1": float(
                    np.mean([row[f"f1_{action}"] for row in records])
                ),
                "normalized_exact_match": float(
                    np.mean([row[f"em_{action}"] for row in records])
                ),
            }
            for action in actions
        }

    row_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        row_counts[(str(row["split"]), str(row["query_id"]), str(row["action"]))] += 1
    expected_cells = sum(expected.values()) * len(actions)
    equal_repeat_protocol = (
        len(rows) == expected_cells * repeat_count
        and len(row_counts) == expected_cells
        and all(value == repeat_count for value in row_counts.values())
        and all(row["split"] in PARTITIONS for row in rows)
    )
    complete = (
        len(train) == expected["train"]
        and len(dev) == expected["dev"]
        and equal_repeat_protocol
    )
    no_failures = all(value == 0 for value in failures.values())
    headroom_pass = (
        point["ac_headroom"] >= practical
        and bootstrap["ac_ci95"][0] >= practical
    )
    if headroom_pass and sensitivity_pass and complete and no_failures:
        decision = "GO"
    elif not headroom_pass and complete and no_failures:
        decision = "STOP"
    else:
        decision = "REVISE"

    generation_cost = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    judge_cost = sum(
        _usage_cost(item.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for item in row.get("answer_correctness", [])
        if isinstance(item, Mapping)
    )
    summary = {
        "phase": 2,
        "revision": "ac_primary_nbd",
        "run_id": run_dir.name,
        "status": "complete" if complete and no_failures else "incomplete",
        "sample": {
            "train": len(train),
            "dev": len(dev),
            "final_holdout_outcomes": 0,
        },
        "protocol": {
            "primary_utility": "answer_correctness",
            "reference_metric": "normalized_token_f1",
            "actions": list(actions),
            "repeats_per_query_action": repeat_count,
            "aggregation": revision["aggregation"],
            "uncertainty": revision["uncertainty"],
        },
        "actions": action_summary,
        "dev_headroom": {
            **{key: value for key, value in point.items() if key != "chosen_actions"},
            **bootstrap,
        },
        "winner_diagnostics": {
            "train": _winner_diagnostics(train, actions=actions, metric="ac"),
            "dev": _winner_diagnostics(dev, actions=actions, metric="ac"),
        },
        "leave_one_repeat_out": leave_one_out,
        "headroom_concentration": concentration,
        "repeat_diagnostics": _all_repeat_diagnostics(rows, repeat_count),
        "failures": failures,
        "estimated_cost_usd": {
            "generation": generation_cost,
            "answer_correctness": judge_cost,
            "total": generation_cost + judge_cost,
        },
        "gate": {
            "ac_headroom_with_nested_uncertainty_passed": headroom_pass,
            "leave_one_repeat_out_passed": sensitivity_pass,
            "equal_repeat_protocol_passed": equal_repeat_protocol,
            "complete_without_failures": complete and no_failures,
            "f1_role": "reference_only_not_a_gate",
            "decision": decision,
        },
    }
    write_metadata_json(run_dir / "summary.json", summary)
    write_metadata_json(output_root / "summary.json", summary)
    _write_action_table(output_root, rows)
    return summary


def main() -> int:
    args = _arguments()
    if args.revision and args.ac_primary:
        raise ValueError("--revision and --ac-primary are mutually exclusive")
    config_path = _resolve_project_path(args.config)
    router = _load_router_config(config_path)
    if args.ac_primary:
        execution_router = _ac_primary_router(router)
    elif args.revision:
        execution_router = _revision_router(router)
    else:
        execution_router = router
    output_root = config_path.parent
    run_dir = output_root / "runs" / args.run_id
    if args.revision and args.run_id == str(router["phase2_revision"]["source_run"]):
        raise ValueError("Phase 2 revision must use a new run directory")
    if args.ac_primary and args.run_id == str(router["phase2_ac_primary"]["source_run"]):
        raise ValueError("AC-primary Phase 2 must use a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    freeze_sample(execution_router, run_dir)
    if args.stage in {"prepare", "all"}:
        if args.ac_primary:
            rows = prepare_ac_primary(router, output_root, run_dir)
        elif args.revision:
            rows = prepare_revision(router, output_root, run_dir)
        else:
            rows = prepare(router, run_dir)
        if args.stage == "prepare":
            print(json.dumps({"stage": "prepare", "rows": len(rows)}))
            return 0
    else:
        rows = _load_rows(run_dir)
    if args.stage in {"generate", "all"}:
        rows = generate_parallel(
            execution_router, run_dir, rows, max_new_calls=args.max_new_calls
        )
        if args.stage == "generate":
            print(json.dumps({"stage": "generate", "rows": len(rows)}))
            return 0
    if args.stage in {"judge", "all"}:
        rows = judge_parallel(
            execution_router, run_dir, rows, max_new_calls=args.max_new_calls
        )
        if args.stage == "judge":
            print(json.dumps({"stage": "judge", "rows": len(rows)}))
            return 0
    if args.ac_primary:
        summary = summarize_ac_primary(router, output_root, run_dir, rows)
    elif args.revision:
        summary = summarize_revision(router, output_root, run_dir, rows)
    else:
        summary = summarize(router, output_root, run_dir, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
