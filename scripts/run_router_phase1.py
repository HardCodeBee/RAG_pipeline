"""Run the frozen HotpotQA Phase 1 four-condition generator sanity pilot."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import validate_config
from src.context_builders.ranked_concat_context import build_context
from src.evaluators.beir_evaluation import compute_streaming_first_stage
from src.evaluators.beir_suite import compute_bm25_first_stage
from src.evaluators.hotpot_answer import (
    answer_correctness_schema,
    answer_metrics,
    build_answer_correctness_prompt,
    linear_weighted_kappa,
    normalize_answer,
    parse_answer_correctness,
)
from src.persistence.run_output_writer import (
    write_metadata_json,
    write_result_checkpoint,
    write_results,
)
from src.pipeline import NaiveRAGPipeline
from src.prompts.fixed_prompt import HOTPOT_SHORT_ANSWER_VERSION, build_prompt
from src.records import ContextPackage, SearchHit
from src.text.token_counters import RegexTokenCounter


ACTIONS = ("no_context", "gold_page_context", "bm25", "dense")
GENERATOR_PRICES = (0.40, 1.60)
JUDGE_PRICES = (0.75, 4.50)
_CHUNK_ID_PATTERN = re.compile(rb'"vector_id":(\d+),"doc_id":"([^"]+)"')


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
    parser.add_argument("--run-id", default="phase1_pilot_v1")
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "judge", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-new-calls",
        type=int,
        default=None,
        help="Stop this invocation after N new provider calls (for a resumable smoke test).",
    )
    parser.add_argument(
        "--calibration-decision",
        choices=("accept", "reject"),
        default=None,
        help="Record the independent review decision after all blind labels are complete.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_router_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("Router config must be a mapping")
    return value


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_pilot_examples(
    router: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split_path = _resolve_project_path(str(router["split"]["path"]))
    split = _read_json(split_path)
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("split.json has no assignments list")
    pilot = [row for row in assignments if row.get("partition") == "pilot"]
    expected = int(router["split"]["pilot_query_count"])
    if len(pilot) != expected:
        raise ValueError(f"Expected {expected} pilot queries, found {len(pilot)}")
    group_by_id = {str(row["query_id"]): str(row["group_id"]) for row in pilot}
    if len(group_by_id) != expected:
        raise ValueError("Pilot query ids must be unique")

    dataset_root = _resolve_project_path(str(router["dataset"]["root"]))
    query_path = dataset_root / "queries" / "queries.jsonl"
    examples: dict[str, dict[str, Any]] = {}
    raw_query_rows = 0
    with query_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_query_rows += 1
            row = json.loads(line)
            query_id = str(row.get("_id", ""))
            if query_id not in group_by_id:
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
            supporting_titles: list[str] = []
            for fact in supporting_facts:
                if (
                    not isinstance(fact, list)
                    or len(fact) != 2
                    or not isinstance(fact[0], str)
                    or not isinstance(fact[1], int)
                ):
                    raise ValueError(f"Malformed supporting fact for query {query_id}")
                if fact[0] not in supporting_titles:
                    supporting_titles.append(fact[0])
            examples[query_id] = {
                "dataset_id": str(router["dataset"]["dataset_id"]),
                "split": "pilot",
                "query_id": query_id,
                "group_id": group_by_id[query_id],
                "question": row["text"],
                "reference_answers": references,
                "supporting_titles": supporting_titles,
                "gold_doc_ids": [],
            }

    qrels_path = dataset_root / "qrels" / "train.tsv"
    raw_qrel_rows = 0
    with qrels_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError("Unexpected HotpotQA train qrels header")
        for line in handle:
            raw_qrel_rows += 1
            query_id, corpus_id, score = line.rstrip("\n\r").split("\t")
            if query_id in examples and float(score) > 0.0:
                examples[query_id]["gold_doc_ids"].append(corpus_id)

    missing_query = sorted(set(group_by_id) - set(examples))
    missing_answer = sorted(
        query_id for query_id, row in examples.items() if not row["reference_answers"]
    )
    missing_evidence = sorted(
        query_id for query_id, row in examples.items() if not row["gold_doc_ids"]
    )
    if missing_query or missing_answer or missing_evidence:
        raise ValueError(
            "Pilot mapping is incomplete: "
            f"missing_query={len(missing_query)}, missing_answer={len(missing_answer)}, "
            f"missing_evidence={len(missing_evidence)}"
        )
    ordered = [examples[str(row["query_id"])] for row in pilot]
    counts = {
        "eligible_query_rows": raw_query_rows,
        "pilot_requested": expected,
        "pilot_mapped": len(ordered),
        "mapping_failures": 0,
        "excluded": 0,
        "train_qrel_rows": raw_qrel_rows,
    }
    return ordered, counts


def _core_config(router: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    dense = router["retrieval"]["dense"]
    generation = router["generation"]
    value: dict[str, Any] = {
        "paths": {
            "corpus": str(_resolve_project_path(str(router["dataset"]["root"]))),
            "artifacts_root": str((PROJECT_ROOT / "artifacts").resolve()),
            "outputs_root": str((PROJECT_ROOT / "outputs").resolve()),
        },
        "loader": {"expected_dataset": router["dataset"]["dataset_id"]},
        "embedding": {
            "backend": dense["backend"],
            "model_name": dense["model"],
            "revision": dense["revision"],
            "normalize": dense["normalize"],
            "batch_size": int(dense.get("batch_size", 128)),
            "encode_call_rows": int(dense.get("encode_call_rows", 2048)),
            "shard_rows": int(dense.get("shard_rows", 25000)),
            "query_prefix": dense["query_prefix"],
            "document_prefix": dense["document_prefix"],
            "max_sequence_length": dense["max_sequence_length"],
            "local_files_only": True,
            "device": dense.get("device", "cuda"),
        },
        "index": {
            "backend": "faiss",
            "type": dense["index"],
            "build_batch_size": 65536,
            "faiss_threads": 8,
        },
        "retrieval": {
            "method": method,
            "final_k": int(router["retrieval"]["final_k"]),
            "candidate_k": int(router["retrieval"]["candidate_k"]),
            "search_threads": 8,
            "reranker": {"provider": "none"},
        },
        "context": {"max_tokens": int(router["context"]["max_tokens"])},
        "prompt": {"version": router["prompt"]["version"]},
        "generation": {
            "provider": generation["provider"],
            "model": generation["model"],
            "temperature": float(generation["temperature"]),
            "max_output_tokens": int(generation["max_output_tokens"]),
            "timeout_seconds": 60.0,
            "max_retries": 2,
        },
        "logging": {"save_retrieved_text": False, "save_prompt": False},
        "_base_dir": str(PROJECT_ROOT),
    }
    if dense["backend"] == "hf_dense":
        value["embedding"].update(
            {
                "family": dense["family"],
                "pooling": dense["pooling"],
                "document_input_format": dense["document_input_format"],
            }
        )
        for key in ("query_model_name", "query_revision"):
            if key in dense:
                value["embedding"][key] = dense[key]
    if method == "dense":
        value["retrieval"].update(
            {"corpus_chunk_size": 25000, "query_batch_size": 100}
        )
    elif method == "bm25":
        bm25 = router["retrieval"]["bm25"]
        value["bm25"] = {
            "backend": "sqlite",
            "method": bm25["method"],
            "k1": float(bm25["k1"]),
            "b": float(bm25["b"]),
            "analyzer": bm25["analyzer"],
            "transaction_documents": 1000,
        }
    else:
        raise ValueError(f"Unsupported retrieval action: {method}")
    return validate_config(value)


def _gold_vector_ids(chunks_path: Path, gold_doc_ids: set[str]) -> dict[str, int]:
    remaining = set(gold_doc_ids)
    found: dict[str, int] = {}
    with chunks_path.open("rb") as handle:
        for line in handle:
            match = _CHUNK_ID_PATTERN.search(line)
            if match is None:
                raise ValueError("Unexpected chunk JSONL record layout")
            doc_id = match.group(2).decode("utf-8")
            if doc_id in remaining:
                found[doc_id] = int(match.group(1))
                remaining.remove(doc_id)
                if not remaining:
                    break
    if remaining:
        raise ValueError(f"Gold qrels reference missing chunk ids: {sorted(remaining)[:5]}")
    return found


def _deduplicated_hits(
    chunk_store: Any,
    vector_ids: Sequence[int],
    scores: Sequence[float],
    *,
    final_k: int,
) -> tuple[SearchHit, ...]:
    chunks = chunk_store.get_many([int(value) for value in vector_ids])
    selected: list[SearchHit] = []
    seen_doc_ids: set[str] = set()
    for chunk, score in zip(chunks, scores):
        if chunk.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(chunk.doc_id)
        selected.append(SearchHit(rank=len(selected) + 1, chunk=chunk, score=float(score)))
        if len(selected) >= final_k:
            break
    return tuple(selected)


def _base_row(
    example: Mapping[str, Any],
    *,
    action: str,
    hits: Sequence[SearchHit],
    retrieval_latency_ms: float | None,
    token_counter: RegexTokenCounter,
    context_max_tokens: int,
) -> dict[str, Any]:
    context = build_context(hits, token_counter, context_max_tokens)
    gold_doc_ids = list(example["gold_doc_ids"])
    retrieved_doc_ids = [hit.chunk.doc_id for hit in hits]
    overlap = len(set(gold_doc_ids) & set(retrieved_doc_ids))
    return {
        "status": "pending_generation",
        "dataset_id": example["dataset_id"],
        "split": example["split"],
        "query_id": example["query_id"],
        "group_id": example["group_id"],
        "action": action,
        "question": example["question"],
        "reference_answers": list(example["reference_answers"]),
        "supporting_titles": list(example["supporting_titles"]),
        "gold_doc_ids": gold_doc_ids,
        "retrieval": {
            "doc_ids": retrieved_doc_ids,
            "vector_ids": [hit.chunk.vector_id for hit in hits],
            "scores": [float(hit.score) for hit in hits],
            "evidence_page_recall": overlap / len(set(gold_doc_ids)),
            "hit": float(overlap > 0),
            "latency_ms": retrieval_latency_ms,
        },
        "context": {
            "text": context.text,
            "token_count": context.token_count,
            "truncated": context.truncated,
        },
    }


def _prompt_for_row(
    row: Mapping[str, Any],
    version: str = HOTPOT_SHORT_ANSWER_VERSION,
) -> str:
    context_value = row["context"]
    context = ContextPackage(
        text=str(context_value["text"]),
        results=(),
        token_count=int(context_value["token_count"]),
        truncated=bool(context_value["truncated"]),
    )
    return build_prompt(str(row["question"]), context, version).text


def _checkpoint_path(checkpoints: Path, position: int) -> Path:
    return checkpoints / f"{position:05d}.json"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Invalid checkpoint row: {path}")
    return value


def _write_all_rows(run_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_results(run_dir / "results.jsonl", list(rows))


def _load_all_rows(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = run_dir / "checkpoints"
    paths = sorted(checkpoints.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No Phase 1 checkpoints found in {checkpoints}")
    return [_load_checkpoint(path) for path in paths]


def prepare(
    router: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    checkpoints = run_dir / "checkpoints"
    existing = sorted(checkpoints.glob("*.json"))
    expected_rows = int(router["generation"]["phase1_total_call_limit"])
    if len(existing) == expected_rows:
        return [_load_checkpoint(path) for path in existing], _read_json(
            run_dir / "mapping_counts.json"
        )
    if existing:
        raise RuntimeError("Partial prepare checkpoints exist; refusing an ambiguous rebuild")

    examples, mapping_counts = _load_pilot_examples(router)
    build_path = _resolve_project_path(str(router["artifacts"]["build"]))
    gold_ids = {doc_id for row in examples for doc_id in row["gold_doc_ids"]}
    token_counter = RegexTokenCounter()
    final_k = int(router["retrieval"]["final_k"])
    candidate_k = int(router["retrieval"]["candidate_k"])
    context_max_tokens = int(router["context"]["max_tokens"])
    questions = [
        {"question_id": row["query_id"], "question": row["question"], "qrels": {}}
        for row in examples
    ]

    dense_config = _core_config(router, method="dense")
    with NaiveRAGPipeline(dense_config) as dense_pipeline:
        if dense_pipeline.build_dir != build_path:
            raise ValueError(
                f"Dense registry resolved {dense_pipeline.build_dir}, expected {build_path}"
            )
        chunks_path = (
            build_path
            / dense_pipeline.manifest["artifacts"]["chunks"]["file"]
        )
        gold_vector_ids = _gold_vector_ids(chunks_path, gold_ids)
        dense_started = time.perf_counter()
        dense_batch = compute_streaming_first_stage(
            dense_pipeline,
            questions,
            candidate_k=candidate_k,
            query_batch_size=len(questions),
        )
        dense_total_ms = (time.perf_counter() - dense_started) * 1000.0
        dense_rows: list[tuple[SearchHit, ...]] = []
        for position in range(len(examples)):
            dense_rows.append(
                _deduplicated_hits(
                    dense_pipeline.chunk_store,
                    dense_batch.vector_ids[position],
                    dense_batch.scores[position],
                    final_k=final_k,
                )
            )
        chunk_store = dense_pipeline.chunk_store
        gold_rows = [
            tuple(
                SearchHit(
                    rank=rank,
                    chunk=chunk_store.get(gold_vector_ids[doc_id]),
                    score=1.0,
                )
                for rank, doc_id in enumerate(example["gold_doc_ids"], start=1)
            )
            for example in examples
        ]

    bm25_config = _core_config(router, method="bm25")
    with NaiveRAGPipeline(bm25_config) as bm25_pipeline:
        if bm25_pipeline.build_dir != build_path:
            raise ValueError(
                f"BM25 registry resolved {bm25_pipeline.build_dir}, expected {build_path}"
            )
        bm25_batch = compute_bm25_first_stage(
            bm25_pipeline,
            questions,
            candidate_k=candidate_k,
            retained_k=candidate_k,
        )
        bm25_rows: list[tuple[SearchHit, ...]] = []
        for position in range(len(examples)):
            scores, vector_ids = bm25_batch.row(position)
            bm25_rows.append(
                _deduplicated_hits(
                    bm25_pipeline.chunk_store,
                    vector_ids,
                    scores,
                    final_k=final_k,
                )
            )

    rows: list[dict[str, Any]] = []
    for position, example in enumerate(examples):
        conditions = {
            "no_context": ((), 0.0),
            "gold_page_context": (gold_rows[position], 0.0),
            "bm25": (
                bm25_rows[position],
                float(bm25_batch.per_question_timings_ms[position].get("total_ms", 0.0)),
            ),
            "dense": (dense_rows[position], dense_total_ms / len(examples)),
        }
        for action in ACTIONS:
            hits, latency_ms = conditions[action]
            rows.append(
                _base_row(
                    example,
                    action=action,
                    hits=hits,
                    retrieval_latency_ms=latency_ms,
                    token_counter=token_counter,
                    context_max_tokens=context_max_tokens,
                )
            )

    repeat_queries = int(router["generation"]["repeat_check_queries"])
    repeated: list[dict[str, Any]] = []
    for row in rows[: repeat_queries * len(ACTIONS)]:
        row["repeat_id"] = 0
        duplicate = copy.deepcopy(row)
        duplicate["repeat_id"] = 1
        repeated.append(duplicate)
    rows.extend(repeated)
    if len(rows) != expected_rows:
        raise ValueError(f"Prepared {len(rows)} rows, expected {expected_rows}")
    for position, row in enumerate(rows):
        write_result_checkpoint(_checkpoint_path(checkpoints, position), row)
    write_metadata_json(run_dir / "mapping_counts.json", mapping_counts)
    _write_all_rows(run_dir, rows)
    return rows, mapping_counts


def _usage_cost(token_usage: Mapping[str, Any], prices: tuple[float, float]) -> float:
    provider = token_usage.get("provider_reported")
    if not isinstance(provider, Mapping):
        return 0.0
    input_tokens = provider.get("input_tokens")
    output_tokens = provider.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000.0


def generate(
    router: Mapping[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_calls: int | None = None,
) -> list[dict[str, Any]]:
    config = _core_config(router, method="bm25")
    from src.generators.answer_generator import LLMGenerator

    generation = config["generation"]
    generator = LLMGenerator(
        provider=generation["provider"],
        model=generation["model"],
        temperature=generation["temperature"],
        max_output_tokens=generation["max_output_tokens"],
        timeout_seconds=generation["timeout_seconds"],
        max_retries=generation["max_retries"],
    )
    call_limit = int(router["generation"]["phase1_total_call_limit"])
    budget = float(router["generation"]["hard_budget_usd"])
    attempted = sum(
        row.get("generation", {}).get("attempted", False)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    spent = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    new_calls = 0
    try:
        for position, row in enumerate(rows):
            if isinstance(row.get("generation"), Mapping) and row["generation"].get(
                "attempted"
            ):
                continue
            if attempted >= call_limit or spent >= budget:
                break
            if max_new_calls is not None and new_calls >= max_new_calls:
                break
            attempted += 1
            new_calls += 1
            try:
                result = generator.generate_from_prompt(
                    _prompt_for_row(row, str(config["prompt"]["version"])),
                    row["question"],
                    [],
                )
                row["prediction"] = result.answer.strip()
                metrics = answer_metrics(row["prediction"], row["reference_answers"])
                row["metrics"] = metrics
                row["generation"] = {
                    "attempted": True,
                    "status": "success",
                    "latency_ms": result.latency_ms,
                    "token_usage": result.token_usage,
                }
                row["status"] = "pending_answer_correctness"
                spent += _usage_cost(result.token_usage, GENERATOR_PRICES)
            except Exception as exc:  # preserve a failure row without imputation
                row["generation"] = {
                    "attempted": True,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                row["status"] = "generation_failure"
            write_result_checkpoint(_checkpoint_path(run_dir / "checkpoints", position), row)
            if new_calls % 20 == 0:
                print(
                    json.dumps(
                        {"generation_new_calls": new_calls, "estimated_cost_usd": spent}
                    ),
                    flush=True,
                )
    finally:
        close = getattr(getattr(generator, "_client", None), "close", None)
        if callable(close):
            close()
    _write_all_rows(run_dir, rows)
    return rows


def _judge_call(
    client: Any,
    router: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    judge = router["answer_correctness"]
    prompt = build_answer_correctness_prompt(
        question=str(row["question"]),
        reference_answers=row["reference_answers"],
        predicted_answer=str(row["prediction"]),
        rubric=judge["rubric"],
    )
    started = time.perf_counter()
    response = client.responses.create(
        model=judge["model"],
        input=prompt,
        reasoning={"effort": judge["reasoning_effort"]},
        max_output_tokens=int(judge["max_output_tokens"]),
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_correctness",
                "strict": True,
                "schema": answer_correctness_schema(),
            }
        },
    )
    parsed = parse_answer_correctness(response.output_text)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    token_usage = {
        "provider_reported": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    }
    result = {
        "attempted": True,
        "status": "success",
        "label": parsed.label,
        "score": parsed.score,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "token_usage": token_usage,
    }
    return result, _usage_cost(token_usage, JUDGE_PRICES)


def judge(
    router: Mapping[str, Any],
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_calls: int | None = None,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    judge_config = router["answer_correctness"]
    call_limit = int(judge_config["phase1_call_limit"])
    budget = float(judge_config["hard_budget_usd"])
    attempted = sum(
        len(row.get("answer_correctness", []))
        for row in rows
        if isinstance(row.get("answer_correctness"), list)
    )
    spent = sum(
        _usage_cost(item.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for item in row.get("answer_correctness", [])
        if isinstance(item, Mapping)
    )
    new_calls = 0
    client = OpenAI(timeout=60.0, max_retries=2)
    try:
        for position, row in enumerate(rows):
            if row.get("status") == "generation_failure":
                continue
            values = row.setdefault("answer_correctness", [])
            if values:
                continue
            if attempted >= call_limit or spent >= budget:
                break
            if max_new_calls is not None and new_calls >= max_new_calls:
                break
            attempted += 1
            new_calls += 1
            try:
                result, cost = _judge_call(client, router, row)
                values.append(result)
                row["metrics"]["answer_correctness"] = result["score"]
                row["status"] = "success"
                spent += cost
            except Exception as exc:
                values.append(
                    {
                        "attempted": True,
                        "status": "failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                row["status"] = "answer_correctness_failure"
            write_result_checkpoint(_checkpoint_path(run_dir / "checkpoints", position), row)
            if new_calls % 20 == 0:
                print(
                    json.dumps(
                        {"judge_new_calls": new_calls, "estimated_cost_usd": spent}
                    ),
                    flush=True,
                )

        repeat_target = int(judge_config["repeat_check_predictions"])
        repeated = 0
        for position, row in enumerate(rows[: int(router["generation"]["phase1_main_call_limit"])]):
            values = row.get("answer_correctness")
            if not isinstance(values, list) or not values or values[0].get("status") != "success":
                continue
            if len(values) >= 2:
                repeated += 1
                continue
            if repeated >= repeat_target or attempted >= call_limit or spent >= budget:
                break
            if max_new_calls is not None and new_calls >= max_new_calls:
                break
            attempted += 1
            new_calls += 1
            try:
                result, cost = _judge_call(client, router, row)
                values.append(result)
                spent += cost
            except Exception as exc:
                values.append(
                    {
                        "attempted": True,
                        "status": "failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            repeated += 1
            write_result_checkpoint(_checkpoint_path(run_dir / "checkpoints", position), row)
            if new_calls % 20 == 0:
                print(
                    json.dumps(
                        {"judge_new_calls": new_calls, "estimated_cost_usd": spent}
                    ),
                    flush=True,
                )
    finally:
        client.close()
    _write_all_rows(run_dir, rows)
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_action: str,
    right_action: str,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    primary = [row for row in rows if row.get("repeat_id", 0) == 0]
    by_key = {(row["query_id"], row["action"]): row for row in primary}
    group_differences: defaultdict[str, list[float]] = defaultdict(list)
    for row in primary:
        if row["action"] != left_action:
            continue
        pair = by_key.get((row["query_id"], right_action))
        if pair is None:
            continue
        left = row.get("metrics", {}).get(metric)
        right = pair.get("metrics", {}).get(metric)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            group_differences[str(row["group_id"])].append(float(left) - float(right))
    groups = sorted(group_differences)
    differences = [value for group in groups for value in group_differences[group]]
    result: dict[str, Any] = {
        "paired_queries": len(differences),
        "effect": _mean(differences),
        "ci95": None,
    }
    if not groups or resamples <= 0:
        return result
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        values = [value for group in sampled for value in group_differences[str(group)]]
        bootstrapped[position] = float(np.mean(values))
    result["ci95"] = [
        float(np.quantile(bootstrapped, 0.025)),
        float(np.quantile(bootstrapped, 0.975)),
    ]
    return result


def _calibration_sample(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    primary = [
        row
        for row in rows
        if row.get("repeat_id", 0) == 0
        and row.get("status") == "success"
        and isinstance(row.get("metrics", {}).get("answer_correctness"), (int, float))
    ]
    allocations = dict(zip(ACTIONS, (13, 13, 12, 12)))
    selected: list[Mapping[str, Any]] = []
    for action in ACTIONS:
        candidates = [row for row in primary if row["action"] == action]
        candidates.sort(
            key=lambda row: (
                -abs(
                    float(row["metrics"]["answer_correctness"])
                    - float(row["metrics"]["normalized_token_f1"])
                ),
                str(row["query_id"]),
            )
        )
        selected.extend(candidates[: allocations[action]])
    sample_path = run_dir / "calibration_sample.jsonl"
    existing_labels: dict[str, Any] = {}
    if sample_path.is_file():
        with sample_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                existing = json.loads(line)
                existing_labels[str(existing["calibration_id"])] = existing.get(
                    "human_label"
                )
    sample = []
    key = []
    for position, row in enumerate(selected, start=1):
        calibration_id = f"c{position:03d}"
        sample.append(
            {
                "calibration_id": calibration_id,
                "question": row["question"],
                "reference_answers": row["reference_answers"],
                "predicted_answer": row["prediction"],
                "human_label": existing_labels.get(calibration_id),
            }
        )
        key.append(
            {
                "calibration_id": calibration_id,
                "query_id": row["query_id"],
                "action": row["action"],
                "normalized_exact_match": row["metrics"]["normalized_exact_match"],
                "normalized_token_f1": row["metrics"]["normalized_token_f1"],
                "answer_correctness_label": row["answer_correctness"][0]["label"],
            }
        )
    write_results(sample_path, sample)
    write_results(run_dir / "calibration_key.jsonl", key)


def _manual_calibration_summary(run_dir: Path) -> dict[str, Any]:
    sample_path = run_dir / "calibration_sample.jsonl"
    key_path = run_dir / "calibration_key.jsonl"
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = [json.loads(line) for line in handle]
    with key_path.open("r", encoding="utf-8") as handle:
        key = {row["calibration_id"]: row for row in map(json.loads, handle)}
    allowed = ("incorrect", "partially_correct", "correct")
    allowed_set = set(allowed)
    invalid = [
        row["calibration_id"]
        for row in sample
        if row.get("human_label") is not None and row.get("human_label") not in allowed_set
    ]
    completed = [row for row in sample if row.get("human_label") in allowed_set]
    result: dict[str, Any] = {
        "requested": len(sample),
        "completed": len(completed),
        "invalid_label_ids": invalid,
        "complete": len(completed) == len(sample) and not invalid,
        "confusion_human_by_ac": None,
        "exact_agreement": None,
        "linear_weighted_kappa": None,
        "two_step_disagreements": None,
        "ac_minus_human_mean_score": None,
        "per_human_label_exact_agreement": None,
    }
    if not result["complete"]:
        return result
    human = [str(row["human_label"]) for row in completed]
    judged = [str(key[row["calibration_id"]]["answer_correctness_label"]) for row in completed]
    result["confusion_human_by_ac"] = {
        label: {other: 0 for other in allowed} for label in allowed
    }
    for left, right in zip(human, judged):
        result["confusion_human_by_ac"][left][right] += 1
    result["exact_agreement"] = _mean(
        [float(left == right) for left, right in zip(human, judged)]
    )
    result["linear_weighted_kappa"] = linear_weighted_kappa(human, judged)
    positions = {label: position for position, label in enumerate(allowed)}
    result["two_step_disagreements"] = sum(
        abs(positions[left] - positions[right]) == 2
        for left, right in zip(human, judged)
    )
    scores = {"incorrect": 0.0, "partially_correct": 0.5, "correct": 1.0}
    result["ac_minus_human_mean_score"] = _mean(
        [scores[right] - scores[left] for left, right in zip(human, judged)]
    )
    result["per_human_label_exact_agreement"] = {
        label: _mean(
            [
                float(left == right)
                for left, right in zip(human, judged)
                if left == label
            ]
        )
        for label in allowed
    }
    return result


def summarize(
    router: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    calibration_decision: str | None = None,
) -> dict[str, Any]:
    for position, row in enumerate(rows):
        row.pop("prompt_text", None)
        write_result_checkpoint(_checkpoint_path(run_dir / "checkpoints", position), row)
    _write_all_rows(run_dir, rows)
    main = [row for row in rows if row.get("repeat_id", 0) == 0]
    action_summary: dict[str, Any] = {}
    for action in ACTIONS:
        action_rows = [row for row in main if row["action"] == action]
        successful = [row for row in action_rows if row.get("status") == "success"]
        action_summary[action] = {
            "rows": len(action_rows),
            "successes": len(successful),
            "normalized_token_f1": _mean(
                [float(row["metrics"]["normalized_token_f1"]) for row in successful]
            ),
            "normalized_exact_match": _mean(
                [float(row["metrics"]["normalized_exact_match"]) for row in successful]
            ),
            "answer_correctness": _mean(
                [float(row["metrics"]["answer_correctness"]) for row in successful]
            ),
            "evidence_page_recall": _mean(
                [float(row["retrieval"]["evidence_page_recall"]) for row in action_rows]
            ),
            "retrieval_hit": _mean(
                [float(row["retrieval"]["hit"]) for row in action_rows]
            ),
            "retrieval_latency_ms": _mean(
                [
                    float(row["retrieval"]["latency_ms"])
                    for row in action_rows
                    if isinstance(row["retrieval"]["latency_ms"], (int, float))
                ]
            ),
        }

    evaluation = router["evaluation"]
    paired_f1 = _paired_bootstrap(
        rows,
        left_action="gold_page_context",
        right_action="no_context",
        metric="normalized_token_f1",
        resamples=int(evaluation["bootstrap_resamples"]),
        seed=int(router["split"]["seed"]),
    )
    paired_ac = _paired_bootstrap(
        rows,
        left_action="gold_page_context",
        right_action="no_context",
        metric="answer_correctness",
        resamples=int(evaluation["bootstrap_resamples"]),
        seed=int(router["split"]["seed"]) + 1,
    )

    repeat_pairs = []
    by_repeat = {
        (row["query_id"], row["action"], row.get("repeat_id", 0)): row for row in rows
    }
    for row in rows:
        if row.get("repeat_id") != 0:
            continue
        pair = by_repeat.get((row["query_id"], row["action"], 1))
        if pair is not None and "prediction" in row and "prediction" in pair:
            repeat_pairs.append((row, pair))
    generation_agreement = _mean(
        [
            float(normalize_answer(left["prediction"]) == normalize_answer(right["prediction"]))
            for left, right in repeat_pairs
        ]
    )
    generation_f1_agreement = _mean(
        [
            float(
                left["metrics"]["normalized_token_f1"]
                == right["metrics"]["normalized_token_f1"]
            )
            for left, right in repeat_pairs
        ]
    )
    generation_em_agreement = _mean(
        [
            float(
                left["metrics"]["normalized_exact_match"]
                == right["metrics"]["normalized_exact_match"]
            )
            for left, right in repeat_pairs
        ]
    )
    generation_ac_agreement = _mean(
        [
            float(
                left["metrics"]["answer_correctness"]
                == right["metrics"]["answer_correctness"]
            )
            for left, right in repeat_pairs
        ]
    )

    ac_first: list[str] = []
    ac_second: list[str] = []
    for row in rows:
        values = row.get("answer_correctness")
        if (
            isinstance(values, list)
            and len(values) >= 2
            and values[0].get("status") == "success"
            and values[1].get("status") == "success"
        ):
            ac_first.append(str(values[0]["label"]))
            ac_second.append(str(values[1]["label"]))

    divergence_rows = [row for row in main if row.get("status") == "success"]
    divergence = {
        "em_vs_ac": _mean(
            [
                float(
                    float(row["metrics"]["normalized_exact_match"])
                    != float(row["metrics"]["answer_correctness"])
                )
                for row in divergence_rows
            ]
        ),
        "f1_vs_ac_absolute_difference": _mean(
            [
                abs(
                    float(row["metrics"]["normalized_token_f1"])
                    - float(row["metrics"]["answer_correctness"])
                )
                for row in divergence_rows
            ]
        ),
    }
    generation_cost = sum(
        _usage_cost(row.get("generation", {}).get("token_usage", {}), GENERATOR_PRICES)
        for row in rows
        if isinstance(row.get("generation"), Mapping)
    )
    judge_cost = sum(
        _usage_cost(value.get("token_usage", {}), JUDGE_PRICES)
        for row in rows
        for value in row.get("answer_correctness", [])
        if isinstance(value, Mapping)
    )
    failures = {
        "generation": sum(row.get("status") == "generation_failure" for row in rows),
        "answer_correctness": sum(
            row.get("status") == "answer_correctness_failure" for row in rows
        ),
        "pending": sum(str(row.get("status", "")).startswith("pending") for row in rows),
    }
    f1_threshold = float(
        evaluation["practical_effect"]["phase1_gold_minus_no_context_f1"]
    )
    ac_threshold = float(
        evaluation["practical_effect"]["phase1_gold_minus_no_context_ac"]
    )
    automated_gate = bool(
        failures == {"generation": 0, "answer_correctness": 0, "pending": 0}
        and isinstance(paired_f1["effect"], float)
        and paired_f1["effect"] >= f1_threshold
        and isinstance(paired_ac["effect"], float)
        and paired_ac["effect"] >= ac_threshold
    )
    _calibration_sample(run_dir, rows)
    manual_calibration = _manual_calibration_summary(run_dir)
    if calibration_decision is not None and not manual_calibration["complete"]:
        raise ValueError("Calibration cannot be accepted or rejected before all labels are valid")
    if not automated_gate or calibration_decision == "reject":
        gate_decision = "REVISE"
        phase_status = "revise"
    elif calibration_decision == "accept":
        gate_decision = "GO"
        phase_status = "complete"
    else:
        gate_decision = "HOLD"
        phase_status = (
            "pending_manual_calibration"
            if not manual_calibration["complete"]
            else "pending_calibration_review"
        )
    summary = {
        "phase": 1,
        "run_id": run_dir.name,
        "status": phase_status,
        "mapping": _read_json(run_dir / "mapping_counts.json"),
        "actions": action_summary,
        "paired_gold_minus_no_context": {
            "normalized_token_f1": paired_f1,
            "answer_correctness": paired_ac,
        },
        "stability": {
            "generation_repeat_pairs": len(repeat_pairs),
            "normalized_answer_agreement": generation_agreement,
            "normalized_token_f1_agreement": generation_f1_agreement,
            "normalized_exact_match_agreement": generation_em_agreement,
            "answer_correctness_score_agreement": generation_ac_agreement,
            "answer_correctness_repeat_pairs": len(ac_first),
            "answer_correctness_exact_agreement": _mean(
                [float(left == right) for left, right in zip(ac_first, ac_second)]
            ),
            "answer_correctness_linear_weighted_kappa": linear_weighted_kappa(
                ac_first, ac_second
            ),
        },
        "metric_divergence": divergence,
        "manual_calibration": manual_calibration,
        "failures": failures,
        "estimated_cost_usd": {
            "generation": generation_cost,
            "answer_correctness": judge_cost,
            "total": generation_cost + judge_cost,
        },
        "gate": {
            "effect_and_failure_checks_passed": automated_gate,
            "manual_calibration_required": not manual_calibration["complete"],
            "calibration_acceptance_requires_review": calibration_decision is None,
            "calibration_review_decision": calibration_decision,
            "decision": gate_decision,
        },
    }
    write_metadata_json(run_dir / "summary.json", summary)
    write_metadata_json(output_root / "summary.json", summary)

    action_rows = []
    repeated_keys = {
        (row["query_id"], row["action"])
        for row in rows
        if row.get("repeat_id") == 1
    }
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
        }
        if (row["query_id"], row["action"]) in repeated_keys:
            value["repeat_id"] = row.get("repeat_id", 0)
        action_rows.append(value)
    write_results(output_root / "action_table.jsonl", action_rows)
    return summary


def main() -> int:
    args = _arguments()
    config_path = _resolve_project_path(args.config)
    router = _load_router_config(config_path)
    output_root = config_path.parent
    run_dir = output_root / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"prepare", "all"}:
        rows, _ = prepare(router, run_dir)
        if args.stage == "prepare":
            print(json.dumps({"stage": "prepare", "rows": len(rows)}, ensure_ascii=False))
            return 0
    else:
        rows = _load_all_rows(run_dir)

    if args.stage in {"generate", "all"}:
        rows = generate(router, run_dir, rows, max_new_calls=args.max_new_calls)
        if args.stage == "generate":
            print(json.dumps({"stage": "generate", "rows": len(rows)}, ensure_ascii=False))
            return 0
    if args.stage in {"judge", "all"}:
        rows = judge(router, run_dir, rows, max_new_calls=args.max_new_calls)
        if args.stage == "judge":
            print(json.dumps({"stage": "judge", "rows": len(rows)}, ensure_ascii=False))
            return 0
    summary = summarize(
        router,
        output_root,
        run_dir,
        rows,
        calibration_decision=args.calibration_decision,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
