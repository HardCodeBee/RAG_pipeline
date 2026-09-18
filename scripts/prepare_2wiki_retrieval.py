"""Prepare frozen 2Wiki question contexts once under the existing RAG configuration.

Only local retrieval and prompt construction run here. No provider is created.
Small experiment bindings are hashed; no extra corpus/index audit is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO.parent / "work" / "router_research"
DEFAULT_RUN = REPO / "outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RESEARCH))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, document):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.int64)
    return {"count": len(values), "sum": int(values.sum()), "min": int(values.min()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": int(values.max())}


def ready_row(row, contract, token_count):
    from src.prompts.fixed_prompt import build_prompt
    from src.records import ContextPackage

    context = row["context"]
    package = ContextPackage(text=context["text"], results=(),
                             token_count=context["token_count"], truncated=context["truncated"])
    prompt = build_prompt(row["question"], package, contract["prompt"]["version"]).text
    tokens = token_count(prompt)
    return {**row, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tiktoken_prompt_tokens": tokens,
            "provider_input_tokens_reserved": max(math.ceil(1.25 * tokens), len(prompt.encode("utf-8"))) + 256}


def prepare(run):
    started = time.perf_counter()
    run = Path(run).resolve()
    # This gate precedes importing retrieval components or constructing a model.
    frozen = read_json(run / "actions_freeze.json")
    if frozen["status"] != "frozen_pre_retrieval_predictions_before_new_answers":
        raise ValueError("The selected pre-retrieval predictions must already be frozen")
    contract = read_json(run / "execution_contract.json")
    binding = {"actions_freeze_sha256": digest(run / "actions_freeze.json"),
               "execution_contract_sha256": digest(run / "execution_contract.json"),
               "questions_sha256": digest(run / "questions.jsonl")}
    manifest_path = run / "contexts_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if any(manifest[key] != value for key, value in binding.items()):
            raise ValueError("Existing contexts belong to different frozen inputs")
        print(json.dumps({"status": "already_complete_no_retrieval", "queries": manifest["queries"]}), flush=True)
        return
    examples = [json.loads(line) for line in (run / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len({row["query_id"] for row in examples}) != len(examples):
        raise ValueError("Duplicate selected question IDs")
    if len(examples) != contract["dataset"]["selected_queries"]:
        raise ValueError("Question count differs from the frozen contract")
    if contract["actions"] != ["bm25", "dense"] or contract["prompt"]["version"] != "hotpot_short_answer_v1":
        raise ValueError("Only the existing two-action short-answer protocol is supported")
    if (contract["retrieval"]["candidate_k"], contract["retrieval"]["final_k"], contract["context"]["max_tokens"]) != (50, 5, 1800):
        raise ValueError("Frozen retrieval/context limits changed")
    for row in examples:
        if not row["question"].strip() or not row["reference_answers"] or any(not isinstance(a, str) for a in row["reference_answers"]):
            raise ValueError("Missing question or offline references")

    from collect_validation_answers import tokenizer_count
    counter = tokenizer_count()  # Local cache only; fail before retrieval if unavailable.
    import prepare_validation_contexts as inherited
    import torch
    import faiss
    from threadpoolctl import threadpool_limits
    from scripts.run_router_phase1 import _deduplicated_hits
    from scripts.run_router_phase28_query_expansion import _ExactNumpySQLiteBM25
    from src.evaluators.beir_evaluation import compute_streaming_first_stage
    from src.indexes.streaming_flat_index import StreamingFlatIPIndex
    from src.retrievers.chunk_store import JsonlOffsetChunkStore
    from src.text.token_counters import RegexTokenCounter
    from preonly_features import configuration
    from src.embedders.text_embedder import create_embedder

    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    build = REPO / contract["artifact_locations"]["build"]
    artifacts = read_json(build / "manifest.json")["artifacts"]
    chunks, offsets = artifacts["chunks"], artifacts["chunk_offsets"]
    index_dir = REPO / contract["artifact_locations"]["bm25_index"]
    sparse_manifest = read_json(index_dir / "manifest.json")
    expected_bm25 = contract["retrieval"]["bm25"]
    actual_bm25 = sparse_manifest["identity"]["bm25"]
    if (sparse_manifest["backend"] != expected_bm25["backend"] or
            any(actual_bm25[key] != expected_bm25[key] for key in ("method", "k1", "b")) or
            sparse_manifest["identity"]["analyzer"]["name"] != expected_bm25["analyzer"] or
            sparse_manifest["identity"]["source_chunks"]["sha256"] != chunks["sha256"]):
        raise ValueError("Current BM25 metadata differs from the fixed environment")
    encoded = REPO / contract["artifact_locations"]["encoded_corpus"]
    if read_json(encoded / "manifest.json")["artifacts"]["chunks"]["sha256"] != chunks["sha256"]:
        raise ValueError("Dense and text corpus metadata differ")
    identity = {**binding, "script": str(Path(__file__)), "script_sha256": digest(__file__),
                "cpu_threads": 2, "dense_query_batch_size": 64, "dense_corpus_chunk_size": 25000,
                "provider_calls": 0}
    connection = inherited.connect_state(run / "contexts_state.sqlite3", identity)
    action_times = {}
    try:
        complete = {(q, a) for q, a in connection.execute("SELECT query_id,action FROM contexts")}
        expected = {(r["query_id"], a) for r in examples for a in ("bm25", "dense")}
        if not complete <= expected:
            raise ValueError("Context checkpoint contains unexpected questions/actions")
        with threadpool_limits(limits=2), JsonlOffsetChunkStore(
                build / chunks["file"], build / offsets["file"], expected_rows=chunks["rows"]) as store:
            for action in ("bm25", "dense"):
                pending = [row for row in examples if (row["query_id"], action) not in complete]
                if not pending:
                    continue
                action_start = time.perf_counter()
                regex_counter = RegexTokenCounter()

                def save(example, ids, scores, latency_ms):
                    hits = _deduplicated_hits(store, ids, scores, final_k=5)
                    row = inherited.context_record(example, action, hits, latency_ms,
                                                   token_counter=regex_counter, max_tokens=1800)
                    row["retrieval"].update(candidate_vector_ids=[int(v) for v in ids],
                                            candidate_scores=[float(v) for v in scores])
                    inherited.save_record(connection, row)

                print(json.dumps({"status": "local_retrieval_started", "action": action, "pending": len(pending)}), flush=True)
                if action == "bm25":
                    cache = run / "retrieval_cache"
                    cache.mkdir(exist_ok=True)
                    lengths = cache / "bm25_doc_lengths.npy"
                    existing_lengths = RESEARCH / "m6_beir_validation_v1/retrieval_cache/bm25_doc_lengths.npy"
                    if not lengths.exists() and existing_lengths.is_file():
                        shutil.copyfile(existing_lengths, lengths)
                    scorer = _ExactNumpySQLiteBM25(index_dir, cache)
                    scorer.postings_cache_max_bytes = 128 * 1024 * 1024
                    try:
                        for position, example in enumerate(pending):
                            query_start = time.perf_counter()
                            ids, scores = scorer.retrieve(example["question"], top_k=50)
                            save(example, ids, scores, (time.perf_counter() - query_start) * 1000)
                            if (position + 1) % 25 == 0 or position + 1 == len(pending):
                                print(json.dumps({"action": action, "completed": position + 1, "pending_at_start": len(pending)}), flush=True)
                    finally:
                        scorer.close()
                else:
                    core, _ = configuration()
                    adapter = SimpleNamespace(embedder=create_embedder(core, role="query"),
                        index=StreamingFlatIPIndex(encoded / "embeddings", corpus_chunk_size=25000))
                    if adapter.index.count != chunks["rows"] or adapter.index.dimension != 384:
                        raise ValueError("Dense index dimensions differ from the existing corpus")
                    questions = [{"question_id": row["query_id"], "question": row["question"]} for row in pending]
                    query_start = time.perf_counter()
                    batch = compute_streaming_first_stage(adapter, questions, candidate_k=50, query_batch_size=64)
                    elapsed = (time.perf_counter() - query_start) * 1000
                    for position, example in enumerate(pending):
                        save(example, batch.vector_ids[position], batch.scores[position], elapsed / len(pending))
                    del adapter
                action_times[action] = time.perf_counter() - action_start
                print(json.dumps({"status": "local_action_complete", "action": action, "new_contexts": len(pending),
                                  "wall_seconds": action_times[action]}), flush=True)
        rows = [ready_row(row, contract, counter) for row in inherited.complete_records(connection, examples)]
    finally:
        connection.close()
    destination = run / "ready_contexts.jsonl"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    token_stats = {}
    for action in ("bm25", "dense", "both"):
        selected = [row for row in rows if action == "both" or row["action"] == action]
        token_stats[action] = {
            "context_regex_tokens": describe([row["context"]["token_count"] for row in selected]),
            "prompt_o200k_tokens": describe([row["tiktoken_prompt_tokens"] for row in selected]),
            "reserved_input_tokens": describe([row["provider_input_tokens_reserved"] for row in selected]),
            "truncated_contexts": sum(row["context"]["truncated"] for row in selected)}
    repeats = contract["generation"]["repeats_per_query_action"]
    summary = {"status": "complete_local_contexts_and_prompts_no_answers", "queries": len(examples),
               "groups": len({row["group_id"] for row in examples}), "query_actions": len(rows),
               "required_successful_generations": len(rows) * repeats, "token_stats": token_stats,
               "three_repeat_input_prompt_tokens": token_stats["both"]["prompt_o200k_tokens"]["sum"] * repeats,
               "three_repeat_reserved_input_tokens": token_stats["both"]["reserved_input_tokens"]["sum"] * repeats,
               "three_repeat_max_output_tokens": len(rows) * repeats * contract["generation"]["max_output_tokens"],
               "action_wall_seconds": action_times, "preparation_wall_seconds": time.perf_counter() - started,
               "provider_calls": 0, "actual_paid_cost_usd": 0,
               "reference_fields_entered_retrieval_or_prompt": False,
               "collector_row_schema": "existing_pool_answer_collection_rows",
               "cost_note": "o200k tokens count the constructed prompt only; provider framing overhead remains estimated by the existing conservative reserve policy"}
    write_json(run / "contexts_summary.json", summary)
    manifest = {"status": "complete", **binding, "queries": len(examples), "query_actions": len(rows),
                "contexts": {"path": destination.name, "sha256": digest(destination)},
                "summary": "contexts_summary.json", "preparation_identity": identity,
                "required_successful_generations": len(rows) * repeats, "new_external_calls": 0,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    prepare(parser.parse_args().run)
