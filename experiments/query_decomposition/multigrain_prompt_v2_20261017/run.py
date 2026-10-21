"""Five isolated prompt controls on a frozen fresh-plus-regression round."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import os
import sqlite3
from threading import Lock
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, canonical, digest
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as old_prompts
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.run import REWRITE_INSTRUCTION, merge_packets, passages
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import load, rows, write, CAMPAIGN
from .prepare import HERE, OUT, ROOT, OUTPUT, OLD_OUT, S02, sha
from . import prompts


RETRIEVAL_LOCK = Lock()


class RoundBudget(ParallelBudget):
    def reserve(self, request_id, namespace, amount):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges").fetchone()[0]
            local = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()[0]
            if total + amount > 195 or local + amount > 5:
                raise RuntimeError("Protected global budget or round subcap reached")
            if self.db.execute("SELECT 1 FROM charges WHERE id=?", (request_id,)).fetchone():
                raise RuntimeError("Existing liability cannot be submitted again")
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def source_paths(dataset):
    historical = (
        OLD_OUT / "musique_baseline_screening_network_v2/checkpoint.sqlite3" if dataset == "musique" else
        CAMPAIGN / dataset / "formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3"
    )
    return [historical, OLD_OUT / "api" / dataset / "checkpoint.sqlite3"]


def seed_sources(owner, selected, resources):
    """Copy immutable retrieval packets and index exact successful old API receipts."""
    cache = {}
    for source in source_paths(selected[0]["dataset"]):
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for case in selected:
                for key, raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval' AND json_extract(value,'$.qid')=?", (case["qid"],)):
                    packet = json.loads(raw)
                    assert key == digest([packet["qid"], packet["query"], packet["k"], resources])
                    owner.put("retrieval", key, packet)
            # Keep only receipts mentioning selected original questions, not gold references.
            for rid, stage, payload, status, raw in db.execute("SELECT * FROM requests WHERE status='success'"):
                if not any(case["question"] in payload for case in selected):
                    continue
                obj = json.loads(payload)
                result = json.loads(raw)
                if result.get("status") == "success" and result.get("finish_reason") == "stop":
                    cache.setdefault(digest(obj), {"id": rid, "stage": stage, "payload": payload,
                        "status": status, "result": raw, "origin": str(source.relative_to(ROOT))})
    for case in selected:
        key = digest([case["qid"], case["question"], 10, resources])
        if owner.get("retrieval", key) is None:
            raise RuntimeError("Missing frozen original-query Top10 packet")
    return cache


def allowed_bindings(selected):
    bindings = []
    for case in selected:
        stages = [variant + "/final" for variant in ("c00_old_prompts", "c01_final_only", "c02_rewrite_only", "c03_both", "c04_single")]
        for branch in ("old", "new", "single"):
            for node in case["qdmr"]["steps"]:
                stages.extend(f"{branch}/{kind}/{node['id']}" for kind in ("rewrite", "answer", "sentence_format"))
        bindings.extend({"qid": case["qid"], "stage": stage} for stage in stages)
    return bindings


def ask(transport, known, bindings, stage, prompt, maximum):
    payload = {"model": transport.model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": maximum, "extra_body": {"enable_thinking": False}}
    receipt = known.get(digest(payload))
    if receipt is not None:
        key = digest([transport.store.namespace, payload])
        if transport.store.get("resume_request_alias", key) is None:
            transport.store.db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?)", (
                receipt["id"], receipt["stage"], receipt["payload"], receipt["status"], receipt["result"]))
            transport.store.db.commit()
            transport.store.put("imported_success_receipt", receipt["id"], {"origin": receipt["origin"],
                "payload_digest": digest(payload), "new_provider_request": False})
            transport.store.put("resume_request_alias", key, {"origin_request_id": receipt["id"], "allowed_bindings": bindings})
    request_id, raw = transport.ask(stage, prompt, maximum)
    if raw.get("status") != "success" or raw.get("finish_reason") != "stop":
        raise RuntimeError("Incomplete model response; retain checkpoint for reconciliation")
    return request_id, raw["answer"].strip()


def execute_case(case, owner, retrieval, known, bindings, client, protocol, examples):
    store = WorkerStore(owner)
    budget = RoundBudget(OUTPUT / "budget.sqlite3", 200)
    transport = ResumeTransport(store, client, budget, protocol["model"])
    transport.qid = case["qid"]
    def query(text):
        with RETRIEVAL_LOCK:
            old_store = retrieval.store
            retrieval.store = store
            try:
                return retrieval.retrieve(case["qid"], text, 10)
            finally:
                retrieval.store = old_store
    def call(stage, text, maximum):
        return ask(transport, known, bindings, stage, text, maximum)
    try:
        original_packet = query(case["question"])
        original_position = load(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{case['dataset']}/formal.positions.json")[case["qid"]]
        variants = {"old": ["c00_old_prompts", "c01_final_only"], "new": ["c02_rewrite_only", "c03_both"], "single": ["c04_single"]}
        completed = []
        for branch, labels in variants.items():
            if all((OUT / "cases" / case["dataset"] / label / (case["qid"] + ".json")).exists() for label in labels):
                completed.extend(labels)
                continue
            history, nodes = [], []
            for step in case["qdmr"]["steps"]:
                sid, question = step["id"], step["request"]
                original = query(question)
                rewritten_query = question
                rewrite_id = None
                rejected = None
                if branch != "single" and history:
                    if branch == "new":
                        prompt = prompts.rewrite(question, history)
                    else:
                        prior = "\n\n".join(f"Question: {r['request']}\nCandidate factual answer: {r['answer']}" for r in history)
                        prompt = REWRITE_INSTRUCTION + "\nOriginal question:\n" + case["question"] + "\n\nPrior question-answer pairs:\n" + prior + "\n\nCurrent question:\n" + question + "\n\nRetrieval question:"
                    rewrite_id, proposed = call(f"{branch}/rewrite/{sid}", prompt, protocol["rewrite_tokens"])
                    if proposed and proposed.endswith("?") and "\n" not in proposed:
                        rewritten_query = proposed
                    else:
                        rejected = "not_one_complete_question; original path retained"
                rewritten = query(rewritten_query) if rewritten_query != question else None
                packet = merge_packets(original, rewritten, "interleave_top10")
                child_prompt = old_prompts.child(case["question"], sid, question, history, passages(packet), examples["child"],
                    role="goal" if sid == case["qdmr"]["goal"] else "ordinary", combine=False)
                answer_id, answer = call(f"{branch}/answer/{sid}", child_prompt, protocol["child_tokens"])
                initial_answer, format_id = answer, None
                if len(answer.split()) < 4:
                    text = (
                        "Express the candidate answer below as a concise complete factual sentence that names its subject and relation. "
                        "Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. "
                        "If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n"
                        + "Question:\n" + question + "\n\nCandidate answer:\n" + answer + "\n\nContext:\n" + passages(packet)
                    )
                    format_id, answer = call(f"{branch}/sentence_format/{sid}", text, protocol["child_tokens"])
                record = {"step_id": sid, "request": question, "answer": answer}
                nodes.append({**record, "prior_complete_answers": list(history), "rewritten_query": rewritten_query,
                    "rewrite_request_id": rewrite_id, "rewrite_rejection": rejected,
                    "original_retrieval": original, "rewritten_retrieval": rewritten, "reader_packet": packet,
                    "answer_request_id": answer_id, "initial_answer": initial_answer, "sentence_format_request_id": format_id})
                history.append(record)
            helper = old_prompts.render_helper(history, None, "Answer")
            blocks = [d["block"] for d in original_packet["docs"]]
            blocks.insert(original_position - 1, helper)
            context = "\n\n".join(blocks)
            for label in labels:
                path = OUT / "cases" / case["dataset"] / label / (case["qid"] + ".json")
                if not path.exists():
                    final_prompt = (
                        prompts.final(case["question"], context, examples["final_new"])
                        if protocol["conditions"][label]["final"] == "new" else
                        old_prompts.final(case["question"], context, examples["final"])
                    )
                    final_id, prediction = call(label + "/final", final_prompt, protocol["final_tokens"])
                    result = {"qid": case["qid"], "dataset": case["dataset"], "question": case["question"],
                        "evaluation_group": case["evaluation_group"], "condition": label, "status": "success",
                        "nodes": nodes, "helper": helper, "final_context": context, "final_prompt": final_prompt,
                        "prediction": prediction, "request_id": final_id, "helper_position": original_position,
                        "original_top10_doc_ids": [d["doc_id"] for d in original_packet["docs"]]}
                    write(path, result)
                    store.put("prediction", label + "/" + case["qid"], {"qid": case["qid"], "condition": label,
                        "prediction": prediction, "status": "success", "request_id": final_id})
                completed.append(label)
        return {"dataset": case["dataset"], "qid": case["qid"], "group": case["evaluation_group"], "conditions": completed}
    finally:
        store.db.close()
        budget.db.close()


def main():
    protocol = load(HERE / "protocol.json")
    checks = {"cases": sha(HERE / "cases.frozen.jsonl") == protocol["cases_sha256"],
        "prompts": sha(HERE / "prompts.py") == protocol["prompt_source_sha256"],
        "examples": sha(HERE / "final_examples.json") == protocol["final_examples_sha256"],
        "child_prompts": sha(S02 / "prompts.py") == protocol["child_prompt_source_sha256"],
        "child_examples": sha(S02 / "examples.json") == protocol["child_examples_sha256"]}
    assert all(checks.values())
    cases = rows(HERE / "cases.frozen.jsonl")
    examples = load(S02 / "examples.json")
    examples["final_new"] = load(HERE / "final_examples.json")
    options = get_api_option(2)
    assert options["model"] == protocol["model"]
    write(OUT / "EXECUTION_FROZEN.json", {"protocol": protocol, "checks": checks, "runner_sha256": sha(HERE / "run.py"),
        "old_outputs_modified": False, "gold_in_generation": False, "sample": {"fresh": 12, "regression": 1}})
    install()
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15)), timeout=120))
        client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=120, max_retries=0, http_client=network))
        streaming = StreamingClient(client)
        for dataset in ("hotpotqa", "2wiki", "musique"):
            selected = [r for r in cases if r["dataset"] == dataset]
            resources = load(CAMPAIGN / dataset / "RESOURCE_FROZEN.json")["resources"]
            owner = Store(OUT / "api" / dataset, {"task": "fresh_prompt_revision_round1", "dataset": dataset,
                "protocol_sha256": sha(HERE / "protocol.json"), "resources": resources})
            known = seed_sources(owner, selected, resources)
            bindings = allowed_bindings(selected)
            retrieval = Retrieval(owner, resources=resources)
            print(json.dumps({"dataset": dataset, "loading": True, "questions": len(selected)}), flush=True)
            try:
                with provider_dns(urlsplit(options["base_url"]).hostname, owner):
                    retrieval.load()
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        tasks = {pool.submit(execute_case, case, owner, retrieval, known, bindings, streaming, protocol, examples): case for case in selected}
                        failures = []
                        for future in as_completed(tasks):
                            try:
                                print(json.dumps(future.result()), flush=True)
                            except BaseException as exc:
                                failures.append({"qid": tasks[future]["qid"], "type": type(exc).__name__, "reason": str(exc)[:240]})
                                for pending in tasks:
                                    pending.cancel()
                        if failures:
                            write(OUT / "ERRORS.json", failures)
                            raise RuntimeError("Generation stopped; inspect checkpoint before resuming")
            finally:
                if retrieval.chunks is not None:
                    retrieval.chunks.close()
                owner.export()
                owner.db.close()
    print("All13 questions and5 conditions completed.", flush=True)


if __name__ == "__main__":
    main()
