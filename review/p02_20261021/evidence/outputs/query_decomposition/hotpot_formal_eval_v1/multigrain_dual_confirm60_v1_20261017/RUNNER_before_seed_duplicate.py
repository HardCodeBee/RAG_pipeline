"""Replicate original v1 behavior on original24 plus36 new questions."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, digest
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import CAMPAIGN, rows, load, write
from .prepare import HERE, OUT, ORIGINAL_OUT, ORIGINAL
from experiments.query_decomposition.multigrain_prompt_v3_20261017.run import ask as cached_ask
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.methods import single, dual


RETRIEVAL_LOCK = Lock()
REWRITE_INSTRUCTION = """Write one optional retrieval rewrite of the current question using relevant prior question-answer pairs. Prior answers are candidate facts, not verified evidence. Use their full statements to identify which entity or relation an answer belongs to.
Preserve the current question's requested object, relation, scope, dates, negation, and comparison direction. Replace a descriptive reference with a relevant explicit entity only when the prior statement actually supplies that identity. Do not concatenate a complete answer sentence into a noun slot. Do not use a statement of missing evidence as an entity, a negative fact, zero, or an empty set. Do not arbitrarily select one member of a multi-valued answer. Do not insert factual comparison answers or solve the question.
When no relevant usable prior information improves the retrieval expression, return the current question exactly unchanged. Return only one complete question, with no explanation or Markdown.
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PilotBudget(ParallelBudget):
    def reserve(self, request_id, namespace, amount):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges").fetchone()[0]
            local = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()[0]
            if total + amount > 195 or local + amount > 5:
                raise RuntimeError("Protected global budget or five-yuan pilot subcap reached")
            if self.db.execute("SELECT 1 FROM charges WHERE id=?", (request_id,)).fetchone():
                raise RuntimeError("Existing liability cannot be replayed")
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def passages(packet):
    return "\n\n".join(d["block"] for d in packet["docs"])


def merge_packets(original, rewritten, policy):
    if rewritten is None:
        return original
    assert policy in {"interleave_top10", "union_top20"}
    ordered = []
    seen = set()
    for rank in range(10):
        for packet in (original, rewritten):
            if rank >= len(packet["docs"]):
                continue
            doc = packet["docs"][rank]
            if doc["doc_id"] not in seen:
                seen.add(doc["doc_id"])
                ordered.append(doc)
    if policy == "interleave_top10":
        ordered = ordered[:10]
    docs = []
    for rank, doc in enumerate(ordered, 1):
        docs.append({**doc, "rank": rank, "block": f'[Chunk {rank} | source={doc["source"]}, pages=1-1]\n{doc["text"]}'})
    return {"qid": original["qid"], "query": original["query"], "k": len(docs), "docs": docs, "merge_policy": policy}


def ask(transport, stage, prompt, maximum):
    return cached_ask(transport, transport.expansion_known, transport.expansion_bindings, stage, prompt, maximum)


def seed_sources(owner, selected, resources):
    dataset = selected[0]['dataset']
    sources = [ORIGINAL_OUT/'api'/dataset/'checkpoint.sqlite3',
        ORIGINAL_OUT/'musique_baseline_screening_network_v2/checkpoint.sqlite3' if dataset=='musique' else
        CAMPAIGN/dataset/'formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3']
    sources += list((CAMPAIGN/dataset/'formal/b00_top10_v1').glob('*/checkpoint.sqlite3'))
    known = {}
    for source in dict.fromkeys(sources):
        if not source.exists():
            continue
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as db:
            for case in selected:
                for key,raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval' AND json_extract(value,'$.qid')=?",(case['qid'],)):
                    packet=json.loads(raw)
                    assert key==digest([packet['qid'],packet['query'],packet['k'],resources])
                    owner.put('retrieval',key,packet)
            if source != sources[0]:
                continue
            for rid,stage,payload,status,raw in db.execute("SELECT * FROM requests WHERE status='success'"):
                result=json.loads(raw)
                if result.get('status')!='success' or result.get('finish_reason')!='stop':
                    continue
                owner.db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?)',(rid,stage,payload,status,raw))
                known[digest(json.loads(payload))]={'id':rid,'stage':stage,'payload':payload,'status':status,'result':raw,'origin':str(source.relative_to(ROOT))}
                owner.put('imported_success_receipt',rid,{'origin':str(source.relative_to(ROOT)),'new_provider_request':False})
            owner.db.commit()
    return known


def run_case(case, protocol, owner, retrieval, client, model, examples, positions):
    store = WorkerStore(owner)
    budget = PilotBudget(OUTPUT / "budget.sqlite3", 200)
    transport = ResumeTransport(store, client, budget, model)
    transport.qid = case["qid"]
    transport.expansion_known = owner.expansion_known
    transport.expansion_bindings = owner.expansion_bindings
    def retrieve(query):
        # Protect the shared encoder/chunk store and use a thread-local SQLite
        # connection. Only exact query/resource packets are shared between arms.
        with RETRIEVAL_LOCK:
            old = retrieval.store
            retrieval.store = store
            try:
                return retrieval.retrieve(case["qid"], query, 10)
            finally:
                retrieval.store = old
    try:
        original_packet = retrieve(case["question"])
        conditions = [single, dual] if protocol["matched_single_control"] else [dual]
        outputs = []
        for condition in conditions:
            destination = OUT / "cases" / case["dataset"] / condition.ID / (case["qid"] + ".json")
            if destination.exists():
                outputs.append(load(destination))
                continue
            history = []
            nodes = []
            for step in case["qdmr"]["steps"]:
                sid, question = step["id"], step["request"]
                is_goal = sid == case["qdmr"]["goal"]
                if is_goal and not protocol["execute_goal"]:
                    continue
                original = retrieve(question)
                rewrite = question
                rewrite_request_id = None
                rewrite_rejection = None
                if condition.USE_REWRITE and history:
                    prior = "\n\n".join(f'Question: {r["request"]}\nCandidate factual answer: {r["answer"]}' for r in history)
                    text = REWRITE_INSTRUCTION + "\nOriginal question:\n" + case["question"] + "\n\nPrior question-answer pairs:\n" + prior + "\n\nCurrent question:\n" + question + "\n\nRetrieval question:"
                    rewrite_request_id, proposed = ask(transport, condition.ID + f"/rewrite/{sid}", text, protocol["rewrite_tokens"])
                    if proposed and proposed.endswith("?") and "\n" not in proposed:
                        rewrite = proposed
                    else:
                        rewrite_rejection = "not_one_complete_question; original path retained"
                rewritten = retrieve(rewrite) if rewrite != question else None
                packet = merge_packets(original, rewritten, protocol["merge_policy"])
                child = prompts.child(case["question"], sid, question, history, passages(packet), examples["child"], role="goal" if is_goal else "ordinary", combine=False)
                answer_request_id, answer = ask(transport, condition.ID + f"/answer/{sid}", child, protocol["child_answer_tokens"])
                format_request_id = None
                initial_answer = answer
                if len(answer.split()) < 4:
                    repair = (
                        "Express the candidate answer below as a concise complete factual sentence that names its subject and relation. "
                        "Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. "
                        "If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n"
                        + "Question:\n" + question + "\n\nCandidate answer:\n" + answer + "\n\nContext:\n" + passages(packet)
                    )
                    format_request_id, answer = ask(transport, condition.ID + f"/sentence_format/{sid}", repair, protocol["child_answer_tokens"])
                # Never reduce answer to an entity, scalar, substring, or JSON value.
                prior_record = {"step_id": sid, "request": question, "answer": answer}
                history.append(prior_record)
                nodes.append({
                    **prior_record, "role": "goal" if is_goal else "ordinary",
                    "prior_complete_answers": history[:-1],
                    "original_retrieval": original,
                    "rewritten_query": rewrite,
                    "rewritten_retrieval": rewritten,
                    "rewrite_request_id": rewrite_request_id,
                    "rewrite_rejection": rewrite_rejection,
                    "reader_packet": packet,
                    "answer_request_id": answer_request_id,
                    "initial_answer": initial_answer,
                    "sentence_format_request_id": format_request_id,
                    "original_doc_ids_displaced": sorted({d["doc_id"] for d in original["docs"]} - {d["doc_id"] for d in packet["docs"]}),
                })
            helper = prompts.render_helper(history, None, "Answer") if history else ""
            blocks = [d["block"] for d in original_packet["docs"]]
            position = positions[case["qid"]]
            if helper:
                blocks.insert(position - 1, helper)
            final_context = "\n\n".join(blocks)
            final_prompt = prompts.final(case["question"], final_context, examples["final"]) if helper else prompts.V1.format(question=case["question"], context=final_context)
            final_id, answer = ask(transport, condition.ID + "/final", final_prompt, protocol["final_answer_tokens"])
            result = {"qid": case["qid"], "dataset": case["dataset"], "question": case["question"], "condition": condition.ID, "nodes": nodes, "helper": helper, "final_context": final_context, "final_prompt": final_prompt, "prediction": answer, "request_id": final_id, "original_top10_doc_ids": [d["doc_id"] for d in original_packet["docs"]], "helper_position": position, "status": "success"}
            write(destination, result)
            outputs.append(result)
        return {"dataset": case["dataset"], "qid": case["qid"], "completed_conditions": [r["condition"] for r in outputs]}
    finally:
        store.db.close()
        budget.db.close()


def main():
    protocol_path = HERE / "protocol.json"
    if not protocol_path.exists():
        raise RuntimeError("Await the three user clarifications; no candidate API calls allowed")
    protocol = load(protocol_path)
    assert type(protocol["execute_goal"]) is bool
    assert type(protocol["matched_single_control"]) is bool
    assert protocol["merge_policy"] in {"interleave_top10", "union_top20"}
    cases = rows(HERE / "cases.frozen.jsonl")
    assert len(cases) == 60
    assert all(sha(ROOT/p)==v for p,v in protocol["frozen_files"].items())
    assert all(sha(ROOT/p)==v for p,v in protocol["original_output_hashes"].items())
    source = ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful"
    examples = load(source / "examples.json")
    options = get_api_option(2)
    assert options["model"] == protocol["model"]
    install()
    write(OUT / "EXECUTION_FROZEN.json", {"protocol": protocol, "protocol_sha256": sha(protocol_path), "cases_sha256": sha(HERE / "cases.frozen.jsonl"), "code_sha256": sha(Path(__file__)), "child_and_final_prompts_sha256": sha(source / "prompts.py"), "historical_outputs_modified": False, "gold_answers_in_generation_inputs": False, "cache_policy": "copy original24 outputs and paid receipts, cache only exact successful payloads; all36 new question IDs unused; per-condition histories isolated"})
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15)), timeout=120))
        client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=120, max_retries=0, http_client=network))
        streaming = StreamingClient(client)
        for dataset in ("hotpotqa", "2wiki", "musique"):
            selected = [r for r in cases if r["dataset"] == dataset]
            resource = load(CAMPAIGN / dataset / "RESOURCE_FROZEN.json")["resources"]
            owner = Store(OUT / "api" / dataset, {"task": "multigrain_dual_retrieval_diagnostic", "dataset": dataset, "protocol_sha256": sha(protocol_path), "model": options["model"], "resources": resource})
            owner.expansion_known = seed_sources(owner,selected,resource)
            bindings=[]
            for case in selected:
                for label in ('single_original','dual_original_rewritten'):
                    bindings.append({'qid':case['qid'],'stage':label+'/final'})
                    for n in case['qdmr']['steps']:
                        bindings += [{'qid':case['qid'],'stage':label+'/'+kind+'/'+str(n['id'])} for kind in ('rewrite','answer','sentence_format')]
            owner.expansion_bindings=bindings
            print(json.dumps({'dataset':dataset,'questions':len(selected),'loading':True}),flush=True)
            retrieval = Retrieval(owner, resources=resource)
            positions = load(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.positions.json")
            try:
                with provider_dns(urlsplit(options["base_url"]).hostname, owner):
                    retrieval.load()
                    fixed={(c["qid"],n["request"]) for c in selected for n in c["qdmr"]["steps"]}
                    retrieval.retrieve_many([{"qid":qid,"query":q} for qid,q in sorted(fixed)],10)
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        pending = {pool.submit(run_case, case, protocol, owner, retrieval, streaming, options["model"], examples, positions): case for case in selected}
                        errors = []
                        for future in as_completed(pending):
                            try:
                                value = future.result()
                                print(json.dumps(value), flush=True)
                            except BaseException as error:
                                errors.append({"qid": pending[future]["qid"], "type": type(error).__name__, "reason": str(error)[:240]})
                                for task in pending:
                                    task.cancel()
                        if errors:
                            write(OUT / "ERRORS.json", errors)
                            raise RuntimeError("Pilot stopped; inspect private checkpoints before any resume")
            finally:
                if retrieval.chunks is not None:
                    retrieval.chunks.close()
                owner.export()
                owner.db.close()
    print("All diagnostic generation completed; run offline report next.", flush=True)


if __name__ == "__main__":
    main()
