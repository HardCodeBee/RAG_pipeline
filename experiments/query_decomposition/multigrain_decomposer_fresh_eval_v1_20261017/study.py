"""Fresh query-only prompt development and sealed confirmation; no QA/training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from experiments.query_decomposition.multigrain_selfcontained_decomposer_v1_20261017 import run_api_pilot as api
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, utc, write_json
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
OLD = api.HERE
SEED = "20261017-fresh-multigrain-01"
QUOTAS = {"hotpotqa": {"bridge": 14, "comparison": 6},
          "2wiki": {"comparison": 5, "inference": 5, "compositional": 5, "bridge_comparison": 5},
          "musique": {"2-hop": 7, "3-hop": 7, "4-hop": 6}}


def rank(value):
    return hashlib.sha256((SEED + value).encode()).hexdigest()


def write_rows(path, values):
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values)
    if path.exists():
        assert path.read_text(encoding="utf-8") == content
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def native_array(path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, started, eof = "", False, False
        while True:
            if not eof and len(buffer) < 1048576:
                chunk = stream.read(1048576)
                buffer += chunk
                eof = not chunk
            buffer = buffer.lstrip(" \r\n\t," if started else " \r\n\t")
            if not started:
                assert buffer.startswith("[")
                buffer, started = buffer[1:], True
                continue
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(1048576)
                buffer += chunk
                eof = not chunk
                continue
            yield value
            buffer = buffer[end:]


def prepare_samples():
    if (HERE / "SELECTION_FROZEN.json").exists():
        return
    excluded_ids, excluded_text, sources = set(), set(), []
    def exclude(path):
        if not path.exists():
            return
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": api.sha(path)})
        for r in api.rows(path):
            qid = r.get("qid", r.get("id"))
            if qid:
                excluded_ids.add(qid)
            q = r.get("question", r.get("query", ""))
            if q:
                excluded_text.add(api.normalize(q))
    train_dir = ROOT / "outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data"
    exclude(train_dir / "runtime_qwen_train.jsonl")
    manifest = json.loads((train_dir / "split_manifest.json").read_text(encoding="utf-8"))
    # The manifest also contains unused candidates and source-held-out records;
    # metadata presence alone is not training or prompt exposure.
    for r in manifest["records"]:
        if r["id"] in excluded_ids or r["project_assignment"] in {"project_dev", "project_test", "reserved_test_family", "reserved_dev_family"}:
            excluded_ids.add(r.get("source", {}).get("id", ""))
    paths = [
        ROOT / "experiments/query_decomposition/multigrain_dual_retrieval_pilot_v1_20261017/cases.frozen.jsonl",
        ROOT / "experiments/query_decomposition/multigrain_dual_retrieval_expand60_v1_20261017/cases.frozen.jsonl",
        OLD / "TRAINING_EXAMPLES_DRAFT.jsonl",
        ROOT / "experiments/query_decomposition/auxiliary_qa_rag_v1/inputs_frozen/queries600.jsonl",
        ROOT / "experiments/query_decomposition/api_qdmr_suitability_v1/data/queries.jsonl",
        ROOT / "analysis/hotpotqa_query_taxonomy_validation_20261007/samples.jsonl",
    ]
    for directory in (ROOT / "experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen",
                      ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/inputs_frozen",
                      ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/compact_100_200_20261014/inputs_frozen"):
        paths.extend(directory.rglob("*.questions.jsonl"))
    for path in paths:
        exclude(path)
    screen = OUTPUT / "multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_FROZEN.json"
    excluded_ids.update(json.loads(screen.read_text(encoding="utf-8"))["ids"])
    native = ROOT / "data/multihop_comprehensive_v1_20261015"
    hp = ROOT / "analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl"
    tw = native / "2wiki/native/dev.json"
    mu = native / "musique/native/musique_ans_v1.0_dev.jsonl"
    records = []
    for dataset, path in (("hotpotqa", hp), ("2wiki", tw), ("musique", mu)):
        iterator = native_array(path) if dataset == "2wiki" else (json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip())
        for r in iterator:
            qid, question = r.get("_id", r.get("id")), r["question"]
            kind = str(len(r["question_decomposition"])) + "-hop" if dataset == "musique" else r["type"]
            records.append({"dataset": dataset, "qid": qid, "question": question, "type": kind})
    pool, seen = [], set()
    for r in sorted(records, key=lambda r: rank("dedup:" + r["qid"])):
        key = api.normalize(r["question"])
        if r["qid"] not in excluded_ids and key not in excluded_text and key not in seen:
            seen.add(key)
            pool.append(r)
    splits, used = {}, set()
    for phase in ("dev", "holdout"):
        chosen = []
        for dataset, quotas in QUOTAS.items():
            for kind, count in quotas.items():
                candidates = sorted([r for r in pool if r["dataset"] == dataset and r["type"] == kind and r["qid"] not in used], key=lambda r: rank(phase + ":" + r["qid"]))
                if len(candidates) < count:
                    raise ValueError(f"Insufficient fresh {dataset}/{kind}: {len(candidates)}; eligible=" + str(Counter((r['dataset'], r['type']) for r in pool)))
                chosen.extend(candidates[:count])
        chosen.sort(key=lambda r: rank("order:" + phase + ":" + r["qid"]))
        for i, r in enumerate(chosen):
            r.update(sample_group="fresh_" + phase, index=i + 1)
            used.add(r["qid"])
        splits[phase] = chosen
        write_rows(HERE / (phase + ".questions.frozen.jsonl"), chosen)
    assert len(used) == 120
    api.immutable_json(HERE / "SELECTION_FROZEN.json", {
        "seed": SEED, "utc": utc(), "queries_each_phase": 60, "quotas_each_phase": QUOTAS,
        "source_counts": dict(Counter(r["dataset"] for r in records)),
        "eligible_counts": dict(Counter(r["dataset"] for r in pool)),
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": api.sha(p)} for p in (hp, tw, mu)],
        "excluded_query_files": sources, "manifest_sha256": api.sha(train_dir / "split_manifest.json"),
        "excluded_ids": len(excluded_ids), "excluded_normalized_queries": len(excluded_text),
        "selection_uses_outcomes_or_answers": False,
        "dev_holdout_disjoint_ids_and_normalized_text": True,
        "holdout_policy": "Do not inspect query text/output until candidate prompt and selection rule are frozen. Never retune on confirmation.",
        "initial_comparison": "v2,v3,v4,v5 on dev60. Select best old version by pass count, then shortest prompt for ties. Optimize only on dev60; compare selected old version with final candidate on holdout60.",
        "target": "90% whole-query F/S/C/U semantic pass; uncertain and unavailable retained in planned denominator",
        "optimization_limit": "At most three new prompt candidates on development; stop if dev >=90%; then one holdout confirmation, no holdout tuning.",
        "limitations": "Type-stratified development/confirmation subsets, current Codex review not human gold. Exclusions use known source IDs and normalized exact text, not complete paraphrase or pretraining decontamination.",
        "rubric_sha256": api.sha(OLD / "PILOT_REVIEW_RULES.txt"),
    })


def prepare_run(phase, version):
    case_phase = "dev" if phase == "probe" else phase
    cases = api.rows(HERE / (case_phase + ".questions.frozen.jsonl"))
    if phase == "probe":
        assert version == "probe"
        cases = cases[2:4]
    prompt = (OLD if version in ("v2", "v3", "v4", "v5") else HERE) / ("TRAINING_PROMPT_" + version + ".txt")
    out = OUT / (phase + "_" + version)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"task": "fresh_query_only_decomposition_prompt_" + phase, "phase": phase, "version": version,
                "questions": len(cases), "model": get_api_option(2)["model"], "max_tokens": 1024,
                "temperature": 0, "enable_thinking": False, "concurrency": 3,
                "prompt_path": str(prompt.relative_to(ROOT)), "prompt_sha256": api.sha(prompt),
                "cases_sha256": api.sha(HERE / (case_phase + ".questions.frozen.jsonl")),
                "selection_sha256": api.sha(HERE / "SELECTION_FROZEN.json"),
                "rubric_sha256": api.sha(OLD / "PILOT_REVIEW_RULES.txt"),
                "inputs": "system=exact frozen prompt,user=original query only; no other fields, labels, answers, passages or prior outputs",
                "global_cap_cny": 200, "protected_stop_cny": 195, "arm_subcap_cny": 1,
                "unknown_submission_policy": "one attempt, retain unknown, no replay", "review": "current Codex explicit query-output inspection, not independent human gold"}
    if version == "v9":
        amendment = json.loads((HERE / "POSTEDIT_PLAN_FROZEN.json").read_text(encoding="utf-8"))
        parent = amendment["base_version"]
        upstream = [OUT / (phase + "_" + parent) / "cases" / parent / c["dataset"] / (c["qid"] + ".json") for c in cases]
        values = [json.loads(p.read_text(encoding="utf-8")) for p in upstream]
        protocol.update({"mode": "one_draft_plus_one_same_model_edit", "upstream_version": parent,
                         "postedit_plan_sha256": api.sha(HERE / "POSTEDIT_PLAN_FROZEN.json"),
                         "upstream_outputs_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in upstream)).hexdigest(),
                         "expected_requests": sum(available(v) for v in values),
                         "inputs": "system=exact editor prompt; user=original query plus literal parent model question-list draft; no answers, labels, critique scores, passages or qrels",
                         "cost_policy": "report inherited parent calls/tokens/cost as well as incremental editing; no free-draft or single-call claim"})
    api.immutable_json(out / "PROTOCOL_FROZEN.json", protocol)
    return cases, prompt, out, protocol


def available(value):
    return value["status"] == "success" and value.get("finish_reason") == "stop" and value.get("mechanical", {}).get("array_of_nonempty_strings", False)


def edit_input(question, draft):
    return "Original question:\n" + question + "\n\nCandidate prerequisite questions:\n" + draft


def run(phase, versions):
    if phase == "holdout":
        assert (HERE / "FINAL_CANDIDATE_FROZEN.json").exists()
    tasks = []
    for version in versions:
        cases, prompt, out, protocol = prepare_run(phase, version)
        for case in cases:
            target = out / "cases" / version / case["dataset"] / (case["qid"] + ".json")
            if target.exists():
                continue
            user_content = None
            if version == "v9":
                parent = protocol["upstream_version"]
                upstream = OUT / (phase + "_" + parent) / "cases" / parent / case["dataset"] / (case["qid"] + ".json")
                draft = json.loads(upstream.read_text(encoding="utf-8"))
                if not available(draft):
                    write_json(target, {**case, "arm": version, "request_id": None, "status": "upstream_unavailable", "upstream_file": str(upstream.relative_to(ROOT)), "upstream_sha256": api.sha(upstream), "replayed": False})
                    continue
                user_content = edit_input(case["question"], draft["answer"])
            tasks.append((case, version, prompt, out, protocol, user_content))
    tasks.sort(key=lambda t: (t[0]["index"], (versions.index(t[1]) + t[0]["index"]) % len(versions)))
    transport_directory = "transport" if phase == "dev" else "transport_" + phase
    owner = Store(OUT / transport_directory, {"task": "fresh_query_only_prompt_comparison", "phase": phase})
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=0)), timeout=90))
            options = get_api_option(2)
            raw = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=90, max_retries=0, http_client=network))
            client = StreamingClient(raw)
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, owner))
            with ThreadPoolExecutor(max_workers=3) as pool:
                unknown_batches = 0
                for i in range(0, len(tasks), 3):
                    batch, unknown, hard = tasks[i:i+3], 0, []
                    futures = [(t, pool.submit(api.run_one, t[0], t[1], t[4], client, pilot_out=t[3], arm_prompts={t[1]: str(t[2])}, user_content=t[5])) for t in batch]
                    for t, future in futures:
                        try:
                            future.result()
                        except Exception as error:
                            p = t[3] / "cases" / t[1] / t[0]["dataset"] / (t[0]["qid"] + ".json")
                            value = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                            if value.get("status") == "unknown":
                                unknown += 1
                            else:
                                hard.append(type(error).__name__)
                    print(json.dumps({"phase": phase, "versions": versions, "attempted_this_run": min(i+3, len(tasks)), "planned_unsubmitted": len(tasks), "batch_unknown": unknown, "hard_failures": hard, "utc": utc()}), flush=True)
                    unknown_batches = unknown_batches + 1 if unknown == len(batch) else 0
                    if hard or unknown_batches >= 2:
                        write_json(OUT / "TRANSPORT_STOP.json", {"utc": utc(), "classes": hard, "all_unknown_batches": unknown_batches})
                        raise RuntimeError("Stopped; all persisted requests retained without replay")
    finally:
        owner.db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--phase", choices=("dev", "holdout", "probe"), default="dev")
    p.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5"])
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare_samples()
        print("Frozen dev60 + sealed holdout60; no answers or model outcomes used.")
    if args.run:
        run(args.phase, args.versions)
