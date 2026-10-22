"""Prepare/dry-run are offline. Only explicit --run can open the existing API.

No retrieval, decomposition, GPU execution or external judge is implemented.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
COHORT = HERE.parent / "p02_noformatter_quick90_v1_20261021"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
SOURCE = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_three_prompts_quick90_v1_20261021/control90.jsonl"
DATASETS = ("hotpotqa", "2wiki", "musique")
SEED = 20261022
GOAL_CAP, FINAL_CAP = 512, 64
from . import templates


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temp.replace(path)


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf8")


def order(label):
    return hashlib.sha256(f"{SEED}|goal_pilot15|{label}".encode()).hexdigest()


def select(questions):
    chosen, quotas = [], {}
    for ds in DATASETS:
        strata = defaultdict(list)
        for row in questions:
            if row["dataset"] == ds:
                strata[row["type"]].append(row)
        population = sum(map(len, strata.values()))
        assert population >= 5
        assert len(strata) <= 5
        quota = {kind: 1 for kind in strata}
        # Preserve every available type, then allocate remaining seats by the
        # greatest proportional deficit; ties and within-type order use hashes.
        while sum(quota.values()) < 5:
            kind = min(strata, key=lambda k: (quota[k] - 5 * len(strata[k]) / population,
                                              order(ds + "|quota|" + k)))
            quota[kind] += 1
        for kind in sorted(strata):
            chosen += sorted(strata[kind], key=lambda v: order(ds + "|" + v["id"]))[:quota[kind]]
        quotas[ds] = quota
    assert len(chosen) == len({q["id"] for q in chosen}) == 15
    return chosen, quotas


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve the frozen pilot; use --dry-run or --run")
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    # Selection is completed before opening trajectories or any local gold.
    questions = [{k: q[k] for k in ("id", "qid", "dataset", "question", "type")}
                 for q in rows(COHORT / "runtime90.frozen.jsonl") if q["supporting_questions"]]
    selected, quotas = select(questions)
    jsonl(HERE / "questions15.frozen.jsonl", selected)
    save(HERE / "COHORT.frozen.json", {
        "seed": SEED, "planned": 15, "per_dataset": 5, "strata": quotas,
        "selection_fields": ["dataset", "type", "id"], "gold_scores_used": False,
        "method": "minimum one per available type, proportional-deficit allocation, SHA256 order",
        "population": "existing exposed quick90 development questions; not independent confirmation",
        "eligible_per_dataset": dict(Counter(q["dataset"] for q in questions)),
        "eligibility": "Nonempty frozen supporting_questions; every original quick90 trajectory is complete; no score-based selection",
        "selected_ids": [q["id"] for q in selected],
    })
    controls = {v["id"]: v for v in rows(SOURCE)}
    bindings = {v["id"]: v for v in rows(COHORT / "SOURCE_BINDINGS.frozen.jsonl")}
    inputs, history = [], []
    for q in selected:
        control = controls[q["id"]]
        t = control["trajectory"]
        assert control["status"] == "success" and t["finish_reason"] == "stop"
        goal = t["nodes"][-1]
        assert goal["role"] == "goal" and goal["request"].strip() == q["question"].strip()
        assert all(n["role"] == "ordinary" for n in t["nodes"][:-1])
        assert len(goal["reader_packet"]["docs"]) == 10
        assert len(t["original_retrieval"]["docs"]) == 10
        assert sha(ROOT / bindings[q["id"]]["source"]) == bindings[q["id"]]["source_sha256"]
        ordinary = [{k: deepcopy(n[k]) for k in ("step_id", "request", "answer")}
                    for n in t["nodes"][:-1]]
        assert ordinary, "Goal-combination pilot excludes empty decomposition routes"
        rec = {**q, "ordinary_nodes": ordinary, "goal_step_id": goal["step_id"],
               "goal_rewritten_query": goal["rewritten_query"],
               "goal_reader_packet": deepcopy(goal["reader_packet"]),
               "goal_original_packet": deepcopy(goal["original_retrieval"]),
               "final_root_packet": deepcopy(t["original_retrieval"]),
               "historical_goal_prompt": goal["child_prompt"],
               "source_sha256": bindings[q["id"]]["source_sha256"]}
        assert not set(rec) & {"prediction", "gold", "answer", "f1", "em", "supporting_facts"}
        inputs.append(rec)
        # Kept outside inference inputs; report reads historical endpoint only.
        history.append({"id": q["id"], "dataset": q["dataset"],
                        "prediction": control["prediction"], "calls": control["calls"],
                        "goal_answer": goal["answer"], "goal_step_id": goal["step_id"],
                        "source": bindings[q["id"]]["source"]})
    jsonl(HERE / "inputs15.frozen.jsonl", inputs)
    jsonl(HERE / "historical15.local.jsonl", history)
    config = read(PACKAGE / "config.json")
    # Do not alter the frozen native config or inherit its obsolete CNY300 cap.
    save(HERE / "GENERATION.frozen.json", config["generation"])
    final_examples = read(PACKAGE / "prompts/final_examples.json")
    save(HERE / "FINAL_EXAMPLES.frozen.json", final_examples)
    for arm, text in (("G1", templates.G1), ("G2", templates.G2)):
        (HERE / (arm + ".frozen.txt")).write_text(text, encoding="utf8")
    holds = [PLAN / "STOP_REQUESTED", ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
             ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    pinned = [HERE / n for n in ["run.py", "templates.py", "questions15.frozen.jsonl",
              "inputs15.frozen.jsonl", "COHORT.frozen.json", "GENERATION.frozen.json",
              "FINAL_EXAMPLES.frozen.json", "G1.frozen.txt", "G2.frozen.txt"]]
    pinned += [PACKAGE / "config.json", PACKAGE / "core/final_prompts.py", SOURCE]
    protocol = {
        "id": HERE.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "User delegated autonomous scoped prompt-binding and goal-synthesis small experiments; root records the exact authority in shared DECISIONS.jsonl.",
        "scope": "Frozen-prefix goal and final regeneration pilot; not a complete fresh-RAG or formal evaluation",
        "arms": {"G0": "historical one-factual-sentence goal prompt; new512 cap and no formatter",
                 "G1": "short factual summary, target80-120 words",
                 "G2": "key facts plus essential composition plus candidate conclusion/gap, target80-120 words"},
        "fixed": "Original quick90 ordinary QA, decomposition, goal rewrite, goal reader packet, original final root, model and final prompt/readout/examples",
        "goal_max_tokens": GOAL_CAP, "final_max_tokens": FINAL_CAP,
        "formatter": False, "goal_presentation": "same neutral Candidate information for the Original question",
        "historical_P02_is_separate": "G0 is not historical P02:512 cap, no formatter and common final packaging",
        "cache": "exact full payload+stop+usage only; no historical128 goal reuse; generated summary reused only within this pilot",
        "retry": "SDK/HTTP retries0; up to5 durable actual attempts across restarts for technical failures, including authorized unknown reissue; retired receipts and unknown reservations preserved; successful and algorithm failures never regenerated for score",
        "expense": "existing provider token rates and global receipts; no old300 or per-run hard ceiling; unknown liabilities preserved; input/output tokens and estimated CNY reported",
        "measurement": "native F1/EM, paired improves/ties/harms, input/output tokens, estimated CNY; no LLM-call-count metric or new timing claim",
        "interpretation": "Overall goal-output protocol comparison. G1/G2 content can differ; not a pure causal estimate of reasoning text alone",
        "followups_prepared_not_default": ["final_goal_none: ordinary QA and root only, no new goal generation",
             "goal_rewrite_off: use saved goal original-route packet; preserve ordinary QA and final root"],
        "main_evaluation_remains_paused": True, "new_retrieval_or_GPU": False,
        "pins": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in pinned},
        "hold_pins": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in holds if p.exists()},
    }
    save(HERE / "PROTOCOL.frozen.json", protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    return dry_run()


def check():
    protocol = read(HERE / "PROTOCOL.frozen.json")
    for group in ("pins", "hold_pins"):
        for name, expected in protocol[group].items():
            assert sha(ROOT / name) == expected, (group, name)
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def settings():
    config = read(PACKAGE / "config.json")
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    return config, read(HERE / "FINAL_EXAMPLES.frozen.json")


def dry_run():
    protocol = check()
    config, examples = settings()
    from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    contracts = []
    for rec in rows(HERE / "inputs15.frozen.jsonl"):
        final_prompts = []
        for arm in templates.ARMS:
            prompt = templates.goal_prompt(rec, arm)
            assert rec["question"] in prompt
            for n in rec["ordinary_nodes"]:
                assert n["request"] in prompt and n["answer"] in prompt
            from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
            assert passages(rec["goal_reader_packet"]) in prompt
            payload = build_payload(config["generation"], prompt, GOAL_CAP)
            assert payload["max_tokens"] == 512
            final, context = templates.final_prompt(rec, "SYNTHETIC SAME GOAL CONTENT", config, examples)
            assert passages(rec["final_root_packet"]) in context
            assert "SYNTHETIC SAME GOAL CONTENT" in final
            final_prompts.append(final)
            contracts.append({"id": rec["id"], "arm": arm,
                "goal_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "goal_max_tokens": GOAL_CAP, "final_max_tokens": FINAL_CAP})
            off = templates.goal_prompt(rec, arm, rewrite_off=True)
            if arm == "G0":
                old_passages = passages(rec["goal_reader_packet"])
                assert prompt.count(old_passages) == 1
                assert off == prompt.replace(old_passages, passages(rec["goal_original_packet"]), 1)
        assert len(set(final_prompts)) == 1
        none_prompt, _ = templates.final_prompt(rec, "", config, examples, include_goal=False)
        assert "SYNTHETIC SAME GOAL CONTENT" not in none_prompt
        assert "Candidate information for the Original question" not in none_prompt
    jsonl(OUT / "DRYRUN_CONTRACTS.jsonl", contracts)
    sample = rows(HERE / "inputs15.frozen.jsonl")[0]
    for arm in templates.ARMS:
        (OUT / (arm + ".rendered.txt")).write_text(templates.goal_prompt(sample, arm), encoding="utf8")
    result = {"passed": True, "pilot_cases": 15, "goal_arm_pairs_checked": len(contracts),
              "shared_final_protocol_verified": True, "followup_final_none_rendered": True,
              "G0_rewrite_off_only_packet_text_changed": True,
              "protocol_sha256": sha(HERE / "PROTOCOL.frozen.json"),
              "new_API_requests": 0, "main_evaluation_remains_paused": True}
    save(OUT / "DRYRUN.json", result)
    return result


def execute(arms, *, rewrite_off=False, final_none=False):
    check()
    authorization = read(HERE / "RUN_AUTHORIZATION.json")
    assert authorization.get("approved") is True and authorization.get("experiment") == HERE.name
    decisions = rows(PLAN / "DECISIONS.jsonl")
    assert any(d.get("id") == authorization["decision_id"] for d in decisions), "Missing shared decision authority"
    from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc
    from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, StrictAsk
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    from urllib.parse import urlsplit
    config, examples = settings()
    records = rows(HERE / "inputs15.frozen.jsonl")
    run_id = "final_goal_none" if final_none else ("rewrite_off" if rewrite_off else "pilot15")
    directory = OUT / run_id
    directory.mkdir(parents=True, exist_ok=True)
    current = {}
    def event(value):
        with (directory / "trace.jsonl").open("a", encoding="utf8") as h:
            h.write(json.dumps({"utc": utc(), **current, **value}, ensure_ascii=False) + "\n")
    class NetworkEvents:
        def event(self, name, **fields):
            event({"event": name, **fields})
    with run_lock(directory), ExitStack() as stack:
        cache = stack.enter_context(Cache(directory / "checkpoint.sqlite3"))
        cache.put("meta", "identity", {"protocol_sha256": sha(HERE / "PROTOCOL.frozen.json"),
                   "run_id": run_id, "goal_cap": GOAL_CAP, "final_cap": FINAL_CAP})
        seed = stack.enter_context(closing(sqlite3.connect(":memory:")))
        seed.execute("CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT)")
        budget = stack.enter_context(closing(NoCapBudget(config, directory)))
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        stack.enter_context(provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()))
        live = stack.enter_context(closing(StrictAsk(cache, seed, config, budget, event,
                     recover_unknown=True, maximum_attempts=5,
                     stop=lambda: (directory / "STOP_REQUESTED").exists())))
        for rec in records:
            if (directory / "STOP_REQUESTED").exists():
                break
            for arm in (("N",) if final_none else arms):
                key = arm + "|" + rec["id"]
                if cache.get("prediction", key) is not None:
                    continue
                current.clear()
                current.update(id=rec["id"], dataset=rec["dataset"], arm=arm)
                try:
                    goal_text, goal_id = "", None
                    if not final_none:
                        goal_id, goal_text, _ = live("goal/" + key,
                                templates.goal_prompt(rec, arm, rewrite_off), GOAL_CAP)
                        receipt = next(r for r in cache.requests() if r["request_id"] == goal_id)
                        assert receipt["result"].get("usage"), "No usage; incomplete receipt"
                    final, context = templates.final_prompt(rec, goal_text, config, examples, include_goal=not final_none)
                    final_id, prediction, _ = live("final/" + key, final, FINAL_CAP)
                    receipt = next(r for r in cache.requests() if r["request_id"] == final_id)
                    assert receipt["result"].get("usage"), "No usage; incomplete receipt"
                    cache.put("prediction", key, {"id": rec["id"], "dataset": rec["dataset"],
                        "arm": arm, "status": "success", "prediction": prediction,
                        "goal_text": goal_text, "goal_request_id": goal_id,
                        "final_request_id": final_id, "final_prompt": final, "final_context": context})
                    print(json.dumps({**current, "status": "success"}), flush=True)
                except Exception as exc:
                    # Ask preserves unresolved receipts and reservations. Resume
                    # reuses all complete successes and holds unknown payloads.
                    event({"event": "case_held", "error_type": type(exc).__name__, "error": str(exc)[:300]})
                    print(json.dumps({**current, "status": "held", "error_type": type(exc).__name__}), flush=True)
    return report(run_id)


def report(run_id="pilot15"):
    # Gold is first opened here, after generation; never enters frozen inputs.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    directory = OUT / run_id
    refs = {r["id"]: r for r in rows(COHORT / "references90.local.frozen.jsonl")}
    hist = {r["id"]: r for r in rows(HERE / "historical15.local.jsonl")}
    # Never construct Cache for monitoring: its constructor changes in_flight
    # to unknown. A report must not mutate a live checkpoint or execution state.
    with closing(sqlite3.connect((directory / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        results = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='prediction' ORDER BY rowid")]
        receipts = []
        for rid, stage, payload, status, started, completed, raw in db.execute("SELECT * FROM requests ORDER BY rowid"):
            receipts.append({"request_id": rid, "stage": stage, "status": status,
                             "result": json.loads(raw) if raw else None})
        retired = []
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_requests'").fetchone():
            for rid, stage, payload, status, started, completed, raw in db.execute("SELECT * FROM retired_requests"):
                retired.append({"request_id": rid, "stage": stage, "status": status,
                                "result": json.loads(raw) if raw else None})
    receipt_index = {r["request_id"]: r for r in receipts}
    for r in results:
        r["native_metrics"] = metric(r["dataset"])(r["prediction"], refs[r["id"]])
        r["historical_native_metrics"] = metric(r["dataset"])(hist[r["id"]]["prediction"], refs[r["id"]])
        used = [receipt_index[i]["result"]["usage"] for i in (r["goal_request_id"], r["final_request_id"]) if i]
        r["input_tokens"] = sum(u["prompt_tokens"] for u in used)
        r["output_tokens"] = sum(u["completion_tokens"] for u in used)
        r["estimated_goal_final_CNY"] = r["input_tokens"] * 0.8e-6 + r["output_tokens"] * 2e-6
        h = hist[r["id"]]
        replaced = {"legacy/final"}
        if run_id != "final_goal_none":
            replaced |= {"legacy/answer/" + str(h["goal_step_id"]),
                         "legacy/sentence_format/" + str(h["goal_step_id"])}
        prefix = [c["usage"] for c in h["calls"] if c["stage"] not in replaced]
        r["fixed_prefix_input_tokens"] = sum(u["prompt_tokens"] for u in prefix)
        r["fixed_prefix_output_tokens"] = sum(u["completion_tokens"] for u in prefix)
        r["full_flow_logical_input_tokens"] = r["input_tokens"] + r["fixed_prefix_input_tokens"]
        r["full_flow_logical_output_tokens"] = r["output_tokens"] + r["fixed_prefix_output_tokens"]
        r["estimated_full_flow_logical_CNY"] = r["full_flow_logical_input_tokens"] * 0.8e-6 + r["full_flow_logical_output_tokens"] * 2e-6
        r["goal_word_count"] = len(r["goal_text"].split())
    groups = {}
    for arm in sorted({r["arm"] for r in results}):
        groups[arm] = {}
        for ds in DATASETS:
            rr = [r for r in results if r["arm"] == arm and r["dataset"] == ds]
            groups[arm][ds] = {"planned": 5, "complete": len(rr),
                "F1": sum(r["native_metrics"]["f1"] for r in rr) / len(rr) if rr else None,
                "EM": sum(r["native_metrics"]["em"] for r in rr) / len(rr) if rr else None,
                "mean_goal_final_input_tokens": sum(r["input_tokens"] for r in rr) / len(rr) if rr else None,
                "mean_goal_final_output_tokens": sum(r["output_tokens"] for r in rr) / len(rr) if rr else None,
                "mean_full_flow_logical_input_tokens": sum(r["full_flow_logical_input_tokens"] for r in rr) / len(rr) if rr else None,
                "mean_full_flow_logical_output_tokens": sum(r["full_flow_logical_output_tokens"] for r in rr) / len(rr) if rr else None,
                "mean_goal_word_count": sum(r["goal_word_count"] for r in rr) / len(rr) if rr else None,
                "estimated_goal_final_CNY": sum(r["estimated_goal_final_CNY"] for r in rr)}
    paired = {}
    indexed = {(r["arm"], r["id"]): r for r in results}
    for a, b in (("G0", "G1"), ("G0", "G2"), ("G1", "G2")):
        counts = Counter()
        for ident in hist:
            if (a, ident) in indexed and (b, ident) in indexed:
                delta = indexed[b, ident]["native_metrics"]["f1"] - indexed[a, ident]["native_metrics"]["f1"]
                counts["improved" if delta > 1e-12 else "harmed" if delta < -1e-12 else "tied"] += 1
        paired[a + "_to_" + b] = dict(counts)
    usage = [r["result"]["usage"] for r in receipts + retired if r.get("result") and r["result"].get("usage")]
    result = {"scope": "15 previously exposed development questions, frozen-prefix goal/final protocol pilot",
        "status": "complete" if len(results) == (15 if run_id == "final_goal_none" else 45) else "partial",
        "by_arm_dataset": groups, "paired_F1": paired,
        "known_receipt_estimated_CNY": sum(u["prompt_tokens"] * 0.8e-6 + u["completion_tokens"] * 2e-6 for u in usage),
        "unresolved_receipts": [{"request_id": r["request_id"], "status": r["status"]} for r in receipts if r["status"] != "success"],
        "retired_receipts_preserved": retired,
        "historical_P02_separate": True, "not_pure_reasoning_causal_effect": True,
        "main_evaluation_remains_paused": True}
    jsonl(directory / "DETAILS.scored.local.jsonl", results)
    save(directory / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=templates.ARMS, default=list(templates.ARMS))
    parser.add_argument("--rewrite-off", action="store_true")
    parser.add_argument("--final-goal-none", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.dry_run:
        result = dry_run()
    elif args.run:
        result = execute(args.arms, rewrite_off=args.rewrite_off, final_none=args.final_goal_none)
    else:
        result = report("final_goal_none" if args.final_goal_none else "rewrite_off" if args.rewrite_off else "pilot15")
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
