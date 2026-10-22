"""User-approved fixed90 G2/I/F goal synthesis; no new retrieval or main resume."""
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.query_decomposition.p02_goal_fewshot_synthesis_v1_20261022 import templates
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import run as pilot
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, complete_receipt, INPUT_RATE, OUTPUT_RATE)
from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc, canonical
from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PLAN = pilot.PLAN
REVIEW = PLAN / "reviews/20261022_prompt_binding_goal_synthesis" / HERE.name
OLD42 = ROOT / "outputs/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022"
DECISION = "P02_GOAL_FEWSHOT_FIXED90_EXPLICIT_NETWORK_AUTH_20261022"
GOAL_CAP, FINAL_CAP = 512, 64
read, rows, save, jsonl, sha = pilot.read, pilot.rows, pilot.save, pilot.jsonl, pilot.sha


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def cost(usage):
    return usage["prompt_tokens"] * INPUT_RATE + usage["completion_tokens"] * OUTPUT_RATE


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve this frozen experiment")
    config, examples = pilot.settings()
    questions = rows(pilot.COHORT / "runtime90.frozen.jsonl")
    controls = {r["id"]: r for r in rows(pilot.SOURCE)}
    bindings = {r["id"]: r for r in rows(pilot.COHORT / "SOURCE_BINDINGS.frozen.jsonl")}
    inputs, history = [], []
    for q in questions:
        old = controls[q["id"]]
        t = old["trajectory"]
        goal = t["nodes"][-1]
        assert old["status"] == "success" and t["finish_reason"] == "stop"
        assert goal["role"] == "goal" and goal["request"] == q["question"]
        assert all(n["role"] == "ordinary" for n in t["nodes"][:-1])
        assert [n["request"] for n in t["nodes"][:-1]] == q["supporting_questions"]
        assert sha(ROOT / bindings[q["id"]]["source"]) == bindings[q["id"]]["source_sha256"]
        ordinary = [{k: n[k] for k in ("step_id", "request", "answer")} for n in t["nodes"][:-1]]
        excluded = {"legacy/final", "legacy/answer/" + str(goal["step_id"]),
                    "legacy/sentence_format/" + str(goal["step_id"])}
        prefix = [c["usage"] for c in old["calls"] if c["stage"] not in excluded]
        record = {**{k: q[k] for k in ("id", "qid", "dataset", "question", "type")},
            "ordinary_nodes": ordinary, "has_ordinary": bool(ordinary), "goal_step_id": goal["step_id"],
            "goal_rewritten_query": goal["rewritten_query"], "goal_reader_packet": goal["reader_packet"],
            "goal_original_packet": goal["original_retrieval"], "final_root_packet": t["original_retrieval"],
            "historical_goal_prompt": goal["child_prompt"], "source_sha256": bindings[q["id"]]["source_sha256"],
            "frozen_prefix_usage": {k: sum(u[k] for u in prefix) for k in ("prompt_tokens", "completion_tokens")}}
        assert not set(record) & {"reference", "gold", "answer", "prediction", "f1", "em"}
        inputs.append(record)
        history.append({"id": q["id"], "dataset": q["dataset"], "prediction": old["prediction"],
                        "source": bindings[q["id"]]["source"],
                        "historical_full_usage": {k: sum(c["usage"][k] for c in old["calls"])
                                                  for k in ("prompt_tokens", "completion_tokens")}})
    assert len(inputs) == len({r["id"] for r in inputs}) == 90
    assert Counter(r["dataset"] for r in inputs) == {d: 30 for d in pilot.DATASETS}
    unknown = []
    with closing(ro(OLD42 / "checkpoint.sqlite3")) as db:
        for rid, stage, payload, status, raw in db.execute("SELECT id,stage,payload,status,result FROM requests"):
            if status in ("unknown", "in_flight"):
                result = json.loads(raw) if raw else {}
                unknown.append({"request_id": rid, "stage": stage, "status": status,
                    "payload_sha256": digest(json.loads(payload)), "source_checkpoint": str((OLD42 / "checkpoint.sqlite3").relative_to(ROOT)),
                    "usage": result.get("usage"), "http_status": result.get("http_status"),
                    "automatic_reissue": False})
    jsonl(HERE / "INPUTS90.frozen.jsonl", inputs)
    jsonl(HERE / "HISTORY90.local.frozen.jsonl", history)
    jsonl(HERE / "EXTERNAL_UNKNOWN.frozen.jsonl", unknown)
    save(HERE / "GENERATION.frozen.json", config["generation"])
    save(HERE / "FINAL_EXAMPLES.frozen.json", examples)
    (HERE / "I.frozen.txt").write_text(templates.INSTRUCTIONS, encoding="utf8")
    (HERE / "F_EXAMPLES.frozen.txt").write_text(templates.EXAMPLES, encoding="utf8")
    pinned = [HERE / n for n in ("run.py", "INPUTS90.frozen.jsonl", "HISTORY90.local.frozen.jsonl",
              "EXTERNAL_UNKNOWN.frozen.jsonl", "GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json",
              "I.frozen.txt", "F_EXAMPLES.frozen.txt")]
    pinned += [Path(templates.__file__), pilot.HERE / "templates.py", pilot.HERE / "run.py", pilot.SOURCE,
               pilot.COHORT / "runtime90.frozen.jsonl", pilot.COHORT / "references90.local.frozen.jsonl",
               pilot.PACKAGE / "config.json", pilot.PACKAGE / "core/final_prompts.py"]
    holds = [PLAN / "STOP_REQUESTED", ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
             ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    protocol = {"id": HERE.name, "utc": utc(), "decision_id": DECISION, "cases": 90, "conditions": 270,
        "authority": "User explicitly answered the external API/data/spending authorization question: 使用固定90题 直接允许 以后不要询问我",
        "approved_external_destination": config["generation"]["base_url"],
        "approved_model": config["generation"]["model"],
        "approved_data": "Fixed90 public questions, existing retrieved passages and previous QA, exact displayed instructions and four fictional demonstrations; credential only authenticates the existing endpoint",
        "future_small_experiments": "User requests no repeated permission questions within this approved small-experiment scope; no main resumption or publication authority",
        "arms": {"G2": "Old synthesis prompt; exact complete payloads reused where present, otherwise generate once",
                 "I": "Displayed revised instructions without examples", "F": "Identical I plus exactly four displayed synthetic examples"},
        "fixed": "Exactly existing fixed90 order/IDs, actual frozen R02/v12, ordinary QA, goal rewrite/both routes/merged packet, final root/common512 synthesis wrapper, final protocol/model/decoding",
        "goal_cap": 512, "final_cap": 64, "formatter": False,
        "empty_decomposition": "Keep all six in full90; separately report ordinary-present84 sensitivity without replacement",
        "cache": "Full payload exact, stop, nonempty, model identity, valid returned usage; no quality rerolls",
        "retry": "SDK/HTTP retries0; at most5 durable actual known technical attempts; unknown from old42 or current requests held, never automatically reissued",
        "external_unknown_guard": {"conditions": len(unknown), "policy": "Retain prior raw receipts/reserves; same payload cannot be submitted by a new directory"},
        "cost": "Existing shared ledger and serialized gate; input/output tokens, logical stage/fullflow estimate and actual new receipt costs separate; unknown not zero",
        "measurements": "Native F1/EM and paired gains/ties/losses; no new timing or LLM-call-count metric",
        "interpretation": "Exposed development fixed90; frozen-prefix goal/final intervention, not fresh complete RAG, independent test, or pure reasoning-text effect",
        "stop_after": "Fixed270 outcomes or documented technical holds; no adaptive candidate changes or extra sample",
        "main_evaluation_resumed": False, "publication_authorized": False,
        "pins": {str(p.relative_to(ROOT)): sha(p) for p in pinned},
        "hold_pins": {str(p.relative_to(ROOT)): sha(p) for p in holds if p.exists()}}
    save(HERE / "PROTOCOL.frozen.json", protocol)
    save(REVIEW / "PROTOCOL.json", protocol)
    with (PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"id": DECISION, "utc": utc(), "experiment": HERE.name,
            "authority": protocol["authority"], "approved": True, "external_destination": protocol["approved_external_destination"],
            "external_data": protocol["approved_data"], "reasonable_API_spending_approved": True,
            "cases": 90, "conditions": 270, "main_evaluation_resumed": False,
            "unknown_reissues_approved": False, "repeated_small_scope_permission_questions": False,
            "protocol_sha256": sha(HERE / "PROTOCOL.frozen.json")}, ensure_ascii=False) + "\n")
    return dry_run()


def check():
    p = read(HERE / "PROTOCOL.frozen.json")
    for group in ("pins", "hold_pins"):
        for name, expected in p[group].items():
            assert sha(ROOT / name) == expected, name
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert any(d.get("id") == DECISION and d.get("approved") is True for d in rows(PLAN / "DECISIONS.jsonl"))
    config, examples = pilot.settings()
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    assert examples == read(HERE / "FINAL_EXAMPLES.frozen.json")
    return config, examples


def dry_run():
    config, examples = check()
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
    unknown = {r["payload_sha256"] for r in rows(HERE / "EXTERNAL_UNKNOWN.frozen.jsonl")}
    blocked = []
    for r in rows(HERE / "INPUTS90.frozen.jsonl"):
        ip, fp = (templates.goal_prompt(r, a) for a in ("I", "F"))
        assert fp.replace("\n" + templates.EXAMPLES, "", 1) == ip
        assert r["question"] in ip and passages(r["goal_reader_packet"]) in ip
        assert all(n["request"] in ip and n["answer"] in ip for n in r["ordinary_nodes"])
        for a in templates.ARMS:
            payload = build_payload(config["generation"], templates.goal_prompt(r, a), GOAL_CAP)
            if digest(payload) in unknown:
                blocked.append({"id": r["id"], "arm": a})
        f, c = templates.old.final_prompt(r, "FIXED CANDIDATE", config, examples)
        assert "FIXED CANDIDATE" in f and passages(r["final_root_packet"]) in c
    result = {"passed": True, "cases": 90, "conditions": 270, "I_F_only_examples_differ": True,
              "external_unknown_conditions_blocked": blocked, "main_evaluation_remains_paused": True,
              "new_API_requests": 0}
    save(OUT / "DRYRUN.json", result)
    return result


def seed_db():
    seed = sqlite3.connect(":memory:")
    seed.execute("CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT)")
    with closing(ro(pilot.OUT / "pilot15/checkpoint.sqlite3")) as old:
        for payload, rid, status, raw in old.execute("SELECT payload,id,status,result FROM requests"):
            result = {**json.loads(raw), "status": status} if raw else {}
            if complete_receipt(result) and result.get("returned_model") == json.loads(payload)["model"]:
                seed.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?)", (payload, rid, canonical(result)))
    return seed


def execute(limit=None):
    config, examples = check()
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    unknown = {r["payload_sha256"]: r for r in rows(HERE / "EXTERNAL_UNKNOWN.frozen.jsonl")}
    current = {}
    def event(value):
        with (OUT / "trace.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps({"utc": utc(), **current, **value}, ensure_ascii=False) + "\n")
    class NetworkEvents:
        def event(self, name, **fields):
            event({"event": name, **fields})
    count = 0
    with run_lock(OUT), ExitStack() as stack:
        cache = stack.enter_context(Cache(OUT / "checkpoint.sqlite3"))
        cache.put("meta", "identity", {"protocol_sha256": sha(HERE / "PROTOCOL.frozen.json")})
        seed = stack.enter_context(closing(seed_db()))
        budget = stack.enter_context(closing(NoCapBudget(config, OUT)))
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        stack.enter_context(provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()))
        live = stack.enter_context(closing(StrictAsk(cache, seed, config, budget, event,
            recover_unknown=False, maximum_attempts=5, stop=lambda: (OUT / "STOP_REQUESTED").exists())))
        for r in rows(HERE / "INPUTS90.frozen.jsonl"):
            for arm in templates.ARMS:
                if (OUT / "STOP_REQUESTED").exists() or (limit is not None and count >= limit):
                    return {"status": "stopped_at_local_limit", "new_conditions": count}
                key = arm + "|" + r["id"]
                if cache.get("prediction", key) is not None:
                    continue
                current.clear()
                current.update(id=r["id"], dataset=r["dataset"], arm=arm, has_ordinary=r["has_ordinary"])
                prompt = templates.goal_prompt(r, arm)
                payload = build_payload(config["generation"], prompt, GOAL_CAP)
                if digest(payload) in unknown:
                    cache.put("prediction", key, {**current, "status": "external_unknown_held",
                        "prediction": None, "goal_text": None, "prior_request": unknown[digest(payload)]})
                    event({"event": "external_unknown_guard", "prior_request_id": unknown[digest(payload)]["request_id"]})
                    print(json.dumps({**current, "status": "external_unknown_held"}), flush=True)
                    count += 1
                    continue
                try:
                    gid, goal, _ = live("goal/" + key, prompt, GOAL_CAP)
                    final, context = templates.old.final_prompt(r, goal, config, examples)
                    final_payload = build_payload(config["generation"], final, FINAL_CAP)
                    if digest(final_payload) in unknown:
                        raise RuntimeError("External unknown final payload is held")
                    fid, prediction, _ = live("final/" + key, final, FINAL_CAP)
                    receipts = {x["request_id"]: x for x in cache.requests()}
                    used = {stage: {"usage": receipts[rid]["result"]["usage"], "request_id": rid}
                            for stage, rid in (("goal", gid), ("final", fid))}
                    cache.put("prediction", key, {**current, "status": "success", "goal_text": goal,
                        "prediction": prediction, "usage_by_stage": used, "goal_request_id": gid,
                        "final_request_id": fid})
                    event({"event": "condition_complete", "goal_final_CNY": sum(cost(u["usage"]) for u in used.values())})
                    print(json.dumps({**current, "status": "success"}), flush=True)
                except Exception as exc:
                    terminal = cache.db.execute("SELECT q.status,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage IN (?,?) ORDER BY q.rowid DESC LIMIT 1", ("goal/" + key, "final/" + key)).fetchone()
                    result = json.loads(terminal[1]) if terminal and terminal[1] else {}
                    algorithm = terminal and terminal[0] == "success" and result.get("usage") and (result.get("finish_reason") in ("length", "content_filter") or result.get("refusal") or not (result.get("answer") or "").strip())
                    status = "algorithm_failure" if algorithm else "technical_held"
                    cache.put("prediction", key, {**current, "status": status, "prediction": "" if algorithm else None,
                        "goal_text": None, "error_type": type(exc).__name__, "error": str(exc)[:300]})
                    event({"event": "condition_terminal", "status": status, "error": str(exc)[:300]})
                    print(json.dumps({**current, "status": status}), flush=True)
                    if terminal and terminal[0] in ("unknown", "in_flight"):
                        return {"status": "new_unknown_network_hold", "new_conditions": count + 1}
                count += 1
    return {"status": "execution_finished", "new_conditions": count}


def report():
    config, examples = check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers = {ds: metric(ds) for ds in pilot.DATASETS}
    refs = {r["id"]: r for r in rows(pilot.COHORT / "references90.local.frozen.jsonl")}
    inputs = {r["id"]: r for r in rows(HERE / "INPUTS90.frozen.jsonl")}
    history = {r["id"]: r for r in rows(HERE / "HISTORY90.local.frozen.jsonl")}
    with closing(ro(OUT / "checkpoint.sqlite3")) as db:
        details = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        receipts = [(status, json.loads(raw) if raw else {}) for status, raw in db.execute("SELECT status,result FROM requests")]
        retired = [(status, json.loads(raw) if raw else {}) for status, raw in db.execute("SELECT status,result FROM retired_requests")]
    for r in details:
        valid = r["status"] in ("success", "algorithm_failure")
        r["native_metrics"] = scorers[r["dataset"]](r["prediction"], refs[r["id"]]) if valid else {"f1": None, "em": None}
        usages = [u["usage"] for u in r.get("usage_by_stage", {}).values()]
        r["input_tokens"] = sum(u["prompt_tokens"] for u in usages) if usages else None
        r["output_tokens"] = sum(u["completion_tokens"] for u in usages) if usages else None
        r["goal_final_CNY"] = sum(cost(u) for u in usages) if usages else None
        prefix = inputs[r["id"]]["frozen_prefix_usage"]
        r["logical_fullflow_CNY"] = cost(prefix) + r["goal_final_CNY"] if usages else None
    index = {(r["arm"], r["id"]): r for r in details}
    summaries, paired = {}, {}
    for subset in ("full90", "ordinary84"):
        summaries[subset], paired[subset] = {}, {}
        for ds in ("all",) + pilot.DATASETS:
            ids = [ident for ident, r in inputs.items() if (subset == "full90" or r["has_ordinary"]) and (ds == "all" or r["dataset"] == ds)]
            summaries[subset][ds], paired[subset][ds] = {}, {}
            for arm in templates.ARMS:
                selected = [index[(arm, ident)] for ident in ids if (arm, ident) in index]
                valid = [r for r in selected if r["native_metrics"]["f1"] is not None]
                success = [r for r in selected if r["status"] == "success"]
                complete = len(valid) == len(ids)
                summaries[subset][ds][arm] = {"planned": len(ids), "terminal": len(selected), "scorable": len(valid),
                    "F1": sum(r["native_metrics"]["f1"] for r in valid) / len(ids) if complete else None,
                    "EM": sum(r["native_metrics"]["em"] for r in valid) / len(ids) if complete else None,
                    "available_only_F1": sum(r["native_metrics"]["f1"] for r in valid) / len(valid) if valid else None,
                    "available_only_EM": sum(r["native_metrics"]["em"] for r in valid) / len(valid) if valid else None,
                    "mean_input_tokens": sum(r["input_tokens"] for r in success) / len(success) if success else None,
                    "mean_output_tokens": sum(r["output_tokens"] for r in success) / len(success) if success else None,
                    "mean_goal_final_CNY": sum(r["goal_final_CNY"] for r in success) / len(success) if success else None,
                    "mean_logical_fullflow_CNY": sum(r["logical_fullflow_CNY"] for r in success) / len(success) if success else None,
                    "statuses": dict(Counter(r["status"] for r in selected))}
            for a, b in (("G2", "I"), ("G2", "F"), ("I", "F")):
                changes, em_changes, counts = [], [], Counter()
                for ident in ids:
                    x, y = index.get((a, ident)), index.get((b, ident))
                    if not x or not y or x["native_metrics"]["f1"] is None or y["native_metrics"]["f1"] is None:
                        continue
                    delta = y["native_metrics"]["f1"] - x["native_metrics"]["f1"]
                    changes.append(delta)
                    em_changes.append(y["native_metrics"]["em"] - x["native_metrics"]["em"])
                    counts["improved" if delta > 1e-12 else "harmed" if delta < -1e-12 else "tied"] += 1
                paired[subset][ds][a + "_to_" + b] = {"paired": len(changes), **dict(counts),
                    "F1_delta": sum(changes) / len(changes) if changes else None,
                    "EM_delta": sum(em_changes) / len(em_changes) if em_changes else None}
    common = [ident for ident in inputs if all((arm, ident) in index and index[(arm, ident)]["native_metrics"]["f1"] is not None for arm in templates.ARMS)]
    common_summary = {}
    for arm in templates.ARMS:
        common_summary[arm] = {"n": len(common),
            "F1": sum(index[(arm, i)]["native_metrics"]["f1"] for i in common) / len(common) if common else None,
            "EM": sum(index[(arm, i)]["native_metrics"]["em"] for i in common) / len(common) if common else None}
    namespace = "p02_prompt_research_v1_20261022/" + str(OUT.resolve())
    with closing(ro(config["budget"]["ledger"])) as db:
        actual, reserved = db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", (namespace,)).fetchone()
    result = {"status": "complete" if len(details) == 270 and len(common) == 90 else "partial_or_documented_holds",
        "terminal_conditions": len(details), "planned_conditions": 270, "by_subset_dataset_arm": summaries,
        "paired_F1": paired, "same_three_arm_complete_subset": common_summary, "common_ids": common,
        "fresh_actual_CNY": actual, "unknown_reserved_CNY": reserved,
        "external_old42_unknown_reserved_CNY": read(OLD42 / "REPORT.json")["unknown_reserved_CNY"],
        "receipt_statuses": dict(Counter(s for s, r in receipts + retired)),
        "not_fresh_complete_RAG": True, "not_independent_confirmation": True, "main_evaluation_remains_paused": True}
    jsonl(OUT / "DETAILS.scored.local.jsonl", details)
    save(OUT / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare", "dry-run", "run", "report"):
        mode.add_argument("--" + option, action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = prepare() if args.prepare else dry_run() if args.dry_run else execute(args.limit) if args.run else report()
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
