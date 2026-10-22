"""42 exposed development cases: cached G2, instructions I, four-shot F.

Only --run opens the provider; old checkpoints and the formal campaign stay intact.
"""
from collections import Counter
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
from . import templates
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import run as pilot
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, complete_receipt, INPUT_RATE, OUTPUT_RATE)
from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc, canonical
from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PLAN = pilot.PLAN
REVIEW = PLAN / "reviews/20261022_prompt_binding_goal_synthesis" / HERE.name
DECISION = "P02_GOAL_FEWSHOT_SYNTHESIS42_SIDE_APPROVED_20261022"
SOURCES = (("pilot15", pilot.HERE, pilot.OUT / "pilot15", "inputs15.frozen.jsonl"),
           ("challenge27", HERE.parent / "p02_goal_composition_challenge_v1_20261022",
            ROOT / "outputs/query_decomposition/p02_goal_composition_challenge_v1_20261022/challenge27",
            "inputs27.frozen.jsonl"))
GOAL_CAP, FINAL_CAP = 512, 64
read, rows, save, jsonl, sha = pilot.read, pilot.rows, pilot.save, pilot.jsonl, pilot.sha


def usage_cost(usage):
    return usage["prompt_tokens"] * INPUT_RATE + usage["completion_tokens"] * OUTPUT_RATE


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve this frozen experiment; use --dry-run or --run")
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    config, examples = pilot.settings()
    records, controls, proofs = [], [], []
    for cohort, source, output, filename in SOURCES:
        with closing(sqlite3.connect((output / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
            by_id = {json.loads(v)["id"]: json.loads(v) for (v,) in db.execute(
                "SELECT value FROM data WHERE kind='prediction'") if json.loads(v)["arm"] == "G2"}
            for record in rows(source / filename):
                assert not set(record) & {"reference", "gold", "answer", "prediction", "f1", "em"}
                record = {**record, "cohort": cohort}
                old = by_id[record["id"]]
                final, _ = templates.old.final_prompt(record, old["goal_text"], config, examples)
                used = {}
                for stage, prompt, cap in (("goal", templates.goal_prompt(record, "G2"), GOAL_CAP),
                                           ("final", final, FINAL_CAP)):
                    rid = old[stage + "_request_id"]
                    payload, status, raw = db.execute(
                        "SELECT payload,status,result FROM requests WHERE id=?", (rid,)).fetchone()
                    receipt = {**json.loads(raw), "status": status}
                    assert complete_receipt(receipt)
                    assert json.loads(payload) == build_payload(config["generation"], prompt, cap)
                    used[stage] = {"usage": receipt["usage"], "source_request_id": rid,
                                   "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()}
                records.append(record)
                controls.append({"id": record["id"], "dataset": record["dataset"], "cohort": cohort,
                    "arm": "G2", "status": "success", "goal_text": old["goal_text"],
                    "prediction": old["prediction"], "usage_by_stage": used,
                    "reused_frozen_control": True})
        proofs.append({"cohort": cohort, "inputs": str((source / filename).relative_to(ROOT)),
                       "sha256": sha(source / filename), "control_results": len(by_id),
                       "source_checkpoint": str((output / "checkpoint.sqlite3").relative_to(ROOT))})
    assert len(records) == len({r["id"] for r in records}) == 42
    jsonl(HERE / "inputs42.frozen.jsonl", records)
    jsonl(HERE / "controls42.frozen.jsonl", controls)
    save(HERE / "GENERATION.frozen.json", config["generation"])
    save(HERE / "FINAL_EXAMPLES.frozen.json", examples)
    (HERE / "I.frozen.txt").write_text(templates.INSTRUCTIONS, encoding="utf8")
    (HERE / "F_EXAMPLES.frozen.txt").write_text(templates.EXAMPLES, encoding="utf8")
    holds = [PLAN / "STOP_REQUESTED", ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
             ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    pins = [HERE / n for n in ("run.py", "templates.py", "inputs42.frozen.jsonl", "controls42.frozen.jsonl",
            "GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json", "I.frozen.txt", "F_EXAMPLES.frozen.txt")]
    pins += [pilot.HERE / "templates.py", pilot.PACKAGE / "config.json", pilot.PACKAGE / "core/final_prompts.py"]
    protocol = {"id": HERE.name, "created_utc": utc(), "decision_id": DECISION,
        "user_authority": "Side-conversation user explicitly requested: 实验一下, following the displayed I/F four-shot candidate and three-way comparison.",
        "scope": "Frozen-prefix goal/final regeneration, not fresh complete RAG or independent formal evidence",
        "cohorts": {"pilot15": "Existing stratified exposed development pilot, 5 per dataset",
                    "challenge27": "Existing targeted historical goal-error sample: 17 2Wiki, 10 HotpotQA"},
        "arms": {"G2": "Exact prompt-matched complete existing goal/final receipts, no paid rerun",
                 "I": "Displayed new instructions, no demonstrations",
                 "F": "Same I instructions plus exactly the four displayed synthetic demonstrations"},
        "fixed": "Frozen actual R02/v12 decomposition, ordinary QA, goal rewrite and retrieval packet, final root/prompt/examples, model and decoding",
        "caps": {"goal": GOAL_CAP, "final": FINAL_CAP}, "formatter": False,
        "output_note": "I/F candidate suffix and no word target differ from G2; I-vs-F isolates demonstration insertion, G2-vs-I is an instruction package contrast",
        "cache": "Only exact complete payload with nonempty stop-ended content and valid returned usage; validate both G2 stages before starting",
        "retry": "Existing serial shared provider gate; SDK/HTTP retries0, maximum5 durable technical attempts; unknown requests held, no automatic paid reissue; algorithm outcomes never rerolled",
        "expense": "Existing shared cumulative ledger, all actual usage/reservations preserved; report goal+final marginal logical and fresh actual cost separately",
        "timing": "No new full-flow timing claim; no GPU, retrieval, decomposition or judge execution",
        "stop_after": "Exactly42 x2 new conditions; no adaptive prompt change or automatic expansion",
        "main_evaluation_resumed": False, "publication_authorized": False,
        "sources": proofs, "pins": {str(p.relative_to(ROOT)): sha(p) for p in pins},
        "hold_pins": {str(p.relative_to(ROOT)): sha(p) for p in holds if p.exists()}}
    save(HERE / "PROTOCOL.frozen.json", protocol)
    save(REVIEW / "PROTOCOL.json", protocol)
    entry = {"id": DECISION, "utc": utc(), "experiment": HERE.name,
             "authority": protocol["user_authority"], "cases": 42, "new_arms": ["I", "F"],
             "cached_control": "G2 exact complete receipts", "main_evaluation_resumed": False,
             "protocol": str((HERE / "PROTOCOL.frozen.json").relative_to(ROOT))}
    with (PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return dry_run()


def check():
    protocol = read(HERE / "PROTOCOL.frozen.json")
    for group in ("pins", "hold_pins"):
        for name, expected in protocol[group].items():
            assert sha(ROOT / name) == expected, name
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert any(d.get("id") == DECISION for d in rows(PLAN / "DECISIONS.jsonl"))
    config, examples = pilot.settings()
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    assert examples == read(HERE / "FINAL_EXAMPLES.frozen.json")
    return config, examples


def dry_run():
    config, examples = check()
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
    records = rows(HERE / "inputs42.frozen.jsonl")
    for record in records:
        assert not set(record) & {"reference", "gold", "answer", "prediction", "f1", "em"}
        ip, fp = (templates.goal_prompt(record, arm) for arm in ("I", "F"))
        assert fp.replace("\n" + templates.EXAMPLES, "", 1) == ip
        assert all(n["request"] in ip and n["answer"] in ip for n in record["ordinary_nodes"])
        assert record["question"] in ip and passages(record["goal_reader_packet"]) in ip
        for arm in templates.ARMS:
            payload = build_payload(config["generation"], templates.goal_prompt(record, arm), GOAL_CAP)
            assert payload["max_tokens"] == 512
        final, context = templates.old.final_prompt(record, "FIXED CANDIDATE", config, examples)
        assert "FIXED CANDIDATE" in context and passages(record["final_root_packet"]) in final
    result = {"passed": True, "cases": len(records), "goal_conditions": len(records) * 3,
              "exact_baseline_receipt_checks": 84, "I_F_only_examples_differ": True,
              "main_evaluation_remains_paused": True, "new_API_requests": 0}
    save(OUT / "DRYRUN.json", result)
    return result


def seed_db():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT)")
    for cohort, source, output, filename in SOURCES:
        with closing(sqlite3.connect((output / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True)) as old:
            for payload, rid, status, raw in old.execute("SELECT payload,id,status,result FROM requests"):
                receipt = {**json.loads(raw), "status": status} if raw else {}
                if complete_receipt(receipt):
                    db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?)", (payload, rid, canonical(receipt)))
    return db


def execute(limit=None):
    config, examples = check()
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    current = {}
    baselines = {r["id"]: r for r in rows(HERE / "controls42.frozen.jsonl")}
    costs = {"I": [], "F": []}
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
        for record in rows(HERE / "inputs42.frozen.jsonl"):
            for arm in ("I", "F"):
                if (OUT / "STOP_REQUESTED").exists() or (limit is not None and count >= limit):
                    return {"status": "stopped_at_local_limit", "new_conditions": count}
                key = arm + "|" + record["id"]
                if cache.get("prediction", key) is not None:
                    continue
                current.clear()
                current.update(id=record["id"], dataset=record["dataset"], cohort=record["cohort"], arm=arm)
                try:
                    gid, goal, _ = live("goal/" + key, templates.goal_prompt(record, arm), GOAL_CAP)
                    prompt, context = templates.old.final_prompt(record, goal, config, examples)
                    fid, prediction, _ = live("final/" + key, prompt, FINAL_CAP)
                    receipts = {r["request_id"]: r for r in cache.requests()}
                    used = {stage: {"usage": receipts[rid]["result"]["usage"], "request_id": rid}
                            for stage, rid in (("goal", gid), ("final", fid))}
                    value = {**current, "status": "success", "goal_text": goal, "prediction": prediction,
                             "goal_request_id": gid, "final_request_id": fid, "usage_by_stage": used}
                    cache.put("prediction", key, value)
                    actual = sum(usage_cost(r["usage"]) for r in used.values())
                    baseline = sum(usage_cost(r["usage"]) for r in baselines[record["id"]]["usage_by_stage"].values())
                    costs[arm].append((actual, baseline))
                    ratio = actual / baseline
                    event({"event": "condition_complete", "estimated_goal_final_CNY": actual,
                           "same_query_G2_cost_ratio": ratio})
                    if ratio > 3 or (len(costs[arm]) >= 20 and
                            sum(a for a, b in costs[arm][-20:]) > 1.5 * sum(b for a, b in costs[arm][-20:])):
                        event({"event": "expense_notice", "scope": "goal_final_stage_pair_only",
                               "same_query_G2_cost_ratio": ratio, "condition_cost_CNY": actual})
                    print(json.dumps({**current, "status": "success", "cost_CNY": round(actual, 6)}, ensure_ascii=True), flush=True)
                except Exception as exc:
                    terminal = cache.db.execute("SELECT r.status,r.result FROM requests r JOIN request_stages s ON s.request_id=r.id WHERE s.stage IN (?,?) ORDER BY r.rowid DESC LIMIT 1", ("goal/" + key, "final/" + key)).fetchone()
                    result = json.loads(terminal[1]) if terminal and terminal[1] else {}
                    algorithm = bool(terminal and terminal[0] == "success" and not complete_receipt({**result, "status": terminal[0]}))
                    status = "algorithm_failure" if algorithm else "technical_held"
                    value = {**current, "status": status, "prediction": "", "goal_text": "",
                             "error_type": type(exc).__name__, "error": str(exc)[:300]}
                    cache.put("prediction", key, value)
                    event({"event": "condition_terminal", **value})
                    print(json.dumps({**current, "status": status, "error_type": type(exc).__name__}), flush=True)
                count += 1
    return {"status": "execution_finished", "new_conditions": count}


def report():
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    refs = {r["id"]: r for r in rows(pilot.COHORT / "references90.local.frozen.jsonl")}
    refs.update({r["id"]: r for r in rows(SOURCES[1][1] / "references27.local.frozen.jsonl")})
    scorers = {dataset: metric(dataset) for dataset in ("hotpotqa", "2wiki", "musique")}
    records = rows(HERE / "inputs42.frozen.jsonl")
    details = rows(HERE / "controls42.frozen.jsonl")
    with closing(sqlite3.connect((OUT / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        details += [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        receipts = [(rid, payload, status, json.loads(raw) if raw else {})
                    for rid, payload, status, raw in db.execute("SELECT id,payload,status,result FROM requests")]
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_requests'").fetchone():
            receipts += [(rid, payload, status, json.loads(raw) if raw else {})
                         for rid, payload, status, raw in db.execute("SELECT id,payload,status,result FROM retired_requests")]
    for r in details:
        r["native_metrics"] = (scorers[r["dataset"]](r["prediction"], refs[r["id"]]) if r["status"] == "success"
            else {"f1": 0., "em": 0.} if r["status"] == "algorithm_failure" else {"f1": None, "em": None})
        usages = [u["usage"] for u in r.get("usage_by_stage", {}).values()]
        r["input_tokens"] = sum(u["prompt_tokens"] for u in usages) if usages else None
        r["output_tokens"] = sum(u["completion_tokens"] for u in usages) if usages else None
        r["estimated_goal_final_CNY"] = sum(usage_cost(u) for u in usages) if usages else None
        r["goal_words"] = len(r["goal_text"].split())
    index = {(r["arm"], r["id"]): r for r in details}
    summaries, paired = {}, {}
    for cohort in ("pilot15", "challenge27"):
        summaries[cohort], paired[cohort] = {}, {}
        for dataset in ("all", "hotpotqa", "2wiki", "musique"):
            ids = [r["id"] for r in records if r["cohort"] == cohort and (dataset == "all" or r["dataset"] == dataset)]
            if not ids:
                continue
            summaries[cohort][dataset] = {}
            for arm in templates.ARMS:
                selected = [index[(arm, ident)] for ident in ids if (arm, ident) in index]
                valid = [r for r in selected if r["native_metrics"]["f1"] is not None]
                successful = [r for r in selected if r["status"] == "success"]
                summaries[cohort][dataset][arm] = {"planned": len(ids), "terminal": len(selected), "scored": len(valid),
                    "F1": sum(r["native_metrics"]["f1"] for r in valid) / len(valid) if valid else None,
                    "EM": sum(r["native_metrics"]["em"] for r in valid) / len(valid) if valid else None,
                    "mean_input_tokens": sum(r["input_tokens"] for r in successful) / len(successful) if successful else None,
                    "mean_output_tokens": sum(r["output_tokens"] for r in successful) / len(successful) if successful else None,
                    "mean_goal_final_CNY": sum(r["estimated_goal_final_CNY"] for r in successful) / len(successful) if successful else None,
                    "mean_goal_words": sum(r["goal_words"] for r in successful) / len(successful) if successful else None,
                    "statuses": dict(Counter(r["status"] for r in selected))}
            paired[cohort][dataset] = {}
            for a, b in (("G2", "I"), ("G2", "F"), ("I", "F")):
                deltas, counts = [], Counter()
                for ident in ids:
                    x, y = index.get((a, ident)), index.get((b, ident))
                    if not x or not y or x["native_metrics"]["f1"] is None or y["native_metrics"]["f1"] is None:
                        continue
                    delta = y["native_metrics"]["f1"] - x["native_metrics"]["f1"]
                    deltas.append(delta)
                    counts["improved" if delta > 1e-12 else "harmed" if delta < -1e-12 else "tied"] += 1
                paired[cohort][dataset][a + "_to_" + b] = {"paired": len(deltas), **dict(counts),
                    "mean_F1_delta": sum(deltas) / len(deltas) if deltas else None}
    config, _ = check()
    namespace = "p02_prompt_research_v1_20261022/" + str(OUT.resolve())
    with closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        actual, unresolved = db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", (namespace,)).fetchone()
    result = {"status": "complete" if len(details) == 126 and all(r["status"] in ("success", "algorithm_failure") for r in details) else "partial",
        "cohorts_reported_separately": True, "by_cohort_dataset_arm": summaries, "paired_F1": paired,
        "fresh_actual_CNY": actual, "unknown_reserved_CNY": unresolved,
        "receipt_statuses": dict(Counter(status for rid, payload, status, raw in receipts)),
        "not_independent_confirmation": True, "not_pure_reasoning_effect": True,
        "no_full_flow_timing_claim": True, "main_evaluation_remains_paused": True}
    jsonl(OUT / "DETAILS.scored.local.jsonl", details)
    save(OUT / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare", "dry-run", "run", "report"):
        mode.add_argument("--" + option, action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = prepare() if args.prepare else dry_run() if args.dry_run else execute(args.limit) if args.run else report()
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
