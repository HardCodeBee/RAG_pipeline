"""Offline preparation and root-authorized fresh synchronous scope/role probe.

No import or --prepare/--dry-run/--report operation submits provider requests.
"""
from collections import Counter
from contextlib import ExitStack, closing
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
CHALLENGE = HERE.parent / "p02_goal_composition_challenge_v1_20261022"
PILOT = HERE.parent / "p02_goal_synthesis_probe_v1_20261022"
SHARED = HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
RUN = OUT / "scope27"
DATASETS = ("2wiki", "hotpotqa")
SEED, GOAL_CAP, FINAL_CAP = 20261022, 512, 64
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022.run import read, rows, save, sha, jsonl
from experiments.query_decomposition.p02_goal_scope_role_probe_v1_20261022 import templates


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def order(data):
    offset = int(digest(str(SEED))[:8], 16) % 3
    result = []
    for i, record in enumerate(data):
        rotation = (offset + i) % 3
        arms = templates.ARMS[rotation:] + templates.ARMS[:rotation]
        result.extend({"case_index": i, "arm_index": j, "id": record["id"], "arm": arm}
                      for j, arm in enumerate(arms))
    return result


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve the frozen experiment")
    for name in ("inputs27.frozen.jsonl", "references27.local.frozen.jsonl", "SELECTION.frozen.jsonl",
                 "PUBLIC_SOURCE_PROOF.jsonl", "PUBLIC_BLOB_METADATA.json", "GENERATION.frozen.json",
                 "FINAL_EXAMPLES.frozen.json", "G2.frozen.txt"):
        (HERE / name).write_bytes((CHALLENGE / name).read_bytes())
    data = rows(HERE / "inputs27.frozen.jsonl")
    assert len(data) == 27 and Counter(r["dataset"] for r in data) == {"2wiki": 17, "hotpotqa": 10}
    assert all(r["ordinary_nodes"] for r in data)
    assert [r["id"] for r in rows(HERE / "SELECTION.frozen.jsonl")] == [r["id"] for r in data]
    assert all(r["member_bytes_exact"] and r["ordinary_QA_and_reader_from_exact_public_source"]
               for r in rows(HERE / "PUBLIC_SOURCE_PROOF.jsonl"))
    jsonl(HERE / "ORDER.frozen.jsonl", order(data))
    save(HERE / "CONSTRAINTS.frozen.json", {"S0": "Unchanged original G2 prompt; fresh requests in this new namespace",
        "S1": {"append_one_sentence": templates.SCOPE, "mechanism": "Preserve query target and predicate; do not add work-specific award or complete-set identity requirements"},
        "S2": {"append_one_sentence": templates.ROLE, "mechanism": "Preserve entity roles, relation direction/generation and requested attribute"},
        "independent": "S1 and S2 are never combined; final prompt and all input packets remain identical"})
    pins = [HERE / n for n in ("run.py", "templates.py", "PROTOCOL.md", "inputs27.frozen.jsonl",
             "references27.local.frozen.jsonl", "SELECTION.frozen.jsonl", "PUBLIC_SOURCE_PROOF.jsonl",
             "PUBLIC_BLOB_METADATA.json", "GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json",
             "G2.frozen.txt", "ORDER.frozen.jsonl", "CONSTRAINTS.frozen.json")]
    pins += [PILOT / "templates.py", PILOT / "run.py", SHARED, PACKAGE / "config.json",
             PACKAGE / "core/cache.py", PACKAGE / "core/llm.py", PACKAGE / "core/final_prompts.py",
             CHALLENGE / "PROTOCOL.frozen.json"]
    protocol = {"id": HERE.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "prepared_only_not_started", "planned": 27, "per_dataset": {"2wiki": 17, "hotpotqa": 10},
        "arms": list(templates.ARMS), "conditions": 81, "goal_cap": GOAL_CAP, "final_cap": FINAL_CAP,
        "selection": "All prior challenge27, no new native-score selection; ordinary-correct/historical-goal-wrong diagnostic selection remains exposed",
        "inference": "Frozen ordinary QA, original goal rewrite and reader passages, final root packet; no retrieval, decomposer, formatter or external judge",
        "scope": "Exploratory targeted constraints; neither historical G2 nor reused27 is independent confirmation",
        "synchronous_control": "S0 is byte-identical old G2 prompt, generated afresh with no cross-experiment seed/cache; current S0 compared to historical G2 only as exploratory temporal/noise sensitivity",
        "cache": "Only exact complete stop+usage same-run payload reuse for resume/deduplication; never import historical G2 requests",
        "ordering": {"seed": SEED, "rule": "Frozen case order; cyclic arm rotation using seed-hash starting offset; each arm starts9 cases"},
        "recovery": "Shared StrictAsk, SDK/HTTP retries0; at most5 actual technical attempts including initial across restart; unknown receipts/reservations retained; only authorized no-return-evidence unknown recovery",
        "terminal": "Known status-success length/empty/refusal is an immutable algorithm condition outcome, no reroll, F1=EM=0 in fixed planned denominator; goal terminal suppresses final; missing usage/model mismatch/technical unknown remain separate held outcomes, never silently zero",
        "reporting": "All arms and datasets; fixed-denominator metrics only if every condition is resolved; known-outcome and complete-only sensitivity denominators, paired F1/EM and terminal counts; scorer constructed once per dataset per report",
        "cost": "Same endpoint/model/rates and shared gate; NoCapBudget ledger preserves all charged usage and unknown reservations; no fixed cap or LLM-call metric",
        "gold_boundary": "Separate local references only join at reporting; no historical answer or final-score field enters model messages",
        "public_provenance": "Challenge27 public source proof byte-copied; root independently revalidated before execution approval",
        "main_evaluation_remains_paused": True, "new_API_requests_during_preparation": 0,
        "pins": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in pins}}
    save(HERE / "PROTOCOL.frozen.json", protocol)
    return dry_run()


def check():
    protocol = read(HERE / "PROTOCOL.frozen.json")
    for name, expected in protocol["pins"].items():
        assert sha(ROOT / name) == expected, name
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def settings():
    config = read(PACKAGE / "config.json")
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    return config, read(HERE / "FINAL_EXAMPLES.frozen.json")


def terminal_reason(status, result):
    """Returned algorithm failure distinct from infrastructure/accounting failure."""
    if status != "success":
        return None
    if result.get("finish_reason") == "length":
        return "length"
    if result.get("refusal") or result.get("finish_reason") == "content_filter":
        return "refusal"
    if result.get("finish_reason") == "stop" and not (result.get("answer") or "").strip():
        return "empty"
    return None


def dry_run():
    check()
    from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
    config, examples = settings()
    data = rows(HERE / "inputs27.frozen.jsonl")
    contracts = []
    for rec in data:
        assert not {"reference", "gold", "answer", "prediction", "f1", "em", "final_correct"} & set(rec)
        original = templates.base.goal_prompt(rec, "G2")
        controls = []
        for arm in templates.ARMS:
            prompt = templates.goal_prompt(rec, arm)
            if arm == "S0":
                assert prompt == original
            else:
                insertion = templates.SENTENCES[arm] + "\n"
                i = len(templates.base.G2)
                assert prompt[:i] + prompt[i + len(insertion):] == original
                assert prompt[i:i + len(insertion)] == insertion
                assert templates.SENTENCES["S2" if arm == "S1" else "S1"] not in prompt
            assert prompt.endswith(original[len(templates.base.G2):])
            assert passages(rec["goal_reader_packet"]) in prompt
            assert all(n["answer"] in prompt for n in rec["ordinary_nodes"])
            assert build_payload(config["generation"], prompt, GOAL_CAP)["max_tokens"] == 512
            final, context = templates.final_prompt(rec, "SYNTHETIC SAME CONTENT", config, examples)
            assert passages(rec["final_root_packet"]) in context
            assert build_payload(config["generation"], final, FINAL_CAP)["max_tokens"] == 64
            controls.append(final)
            contracts.append({"id": rec["id"], "arm": arm, "goal_prompt_sha256": digest(prompt),
                "S0_equals_original_G2": templates.goal_prompt(rec, "S0") == original,
                "only_one_instruction_sentence_added": arm != "S0", "final_same_content_sha256": digest(final)})
        assert len(set(controls)) == 1
    assert rows(HERE / "ORDER.frozen.jsonl") == order(data)
    assert Counter(r["arm"] for r in order(data) if r["arm_index"] == 0) == {a: 9 for a in templates.ARMS}
    assert terminal_reason("success", {"finish_reason": "length", "answer": "partial"}) == "length"
    assert terminal_reason("success", {"finish_reason": "stop", "answer": ""}) == "empty"
    assert terminal_reason("unknown", {"finish_reason": "length"}) is None
    assert terminal_reason("success", {"finish_reason": "stop", "answer": "ok"}) is None
    jsonl(OUT / "DRYRUN_CONTRACTS.jsonl", contracts)
    result = {"passed": True, "cases": 27, "conditions_checked": len(contracts),
        "one_sentence_only": True, "same_final_packaging": True, "fresh_control_no_seed": True,
        "public_source_proofs_copied": 27, "new_API_requests": 0, "main_evaluation_remains_paused": True}
    save(OUT / "DRYRUN.json", result)
    return result


def active_receipt(cache, config, prompt, cap):
    from src.p02_multigrain_timing_cachefix_v1.core.cache import canonical
    from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    raw = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?",
                           (canonical(build_payload(config["generation"], prompt, cap)),)).fetchone()
    if raw is None:
        return None
    return {"request_id": raw[0], "status": raw[1], "result": json.loads(raw[2]) if raw[2] else {}}


def execute():
    check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert any(d.get("id") == authority["decision_id"] for d in rows(PLAN / "DECISIONS.jsonl"))
    from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc
    from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import StrictAsk, NoCapBudget
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    from urllib.parse import urlsplit
    config, examples = settings()
    data = {r["id"]: r for r in rows(HERE / "inputs27.frozen.jsonl")}
    RUN.mkdir(parents=True, exist_ok=True)
    current = {}
    def event(value):
        with (RUN / "trace.jsonl").open("a", encoding="utf8") as h:
            h.write(json.dumps({"utc": utc(), **current, **value}, ensure_ascii=False) + "\n")
    class NetworkEvents:
        def event(self, name, **fields): event({"event": name, **fields})
    with run_lock(RUN), ExitStack() as stack:
        cache = stack.enter_context(Cache(RUN / "checkpoint.sqlite3"))
        cache.put("meta", "identity", {"protocol_sha256": sha(HERE / "PROTOCOL.frozen.json"), "run_id": "scope27"})
        budget = stack.enter_context(closing(NoCapBudget(config, RUN)))
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        stack.enter_context(provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()))
        live = stack.enter_context(closing(StrictAsk(cache, None, config, budget, event,
            recover_unknown=True, maximum_attempts=5, stop=lambda: (RUN / "STOP_REQUESTED").exists())))
        for item in rows(HERE / "ORDER.frozen.jsonl"):
            if (RUN / "STOP_REQUESTED").exists(): break
            rec, arm = data[item["id"]], item["arm"]
            key = arm + "|" + rec["id"]
            if cache.get("condition", key) is not None: continue
            current.clear(); current.update(id=rec["id"], dataset=rec["dataset"], arm=arm)
            stage = "goal"
            prompt, cap = templates.goal_prompt(rec, arm), GOAL_CAP
            saved = {**current}
            try:
                gid, goal, _ = live("goal/" + key, prompt, cap)
                saved.update(goal_text=goal, goal_request_id=gid)
                stage, cap = "final", FINAL_CAP
                prompt, context = templates.final_prompt(rec, goal, config, examples)
                saved.update(final_prompt=prompt, final_context=context)
                fid, prediction, _ = live("final/" + key, prompt, cap)
                saved.update(status="success", prediction=prediction, final_request_id=fid)
                cache.put("condition", key, saved)
                print(json.dumps({**current, "status": "success"}), flush=True)
            except Exception as exc:
                receipt = active_receipt(cache, config, prompt, cap)
                reason = terminal_reason(receipt["status"], receipt["result"]) if receipt else None
                if reason:
                    saved.update(status="algorithm_terminal", terminal_stage=stage, terminal_reason=reason,
                        terminal_request_id=receipt["request_id"], terminal_receipt=receipt, prediction=None)
                    cache.put("condition", key, saved)
                    event({"event": "condition_terminal", "stage": stage, "reason": reason,
                           "request_id": receipt["request_id"], "no_reroll": True})
                    print(json.dumps({**current, "status": "algorithm_terminal", "reason": reason}), flush=True)
                else:
                    event({"event": "condition_held", "stage": stage, "error_type": type(exc).__name__,
                           "error": str(exc)[:300], "receipt_id": receipt["request_id"] if receipt else None})
                    print(json.dumps({**current, "status": "held"}), flush=True)
    return report()


def report():
    """Read-only snapshot; scoring references stay local; each scorer built once."""
    check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    refs = {r["id"]: r for r in rows(HERE / "references27.local.frozen.jsonl")}
    scorers = {ds: metric(ds) for ds in DATASETS}
    with closing(sqlite3.connect((RUN / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        results = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='condition' ORDER BY rowid")]
        receipts = [{"request_id": rid, "status": status, "result": json.loads(raw) if raw else None}
                    for rid, status, raw in db.execute("SELECT id,status,result FROM requests ORDER BY rowid")]
        retired = [{"request_id": rid, "status": status, "result": json.loads(raw) if raw else None}
                   for rid, status, raw in db.execute("SELECT id,status,result FROM retired_requests ORDER BY rowid")]
    indexed_receipts = {r["request_id"]: r for r in receipts}
    for r in results:
        r["native_metrics"] = scorers[r["dataset"]](r["prediction"], refs[r["id"]]) if r["status"] == "success" else {"f1": 0.0, "em": 0.0}
        request_ids = {r[k] for k in ("goal_request_id", "final_request_id", "terminal_request_id") if r.get(k)}
        usage = [(indexed_receipts[rid].get("result") or {}).get("usage") or {} for rid in request_ids]
        r["usage_complete"] = all(all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens")) for u in usage)
        r["goal_final_input_tokens"] = sum(u.get("prompt_tokens") or 0 for u in usage)
        r["goal_final_output_tokens"] = sum(u.get("completion_tokens") or 0 for u in usage)
        r["goal_final_estimated_CNY"] = r["goal_final_input_tokens"] * 0.8e-6 + r["goal_final_output_tokens"] * 2e-6
    groups = {}
    for arm in templates.ARMS:
        groups[arm] = {}
        for ds, planned in (("2wiki", 17), ("hotpotqa", 10), ("all", 27)):
            rr = [r for r in results if r["arm"] == arm and (ds == "all" or r["dataset"] == ds)]
            success = [r for r in rr if r["status"] == "success"]
            group = {"planned": planned, "resolved": len(rr), "normal_complete": len(success),
                "algorithm_terminal": len(rr) - len(success), "held_or_pending": planned - len(rr),
                "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rr if r["status"] != "success")),
                "fixed_denominator_F1": sum(r["native_metrics"]["f1"] for r in rr) / planned if len(rr) == planned else None,
                "fixed_denominator_EM": sum(r["native_metrics"]["em"] for r in rr) / planned if len(rr) == planned else None,
                "known_outcome_F1": sum(r["native_metrics"]["f1"] for r in rr) / len(rr) if rr else None,
                "known_outcome_EM": sum(r["native_metrics"]["em"] for r in rr) / len(rr) if rr else None,
                "complete_only_F1": sum(r["native_metrics"]["f1"] for r in success) / len(success) if success else None,
                "complete_only_EM": sum(r["native_metrics"]["em"] for r in success) / len(success) if success else None}
            for key in ("goal_final_input_tokens", "goal_final_output_tokens", "goal_final_estimated_CNY"):
                group[key] = sum(r[key] for r in rr)
            groups[arm][ds] = group
    indexed = {(r["arm"], r["id"]): r for r in results}
    paired = {}
    for a, b in (("S0", "S1"), ("S0", "S2"), ("S1", "S2")):
        outcomes = [(indexed[a, ident], indexed[b, ident]) for ident in refs if (a, ident) in indexed and (b, ident) in indexed]
        summary = {"paired_resolved": len(outcomes), "planned": 27}
        for endpoint in ("f1", "em"):
            deltas = [y["native_metrics"][endpoint] - x["native_metrics"][endpoint] for x, y in outcomes]
            summary[endpoint] = {"improved": sum(d > 1e-12 for d in deltas), "harmed": sum(d < -1e-12 for d in deltas),
                "tied": sum(abs(d) <= 1e-12 for d in deltas), "mean_delta": sum(deltas) / len(deltas) if deltas else None}
        paired[a + "_to_" + b] = summary
    all_receipts = receipts + retired
    known_usage = [(r.get("result") or {}).get("usage") or {} for r in all_receipts]
    config, _ = settings()
    from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget
    namespace = "p02_prompt_research_v1_20261022/" + str(RUN.resolve())
    ledger = Path(config["budget"]["ledger"]).resolve()
    with closing(sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True)) as db:
        charges = [{"id": rid, "reserved_CNY": reserved, "actual_CNY": actual, "status": status}
            for rid, reserved, actual, status in db.execute("SELECT id,reserved,actual,status FROM charges WHERE namespace=?", (namespace,))]
    result = {"scope": "Exposed diagnostic-selected27; exploratory scope/role constraints with fresh synchronous control",
        "status": "complete" if len(results) == 81 else "partial", "by_arm_dataset": groups, "paired": paired,
        "known_receipt_CNY": sum((u.get("prompt_tokens") or 0) * 0.8e-6 + (u.get("completion_tokens") or 0) * 2e-6 for u in known_usage),
        "unresolved_receipts": [r for r in receipts if r["status"] != "success"], "retired_receipts_preserved": retired,
        "charges": charges, "unknown_or_unresolved_reserved_CNY": sum(r["reserved_CNY"] or 0 for r in charges if r["actual_CNY"] is None),
        "terminal_algorithm_outcomes_count_zero_not_missing": True, "no_independent_confirmation_claim": True,
        "historical_G2_not_imported": True, "main_evaluation_remains_paused": True}
    jsonl(RUN / "DETAILS.scored.local.jsonl", results)
    save(RUN / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--report", action="store_true")
    args = parser.parse_args()
    value = prepare() if args.prepare else dry_run() if args.dry_run else execute() if args.run else report()
    print(json.dumps(value, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
