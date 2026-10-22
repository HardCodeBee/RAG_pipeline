"""Offline preparation and root-authorized fresh synchronous fact-sentence/role probe.

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
RUN = OUT / "fact27"
DATASETS = ("2wiki", "hotpotqa")
SEED, GOAL_CAP, FINAL_CAP = 20261022, 512, 64
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022.run import read, rows, save, sha, jsonl
from experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run import receipt_integrity, algorithm_failure
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import complete_receipt
from experiments.query_decomposition.p02_goal_fact_role_probe_v1_20261022 import templates


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def order(data):
    offset = int(digest(str(SEED))[:8], 16) % 2
    result = []
    for i, record in enumerate(data):
        rotation = (offset + i) % 2
        arms = templates.ARMS[rotation:] + templates.ARMS[:rotation]
        result.extend({"case_index": i, "arm_index": j, "id": record["id"], "arm": arm}
                      for j, arm in enumerate(arms))
    return result


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve this frozen experiment")
    copied = ("inputs27.frozen.jsonl", "references27.local.frozen.jsonl", "SELECTION.frozen.jsonl",
              "PUBLIC_SOURCE_PROOF.jsonl", "PUBLIC_BLOB_METADATA.json", "GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json")
    for name in copied:
        (HERE / name).write_bytes((CHALLENGE / name).read_bytes())
    (HERE / "ROLE.frozen.txt").write_text(templates.ROLE + "\n", encoding="utf8")
    data = rows(HERE / "inputs27.frozen.jsonl")
    assert len(data) == 27 and Counter(r["dataset"] for r in data) == {"2wiki": 17, "hotpotqa": 10}
    assert all(r["ordinary_nodes"] for r in data)
    assert [r["id"] for r in rows(HERE / "SELECTION.frozen.jsonl")] == [r["id"] for r in data]
    proof = rows(HERE / "PUBLIC_SOURCE_PROOF.jsonl")
    assert len(proof) == 27 and all(r["member_bytes_exact"] and r["ordinary_QA_and_reader_from_exact_public_source"] for r in proof)
    jsonl(HERE / "ORDER.frozen.jsonl", order(data))
    pins = [HERE / n for n in ("run.py", "templates.py", "PROTOCOL.md", *copied, "ORDER.frozen.jsonl", "ROLE.frozen.txt")]
    pins += [PILOT / "templates.py", PILOT / "run.py", SHARED, PACKAGE / "config.json",
             PACKAGE / "core/cache.py", PACKAGE / "core/llm.py", PACKAGE / "core/final_prompts.py",
             CHALLENGE / "PROTOCOL.frozen.json",
             HERE.parent / "p02_goal_scope_role_probe_v1_20261022/templates.py",
             HERE.parent / "p02_ordinary_fewshot_probe_v1_20261022/run.py"]
    protocol = {"id": HERE.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "prepared_only_not_authorized", "planned": 27, "conditions": 54,
        "per_dataset": {"2wiki": 17, "hotpotqa": 10}, "arms": list(templates.ARMS),
        "goal_cap": GOAL_CAP, "final_cap": FINAL_CAP,
        "not_native128_reproduction": True,
        "selection": "Exactly all original challenge27, no new score selection; frozen actual ordinary decomposition and reader routes; not manual60",
        "T0": "Original G0 historical goal prompt byte-identical, generated afresh at common512 in independent namespace, no historical seed",
        "T1": "Only append exact S2 ROLE sentence to G0 main instruction prefix immediately before the unique Original question input header (formal G0 has no worked examples); preserve all original prompt bytes",
        "ROLE": templates.ROLE, "insertion_marker": templates.MARKER,
        "fixed": "Ordinary QA, decomposition D, goal rewrite/reader packet, final root, all existing G0 instructions and goal rule, model, final packaging; no formatter/retrieval/training/judge",
        "stage_payload_upper_bound": 108,
        "stage_upper_bound_meaning": "54 goal plus at most54 final logical payloads before authorized technical retries; complete identical final payloads may share current-run receipts",
        "previous_G2_reference": "Exposed S0/S2 only historical staged references, no cross-experiment response import; not synchronous full factorial or independent confirmation of format interaction",
        "ordering": {"seed": SEED, "rule": "Frozen cases, cyclic alternating T0/T1 arm start;14/13 first-position cases"},
        "cache": "No seed; same-experiment exact full-payload complete matching-model stop+usage reuse only, no success or algorithm rerolls",
        "recovery": "Same frozen StrictAsk/NoCapBudget/gate, SDK/HTTP0; maximum5 actual technical attempts including initial across restarts; unknown/retired receipts and liabilities preserved",
        "terminal": "Reuse frozen fewshot receipt_integrity/algorithm_failure helper: exact requested/returned model plus nonnegative integer usage required. Resolved length/empty/refusal/contentfilter remain immutable0 conditions; goal terminal skips final; missing usage/mismatch/technical unknown held and never zero",
        "report": "All arms/datasets and planned denominator; complete-only sensitivity, paired F1/EM, tokens/cost; scorer once per dataset; only read-only DB monitoring and local reference join",
        "cost": "Same endpoint/rates no fixed cap; known paid receipts and unknown reservations retained; no LLM-call-count metric",
        "public_provenance": "Byte-identical original challenge27 inputs and approved publicsource proofs; no new network or remote-public verification claimed",
        "execution_gate": "No authorization created by preparation; root decides after W90 and stage review, then separate concrete decision/authentication file required",
        "new_API_requests_during_preparation": 0, "main_evaluation_remains_paused": True,
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
    model = read(HERE / "GENERATION.frozen.json")["model"]
    reason = algorithm_failure({**result, "status": status}, model)
    return reason.removeprefix("terminal_") if reason else None


def validate_success_receipt(cache, request_id):
    row = cache.db.execute("SELECT status,result FROM requests WHERE id=?", (request_id,)).fetchone()
    assert row
    receipt = {**(json.loads(row[1]) if row[1] else {}), "status": row[0]}
    model = read(HERE / "GENERATION.frozen.json")["model"]
    if not (receipt_integrity(receipt, model) and complete_receipt(receipt)):
        raise RuntimeError("Hold incomplete or mismatched model/usage receipt")


def dry_run():
    check()
    from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
    config, examples = settings()
    data = rows(HERE / "inputs27.frozen.jsonl")
    contracts = []
    for rec in data:
        assert not {"reference", "gold", "answer", "prediction", "f1", "em", "final_correct"} & set(rec)
        original = templates.base.goal_prompt(rec, "G0")
        assert original == rec["historical_goal_prompt"]
        assert templates.goal_prompt(rec, "T0") == original
        changed = templates.goal_prompt(rec, "T1")
        added = "\n" + templates.ROLE
        i = original.index(templates.MARKER)
        assert changed[:i] + changed[i + len(added):] == original
        assert changed[i:i + len(added)] == added
        assert changed.count(templates.ROLE) == 1 and templates.ROLE not in original
        assert changed.split(templates.MARKER, 1)[1] == original.split(templates.MARKER, 1)[1]
        finals = []
        for arm in templates.ARMS:
            prompt = templates.goal_prompt(rec, arm)
            assert passages(rec["goal_reader_packet"]) in prompt
            assert all(n["answer"] in prompt for n in rec["ordinary_nodes"])
            payload = build_payload(config["generation"], prompt, GOAL_CAP)
            assert payload["max_tokens"] == 512
            final, context = templates.final_prompt(rec, "SYNTHETIC SAME CONTENT", config, examples)
            assert passages(rec["final_root_packet"]) in context
            assert build_payload(config["generation"], final, FINAL_CAP)["max_tokens"] == 64
            finals.append(final)
            contracts.append({"id": rec["id"], "arm": arm, "goal_prompt_sha256": digest(prompt),
                "T0_equals_original_G0": True, "T1_exact_ROLE_only": True,
                "goal_cap_common512_not_native128": True, "final_same_content_sha256": digest(final)})
        assert len(set(finals)) == 1
    assert rows(HERE / "ORDER.frozen.jsonl") == order(data)
    starts = Counter(r["arm"] for r in order(data) if r["arm_index"] == 0)
    assert sorted(starts.values()) == [13, 14]
    model = config["generation"]["model"]
    valid = {"status": "success", "requested_model": model, "returned_model": model,
             "usage": {"prompt_tokens": 100, "completion_tokens": 64}, "answer": "partial", "finish_reason": "length"}
    assert terminal_reason("success", valid) == "length"
    assert terminal_reason("success", {**valid, "finish_reason": "stop", "answer": ""}) == "empty"
    assert terminal_reason("success", {**valid, "finish_reason": "stop", "refusal": "declined"}) == "refusal"
    assert terminal_reason("success", {**valid, "finish_reason": "content_filter"}) == "content_filter"
    for invalid in ({**valid, "usage": None}, {**valid, "returned_model": "wrong-model"},
                    {**valid, "usage": {"prompt_tokens": True, "completion_tokens": 64}},
                    {**valid, "usage": {"prompt_tokens": 100, "completion_tokens": -1}}):
        assert terminal_reason("success", invalid) is None
    assert terminal_reason("unknown", valid) is None
    jsonl(OUT / "DRYRUN_CONTRACTS.jsonl", contracts)
    result = {"passed": True, "cases": 27, "conditions_checked": len(contracts), "goal_cap": 512, "final_cap": 64,
        "one_ROLE_sentence_only": True, "same_final_packaging": True, "fresh_T0_no_seed": True,
        "strict_terminal_integrity_contracts": True, "first_arm_counts": dict(starts),
        "public_source_proofs_copied": 27, "not_native128_reproduction": True,
        "new_API_requests": 0, "execution_authorized": False, "main_evaluation_remains_paused": True}
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
    assert authority.get("protocol_sha256") == sha(HERE / "PROTOCOL.frozen.json")
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
        cache.put("meta", "identity", {"protocol_sha256": sha(HERE / "PROTOCOL.frozen.json"), "run_id": "fact27"})
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
                validate_success_receipt(cache, gid)
                saved.update(goal_text=goal, goal_request_id=gid)
                stage, cap = "final", FINAL_CAP
                prompt, context = templates.final_prompt(rec, goal, config, examples)
                saved.update(final_prompt=prompt, final_context=context)
                fid, prediction, _ = live("final/" + key, prompt, cap)
                validate_success_receipt(cache, fid)
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
    if not (RUN / "checkpoint.sqlite3").exists():
        return {"status": "prepared_not_started", "planned": 54, "resolved": 0, "new_API_requests": 0, "main_evaluation_remains_paused": True}
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
    for a, b in (("T0", "T1"),):
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
    result = {"scope": "Exposed diagnostic-selected27; original G0 fact sentence with/without ROLE, common512, fresh synchronous T0; not native128 or full concurrent factorial",
        "status": "complete" if len(results) == 54 else "partial", "by_arm_dataset": groups, "paired": paired,
        "known_receipt_CNY": sum((u.get("prompt_tokens") or 0) * 0.8e-6 + (u.get("completion_tokens") or 0) * 2e-6 for u in known_usage),
        "unresolved_receipts": [r for r in receipts if r["status"] != "success"], "retired_receipts_preserved": retired,
        "charges": charges, "unknown_or_unresolved_reserved_CNY": sum(r["reserved_CNY"] or 0 for r in charges if r["actual_CNY"] is None),
        "terminal_algorithm_outcomes_count_zero_not_missing": True, "no_independent_confirmation_claim": True,
        "historical_G0_or_G2_not_imported": True, "main_evaluation_remains_paused": True}
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
