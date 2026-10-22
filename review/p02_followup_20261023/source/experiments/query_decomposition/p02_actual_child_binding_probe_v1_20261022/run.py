"""Prepare frozen15 actual ordinary nodes; only root-authorized run calls API.

R0 must reuse the contemporaneous fewshot E0 receipts. Missing E0 never causes a
new request, replacement node or fallback to a convenient historical answer.
"""
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, atomic_json, complete_receipt, INPUT_RATE, OUTPUT_RATE,
)
from experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run import (
    algorithm_failure, receipt_integrity,
)
from experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022 import templates

FEWSHOT = HERE.parent / "p02_ordinary_fewshot_probe_v1_20261022"
E0_OUT = ROOT / "outputs/query_decomposition" / FEWSHOT.name
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
INPUTS = HERE / "INPUTS15.frozen.jsonl"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
CONTROLS = HERE / "E0_SOURCE15.frozen.json"
ORDER = HERE / "TASK_ORDER.frozen.jsonl"
SEED, CAP = 20261022, 128


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf8")


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def cost(result):
    usage = result.get("usage") or {}
    return (usage.get("prompt_tokens") or 0) * INPUT_RATE + (usage.get("completion_tokens") or 0) * OUTPUT_RATE


def arm_order(case_id):
    offset = int(hashlib.sha256(f"{SEED}|{case_id}|actual_child_binding_order".encode()).hexdigest(), 16) % 2
    return ("R1", "R2")[offset:] + ("R1", "R2")[:offset]


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    original = [r for r in rows(FEWSHOT / "INPUTS.frozen.jsonl") if r["group"] == "with_history"]
    assert len(original) == 15
    items = []
    for row in original:
        assert sha(ROOT / row["source"]) == row["source_sha256"]
        source = read(ROOT / row["source"])
        node = next(n for n in source["nodes"] if n["step_id"] == row["step_id"])
        assert node["role"] == row["role"] == "ordinary" and row["history"]
        assert node["child_prompt"] == row["control_prompt"]
        assert node["reader_packet"] == row["reader_packet"]
        assert node["request"] == row["current_question"]
        assert isinstance(node["rewritten_query"], str) and node["rewritten_query"].strip()
        item = {k: row[k] for k in ("probe_id", "case_id", "qid", "dataset", "role", "step_id", "source", "source_sha256",
                "original_question", "current_question", "history", "reader_packet", "control_prompt", "control_payload",
                "control_payload_sha256", "control_source_request_id")}
        item.update(frozen_rewritten_query=node["rewritten_query"],
            interface_group="direct_named_control" if node["rewritten_query"] == row["current_question"] else "descriptive_binding_candidate",
            original_supporting_questions=[n["request"] for n in source["nodes"] if n["role"] == "ordinary"],
            decomposition_source_prefix=source.get("source_prefix"), parent_source=source.get("parent_source"))
        items.append(item)
    assert Counter(r["interface_group"] for r in items) == {"direct_named_control": 9, "descriptive_binding_candidate": 6}
    jsonl(INPUTS, items)
    proof = read(FEWSHOT / "PUBLIC_PAYLOAD_PROVENANCE.json")
    ids = {r["probe_id"] for r in items}
    proof["cases"] = [r for r in proof["cases"] if r["probe_id"] in ids]
    proof.update(verified_nodes=15, candidate_changes="Only Actual task candidate block, filled by identical original current question or existing public frozen rewritten_query",
        verification="Reused live-verified public manifest identity through frozen fewshot proof;15 source hashes and reader/history/current/prompt plus rewritten query rechecked offline", new_provider_requests=0)
    atomic_json(HERE / "PUBLIC_PAYLOAD_PROVENANCE.json", proof)
    jsonl(ORDER, [{"index": i * 2 + j, "task_id": r["probe_id"] + "/" + arm,
           "probe_id": r["probe_id"], "arm": arm} for i, r in enumerate(items) for j, arm in enumerate(arm_order(r["case_id"]))])
    config = read(PACKAGE / "config.json")
    (HERE / "GENERATION.frozen.json").write_text(json.dumps(config["generation"], indent=2) + "\n", encoding="utf8")
    contracts = [dict(probe_id=r["probe_id"], arm=arm, payload_sha256=digest(build_payload(config["generation"], templates.prompt(r, arm), CAP)),
                      interface_group=r["interface_group"]) for r in items for arm in templates.ARMS]
    jsonl(HERE / "CONTRACTS.frozen.jsonl", contracts)
    paths = [HERE / n for n in ("run.py", "templates.py", "__init__.py", "PROTOCOL.md", "INPUTS15.frozen.jsonl",
         "TASK_ORDER.frozen.jsonl", "PUBLIC_PAYLOAD_PROVENANCE.json", "GENERATION.frozen.json", "CONTRACTS.frozen.jsonl")]
    paths += [FEWSHOT / "run.py", FEWSHOT / "PROTOCOL.frozen.json", FEWSHOT / "INPUTS.frozen.jsonl",
         FEWSHOT / "PUBLIC_PAYLOAD_PROVENANCE.json", PACKAGE / "config.json", PACKAGE / "core/cache.py", PACKAGE / "core/llm.py",
         HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py"]
    protocol = dict(id=HERE.name, prepared_utc=utc(), state="prepared_inputs_only_control_receipts_pending", nodes=15,
        arms=list(templates.ARMS), planned_conditions=45, groups={"direct_named_control": 9, "descriptive_binding_candidate": 6},
        sample="All15 existing with-history ordinary nodes from frozen actual P02 source traces, same order; no manual60, no gold/native-score filtering, no replacement or changed decomposition/training",
        interventions={"R0": "Byte-identical original prompt; ONLY reuse these15 contemporaneous fewshot E0 receipts, never submit R0",
           "R1": "Append fixed candidate block after actual current subquestion; candidate repeats original current question verbatim",
           "R2": "Same block and site as R1; only candidate text becomes existing frozen rewritten_query; original current question remains authoritative"},
        exact_block_header=templates.HEADER, exact_caution=templates.CAUTION,
        fixed="Actual decomposition, original/current questions, history, reader/order, four worked examples and other instructions, same model/settings,128 cap, no formatter; no goal entity replacement",
        important_limits="Six rewrite-different nodes are not six successful entity resolutions: FortHill only reorders its description. Nine direct controls have identical R1/R2 payloads and therefore same-run exact reuse, not independent repeated model draws.",
        incremental_distinct_payloads=21, new_condition_limit=30,
        control_freeze="Root must run freeze-e0 after all15 E0 outcomes resolve with complete model/usage receipts. Missing or held E0 blocks entire launch; no fallback. Preserve terminal E0 outcomes rather than replace/reroll. Source protocol, outcome, actual payload and request ID are frozen per node; source DB read-only transaction.",
        costs="R0 receipts charged by source E0 experiment; separate attributed reuse cost from new actual charges. New R1/R2 exact duplicates can share receipts. NoCapBudget preserves unknown/retired liabilities and shared gate.",
        recovery="Frozen shared StrictAsk, SDK/HTTP0; max5 actual technical attempts including initial across restarts. No output rerolls; algorithm_failure and receipt_integrity reuse frozen fewshot helper. Missing/invalid usage or model mismatch stay held even when length/empty/refusal observed.",
        output="Coverage, terminal outcomes, paired answer changes and tokens/cost only; no component accuracy substitutes for complete-RAG F1. Later separate blinded semantic review, then approved frozen full-pipeline validation.",
        new_provider_submissions_at_prepare=0, main_evaluation_remains_paused=True,
        pins={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths})
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def dry_run():
    check()
    items = rows(INPUTS)
    config = read(PACKAGE / "config.json")
    contracts = []
    for item in items:
        assert sha(ROOT / item["source"]) == item["source_sha256"]
        source = read(ROOT / item["source"])
        node = next(n for n in source["nodes"] if n["step_id"] == item["step_id"])
        assert node["request"] == item["current_question"] and node["rewritten_query"] == item["frozen_rewritten_query"]
        assert node["reader_packet"] == item["reader_packet"] and node["prior_complete_answers"] == item["history"]
        assert [n["request"] for n in source["nodes"] if n["role"] == "ordinary"] == item["original_supporting_questions"]
        assert source.get("source_prefix") == item["decomposition_source_prefix"]
        assert templates.prompt(item, "R0") == item["control_prompt"]
        assert build_payload(config["generation"], templates.prompt(item, "R0"), CAP) == item["control_payload"]
        for arm in templates.ARMS:
            prompt = templates.prompt(item, arm)
            old_prefix, old_actual = item["control_prompt"].split("\n\nActual task:\n", 1)
            new_prefix, new_actual = prompt.split("\n\nActual task:\n", 1)
            assert old_prefix == new_prefix
            if arm != "R0":
                candidate = item["current_question"] if arm == "R1" else item["frozen_rewritten_query"]
                added = "\n\n" + templates.block(candidate)
                assert new_actual.count(added) == 1
                assert new_actual.replace(added, "", 1) == old_actual
                assert f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\n' in new_actual
            payload = build_payload(config["generation"], prompt, CAP)
            assert {k: v for k, v in payload.items() if k != "messages"} == {k: v for k, v in item["control_payload"].items() if k != "messages"}
            contracts.append(dict(probe_id=item["probe_id"], arm=arm, payload_sha256=digest(payload), interface_group=item["interface_group"]))
    assert contracts == rows(HERE / "CONTRACTS.frozen.jsonl")
    candidates = [r for r in contracts if r["arm"] != "R0"]
    assert len(candidates) == 30 and len({r["payload_sha256"] for r in candidates}) == 21
    assert Counter(r["dataset"] for r in items) == {ds: 5 for ds in ("hotpotqa", "2wiki", "musique")}
    assert len(read(HERE / "PUBLIC_PAYLOAD_PROVENANCE.json")["cases"]) == 15
    result = dict(passed=True, nodes=15, logical_conditions=45, R0_requires_external_E0_freeze=True,
        new_candidate_conditions=30, distinct_new_payloads=21, no_original_current_question_replacement=True,
        exact_append_only_contracts=45, reader_decomposition_examples_other_instructions_unchanged=True,
        new_API_requests=0, E0_source_receipts_frozen=CONTROLS.exists(), main_evaluation_remains_paused=True)
    atomic_json(OUT / "DRYRUN.json", result)
    return result


def freeze_e0():
    """Root-invoked local snapshot only; never substitutes a missing control."""
    check()
    if CONTROLS.exists():
        return controls_check()
    config = read(PACKAGE / "config.json")
    items = rows(INPUTS)
    snapshots, missing = [], []
    path = E0_OUT / "checkpoint.sqlite3"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        for item in items:
            key = item["probe_id"] + "/E0"
            row = db.execute("SELECT value FROM data WHERE kind='outcome' AND key=?", (key,)).fetchone()
            if row is None:
                missing.append(key); continue
            outcome = json.loads(row[0])
            raw = db.execute("SELECT id,payload,status,result FROM requests WHERE id=?", (outcome["request_id"],)).fetchone()
            assert raw, key
            rid, payload, status, result_json = raw
            result = {**json.loads(result_json), "status": status}
            assert json.loads(payload) == item["control_payload"] and digest(json.loads(payload)) == item["control_payload_sha256"]
            assert receipt_integrity(result, config["generation"]["model"]), key
            if outcome["status"] == "success":
                assert complete_receipt(result) and outcome["answer"] == result["answer"].strip()
            else:
                assert outcome["status"] == "algorithm_failure" and algorithm_failure(result, config["generation"]["model"]) == outcome["reason"]
            assert outcome["usage"] == result["usage"] and outcome["finish_reason"] == result["finish_reason"]
            assert outcome["group"] == "with_history" and outcome["arm"] == "E0"
            snapshots.append(dict(probe_id=item["probe_id"], source_task_id=key, source_request_id=rid,
                source_database=str(path.relative_to(ROOT)).replace("\\", "/"), payload=json.loads(payload),
                payload_sha256=item["control_payload_sha256"], outcome=outcome, result=result,
                outcome_sha256=digest(outcome), receipt_sha256=digest(result), source_charge_already_recorded=True))
    if missing:
        raise RuntimeError(f"E0 source incomplete: {len(missing)} missing controls; no fallback or provider requests")
    atomic_json(CONTROLS, dict(experiment=HERE.name, utc=utc(), source_experiment=FEWSHOT.name,
        source_protocol_sha256=sha(FEWSHOT / "PROTOCOL.frozen.json"), source_inputs_sha256=sha(FEWSHOT / "INPUTS.frozen.jsonl"),
        controls=snapshots, count=15, new_provider_requests=0, no_R0_resubmission=True))
    return controls_check()


def controls_check():
    check()
    frozen = read(CONTROLS)
    assert frozen["count"] == len(frozen["controls"]) == 15
    assert frozen["source_protocol_sha256"] == sha(FEWSHOT / "PROTOCOL.frozen.json")
    assert frozen["source_inputs_sha256"] == sha(FEWSHOT / "INPUTS.frozen.jsonl")
    expected = {r["probe_id"]: r for r in rows(INPUTS)}
    assert {r["probe_id"] for r in frozen["controls"]} == set(expected)
    config = read(PACKAGE / "config.json")
    for row in frozen["controls"]:
        assert row["payload"] == expected[row["probe_id"]]["control_payload"]
        assert digest(row["payload"]) == row["payload_sha256"] == expected[row["probe_id"]]["control_payload_sha256"]
        assert digest(row["result"]) == row["receipt_sha256"] and digest(row["outcome"]) == row["outcome_sha256"]
        assert receipt_integrity(row["result"], config["generation"]["model"])
        assert complete_receipt(row["result"]) or algorithm_failure(row["result"], config["generation"]["model"])
    return dict(passed=True, controls_frozen=15, source=FEWSHOT.name, controls_sha256=sha(CONTROLS), new_provider_requests=0)


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as h:
        h.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def run(limit=None):
    controls_check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert authority.get("controls_sha256") == sha(CONTROLS), "Root authorization must approve the exact E0 snapshot"
    assert any(r.get("id") == authority["decision_id"] for r in rows(PLAN / "DECISIONS.jsonl"))
    config = read(PACKAGE / "config.json")
    items = {r["probe_id"]: r for r in rows(INPUTS)}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self, name, **fields): event({"event": name, **fields})
    with run_lock(OUT), ExitStack() as stack:
        cache = stack.enter_context(Cache(OUT / "checkpoint.sqlite3"))
        cache.put("meta", "identity", dict(protocol_sha256=sha(PROTOCOL), controls_sha256=sha(CONTROLS), inputs_sha256=sha(INPUTS)))
        for row in read(CONTROLS)["controls"]:
            item, source = items[row["probe_id"]], row["outcome"]
            base = dict(task_id=item["probe_id"] + "/R0", probe_id=item["probe_id"], case_id=item["case_id"], dataset=item["dataset"],
                step_id=item["step_id"], arm="R0", interface_group=item["interface_group"], status=source["status"],
                answer=source["answer"], request_id=row["source_request_id"], usage=source["usage"],
                origin="reused_contemporaneous_fewshot_E0", source_experiment=FEWSHOT.name,
                source_task_id=row["source_task_id"], actual_new_CNY=0, attributed_reused_CNY=cost(source),
                finish_reason=source["finish_reason"])
            if source.get("reason"): base["reason"] = source["reason"]
            cache.put("outcome", base["task_id"], base)
        budget = stack.enter_context(closing(NoCapBudget(config, OUT)))
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        stack.enter_context(provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()))
        stop = lambda: (OUT / "STOP_REQUESTED").exists()
        ask = stack.enter_context(closing(StrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)))
        controls = {r["probe_id"]: r for r in read(CONTROLS)["controls"]}
        recent = []
        processed = 0
        for task in rows(ORDER):
            if cache.get("outcome", task["task_id"]) is not None: continue
            if stop() or (limit is not None and processed >= limit): break
            processed += 1
            item, arm = items[task["probe_id"]], task["arm"]
            prompt = templates.prompt(item, arm)
            base = dict(task_id=task["task_id"], probe_id=item["probe_id"], case_id=item["case_id"], dataset=item["dataset"],
                step_id=item["step_id"], arm=arm, interface_group=item["interface_group"], origin="new_experiment_exact_cache_or_provider")
            try:
                rid, answer, finish = ask("actual_child_binding/" + task["task_id"], prompt, CAP)
                result = json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()[0])
                if not receipt_integrity(result, config["generation"]["model"]):
                    raise RuntimeError("Held incomplete usage or mismatched model receipt")
                outcome = {**base, "status": "success", "answer": answer, "request_id": rid,
                           "finish_reason": finish, "usage": result["usage"], "utc": utc()}
            except (RuntimeError, InterruptedError) as exc:
                payload = canonical(build_payload(config["generation"], prompt, CAP))
                row = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
                result = {**(json.loads(row[2]) if row and row[2] else {}), "status": row[1] if row else None}
                reason = algorithm_failure(result, config["generation"]["model"])
                if reason:
                    outcome = {**base, "status": "algorithm_failure", "answer": result.get("answer", ""),
                        "request_id": row[0], "finish_reason": result.get("finish_reason"), "refusal": result.get("refusal"),
                        "usage": result["usage"], "reason": reason, "utc": utc()}
                else:
                    atomic_json(OUT / "held" / (digest(task["task_id"]) + ".json"), {**base,
                        "status": "held_technical", "error": str(exc), "request_id": row[0] if row else None, "utc": utc()})
                    event(dict(event="held_technical", task_id=task["task_id"], error=str(exc))); continue
            cache.put("outcome", task["task_id"], outcome)
            # Operational stage expense comparison; not a complete-query mean.
            reference, actual = cost(controls[item["probe_id"]]["result"]), cost(outcome)
            recent.append((reference, actual))
            reasons = []
            if reference and actual > 3 * reference: reasons.append("component_above_3x_same_E0_stage")
            if len(recent) >= 20 and sum(x[1] for x in recent[-20:]) > 1.5 * sum(x[0] for x in recent[-20:]):
                reasons.append("recent20_component_above_1.5x_same_E0_stages")
            if reasons: event(dict(event="expense_alert", task_id=task["task_id"], reasons=reasons,
                                   source_E0_stage_CNY=reference, logical_stage_CNY=actual))
            emit(dict(task_id=task["task_id"], status=outcome["status"]))
    return report()


def report():
    check()
    path = OUT / "checkpoint.sqlite3"
    if not path.exists():
        return dict(status="prepared_not_started", planned=45, complete=0, controls_frozen=CONTROLS.exists(), new_provider_requests=0)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        outcomes = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='outcome' ORDER BY rowid")]
        receipts = [{"request_id": rid, "result": json.loads(raw) if raw else None, "status": status}
                    for rid, raw, status in db.execute("SELECT id,result,status FROM requests")]
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_requests'").fetchone():
            retired = [{"request_id": rid, "result": json.loads(raw) if raw else None, "status": status}
                    for rid, raw, status in db.execute("SELECT id,result,status FROM retired_requests")]
        else: retired = []
    indexed = {(r["probe_id"], r["arm"]): r for r in outcomes}
    sections = {}
    for group in ("all", "direct_named_control", "descriptive_binding_candidate"):
        for arm in templates.ARMS:
            rr = [r for r in outcomes if r["arm"] == arm and (group == "all" or r["interface_group"] == group)]
            sections[group + "/" + arm] = dict(planned=15 if group == "all" else 9 if group == "direct_named_control" else 6,
                resolved=len(rr), complete=sum(r["status"] == "success" for r in rr), algorithm_terminal=sum(r["status"] == "algorithm_failure" for r in rr),
                logical_input_tokens=sum(r["usage"]["prompt_tokens"] for r in rr), logical_output_tokens=sum(r["usage"]["completion_tokens"] for r in rr),
                logical_attributed_CNY=sum(cost(r) for r in rr))
    pairs = {}
    for a, b in (("R0", "R1"), ("R1", "R2"), ("R0", "R2")):
        for group in ("all", "direct_named_control", "descriptive_binding_candidate"):
            ids = [r["probe_id"] for r in rows(INPUTS) if group == "all" or r["interface_group"] == group]
            eligible = [p for p in ids if (p, a) in indexed and (p, b) in indexed]
            pairs[group + "/" + a + "_to_" + b] = dict(resolved_pairs=len(eligible),
                text_changed_ids=[p for p in eligible if indexed[p, a]["answer"] != indexed[p, b]["answer"]],
                same_actual_receipt_pairs=sum(indexed[p, a]["request_id"] == indexed[p, b]["request_id"] for p in eligible))
    config = read(PACKAGE / "config.json")
    with closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        actual, reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",
            ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),)).fetchone()
    done = {r["task_id"] for r in outcomes}
    holds = [read(p) for p in (OUT / "held").glob("*.json") if read(p)["task_id"] not in done]
    result = dict(status="complete" if len(outcomes) == 45 else "partial", planned=45, resolved=len(outcomes),
        sections=sections, pairs=pairs, actual_new_ledger_CNY=actual, unknown_reserved_CNY=reserved,
        R0_attributed_cost_already_charged_elsewhere=sum(r.get("attributed_reused_CNY", 0) for r in outcomes),
        reused_control_receipts_never_counted_in_new_actual_fees=True, held=holds, retired_receipts_preserved=retired,
        logical_conditions_are_not_independent_draws=True, component_results_not_complete_RAG_F1=True,
        main_evaluation_remains_paused=True)
    atomic_json(OUT / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "freeze-e0", "run", "report"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    value = prepare() if args.action == "prepare" else dry_run() if args.action == "dry-run" else freeze_e0() if args.action == "freeze-e0" else run(args.limit) if args.action == "run" else report()
    emit(value)


if __name__ == "__main__":
    main()
