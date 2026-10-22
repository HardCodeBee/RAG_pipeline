"""Isolated full-RAG factor90 controller; preparation is offline, run needs root AUTH."""
from collections import Counter
from contextlib import closing
from copy import deepcopy
import argparse
import json
import sqlite3
import time
from urllib.parse import urlsplit
from .preparation import HERE, ROOT, OUT, PLAN, PRIOR, STAGE, PROTOCOL, INPUTS, ORDER, HISTORY
from .preparation import base, read, rows, sha, rel, jsonl, prepare, check, freeze, templates, atomic_json
from . import flow
from p02_multigrain_timing_cachefix_v1.core.cache import utc
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, passages, render_final
from p02_multigrain_timing_cachefix_v1.core import child_prompts
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import (
    NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld, stage_success, terminal_reason)
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.message_transport import build_payload
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.retrieval_bridge import (
    identity as retrieval_identity, key as retrieval_key, check_packet)
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget

PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
ro, digest, cost = base.ro, base.digest, base.cost


def held_rows():
    return [read(path) for path in (OUT / "held").glob("*.json")]


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(dict(utc=utc(), **value), ensure_ascii=False) + "\n")


def stage_completion_status():
    folder = ROOT / "outputs/query_decomposition" / STAGE.name
    if not (folder / "checkpoint.sqlite3").is_file():
        return dict(finished=False, reason="stage_checkpoint_missing")
    try:
        expected = dict(protocol_sha256=sha(STAGE / "PROTOCOL.frozen.json"),
            inputs_sha256=sha(STAGE / "INPUTS.frozen.jsonl"), order_sha256=sha(STAGE / "TASK_ORDER.frozen.jsonl"), historical_seed=None)
        tasks = rows(STAGE / "TASK_ORDER.frozen.jsonl"); planned = {r["task_id"] for r in tasks}
        if len(tasks) != 152 or len(planned) != 152:
            return dict(finished=False, reason="stage_order_not_exact152")
        with closing(ro(folder / "checkpoint.sqlite3")) as db:
            db.execute("BEGIN")
            item = db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
            identity = json.loads(item[0]) if item else {}
            done = {r[0] for r in db.execute("SELECT key FROM data WHERE kind='outcome'")}
            inflight = db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
        held = {read(path)["task_id"] for path in (folder / "held").glob("*.json")}
        if identity != expected or not done <= planned or not held <= planned:
            return dict(finished=False, reason="stage_identity_or_tasks_mismatch")
        return dict(finished=done | held == planned and inflight == 0, planned=152,
                    attempted=len(done | held), in_flight=inflight, protocol_sha256=expected["protocol_sha256"])
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
        return dict(finished=False, reason="invalid_stage_metadata")


def predecessor_status():
    pins = read(HERE / "PREDECESSORS.frozen.json")
    for label, folder in (("context90", PRIOR), ("stage_probe", STAGE)):
        if any(sha(folder / name) != expected for name, expected in pins[label].items()):
            return dict(finished=False, reason="predecessor_frozen_identity_mismatch")
    from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022.run import context90_completion_status
    context = context90_completion_status(); stage = stage_completion_status()
    return dict(finished=context["finished"] and stage["finished"], context90=context, stage_probe=stage)


def await_predecessors(wait):
    """No writable Cache, provider key, ledger, SSH or worker exists during wait."""
    while True:
        check(); status = predecessor_status()
        atomic_json(OUT / "PREDECESSOR_WAIT_STATUS.json", dict(utc=utc(), protocol_sha256=sha(PROTOCOL),
                    API_SSH_GPU_requests=0, **status))
        if status["finished"]:
            return True
        if not wait or (OUT / "STOP_REQUESTED").exists():
            return False
        time.sleep(30)


def authorized():
    check(); auth = read(HERE / "RUN_AUTHORIZATION.json")
    assert auth.get("approved") is True and auth.get("experiment") == HERE.name
    assert auth.get("protocol_sha256") == sha(PROTOCOL)
    assert auth.get("current_namespace_returnless_recovery_only") is True
    assert auth.get("maximum_technical_attempts") == 5
    assert any(r.get("id") == auth.get("decision_id") for r in rows(PLAN / "DECISIONS.jsonl"))
    return auth


def engineering_gate(cache, tasks, config):
    """Validate first24 wire/receipt/ordinary-only invariants without native scores."""
    selected = [t for t in tasks if t["engineering_initial"]]; outcomes = cache.all("outcome")
    missing = [t["task_id"] for t in selected if t["task_id"] not in outcomes]
    if missing:
        return dict(passed=False, status="engineering_technical_review_required", missing_task_ids=missing,
                    quality_or_predictions_inspected=False)
    pipeline = Pipeline(config); inputs = {r["id"]: r for r in rows(INPUTS)}; checked = nodes_checked = 0
    try:
        for task in selected:
            outcome = outcomes[task["task_id"]]; case = inputs[task["id"]]; stage_payloads = {}; terminals = []
            for stage in outcome["stages"]:
                row = cache.db.execute("SELECT payload,status,result FROM requests WHERE id=?", (stage["request_id"],)).fetchone()
                assert row is not None
                payload = json.loads(row[0]); receipt = {**json.loads(row[2]), "status": row[1]}
                assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
                assert payload == build_payload(config["generation"], payload["messages"][0]["content"], payload["max_tokens"])
                assert digest(payload) == stage["payload_sha256"]
                success = stage_success(receipt, config["generation"]["model"], stage["stage"])
                reason = None if success else terminal_reason(receipt, config["generation"]["model"])
                assert success or reason
                if reason: terminals.append((stage["stage"], reason))
                stage_payloads[stage["stage"]] = payload; checked += 1
            if outcome["status"] == "algorithm_failure":
                assert terminals == [(outcome["terminal_stage"], outcome["terminal_reason"])]
            else:
                assert not terminals
            trajectory = outcome.get("trajectory") or {}; nodes = trajectory.get("nodes", outcome.get("partial_nodes", []))
            history = []
            for node in nodes:
                sid = node["step_id"]; ordinary = sid <= len(case["supporting_questions"])
                question = (case["supporting_questions"] + [case["question"]])[sid - 1]
                raw = child_prompts.child(case["question"], sid, question, history, passages(node["reader_packet"]),
                    pipeline.child_examples, role="ordinary" if ordinary else "goal")
                if not ordinary: raw = raw[:-len("Answer:")] + child_prompts.GOAL_RULE + "\n\nAnswer:"
                assert node["raw_P02_child_prompt"] == raw and node["prior_complete_answers"] == history
                item = dict(role="ordinary", step_id=sid, original_question=case["question"], current_question=question,
                    history=history, control_prompt=raw, accepted_rewrite=node["rewritten_query"])
                actual = templates.prompt(item, task["arm"]) if ordinary else raw
                assert node["child_prompt"] == actual
                assert stage_payloads[f"legacy/answer/{sid}"] == build_payload(config["generation"], actual, 128)
                if node["rewrite_request_id"]:
                    assert stage_payloads[f"legacy/rewrite/{sid}"] == build_payload(config["generation"], node["rewrite_prompt"], 256)
                if node["sentence_format_request_id"]:
                    item["reader_packet"] = node["reader_packet"]
                    assert stage_payloads[f"legacy/sentence_format/{sid}"] == build_payload(config["generation"], templates.formatter_prompt(item, node["initial_answer"]), 128)
                history.append(dict(step_id=sid, request=question, answer=node["answer"])); nodes_checked += 1
            if trajectory:
                final, _, _ = render_final(case["question"], nodes, trajectory["original_retrieval"], pipeline.readout, pipeline.final_examples)
                assert stage_payloads["legacy/final"] == build_payload(config["generation"], final, 64)
        return dict(passed=True, status="engineering_contracts_passed_continue_fixed84", planned_conditions=24,
                    stages_checked=checked, nodes_checked=nodes_checked, quality_or_predictions_inspected=False)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return dict(passed=False, status="engineering_contract_failure_review_required", error_type=type(exc).__name__,
                    stages_checked=checked, nodes_checked=nodes_checked, quality_or_predictions_inspected=False)


def check_expense(cache, budget, historical, task):
    paid, unattributed = base.paid_query_attribution(cache.db, budget.db, rows(ORDER), list(cache.all("outcome").values()),
        historical, "p02_prompt_research_v1_20261022/" + str(OUT.resolve()))
    current = next(r for r in paid if r["task_id"] == task["task_id"])
    reasons = base.expense_reasons(current, paid)
    if reasons: event(dict(event="expense_alert", task_id=task["task_id"], reasons=reasons, paid_query=current, unattributed_charges=unattributed))


def run_with_bridge(bridge):
    authorized(); assert predecessor_status()["finished"]
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config = read(PACKAGE / "config.json"); data = {r["id"]: r for r in rows(INPUTS)}
    resources = read(HERE / "RESOURCE_CONTRACTS.json"); historical = {r["id"]: r for r in rows(HISTORY)}
    class NetworkEvents:
        def event(self, name, **fields): event(dict(event=name, **fields))
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config, OUT)) as budget, \
         temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
         provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
        expected = dict(protocol_sha256=sha(PROTOCOL), inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER),
                        resources=resources, no_historical_LLM_seed=True)
        previous = cache.get("meta", "identity")
        if previous is None:
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0 and not cache.all("outcome")
            cache.put("meta", "identity", expected)
        else: assert previous == expected
        stop = lambda: (OUT / "STOP_REQUESTED").exists()
        pipeline = Pipeline(config)
        with closing(SubmissionAwareStrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as strict:
            tasks = rows(ORDER); engineering_passed = False
            for task in tasks:
                if cache.get("outcome", task["task_id"]) is not None or task["task_id"] in {r["task_id"] for r in held_rows()}: continue
                if stop(): break
                allocation = bridge.can_start_condition()
                if not allocation["allowed"]:
                    event(dict(event="allocation_dispatch_guard_no_new_condition", next_task_id=task["task_id"], **allocation))
                    break
                if not task["engineering_initial"] and not engineering_passed:
                    gate = engineering_gate(cache, tasks, config); atomic_json(OUT / "ENGINEERING_GATE.json", gate)
                    if not gate["passed"]:
                        event(dict(event="engineering_gate_paused", **gate)); break
                    engineering_passed = True
                case = data[task["id"]]; remote = bridge.for_dataset(case["dataset"])
                def retrieve(query):
                    request = retrieval_identity(resources[case["dataset"]]["identity"], case["dataset"], case["qid"], query, 10)
                    key = retrieval_key(request); packet = cache.get("retrieval", key); hit = packet is not None
                    if not hit:
                        packet = check_packet(request, remote.retrieve(case["qid"], query, 10)); cache.put("retrieval", key, packet)
                    else: packet = check_packet(request, packet)
                    event(dict(event="logical_retrieval", task_id=task["task_id"], request_id=key,
                               source="current_run_exact" if hit else "remote_search", **request))
                    return deepcopy(packet)
                native = NativeAsk(cache, strict, config, task["task_id"], event=event); partial = []
                def record(node):
                    cache.put("node", task["task_id"] + "/" + str(node["step_id"]), node); partial.append(deepcopy(node))
                try:
                    result = flow.execute_arm(case, retrieve, native, task["arm"], pipeline=pipeline, record_node=record,
                                              record_event=lambda value: event(dict(task_id=task["task_id"], **value)))
                    outcome = dict(**task, status=result["status"], prediction=result["prediction"], trajectory=result,
                                   stages=native.stages, utc=utc(), terminal_reason=None)
                except AlgorithmTerminal as exc:
                    outcome = dict(**task, status="algorithm_failure", prediction="", partial_nodes=partial,
                        stages=native.stages, terminal_reason=exc.value["terminal_reason"], terminal_stage=exc.value["stage"], utc=utc())
                except (TechnicalHeld, RuntimeError, ValueError, TypeError, InterruptedError) as exc:
                    held = dict(**task, status="held_technical", error_type=type(exc).__name__, utc=utc(),
                        completed_stage_request_ids=[r["request_id"] for r in native.stages])
                    atomic_json(OUT / "held" / (digest(task["task_id"]) + ".json"), held)
                    event(dict(event="held_technical", **held)); check_expense(cache, budget, historical, task); continue
                cache.put("outcome", task["task_id"], outcome); check_expense(cache, budget, historical, task)
                terminal = dict(event="condition_terminal", task_id=task["task_id"], status=outcome["status"])
                event(terminal); print(json.dumps(terminal), flush=True)
    return report()


def report():
    check(); tasks = rows(ORDER); found = []; states = {}
    if (OUT / "checkpoint.sqlite3").exists():
        with closing(ro(OUT / "checkpoint.sqlite3")) as db:
            found = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
            states = dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
    completed = {r["task_id"] for r in found}; held = {r["task_id"] for r in held_rows()} - completed
    planned = {r["task_id"] for r in tasks}; assert completed | held <= planned
    value = dict(utc=utc(), status="all_conditions_attempted" if completed | held == planned else "partial" if found or held else "prepared_not_started",
        planned_conditions=360, quality_terminal=len(found), held_technical=len(held), not_started=360-len(completed | held),
        request_statuses=states, arms={arm: dict(planned=90, quality_terminal=sum(r["arm"] == arm for r in found),
            logical_prompt_tokens=sum(s["usage"]["prompt_tokens"] for r in found if r["arm"] == arm for s in r["stages"]),
            logical_completion_tokens=sum(s["usage"]["completion_tokens"] for r in found if r["arm"] == arm for s in r["stages"])) for arm in templates.ARMS},
        engineering_gate=read(OUT / "ENGINEERING_GATE.json") if (OUT / "ENGINEERING_GATE.json").exists() else None,
        native_quality_not_computed=True, main_evaluation_remains_paused=True)
    if (OUT / "checkpoint.sqlite3").exists():
        with closing(ro(read(PACKAGE / "config.json")["budget"]["ledger"])) as ledger:
            actual, reserve = ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",
                ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),)).fetchone()
        value.update(actual_new_returned_usage_CNY=actual, unresolved_reserved_CNY=reserve)
    atomic_json(OUT / "STATUS.json", value); return value


def run(wait=False):
    auth = authorized(); status = report()
    if status["not_started"] == 0: return status
    if not await_predecessors(wait): return dict(state="waiting_for_predecessors", **predecessor_status())
    if (OUT / "STOP_REQUESTED").exists(): return dict(state="local_stop_preserved")
    from .remote_bridge import open_bridge
    with open_bridge(resources=read(HERE / "RESOURCE_CONTRACTS.json"), output=OUT, authorization=auth) as bridge:
        return run_with_bridge(bridge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "dry-run", "report", "predecessors", "run"))
    parser.add_argument("--wait-for-predecessors", action="store_true")
    args = parser.parse_args()
    if args.wait_for_predecessors and args.action != "run": parser.error("--wait-for-predecessors is run-only")
    value = run(args.wait_for_predecessors) if args.action == "run" else {
        "prepare": prepare, "freeze": freeze, "dry-run": check, "report": report, "predecessors": predecessor_status}[args.action]()
    print(json.dumps(value, ensure_ascii=True), flush=True)


if __name__ == "__main__": main()
