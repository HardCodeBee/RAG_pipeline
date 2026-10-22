"""Independent full-RAG controller; no service before exact AUTH and direct predecessor closure."""
from collections import Counter
from contextlib import closing
from copy import deepcopy
import argparse
import hashlib
import json
import time
from urllib.parse import urlsplit
from .preparation import HERE, ROOT, OUT, PACKAGE, PLAN, INPUTS, ORDER, HISTORY, PROTOCOL, DATASETS
from .preparation import read, rows, sha, ro, prepare, validate, check, freeze, atomic_json
from . import flow, predecessors
from .accounting import paid_query_attribution, expense_reasons
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, passages, render_final
from p02_multigrain_timing_cachefix_v1.core import child_prompts, rewrite_prompts
from experiments.query_decomposition.p02_local_cot_stage38_v1_20261022 import templates
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022.templates import formatter_prompt, needs_formatter
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import (
    NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld, stage_success, terminal_reason)
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.message_transport import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, INPUT_RATE, OUTPUT_RATE
from .retrieval_bridge import identity as retrieval_identity, key as retrieval_key, check_packet


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def cost(usage): return usage["prompt_tokens"] * INPUT_RATE + usage["completion_tokens"] * OUTPUT_RATE
def held_rows(): return [read(p) for p in (OUT / "held").glob("*.json")]


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def expected_identity():
    return dict(protocol_sha256=sha(PROTOCOL), inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER),
        resources=read(HERE / "RESOURCE_CONTRACTS.json"), no_historical_LLM_seed=True)


def authorized():
    check(); auth = read(HERE / "RUN_AUTHORIZATION.json")
    assert auth.get("approved") is True and auth.get("experiment") == HERE.name
    assert auth.get("protocol_sha256") == sha(PROTOCOL) and auth.get("approved_conditions") == 360
    assert auth.get("current_namespace_returnless_recovery_only") is True and auth.get("maximum_technical_attempts") == 5
    decisions = [r for r in rows(PLAN / "DECISIONS.jsonl") if r.get("id") == auth.get("decision_id")]
    assert len(decisions) == 1 and decisions[0].get("approved") is True
    assert decisions[0].get("experiment") == HERE.name and decisions[0].get("protocol_sha256") == sha(PROTOCOL)
    return auth


def await_predecessors(wait):
    while True:
        authorized(); state = predecessors.status()
        atomic_json(OUT / "PREDECESSOR_WAIT_STATUS.json", dict(utc=utc(), new_API_SSH_GPU_requests=0, **state))
        if state["finished"]: return True
        if not wait or (OUT / "STOP_REQUESTED").exists(): return False
        time.sleep(30)


def engineering_gate(cache, tasks, config):
    """First24 structural contracts, with no predictions or native scores as selection criteria."""
    initial = [t for t in tasks if t["engineering_initial"]]
    available = {**{v["task_id"]: v for v in held_rows()}, **cache.all("outcome")}
    missing = [t["task_id"] for t in initial if t["task_id"] not in available]
    if missing: return dict(passed=False, status="engineering_not_all_attempted", missing_task_ids=missing, quality_not_a_gate=True)
    pipeline = Pipeline(config); inputs = {r["id"]: r for r in rows(INPUTS)}
    checked = format_failures = algorithm_failures = technical_holds = 0
    try:
        assert cache.get("meta", "identity") == expected_identity() and len(initial) == 24
        for task in initial:
            value = available[task["task_id"]]; case = inputs[task["id"]]
            assert all(value[k] == task[k] for k in task)
            trajectory = value.get("trajectory") or {}
            nodes = trajectory.get("nodes", value.get("partial_nodes", []))
            by_sid = {n["step_id"]: n for n in nodes}
            emitted = {r["step_id"]: r for r in value["candidate_outputs"]}
            stages = {s["stage"]: s for s in value["stages"]}
            native_terminals = []
            for sid, output in emitted.items():
                ordinary = sid <= len(case["supporting_questions"])
                question = (case["supporting_questions"] + [case["question"]])[sid - 1]
                history = [dict(step_id=i, request=by_sid[i]["request"], answer=by_sid[i]["answer"]) for i in range(1, sid)]
                assert output["prior_complete_answers"] == history and output["current_question"] == question
                raw = child_prompts.child(case["question"], sid, question, history, passages(output["reader_packet"]),
                    pipeline.child_examples, role="ordinary" if ordinary else "goal")
                if not ordinary: raw = raw[:-len("Answer:")] + child_prompts.GOAL_RULE + "\n\nAnswer:"
                actual = templates.prompt(dict(role="ordinary", control_prompt=raw), task["arm"]) if ordinary else raw
                assert output["raw_P02_child_prompt"] == raw and output["actual_child_prompt"] == actual
                assert output["output_cap"] == (templates.cap(task["arm"]) if ordinary else 128)
                assert output["raw_output"] == stages["legacy/answer/" + str(sid)]["answer"]
                if output["parser_status"] == "parsed":
                    parsed = templates.parse(task["arm"], output["raw_output"]) if ordinary else dict(support=None, answer=output["raw_output"].strip())
                    assert parsed == dict(support=output["support"], answer=output["parsed_answer"])
                elif output["parser_status"] == "format_failure":
                    assert value["status"] == "algorithm_failure" and value["terminal_reason"] == "invalid_support_answer_format"
                    try: templates.parse(task["arm"], output["raw_output"])
                    except templates.FormatFailure: pass
                    else: raise AssertionError("Recorded parser failure not reproducible")
                    format_failures += 1
                else: assert output.get("generation_terminal_reason")
            for stage in value["stages"]:
                record = cache.db.execute("SELECT payload,status,result FROM requests WHERE id=?", (stage["request_id"],)).fetchone()
                assert record is not None
                payload = json.loads(record[0]); receipt = {**json.loads(record[2]), "status": record[1]}
                name = stage["stage"]
                assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
                assert digest(payload) == stage["payload_sha256"]
                reason = terminal_reason(receipt, config["generation"]["model"])
                assert stage_success(receipt, config["generation"]["model"], name) or reason
                if not stage_success(receipt, config["generation"]["model"], name): native_terminals.append(name)
                if name.startswith("legacy/answer/"):
                    o = emitted[int(name.rsplit("/", 1)[1])]
                    expected = build_payload(config["generation"], o["actual_child_prompt"], o["output_cap"])
                elif name.startswith("legacy/sentence_format/"):
                    o = emitted[int(name.rsplit("/", 1)[1])]
                    assert needs_formatter(o["parsed_answer"])
                    expected = build_payload(config["generation"], formatter_prompt(dict(current_question=o["current_question"], reader_packet=o["reader_packet"]), o["parsed_answer"]), 128)
                elif name.startswith("legacy/rewrite/"):
                    sid = int(name.rsplit("/", 1)[1]); question = (case["supporting_questions"] + [case["question"]])[sid - 1]
                    history = [dict(step_id=i, request=by_sid[i]["request"], answer=by_sid[i]["answer"]) for i in range(1, sid)]
                    expected = build_payload(config["generation"], rewrite_prompts.rewrite(question, history), 256)
                else:
                    assert name == "legacy/final"
                    root_packet = trajectory.get("original_retrieval") or emitted[len(case["supporting_questions"]) + 1]["original_retrieval"]
                    final, _, _ = render_final(case["question"], nodes, root_packet, pipeline.readout, pipeline.final_examples)
                    expected = build_payload(config["generation"], final, 64)
                assert payload == expected; checked += 1
            for node in nodes:
                o = emitted[node["step_id"]]; local = node["local_cot"]
                assert node["initial_answer"] == o["parsed_answer"] == local["parsed_answer"]
                assert local["raw_output"] == o["raw_output"] and local["support"] == o["support"]
                assert local["accepted_answer"] == node["answer"] and local["support_forwarded"] is False
                formatter_stage = "legacy/sentence_format/" + str(node["step_id"])
                expected_answer = stages[formatter_stage]["answer"] if node["sentence_format_request_id"] else o["parsed_answer"]
                assert node["answer"] == expected_answer
            if value["status"] == "algorithm_failure":
                assert value["terminal_reason"] == "invalid_support_answer_format" or native_terminals
                algorithm_failures += 1
            else: assert not native_terminals
            technical_holds += value["status"] == "held_technical"
        return dict(passed=True, status="engineering_pass_continue_all90", planned_conditions=24, stages_checked=checked,
            format_failures=format_failures, algorithm_failures=algorithm_failures, technical_holds=technical_holds, quality_not_a_gate=True)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return dict(passed=False, status="engineering_contract_failure", error_type=type(exc).__name__,
            stages_checked=checked, quality_not_a_gate=True)


def check_expense(cache, budget, historical, task):
    paid, unattributed = paid_query_attribution(cache.db, budget.db, rows(ORDER), list(cache.all("outcome").values()),
        historical, "p02_prompt_research_v1_20261022/" + str(OUT.resolve()))
    current = next(r for r in paid if r["task_id"] == task["task_id"])
    reasons = expense_reasons(current, paid)
    if reasons: event(dict(event="expense_alert", task_id=task["task_id"], reasons=reasons, paid_query=current, unattributed_charges=unattributed))


class RetrievalHeld(RuntimeError): pass


def perform(task, case, retrieve, native, cache, pipeline, partial, candidates):
    """One condition; genuine implementation errors propagate to the controller."""
    def record_node(node):
        cache.put("node", task["task_id"] + "/" + str(node["step_id"]), node); partial.append(deepcopy(node))
    def record_output(value):
        cache.put("candidate_output", task["task_id"] + "/" + str(value["step_id"]), value); candidates.append(deepcopy(value))
    try:
        result = flow.execute_arm(case, retrieve, native, task["arm"], pipeline=pipeline, record_node=record_node,
            record_output=record_output, record_event=lambda value: event(dict(task_id=task["task_id"], **value)))
        return dict(**task, status=result["status"], prediction=result["prediction"], trajectory=result,
            stages=native.stages, candidate_outputs=candidates, terminal_reason=None, utc=utc())
    except AlgorithmTerminal as exc:
        return dict(**task, status="algorithm_failure", prediction="", partial_nodes=partial, stages=native.stages,
            candidate_outputs=candidates, terminal_reason=exc.value["terminal_reason"], terminal_stage=exc.value["stage"], utc=utc())


def retrieve_cached(cache, remote, resources, case, query):
    request = retrieval_identity(resources[case["dataset"]]["identity"], case["dataset"], case["qid"], query, 10)
    key = retrieval_key(request); packet = cache.get("retrieval", key); hit = packet is not None
    if not hit:
        try: packet = check_packet(request, remote.retrieve(case["qid"], query, 10))
        except (RuntimeError, OSError, ValueError, KeyError) as exc: raise RetrievalHeld(type(exc).__name__) from None
        cache.put("retrieval", key, packet)
    else:
        # Corrupt cached identity is an engineering failure, never a service retry.
        packet = check_packet(request, packet)
    return deepcopy(packet), key, request, hit


def run_with_bridge(bridge):
    authorized(); assert predecessors.status()["finished"]
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config = read(PACKAGE / "config.json"); inputs = {r["id"]: r for r in rows(INPUTS)}
    resources = read(HERE / "RESOURCE_CONTRACTS.json"); historical = {r["id"]: r for r in rows(HISTORY)}
    class NetworkEvents:
        def event(self, name, **fields): event(dict(event=name, **fields))
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache:
        expected = expected_identity(); previous = cache.get("meta", "identity")
        if previous is None:
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0 and not cache.all("outcome")
            cache.put("meta", "identity", expected)
        else: assert previous == expected
        with closing(NoCapBudget(config, OUT)) as budget, temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
                provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
            stop = lambda: (OUT / "STOP_REQUESTED").exists()
            with closing(SubmissionAwareStrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as strict:
                tasks = rows(ORDER); pipeline = Pipeline(config); engineering_passed = False
                for task in tasks:
                    if cache.get("outcome", task["task_id"]) is not None or task["task_id"] in {r["task_id"] for r in held_rows()}: continue
                    if stop(): break
                    allocation = bridge.can_start_condition()
                    if not allocation["allowed"]:
                        event(dict(event="allocation_no_new_condition", next_task_id=task["task_id"], **allocation)); break
                    if not task["engineering_initial"] and not engineering_passed:
                        gate = engineering_gate(cache, tasks, config); atomic_json(OUT / "ENGINEERING_GATE.json", gate)
                        if not gate["passed"]: event(dict(event="engineering_gate_stopped", **gate)); break
                        engineering_passed = True
                    case = inputs[task["id"]]; remote = bridge.for_dataset(case["dataset"])
                    def retrieve(query):
                        packet, key, request, hit = retrieve_cached(cache, remote, resources, case, query)
                        event(dict(event="logical_retrieval", task_id=task["task_id"], request_id=key,
                            source="current_run_exact" if hit else "new_remote_search", **request))
                        return deepcopy(packet)
                    native = NativeAsk(cache, strict, config, task["task_id"], event=event)
                    partial, candidates = [], []
                    try:
                        outcome = perform(task, case, retrieve, native, cache, pipeline, partial, candidates)
                    except (TechnicalHeld, RetrievalHeld, InterruptedError) as exc:
                        held = dict(**task, status="held_technical", error_type=type(exc).__name__, partial_nodes=partial,
                            stages=native.stages, candidate_outputs=candidates, utc=utc())
                        atomic_json(OUT / "held" / (digest(task["task_id"]) + ".json"), held)
                        event(dict(event="held_technical", **held)); check_expense(cache, budget, historical, task); continue
                    cache.put("outcome", task["task_id"], outcome); check_expense(cache, budget, historical, task)
                    summary = dict(event="condition_terminal", task_id=task["task_id"], status=outcome["status"])
                    event(summary); print(json.dumps(summary), flush=True)
    return report()


def report():
    check(); planned = {t["task_id"] for t in rows(ORDER)}; found = []; states = {}
    if (OUT / "checkpoint.sqlite3").exists():
        with closing(ro(OUT / "checkpoint.sqlite3")) as db:
            found = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
            states = dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
    done = {r["task_id"] for r in found}; held = {r["task_id"] for r in held_rows()}
    assert done <= planned and held <= planned and not done.intersection(held)
    value = dict(utc=utc(), status="all_attempted" if done | held == planned else "partial" if done | held else "prepared_not_started",
        planned_conditions=360, attempted=len(done | held), terminal=len(done), held_technical=len(held), not_started=len(planned - (done | held)),
        request_statuses=states, by_arm={arm: dict(planned=90, terminal=sum(r["arm"] == arm for r in found),
            statuses=dict(Counter(r["status"] for r in found if r["arm"] == arm)),
            logical_input_tokens=sum(s["usage"]["prompt_tokens"] for r in found if r["arm"] == arm for s in r["stages"]),
            logical_output_tokens=sum(s["usage"]["completion_tokens"] for r in found if r["arm"] == arm for s in r["stages"])) for arm in flow.ARMS},
        native_quality_not_computed=True, main_evaluation_remains_paused=True,
        engineering_gate=read(OUT / "ENGINEERING_GATE.json") if (OUT / "ENGINEERING_GATE.json").exists() else None)
    if (OUT / "checkpoint.sqlite3").exists():
        with closing(ro(read(PACKAGE / "config.json")["budget"]["ledger"])) as ledger:
            actual, reserve = ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),)).fetchone()
        value.update(actual_new_returned_usage_CNY=actual, unresolved_reserved_CNY=reserve)
    atomic_json(OUT / "STATUS.json", value); return value


def run(wait=False):
    auth = authorized(); current = report()
    if current["not_started"] == 0: return current
    if not await_predecessors(wait): return dict(state="waiting_predecessors", **predecessors.status())
    if (OUT / "STOP_REQUESTED").exists(): return dict(state="local_stop_preserved")
    from .remote_bridge import open_bridge
    with open_bridge(resources=read(HERE / "RESOURCE_CONTRACTS.json"), output=OUT, authorization=auth) as bridge:
        return run_with_bridge(bridge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "check", "freeze", "report", "predecessors", "run"))
    parser.add_argument("--wait", action="store_true"); args = parser.parse_args()
    if args.wait and args.action != "run": parser.error("--wait is run-only")
    value = run(args.wait) if args.action == "run" else {"prepare": prepare, "validate": validate, "check": check,
        "freeze": freeze, "report": report, "predecessors": predecessors.status}[args.action]()
    print(json.dumps(value, ensure_ascii=True), flush=True)


if __name__ == "__main__": main()
