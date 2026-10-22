"""Bounded ordinary-stage support/operation probe. Preparation opens no services.

Only an exact root authorization can execute; the main evaluation stays paused.
No source labels, gold answers, new retrieval or downstream F1 are consumed.
"""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE)
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.message_transport import build_payload
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import (
    NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld, stage_success, terminal_reason)
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022.templates import (
    needs_formatter, formatter_prompt)
from . import templates

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
STAGE = HERE.parent / "p02_reference_binding_priority_probe_v1_20261022"
OPERATOR = HERE.parent / "p02_result_binding_operator_probe_v1_20261022"
FACTOR = HERE.parent / "p02_reference_binding_priority_fixed90_v1_20261022"
CONTEXT = HERE.parent / "p02_shared_context_fixed90_v1_20261022"
INPUTS = HERE / "INPUTS.frozen.jsonl"
ORDER = HERE / "TASK_ORDER.frozen.jsonl"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
SEED = 20261022
FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path): return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def ro(path): return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
def cost(usage): return usage["prompt_tokens"] * INPUT_RATE + usage["completion_tokens"] * OUTPUT_RATE


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf8")


def task_order(items):
    assert len(items) == 38 and len({r["probe_id"] for r in items}) == 38
    assert all(r["is_stress"] for r in items[:8]) and not any(r["is_stress"] for r in items[8:])
    tasks = []
    for item in items:
        offset = int(hashlib.sha256(f"{SEED}|{item['probe_id']}|local_cot_order".encode()).hexdigest(), 16) % 4
        for arm in templates.ARMS[offset:] + templates.ARMS[:offset]:
            tasks.append(dict(index=len(tasks), task_id=item["probe_id"] + "/" + arm,
                probe_id=item["probe_id"], case_id=item["case_id"], dataset=item["dataset"],
                arm=arm, is_stress=item["is_stress"]))
    return tasks


def predecessor_pins():
    files = {
        "stage": (STAGE, ("PROTOCOL.frozen.json", "INPUTS.frozen.jsonl", "TASK_ORDER.frozen.jsonl")),
        "operator": (OPERATOR, ("PROTOCOL.frozen.json", "TASKS.frozen.jsonl")),
        "factor": (FACTOR, ("PROTOCOL.frozen.json", "INPUTS90.jsonl", "ORDER360.jsonl", "RESOURCE_CONTRACTS.json")),
    }
    return {label: {rel(folder / name): sha(folder / name) for name in names}
            for label, (folder, names) in files.items()}


def prepare():
    if PROTOCOL.exists(): return check()
    assert not (HERE / "RUN_AUTHORIZATION.json").exists() and not (OUT / "checkpoint.sqlite3").exists()
    source = STAGE / "INPUTS.frozen.jsonl"
    if INPUTS.exists(): assert INPUTS.read_bytes() == source.read_bytes()
    else: INPUTS.write_bytes(source.read_bytes())
    items = rows(INPUTS)
    forbidden = {"label", "labels", "gold", "gold_answer", "reference_answer", "metrics", "f1", "em"}
    assert all(r["role"] == "ordinary" and not forbidden.intersection(r) for r in items)
    jsonl(ORDER, task_order(items))
    generation = read(PACKAGE / "config.json")["generation"]
    atomic_json(HERE / "GENERATION.frozen.json", generation)
    atomic_json(HERE / "PREDECESSORS.frozen.json", predecessor_pins())
    jsonl(HERE / "PROMPT_CONTRACTS.frozen.jsonl", [dict(probe_id=r["probe_id"], arm=a,
        max_tokens=templates.cap(a), payload_sha256=digest(build_payload(generation, templates.prompt(r, a), templates.cap(a))))
        for r in items for a in templates.ARMS])
    return dict(state="prepared_not_frozen_not_authorized", nodes=38, conditions=152,
        initial_engineering_conditions=32, new_API_SSH_GPU_requests=0)


def validate():
    if not INPUTS.exists(): prepare()
    assert INPUTS.read_bytes() == (STAGE / "INPUTS.frozen.jsonl").read_bytes()
    items = rows(INPUTS); assert rows(ORDER) == task_order(items)
    assert tuple(templates.ARMS) == ("N128", "N256", "F256", "R256")
    generation = read(HERE / "GENERATION.frozen.json")
    assert generation == read(PACKAGE / "config.json")["generation"]
    contracts = []
    for item in items:
        assert templates.prompt(item, "N128") == templates.prompt(item, "N256") == item["control_prompt"]
        for arm in templates.ARMS:
            prompt = templates.prompt(item, arm)
            payload = build_payload(generation, prompt, templates.cap(arm))
            assert len(payload["messages"]) == 1 and payload["messages"][0] == dict(role="user", content=prompt)
            assert templates.cap(arm) == (128 if arm == "N128" else 256)
            contracts.append(dict(probe_id=item["probe_id"], arm=arm, max_tokens=templates.cap(arm), payload_sha256=digest(payload)))
        assert formatter_prompt(item, "VALUE").endswith("\n\n".join(d["block"] for d in item["reader_packet"]["docs"]))
    assert contracts == rows(HERE / "PROMPT_CONTRACTS.frozen.jsonl")
    assert read(HERE / "PREDECESSORS.frozen.json") == predecessor_pins()
    value = dict(passed=True, nodes=38, conditions=152, source_bytes_exact=True,
        native_prompts_identical_only_cap_differs=True, isolated_seed=None,
        downstream_support_never_sent=True, native_formatter_cap=128,
        new_API_SSH_GPU_requests=0)
    atomic_json(HERE / "OFFLINE_CONTRACTS.json", value)
    return value


def check():
    protocol = read(PROTOCOL)
    for name, expected in {**protocol["pins"], **protocol["protected_holds"]}.items():
        assert sha(ROOT / name) == expected, name
    assert protocol["experiment"] == HERE.name and protocol["planned_conditions"] == 152
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert rows(ORDER) == task_order(rows(INPUTS))
    return dict(passed=True, state="frozen_not_execution_authorized", protocol_sha256=sha(PROTOCOL),
        nodes=38, conditions=152, new_API_SSH_GPU_requests=0)


def freeze():
    """Only root invokes after review; this never creates execution authorization."""
    if PROTOCOL.exists(): return check()
    prepare(); validate()
    assert (HERE / "PROTOCOL.md").is_file()
    assert read(HERE / "TEMPLATE_CONTRACTS.json")["passed"] is True
    assert read(HERE / "CONTROLLER_OFFLINE_CHECKS.json")["passed"] is True
    paths = [p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md")
             and p.name not in ("PROTOCOL.frozen.json", "RUN_AUTHORIZATION.json")]
    paths += [STAGE / "templates.py", STAGE / "INPUTS.frozen.jsonl", CONTEXT / "native_ask.py",
        CONTEXT / "message_transport.py", HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        HERE.parent / "p02_formal1500_v1_20261021/network.py", ROOT / "scripts/cli_support.py", PACKAGE / "config.json"]
    paths += list((PACKAGE / "core").glob("*.py"))
    paths += [templates.EXAMPLES_PATH]
    pause = read(HERE / "FACTOR_TECHNICAL_PAUSE_AUTHORIZATION.json")
    paths += [ROOT / pause["scheduler_receipt_path"]]
    for group in read(HERE / "PREDECESSORS.frozen.json").values(): paths += [ROOT / p for p in group]
    atomic_json(PROTOCOL, dict(experiment=HERE.name, revision=1, state="frozen_not_authorized", utc=utc(),
        planned_nodes=38, planned_conditions=152, arms=list(templates.ARMS), initial_engineering_conditions=32,
        fresh_seed=None, scope="ordinary QA plus Answer-only native formatter; no final/RAG scores",
        complete_response_and_accepted_answer_correctness_reviewed_separately=True,
        technical_recovery="Only this fresh namespace, max5 durable returnless technical attempts; held skipped, reservations retained",
        predecessor_guard="Exact stage152 and operator346 complete/no-inflight; exact factor90 first24 engineering PASS OR root-authorized verified technical pause",
        engineering="First stress32 wire/receipt/parser integrity only; algorithm format failures retained, never quality selection",
        main_evaluation_remains_paused=True,
        pins={rel(p): sha(p) for p in paths}, protected_holds=read(STAGE / "PROTOCOL.frozen.json")["protected_holds"]))
    return check()


def _metadata(folder, kind, meta_key):
    with closing(ro(folder / "checkpoint.sqlite3")) as db:
        db.execute("BEGIN")
        row = db.execute("SELECT value FROM data WHERE kind='meta' AND key=?", (meta_key,)).fetchone()
        identity = json.loads(row[0]) if row else None
        records = dict(db.execute("SELECT key,json_extract(value,'$.status') FROM data WHERE kind=?", (kind,)))
        inflight = db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
    return identity, records, inflight


def factor_technical_pause_status(expected, identity_value, done, active, planned):
    """A closed allocation is not a completed experiment. This is a distinct root-approved scheduling release."""
    authority = read(HERE / "FACTOR_TECHNICAL_PAUSE_AUTHORIZATION.json")
    assert authority["approved"] is True and authority["experiment"] == HERE.name
    assert authority["factor_experiment"] == FACTOR.name
    assert authority["factor_protocol_sha256"] == expected["protocol_sha256"]
    assert authority["reason"] == "technical_execution_failure_not_quality"
    decisions = [r for r in rows(PLAN / "DECISIONS.jsonl") if r.get("id") == authority["decision_id"]]
    assert len(decisions) == 1 and decisions[0].get("approved") is True
    assert decisions[0].get("experiment") == HERE.name
    receipt_path = ROOT / authority["scheduler_receipt_path"]
    stop_path = ROOT / authority["factor_stop_path"]
    factor_out = ROOT / "outputs/query_decomposition" / FACTOR.name
    assert receipt_path.resolve() == (factor_out / "SCHEDULER_TERMINAL.json").resolve()
    assert stop_path.resolve() == (factor_out / "STOP_REQUESTED").resolve()
    assert sha(receipt_path) == authority["scheduler_receipt_sha256"] and sha(stop_path) == authority["factor_stop_sha256"]
    receipt = read(receipt_path); job = read(factor_out / "REMOTE_JOB_ID.json")
    job_id = authority["job_id"]
    assert isinstance(job_id, str) and job_id.isdigit()
    directory = job["directory"]
    assert job["job_id"] == job_id and job["protocol_sha256"] == expected["protocol_sha256"]
    assert directory.endswith("/factor90_" + expected["protocol_sha256"][:16])
    assert receipt["experiment"] == FACTOR.name and receipt["protocol_sha256"] == expected["protocol_sha256"]
    assert receipt["job_id"] == job_id and receipt["remote_directory"] == directory
    assert receipt["read_only_remote"] is True and receipt["new_jobs"] == receipt["API_requests"] == 0
    terminal = receipt["terminal"]
    assert terminal["verified"] is True and terminal["state"] == "COMPLETED"
    assert terminal["job_id"] == job_id and terminal["remote_directory"] == directory
    commands = receipt["commands"]
    assert commands["squeue"]["argv"] == ["squeue", "-h", "-j", job_id, "-o", "%i|%T|%j"]
    assert commands["squeue"]["code"] == 0 and not commands["squeue"]["stdout"].strip()
    assert commands["sacct"]["argv"] == ["sacct", "-n", "-P", "-j", job_id,
        "--format=JobIDRaw,JobName,State,ExitCode,Start,End,Timelimit,NodeList"]
    assert commands["scontrol"]["argv"] == ["scontrol", "show", "job", "-o", job_id]
    assert commands["sacct"]["code"] == commands["scontrol"]["code"] == 0
    accounting = [s.split("|") for s in commands["sacct"]["stdout"].splitlines() if s.strip()]
    assert accounting and sum(row[0] == job_id for row in accounting) == 1
    assert all(len(row) >= 6 and (row[0] == job_id or row[0].startswith(job_id + "."))
        and row[2] == "COMPLETED" and row[3] == "0:0" and row[5] not in ("", "Unknown") for row in accounting)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=([^\s]*)", commands["scontrol"]["stdout"]))
    assert fields.get("JobId") == job_id and fields.get("JobState") == "COMPLETED"
    assert fields.get("JobName") == "p02-factor90" and fields.get("WorkDir") == directory
    assert fields.get("UserId", "").split("(", 1)[0] == "u7830488"
    held = {read(p)["task_id"] for p in (factor_out / "held").glob("*.json")}
    assert identity_value == expected and set(done) <= planned and held <= planned and not set(done).intersection(held)
    assert set(done.values()) <= {"success", "algorithm_failure"}
    attempted = len(set(done) | held)
    assert active == authority["in_flight"] == 0 and attempted == authority["attempted"] == 3
    assert receipt["progress"]["valid"] is True and receipt["progress"]["attempted"] == attempted
    assert receipt["progress"]["in_flight"] == 0 and receipt["progress"]["planned"] == len(planned) == 360
    return dict(finished=True, mode="verified_technical_pause", attempted=attempted, in_flight=active,
        factor_quality_not_a_gate=True, job_id=job_id, decision_id=authority["decision_id"])


def predecessor_status():
    """Metadata-only, read-only. Never import a predecessor runner or its scorer."""
    result = {}
    try:
        for group in read(HERE / "PREDECESSORS.frozen.json").values():
            for name, expected in group.items():
                if sha(ROOT / name) != expected: return dict(finished=False, reason="predecessor_pin_changed")
        stage_out = ROOT / "outputs/query_decomposition" / STAGE.name
        identity, done, active = _metadata(stage_out, "outcome", "identity")
        expected = dict(protocol_sha256=sha(STAGE / "PROTOCOL.frozen.json"), inputs_sha256=sha(STAGE / "INPUTS.frozen.jsonl"),
            order_sha256=sha(STAGE / "TASK_ORDER.frozen.jsonl"), historical_seed=None)
        tasks = rows(STAGE / "TASK_ORDER.frozen.jsonl"); planned = {t["task_id"] for t in tasks}
        held = {read(p)["task_id"] for p in (stage_out / "held").glob("*.json")}
        valid = (identity == expected and len(tasks) == len(planned) == 152 and set(done) <= planned and held <= planned
                 and set(done.values()) <= {"success", "algorithm_failure"} and not set(done).intersection(held))
        result["stage"] = dict(finished=bool(valid and (set(done) | held) == planned and active == 0), attempted=len(set(done) | held), in_flight=active)
        op_out = ROOT / "outputs/query_decomposition" / OPERATOR.name
        identity, done, active = _metadata(op_out, "prediction", "protocol")
        tasks = rows(OPERATOR / "TASKS.frozen.jsonl"); planned = {t["key"] for t in tasks}
        valid = (identity == dict(sha256=sha(OPERATOR / "PROTOCOL.frozen.json")) and len(tasks) == len(planned) == 346
            and set(done) == planned and set(done.values()) <= {"success", "algorithm_failure", "technical_held"})
        result["operator"] = dict(finished=bool(valid and active == 0), attempted=len(done), in_flight=active)
        factor_out = ROOT / "outputs/query_decomposition" / FACTOR.name
        identity, done, active = _metadata(factor_out, "outcome", "identity")
        expected = dict(protocol_sha256=sha(FACTOR / "PROTOCOL.frozen.json"), inputs_sha256=sha(FACTOR / "INPUTS90.jsonl"),
            order_sha256=sha(FACTOR / "ORDER360.jsonl"), resources=read(FACTOR / "RESOURCE_CONTRACTS.json"), no_historical_LLM_seed=True)
        tasks = rows(FACTOR / "ORDER360.jsonl"); planned = {t["task_id"] for t in tasks}
        first = {t["task_id"] for t in tasks if t["engineering_initial"]}
        gate = read(factor_out / "ENGINEERING_GATE.json") if (factor_out / "ENGINEERING_GATE.json").exists() else {}
        valid = (identity == expected and len(tasks) == len(planned) == 360 and len(first) == 24
            and first <= set(done) <= planned and set(done.values()) <= {"success", "algorithm_failure"}
            and gate.get("passed") is True and gate.get("planned_conditions") == 24
            and gate.get("quality_or_predictions_inspected") is False)
        # Factor90 may continue running: only its completed first24 is the barrier.
        result["factor"] = (dict(finished=True, mode="first24_engineering_pass", engineering_attempted=len(first.intersection(done)), engineering_conditions=24)
            if valid else factor_technical_pause_status(expected, identity, done, active, planned))
        return dict(finished=all(s["finished"] for s in result.values()), **result)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, AssertionError) as exc:
        return dict(finished=False, reason="missing_or_invalid_predecessor_metadata", error_type=type(exc).__name__, **result)


def authorized():
    check(); authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert authority.get("protocol_sha256") == sha(PROTOCOL)
    assert authority.get("current_namespace_returnless_recovery_only") is True
    assert authority.get("maximum_technical_attempts") == 5 and authority.get("approved_conditions") == 152
    matches = [r for r in rows(PLAN / "DECISIONS.jsonl") if r.get("id") == authority.get("decision_id")]
    assert len(matches) == 1 and matches[0].get("approved") is True
    assert matches[0].get("experiment") == HERE.name and matches[0].get("protocol_sha256") == sha(PROTOCOL)
    return authority


def await_predecessors(wait):
    while True:
        authorized(); status = predecessor_status()
        atomic_json(OUT / "PREDECESSOR_WAIT_STATUS.json", dict(utc=utc(), new_API_SSH_GPU_requests=0, **status))
        if status["finished"]: return True
        if not wait or (OUT / "STOP_REQUESTED").exists(): return False
        time.sleep(30)


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def identity():
    return dict(protocol_sha256=sha(PROTOCOL), inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER), historical_seed=None)


def held_outcomes(): return [read(p) for p in (OUT / "held").glob("*.json")]


def outcomes():
    if not (OUT / "checkpoint.sqlite3").exists(): return []
    with closing(ro(OUT / "checkpoint.sqlite3")) as db:
        return [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='outcome'")]


def perform(item, arm, native):
    prompt = templates.prompt(item, arm)
    result = dict(raw_current_question=item["current_question"], actual_child_prompt=prompt,
        actual_reader_packet=item["reader_packet"], raw_output=None, support=None,
        parsed_answer=None, accepted_answer=None, formatted_answer=None,
        formatter_triggered=False, parser_status="not_reached")
    try:
        rid, raw, finish = native("legacy/answer/" + str(item["step_id"]), prompt, templates.cap(arm))
        result.update(raw_output=raw, request_id=rid, finish_reason=finish)
        try:
            parsed = templates.parse(arm, raw)
        except templates.FormatFailure:
            result.update(status="algorithm_failure", parser_status="format_failure", reason="invalid_support_answer_format",
                failure_stage="parse")
        else:
            result.update(support=parsed["support"], parsed_answer=parsed["answer"], parser_status="parsed")
            answer = parsed["answer"]
            result["formatter_triggered"] = needs_formatter(answer)
            if result["formatter_triggered"]:
                rid, answer, finish = native("legacy/sentence_format/" + str(item["step_id"]), formatter_prompt(item, answer), 128)
                result["formatted_answer"] = answer
            result.update(status="success", accepted_answer=answer, request_id=rid, finish_reason=finish)
    except AlgorithmTerminal as exc:
        stage = exc.value
        if result["raw_output"] is None: result["raw_output"] = stage["answer"]
        result.update(status="algorithm_failure", reason=stage["terminal_reason"], failure_stage=stage["stage"],
            request_id=stage["request_id"], finish_reason=stage["finish_reason"])
    except (TechnicalHeld, InterruptedError) as exc:
        result.update(status="held_technical", error_type=type(exc).__name__)
    return dict(**result, stages=native.stages,
        usage={k: sum(s["usage"][k] for s in native.stages) for k in FIELDS}, utc=utc())


def engineering_gate(cache, items, config):
    """Structural validation only; model formatting/algorithm failures do not stop the cohort."""
    selected = [t for t in rows(ORDER) if t["is_stress"]]
    found = cache.all("outcome"); held = {r["task_id"]: r for r in held_outcomes()}
    merged = {**held, **found}; checked = format_failures = algorithm_failures = technical_holds = 0
    try:
        assert cache.get("meta", "identity") == identity()
        assert len(selected) == 32 and all(t["task_id"] in merged for t in selected)
        for task in selected:
            value = merged[task["task_id"]]; item = items[task["probe_id"]]; sid = str(item["step_id"])
            assert all(value[k] == task[k] for k in task)
            assert value["actual_child_prompt"] == templates.prompt(item, task["arm"])
            assert value["status"] in ("success", "algorithm_failure", "held_technical")
            expected = {"legacy/answer/" + sid: build_payload(config["generation"], value["actual_child_prompt"], templates.cap(task["arm"]))}
            if value["formatter_triggered"]:
                assert value["parser_status"] == "parsed" and needs_formatter(value["parsed_answer"])
                expected["legacy/sentence_format/" + sid] = build_payload(config["generation"], formatter_prompt(item, value["parsed_answer"]), 128)
            for stage in value["stages"]:
                stored = cache.db.execute("SELECT payload,status,result FROM requests WHERE id=?", (stage["request_id"],)).fetchone()
                assert stored is not None
                payload = json.loads(stored[0]); receipt = dict(json.loads(stored[2]), status=stored[1])
                assert payload == expected[stage["stage"]] and digest(payload) == stage["payload_sha256"]
                assert stage_success(receipt, config["generation"]["model"], stage["stage"]) or terminal_reason(receipt, config["generation"]["model"])
                checked += 1
            # Verify wire identity even for a technical receipt omitted from native.stages.
            for name, payload in expected.items():
                label = "full8/" + task["task_id"] + "/" + name
                for stored, in cache.db.execute("SELECT q.payload FROM requests q JOIN request_stages s ON q.id=s.request_id WHERE s.stage=?", (label,)):
                    assert json.loads(stored) == payload
            if value["parser_status"] == "format_failure":
                format_failures += 1
                try: templates.parse(task["arm"], value["raw_output"])
                except templates.FormatFailure: pass
                else: raise AssertionError("Recorded parse failure is not reproducible")
                assert value["status"] == "algorithm_failure" and not value["formatter_triggered"]
            elif value["parser_status"] == "parsed":
                assert templates.parse(task["arm"], value["raw_output"]) == dict(support=value["support"], answer=value["parsed_answer"])
                if value["status"] == "success":
                    assert value["accepted_answer"] == (value["formatted_answer"] if value["formatter_triggered"] else value["parsed_answer"])
            assert value["status"] == "success" or value["accepted_answer"] is None
            algorithm_failures += value["status"] == "algorithm_failure"
            technical_holds += value["status"] == "held_technical"
        return dict(passed=True, status="engineering_valid_continue_all38", protocol_sha256=sha(PROTOCOL),
            planned_conditions=32, stages_checked=checked, format_failures=format_failures,
            algorithm_failures=algorithm_failures, technical_holds=technical_holds, semantic_quality_not_a_gate=True)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return dict(passed=False, status="engineering_contract_failure", error_type=type(exc).__name__,
            planned_conditions=32, stages_checked=checked, semantic_quality_not_a_gate=True)


def accounting():
    ledger = Path(read(PACKAGE / "config.json")["budget"]["ledger"])
    with closing(ro(ledger)) as db:
        charges = list(db.execute("SELECT actual,reserved FROM charges WHERE namespace=?", ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),)))
    return dict(actual_new_returned_usage_CNY=sum(a or 0.0 for a, _ in charges),
        unresolved_reserved_CNY=sum(r for a, r in charges if a is None))


def report():
    check(); tasks = rows(ORDER); planned = {t["task_id"] for t in tasks}
    found = outcomes(); held = held_outcomes(); done = {r["task_id"] for r in found}; held_ids = {r["task_id"] for r in held}
    assert done <= planned and held_ids <= planned and not done.intersection(held_ids)
    attempted = done | held_ids
    successes = {a: {r["probe_id"] for r in found if r["arm"] == a and r["status"] == "success"} for a in templates.ARMS}
    common = set.intersection(*successes.values())
    def arm_stats(arm):
        values = [r for r in found if r["arm"] == arm]
        matched = [r for r in values if r["probe_id"] in common]
        return dict(planned=38, terminal=len(values), held_technical=sum(r["arm"] == arm for r in held),
            statuses=dict(Counter(r["status"] for r in values)),
            format_failures=sum(r["parser_status"] == "format_failure" for r in values),
            formatter_triggered=sum(r["formatter_triggered"] for r in values),
            logical_usage={k: sum(r["usage"][k] for r in values) for k in FIELDS},
            common_success_n=len(matched), common_success_mean_usage={k: sum(r["usage"][k] for r in matched) / len(matched) if matched else None for k in FIELDS},
            common_success_mean_component_CNY=sum(cost(r["usage"]) for r in matched) / len(matched) if matched else None)
    result = dict(utc=utc(), status="all_attempted" if attempted == planned else "partial", planned_conditions=152,
        terminal=len(found), held_technical=len(held_ids), attempted=len(attempted), not_started=len(planned - attempted),
        by_arm={a: arm_stats(a) for a in templates.ARMS}, main_evaluation_remains_paused=True,
        scope="OrdinaryQA component only; no final F1. Correctness requires later separate whole-response and accepted-answer review.",
        missing_not_zero=True, engineering_gate=read(OUT / "ENGINEERING_GATE.json") if (OUT / "ENGINEERING_GATE.json").exists() else None,
        **accounting())
    atomic_json(OUT / "REPORT.json", result)
    return result


def run(limit=None, wait=False):
    authorized(); attempted = {r["task_id"] for r in outcomes() + held_outcomes()}
    if attempted == {r["task_id"] for r in rows(ORDER)}: return report()
    if not await_predecessors(wait): return dict(state="waiting_predecessors", new_API_SSH_GPU_requests=0)
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config = read(PACKAGE / "config.json"); items = {r["probe_id"]: r for r in rows(INPUTS)}
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    class NetworkEvents:
        def event(self, name, **fields): event(dict(event=name, **fields))
    with run_lock(OUT):
        authorized(); assert predecessor_status()["finished"]
        with Cache(OUT / "checkpoint.sqlite3") as cache:
            old = cache.get("meta", "identity")
            if old is None:
                assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0 and not cache.all("outcome")
                cache.put("meta", "identity", identity())
            else: assert old == identity()
            with closing(NoCapBudget(config, OUT)) as budget, temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
                    provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
                stop = lambda: (OUT / "STOP_REQUESTED").exists()
                with closing(SubmissionAwareStrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as strict:
                    processed = 0; engineering_passed = False
                    for task in rows(ORDER):
                        if task["task_id"] in {r["task_id"] for r in held_outcomes()} or cache.get("outcome", task["task_id"]) is not None: continue
                        if stop() or (limit is not None and processed >= limit): break
                        if not task["is_stress"] and not engineering_passed:
                            gate = engineering_gate(cache, items, config); atomic_json(OUT / "ENGINEERING_GATE.json", gate)
                            if not gate["passed"]: event(dict(event="engineering_gate_stopped", **gate)); break
                            engineering_passed = True
                        native = NativeAsk(cache, strict, config, task["task_id"], event=event)
                        item = items[task["probe_id"]]; result = dict(**task, **perform(item, task["arm"], native))
                        if result["status"] == "held_technical": atomic_json(OUT / "held" / (digest(task["task_id"]) + ".json"), result)
                        else: cache.put("outcome", task["task_id"], result)
                        processed += 1
                        reference = item.get("historical_stage_usage_reference")
                        if native.stages and reference and cost(reference) > 0 and cost(native.stages[0]["usage"]) > 3 * cost(reference):
                            event(dict(event="expense_alert", task_id=task["task_id"], reason="ordinaryQA_component_over3x_same_historical_component",
                                ratio=cost(native.stages[0]["usage"]) / cost(reference), not_fullquery_cost=True))
                        summary = dict(event="condition_status", task_id=task["task_id"], status=result["status"], parser_status=result["parser_status"])
                        event(summary); print(json.dumps(summary), flush=True)
    return report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "check", "freeze", "predecessors", "report", "run"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and (args.action != "run" or args.limit < 1): parser.error("--limit is a positive run-only count")
    if args.wait and args.action != "run": parser.error("--wait is run-only")
    result = run(args.limit, args.wait) if args.action == "run" else {
        "prepare": prepare, "validate": validate, "check": check, "freeze": freeze,
        "predecessors": predecessor_status, "report": report}[args.action]()
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__": main()
