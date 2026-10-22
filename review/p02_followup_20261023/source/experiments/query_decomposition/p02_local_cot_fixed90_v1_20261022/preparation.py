"""Offline copies and provenance only. This module cannot authorize or submit work."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.pipeline import safe_case
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json
from .flow import ARMS

REFERENCE = HERE.parent / "p02_reference_binding_priority_fixed90_v1_20261022"
STAGE = HERE.parent / "p02_local_cot_stage38_v1_20261022"
CONTEXT = HERE.parent / "p02_shared_context_fixed90_v1_20261022"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
INPUTS = HERE / "INPUTS90.jsonl"
ORDER = HERE / "ORDER360.jsonl"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
HISTORY = HERE / "HISTORICAL_REFERENCES90.local.jsonl"
DATASETS = ("hotpotqa", "2wiki", "musique")
COPIED = ("INPUTS90.jsonl", "RESOURCE_CONTRACTS.json", "QUERY_MODEL.frozen.json", "EXECUTION_SETTINGS.json",
    "INITIAL_ENGINEERING6.frozen.json", "PUBLIC_INPUT_PROVENANCE.json", "SOURCE_PROVENANCE90.jsonl",
    "remote_worker.py", "resource_contract.py", "retrieval_bridge.py")
LOCAL_ONLY = ("HISTORICAL_REFERENCES90.local.jsonl", "HISTORICAL_COST_PROXIES.local.json", "references90.local.jsonl")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path): return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def jsonl(path, values): Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf8")
def ro(path): return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def order(data):
    first = read(HERE / "INITIAL_ENGINEERING6.frozen.json")["ids"]
    assert len(first) == len(set(first)) == 6
    index = {r["id"]: r for r in data}
    ordered = [index[i] for i in first] + [r for r in data if r["id"] not in first]
    result = []
    offset = int(hashlib.sha256(b"20261022|local_cot_fixed90").hexdigest(), 16) % 4
    for i, item in enumerate(ordered):
        shift = (i + offset) % 4
        for arm in ARMS[shift:] + ARMS[:shift]:
            result.append(dict(index=len(result), task_id=item["id"] + "/" + arm, id=item["id"],
                dataset=item["dataset"], arm=arm, engineering_initial=item["id"] in first))
    return result


def prepare():
    if PROTOCOL.exists(): return check()
    assert not (HERE / "RUN_AUTHORIZATION.json").exists() and not (OUT / "checkpoint.sqlite3").exists()
    for name in COPIED + LOCAL_ONLY:
        source, target = REFERENCE / name, HERE / name
        if target.exists(): assert target.read_bytes() == source.read_bytes(), name
        else: target.write_bytes(source.read_bytes())
    data = rows(INPUTS)
    assert len(data) == len({r["id"] for r in data}) == 90
    assert Counter(r["dataset"] for r in data) == {d: 30 for d in ("hotpotqa", "2wiki", "musique")}
    for item in data: safe_case(item)
    jsonl(ORDER, order(data))
    atomic_json(HERE / "GENERATION.prepared.json", read(PACKAGE / "config.json")["generation"])
    atomic_json(HERE / "PREDECESSORS.prepared.json", {
        label: {rel(folder / name): sha(folder / name) for name in names}
        for label, folder, names in (("stage", STAGE, ("PROTOCOL.frozen.json", "INPUTS.frozen.jsonl", "TASK_ORDER.frozen.jsonl")),
            ("reference", REFERENCE, ("PROTOCOL.frozen.json", "INPUTS90.jsonl", "ORDER360.jsonl", "RESOURCE_CONTRACTS.json")))})
    atomic_json(HERE / "PREPARATION_MANIFEST.json", dict(state="prepared_not_frozen_not_authorized", source=rel(REFERENCE),
        source_protocol_sha256=sha(REFERENCE / "PROTOCOL.frozen.json"), stage_protocol_sha256=sha(STAGE / "PROTOCOL.frozen.json"),
        source_template=rel(STAGE / "templates.py"), source_template_sha256=sha(STAGE / "templates.py"),
        source_stage_outcomes_not_loaded=True, nodes_or_quality_not_selected=True,
        copies=[dict(path=rel(HERE / n), source=rel(REFERENCE / n), sha256=sha(HERE / n),
            local_scoring_only=n in LOCAL_ONLY) for n in COPIED + LOCAL_ONLY],
        inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER), queries=90, conditions=360,
        goal_only_cases=sum(not r["supporting_questions"] for r in data),
        planned_first_engineering_conditions=24, engineering_selection_copied_no_quality_filter=True,
        proposed_remote_namespace="cot90_<new_protocol_sha256_prefix>", existing_workers_not_shared=True,
        no_new_API_SSH_GPU_requests=True, no_execution_authority=True))
    return validate()


def validate():
    manifest = read(HERE / "PREPARATION_MANIFEST.json")
    assert manifest["state"] == "prepared_not_frozen_not_authorized"
    assert manifest["source_template_sha256"] == sha(STAGE / "templates.py")
    assert manifest["source_protocol_sha256"] == sha(REFERENCE / "PROTOCOL.frozen.json")
    assert manifest["stage_protocol_sha256"] == sha(STAGE / "PROTOCOL.frozen.json")
    for item in manifest["copies"]:
        assert sha(ROOT / item["path"]) == sha(ROOT / item["source"]) == item["sha256"]
    assert rows(ORDER) == order(rows(INPUTS)) and manifest["order_sha256"] == sha(ORDER)
    assert read(HERE / "GENERATION.prepared.json") == read(PACKAGE / "config.json")["generation"]
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return dict(passed=True, state="prepared_not_frozen_not_authorized", queries=90, conditions=360,
        engineering_conditions=24, goal_only_cases=manifest["goal_only_cases"], no_execution_authority=True,
        new_API_SSH_GPU_requests=0)


def check():
    protocol = read(PROTOCOL)
    assert protocol["experiment"] == HERE.name and protocol["planned_conditions"] == 360
    for name, expected in {**protocol["pins"], **protocol["protected_holds"]}.items():
        assert sha(ROOT / name) == expected, name
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert rows(ORDER) == order(rows(INPUTS))
    return dict(passed=True, state="frozen_execution_requires_exact_AUTH", protocol_sha256=sha(PROTOCOL),
        queries=90, conditions=360, new_API_SSH_GPU_requests=0)


def freeze():
    """Root only after independent review. Freeze is not execution approval."""
    if PROTOCOL.exists(): return check()
    prepare(); validate()
    check_test_receipts()
    paths = [f for f in HERE.iterdir() if f.is_file() and f.suffix in (".py", ".json", ".jsonl", ".md")
        and f.name not in ("PROTOCOL.frozen.json", "RUN_AUTHORIZATION.json")]
    paths += list((PACKAGE / "core").glob("*.py")) + list((PACKAGE / "prompts").glob("*.json"))
    paths += [PACKAGE / "__init__.py", PACKAGE / "config.json", STAGE / "templates.py", STAGE / "PROTOCOL.frozen.json",
        CONTEXT / "native_ask.py", CONTEXT / "message_transport.py",
        HERE.parent / "p02_reference_binding_priority_probe_v1_20261022/templates.py",
        HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        HERE.parent / "p02_formal1500_v1_20261021/network.py", ROOT / "scripts/cli_support.py",
        HERE.parent / "comprehensive_eval_v1_20261015/scoring.py"]
    from . import retrieval_transport
    paths += [retrieval_transport.PORTABLE] + [ROOT / name for name in retrieval_transport.DEPENDENCIES]
    for group in read(HERE / "PREDECESSORS.prepared.json").values(): paths += [ROOT / name for name in group]
    # Explicit official scorer dependencies previously audited for the same data.
    for name in read(CONTEXT / "RUN_AUTHORIZATION.json")["additional_scorer_dependencies"]: paths.append(ROOT / name)
    settings = read(HERE / "EXECUTION_SETTINGS.json")
    assert settings["partition_limit_verified_before_freeze"] is True
    paths.append(ROOT / settings["partition_preflight"])
    atomic_json(PROTOCOL, dict(experiment=HERE.name, revision=1, state="frozen_not_authorized", planned_queries=90,
        planned_conditions=360, arms=list(ARMS), engineering_conditions=24, no_historical_LLM_seed=True,
        own_dynamic_accepted_answer_history=True, support_forwarded=False, ordinary_only=True,
        predecessor_barrier="CoT152 attempted/0 in-flight plus reference360 attempted/0 in-flight/all owned jobs closed",
        remote_directory_prefix="cot90_", main_evaluation_remains_paused=True,
        protected_holds=read(REFERENCE / "PROTOCOL.frozen.json")["protected_holds"],
        pins={rel(f): sha(f) for f in paths}))
    return check()


def check_test_receipts():
    """A prior PASS cannot approve source edits made after that check."""
    common = dict(flow_sha256=HERE / "flow.py", template_sha256=STAGE / "templates.py", inputs_sha256=INPUTS)
    sources = {
        "OFFLINE_FLOW_CHECKS.json": dict(**common, order_sha256=ORDER),
        "CONTROLLER_OFFLINE_CHECKS.json": dict(**common, controller_sha256=HERE / "run.py",
            remote_bridge_sha256=HERE / "remote_bridge.py", retrieval_transport_sha256=HERE / "retrieval_transport.py"),
        "SCORER_OFFLINE_CHECKS.json": dict(scorer_sha256=HERE / "score.py"),
    }
    for name, expected in sources.items():
        receipt = read(HERE / name)
        assert receipt["passed"] is True
        for field, source in expected.items():
            assert receipt[field] == sha(source), "Stale offline check: " + name + "/" + field


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("prepare", "validate", "check", "freeze"))
    print(json.dumps({"prepare": prepare, "validate": validate, "check": check, "freeze": freeze}[parser.parse_args().action](), ensure_ascii=True))
