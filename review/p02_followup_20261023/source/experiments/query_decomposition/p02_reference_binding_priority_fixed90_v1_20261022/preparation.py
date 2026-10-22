"""Offline source/order/protocol preparation; no services are created here."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import utc
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022 import run as base
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022 import templates
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json

PRIOR = HERE.parent / "p02_shared_context_fixed90_v1_20261022"
STAGE = HERE.parent / "p02_reference_binding_priority_probe_v1_20261022"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "INPUTS90.jsonl"
ORDER = HERE / "ORDER360.jsonl"
HISTORY = HERE / "HISTORICAL_REFERENCES90.local.jsonl"
read, rows, sha, rel, jsonl = base.read, base.rows, base.sha, base.rel, base.jsonl
COPIED = ("INPUTS90.jsonl", "HISTORICAL_REFERENCES90.local.jsonl", "references90.local.jsonl",
          "SOURCE_PROVENANCE90.jsonl", "PUBLIC_INPUT_PROVENANCE.json", "RESOURCE_CONTRACTS.json",
          "HISTORICAL_COST_PROXIES.local.json", "QUERY_MODEL.frozen.json", "remote_worker.py",
          "resource_contract.py", "retrieval_bridge.py")


def order(data):
    first = read(PRIOR / "INITIAL_ENGINEERING6.frozen.json")["ids"]
    assert len(first) == len(set(first)) == 6
    index = {row["id"]: row for row in data}
    ordered = [index[i] for i in first] + [r for r in data if r["id"] not in first]
    start = int(hashlib.sha256(b"20261022").hexdigest(), 16) % 4
    tasks = []
    for i, row in enumerate(ordered):
        arms = list(templates.ARMS); offset = (i + start) % 4
        for arm in arms[offset:] + arms[:offset]:
            tasks.append(dict(index=len(tasks), id=row["id"], dataset=row["dataset"], arm=arm,
                              task_id=row["id"] + "/" + arm, engineering_initial=row["id"] in first))
    return tasks


def prepare():
    if PROTOCOL.exists():
        return check()
    assert not (HERE / "RUN_AUTHORIZATION.json").exists()
    assert not (OUT / "checkpoint.sqlite3").exists()
    base.check()
    from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022.run import check as stage_check
    stage_check()
    for name in COPIED:
        target = HERE / name
        if target.exists():
            assert target.read_bytes() == (PRIOR / name).read_bytes(), name
        else:
            target.write_bytes((PRIOR / name).read_bytes())
    data = rows(INPUTS)
    assert len(data) == 90 and Counter(r["dataset"] for r in data) == {ds: 30 for ds in base.DATASETS}
    jsonl(ORDER, order(data))
    atomic_json(HERE / "INITIAL_ENGINEERING6.frozen.json", dict(
        ids=read(PRIOR / "INITIAL_ENGINEERING6.frozen.json")["ids"], cases=6, conditions=24,
        committed_full_cohort=90, committed_conditions=360, quality_or_final_F1_not_a_gate=True,
        source=rel(PRIOR / "INITIAL_ENGINEERING6.frozen.json"), source_sha256=sha(PRIOR / "INITIAL_ENGINEERING6.frozen.json")))
    predecessors = dict(context90={name: sha(PRIOR / name) for name in
        ("PROTOCOL.frozen.json", "INPUTS90.jsonl", "ORDER270.jsonl")},
        stage_probe={name: sha(STAGE / name) for name in
        ("PROTOCOL.frozen.json", "INPUTS.frozen.jsonl", "TASK_ORDER.frozen.jsonl")})
    atomic_json(HERE / "PREDECESSORS.frozen.json", predecessors)
    atomic_json(HERE / "REUSED_COMPONENTS.json", dict(components=[dict(
        path=rel(HERE / name), source=rel(PRIOR / name), sha256=sha(HERE / name), byte_identical=True)
        for name in COPIED], imported_runtime=rel(PRIOR), stage_templates_sha256=sha(STAGE / "templates.py"),
        isolated_remote_private_module=True, original_modules_and_outputs_not_mutated=True))
    return dict(state="offline_prepared_not_frozen_not_authorized", cases=90, conditions=360,
                engineering_conditions=24, API_SSH_GPU_requests=0)


def check():
    protocol = read(PROTOCOL)
    for path, expected in {**protocol["pins"], **protocol["protected_holds"]}.items():
        assert sha(ROOT / path) == expected, path
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert rows(ORDER) == order(rows(INPUTS))
    assert len(rows(ORDER)) == 360 and len({r["task_id"] for r in rows(ORDER)}) == 360
    return dict(passed=True, state="frozen_prepared", protocol_sha256=sha(PROTOCOL),
                cases=90, conditions=360, engineering_conditions=24, API_SSH_GPU_requests=0)


def freeze():
    """Root may freeze after independent review; this does not authorize execution."""
    if PROTOCOL.exists():
        return check()
    prepare()
    assert read(HERE / "OFFLINE_CHECKS.json")["passed"] is True
    assert read(HERE / "CONTROLLER_OFFLINE_CHECKS.json")["passed"] is True
    assert read(HERE / "EXECUTION_SETTINGS.json")["partition_limit_verified_before_freeze"] is True
    settings = read(HERE / "EXECUTION_SETTINGS.json")
    preflight = ROOT / settings["partition_preflight"]
    checked_remote = read(preflight)
    assert checked_remote["read_only"] and checked_remote["new_jobs"] == 0 and checked_remote["new_model_requests"] == 0
    assert "db4ai|infinite|up|gpu:a6000:1" in checked_remote["partitions"]["stdout"]
    assert checked_remote["verified_address"] == settings.get("verified_host_address")
    prior = read(PRIOR / "PROTOCOL.frozen.json")
    stage = read(STAGE / "PROTOCOL.frozen.json")
    pins = {**prior["pins"], **stage["pins"]}
    pins[rel(preflight)] = sha(preflight)
    for folder in (PRIOR, STAGE):
        pins[rel(folder / "PROTOCOL.frozen.json")] = sha(folder / "PROTOCOL.frozen.json")
    # The predecessor authorization separately added these scorer dependencies.
    for path, expected in read(PRIOR / "RUN_AUTHORIZATION.json")["additional_scorer_dependencies"].items():
        assert sha(ROOT / path) == expected
        pins[path] = expected
    for path in (HERE.parent / "comprehensive_eval_v1_20261015/scoring.py",):
        pins[rel(path)] = sha(path)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".json", ".jsonl", ".md") and path.name != "RUN_AUTHORIZATION.json":
            pins[rel(path)] = sha(path)
    atomic_json(PROTOCOL, dict(experiment=HERE.name, revision=1, state="frozen_not_authorized", prepared_utc=utc(),
        planned_queries=90, planned_conditions=360, arms=list(templates.ARMS), dataset_counts={ds: 30 for ds in base.DATASETS},
        cohort="Same exposed fixed90 actual frozen R02/v12 arrays; all no-op and goal-only cases retained",
        scope="Independent ordinaryQA C0/S1/B1/P1 factors; each arm uses own new accepted history/live rewrite and dynamic retrieval",
        unchanged="GoalQA, optional formatter, final64, native retrieval and source decompositions; all model requests single user strings",
        engineering="First fixed6 x4=24; wire/receipt contracts only; continue remaining84 regardless F1 after contracts pass",
        predecessor_barrier="Before API/SSH/GPU: exact context90 all270 attempted/no in-flight/matching stopped worker AND stage152 attempted/no in-flight; quality is never a gate",
        fresh="seedNone; only exact current-namespace fullpayload/resourcequery reuse; no historical answers, rewrite, reader or mappings seeded",
        runtime="At most5 durable technical attempts incl first/restarts; returned content/model/id/finish/usage/ledger guarded; all reservations retained; held tasks skipped",
        metrics="Native F1/EM and paired gains/harms vs freshC0; old P02 same90 reference separate; input/output tokens/known fees/unknown reserves; no LLM call metric",
        remote="New factor90_protocolhash16 namespace; A6000/4CPU32G retrieval-only worker with explicit pinned EXECUTION_SETTINGS; no key/messages/gold on remote; no automatic job resubmission",
        execution_settings=read(HERE / "EXECUTION_SETTINGS.json"),
        execution="Root exact-hash authorization plus decision required; no services during preparation, validation or metadata wait",
        main_evaluation_remains_paused=True, protected_holds=prior["protected_holds"], pins=pins))
    return check()
