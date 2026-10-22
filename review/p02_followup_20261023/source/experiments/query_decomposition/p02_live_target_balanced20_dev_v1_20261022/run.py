"""Independent balanced20 preparation; no inference without new authorization.

prepare/dry-run/report never create Cache, provider, remote connection or GPU.
Only run imports the root-owned remote bridge after protocol+auth verification.
"""
from collections import Counter
from contextlib import closing
from copy import deepcopy
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
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE
from . import flow, resource_contract
from .native_ask import NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld, usage_ok
from .retrieval_bridge import identity as retrieval_identity, key as retrieval_key, check_packet

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
I8 = HERE.parent / "p02_ordinary_failure_challenge8_v1_20261022"
ORIGIN = HERE.parent / "p02_failure8_dynamic_target_fullrag_draft_20261022"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
FORMAL = HERE.parent / "p02_formal1500_v1_20261021"
PREFLIGHT = ROOT / "outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
DATASETS = ("hotpotqa", "2wiki")
SEED = 20261022
SCORERS = None
EXPECTED_CASES = 20
INPUTS = HERE/"INPUTS20.jsonl"
ORDER = HERE/"ORDER40.jsonl"
SELECTION = HERE/"COHORT_SELECTION20.frozen.jsonl"
HISTORY = HERE/"HISTORICAL_REFERENCES20.local.jsonl"
FORMAL_OUT = ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in data), encoding="utf-8")


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)


def cost(usage):
    return usage["prompt_tokens"]*INPUT_RATE+usage["completion_tokens"]*OUTPUT_RATE


def paid_query_attribution(cache_db,ledger_db,tasks,outcomes,historical,namespace):
    """Actual charges belong to the request's FIRST submitting condition.

    Include retired technical attempts; exact shared replays do not create a
    second paid owner. Unknown reservations remain visible and never dilute a
    complete-query expense window. No reads create/alter a Cache or ledger row.
    """
    task_ids={t["task_id"] for t in tasks}
    bytask={t:dict(actual_new_returned_usage_CNY=0.,unresolved_reserved_CNY=0.) for t in task_ids}
    owner={}
    for table in("requests","retired_requests"):
        for rid,stage in cache_db.execute("SELECT id,stage FROM "+table):
            matched=[t for t in task_ids if stage.startswith("full8/"+t+"/")]
            assert len(matched)==1,(table,rid,"unknown_first_owner")
            if rid in owner:assert owner[rid]==matched[0]
            owner[rid]=matched[0]
    unattributed=dict(actual_new_returned_usage_CNY=0.,unresolved_reserved_CNY=0.)
    for rid,actual,reserved in ledger_db.execute("SELECT id,actual,reserved FROM charges WHERE namespace=?",(namespace,)):
        dest=bytask[owner[rid]] if rid in owner else unattributed
        dest["actual_new_returned_usage_CNY"]+=actual if actual is not None else 0.
        dest["unresolved_reserved_CNY"]+=reserved if actual is None else 0.
    outindex={r["task_id"]:r for r in outcomes}
    result=[]
    for task in tasks:
        value=bytask[task["task_id"]]
        outcome=outindex.get(task["task_id"])
        status=outcome["status"] if outcome else "held_or_unstarted"
        baseline=historical[task["id"]]["reference_CNY"]
        eligible=(status in("success","success_truncated") and value["actual_new_returned_usage_CNY"]>0
            and value["unresolved_reserved_CNY"]==0)
        result.append({**task,**value,"status":status,"historical_complete_query_CNY":baseline,
            "eligible_for_complete_paid_query_trend":eligible,
            "historical_expense_reference_kind":historical[task["id"]]["reference_kind"],
            "attribution":"First submitting condition; complete flow with shared prefix is a same-period variant proxy, not independent cold full-cost measurement",
            "utc":outcome.get("utc") if outcome else None})
    return result,unattributed


def expense_reasons(current,all_queries):
    reasons=[]
    if current["historical_complete_query_CNY"] and current["actual_new_returned_usage_CNY"]>3*current["historical_complete_query_CNY"]:
        reasons.append("paid_query_above3x_historical_complete_query_reference")
    window=sorted([r for r in all_queries if r["eligible_for_complete_paid_query_trend"]],key=lambda r:r["utc"])[-20:]
    if len(window)==20 and sum(r["actual_new_returned_usage_CNY"]for r in window)>1.5*sum(r["historical_complete_query_CNY"]for r in window):
        reasons.append("recent20_complete_paid_queries_above1.5x_historical_references")
    return reasons


def old_cost_records(public):
    """Costs only from strict complete historical formal flows, never selection."""
    records={};perds={ds:[]for ds in DATASETS}
    for ds in DATASETS:
        directory=FORMAL_OUT/"rounds/r000/cases"/ds/"p02_multigrain_timing_cachefix_v1"
        for path in sorted(directory.glob("*.json")):
            source=read(path)
            calls=source.get("logical_calls") or []
            if(source.get("status")not in("success","success_truncated")or not calls
                or not isinstance(source.get("prediction"),str)
                or not all(usage_ok(c)and c.get("status")=="success"and c.get("full_payload_match_verified")is True
                    and c.get("finish_reason")in(("stop","length")if c["stage"]=="legacy/final"else("stop",))for c in calls)):
                continue
            source_sha=sha(path)
            if rel(path)not in public or public[rel(path)]["sha256"]!=source_sha:continue
            value=dict(id=source["id"],dataset=ds,source=rel(path),source_sha256=source_sha,
                complete_query_CNY=sum(cost(c["usage"])for c in calls))
            records[source["id"]]=(value,source)
            perds[ds].append(value)
    assert all(perds.values()),"A same-dataset historical complete-query proxy is required, never0"
    proxies={ds:dict(dataset=ds,reference_kind="same_method_same_dataset_complete_query_mean_proxy",
        complete_queries=len(r),mean_CNY=sum(v["complete_query_CNY"]for v in r)/len(r),sources=r)
        for ds,r in perds.items()}
    return records,proxies


def assemble():
    """Fixed selected formal questions/rawD only; no quality/status selection."""
    from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.prepare import public_index
    public,publicproof=public_index()
    qfile=FORMAL/"queries1500.id_query.frozen.jsonl"
    qrows=rows(qfile);qindex={r["id"]:(i+1,r)for i,r in enumerate(qrows)}
    assert sha(qfile)==public[rel(qfile)]["sha256"]
    assert sha(FORMAL/"v12.txt")==public[rel(FORMAL/"v12.txt")]["sha256"]
    inputs,provenance=[],[]
    for item in rows(SELECTION):
        position,q=qindex[item["id"]]
        rawpath=ROOT/item["raw_file"]
        raw=read(rawpath)
        question_file=ROOT/item["question_file"]
        assert position==item["formal_position_one_based"] and sha(question_file)==item["question_file_sha256"]==public[rel(question_file)]["sha256"]
        assert sha(rawpath)==item["raw_sha256"]
        assert raw["index"]==position and raw["id"]==q["id"]==item["id"]
        assert raw["query"]==q["query"]==item["question"]
        assert raw["parsed"]==item["supporting_questions"] and item["actual_R02_array_exact"]is True
        assert item["semantic_repairs_applied"]is False
        assert item["source_R02_remote_inputs_sha256"]==sha(FORMAL/"REMOTE_INPUTS.frozen.json")
        assert item["selection_rank_sha256"]==hashlib.sha256(("p02-fixedD-balanced20-v1|20261022|"+item["id"]).encode()).hexdigest()
        assert raw["format_valid"]and not raw["truncated"]and not raw["generation_error"]
        assert raw["actual_prompt"].startswith((FORMAL/"v12.txt").read_text(encoding="utf-8").rstrip())
        clean=safe_case(dict(id=item["id"],qid=item["qid"],dataset=item["dataset"],
            question=item["question"],supporting_questions=item["supporting_questions"]))
        assert clean["supporting_questions"],"Keep user's ordinary-present scope"
        runtime=FORMAL_OUT/"runtime"/clean["dataset"]/(clean["qid"]+".json")
        assert read(runtime)==clean and sha(runtime)==public[rel(runtime)]["sha256"]
        inputs.append(clean)
        provenance.append(dict(id=clean["id"],formal_position_one_based=position,raw_file=rel(rawpath),raw_sha256=sha(rawpath),
            actual_R02_array_exact=True,manual_or_semantic_repairs_applied=False,
            source_R02_remote_inputs_sha256=item["source_R02_remote_inputs_sha256"],
            source_R02_checkpoint_manifest_sha256=item["source_R02_checkpoint_manifest_sha256"],
            prior723_semantic_exposure=item["prior723_semantic_exposure"],
            passive270_review_not_an_exclusion=True,old_quality_or_availability_not_selection_inputs=True,
            public_runtime=rel(runtime),public_runtime_sha256=sha(runtime),
            public_container=public[rel(runtime)]["container"],
            raw_file_itself_prior_publication_not_claimed=True,
            all_inference_fields_exact_public_runtime=True))
    assert len(inputs)==len({r["id"]for r in inputs})==EXPECTED_CASES
    assert Counter(r["dataset"]for r in inputs)=={ds:EXPECTED_CASES//len(DATASETS)for ds in DATASETS}
    # Only AFTER fixed selected inputs exist, inspect old expenses/optional
    # quality references. Never filter, replace or order selected inputs here.
    historical,proxies=old_cost_records(public)
    costs=[]
    for clean in inputs:
        old=historical.get(clean["id"])
        if old:
            expense,source=old
            assert safe_case({k:source[k]for k in clean})==clean
            costs.append(dict(id=clean["id"],dataset=clean["dataset"],reference_CNY=expense["complete_query_CNY"],
                reference_kind="same_query_complete_historical_flow",historical_quality_available=True,
                final_prediction_reference_only=source["prediction"],source=expense["source"],source_sha256=expense["source_sha256"]))
        else:
            proxy=proxies[clean["dataset"]]
            costs.append(dict(id=clean["id"],dataset=clean["dataset"],reference_CNY=proxy["mean_CNY"],
                reference_kind=proxy["reference_kind"],historical_quality_available=False,
                final_prediction_reference_only=None,proxy_complete_queries=proxy["complete_queries"]))
    # References are a separate local-only report input and NEVER part of inputs
    # or the arguments passed to Pipeline, the remote worker or provider.
    wanted = {r["id"] for r in inputs}
    refs = []
    for r in rows(FORMAL / "references1500.local.frozen.jsonl"):
        if r["id"] in wanted:
            keys = ("id", "qid", "dataset", "answer", "answer_id") if r["dataset"] == "2wiki" else ("id", "qid", "dataset", "answer")
            refs.append({k: deepcopy(r[k]) for k in keys})
    assert len(refs) == len(inputs) and {r["id"] for r in refs} == wanted
    remote = read(PREFLIGHT)
    resource_records = {}
    for ds in DATASETS:
        source = remote["manifests"][ds]
        resource_records[ds] = dict(identity=resource_contract.identity(source["manifest"], source["sha256"]),
            contract=resource_contract.contract(source["manifest"], source["sha256"]))
    publicproof.update(queries_file=rel(qfile),queries_sha256=sha(qfile),selected_inference_fields_exact_public_runtime=True,
        runtime_proofs=provenance,corpus_origin="Frozen portable publicbenchmark Wikipedia corpora; resource fileSHA contract retained",
        all_future_retrieved_text_previously_pushed_claimed=False,no_credentials_or_private_user_documents_in_inference=True,
        endpoint_configuration_sha256=sha(PACKAGE/"config.json"))
    return inputs,costs,provenance,refs,resource_records,proxies,publicproof


def order(data):
    # Rotate by case index with a seed-determined start; balanced starts for an
    # evenN (10D0-first/10D1-first here). No score/label influences inputs/order.
    start = int(hashlib.sha256(str(SEED).encode()).hexdigest(), 16) % 2
    return [dict(index=i*2+j, id=r["id"], dataset=r["dataset"], arm=a, task_id=r["id"]+"/"+a)
        for i, r in enumerate(data) for j, a in enumerate(flow.ARMS[(i+start)%2:]+flow.ARMS[:(i+start)%2])]


def prepare(selection_path=None):
    if PROTOCOL.exists():
        return dry_run()
    assert not (OUT / "checkpoint.sqlite3").exists() and not (HERE / "RUN_AUTHORIZATION.json").exists()
    selection_source=HERE/"SELECTION_SOURCE.json"
    if selection_path is not None:
        selection_path=Path(selection_path).resolve()
        assert selection_path.is_relative_to(ROOT)
        original=selection_path.read_bytes()
        if SELECTION.exists():assert SELECTION.read_bytes()==original,"Do not silently change fixedselected20"
        else:SELECTION.write_bytes(original)
        metadata=dict(source=rel(selection_path),source_sha256=sha(selection_path),
            selection_rule="Within dataset SHA256('p02-fixedD-balanced20-v1|20261022|'+canonicalID), recent actual experimentIDs excluded, passive270 not excluded",
            no_old_score_status_availability_filter=True)
        if selection_source.exists():assert read(selection_source)==metadata
        else:atomic_json(selection_source,metadata)
    selected=read(selection_source)
    assert sha(ROOT/selected["source"])==selected["source_sha256"]==sha(SELECTION)
    feasibility=Path(ROOT/selected["source"]).parent/"FEASIBILITY.json"
    assert read(feasibility)["selection_used_no_final_or_semantic_values"]is True
    assert read(feasibility)["old_full_RAG_or_score_availability_not_a_filter"]is True
    data,costs,proof,refs,resources,proxies,publicproof=assemble()
    jsonl(INPUTS,data)
    jsonl(HISTORY,costs)
    jsonl(HERE/"SOURCE_PROVENANCE20.jsonl",proof)
    jsonl(HERE/"references20.local.jsonl",refs)
    jsonl(ORDER,order(data))
    atomic_json(HERE/"HISTORICAL_COST_PROXIES.local.json",proxies)
    atomic_json(HERE/"PUBLIC_INPUT_PROVENANCE.json",publicproof)
    atomic_json(HERE / "RESOURCE_CONTRACTS.json", resources)
    return dict(state="offline_prepared_not_frozen_not_authorized",planned_cases=len(data),
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"]) for r in data),conditions=len(data)*len(flow.ARMS),
        prior723_semantic_exposure=sum(r["prior723_semantic_exposure"]for r in proof),
        historical_complete_quality_coverage=sum(r["historical_quality_available"]for r in costs),
        new_API_GPU_SSH_requests=0, resources={ds:r["identity"] for ds,r in resources.items()})


def freeze():
    if PROTOCOL.exists():
        return dry_run()
    prepare()
    data=rows(INPUTS)
    # Root owns these two files; do not freeze half-written transport code.
    needed = [HERE / "remote_bridge.py", HERE / "remote_worker.py", HERE / "validate_offline.py",
        HERE / "validate_remote_offline.py", HERE / "PROTOCOL.md", HERE / "REMOTE_OFFLINE_CONTRACTS.json",
        HERE / "QUERY_MODEL.frozen.json"]
    assert all(p.is_file() for p in needed), "Finish and review all local/remote code before freeze"
    assert read(HERE / "OFFLINE_CONTRACTS.json")["passed"] is True
    assert read(HERE / "REMOTE_OFFLINE_CONTRACTS.json")["passed"] is True
    pins = {rel(p): sha(p) for p in HERE.glob("*.py")}
    pins.update({rel(p):sha(p) for p in PACKAGE.rglob("*.py") if "__pycache__" not in str(p)})
    extras = [PACKAGE / "config.json", PACKAGE / "prompts/child_examples.json", PACKAGE / "prompts/final_examples.json",
        HERE / "PROTOCOL.md", HERE / "OFFLINE_CONTRACTS.json", HERE/"REMOTE_OFFLINE_CONTRACTS.json",
        HERE/"QUERY_MODEL.frozen.json", I8 / "PUBLIC_PAYLOAD_PROOF.json",
        I8 / "DECOMPOSITION_PROOF8.frozen.jsonl", I8 / "SOURCE_PROOF8.frozen.jsonl",
        HERE.parent / "p02_failure8_actual_target_probe_v1_20261022/templates.py",
        HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        HERE.parent / "robustness_models_retrievers_v1_20261017/remote_resources.py", PREFLIGHT,
        ROOT/"src/indexes/streaming_flat_index.py",ROOT/"src/embedders/text_embedder.py",
        ROOT/"src/persistence/artifact_io.py",ROOT/"src/provenance.py",
        HERE.parent/"comprehensive_eval_v1_20261015/scoring.py",
        HERE.parent/"comprehensive_eval_v1_20261015/common.py",
        HERE.parent/"hotpot_formal_eval_v1/common/scoring.py",
        ROOT/"data/multihop_comprehensive_v1_20261015/2wiki/official/2wikimultihop_evaluate_v1.1.py",
        ROOT/"data/multihop_comprehensive_v1_20261015/2wiki/native/id_aliases.json",
        ROOT / "scripts/cli_support.py", HERE.parent / "p02_formal1500_v1_20261021/network.py"]
    extras += list(HERE.glob("*.jsonl"))+[HERE/"RESOURCE_CONTRACTS.json",HERE/"SELECTION_SOURCE.json",
        HERE/"REUSED_COMPONENTS.json",HERE/"HISTORICAL_COST_PROXIES.local.json",HERE/"PUBLIC_INPUT_PROVENANCE.json"]
    extras += [ROOT/read(HERE/"SELECTION_SOURCE.json")["source"]]
    selection_parent=(ROOT/read(HERE/"SELECTION_SOURCE.json")["source"]).parent
    extras += [selection_parent/"FEASIBILITY.json",selection_parent/"select_prospective.py"]
    def selection_dependencies(value):
        if isinstance(value,dict):
            if isinstance(value.get("path"),str)and isinstance(value.get("sha256"),str):
                path=ROOT/value["path"]
                assert path.resolve().is_relative_to(ROOT)and sha(path)==value["sha256"]
                extras.append(path)
            for child in value.values():selection_dependencies(child)
        elif isinstance(value,list):
            for child in value:selection_dependencies(child)
    selection_dependencies(read(selection_parent/"FEASIBILITY.json"))
    extras += [ROOT/r["raw_file"]for r in rows(HERE/"SOURCE_PROVENANCE20.jsonl")]
    extras += [ROOT/r["public_runtime"]for r in rows(HERE/"SOURCE_PROVENANCE20.jsonl")]
    extras += [ORIGIN/name for name in("flow.py","native_ask.py","resource_contract.py","retrieval_bridge.py","remote_worker.py","QUERY_MODEL.frozen.json")]
    for component in read(HERE/"REUSED_COMPONENTS.json")["components"]:
        original=ROOT/component["source"]
        assert sha(original)==component["source_sha256"]==sha(HERE/component["component"])
        extras.append(original)
    extras += [FORMAL/"questions1500.frozen.jsonl",FORMAL/"queries1500.id_query.frozen.jsonl",FORMAL/"v12.txt",FORMAL/"REMOTE_INPUTS.frozen.json",
        I8/"prepare.py",ROOT/"tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json",
        PLAN/"reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"]
    pins.update({rel(p):sha(p) for p in extras})
    holds = [ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
        PLAN / "STOP_REQUESTED", ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    pin_holds = {rel(p):sha(p) for p in holds if p.exists()}
    exposure=rows(HERE/"SOURCE_PROVENANCE20.jsonl")
    atomic_json(PROTOCOL, dict(experiment=HERE.name, revision=1, prepared_utc=utc(), state="frozen_not_authorized",
        planned_queries=len(data),planned_conditions=len(data)*len(flow.ARMS),
        dataset_counts=dict(Counter(r["dataset"]for r in data)),
        prior723_semantic_exposure=dict(Counter(r["id"].split(":",1)[0]for r in exposure if r["prior723_semantic_exposure"])),
        not_independent_unseen_confirmation=True,
        cohort="Fixed20=HP10+Wiki10 by prespecified IDhash; exclude recent actual experiments only, not passive270/oldscores/availability; prior723 exposure separately disclosed",
        input_scope="Actual accepted formal R02/v12 D fixed; no manual60/weight change; complete dynamic RAG after decomposition",
        comparison="D0 fresh originalP02 ordinary current; D1 allordinary uses this arm's live accepted rewrite in actualQA current only",
        preserved="rawD/history row.request/original rewrite acceptance/dualroute Top10 merge/ordinary+goal128/<4word formatter128 rawQuestion/final64 rawpackaging+validlength",
        controls="Fresh same-experiment control; StrictAsk seed=None. No old LLM responses or retrieval packets imported. Same new exactpayload may serve both arms with attribution; charges dedup.",
        downstream="All subsequent rewrites recomputed from current history; changedqueries real frozenBGE remoteGPUqueryencoder+CPUexactFAISS, never frozen oldreaders",
        runtime=dict(shared_gate=True, gate_interval_seconds=3, SDK_HTTP_retries=0, maximum_actual_attempts_including_restart=5,
            unknown="bounded only if no returnevidence/returnedledger; retain retiredreceipt+unknown reservation",
            algorithm="valid fullmodel+3intusage:goal/QA/rewrite/formatter length,empty,refusal/filter →0; never reroll",
            final="nonempty attributable matchingmodel/no-refusal length accepted and native scored success_truncated",
            technical="missingusage/modelmismatch/unknown/incomplete/exhausted →held, separate denominator bounds",
            no_fixed_budget_cap=True),
        metrics="Native official answerF1/EM; fixedN20 perarm including algorithm0, technical bounds; paired gains/ties/harms. Nativefullcorrect=F1==1orEM==1 separately from semanticlabels. Logical fullflow tokens/cost; actual charges/unknown separately. No LLMcallcount or timingbenchmark.",
        order="case-fixed seed20261022 rotating arm order; no score-based filtering",order_first_counts=dict(Counter(r["arm"]for r in rows(ORDER)if r["index"]%2==0)),
        auth="root reviewed RUN_AUTHORIZATION must bind thishash+decisionID+approvedremote deployment/preflight; no permission implied by prepare/freeze",
        source_public_proof="Every selected5field projection equals hash-verified public formalruntime; queries+v12 publicmanifest exact; not independent unseen confirmation",
        expense_alert="Actualpaidquery including technicalattempts >3x completequeryreference (samequery or labelledsame-datasetproxy); recent20 completepaidqueries(no purecache/partial/unknown)>1.5x references; unknown separate",
        inference_reference_files_forbidden=True, main_evaluation_remains_paused=True, pins=pins, protected_holds=pin_holds))
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for p, expected in {**protocol["pins"], **protocol["protected_holds"]}.items():
        assert sha(ROOT / p) == expected, p
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def dry_run():
    check()
    data,costs,proof,refs,resources,proxies,publicproof=assemble()
    for filename,expected in((INPUTS.name,data),(HISTORY.name,costs),
        ("SOURCE_PROVENANCE20.jsonl",proof),("references20.local.jsonl",refs),(ORDER.name,order(data))):
        assert rows(HERE / filename) == expected, filename
    assert read(HERE / "RESOURCE_CONTRACTS.json") == resources
    assert read(HERE/"HISTORICAL_COST_PROXIES.local.json")==proxies
    assert read(HERE/"PUBLIC_INPUT_PROVENANCE.json")==publicproof
    assert all(set(c)=={"id","qid","dataset","question","supporting_questions"} for c in data)
    result=dict(passed=True,state="frozen_prepared",cases=len(data),planned_conditions=len(data)*len(flow.ARMS),
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"]) for r in data), source_and_actual_R02_arrays_exact=True,
        no_C_or_other_new_candidates_read=True, inference_refs_forbidden=True,
        checkpoint_created=False, new_API_GPU_SSH=0, protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT / "DRYRUN.json", result)
    return result


def authorized():
    check()
    auth = read(HERE / "RUN_AUTHORIZATION.json")
    assert auth.get("approved") is True and auth.get("experiment") == HERE.name
    assert auth.get("protocol_sha256") == sha(PROTOCOL)
    assert any(r.get("id") == auth.get("decision_id") for r in rows(PLAN / "DECISIONS.jsonl"))
    return auth


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps({"utc":utc(),**value},ensure_ascii=False)+"\n")


def check_expense(cache,budget,historic,task):
    paid,unattributed=paid_query_attribution(cache.db,budget.db,rows(ORDER),
        list(cache.all("outcome").values()),historic,"p02_prompt_research_v1_20261022/"+str(OUT.resolve()))
    current=next(r for r in paid if r["task_id"]==task["task_id"])
    reasons=expense_reasons(current,paid)
    if reasons:event(dict(event="expense_alert",task_id=task["task_id"],reasons=reasons,
        paid_query=current,unattributed_charges=unattributed))


def run_with_bridge(bridge):
    """No connection is created here; caller must independently open auth bridge."""
    authorized()
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config = read(PACKAGE / "config.json")
    data={r["id"]:r for r in rows(INPUTS)}
    assert len(data)==EXPECTED_CASES
    resources = read(HERE / "RESOURCE_CONTRACTS.json")
    historic={r["id"]:r for r in rows(HISTORY)}
    class NetworkEvents:
        def event(self, name, **fields): event(dict(event=name,**fields))
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config,OUT)) as budget, \
        temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"), \
        provider_dns(urlsplit(config["generation"]["base_url"]).hostname,NetworkEvents()):
        expected=dict(protocol_sha256=sha(PROTOCOL),inputs_sha256=sha(INPUTS),
            order_sha256=sha(ORDER),resources=resources,no_historical_LLM_seed=True)
        previous = cache.get("meta","identity")
        if previous is None:
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0 and not cache.all("outcome")
            cache.put("meta","identity",expected)
        else: assert previous == expected
        stop = lambda:(OUT / "STOP_REQUESTED").exists()
        pipeline = Pipeline(config)
        with closing(SubmissionAwareStrictAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop)) as strict:
            for task in rows(ORDER):
                if cache.get("outcome",task["task_id"]) is not None:
                    continue
                if stop():break
                case = data[task["id"]]
                resource = resources[case["dataset"]]["identity"]
                remote = bridge.for_dataset(case["dataset"])
                def retrieve(query):
                    request = retrieval_identity(resource,case["dataset"],case["qid"],query,10)
                    request_id = retrieval_key(request)
                    packet = cache.get("retrieval",request_id)
                    hit = packet is not None
                    if not hit:
                        packet = check_packet(request,remote.retrieve(case["qid"],query,10))
                        cache.put("retrieval",request_id,packet)
                    else:packet=check_packet(request,packet)
                    event(dict(event="logical_retrieval",task_id=task["task_id"],
                        request_id=request_id,source="current_run_exact" if hit else "remote_search",**request))
                    return deepcopy(packet)
                native = NativeAsk(cache,strict,config,task["task_id"],event=event)
                partial_nodes = []
                def node_record(node):
                    cache.put("node",task["task_id"]+"/"+str(node["step_id"]),node)
                    partial_nodes.append(deepcopy(node))
                try:
                    result = flow.execute_arm(case,retrieve,native,task["arm"],pipeline=pipeline,
                        record_node=node_record,record_event=lambda e:event(dict(task_id=task["task_id"],**e)))
                    outcome = {**task,"status":result["status"],"prediction":result["prediction"],
                        "trajectory":result,"stages":native.stages,"utc":utc(),"terminal_reason":None}
                except AlgorithmTerminal as exc:
                    outcome = {**task,"status":"algorithm_failure","prediction":"","partial_nodes":partial_nodes,
                        "stages":native.stages,"terminal_reason":exc.value["terminal_reason"],
                        "terminal_stage":exc.value["stage"],"utc":utc()}
                except (TechnicalHeld,RuntimeError,ValueError,TypeError,InterruptedError) as exc:
                    held = {**task,"status":"held_technical","error_type":type(exc).__name__,"utc":utc(),
                        "completed_stage_request_ids":[r["request_id"] for r in native.stages]}
                    atomic_json(OUT / "held" / (digest(task["task_id"])+".json"),held)
                    event(dict(event="held_technical",**held))
                    # A failed/partial query can already have abnormally high
                    # known fees. Compare to the COMPLETE historical reference;
                    # it still never enters the complete-query20 trend window.
                    check_expense(cache,budget,historic,task)
                    continue
                cache.put("outcome",task["task_id"],outcome)
                check_expense(cache,budget,historic,task)
                emit = dict(event="condition_terminal",task_id=task["task_id"],status=outcome["status"])
                event(emit);print(json.dumps(emit,ensure_ascii=True),flush=True)
    return report()


def run():
    authority = authorized()
    # A completed run returns through read-only report before SSH/deployment or
    # Cache construction; repeating CLI never allocates another GPU merely to
    # discover that every plannedcondition was already terminal.
    if (OUT/"checkpoint.sqlite3").exists():
        with closing(ro(OUT/"checkpoint.sqlite3"))as db:
            completed={r[0]for r in db.execute("SELECT key FROM data WHERE kind='outcome'")}
        if completed=={r["task_id"]for r in rows(ORDER)}:
            return report()
    # Root owns transport. Its creation is delayed until every local/auth check.
    from .remote_bridge import open_bridge
    with open_bridge(resources=read(HERE/"RESOURCE_CONTRACTS.json"),output=OUT,authorization=authority) as bridge:
        return run_with_bridge(bridge)


def report():
    check()
    if not (OUT/"checkpoint.sqlite3").exists():
        return dict(status="prepared_not_started",planned_conditions=len(rows(ORDER)),new_API_requests=0)
    with closing(ro(OUT/"checkpoint.sqlite3")) as db:
        outcomes=[json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        states=dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
        retired=dict(db.execute("SELECT status,count(*) FROM retired_requests GROUP BY status"))
    tasks=rows(ORDER);expected={r["task_id"]for r in tasks};n=len(rows(INPUTS));planned=len(tasks)
    assert len(outcomes)==len({r["task_id"] for r in outcomes}) and all(r["task_id"] in expected for r in outcomes)
    global SCORERS
    if SCORERS is None:
        from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
        SCORERS={ds:metric(ds) for ds in DATASETS}
    references={r["id"]:r for r in rows(HERE/"references20.local.jsonl")}
    details=[]
    for r in outcomes:
        native=SCORERS[r["dataset"]](r["prediction"],references[r["id"]]) if r["status"] in ("success","success_truncated") else dict(f1=0.,em=0.)
        usage={k:sum(s["usage"][k] for s in r["stages"]) for k in ("prompt_tokens","completion_tokens","total_tokens")}
        # Large trajectories remain incheckpoint; this report carries audit keys.
        detail={k:v for k,v in r.items() if k not in ("trajectory","partial_nodes")}
        details.append({**detail,**native,"native_full_correct":native["f1"]==1. or native["em"]==1.,
            "full_flow_logical_usage":usage,"full_flow_logical_estimated_CNY":cost(usage)})
    def summary(arm,ds=None):
        planned=[t for t in tasks if t["arm"]==arm and(ds is None or t["dataset"]==ds)]
        found=[d for d in details if d["arm"]==arm and(ds is None or d["dataset"]==ds)]
        n=len(planned);missing=n-len(found);f=sum(d["f1"] for d in found);e=sum(d["em"] for d in found)
        return dict(planned=n,quality_terminal=len(found),held_or_unstarted=missing,
            F1=f/n if not missing else None,EM=e/n if not missing else None,
            F1_fixed_denominator_bounds=[f/n,(f+missing)/n],EM_fixed_denominator_bounds=[e/n,(e+missing)/n],
            algorithm_failures=sum(d["status"]=="algorithm_failure" for d in found),
            native_full_correct_count=sum(d["native_full_correct"]for d in found),
            success_truncated=sum(d["status"]=="success_truncated" for d in found),
            logical_prompt_tokens=sum(d["full_flow_logical_usage"]["prompt_tokens"] for d in found),
            logical_completion_tokens=sum(d["full_flow_logical_usage"]["completion_tokens"] for d in found),
            logical_estimated_CNY=sum(d["full_flow_logical_estimated_CNY"] for d in found))
    index={(d["id"],d["arm"]):d for d in details}
    pairs=[dict(id=i,dataset=references[i]["dataset"],D0=index[i,"D0"]["f1"],D1=index[i,"D1"]["f1"],
        delta_F1=index[i,"D1"]["f1"]-index[i,"D0"]["f1"],delta_EM=index[i,"D1"]["em"]-index[i,"D0"]["em"])
        for i in references if(i,"D0")in index and(i,"D1")in index]
    historical_rows=[r for r in rows(HISTORY)if r["historical_quality_available"]]
    historic_index={r["id"]:SCORERS[r["dataset"]](r["final_prediction_reference_only"],references[r["id"]])for r in historical_rows}
    historic=list(historic_index.values())
    def matched_old(ids):
        values=[historic_index[i]for i in ids]
        return dict(n=len(values),F1=sum(v["f1"]for v in values)/len(values)if values else None,
            EM=sum(v["em"]for v in values)/len(values)if values else None)
    def matched_fresh(arm,ids):
        values=[d for d in details if d["arm"]==arm and d["id"]in ids]
        count=len(ids);missing=count-len(values);f=sum(d["f1"]for d in values);e=sum(d["em"]for d in values)
        return dict(planned=count,quality_terminal=len(values),held_or_unstarted=missing,
            F1=f/count if count and not missing else None,EM=e/count if count and not missing else None,
            F1_fixed_denominator_bounds=[f/count,(f+missing)/count]if count else None,
            EM_fixed_denominator_bounds=[e/count,(e+missing)/count]if count else None)
    matched_ids=set(historic_index)
    matched=dict(scope="Secondary same15 frozen historicalavailability subset; no scorefilter/recovery/imputation; never compare old15 as oldfull20",
        n=len(matched_ids),missing_historical=n-len(matched_ids),missing_historical_ids=sorted(set(references)-matched_ids),
        old_native=matched_old(matched_ids),D0_native=matched_fresh("D0",matched_ids),D1_native=matched_fresh("D1",matched_ids),
        by_dataset={ds:dict(n=sum(references[i]["dataset"]==ds for i in matched_ids),
            old_native=matched_old({i for i in matched_ids if references[i]["dataset"]==ds}),
            D0_native=matched_fresh("D0",{i for i in matched_ids if references[i]["dataset"]==ds}),
            D1_native=matched_fresh("D1",{i for i in matched_ids if references[i]["dataset"]==ds}))for ds in DATASETS})
    with closing(ro(read(PACKAGE/"config.json")["budget"]["ledger"])) as ledger,closing(ro(OUT/"checkpoint.sqlite3"))as checkpoint:
        namespace="p02_prompt_research_v1_20261022/"+str(OUT.resolve())
        actual,reserve=ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",(namespace,)).fetchone()
        paid,unattributed=paid_query_attribution(checkpoint,ledger,tasks,outcomes,
            {r["id"]:r for r in rows(HISTORY)},namespace)
    paidindex={r["task_id"]:r for r in paid}
    for detail in details:
        detail["actual_new_spend_first_owner_attribution"]=paidindex[detail["task_id"]]
    completed={r["task_id"] for r in outcomes}
    holds=[read(p) for p in (OUT/"held").glob("*.json")] if(OUT/"held").exists() else[]
    held_ids=sorted({h["task_id"] for h in holds if h["task_id"] not in completed})
    value=dict(status="completed"if len(outcomes)==planned else"partial",utc=utc(),
        scope="Fixed actualR02-D complete dynamic downstream RAG balanced20 development; prior723exposure disclosed, not newD/timing/formal500 or untouchedconfirmation",
        planned_conditions=planned,quality_terminal=len(outcomes),held_technical_tasks=len(held_ids),held_task_ids=held_ids,
        not_started_tasks=planned-len(outcomes)-len(held_ids),arms={a:summary(a)for a in flow.ARMS},
        by_dataset={ds:{a:summary(a,ds) for a in flow.ARMS} for ds in DATASETS},
        paired=dict(complete_pairs=len(pairs),delta_F1=sum(p["delta_F1"]for p in pairs)/n if len(pairs)==n else None,
            delta_EM=sum(p["delta_EM"]for p in pairs)/n if len(pairs)==n else None,
            improved=sum(p["delta_F1"]>0 for p in pairs),tied=sum(p["delta_F1"]==0 for p in pairs),harmed=sum(p["delta_F1"]<0 for p in pairs)),
        historical_source_reference_only=dict(scope="Strict complete public formalr000 snapshot; not a claim about all historical runs",planned=n,available=len(historic),missing=n-len(historic),
            F1=sum(r["f1"]for r in historic)/n if len(historic)==n else None,
            EM=sum(r["em"]for r in historic)/n if len(historic)==n else None,
            F1_available=sum(r["f1"]for r in historic)/len(historic)if historic else None,
            EM_available=sum(r["em"]for r in historic)/len(historic)if historic else None,missing_not_imputed=True),
        matched_historical_subset=matched,
        actual_new_returned_usage_CNY=actual,unresolved_reserved_CNY=reserve,
        per_query_new_paid_attribution=paid,unattributed_charges=unattributed,
        expense_trend=dict(unit="complete_paid_query",eligible=sum(r["eligible_for_complete_paid_query_trend"]for r in paid),
            threshold=20,enough_queries_for_window=sum(r["eligible_for_complete_paid_query_trend"]for r in paid)>=20,
            note="Only completepaidqueries; no purecache/partial/unknown stagecount proxy; sharedprefix variants labelledcostproxy"),
        request_statuses=states,retired_request_statuses=retired,scorer_factory_calls_once_per_dataset=2,
        main_evaluation_remains_paused=True)
    jsonl(OUT/"DETAILS.jsonl",details);jsonl(OUT/"PAIRED.jsonl",pairs);atomic_json(OUT/"REPORT.json",value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","freeze","dry-run","report","run"))
    p.add_argument("--selection")
    args=p.parse_args();action=args.action
    if args.selection is not None and action!="prepare":p.error("--selection is offlineprepare only")
    value=prepare(args.selection)if action=="prepare"else{"freeze":freeze,"dry-run":dry_run,"report":report,"run":run}[action]()
    print(json.dumps(value,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
