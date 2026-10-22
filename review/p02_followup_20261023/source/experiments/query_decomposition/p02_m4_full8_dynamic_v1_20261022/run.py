"""Independent M4full8 preparation; no inference without new authorization.

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
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
FORMAL = HERE.parent / "p02_formal1500_v1_20261021"
PREFLIGHT = ROOT / "outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
DATASETS = ("2wiki",)
SEED = 20261022
SCORERS = None
EXPECTED_CASES = 8
INPUTS = HERE/"INPUTS8.jsonl"
ORDER = HERE/"ORDER16.jsonl"
SELECTION = HERE/"STAGE_SELECTION8.frozen.json"
HISTORY = HERE/"HISTORICAL_REFERENCES8.local.jsonl"
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


def assemble():
    from .source_inputs import assemble as build
    return build()


def order(data):
    # Rotate by case index with a seed-determined start; balanced starts for an
    # evenN (4M0-first/4M4-first here). No score/label influences inputs/order.
    start = int(hashlib.sha256(str(SEED).encode()).hexdigest(), 16) % 2
    return [dict(index=i*2+j, id=r["id"], dataset=r["dataset"], arm=a, task_id=r["id"]+"/"+a)
        for i, r in enumerate(data) for j, a in enumerate(flow.ARMS[(i+start)%2:]+flow.ARMS[:(i+start)%2])]


def prepare(selection_path=None):
    if PROTOCOL.exists():return dry_run()
    assert selection_path is None,"Source is exactly frozenM4stage8, no new cohort option"
    assert not(OUT/"checkpoint.sqlite3").exists()and not(HERE/"RUN_AUTHORIZATION.json").exists()
    data,costs,proof,refs,resources,proxies,publicproof=assemble()
    jsonl(INPUTS,data);jsonl(HISTORY,costs);jsonl(HERE/"SOURCE_PROVENANCE8.jsonl",proof)
    jsonl(HERE/"references8.local.jsonl",refs);jsonl(ORDER,order(data))
    atomic_json(HERE/"HISTORICAL_COST_PROXIES.local.json",proxies)
    atomic_json(HERE/"PUBLIC_INPUT_PROVENANCE.json",publicproof)
    atomic_json(HERE/"RESOURCE_CONTRACTS.json",resources)
    from .source_inputs import STAGE
    atomic_json(HERE/"SELECTION_SOURCE.json",dict(source=rel(STAGE/"INPUTS8.frozen.jsonl"),source_sha256=sha(STAGE/"INPUTS8.frozen.jsonl"),rule="All8 frozenstagequeries;no new outcome or dateprecision selection"))
    (HERE/"STAGE_SELECTION8.frozen.json").write_bytes((STAGE/"SELECTION.frozen.json").read_bytes())
    base=HERE.parent/"p02_live_target_balanced20_dev_v1_20261022"
    components=[]
    for name in("native_ask.py","resource_contract.py","retrieval_bridge.py","remote_worker.py","QUERY_MODEL.frozen.json"):
        assert sha(base/name)==sha(HERE/name)
        components.append(dict(component=name,source=rel(base/name),source_sha256=sha(base/name)))
    atomic_json(HERE/"REUSED_COMPONENTS.json",dict(components=components))
    return dict(state="offline_prepared_not_frozen_not_authorized",planned_cases=8,conditions=16,
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"])for r in data),historical_complete_quality_coverage=8,new_API_GPU_SSH_requests=0)


def freeze():
    if PROTOCOL.exists():return dry_run()
    prepare()
    assert read(HERE/"OFFLINE_CONTRACTS.json")["passed"]is True
    assert read(HERE/"REMOTE_OFFLINE_CONTRACTS.json")["passed"]is True
    assert not(HERE/"RUN_AUTHORIZATION.json").exists()and not(OUT/"checkpoint.sqlite3").exists()
    from .source_inputs import STAGE
    stage_protocol=read(STAGE/"PROTOCOL.frozen.json")
    pins={**stage_protocol["pins"],rel(STAGE/"PROTOCOL.frozen.json"):sha(STAGE/"PROTOCOL.frozen.json")}
    extras=[*HERE.glob("*.py"),*HERE.glob("*.jsonl"),HERE/"PROTOCOL.md",HERE/"OFFLINE_CONTRACTS.json",
        HERE/"REMOTE_OFFLINE_CONTRACTS.json",HERE/"QUERY_MODEL.frozen.json",HERE/"RESOURCE_CONTRACTS.json",
        HERE/"SELECTION_SOURCE.json",HERE/"STAGE_SELECTION8.frozen.json",HERE/"REUSED_COMPONENTS.json",
        HERE/"HISTORICAL_COST_PROXIES.local.json",HERE/"PUBLIC_INPUT_PROVENANCE.json",
        STAGE/"__init__.py",STAGE/"selection.py",
        HERE.parent/"robustness_models_retrievers_v1_20261017/remote_resources.py",PREFLIGHT,
        ROOT/"src/indexes/streaming_flat_index.py",ROOT/"src/embedders/text_embedder.py",
        ROOT/"src/persistence/artifact_io.py",ROOT/"src/provenance.py",
        HERE.parent/"comprehensive_eval_v1_20261015/scoring.py",HERE.parent/"comprehensive_eval_v1_20261015/common.py",
        HERE.parent/"hotpot_formal_eval_v1/common/scoring.py",
        ROOT/"data/multihop_comprehensive_v1_20261015/2wiki/official/2wikimultihop_evaluate_v1.1.py",
        ROOT/"data/multihop_comprehensive_v1_20261015/2wiki/native/id_aliases.json",
        PACKAGE/"prompts/final_examples.json",FORMAL/"references1500.local.frozen.jsonl"]
    for component in read(HERE/"REUSED_COMPONENTS.json")["components"]:
        original=ROOT/component["source"]
        assert sha(original)==component["source_sha256"]==sha(HERE/component["component"])
        extras.append(original)
    extras +=[ROOT/r["public_runtime"]for r in rows(HERE/"SOURCE_PROVENANCE8.jsonl")]
    pins.update({rel(p):sha(p)for p in extras})
    pins.update({rel(p):sha(p)for p in PACKAGE.rglob("*.py")if"__pycache__"not in str(p)})
    atomic_json(PROTOCOL,dict(experiment=HERE.name,revision=1,prepared_utc=utc(),state="frozen_not_authorized",
        planned_queries=8,planned_conditions=16,dataset_counts={"2wiki":8},ordinary_nodes_per_arm=17,
        cohort="Exactly8 frozenM4stagequeries;allpriorsemanticallyexposed;selectionincludesoldfullscoreandnonfullstrata;notunseenconfirmation",
        input_scope="Only5fields;actualformalR02/v12 rawDfixed;source/publicruntime/oldsame8referenceexact;no stage outputs or gold enter inference",
        comparison="M0 freshnativeinterleave;M4 same-sourceorderedcaseword+punctuationformatduplicate fill at ALLordinaryandgoalmerge calls",
        implementation="StaticisolatednativePipeline.execute copy:onlypacket=merge line differs;M0 all8 realwire/trajectoryreplay andnativefakeequivalence",
        preserved="RawCurrent/history/rewriteacceptance/allpromptinstructions/fewshots/goalrule/QA128/<4wordsformatter128/rootoriginal/final64validlength",
        controls="AllfreshseedNone;no historicalLLM/retrievalcache import;onlyexactpayload/currentnamespace sharing with first-owner newcharges",
        downstream="Allown-historyQA/rewrite/newqueryretrieval/formatter/goal/finalrecomputed;remote A6000BGEqueryencoder+CPUexactFAISS samefrozenindex",
        runtime=dict(shared_gate=True,gate_interval_seconds=3,SDK_HTTP_retries=0,maximum_actual_attempts_including_restart=5,
            unknown="Returnevidenceorreturnedledgerheld;no-returnboundedrecovery;retain retiredreceipts/reservations",
            algorithm="Complete model/all3intusage:QA/rewrite/formatter length/empty/refusal/filter terminal0 without reroll",
            final="Nativecomplete attributable nonempty no-refusal length is success_truncated and scored actual text",
            technical="Missingusage/modelmismatch/providerblock/exhaustion held;separate fixedN bounds",no_fixed_budget_cap=True),
        metrics="OfficialnativeF1/EM fixedN8perarm;oldsame8reference/freshM0/freshM4separate;pairedgains/ties/harms;nativeF1orEM1fullcorrectseparatefromsemantics",
        blind_interpretation="Lockmaskedfinaltextdifferences before endpoint interpretation;no automatic promotion from stage dateprecision gain",
        expenses="Logicalfullflow tokens/cost;distinct actual paidattempts/unknown separate;3xcompletequeryreference alert;20querytrend cannottriggerfrom16flows;notstageproxy",
        resources="Oldportablemanifest/model/dependency hashes pinned;oldpreflightnotcurrentreadiness;fresh approvedbridge preflight rechecks beforeGPU",
        remote_isolation="m4full8_protocolhash16;onlyquery/qid/resourceidentitycrossSSH;noAPIkey,history,gold,modelmessages;deploymentlock+durablesubmissionguard;finishedworkernewmissheld",
        authorization="NoAUTH/API/GPU at preparation;root reviewsstage16lockedsemantics+integrity(no clearregression andsubstantivenecessaryinformationrecovery) then separately authorizes exactprotocolhash+decision",
        main_evaluation_remains_paused=True,protected_holds=stage_protocol["protected_holds"],pins=pins))
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
        ("SOURCE_PROVENANCE8.jsonl",proof),("references8.local.jsonl",refs),(ORDER.name,order(data))):
        assert rows(HERE / filename) == expected, filename
    assert read(HERE / "RESOURCE_CONTRACTS.json") == resources
    assert read(HERE/"HISTORICAL_COST_PROXIES.local.json")==proxies
    assert read(HERE/"PUBLIC_INPUT_PROVENANCE.json")==publicproof
    assert all(set(c)=={"id","qid","dataset","question","supporting_questions"} for c in data)
    result=dict(passed=True,state="frozen_prepared",cases=len(data),planned_conditions=len(data)*len(flow.ARMS),
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"]) for r in data), source_and_actual_R02_arrays_exact=True,
        no_new_stage_or_full8_outputs_used_for_preparation=True, inference_refs_forbidden=True,
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
    references={r["id"]:r for r in rows(HERE/"references8.local.jsonl")}
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
    pairs=[dict(id=i,dataset=references[i]["dataset"],M0=index[i,"M0"]["f1"],M4=index[i,"M4"]["f1"],
        delta_F1=index[i,"M4"]["f1"]-index[i,"M0"]["f1"],delta_EM=index[i,"M4"]["em"]-index[i,"M0"]["em"])
        for i in references if(i,"M0")in index and(i,"M4")in index]
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
    matched=dict(scope="Same8 frozen source historical references; noqualityselection;fresh controls separate from oldnonconcurrent",
        n=len(matched_ids),missing_historical=n-len(matched_ids),missing_historical_ids=sorted(set(references)-matched_ids),
        old_native=matched_old(matched_ids),M0_native=matched_fresh("M0",matched_ids),M4_native=matched_fresh("M4",matched_ids),
        by_dataset={ds:dict(n=sum(references[i]["dataset"]==ds for i in matched_ids),
            old_native=matched_old({i for i in matched_ids if references[i]["dataset"]==ds}),
            M0_native=matched_fresh("M0",{i for i in matched_ids if references[i]["dataset"]==ds}),
            M4_native=matched_fresh("M4",{i for i in matched_ids if references[i]["dataset"]==ds}))for ds in DATASETS})
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
        scope="Same8 exposedstagequeries;complete dynamic M0 originalmerge vs M4 readerselector;rawR02 fixed;notformal500/unseenconfirmation",
        planned_conditions=planned,quality_terminal=len(outcomes),held_technical_tasks=len(held_ids),held_task_ids=held_ids,
        not_started_tasks=planned-len(outcomes)-len(held_ids),arms={a:summary(a)for a in flow.ARMS},
        by_dataset={ds:{a:summary(a,ds) for a in flow.ARMS} for ds in DATASETS},
        paired=dict(complete_pairs=len(pairs),delta_F1=sum(p["delta_F1"]for p in pairs)/n if len(pairs)==n else None,
            delta_EM=sum(p["delta_EM"]for p in pairs)/n if len(pairs)==n else None,
            improved=sum(p["delta_F1"]>0 for p in pairs),tied=sum(p["delta_F1"]==0 for p in pairs),harmed=sum(p["delta_F1"]<0 for p in pairs)),
        historical_source_reference_only=dict(scope="Exact8 public formal source cases selected by priorstage protocol (r000/r100); nonconcurrent reference only",planned=n,available=len(historic),missing=n-len(historic),
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
        request_statuses=states,retired_request_statuses=retired,scorer_factory_calls_once_per_dataset=len(DATASETS),
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
