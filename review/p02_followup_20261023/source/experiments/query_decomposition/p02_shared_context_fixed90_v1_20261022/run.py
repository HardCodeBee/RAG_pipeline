"""Independent fixed90 reset/flat/chat preparation; no inference without new authorization.

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
from .message_transport import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE
from . import flow, resource_contract
from .native_ask import NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld, usage_ok, stage_success, terminal_reason
from .retrieval_bridge import identity as retrieval_identity, key as retrieval_key, check_packet

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
FORMAL = HERE.parent / "p02_formal1500_v1_20261021"
PREFLIGHT = ROOT / "outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
DATASETS = ("hotpotqa", "2wiki", "musique")
SEED = 20261022
SCORERS = None
EXPECTED_CASES = 90
INPUTS = HERE/"INPUTS90.jsonl"
ORDER = HERE/"ORDER270.jsonl"
HISTORY = HERE/"HISTORICAL_REFERENCES90.local.jsonl"
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


def initial_ids():
    from .source_inputs import fixed_sample_ids
    value=fixed_sample_ids()
    assert len(value)==6 and len(set(value))==6
    return value


def order(data):
    index={row["id"]:row for row in data}
    ids=initial_ids(); assert all(i in index for i in ids)
    arranged=[index[i] for i in ids]+[row for row in data if row["id"] not in set(ids)]
    start=int(hashlib.sha256(str(SEED).encode()).hexdigest(),16)%len(flow.ARMS)
    result=[]
    for i,row in enumerate(arranged):
        offset=(i+start)%len(flow.ARMS)
        for j,arm in enumerate(flow.ARMS[offset:]+flow.ARMS[:offset]):
            result.append(dict(index=i*len(flow.ARMS)+j,id=row["id"],dataset=row["dataset"],
                arm=arm,task_id=row["id"]+"/"+arm,engineering_initial=row["id"]in set(ids)))
    return result


def prepare(selection_path=None):
    if PROTOCOL.exists():return dry_run()
    assert selection_path is None,"Same fixed90; no cohort substitution"
    assert not(OUT/"checkpoint.sqlite3").exists()and not(HERE/"RUN_AUTHORIZATION.json").exists()
    data,costs,proof,refs,resources,proxies,publicproof=assemble()
    for path,values in ((INPUTS,data),(HISTORY,costs),(HERE/"SOURCE_PROVENANCE90.jsonl",proof),
        (HERE/"references90.local.jsonl",refs)):
        assert rows(path)==values,path
    assert read(HERE/"RESOURCE_CONTRACTS.json")==resources
    assert read(HERE/"HISTORICAL_COST_PROXIES.local.json")==proxies
    assert read(HERE/"PUBLIC_INPUT_PROVENANCE.json")==publicproof
    assert len(data)==90 and Counter(r["dataset"]for r in data)=={ds:30 for ds in DATASETS}
    jsonl(ORDER,order(data))
    base=HERE.parent/"p02_m4_wiki20_dynamic_dev_v1_20261022"
    components=[]
    for name in ("remote_worker.py","QUERY_MODEL.frozen.json"):
        assert sha(base/name)==sha(HERE/name)
        components.append(dict(component=name,source=rel(base/name),source_sha256=sha(base/name),byte_identical=True))
    atomic_json(HERE/"REUSED_COMPONENTS.json",dict(components=components,
        changed_isolated_modules="flow/context use unchanged core Pipeline; message-aware transport; retrieval/resource allowlist adds MuSiQue only; independent90 driver/bridge"))
    return dict(state="offline_prepared_not_frozen_not_authorized",planned_cases=90,conditions=270,
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"])for r in data),
        goal_only_queries=sum(not r["supporting_questions"]for r in data),
        engineering_cases=6,engineering_conditions=18,new_API_GPU_SSH_requests=0)


def freeze():
    if PROTOCOL.exists():return dry_run()
    prepare()
    for name in ("OFFLINE_CONTRACTS.json","TRANSPORT_CONTRACTS.json","REMOTE_OFFLINE_CONTRACTS.json"):
        assert read(HERE/name)["passed"]is True,name
    assert not(HERE/"RUN_AUTHORIZATION.json").exists()and not(OUT/"checkpoint.sqlite3").exists()
    from . import source_inputs
    # Bind the prior native fixed90 sources and all imported transport dependencies.
    priorpath=HERE.parent/"p02_goal_native_role_fixed90_v1_20261022/PROTOCOL.frozen.json"
    prior=read(priorpath)
    pins={**prior["pins"],rel(priorpath):sha(priorpath)}
    hold_source=HERE.parent/"p02_m4_wiki20_dynamic_dev_v1_20261022/PROTOCOL.frozen.json"
    holds=read(hold_source)["protected_holds"]
    assert len(holds)==3 and all(sha(ROOT/path)==expected for path,expected in holds.items())
    pins[rel(hold_source)]=sha(hold_source)
    extras=[*HERE.glob("*.py"),*HERE.glob("*.jsonl"),HERE/"PROTOCOL.md",*HERE.glob("*.json")]
    extras.extend(source_inputs.pinned_paths())
    for name in ("L_INPUTS","W_INPUTS","CONTROL","REPLAY","LINEAGE","LINEAGE_SUMMARY","MANIFEST","PRIOR_PUBLIC","PREFLIGHT","BASE"):
        path=getattr(source_inputs,name,None)
        if isinstance(path,Path)and path.is_file():extras.append(path)
    for path in PACKAGE.rglob("*"):
        if path.is_file()and path.suffix in(".py",".json",".txt"):extras.append(path)
    extras.extend([ROOT/"scripts/cli_support.py",FORMAL/"network.py",
        HERE.parent/"p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        ROOT/"src/indexes/streaming_flat_index.py",ROOT/"src/embedders/text_embedder.py",
        ROOT/"src/persistence/artifact_io.py",ROOT/"src/provenance.py",
        HERE.parent/"robustness_models_retrievers_v1_20261017/remote_resources.py"])
    def dependencies(value):
        if isinstance(value,dict):
            for field,hashfield in (("path","sha256"),("source","source_sha256"),("raw_file","raw_sha256"),("lineage_source","lineage_source_sha256")):
                if isinstance(value.get(field),str)and isinstance(value.get(hashfield),str):
                    path=(ROOT/value[field]).resolve()
                    if path.is_relative_to(ROOT)and path.is_file():
                        assert sha(path)==value[hashfield],value[field];extras.append(path)
            for child in value.values():dependencies(child)
        elif isinstance(value,list):
            for child in value:dependencies(child)
    for path in (HERE/"PUBLIC_INPUT_PROVENANCE.json",HERE/"INITIAL_ENGINEERING6.frozen.json",HERE/"SOURCE_PROVENANCE90.jsonl"):
        dependencies(rows(path)if path.suffix==".jsonl"else read(path))
    for row in rows(HISTORY):
        if row["historical_quality_available"]:extras.append(ROOT/row["source"])
    for component in read(HERE/"REUSED_COMPONENTS.json")["components"]:extras.append(ROOT/component["source"])
    pins.update({rel(path):sha(path)for path in extras if path.name!="PROTOCOL.frozen.json"and path.name!="RUN_AUTHORIZATION.json"})
    data=rows(INPUTS)
    atomic_json(PROTOCOL,dict(experiment=HERE.name,revision=1,state="frozen_not_authorized",prepared_utc=utc(),
        planned_queries=90,planned_conditions=270,dataset_counts={ds:30 for ds in DATASETS},arms=list(flow.ARMS),
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"])for r in data),goal_only_queries=sum(not r["supporting_questions"]for r in data),
        cohort="Exact same historical W/L fixed90; oldactual R02/v12 arrays, not manual60 or formal1500 generation; reused exposed development cohort",
        context="reset exact original current prompt; flat previous raw prompts+accepted answers labelledUser/Assistant then current; chat same content as alternating roles. No previous expanded wire nested; per-original-query isolation",
        unchanged="Unmodified core Pipeline.execute; original current child prompt/allhistory/fourfewshots/native merge; rewrite256 formatter128 final64 remain independent strings; QAandgoal128; no M4/ROLE/O1/D1",
        accepted_history="record_node succeeds then commit accepted answer AFTER optional<4wordformatter; raw initial stored separately, formatter payload original",
        growth="Everyrawturnretained, including its native priorQA and examples; possible quadratic repetition explicitly measured; not recursive/exponential replay; all prefixusage charged",
        context_guard="UTF8contentbytes+64/message+max_tokens<=128000 before cache/provider; overlimit technicalheld/no truncation; frozen low-input-price research range, not monetarycap/modelwindowclaim",
        engineering_gate="Prespecified first6 HP2Wiki2Mu2/18conditions; validates realwire/receipt/contextisolation only. No prediction/F1 inspected; valid algorithmterminals not screened out. Technicalheld or contractfail pauses before remaining84 for engineering review",
        stopping="After engineeringpass automatically continue fixedremaining84; no quality-based small-sample stop or candidate tuning. Report all270 outcomes; no automatic mainpromotion",
        fresh="seedNone;no historicalLLM/retrievalseeding; same-run exact fullpayload/resourcequery shared andpaidonce",
        runtime="SDK0/HTTP0/globalgate3s/max5durabletechnical/return+ledgerguard/unknownreserves;algorithmfailure0/noqualityretry;technicalheldbounds;nativefinallegalnonemptylengthaccepted",
        metrics="Official scorer once per3datasets; fixed90F1EM plus reset-flat/reset-chat/flat-chat paired comparisons; oldsame90 separate from freshreset; no hidden semanticlabels/as-ifformal claims",
        expense="Logicalentireprefix prompt/completiontokens andexpense vs actualfirstownercharges; historicalcompletequery3x/recent20paidquery1.5x alerts retained despiteexpectedgrowth;unknownseparate;noLLMcall/timebenchmark",
        remote="NEWcontext90_protocolhash16, one A6000query-only/4CPU32Gmax4h idle30m;onlyquery/qid/resource toworker;threefrozenresources;unknownsubmissionheld/no silentGPUrestart",
        authorization="OfflineonlyuntilrootexacthashAUTH+decision;mainformal500/timing/baselines/publication remainpaused",
        main_evaluation_remains_paused=True,protected_holds=holds,pins=pins))
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
    for filename,expected in((INPUTS.name,data),(HISTORY.name,costs),("SOURCE_PROVENANCE90.jsonl",proof),
        ("references90.local.jsonl",refs),(ORDER.name,order(data))):
        assert rows(HERE/filename)==expected,filename
    assert read(HERE/"RESOURCE_CONTRACTS.json")==resources
    assert read(HERE/"PUBLIC_INPUT_PROVENANCE.json")==publicproof
    assert read(HERE/"HISTORICAL_COST_PROXIES.local.json")==proxies
    assert all(set(row)=={"id","qid","dataset","question","supporting_questions"}for row in data)
    tasks=rows(ORDER);assert len(tasks)==270 and len({t["task_id"]for t in tasks})==270
    assert all(t["engineering_initial"]for t in tasks[:18])and not any(t["engineering_initial"]for t in tasks[18:])
    result=dict(passed=True,state="frozen_prepared",cases=90,planned_conditions=270,
        engineering_initial_cases=6,engineering_initial_conditions=18,
        ordinary_nodes_per_arm=sum(len(r["supporting_questions"])for r in data),
        source_and_actual_R02_arrays_exact=True,new_API_GPU_SSH=0,protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT/"DRYRUN.json",result)
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


def engineering_gate(cache, tasks, config):
    """Inspect wire/receipt invariants only; never inspect native scores/gold.

    Legitimate algorithm terminal outcomes remain zero and do not fail this
    engineering gate. Missing technical receipts or message corruption pause
    before the remaining84; no question is replaced or quality rerolled.
    """
    from .context import Conversation, digest as context_digest, messages
    from p02_multigrain_timing_cachefix_v1.core.child_prompts import CHILD_INSTRUCTIONS
    selected=[task for task in tasks if task["engineering_initial"]]
    outcomes=cache.all("outcome")
    missing=[task["task_id"]for task in selected if task["task_id"]not in outcomes]
    if missing:
        return dict(passed=False,status="engineering_incomplete_technical_review_required",
            planned_conditions=len(selected),terminal_conditions=len(selected)-len(missing),
            missing_task_ids=missing,quality_or_predictions_inspected=False)
    inputs={row["id"]:row for row in rows(INPUTS)}
    checked_stages=checked_QA=0
    try:
        for task in selected:
            result=outcomes[task["task_id"]]
            terminal_stages=[]
            for stage in result["stages"]:
                row=cache.db.execute("SELECT payload,status,result FROM requests WHERE id=?",(stage["request_id"],)).fetchone()
                assert row is not None
                payload,receipt=json.loads(row[0]),json.loads(row[2])
                receipt={**receipt,"status":row[1]}
                assert digest(payload)==stage["payload_sha256"]
                assert payload==build_payload(config["generation"],payload["messages"],payload["max_tokens"])
                success=stage_success(receipt,config["generation"]["model"],stage["stage"])
                reason=None if success else terminal_reason(receipt,config["generation"]["model"])
                assert success or reason
                if reason:terminal_stages.append((stage["stage"],reason))
                checked_stages+=1
            if result["status"]=="algorithm_failure":
                assert terminal_stages==[(result["terminal_stage"],result["terminal_reason"])]
            else:assert not terminal_stages
            trajectory=result.get("trajectory")or{}
            nodes=trajectory.get("nodes",result.get("partial_nodes",[]))
            nodeindex={node["step_id"]:node for node in nodes}
            conversation=Conversation(task["arm"])
            case=inputs[task["id"]]
            qa=sorted((value for key,value in cache.all("qa_input").items()if key.startswith(task["task_id"]+"/")),
                key=lambda value:int(value["stage"].rsplit("/",1)[1]))
            assert qa or result["status"]=="algorithm_failure"
            for value in qa:
                sid=int(value["stage"].rsplit("/",1)[1]);raw=value["raw_current_prompt"]
                assert conversation.render(sid,raw)==value["wire_input"]
                assert context_digest(messages(value["wire_input"]))==value["wire_messages_sha256"]
                if sid<=len(case["supporting_questions"]):
                    assert raw.count("\n\nActual task:\n")==1
                    actual=raw.split("\n\nActual task:\n",1)[1]
                else:
                    # Native goal has no worked examples/Actual-task delimiter.
                    prefix=CHILD_INSTRUCTIONS+"\n\n"
                    assert raw.startswith(prefix)
                    actual=raw[len(prefix):]
                assert actual.startswith("Original question:\n"+case["question"]+"\n\nRelevant prior question-answer pairs:\n")
                question=(case["supporting_questions"]+[case["question"]])[sid-1]
                assert actual.count(f"Current subquestion #{sid}:\n{question}\n\nRetrieved passages:\n")==1
                if sid in nodeindex:
                    node=nodeindex[sid]
                    assert node["request"]==question and node["child_prompt"]==raw
                    assert node["shared_context"]["wire_input"]==value["wire_input"]
                    conversation.commit(node)
                else:
                    assert result["status"]=="algorithm_failure"
                    assert result["terminal_stage"]in(value["stage"],f"legacy/sentence_format/{sid}")
                    assert sid==int(qa[-1]["stage"].rsplit("/",1)[1])
                checked_QA+=1
        return dict(passed=True,status="engineering_contracts_passed_continue_fixed84",
            planned_cases=6,planned_conditions=len(selected),terminal_conditions=len(selected),
            stages_checked=checked_stages,QA_contexts_checked=checked_QA,
            quality_or_predictions_inspected=False,missing_task_ids=[])
    except (AssertionError,KeyError,TypeError,ValueError)as exc:
        return dict(passed=False,status="engineering_contract_failure_review_required",
            error_type=type(exc).__name__,quality_or_predictions_inspected=False,
            stages_checked=checked_stages,QA_contexts_checked=checked_QA)


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
            tasks=rows(ORDER)
            engineering_passed=False
            for task in tasks:
                if cache.get("outcome",task["task_id"]) is not None:
                    continue
                if stop():break
                if not task["engineering_initial"]and not engineering_passed:
                    gate=engineering_gate(cache,tasks,config)
                    atomic_json(OUT/"ENGINEERING_GATE.json",gate)
                    if not gate["passed"]:
                        event(dict(event="engineering_gate_paused",**gate))
                        break
                    engineering_passed=True
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
                def context_event(value):
                    value=deepcopy(value)
                    if value["event"]=="shared_context_request":
                        key=task["task_id"]+"/"+value["stage"]
                        previous=cache.get("qa_input",key)
                        assert previous is None or previous==value,"QA context changed on resume"
                        cache.put("qa_input",key,value)
                        # Full strings remain in checkpoint and exact request,
                        # not progress logs. The hash/size is enough to monitor.
                        value.pop("raw_current_prompt");value.pop("wire_input")
                    event(dict(task_id=task["task_id"],**value))
                try:
                    result = flow.execute_arm(case,retrieve,native,task["arm"],pipeline=pipeline,
                        record_node=node_record,record_event=context_event)
                    outcome = {**task,"status":result["status"],"prediction":result["prediction"],
                        "trajectory":result,"stages":native.stages,"utc":utc(),"terminal_reason":None}
                except AlgorithmTerminal as exc:
                    outcome = {**task,"status":"algorithm_failure","prediction":"","partial_nodes":partial_nodes,
                        "stages":native.stages,"terminal_reason":exc.value["terminal_reason"],
                        "terminal_stage":exc.value["stage"],"utc":utc()}
                except (TechnicalHeld,RuntimeError,ValueError,TypeError,InterruptedError) as exc:
                    held = {**task,"status":"held_technical","error_type":type(exc).__name__,"utc":utc(),
                        "completed_stage_request_ids":[r["request_id"] for r in native.stages]}
                    if getattr(exc,"estimated_units",None)is not None:
                        held.update(estimated_context_units=exc.estimated_units,maximum_context_units=exc.maximum_units)
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
    # discover that every planned condition was already terminal.
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
    references={r["id"]:r for r in rows(HERE/"references90.local.jsonl")}
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
    comparisons=(("flat_minus_reset","flat","reset"),("chat_minus_reset","chat","reset"),("chat_minus_flat","chat","flat"))
    pairs=[dict(id=i,dataset=references[i]["dataset"],comparison=label,
        reference_arm=control,candidate_arm=candidate,reference_F1=index[i,control]["f1"],candidate_F1=index[i,candidate]["f1"],
        delta_F1=index[i,candidate]["f1"]-index[i,control]["f1"],delta_EM=index[i,candidate]["em"]-index[i,control]["em"])
        for label,candidate,control in comparisons for i in references if(i,control)in index and(i,candidate)in index]
    def paired_summary(label,ds=None):
        found=[row for row in pairs if row["comparison"]==label and(ds is None or row["dataset"]==ds)]
        count=sum(row["dataset"]==ds for row in references.values())if ds else n
        missing=count-len(found);f=sum(row["delta_F1"]for row in found);e=sum(row["delta_EM"]for row in found)
        return dict(planned_pairs=count,complete_pairs=len(found),missing_pairs=missing,
            delta_F1=f/count if not missing else None,delta_EM=e/count if not missing else None,
            available_delta_F1=f/len(found)if found else None,available_delta_EM=e/len(found)if found else None,
            delta_F1_fixed_denominator_bounds=[(f-missing)/count,(f+missing)/count],
            improved=sum(row["delta_F1"]>0 for row in found),tied=sum(row["delta_F1"]==0 for row in found),harmed=sum(row["delta_F1"]<0 for row in found))
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
    matched=dict(scope="Same frozen historicalavailability subset; oldhistorical/nonconcurrent versus all3fresh separately; primaryfreshchat/reset/flat",
        n=len(matched_ids),missing_historical=n-len(matched_ids),missing_historical_ids=sorted(set(references)-matched_ids),
        old_native=matched_old(matched_ids),fresh={arm:matched_fresh(arm,matched_ids)for arm in flow.ARMS},
        by_dataset={ds:dict(n=sum(references[i]["dataset"]==ds for i in matched_ids),
            old_native=matched_old({i for i in matched_ids if references[i]["dataset"]==ds}),
            fresh={arm:matched_fresh(arm,{i for i in matched_ids if references[i]["dataset"]==ds})for arm in flow.ARMS})for ds in DATASETS})
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
        scope="Same exposedfixed90 actualR02; freshreset/flat/chat complete dynamic contextcomparison; ordinary+goal contextonly; notformal500 or newsharedbaselinev2",
        planned_conditions=planned,quality_terminal=len(outcomes),held_technical_tasks=len(held_ids),held_task_ids=held_ids,
        not_started_tasks=planned-len(outcomes)-len(held_ids),arms={a:summary(a)for a in flow.ARMS},
        by_dataset={ds:{a:summary(a,ds) for a in flow.ARMS} for ds in DATASETS},
        paired={label:paired_summary(label)for label,_,_ in comparisons},
        paired_by_dataset={ds:{label:paired_summary(label,ds)for label,_,_ in comparisons}for ds in DATASETS},
        engineering_gate=read(OUT/"ENGINEERING_GATE.json")if(OUT/"ENGINEERING_GATE.json").exists()else None,
        historical_source_reference_only=dict(scope="Exact fixed90 native source/control samefields/payload/model3usage/retrieval; no scoreselection; nonconcurrent",planned=n,available=len(historic),missing=n-len(historic),
            F1=sum(r["f1"]for r in historic)/n if len(historic)==n else None,
            EM=sum(r["em"]for r in historic)/n if len(historic)==n else None,
            F1_available=sum(r["f1"]for r in historic)/len(historic)if historic else None,
            EM_available=sum(r["em"]for r in historic)/len(historic)if historic else None,missing_not_imputed=True),
        historical_reference_sources=[dict(id=r["id"],reference_source=r.get("reference_source"),reference_round=r.get("reference_round"),
            reference_kind=r["reference_kind"],historical_quality_available=r["historical_quality_available"])for r in rows(HISTORY)],
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
