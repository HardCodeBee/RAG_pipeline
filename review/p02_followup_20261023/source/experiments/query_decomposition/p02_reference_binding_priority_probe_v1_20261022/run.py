"""Fresh independent ordinary-QA probes; only exact root-authorized run submits.

Offline prepare/validate do not open a provider, Cache, SSH or GPU. No goal,
retrieval, decomposition or final-answer generation occurs in this package.
"""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from urllib.parse import urlsplit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical,utc
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget,atomic_json,INPUT_RATE,OUTPUT_RATE
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.message_transport import build_payload
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import NativeAsk,SubmissionAwareStrictAsk,AlgorithmTerminal,TechnicalHeld
from . import templates
from .inputs import source

OUT=ROOT/"outputs/query_decomposition"/HERE.name
PACKAGE=ROOT/"src/p02_multigrain_timing_cachefix_v1"
PLAN=HERE.parent/"unified_rag_eval_v1_20261021"
CONTEXT=HERE.parent/"p02_shared_context_fixed90_v1_20261022"
CONTEXT_OUT=ROOT/"outputs/query_decomposition"/CONTEXT.name
INPUTS=HERE/"INPUTS.frozen.jsonl"
ORDER=HERE/"TASK_ORDER.frozen.jsonl"
PROTOCOL=HERE/"PROTOCOL.frozen.json"
CAP,SEED=128,20261022


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):return [json.loads(line)for line in Path(path).read_text(encoding="utf-8-sig").splitlines()if line.strip()]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def rel(path):return str(Path(path).resolve().relative_to(ROOT)).replace("\\","/")


def jsonl(path,values):Path(path).write_text("".join(json.dumps(row,ensure_ascii=False)+"\n"for row in values),encoding="utf8")


def ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)


def cost(usage):return usage["prompt_tokens"]*INPUT_RATE+usage["completion_tokens"]*OUTPUT_RATE


def task_order(items):
    result=[]
    assert len(items)==38 and all(item["is_stress"]for item in items[:8])and not any(item["is_stress"]for item in items[8:])
    for index,item in enumerate(items):
        offset=int(hashlib.sha256(f"{SEED}|{item['case_id']}|reference_binding_priority_order".encode()).hexdigest(),16)%4
        for arm in templates.ARMS[offset:]+templates.ARMS[:offset]:
            result.append(dict(index=len(result),task_id=item["probe_id"]+"/"+arm,probe_id=item["probe_id"],
                case_id=item["case_id"],dataset=item["dataset"],arm=arm,is_stress=item["is_stress"]))
    return result


def prepare():
    if PROTOCOL.exists():return dry_run()
    assert not(HERE/"RUN_AUTHORIZATION.json").exists()and not(OUT/"checkpoint.sqlite3").exists()
    items=source.assemble();assert rows(INPUTS)==items
    config=read(PACKAGE/"config.json")
    jsonl(ORDER,task_order(items))
    atomic_json(HERE/"GENERATION.frozen.json",config["generation"])
    jsonl(HERE/"MAPPINGS.frozen.jsonl",[dict(probe_id=item["probe_id"],candidate_bindings=templates.bindings(item))for item in items])
    jsonl(HERE/"PROMPT_CONTRACTS.frozen.jsonl",[dict(probe_id=item["probe_id"],arm=arm,
        payload_sha256=digest(build_payload(config["generation"],templates.prompt(item,arm),CAP)))for item in items for arm in templates.ARMS])
    return dict(state="offline_prepared_not_authorized",nodes=38,planned_conditions=152,stress_initial_conditions=32,
        deferred_conditions=120,new_API_SSH_GPU_requests=0)


def validate():
    if not PROTOCOL.exists():prepare()
    items=source.assemble();assert rows(INPUTS)==items and rows(ORDER)==task_order(items)
    config=read(PACKAGE/"config.json");contracts=[];mapping_nodes=0;mapping_count=0
    for item in items:
        prefix,actual,marker=templates.parts(item)
        assert templates.prompt(item,"C0")==item["control_prompt"]
        bindings=templates.bindings(item);mapping_nodes+=bool(bindings);mapping_count+=len(bindings)
        history={row["step_id"]:row for row in item["history"]}
        for binding in bindings:
            start,end=binding["raw_span"];assert item["current_question"][start:end]==binding["raw_text"]
            start,end=binding["rewrite_span"];assert item["accepted_rewrite"][start:end]==binding["rewrite_text"]
            assert binding["candidate_text"]in[history[s["step_id"]]["answer"][s["answer_span"][0]:s["answer_span"][1]]for s in binding["supports"]]
            assert binding["supporting_step_ids"]==sorted({s["step_id"]for s in binding["supports"]})
            for support in binding["supports"]:
                start,end=support["answer_span"];assert history[support["step_id"]]["answer"][start:end]==support["answer_text"]
        for arm in templates.ARMS:
            prompt=templates.prompt(item,arm);newprefix,newactual=prompt.split("\n\nActual task:\n",1)
            # Everything from the raw actual question value through reader/end
            # is immutable. S1 can only rename its current heading.
            anchor=marker if arm!="S1"else marker.replace(f"#{item['step_id']}",f"@{item['step_id']}",1)
            assert newactual.split(anchor,1)[1]==actual.split(marker,1)[1]
            assert newactual.split("Relevant prior question-answer pairs:\n",1)[0]==actual.split("Relevant prior question-answer pairs:\n",1)[0]
            if arm=="B1":
                assert newprefix==prefix
                if bindings:assert newactual.replace(templates.binding_block(bindings)+"\n\n","",1)==actual
                else:assert prompt==item["control_prompt"]
            if arm=="P1":
                assert newprefix==prefix.replace(templates.SCOPE+"\n","",1)
                assert newactual.replace(templates.SCOPE+"\n"+marker,marker,1)==actual
                assert prompt.count(templates.SCOPE)==1
            if arm=="S1":
                assert "Resolve each @N"in newprefix and "Resolve each #N"not in newprefix
                old_examples=prefix.split("Worked examples:\n\n",1)[1]
                new_examples=newprefix.split("Worked examples:\n\n",1)[1]
                # Four example retrieved/answer blocks remain byte-identical.
                import re
                old_blocks=[block.split("\n\nRetrieved passages:\n",1)[1]for block in re.split(r"(?m)^Example \d+\n",old_examples)if block]
                new_blocks=[block.split("\n\nRetrieved passages:\n",1)[1]for block in re.split(r"(?m)^Example \d+\n",new_examples)if block]
                assert old_blocks==new_blocks and len(old_blocks)==4
                for prior in item["history"]:
                    literal=f'@{prior["step_id"]} Question: {prior["request"]}\nAnswer: {prior["answer"]}'
                    assert literal in newactual
            payload=build_payload(config["generation"],prompt,CAP)
            contracts.append(dict(probe_id=item["probe_id"],arm=arm,payload_sha256=digest(payload)))
        formatter=templates.formatter_prompt(item,"Synthetic candidate")
        assert "Question:\n"+item["current_question"]+"\n\n"in formatter
        assert formatter.endswith("\n\n".join(doc["block"]for doc in item["reader_packet"]["docs"]))
    assert contracts==rows(HERE/"PROMPT_CONTRACTS.frozen.jsonl")
    assert [dict(probe_id=item["probe_id"],candidate_bindings=templates.bindings(item))for item in items]==rows(HERE/"MAPPINGS.frozen.jsonl")
    assert templates.needs_formatter("one two three")and not templates.needs_formatter("one two three four")
    value=dict(passed=True,nodes=38,planned_conditions=152,stress_initial_conditions=32,
        B1_nodes_with_candidate_spans=mapping_nodes,B1_candidate_spans=mapping_count,
        B1_no_map_equal_C0=True,S1_actual_questions_reader_history_values_and_example_evidence_immutable=True,
        P1_exact_scope_sentence_once_relocated=True,formatter_raw_current_same_reader_128=True,
        no_gold_answers_or_current_node_response_used_by_mapper=True,new_API_SSH_GPU_requests=0)
    atomic_json(HERE/"OFFLINE_CONTRACTS.json",value)
    return value


def check():
    protocol=read(PROTOCOL)
    for path,expected in {**protocol["pins"],**protocol["protected_holds"]}.items():assert sha(ROOT/path)==expected,path
    assert read(PLAN/"PLAN.json")["execution_approval"]["approved"]is False
    return protocol


def dry_run():
    check()
    assert rows(ORDER)==task_order(rows(INPUTS))
    return dict(passed=True,protocol_sha256=sha(PROTOCOL),nodes=38,planned_conditions=152,
        stress_initial_conditions=32,source_and_order_exact=True,new_API_SSH_GPU_requests=0)


def freeze():
    if PROTOCOL.exists():return dry_run()
    validate();assert(HERE/"PROTOCOL.md").is_file()
    assert not(HERE/"RUN_AUTHORIZATION.json").exists()and not(OUT/"checkpoint.sqlite3").exists()
    paths=[*HERE.glob("*.py"),*HERE.glob("*.json"),*HERE.glob("*.jsonl"),HERE/"PROTOCOL.md",*HERE.glob("inputs/*.py")]
    paths.extend(source.pinned_paths())
    paths.extend([CONTEXT/"native_ask.py",CONTEXT/"message_transport.py",CONTEXT/"PROTOCOL.frozen.json",
        CONTEXT/"INPUTS90.jsonl",CONTEXT/"ORDER270.jsonl",
        HERE.parent/"p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        PACKAGE/"config.json",*list((PACKAGE/"core").glob("*.py")),ROOT/"scripts/cli_support.py",
        HERE.parent/"p02_formal1500_v1_20261021/network.py"])
    holds=[PLAN/"STOP_REQUESTED",ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
        ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    atomic_json(PROTOCOL,dict(experiment=HERE.name,revision=1,state="frozen_not_authorized",prepared_utc=utc(),
        arms=list(templates.ARMS),nodes=38,planned_conditions=152,stress_initial_conditions=32,
        fresh_seed_cache=None,scope="ordinaryQA128+native<4wordformatter128 only; no RAG F1/no goal/final/retrieval",
        deferred_queue="Remaining120 require --defer afterguard, explicit deferred AUTH, and exact context90 identity/allplannedattempted+FINISHED+noinflight",
        pins={rel(path):sha(path)for path in paths if path.resolve()not in
            {PROTOCOL.resolve(),(HERE/"RUN_AUTHORIZATION.json").resolve()}},
        protected_holds={rel(path):sha(path)for path in holds},main_evaluation_remains_paused=True))
    return dry_run()


def context90_completion_status():
    """Read predecessor metadata only; fixed-path stale artifacts cannot release it.

    FINISHED is the worker's existing leaf receipt (it has no job/protocol
    fields). Bind it through the matching job/READY receipt and GPU identity;
    do not invent missing fields or inspect predecessor predictions/scores.
    """
    required=[CONTEXT/"PROTOCOL.frozen.json",CONTEXT/"INPUTS90.jsonl",CONTEXT/"ORDER270.jsonl",
        CONTEXT_OUT/"checkpoint.sqlite3",CONTEXT_OUT/"REMOTE_JOB_ID.json",
        CONTEXT_OUT/"REMOTE_GPU_READY.json",CONTEXT_OUT/"REMOTE_GPU_FINISHED.json"]
    if any(not path.is_file()for path in required):return dict(finished=False,reason="missing_predecessor_completion_artifact")
    try:
        protocol_sha=sha(CONTEXT/"PROTOCOL.frozen.json")
        expected=dict(protocol_sha256=protocol_sha,inputs_sha256=sha(CONTEXT/"INPUTS90.jsonl"),
            order_sha256=sha(CONTEXT/"ORDER270.jsonl"))
        ordered=rows(CONTEXT/"ORDER270.jsonl");planned={row["task_id"]for row in ordered}
        if len(ordered)!=270 or len(planned)!=270:return dict(finished=False,reason="predecessor_order_not_exact270")
        with closing(ro(CONTEXT_OUT/"checkpoint.sqlite3"))as db:
            db.execute("BEGIN")
            identity_row=db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
            identity=json.loads(identity_row[0])if identity_row else{}
            completed={key for key, in db.execute("SELECT key FROM data WHERE kind='outcome'")}
            inflight=db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
        if any(identity.get(key)!=value for key,value in expected.items()):
            return dict(finished=False,reason="predecessor_checkpoint_identity_mismatch")
        held={read(path)["task_id"]for path in(CONTEXT_OUT/"held").glob("*.json")}
        if not completed<=planned or not held<=planned:return dict(finished=False,reason="predecessor_foreign_task_metadata")
        job=read(CONTEXT_OUT/"REMOTE_JOB_ID.json");ready=read(CONTEXT_OUT/"REMOTE_GPU_READY.json")
        finished=read(CONTEXT_OUT/"REMOTE_GPU_FINISHED.json")
        job_id=job.get("job_id")
        if not(isinstance(job_id,str)and job_id.isdigit()and ready.get("job_id")==job_id
            and job.get("protocol_sha256")==ready.get("protocol_sha256")==protocol_sha
            and ready.get("status")=="ready"and ready.get("no_LLM_credentials_or_requests")is True
            and str(job.get("directory","")).endswith("/context90_"+protocol_sha[:16])):
            return dict(finished=False,reason="predecessor_job_ready_identity_mismatch")
        processed=finished.get("retrieval_requests_processed")
        if not(finished.get("status")=="stopped"and finished.get("no_LLM_requests")is True
            and finished.get("GPU")==ready.get("gpu")and type(processed)is int and processed>=0):
            return dict(finished=False,reason="invalid_predecessor_worker_finished_leaf")
        attempted=len(completed|held)
        return dict(finished=attempted==270 and inflight==0,reason="complete"if attempted==270 and not inflight else"predecessor_pending",
            planned=270,attempted=attempted,in_flight=inflight,protocol_sha256=protocol_sha,job_id=job_id)
    except(KeyError,TypeError,ValueError,sqlite3.Error,OSError):
        return dict(finished=False,reason="invalid_predecessor_completion_metadata")


def context90_finished():return context90_completion_status()["finished"]


def wait_for_context90():
    """Durable thirty-second metadata wait; no Cache/client/ledger/SSH is open."""
    while True:
        status=context90_completion_status()
        atomic_json(OUT/"DEFERRED_WAIT_STATUS.json",dict(utc=utc(),protocol_sha256=sha(PROTOCOL),
            requested_scope="deferred120",API_SSH_GPU_requests=0,**status))
        if status["finished"]:return True
        if(OUT/"STOP_REQUESTED").exists():return False
        time.sleep(30)


def outcomes():
    if not(OUT/"checkpoint.sqlite3").exists():return[]
    with closing(ro(OUT/"checkpoint.sqlite3"))as db:
        return[json.loads(value)for value, in db.execute("SELECT value FROM data WHERE kind='outcome'")]


def held_outcomes():return[read(path)for path in(OUT/"held").glob("*.json")]


def accounting():
    ledger=Path(read(PACKAGE/"config.json")["budget"]["ledger"])
    namespace="p02_prompt_research_v1_20261022/"+str(OUT.resolve())
    with closing(ro(ledger))as db:
        charges=list(db.execute("SELECT actual,reserved FROM charges WHERE namespace=?",(namespace,)))
    return dict(actual_new_returned_usage_CNY=sum(actual or 0. for actual,_ in charges),
        unresolved_reserved_CNY=sum(reserved for actual,reserved in charges if actual is None))


def report():
    check();tasks=rows(ORDER);found=outcomes();completed={row["task_id"]for row in found}
    held=[row for row in held_outcomes()if row["task_id"]not in completed]
    attempted=completed|{row["task_id"]for row in held}
    value=dict(status="all_conditions_terminal"if len(found)==len(tasks)else
        "all_conditions_attempted_with_technical_holds"if len(attempted)==len(tasks)else"partial",utc=utc(),
        scope="Independent ordinaryQA component probes; no complete RAG F1 or populationaccuracy",
        planned_nodes=38,planned_conditions=152,terminal=len(found),held_technical=len(held),
        attempted_conditions=len(attempted),not_started_conditions=len(tasks)-len(attempted),
        by_arm={arm:dict(planned=38,terminal=sum(row["arm"]==arm for row in found),
            statuses=dict(Counter(row["status"]for row in found if row["arm"]==arm)),
            formatter_triggered=sum(row["arm"]==arm and row["formatter_triggered"]for row in found),
            logical_prompt_tokens=sum(row["usage"]["prompt_tokens"]for row in found if row["arm"]==arm),
            logical_completion_tokens=sum(row["usage"]["completion_tokens"]for row in found if row["arm"]==arm))for arm in templates.ARMS},
        stress_conditions_terminal=sum(row["is_stress"]for row in found),context90_finished=context90_finished(),
        fresh_same_run_fullpayload_reuse_only=True,no_complete_query_trend_from_stage_count=True,
        main_evaluation_remains_paused=True,**accounting())
    atomic_json(OUT/"REPORT.json",value)
    return value


def event(value):
    OUT.mkdir(parents=True,exist_ok=True)
    with(OUT/"events.jsonl").open("a",encoding="utf8")as handle:handle.write(json.dumps({"utc":utc(),**value},ensure_ascii=False)+"\n")


def perform(item,arm,native):
    initial=None;triggered=False
    base=dict(raw_current_question=item["current_question"],actual_child_prompt=templates.prompt(item,arm),
        candidate_bindings=templates.bindings(item)if arm=="B1"else[],actual_reader_packet=item["reader_packet"])
    try:
        rid,answer,finish=native("legacy/answer/"+str(item["step_id"]),base["actual_child_prompt"],CAP)
        initial=answer;triggered=templates.needs_formatter(initial)
        if triggered:
            rid,answer,finish=native("legacy/sentence_format/"+str(item["step_id"]),templates.formatter_prompt(item,initial),CAP)
        result={**base,"status":"success","answer":answer,"request_id":rid,"finish_reason":finish,"formatted_answer":answer}
    except AlgorithmTerminal as exc:
        stage=exc.value
        result={**base,"status":"algorithm_failure","answer":stage["answer"],"request_id":stage["request_id"],
            "finish_reason":stage["finish_reason"],"failure_stage":stage["stage"],"reason":stage["terminal_reason"],"formatted_answer":None}
        if initial is None:initial=stage["answer"]
    except(TechnicalHeld,RuntimeError,ValueError,TypeError,InterruptedError)as exc:
        result={**base,"status":"held_technical","error_type":type(exc).__name__,"formatted_answer":None}
    usage={key:sum(stage["usage"][key]for stage in native.stages)for key in("prompt_tokens","completion_tokens","total_tokens")}
    return{**result,"initial_answer":initial,"formatter_triggered":triggered,"stages":native.stages,"usage":usage,"utc":utc()}


def authorized(deferred=False):
    check();authority=read(HERE/"RUN_AUTHORIZATION.json")
    assert authority.get("approved")is True and authority.get("experiment")==HERE.name and authority.get("protocol_sha256")==sha(PROTOCOL)
    assert any(row.get("id")==authority.get("decision_id")for row in rows(PLAN/"DECISIONS.jsonl"))
    if deferred:assert authority.get("approved_deferred_conditions")is True,"Deferred120 not authorized by initial32 pilot AUTH"
    return authority


def run(limit=None,deferred=False,wait_for_guard=False):
    authorized(deferred);planned=rows(ORDER)
    held_ids={row["task_id"]for row in held_outcomes()}
    done={row["task_id"]for row in outcomes()}|held_ids
    if done=={row["task_id"]for row in planned}:return report()
    pending=[task for task in planned if task["task_id"]not in done and(task["is_stress"]or deferred)]
    if not pending:return report()
    if deferred and not context90_finished():
        if not wait_for_guard or not wait_for_context90():return report()
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache,run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config=read(PACKAGE/"config.json");assert config["generation"]==read(HERE/"GENERATION.frozen.json")
    items={row["probe_id"]:row for row in rows(INPUTS)}
    class NetworkEvents:
        def event(self,name,**values):event(dict(event=name,**values))
    with run_lock(OUT),Cache(OUT/"checkpoint.sqlite3")as cache,closing(NoCapBudget(config,OUT))as budget, \
        temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"), \
        provider_dns(urlsplit(config["generation"]["base_url"]).hostname,NetworkEvents()):
        # Another initial32 invocation may have completed or held a condition
        # while this process waited for the predecessor, before this lock.
        held_ids={row["task_id"]for row in held_outcomes()}
        identity=dict(protocol_sha256=sha(PROTOCOL),inputs_sha256=sha(INPUTS),order_sha256=sha(ORDER),historical_seed=None)
        old=cache.get("meta","identity")
        if old is None:
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==0
            cache.put("meta","identity",identity)
        else:assert old==identity
        stop=lambda:(OUT/"STOP_REQUESTED").exists()
        with closing(SubmissionAwareStrictAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop))as strict:
            processed=0
            for task in planned:
                held_ids.update(row["task_id"]for row in held_outcomes())
                if task["task_id"]in held_ids or cache.get("outcome",task["task_id"])is not None:continue
                if not task["is_stress"]and not deferred:break
                if stop()or(limit is not None and processed>=limit):break
                if not task["is_stress"]and not context90_finished():
                    event(dict(event="deferred_until_context90_finished",remaining_scope="nonstress120"));break
                processed+=1;item=items[task["probe_id"]]
                native=NativeAsk(cache,strict,config,task["task_id"],event=event)
                result={**task,**perform(item,task["arm"],native)}
                if result["status"]=="held_technical":
                    atomic_json(OUT/"held"/(digest(task["task_id"])+".json"),result)
                else:cache.put("outcome",task["task_id"],result)
                reference=item.get("historical_stage_usage_reference")
                if native.stages and reference and cost(native.stages[0]["usage"])>3*cost(reference):
                    event(dict(event="expense_alert",task_id=task["task_id"],reason="QA_component_above3x_same_historical_component",
                        scope="Stageonly; not fullquery expense trend"))
                summary=dict(event="condition_status",task_id=task["task_id"],status=result["status"])
                event(summary);print(json.dumps(summary,ensure_ascii=True),flush=True)
    return report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","validate","freeze","dry-run","report","run"))
    parser.add_argument("--limit",type=int)
    parser.add_argument("--defer",choices=("afterguard",),help="Explicitly request separately authorized deferred120 after predecessor identity/completion guard")
    parser.add_argument("--wait-for-guard",action="store_true",help="With --defer afterguard, persist thirty-second read-only predecessor checks until finished or local STOP")
    args=parser.parse_args()
    if args.limit is not None and(args.action!="run"or args.limit<1):parser.error("--limit is a positive run-only count")
    if args.defer is not None and args.action!="run":parser.error("--defer is run-only")
    if args.wait_for_guard and args.defer!="afterguard":parser.error("--wait-for-guard requires --defer afterguard")
    value=run(args.limit,deferred=args.defer=="afterguard",wait_for_guard=args.wait_for_guard)if args.action=="run"else{"prepare":prepare,"validate":validate,"freeze":freeze,"dry-run":dry_run,"report":report}[args.action]()
    print(json.dumps(value,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
