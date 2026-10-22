"""Independent M4 ordinary/goal reader-selection stage comparison; offline actions never create Cache/client."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import json
import sqlite3
from urllib.parse import urlsplit

from . import prepare as frozen, templates
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE,
)
from experiments.query_decomposition.p02_live_target_balanced20_dev_v1_20261022.native_ask import (
    NativeAsk, SubmissionAwareStrictAsk, AlgorithmTerminal, TechnicalHeld,
    integrity, stage_success, terminal_reason,
)

HERE,ROOT,OUT,PACKAGE,PLAN=frozen.HERE,frozen.ROOT,frozen.OUT,frozen.PACKAGE,frozen.PLAN
ARMS,CAP=templates.ARMS,frozen.CAP
read,rows,sha,digest=frozen.read,frozen.rows,frozen.sha,frozen.digest


def emit(value):print(json.dumps(value,ensure_ascii=True),flush=True)


def event(value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"events.jsonl").open("a",encoding="utf8") as handle:
        handle.write(json.dumps({"utc":utc(),**value},ensure_ascii=False)+"\n")


def cost(value):
    usage=value.get("usage") or {}
    return usage.get("prompt_tokens",0)*INPUT_RATE+usage.get("completion_tokens",0)*OUTPUT_RATE


def freeze():
    if frozen.PROTOCOL.exists():return frozen.dry_run()
    frozen.prepare()
    assert read(OUT/"OFFLINE_CONTRACTS.json")["passed"] is True
    assert not(HERE/"RUN_AUTHORIZATION.json").exists() and not(OUT/"checkpoint.sqlite3").exists()
    pins={}
    extras=[*HERE.glob("*.py"),*HERE.glob("*.jsonl"),HERE/"PUBLIC_PAYLOAD_PROOF.json",HERE/"GENERATION.frozen.json",
        HERE/"SELECTION.frozen.json",HERE/"PROTOCOL.md",OUT/"OFFLINE_CONTRACTS.json",
        frozen.INVENTORY,frozen.EVIDENCE/"policies.py",frozen.source.HERE/"prepare.py",
        frozen.source.MANIFEST,frozen.source.PRIOR,frozen.source.QUERIES,
        frozen.source.FORMAL/"REMOTE_INPUTS.frozen.json",frozen.source.FORMAL/"v12.txt",
        PACKAGE/"config.json",PACKAGE/"core/child_prompts.py",PACKAGE/"core/pipeline.py",PACKAGE/"core/llm.py",PACKAGE/"prompts/child_examples.json",
        HERE.parent/"p02_live_target_balanced20_dev_v1_20261022/native_ask.py",
        HERE.parent/"p02_prompt_binding_probe_v1_20261022/shared_runtime.py",
        PACKAGE/"core/cache.py",PACKAGE/"core/runtime.py",ROOT/"scripts/cli_support.py",
        HERE.parent/"p02_formal1500_v1_20261021/network.py"]
    pins.update({frozen.rel(path):sha(path)for path in extras})
    pins.update({r["source"]:r["source_sha256"]for r in rows(frozen.INPUTS)})
    pins.update({r["raw_file"]:r["raw_sha256"]for r in rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")})
    holds=[PLAN/"STOP_REQUESTED",ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED",
        ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED"]
    # Existing STOP files are protected; do not manufacture or clear any hold.
    assert all(p.exists()for p in holds)
    atomic_json(frozen.PROTOCOL,dict(id=HERE.name,state="offline_prepared_not_authorized",prepared_utc=utc(),
        planned_cases=8,planned_conditions=16,arms=list(ARMS),max_tokens=CAP,
        seed_cache=None,contemporaneous_fresh_control=True,
        contrast="M0 originalmerge exactsourcewire;M4 onlyactualreader selection same-source orderedtokens formatduplicate fill;nootherpolicy",
        fixed="Raw R02/v12 D,raw Current,history,originalrootquestion,instructions,examples,GOAL_RULE,model128;nativeformatter128;<4words;no rewrite/retrieval/final",
        scope_target="Raw Current subquestion remains evaluation target; true answer to another entity is not a repair",
        gate="No automatic followup;no substantive necessary-information/factual repair or any clear regression requires stop/review;technicalmissing prevents success;stageonlynotRAGF1",
        receipts="Model exact; all3 nonnegative integer usage fields and sum; ordinary stop nonempty no-refusal; valid length/filter/refusal/empty terminal without reroll; incomplete held",
        recovery="SDK0/HTTP0,max5 actual technical attempts total across restart; preserve returned/ledger evidence and all unknown reservations",
        scheduling="Shared provider gate serialized with3-second minimum starts;NoCapBudget;newnamespace;onlyrootauthorizationcanrun",
        expenses="QA+triggeredformatter logicalstageusage;allactual ledger charges andunknown separately;3xQA historical diagnostic;no20-stage-as-querytrend",
        limitation="Exposed8distinctWikiqueries:fixedWiki721+3oldfullscoregoal+2nonfullordinary+2fullscoreordinary;4ordinary4goal;notunseen/generalization/F1;contextmembershipandlengthvary",
        report="All16 statuses,initial/formattedtexts,formattertrigger,truthvsneededinformation,usage/newcost/unknown;no scoreror gold",
        main_evaluation_remains_paused=True,no_new_API_GPU_at_preparation=0,pins=pins,
        protected_holds={frozen.rel(p):sha(p)for p in holds}))
    return frozen.dry_run()


def identity(cache):
    expected=dict(protocol_sha256=sha(frozen.PROTOCOL),inputs_sha256=sha(frozen.INPUTS),order_sha256=sha(frozen.ORDER))
    saved=cache.get("meta","identity")
    if saved is None:
        assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==0 and not cache.all("outcome")
        cache.put("meta","identity",expected)
    else:assert saved==expected,"Changed frozen identity; do not resume"


def snapshot(cache=None):
    if cache is not None:return list(cache.all("outcome").values())
    path=OUT/"checkpoint.sqlite3"
    if not path.exists():return []
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as db:
        return [json.loads(v)for(v,)in db.execute("SELECT value FROM data WHERE kind='outcome'")]


def accounting(cache=None):
    ledger=Path(read(PACKAGE/"config.json")["budget"]["ledger"])
    namespace="p02_prompt_research_v1_20261022/"+str(OUT.resolve())
    with closing(sqlite3.connect(ledger.resolve().as_uri()+"?mode=ro",uri=True)) as db:
        charges=[dict(id=r[0],actual=r[1],reserved=r[2],status=r[3])for r in db.execute(
            "SELECT id,actual,reserved,status FROM charges WHERE namespace=?",(namespace,))]
    def ids(db):
        result={r[0]for r in db.execute("SELECT id FROM requests")}
        if db.execute("SELECT count(*) FROM sqlite_master WHERE name='retired_requests'").fetchone()[0]:
            result.update(r[0]for r in db.execute("SELECT id FROM retired_requests"))
        return result
    if cache is not None:known=ids(cache.db)
    elif(OUT/"checkpoint.sqlite3").exists():
        with closing(sqlite3.connect((OUT/"checkpoint.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as db:known=ids(db)
    else:known=set()
    return dict(new_actual_returned_usage_CNY=sum(r["actual"]for r in charges if r["actual"]is not None),
        unresolved_reserved_CNY=sum(r["reserved"]for r in charges if r["actual"]is None),
        reservations_without_current_or_retired_cache_record_ids=sorted(r["id"]for r in charges if r["actual"]is None and r["id"]not in known),
        unknown_reservations_retained=True)


def report(cache=None):
    frozen.check()
    planned=rows(frozen.ORDER);items=rows(frozen.INPUTS);outcomes=snapshot(cache)
    keys={t["task_id"]for t in planned};completed={r["task_id"]for r in outcomes}
    assert len(outcomes)==len(completed) and completed<=keys
    held=[read(p)for p in(OUT/"held").glob("*.json")]if(OUT/"held").exists()else[]
    held=[r for r in held if r["task_id"]not in completed]
    assert len(held)==len({r["task_id"]for r in held}) and all(r["task_id"]in keys for r in held)
    index={(r["probe_id"],r["arm"]):r for r in outcomes}
    pairs=[]
    for item in items:
        a,b=(index.get((item["probe_id"],arm))for arm in ARMS)
        if a is not None and b is not None:
            pairs.append(dict(probe_id=item["probe_id"],id=item["id"],dataset=item["dataset"],step_id=item["step_id"],
                M0_status=a["status"],M4_status=b["status"],both_normal=a["status"]==b["status"]=="success",
                text_changed=a["answer"]!=b["answer"],M0_answer=a["answer"],M4_answer=b["answer"],
                M0_initial_answer=a.get("initial_answer"),M4_initial_answer=b.get("initial_answer"),role=item["role"]))
    def summary(arm):
        data=[r for r in outcomes if r["arm"]==arm]
        held_stages=[s for h in held if h["arm"]==arm for s in h.get("completed_stages",[])]
        return dict(planned=sum(t["arm"]==arm for t in planned),terminal=len(data),
            normal_complete=sum(r["status"]=="success"for r in data),algorithm_terminal=sum(r["status"]=="algorithm_failure"for r in data),
            formatter_triggered=sum(r.get("formatter_triggered",False)for r in data),
            pending_or_held=sum(t["arm"]==arm for t in planned)-len(data),
            logical_prompt_tokens=sum(r["usage"]["prompt_tokens"]for r in data),logical_completion_tokens=sum(r["usage"]["completion_tokens"]for r in data),
            held_completed_stage_prompt_tokens=sum(s["usage"]["prompt_tokens"]for s in held_stages),
            held_completed_stage_completion_tokens=sum(s["usage"]["completion_tokens"]for s in held_stages),
            held_completed_stage_returned_usage_CNY=sum(cost(s)for s in held_stages),
            logical_returned_usage_CNY=sum(cost(r)for r in data))
    value=dict(utc=utc(),scope="Selected ordinary/goal reader-membership mechanism challenge; not full RAG/F1 or populationaccuracy",
        status="all_conditions_terminal"if completed==keys else"partial",planned_cases=len(items),planned_conditions=len(planned),terminal=len(outcomes),
        held_technical_tasks=len(held),held_task_ids=sorted(r["task_id"]for r in held),
        unstarted_or_unresolved_task_ids=sorted(keys-completed-{r["task_id"]for r in held}),by_arm={a:summary(a)for a in ARMS},
        complete_terminal_pairs=len(pairs),normal_pairs=sum(r["both_normal"]for r in pairs),
        changed_text_normal_pair_ids=[r["probe_id"]for r in pairs if r["both_normal"]and r["text_changed"]],
        no_scorer_or_final_F1=True,no_LLM_call_count_metric=True,semantic_gate="Await blinded stage review and root decision; no automaticpromotion",
        fresh_both_arms_seed_None=True,all_outcomes_retained=True,main_evaluation_remains_paused=True,
        no_complete_query_cost_trend_from_stage_count=True,**accounting(cache))
    frozen.jsonl(OUT/"DETAILS.local.jsonl",outcomes);frozen.jsonl(OUT/"PAIRED_TEXT.local.jsonl",pairs)
    atomic_json(OUT/"REPORT.json",value)
    return value


def authorize():
    authority=read(HERE/"RUN_AUTHORIZATION.json")
    assert authority.get("approved")is True and authority.get("experiment")==HERE.name
    assert authority.get("protocol_sha256")==sha(frozen.PROTOCOL)
    assert any(r.get("id")==authority.get("decision_id")for r in rows(PLAN/"DECISIONS.jsonl"))


def run(limit=None):
    frozen.check();authorize()
    # Completed reentry reads SQLite only, before any Cache/client/network setup.
    planned=rows(frozen.ORDER)
    if {r["task_id"]for r in snapshot()}=={t["task_id"]for t in planned}:return report()
    config=read(PACKAGE/"config.json")
    assert config["generation"]==read(HERE/"GENERATION.frozen.json")
    items={r["probe_id"]:r for r in rows(frozen.INPUTS)}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self,name,**fields):event(dict(event=name,**fields))
    with run_lock(OUT),Cache(OUT/"checkpoint.sqlite3")as cache,closing(NoCapBudget(config,OUT))as budget, \
            temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname,NetworkEvents()):
        identity(cache);stop=lambda:(OUT/"STOP_REQUESTED").exists()
        with closing(SubmissionAwareStrictAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop))as strict:
            processed=0
            for task in planned:
                key=task["task_id"]
                if cache.get("outcome",key)is not None:continue
                if stop()or(limit is not None and processed>=limit):return report(cache)
                processed+=1;item=items[task["probe_id"]]
                native=NativeAsk(cache,strict,config,key,event=event)
                packet=templates.reader(item,task["arm"])
                base={**task,"raw_current_question":item["current_question"],"fresh_experiment":True,
                    "actual_reader_packet":packet,"actual_reader_doc_ids":[d["doc_id"]for d in packet["docs"]]}
                initial=None;formatter_triggered=False
                try:
                    rid,answer,finish=native("legacy/answer/"+str(item["step_id"]),templates.prompt(item,task["arm"]),CAP)
                    initial=answer
                    formatter_triggered=templates.needs_formatter(initial)
                    if formatter_triggered:
                        rid,answer,finish=native("legacy/sentence_format/"+str(item["step_id"]),templates.formatter_prompt(item,task["arm"],initial),CAP)
                    stage=native.stages[-1]
                    outcome={**base,"status":"success","answer":answer,"request_id":rid,"finish_reason":finish}
                except AlgorithmTerminal as exc:
                    stage=exc.value
                    if initial is None:initial=stage["answer"]
                    outcome={**base,"status":"algorithm_failure","answer":stage["answer"],"request_id":stage["request_id"],
                        "finish_reason":stage["finish_reason"],"reason":stage["terminal_reason"],"failure_stage":stage["stage"]}
                except TechnicalHeld as exc:
                    atomic_json(OUT/"held"/(digest(key)+".json"),{**base,"status":"held_technical","initial_answer":initial,
                        "formatter_triggered":formatter_triggered,"completed_stages":native.stages,"reason":str(exc),"utc":utc()})
                    event(dict(event="held_technical",task_id=key,reason=str(exc)));report(cache);continue
                usage={k:sum(s["usage"][k]for s in native.stages)for k in("prompt_tokens","completion_tokens","total_tokens")}
                outcome.update(usage=usage,stages=native.stages,initial_answer=initial,formatter_triggered=formatter_triggered,
                    formatted_answer=outcome["answer"]if outcome["status"]=="success"else None,utc=utc())
                cache.put("outcome",key,outcome)
                reference=cost({"usage":item["historical_stage_usage_reference"]})
                actual_qa=cost(native.stages[0])
                if reference and actual_qa>3*reference:
                    alert=dict(event="expense_alert",task_id=key,reason="QA_stage_above3x_historical_reference",
                        historical_stage_CNY=reference,actual_stage_CNY=actual_qa,scope="Oneordinary/goalQAstageonly;notcompletequeryexpense")
                    event(alert);emit(alert)
                emit(dict(event="condition_terminal",task_id=key,status=outcome["status"]));report(cache)
    return report()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","freeze","dry-run","report","run"));parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error("--limit must be positive")
    emit({"prepare":frozen.prepare,"freeze":freeze,"dry-run":frozen.dry_run,"report":report,"run":lambda:run(args.limit)}[args.action]())
