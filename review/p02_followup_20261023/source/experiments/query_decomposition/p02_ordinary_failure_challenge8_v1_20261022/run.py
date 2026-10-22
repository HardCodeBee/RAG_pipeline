"""Independent16-condition I8 runner. Offline actions never create clients/Cache."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE,
)
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import prepare as frozen,templates

OUT=frozen.OUT
PACKAGE=frozen.PACKAGE
PLAN=HERE.parent/"unified_rag_eval_v1_20261021"
RUNTIME=HERE.parent/"p02_prompt_binding_probe_v1_20261022/shared_runtime.py"
EXECUTION=HERE/"EXECUTION_PROTOCOL.frozen.json"
ARMS=templates.ARMS
CAP=128
read,rows,sha,digest=frozen.read,frozen.rows,frozen.sha,frozen.digest


def emit(value):
    print(json.dumps(value,ensure_ascii=True),flush=True)


def event(value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"events.jsonl").open("a",encoding="utf8") as f:
        f.write(json.dumps({"utc":utc(),**value},ensure_ascii=False)+"\n")


def cost(value):
    usage=value.get("usage") or {}
    return usage.get("prompt_tokens",0)*INPUT_RATE+usage.get("completion_tokens",0)*OUTPUT_RATE


def receipt_integrity(result,model):
    u=result.get("usage") or {}
    return (result.get("status")=="success" and result.get("requested_model")==result.get("returned_model")==model
        and all(type(u.get(k)) is int and u[k]>=0 for k in ("prompt_tokens","completion_tokens","total_tokens"))
        and u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"])


def algorithm_failure(result,model):
    if not receipt_integrity(result,model):
        return None
    if result.get("finish_reason") in ("length","content_filter"):
        return "terminal_"+result["finish_reason"]
    if result.get("refusal"):
        return "terminal_refusal"
    if not isinstance(result.get("answer"),str) or not result["answer"].strip():
        return "terminal_empty"
    return None


def normal_complete(result,model):
    return (receipt_integrity(result,model) and result.get("finish_reason")=="stop"
        and isinstance(result.get("answer"),str) and bool(result["answer"].strip()) and not result.get("refusal"))


class SubmissionAwareStrictAsk(StrictAsk):
    """Keep shared recovery, holding any additional evidence of a returned call.

    In particular ledger.finish can commit before cache.finish_request. A crash
    between them must not turn known returned usage into a new blind submission.
    This read-only guard does not change the frozen shared runtime.
    """
    def _retire(self,rid,stage,payload,attempts,reason):
        row=self.cache.db.execute("SELECT result FROM requests WHERE id=?",(rid,)).fetchone()
        result=json.loads(row[0]) if row and row[0] else {}
        returned=any(result.get(k) is not None for k in ("response_id","raw_response","usage","returned_model"))
        returned=returned or bool((result.get("answer") or result.get("content") or "").strip())
        charge=self.budget.db.execute("SELECT actual,status FROM charges WHERE id=?",(rid,)).fetchone()
        if returned or(charge and(charge[0] is not None or charge[1]=="returned_usage")):
            raise RuntimeError("Held returned evidence in request or accounting ledger; preserve receipt/reservation for reconciliation")
        return super()._retire(rid,stage,payload,attempts,reason)


def base_check():
    old=read(frozen.PROTOCOL)
    assert sha(frozen.PROTOCOL)=="f7cfa47e0552380e4fb95333e827926533907a0b56ae99b3b085a2278912f6dd"
    for path,expected in old["pins"].items():
        assert sha(ROOT/path)==expected,path
    assert read(PLAN/"PLAN.json")["execution_approval"]["approved"] is False
    return old


def prepare_runner():
    base_check()
    if EXECUTION.exists():
        return dry_run()
    assert not (OUT/"checkpoint.sqlite3").exists() and not (HERE/"RUN_AUTHORIZATION.json").exists()
    # Original protocol/artifacts are retained byte-for-byte in place and pinned;
    # this inventory documents the unexecuted preparation revision explicitly.
    originals={path:expected for path,expected in read(frozen.PROTOCOL)["pins"].items() if path.startswith(frozen.rel(HERE)+"/")}
    originals[frozen.rel(frozen.PROTOCOL)]=sha(frozen.PROTOCOL)
    originals[frozen.rel(OUT/"DRYRUN.json")]=sha(OUT/"DRYRUN.json")
    atomic_json(HERE/"OFFLINE_PREPARE_REV001_INVENTORY.json",dict(state="unexecuted_offline_preparation_preserved_in_place",files=originals))
    pins={**read(frozen.PROTOCOL)["pins"],frozen.rel(frozen.PROTOCOL):sha(frozen.PROTOCOL)}
    extra=[Path(__file__),HERE/"RUNNER_PROTOCOL.md",HERE/"validate_runner_offline.py",HERE/"OFFLINE_PREPARE_REV001_INVENTORY.json",
        RUNTIME,PACKAGE/"core/cache.py",PACKAGE/"core/runtime.py",ROOT/"scripts/cli_support.py",
        HERE.parent/"p02_formal1500_v1_20261021/network.py"]
    pins.update({frozen.rel(p):sha(p) for p in extra})
    atomic_json(EXECUTION,dict(id=HERE.name,execution_revision=2,prepared_utc=utc(),state="prepared_not_authorized_or_executed",
        original_prepare_protocol_sha256=sha(frozen.PROTOCOL),planned_nodes=8,planned_conditions=16,arms=list(ARMS),cap=CAP,
        inputs_and_original_single_sentence_contracts_unchanged=True,sample_expansion=False,
        fresh="BothI0/I1 in one new checkpoint; StrictAsk seed=None. Never import original responses; resume sameDB only",
        execution_policy=dict(maximum_actual_attempts=5,attempts_durable_across_restarts=True,SDK_retries=0,HTTP_retries=0,
            shared_gate=True,minimum_start_interval_seconds=3,no_fixed_cost_cap=True,ledger="Existing cumulative ledger viaNoCapBudget",
            technical_returned_unknown="Hold with original receipt/reservation; never silently resubmit",
            unknown_without_attributable_return="Existing explicitly bounded recovery; preserve old receipts and reservations",
            strict_new_receipts="Requested/returned model match; all3 token fields nonnegative int excludingbool andsum; stop/nonempty/no refusal",
            algorithm_terminal="Length,filter,refusal,empty with valididentity+usage are retained terminal; no reroll",
            technical_held="Missingusage/modelmismatch/incomplete/unknown or5attempt exhaustion separately held; no pseudoF1"),
        reporting="Status/candidates, groups6failure+2abstention, paired text changes,tokens/cost/reservations; no gold/native scorer/F1/LLMcall metric",
        expenses="Existing single-stage3x and recent20-stage1.5x paired historical-stage alert policy; all fees and unknown reservations retained",
        authorization="Separate root-reviewed RUN_AUTHORIZATION approved with this protocolhash and existing decisionID required before network",
        no_main_evaluation_resumed=True,no_new_provider_requests_at_prepare=0,pins=pins))
    return dry_run()


def check():
    base_check()
    protocol=read(EXECUTION)
    for path,expected in protocol["pins"].items():
        assert sha(ROOT/path)==expected,path
    return protocol


def dry_run():
    check()
    items,proofs,receipts,dec,public=frozen.assemble()
    assert items==rows(frozen.INPUTS)
    assert proofs==rows(HERE/"SOURCE_PROOF8.frozen.jsonl")
    assert receipts==rows(HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl")
    assert dec==rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public==read(HERE/"PUBLIC_PAYLOAD_PROOF.json")
    contracts=frozen.make_contracts(items)
    assert contracts==rows(HERE/"CONTRACTS16.frozen.jsonl") and len(contracts)==16
    assert sha(frozen.PROTOCOL)==read(EXECUTION)["original_prepare_protocol_sha256"]
    value=dict(utc=utc(),passed=True,nodes=8,conditions=16,all_original_inputs_contracts_public_raw_proofs_unchanged=True,
        fresh_same_checkpoint_seed_None=True,new_receipt_identity_and_all3_usage_fields_required=True,
        no_client_created=True,no_Cache_created=True,no_API_requests=0,no_checkpoint_created=not(OUT/"checkpoint.sqlite3").exists(),
        run_authorization_present=(HERE/"RUN_AUTHORIZATION.json").exists(),main_evaluation_remains_paused=True,
        execution_protocol_sha256=sha(EXECUTION))
    atomic_json(OUT/"RUNNER_DRYRUN.json",value)
    return value


def identity(cache):
    expected=dict(execution_protocol_sha256=sha(EXECUTION),inputs_sha256=sha(frozen.INPUTS),
        order_sha256=sha(HERE/"TASK_ORDER16.frozen.jsonl"))
    current=cache.get("meta","identity")
    if current is not None:
        assert current==expected,"Do not resume a changed experiment in this checkpoint"
    else:
        assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==0 and not cache.all("outcome")
        cache.put("meta","identity",expected)


def snapshot(cache=None):
    if cache is not None:
        return list(cache.all("outcome").values())
    path=OUT/"checkpoint.sqlite3"
    if not path.exists():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as db:
        return [json.loads(v) for(v,) in db.execute("SELECT value FROM data WHERE kind='outcome'")]


def report(cache=None):
    check()
    outcomes=snapshot(cache)
    planned=rows(HERE/"TASK_ORDER16.frozen.jsonl")
    keys={t["task_id"] for t in planned}
    assert len(outcomes)==len({r["task_id"] for r in outcomes}) and all(r["task_id"] in keys for r in outcomes)
    groups={x["id"]+"/ordinary/"+str(x["step_id"]):("local_QA_failure" if x["eligible_for_same_reader_qa_challenge"] else "reasonable_abstention_boundary") for x in rows(HERE/"SELECTION8.frozen.jsonl")}
    completed={r["task_id"] for r in outcomes}
    held=[read(p) for p in (OUT/"held").glob("*.json")] if (OUT/"held").exists() else []
    held=[r for r in held if r["task_id"] not in completed]
    index={(r["probe_id"],r["arm"]):r for r in outcomes}
    pairs=[]
    for item in rows(frozen.INPUTS):
        pid=item["probe_id"]
        a,b=index.get((pid,"I0")),index.get((pid,"I1"))
        if a is not None and b is not None:
            pairs.append(dict(probe_id=pid,group=groups[pid],I0_status=a["status"],I1_status=b["status"],
                both_normal=a["status"]==b["status"]=="success",text_changed=a["answer"]!=b["answer"],I0_answer=a["answer"],I1_answer=b["answer"]))
    def summarize(arm,group=None):
        tasks=[t for t in planned if t["arm"]==arm and(group is None or groups[t["probe_id"]]==group)]
        data=[r for r in outcomes if r["arm"]==arm and(group is None or groups[r["probe_id"]]==group)]
        return dict(planned=len(tasks),terminal=len(data),normal_complete=sum(r["status"]=="success" for r in data),
            algorithm_terminal=sum(r["status"]=="algorithm_failure" for r in data),
            pending_or_held=len(tasks)-len(data),prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in data),
            completion_tokens=sum(r["usage"]["completion_tokens"] for r in data),returned_usage_CNY=sum(cost(r) for r in data))
    value=dict(utc=utc(),scope="Selected8 ordinaryQA mechanism development challenge, not completeRAG or population accuracy",
        status="all_conditions_terminal" if len(outcomes)==16 else "partial",planned_nodes=8,planned_conditions=16,
        terminal=len(outcomes),held_technical_tasks=len(held),held_task_ids=[r["task_id"] for r in held],
        unstarted_or_unresolved_task_ids=sorted(keys-completed-{r["task_id"] for r in held}),
        by_arm={a:summarize(a) for a in ARMS},by_group={g:{a:summarize(a,g) for a in ARMS} for g in sorted(set(groups.values()))},
        complete_terminal_pairs=len(pairs),normal_pairs=sum(r["both_normal"] for r in pairs),
        changed_text_normal_pair_ids=[r["probe_id"] for r in pairs if r["both_normal"] and r["text_changed"]],
        semantic_correctness_not_automatically_assigned=True,no_F1_or_gold_scorer=True,no_LLM_call_count_metric=True,
        fresh_I0_not_historical_replay=True,all_outcomes_retained=True,main_evaluation_remains_paused=True)
    ledger=Path(read(PACKAGE/"config.json")["budget"]["ledger"])
    with closing(sqlite3.connect(ledger.resolve().as_uri()+"?mode=ro",uri=True)) as db:
        actual,reserved=db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",("p02_prompt_research_v1_20261022/"+str(OUT.resolve()),)).fetchone()
        pending_ids={r[0] for r in db.execute("SELECT id FROM charges WHERE namespace=? AND actual IS NULL",("p02_prompt_research_v1_20261022/"+str(OUT.resolve()),))}
    def recorded_ids(db):
        ids={r[0] for r in db.execute("SELECT id FROM requests")}
        if db.execute("SELECT count(*) FROM sqlite_master WHERE name='retired_requests'").fetchone()[0]:
            ids.update(r[0] for r in db.execute("SELECT id FROM retired_requests"))
        return ids
    if cache is not None:
        known_ids=recorded_ids(cache.db)
    elif(OUT/"checkpoint.sqlite3").exists():
        with closing(sqlite3.connect((OUT/"checkpoint.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as db:
            known_ids=recorded_ids(db)
    else:
        known_ids=set()
    value.update(new_experiment_actual_returned_usage_CNY=actual,unresolved_reserved_CNY=reserved,
        reservations_without_current_or_retired_cache_record_ids=sorted(pending_ids-known_ids),
        orphan_reservation_limit="Preserved separately; absence of cache row does not establish submission or permit deleting a reservation")
    frozen.jsonl(OUT/"DETAILS.local.jsonl",outcomes)
    frozen.jsonl(OUT/"PAIRED_TEXT.local.jsonl",pairs)
    atomic_json(OUT/"REPORT.json",value)
    return value


def run(limit=None):
    check()
    authority=read(HERE/"RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment")==HERE.name
    assert authority.get("protocol_sha256")==sha(EXECUTION)
    assert any(r.get("id")==authority.get("decision_id") for r in rows(PLAN/"DECISIONS.jsonl"))
    config=read(PACKAGE/"config.json")
    assert config["generation"]==read(HERE/"GENERATION.frozen.json")
    items={r["probe_id"]:r for r in rows(frozen.INPUTS)}
    historical={r["probe_id"]:r["result"] for r in rows(HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl")}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self,name,**fields):event(dict(event=name,**fields))
    with run_lock(OUT),Cache(OUT/"checkpoint.sqlite3") as cache,closing(NoCapBudget(config,OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname,NetworkEvents()):
        identity(cache)
        stop=lambda:(OUT/"STOP_REQUESTED").exists()
        recent=[(cost(historical[r["probe_id"]]),cost(r)) for r in sorted(cache.all("outcome").values(),key=lambda r:r["utc"])]
        with closing(SubmissionAwareStrictAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop)) as ask:
            processed=0
            for task in rows(HERE/"TASK_ORDER16.frozen.jsonl"):
                key=task["task_id"]
                if cache.get("outcome",key) is not None:
                    continue
                if stop() or(limit is not None and processed>=limit):
                    return report(cache)
                processed+=1
                item=items[task["probe_id"]]
                prompt=templates.prompt(item["control_prompt"],task["arm"])
                base={**task,"id":item["id"],"dataset":item["dataset"],"step_id":item["step_id"],"fresh_experiment":True}
                try:
                    rid,answer,finish=ask("ordinary_failure/"+key,prompt,CAP)
                    result=json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?",(rid,)).fetchone()[0])
                    if not normal_complete(result,config["generation"]["model"]):
                        raise RuntimeError("Held: new complete matching-model receipt and all3 integer usage fields required")
                    outcome={**base,"status":"success","answer":answer,"request_id":rid,"finish_reason":finish,"usage":result["usage"],"utc":utc()}
                except(RuntimeError,InterruptedError) as exc:
                    payload=canonical(build_payload(config["generation"],prompt,CAP))
                    row=cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?",(payload,)).fetchone()
                    result={**(json.loads(row[2]) if row and row[2] else {}),"status":row[1] if row else None}
                    reason=algorithm_failure(result,config["generation"]["model"])
                    if reason:
                        outcome={**base,"status":"algorithm_failure","answer":result.get("answer", ""),"request_id":row[0],
                            "finish_reason":result.get("finish_reason"),"refusal":result.get("refusal"),"usage":result["usage"],"reason":reason,"utc":utc()}
                        event(dict(event="terminal_algorithm_outcome",task_id=key,reason=reason))
                    else:
                        held={**base,"status":"held_technical","reason":str(exc),"request_id":row[0] if row else None,"utc":utc()}
                        atomic_json(OUT/"held"/(digest(key)+".json"),held)
                        event(dict(event="held_technical",task_id=key,reason=str(exc)))
                        report(cache)
                        continue
                cache.put("outcome",key,outcome)
                reference,actual=cost(historical[item["probe_id"]]),cost(outcome)
                recent.append((reference,actual))
                reasons=[]
                if reference and actual>3*reference:
                    reasons.append("ordinary_stage_above3x_paired_historical")
                if len(recent)>=20 and sum(c for _,c in recent[-20:])>1.5*sum(b for b,_ in recent[-20:]):
                    reasons.append("recent20_ordinary_stage_above1.5x_paired_historical")
                if reasons:
                    alert=dict(event="expense_alert",task_id=key,reasons=reasons,historical_stage_CNY=reference,actual_stage_CNY=actual,scope="OrdinaryQA stage only, not complete-query cost")
                    event(alert);emit(alert)
                emit(dict(event="condition_terminal",task_id=key,status=outcome["status"]))
                report(cache)
    return report()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare-runner","dry-run","report","run"))
    p.add_argument("--limit",type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:
        p.error("--limit must be positive")
    emit({"prepare-runner":prepare_runner,"dry-run":dry_run,"report":report,"run":lambda:run(args.limit)}[args.action]())
