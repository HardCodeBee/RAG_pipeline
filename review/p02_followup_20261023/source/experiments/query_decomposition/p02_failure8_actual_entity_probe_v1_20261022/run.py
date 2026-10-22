"""Root-authorized candidate execution; offline prepare never calls this path."""
from contextlib import closing
from pathlib import Path
import argparse
import json
import sqlite3
from urllib.parse import urlsplit
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022 import prepare as frozen, templates
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.runtime import CandidateAsk, normal_complete, algorithm_failure
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload

HERE, ROOT, OUT, PACKAGE = frozen.HERE, frozen.ROOT, frozen.OUT, frozen.PACKAGE
read, rows, sha, digest = frozen.read, frozen.rows, frozen.sha, frozen.digest


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as h:
        h.write(json.dumps({"utc":utc(), **value}, ensure_ascii=False) + "\n")


def cost(result):
    u = result["usage"]
    return u["prompt_tokens"] * INPUT_RATE + u["completion_tokens"] * OUTPUT_RATE


def identity(cache):
    expected=dict(protocol_sha256=sha(frozen.PROTOCOL), controls_sha256=sha(frozen.CONTROLS),
        inputs_sha256=sha(frozen.INPUTS), order_sha256=sha(frozen.ORDER))
    previous=cache.get("meta","identity")
    if previous is not None:
        assert previous==expected, "Checkpoint inputs/protocol changed; do not resume"
    else:
        assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==0 and not cache.all("outcome")
        cache.put("meta","identity",expected)


def import_controls(cache,items):
    for row in read(frozen.CONTROLS)["controls"]:
        item,source=items[row["probe_id"]],row["source_outcome"]
        value=dict(task_id=item["probe_id"]+"/B0",probe_id=item["probe_id"],id=item["id"],dataset=item["dataset"],
            step_id=item["step_id"],arm="B0",interface_group=item["interface_group"],mechanism_group=item["mechanism_group"],
            status="success",answer=source["answer"],request_id=row["source_request_id"],usage=row["result"]["usage"],
            finish_reason="stop",origin="reused_exact_fresh_I8_I0",source_database=row["source_database"],
            source_task_id=row["source_task_id"],actual_new_CNY=0,attributed_already_charged_CNY=row["source_charge"]["actual_CNY"],
            payload_sha256=row["payload_sha256"],utc=source["utc"])
        old=cache.get("outcome",value["task_id"])
        assert old is None or old==value
        if old is None:
            cache.put("outcome",value["task_id"],value)


def unique_candidate_outcomes(outcomes):
    seen={}
    for r in sorted(outcomes,key=lambda x:x.get("utc","")):
        if r["arm"]!="B0":
            seen.setdefault(r["request_id"],r)
    return list(seen.values())


def report(cache=None):
    frozen.check()
    if cache is not None:
        outcomes=list(cache.all("outcome").values())
    elif (OUT/"checkpoint.sqlite3").exists():
        with closing(sqlite3.connect((OUT/"checkpoint.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as db:
            outcomes=[json.loads(v) for(v,) in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    else:
        return dict(status="prepared_not_started",planned_nodes=8,planned_logical_conditions=24,
            B0_exact_snapshots_ready=8,distinct_planned_new_payloads=13,main_evaluation_remains_paused=True)
    planned={r["probe_id"]+"/"+arm for r in rows(frozen.INPUTS) for arm in templates.ARMS}
    done={r["task_id"] for r in outcomes}
    assert len(done)==len(outcomes) and done<=planned
    held=[read(p) for p in (OUT/"held").glob("*.json") if read(p)["task_id"] not in done]
    unique=unique_candidate_outcomes(outcomes)
    config=read(PACKAGE/"config.json")
    with closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri()+"?mode=ro",uri=True)) as db:
        actual,reserved=db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",
            ("p02_prompt_research_v1_20261022/"+str(OUT.resolve()),)).fetchone()
        pending_ids={r[0] for r in db.execute("SELECT id FROM charges WHERE namespace=? AND actual IS NULL",
            ("p02_prompt_research_v1_20261022/"+str(OUT.resolve()),))}
    def recorded_ids(db):
        found={r[0] for r in db.execute("SELECT id FROM requests")}
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_requests'").fetchone():
            found.update(r[0] for r in db.execute("SELECT id FROM retired_requests"))
        return found
    if cache is not None:
        known_ids=recorded_ids(cache.db)
    else:
        with closing(sqlite3.connect((OUT/"checkpoint.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as db:
            known_ids=recorded_ids(db)
    index={(r["probe_id"],r["arm"]):r for r in outcomes}
    comparisons={}
    for a,b in (("B0","B1"),("B1","B2"),("B0","B2")):
        pp=[p["probe_id"] for p in rows(frozen.INPUTS) if (p["probe_id"],a) in index and (p["probe_id"],b) in index]
        comparisons[a+"_to_"+b]=dict(terminal_pairs=len(pp),changed_text_ids=[p for p in pp if index[p,a]["answer"]!=index[p,b]["answer"]],
            same_actual_receipt_pairs=sum(index[p,a]["request_id"]==index[p,b]["request_id"] for p in pp))
    sections={}
    for arm in templates.ARMS:
        data=[r for r in outcomes if r["arm"]==arm]
        sections[arm]=dict(planned=8,terminal=len(data),normal_complete=sum(r["status"]=="success" for r in data),
            algorithm_terminal=sum(r["status"]=="algorithm_failure" for r in data),
            logical_input_tokens=sum(r["usage"]["prompt_tokens"] for r in data),logical_output_tokens=sum(r["usage"]["completion_tokens"] for r in data))
    value=dict(status="all_conditions_terminal" if len(done)==24 else "partial",planned_nodes=8,planned_logical_conditions=24,
        terminal=len(done),sections=sections,comparisons=comparisons,held_technical=held,pending_task_ids=sorted(planned-done),
        distinct_candidate_returned_receipts=len(unique),distinct_planned_new_payloads=13,
        candidate_unique_input_tokens=sum(r["usage"]["prompt_tokens"] for r in unique),candidate_unique_output_tokens=sum(r["usage"]["completion_tokens"] for r in unique),
        actual_new_known_returned_usage_CNY=actual,unresolved_reserved_CNY=reserved,
        unresolved_reservation_ids=sorted(pending_ids),reservations_without_cache_record_ids=sorted(pending_ids-known_ids),
        B0_attributed_already_charged_CNY=sum(r.get("attributed_already_charged_CNY",0) for r in outcomes),
        B0_new_actual_CNY=0,duplicate_conditions_not_independent_draws=True,no_gold_or_native_F1=True,
        semantic_correctness_not_automatically_assigned=True,scope="Exposed I8 ordinaryQA mechanism follow-up, not completeRAG",
        main_evaluation_remains_paused=True)
    frozen.jsonl(OUT/"DETAILS.local.jsonl",outcomes)
    atomic_json(OUT/"REPORT.json",value)
    return value


def run(limit=None):
    frozen.check()
    authority=read(HERE/"RUN_AUTHORIZATION.json")  # Required before Cache, ledger, key or client creation.
    assert authority.get("approved") is True and authority.get("experiment")==HERE.name
    assert authority.get("protocol_sha256")==sha(frozen.PROTOCOL)
    assert authority.get("controls_sha256")==sha(frozen.CONTROLS)
    assert any(r.get("id")==authority.get("decision_id") for r in rows(frozen.PLAN/"DECISIONS.jsonl"))
    config=read(PACKAGE/"config.json")
    assert config["generation"]==read(HERE/"GENERATION.frozen.json")
    items={r["probe_id"]:r for r in rows(frozen.INPUTS)}
    controls={r["probe_id"]:r for r in read(frozen.CONTROLS)["controls"]}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self,name,**fields):event(dict(event=name,**fields))
    with run_lock(OUT),Cache(OUT/"checkpoint.sqlite3") as cache,closing(NoCapBudget(config,OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname,NetworkEvents()):
        identity(cache)
        import_controls(cache,items)
        stop=lambda:(OUT/"STOP_REQUESTED").exists()
        recent=[(cost(controls[r["probe_id"]]["result"]),cost(r)) for r in unique_candidate_outcomes(list(cache.all("outcome").values()))]
        seen={r["request_id"] for r in unique_candidate_outcomes(list(cache.all("outcome").values()))}
        with closing(CandidateAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop)) as ask:
            processed=0
            for task in rows(frozen.ORDER):
                key=task["task_id"]
                if cache.get("outcome",key) is not None:
                    continue
                if stop() or(limit is not None and processed>=limit):
                    return report(cache)
                processed+=1
                item=items[task["probe_id"]]
                prompt=templates.prompt(item,task["arm"])
                payload=build_payload(config["generation"],prompt,frozen.CAP)
                base={**task,"id":item["id"],"dataset":item["dataset"],"step_id":item["step_id"],
                    "interface_group":item["interface_group"],"mechanism_group":item["mechanism_group"],
                    "payload_sha256":digest(payload),"origin":"candidate_provider_or_exact_same_experiment_receipt"}
                try:
                    rid,answer,finish=ask("failure8_candidate/"+key,prompt,frozen.CAP)
                    result=json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?",(rid,)).fetchone()[0])
                    if not normal_complete(result,config["generation"]["model"]):
                        raise RuntimeError("Held: matchingmodel, stop/nonempty/no refusal and all3 valid usage fields required")
                    outcome={**base,"status":"success","answer":answer,"request_id":rid,"finish_reason":finish,"usage":result["usage"],"utc":utc()}
                except(RuntimeError,InterruptedError) as exc:
                    row=cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?",(canonical(payload),)).fetchone()
                    result={**(json.loads(row[2]) if row and row[2] else {}),"status":row[1] if row else None}
                    reason=algorithm_failure(result,config["generation"]["model"])
                    if reason:
                        outcome={**base,"status":"algorithm_failure","answer":result.get("answer",""),"request_id":row[0],
                            "finish_reason":result.get("finish_reason"),"refusal":result.get("refusal"),"usage":result["usage"],"reason":reason,"utc":utc()}
                    else:
                        atomic_json(OUT/"held"/(digest(key)+".json"),{**base,"status":"held_technical","reason":str(exc),"request_id":row[0] if row else None,"utc":utc()})
                        event(dict(event="held_technical",task_id=key,reason=str(exc)))
                        report(cache)
                        continue
                cache.put("outcome",key,outcome)
                if outcome["request_id"] not in seen:
                    seen.add(outcome["request_id"])
                    reference,actual=cost(controls[item["probe_id"]]["result"]),cost(outcome)
                    recent.append((reference,actual))
                    flags=[]
                    if reference and actual>3*reference:flags.append("stage_above3x_same_node_I0")
                    if len(recent)>=20 and sum(c for _,c in recent[-20:])>1.5*sum(c for c,_ in recent[-20:]):flags.append("recent20_unique_stage_above1.5x_same_node_I0")
                    if flags:
                        alert=dict(event="expense_alert",task_id=key,reasons=flags,I0_stage_CNY=reference,actual_stage_CNY=actual,scope="Unique candidate stage receipt, not completequery cost")
                        event(alert);emit(alert)
                emit(dict(event="condition_terminal",task_id=key,status=outcome["status"]))
                report(cache)
    return report()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("dry-run","report","run"))
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error("--limit must be positive")
    emit(frozen.dry_run() if args.action=="dry-run" else report() if args.action=="report" else run(args.limit))
