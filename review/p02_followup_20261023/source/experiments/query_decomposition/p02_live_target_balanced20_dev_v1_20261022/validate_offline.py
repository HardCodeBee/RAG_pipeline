"""Source replay and isolated fake-only contracts. Never opens a real provider."""
from contextlib import closing
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import sqlite3
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.cache import Cache,canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import StrictAsk,atomic_json
from . import run,flow,retrieval_bridge as bridge
from .native_ask import NativeAsk,SubmissionAwareStrictAsk,AlgorithmTerminal,TechnicalHeld,stage_success,integrity


def source_replay():
    """All historical wire payloads/receipts, readers and renderers exact8."""
    config=run.read(run.PACKAGE/"config.json");pipeline=Pipeline(config)
    audits=[]
    for item in run.rows(run.I8/"INPUTS8.frozen.jsonl"):
        source=run.read(ROOT/item["source"])
        assert run.sha(ROOT/item["source"])==item["source_sha256"]
        byquery={}
        for n in source["nodes"]:
            for field in ("original_retrieval","rewritten_retrieval"):
                packet=n.get(field)
                if packet is not None:
                    if packet["query"] in byquery:assert byquery[packet["query"]]==packet
                    else:byquery[packet["query"]]=packet
        calls={c["stage"]:c for c in source["logical_calls"]}
        source_dbs={}
        seen=[]
        try:
            def oldask(stage,prompt,cap,allow_final_truncation=False):
                c=calls[stage];dbpath=ROOT/c["source"]
                if str(dbpath) not in source_dbs:source_dbs[str(dbpath)]=run.ro(dbpath)
                row=source_dbs[str(dbpath)].execute("SELECT payload,status,result FROM requests WHERE id=?",(c["request_id"],)).fetchone()
                payload=build_payload(config["generation"],prompt,cap)
                assert row and row[0]==canonical(payload) and row[1]=="success"
                receipt=json.loads(row[2]);assert receipt["status"]=="success"
                assert receipt.get("returned_model")==payload["model"] and run.usage_ok(receipt)
                assert not receipt.get("refusal") and receipt.get("finish_reason") in(("stop","length")if allow_final_truncation else("stop",))
                assert receipt["answer"].strip() and c["usage"]==receipt["usage"]
                assert c["payload_sha256"]==run.digest(payload)
                seen.append(stage)
                return c["request_id"],receipt["answer"].strip(),receipt["finish_reason"]
            clean=safe_case({k:source[k]for k in("id","qid","dataset","question","supporting_questions")})
            result=flow.execute_arm(clean,lambda q:deepcopy(byquery[q]),oldask,"D0",pipeline=pipeline)
            assert result["prediction"]==source["prediction"] and result["final_prompt"]==source["final_prompt"]
            assert result["final_context"]==source["final_context"] and result["helper"]==source["helper"]
            assert result["history_all"]==source["history_all"] and result["supporting_questions"]==source["supporting_questions"]
            for actual,old in zip(result["nodes"],source["nodes"]):
                for field in("request","answer","initial_answer","child_prompt","rewrite_prompt","raw_rewrite","rewritten_query","reader_packet","prior_complete_answers"):
                    assert actual[field]==old[field],(source["id"],field)
            assert len(seen)==len(calls)
            audits.append(dict(id=clean["id"],source_sha256=item["source_sha256"],native_full_payload_receipts_all_exact=True,
                supporting_questions_unchanged=True,raw_history_reader_formatter_final_exact=True,
                legacy_requested_model_bound_by_exact_payload=True))
        finally:
            for db in source_dbs.values():db.close()
    assert len(audits)==8
    return audits


def packet(request):
    docs=[]
    for rank in range(1,11):
        docid=hashlib.sha256((request["query"]+str(rank)).encode()).hexdigest()
        text="Synthetic article preserves its subject and relation."
        docs.append(dict(rank=rank,doc_id=docid,score=1.-rank/100,source="Synthetic article",text=text,
            block=f"[Chunk {rank} | source=Synthetic article, pages=1-1]\n{text}"))
    return dict(qid=request["qid"],query=request["query"],k=10,docs=docs)


def flow_contracts():
    config=run.read(run.PACKAGE/"config.json");pipeline=Pipeline(config)
    cases=run.rows(run.INPUTS);total=0;downstream=0
    for case in cases:
        paired={}
        for arm in flow.ARMS:
            asks=[];queries=[];records=[]
            def fake_retrieve(query):
                queries.append(query)
                return packet(bridge.identity("fake-resource",case["dataset"],case["qid"],query))
            def fake_ask(stage,prompt,cap,allow_final_truncation=False):
                asks.append(dict(stage=stage,prompt=prompt,cap=cap,trunc=allow_final_truncation))
                short=hashlib.sha256(prompt.encode()).hexdigest()[:12]
                if "/rewrite/" in stage:answer=f"Explicit resolved entity {short}?"
                elif "/sentence_format/"in stage:answer="The synthetic subject has the candidate relation."
                elif stage=="legacy/final":answer="Synthetic final answer."
                elif stage=="legacy/answer/2":answer="Candidate"  # force real<4word formatter
                else:answer=f"The synthetic subject has value {short}."
                return "fake-"+short,answer,"length"if stage=="legacy/final"else"stop"
            result=flow.execute_arm(case,fake_retrieve,fake_ask,arm,pipeline=pipeline,record_node=records.append)
            assert result["status"]=="success_truncated" and result["nodes"]==records
            assert result["history_all"]==[dict(step_id=n["step_id"],request=n["request"],answer=n["answer"])for n in records]
            assert [n["request"]for n in records[:-1]]==case["supporting_questions"]
            for node in records:
                interface=node["actual_question_interface"]
                if node["role"]=="ordinary":
                    total+=arm=="D1"
                    assert interface["evaluation_target"]==node["request"]
                    expected=node["rewritten_query"]if arm=="D1"else node["request"]
                    assert interface["model_current"]==expected
                    current=f'Current subquestion #{node["step_id"]}:\n{expected}\n\nRetrieved passages:'
                    assert current in node["child_prompt"].split("\n\nActual task:\n",1)[1]
                else:assert node["child_prompt"]==node["raw_P02_child_prompt"]
                if node["sentence_format_request_id"]:
                    sent=next(a for a in asks if a["stage"]==f'legacy/sentence_format/{node["step_id"]}')
                    assert "\n\nQuestion:\n"+node["request"]+"\n\nCandidate answer:"in sent["prompt"]
                    assert sent["cap"]==128 and not sent["trunc"]
            assert asks[-1]["cap"]==64 and asks[-1]["trunc"]
            assert asks[-1]["prompt"]==result["final_prompt"]
            assert all(a["cap"]==256 for a in asks if"/rewrite/"in a["stage"])
            paired[arm]=(result,asks,queries)
        a,b=paired["D0"][0],paired["D1"][0]
        assert a["nodes"][0]["child_prompt"]==b["nodes"][0]["child_prompt"]
        for raw,changed in zip(a["nodes"],b["nodes"]):
            if raw["prior_complete_answers"]!=changed["prior_complete_answers"]:
                assert raw["rewrite_prompt"]!=changed["rewrite_prompt"]
                assert raw["rewritten_query"]!=changed["rewritten_query"]
                assert changed["rewritten_query"]in paired["D1"][2]
                downstream+=1
    assert total==sum(len(c["supporting_questions"])for c in cases)and downstream>0
    # A malformed stop rewrite is rejected using the existing exact rule.
    for value in("not a question","Two lines?\nSecond?",""):
        assert flow.accepted_rewrite(value,"Original question?")=="Original question?"
    assert flow.accepted_rewrite("  Accepted question?  ","Original question?")=="Accepted question?"
    # Goal-only exists outside selected8; ensure wrapper never entityizes it.
    empty=dict(id="2wiki:fixture",qid="fixture",dataset="2wiki",question="Goal fixture?",supporting_questions=[])
    captured=[]
    def fake(stage,prompt,cap,allow_final_truncation=False):
        captured.append((stage,prompt));return "fixture","This is one complete factual sentence.","stop"
    result=flow.execute_arm(empty,lambda q:packet(bridge.identity("f","2wiki","fixture",q)),fake,"D1",pipeline=pipeline)
    assert result["nodes"][0]["child_prompt"]==result["nodes"][0]["raw_P02_child_prompt"]
    return dict(cases_both_arms=len(cases),all_ordinary_nodes_covered_by_policy=total,
        accepted_rewrite_is_live_node_rewrite=True,subsequent_new_history_rewrites_and_queries=downstream,
        formatter_uses_original_raw_question=True,goal_no_direct_prompt_change=True,
        actual_sent_prompt_recorded=True,raw_history_request_preserved=True,
        malformed_rewrite_falls_back=True,goal_only_no_change=True,native_final_length_preserved=True)


def good(answer="One complete factual answer.",finish="stop",model="fake-model"):
    return dict(status="success",requested_model=model,returned_model=model,answer=answer,
        finish_reason=finish,usage=dict(prompt_tokens=2,completion_tokens=3,total_tokens=5))


class Budget:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,actual REAL,status TEXT)")
    def reserve(self,rid,payload):
        with self.db:self.db.execute("INSERT INTO charges VALUES(?,NULL,'reserved')",(rid,))
    def finish(self,rid,result):
        if result.get("usage"):
            with self.db:self.db.execute("UPDATE charges SET actual=1,status='returned_usage'WHERE id=?",(rid,))
    def close(self):self.db.close()


def native_contracts():
    # Every Cache is newly created inside TemporaryDirectory; never a real DB.
    config=dict(generation=dict(model="fake-model",base_url="https://invalid.invalid/v1",temperature=0,
        api_key_env="ABSENT_TEST_KEY",timeout_seconds=1,enable_thinking=False))
    model="fake-model";checks={}
    for stage in("legacy/rewrite/2","legacy/answer/2","legacy/sentence_format/2","legacy/final"):
        assert stage_success(good(),model,stage)
        assert stage_success(good(finish="length"),model,stage)==(stage=="legacy/final")
        for change in({"usage":None},{"usage":{"prompt_tokens":True,"completion_tokens":3,"total_tokens":4}},
            {"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":6}},{"returned_model":"other"}):
            assert not integrity({**good(),**change},model)
    checks["all_stages_strict_model_three_int_usage_and_sum"]=True
    with TemporaryDirectory(prefix="full8-native-")as temporary:
        path=Path(temporary)/"cache.sqlite3"
        provider=[]
        for restart in range(2):
            with Cache(path)as cache:
                def strict(stage,prompt,cap):
                    provider.append(stage)
                    rid,_=cache.begin_request(stage,build_payload(config["generation"],prompt,cap))
                    cache.finish_request(rid,good(finish="length"))
                    raise RuntimeError("StrictAsk held length")
                adapter=NativeAsk(cache,strict,config,"case/D0")
                result=adapter("legacy/final","Synthetic final prompt",64,allow_final_truncation=True)
                assert result[2]=="length"
        assert len(provider)==1
        checks["final_length_one_fake_provider_then_resume_zero"]=True
    for changed,expected in((good(finish="length"),"terminal_length"),(good(answer=""),"terminal_empty"),
        ({**good(),"refusal":"declined"},"terminal_refusal"),(good(finish="content_filter"),"terminal_content_filter")):
        with TemporaryDirectory(prefix="full8-terminal-")as temporary:
            with Cache(Path(temporary)/"cache.sqlite3")as cache:
                payload=build_payload(config["generation"],"Terminal prompt",128)
                rid,_=cache.begin_request("old",payload);cache.finish_request(rid,changed)
                def forbidden(*a,**k):raise AssertionError("Terminal outcome must never call provider")
                for restart in range(2):
                    adapter=NativeAsk(cache,forbidden,config,"case/D0")
                    try:adapter("legacy/answer/2","Terminal prompt",128)
                    except AlgorithmTerminal as exc:assert exc.value["terminal_reason"]==expected
                    else:raise AssertionError("Expected algorithm terminal")
        checks[expected+"_never_retried"]=True
    for changed in({**good(finish="length"),"usage":None},{**good(),"returned_model":"other"}):
        with TemporaryDirectory(prefix="full8-held-")as temporary:
            with Cache(Path(temporary)/"cache.sqlite3")as cache:
                payload=build_payload(config["generation"],"Held prompt",128)
                rid,_=cache.begin_request("old",payload);cache.finish_request(rid,changed)
                def held(*a,**k):raise RuntimeError("Existing receipt is held")
                try:NativeAsk(cache,held,config,"case/D0")("legacy/answer/2","Held prompt",128)
                except TechnicalHeld:pass
                else:raise AssertionError("Invalid receipt not algorithm0")
    checks["missing_usage_and_model_mismatch_held_not_zero"]=True
    # Returnevidence/ledger crash window prevents retirement/reissue, including
    # isolated finish_reason/refusal that the shared older guard omits.
    fields=[dict(response_id="r"),dict(raw_response={}),dict(usage={}),dict(returned_model=model),
        dict(answer="partial"),dict(content="partial"),dict(refusal="no"),dict(finish_reason="length")]
    for extra in fields+[{}]:
        with TemporaryDirectory(prefix="full8-guard-")as temporary:
            with Cache(Path(temporary)/"cache.sqlite3")as cache:
                budget=Budget(Path(temporary)/"budget.sqlite3")
                payload=build_payload(config["generation"],"Unknown prompt",128)
                rid,_=cache.begin_request("unknown",payload,before_start=lambda r:budget.reserve(r,payload))
                with cache.db:cache.db.execute("UPDATE requests SET status='unknown',result=?WHERE id=?",(json.dumps(extra),rid))
                if not extra:
                    with budget.db:budget.db.execute("UPDATE charges SET actual=1,status='returned_usage'WHERE id=?",(rid,))
                strict=SubmissionAwareStrictAsk(cache,None,config,budget,lambda e:None)
                try:strict._retire(rid,"stage",canonical(payload),1,"fake")
                except RuntimeError:pass
                else:raise AssertionError("Returned evidence must hold")
                assert cache.db.execute("SELECT count(*)FROM requests").fetchone()[0]==1
                strict.close();budget.close()
    checks["eight_return_signs_and_ledger_before_cache_all_held"]=True
    # Five total unknown attempts persist across separate Cache constructions.
    # Replace ALL provider and gate/time facilities with local fakes.
    from contextlib import nullcontext
    shared="experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime"
    with TemporaryDirectory(prefix="full8-five-")as temporary:
        submitted=[]
        class Client:
            def request_once(self,payload):submitted.append(1);return dict(status="unknown")
            def close(self):pass
        with patch(shared+".provider_gate",lambda *a,**k:nullcontext()),patch(shared+".time.sleep",lambda s:None):
            for restart in range(7):
                with Cache(Path(temporary)/"cache.sqlite3")as cache:
                    budget=Budget(Path(temporary)/"budget.sqlite3")
                    strict=SubmissionAwareStrictAsk(cache,None,config,budget,lambda e:None,maximum_attempts=5)
                    strict.client=Client()
                    try:strict("fake/unknown","Five attempt prompt",128)
                    except RuntimeError:pass
                    else:raise AssertionError("Unknown must remain held")
                    assert len(submitted)==5
                    assert cache.db.execute("SELECT count(*)FROM retired_requests").fetchone()[0]==4
                    strict.close();budget.close()
        checks["five_actual_attempts_durable_seven_restarts"]=True
    return checks


def bridge_contracts():
    request=bridge.identity("resource","2wiki","fixture","Resolved question?")
    fetches=[];events=[]
    def fetch(r):fetches.append(r);return packet(r)
    router=bridge.PacketRouter("resource","2wiki","fixture",fetch,event=events.append)
    assert router("Resolved question?")==router("Resolved question?")
    assert len(fetches)==1
    router("Changed resolved question?")
    assert len(fetches)==2 and fetches[0]["query"]!=fetches[1]["query"]
    line=bridge.canonical(dict(request_id=bridge.key(request),**request))+"\n"
    returned=bridge.serve_one(line,fetch,expected_resources={"2wiki":"resource"})
    writer=StringIO()
    assert bridge.client_exchange(request,writer,StringIO(returned))==packet(request)
    assert writer.getvalue()==line
    for invalid in({**request,"resource_identity":"wrong"},{**request,"query":"Wrong query?"}):
        badline=bridge.canonical(dict(request_id=bridge.key(request),**invalid))
        try:bridge.serve_one(badline,fetch,expected_resources={"2wiki":"resource"})
        except ValueError:pass
        else:raise AssertionError("Identity mismatch accepted")
    assert not any(k in request for k in("answer","gold","history","prompt","api_key","password"))
    return dict(exact_query_cache_only=True,changed_query_real_fetch=True,strict_identity_and_packet=True,
        JSONL_roundtrip=True,no_model_messages_or_credentials_in_request=True)


def expense_and_completed_contracts():
    tasks=run.rows(run.ORDER)
    historic={r["id"]:r for r in run.rows(run.HISTORY)}
    a,b,c,d=[t["task_id"]for t in tasks[:4]]
    with TemporaryDirectory(prefix="full8-paid-")as temporary:
        with closing(sqlite3.connect(Path(temporary)/"cache.sqlite3"))as cache,closing(sqlite3.connect(Path(temporary)/"ledger.sqlite3"))as ledger:
            cache.executescript("CREATE TABLE requests(id TEXT,stage TEXT);CREATE TABLE retired_requests(id TEXT,stage TEXT);")
            ledger.execute("CREATE TABLE charges(id TEXT,namespace TEXT,actual REAL,reserved REAL)")
            cache.executemany("INSERT INTO requests VALUES(?,?)",[("good","full8/"+a+"/legacy/answer/1"),
                ("partial","full8/"+c+"/legacy/answer/1"),("uncertain","full8/"+d+"/legacy/answer/1")])
            cache.execute("INSERT INTO retired_requests VALUES(?,?)",("failed-paid","full8/"+a+"/legacy/answer/1"))
            ledger.executemany("INSERT INTO charges VALUES(?,'test',?,?)",[("good",2.,0.),("failed-paid",3.,0.),
                ("partial",7.,0.),("uncertain",None,4.),("unmapped",None,6.)])
            outcomes=[dict(task_id=t["task_id"],id=t["id"],status="success",utc=f"2026-10-22T00:00:{i:02d}Z")for i,t in enumerate(tasks)if t["task_id"]!=c]
            values,unattributed=run.paid_query_attribution(cache,ledger,tasks,outcomes,historic,"test")
            index={r["task_id"]:r for r in values}
            assert index[a]["actual_new_returned_usage_CNY"]==5.  # includes failed technical attempt
            assert index[a]["eligible_for_complete_paid_query_trend"]
            assert index[b]["actual_new_returned_usage_CNY"]==0 and not index[b]["eligible_for_complete_paid_query_trend"]
            assert index[c]["actual_new_returned_usage_CNY"]==7. and not index[c]["eligible_for_complete_paid_query_trend"]
            assert index[d]["unresolved_reserved_CNY"]==4. and not index[d]["eligible_for_complete_paid_query_trend"]
            assert unattributed["unresolved_reserved_CNY"]==6.
            assert not any("recent20"in r for r in run.expense_reasons(index[a],values))
    # Every plannedcondition completed must return before opening any bridge,
    # constructing Cache or allocating GPU. Use only temporary read-only data.
    with TemporaryDirectory(prefix="full8-complete-")as temporary:
        folder=Path(temporary)
        with closing(sqlite3.connect(folder/"checkpoint.sqlite3"))as db:
            db.execute("CREATE TABLE data(kind TEXT,key TEXT,value TEXT)")
            db.executemany("INSERT INTO data VALUES('outcome',?,'{}')",[(t["task_id"],)for t in tasks])
            db.commit()
        with patch.object(run,"OUT",folder),patch.object(run,"authorized",return_value={"approved":True}),\
            patch.object(run,"report",return_value={"status":"completed_read_only"}),\
            patch("p02_multigrain_timing_cachefix_v1.core.cache.Cache",side_effect=AssertionError("No Cache constructor after completion")):
            assert run.run()=={"status":"completed_read_only"}
    # The trend is now possible:19 complete queries cannot trigger20, whereas
    #20 eligible paidqueries above1.5x reference do. Arbitrary stage counts are
    # irrelevant to this query-level interface; no actual ledger is opened.
    twenty=[dict(eligible_for_complete_paid_query_trend=True,actual_new_returned_usage_CNY=2.,
        historical_complete_query_CNY=1.,utc=f"2026-10-22T00:00:{i:02d}Z")for i in range(20)]
    assert not any("recent20"in r for r in run.expense_reasons(twenty[-1],twenty[:19]))
    assert any("recent20"in r for r in run.expense_reasons(twenty[-1],twenty))
    return dict(failed_paid_attempts_included=True,pure_cache_and_partial_and_unknown_excluded_from_trend=True,
        unknown_unattributed_reserves_preserved=True,trend_uses_twenty_complete_paid_queries_not_stages=True,
        already_completed_run_returns_before_Cache_or_bridge=True)


def report_contract():
    """Fake fullN and matched-oldsubset metrics, never actual model endpoints."""
    tasks=run.rows(run.ORDER);history=run.rows(run.HISTORY)
    expected_old={r["id"]for r in history if r["historical_quality_available"]}
    expected_ds={ds:sum(r["dataset"]==ds and r["historical_quality_available"]for r in history)for ds in run.DATASETS}
    with TemporaryDirectory(prefix="balanced20-report-")as temporary:
        folder=Path(temporary)
        with closing(sqlite3.connect(folder/"checkpoint.sqlite3"))as db:
            db.executescript("CREATE TABLE data(kind TEXT,key TEXT,value TEXT);CREATE TABLE requests(id TEXT,stage TEXT,status TEXT);CREATE TABLE retired_requests(id TEXT,stage TEXT,status TEXT);")
            for task in tasks:
                value={**task,"status":"success","prediction":"FAKE_"+task["arm"],"stages":[],"utc":"2026-10-22T00:00:00Z","terminal_reason":None}
                db.execute("INSERT INTO data VALUES('outcome',?,?)",(task["task_id"],json.dumps(value)))
            db.commit()
        with closing(sqlite3.connect(folder/"ledger.sqlite3"))as ledger:
            ledger.execute("CREATE TABLE charges(id TEXT,namespace TEXT,actual REAL,reserved REAL)")
            ledger.commit()
        original_read=run.read
        def fake_read(path):
            if Path(path)==run.PACKAGE/"config.json":return dict(budget=dict(ledger=str(folder/"ledger.sqlite3")))
            return original_read(path)
        def score(pred,ref):
            f=.2 if pred=="FAKE_D0"else(.3 if pred=="FAKE_D1"else .1)
            return dict(f1=f,em=0.)
        saved_reports={}
        with patch.object(run,"OUT",folder),patch.object(run,"check",return_value={}),\
            patch.object(run,"read",side_effect=fake_read),patch.object(run,"SCORERS",{ds:score for ds in run.DATASETS}),\
            patch.object(run,"atomic_json",side_effect=lambda path,value:saved_reports.setdefault(Path(path).name,deepcopy(value))),\
            patch("p02_multigrain_timing_cachefix_v1.core.cache.Cache",side_effect=AssertionError("Report must remain readonly")):
            result=run.report()
        assert saved_reports["REPORT.json"]==result
        n=len(run.rows(run.INPUTS));matched=result["matched_historical_subset"]
        assert result["planned_conditions"]==len(tasks)==n*len(flow.ARMS)
        assert result["arms"]["D0"]["planned"]==result["arms"]["D1"]["planned"]==n
        assert abs(result["paired"]["delta_F1"]-.1)<1e-12
        assert result["historical_source_reference_only"]["F1"]is None
        assert matched["n"]==len(expected_old)==15 and matched["missing_historical"]==n-len(expected_old)
        assert abs(matched["old_native"]["F1"]-.1)<1e-12
        assert abs(matched["D0_native"]["F1"]-.2)<1e-12 and abs(matched["D1_native"]["F1"]-.3)<1e-12
        assert all(matched["by_dataset"][ds]["n"]==expected_ds[ds]for ds in run.DATASETS)
        assert all(result["by_dataset"][ds][a]["planned"]==n//len(run.DATASETS)for ds in run.DATASETS for a in flow.ARMS)
    return dict(fake_only=True,full_N_denominators_and_pairs=True,matched_historical_same_membership=True,
        old_missing_not_imputed=True,old_subset_not_compared_as_full_N=True,report_no_Cache_constructor=True)


def main():
    # No actual Cache path, API key, network, SSH, GPU or scorer is opened here.
    assert not(run.OUT/"checkpoint.sqlite3").exists()
    run.prepare()
    report=dict(passed=True,state="offline_contracts_only_not_authorized",
        original_native_source_replay=source_replay(),flow=flow_contracts(),native=native_contracts(),bridge=bridge_contracts(),
        expense=expense_and_completed_contracts(),
        report=report_contract(),
        real_provider_calls=0,real_retrieval_searches=0,SSH_GPU_starts=0,temporary_fake_SQLite_only=True)
    atomic_json(HERE/"OFFLINE_CONTRACTS.json",report)
    print(json.dumps({k:v for k,v in report.items()if k!="original_native_source_replay"},ensure_ascii=True),flush=True)


if __name__=="__main__":main()
