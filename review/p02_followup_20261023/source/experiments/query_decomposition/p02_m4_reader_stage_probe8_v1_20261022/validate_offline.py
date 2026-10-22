"""Real adapter with fake provider/budget/gate and temporary SQLite only."""
from contextlib import contextmanager,ExitStack,closing
from copy import deepcopy
from pathlib import Path
import json
import importlib.util
import sqlite3
import tempfile
from unittest.mock import patch

from . import run as app,templates,selection
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as shared


@contextmanager
def fake_context(*args,**kwargs):yield


class Budget:
    def __init__(self,path,output):
        self.db=sqlite3.connect(path)
        self.namespace="p02_prompt_research_v1_20261022/"+str(output.resolve())
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,namespace TEXT,reserved REAL,actual REAL,status TEXT)");self.db.commit()
    def reserve(self,rid,payload):
        with self.db:self.db.execute("INSERT INTO charges VALUES(?,?,.1,NULL,'reserved')",(rid,self.namespace))
    def finish(self,rid,result):
        if result.get("usage"):
            with self.db:self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?",(app.cost(result),rid))
    def close(self):self.db.close()


class Client:
    def __init__(self,answers):self.answers=deepcopy(answers);self.calls=0;self.payloads=[]
    def request_once(self,payload):
        self.calls+=1;self.payloads.append(payload)
        assert self.answers,"Unexpected fake provider resubmission"
        return self.answers.pop(0)
    def close(self):pass


class Ask(app.SubmissionAwareStrictAsk):
    def __init__(self,cache,config,budget,client):
        super().__init__(cache,None,config,budget,lambda event:None,maximum_attempts=5,recover_unknown=True)
        self.fixture_client=client
    def _client(self):return self.fixture_client


def validate():
    app.OUT.mkdir(parents=True,exist_ok=True)
    items,manifest,proofs,receipts,decomp,public=app.frozen.assemble()
    assert items==app.rows(app.frozen.INPUTS)
    assert app.frozen.contracts(items)==app.rows(app.HERE/"CONTRACTS16.frozen.jsonl")
    assert len(app.rows(app.frozen.ORDER))==16
    config=app.read(app.PACKAGE/"config.json");model=config["generation"]["model"]
    good=dict(status="success",answer="Fixture complete answer is available.",finish_reason="stop",requested_model=model,returned_model=model,
        usage=dict(prompt_tokens=7,completion_tokens=2,total_tokens=9))
    unknown=dict(status="unknown",requested_model=model)
    prompt="Offline ordinary fixture";stage="legacy/answer/2"
    payload=app.build_payload(config["generation"],prompt,128)
    checks=dict(all8_public_sources_rawD_actual_messages_exact=True,all16_only_actual_reader_block_changes=True,
        original_instructions_fewshots_history_rawCurrent_root_goalrule_preserved=True,no_gold_or_expected_answer_added=True)
    spec=importlib.util.spec_from_file_location("prior_m4_policy",app.frozen.EVIDENCE/"policies.py")
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    cases={};checked=0
    for row in app.rows(app.frozen.INVENTORY):
        case=cases.setdefault(row["source"],app.read(app.ROOT/row["source"]))
        node=next(n for n in case["nodes"]if n["step_id"]==row["step_id"])
        actual=selection.packet(node["original_retrieval"],node["rewritten_retrieval"],"M4")
        expected=prior.select(node["original_retrieval"],node["rewritten_retrieval"],node["request"],"M4_format_duplicate_fill")
        assert [d["doc_id"]for d in actual["docs"]]==[d["doc_id"]for d in expected]
        checked+=1
        if(row["id"],row["step_id"])==app.frozen.TARGET:
            assert "604e2bf53393913c1d2efa88274bbca4e5e46736e6ed4bcedf03043bfdc5275e"in{d["doc_id"]for d in actual["docs"]}
        if row["id"]=="hotpotqa:5ab3eeb5554299233954fff4"and row["step_id"]==2:
            assert actual["docs"]==selection.packet(node["original_retrieval"],node["rewritten_retrieval"],"M0")["docs"]
    assert checked==696
    fp=lambda source,text:selection.fingerprint(dict(source=source,text=text))
    assert fp("Same title","A (B).") == fp("Same title","A( B ).")
    for source,text in[("Other title","A (B)."),("Same title","a (B)."),("Same title","A (B)!"),("Same title","AB.")]:
        assert fp("Same title","A (B).")!=fp(source,text)
    for a,b in[("New York","NewYork"),("father","grandfather"),("1900","1901"),("5 cm","5 m"),("yes","not yes")]:
        assert fp("Title",a)!=fp("Title",b)
    assert templates.needs_formatter("one two three")and not templates.needs_formatter("one two three four")
    checks.update(M4_same_as_prior_rule_all696_nodes=True,Wiki721_recovered_HP7_unchanged=True,
        exact_title_ordered_tokens_word_boundaries_relations_dates_units_case_preserved=True)
    def expect(native,kind):
        try:native(stage,prompt,128)
        except kind:return
        raise AssertionError("Expected "+kind.__name__)
    with tempfile.TemporaryDirectory(prefix="M4_fake_",dir=app.OUT)as directory,ExitStack()as resources, \
            patch.object(shared,"provider_gate",fake_context),patch.object(shared.time,"sleep",lambda seconds:None):
        root=Path(directory)
        def open_case(name):
            out=root/name;out.mkdir(exist_ok=True)
            cache,budget=app.Cache(out/"checkpoint.sqlite3"),Budget(out/"ledger.sqlite3",out)
            resources.callback(cache.close);resources.callback(budget.close)
            return out,cache,budget
        def native(cache,budget,client,task="fixture/M0"):
            return app.NativeAsk(cache,Ask(cache,config,budget,client),config,task)
        out,cache,budget=open_case("fresh_resume")
        old=sqlite3.connect(out/"old.sqlite3");old.execute("CREATE TABLE requests(payload TEXT,result TEXT)")
        old.execute("INSERT INTO requests VALUES(?,?)",(app.canonical(payload),json.dumps({**good,"answer":"Old answer"})));old.commit();old.close()
        client=Client([good]);assert native(cache,budget,client)(stage,prompt,128)[1]=="Fixture complete answer is available."and client.calls==1
        cache.close();budget.close();out,cache,budget=open_case("fresh_resume")
        client=Client([]);assert native(cache,budget,client)(stage,prompt,128)[1]=="Fixture complete answer is available."and client.calls==0
        cache.close();budget.close();checks["fresh_seed_None_and_resume_no_reroll"]=True

        out,cache,budget=open_case("unknown_max5")
        client=Client([unknown]*5);expect(native(cache,budget,client),app.TechnicalHeld)
        assert client.calls==5 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==4
        cache.close();budget.close();out,cache,budget=open_case("unknown_max5")
        client=Client([]);expect(native(cache,budget,client),app.TechnicalHeld)
        assert client.calls==0 and budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0]==5
        cache.close();budget.close();checks["max5_durable_unknown_and5_reservations"]=True

        out,cache,budget=open_case("inflight")
        cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload));cache.close();budget.close()
        out,cache,budget=open_case("inflight");client=Client([unknown]*4);expect(native(cache,budget,client),app.TechnicalHeld)
        assert client.calls==4 and budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==5
        cache.close();budget.close();checks["inflight_restart_counts_original_attempt"]=True

        for field,value in [("response_id","id"),("usage",{}),("raw_response",{}),("returned_model",model),
                ("answer","partial"),("content","partial"),("refusal","declined"),("finish_reason","length")]:
            out,cache,budget=open_case("returned_"+field)
            rid,_=cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload))
            cache.finish_request(rid,{**unknown,field:value});cache.close();budget.close()
            out,cache,budget=open_case("returned_"+field);client=Client([]);expect(native(cache,budget,client),app.TechnicalHeld)
            assert client.calls==0 and budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==1
            cache.close();budget.close()
        checks["each_of8_returned_signs_holds_without_resubmission"]=True

        out,cache,budget=open_case("ledger_window")
        rid,_=cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload))
        budget.finish(rid,good);cache.close();budget.close();out,cache,budget=open_case("ledger_window")
        client=Client([]);expect(native(cache,budget,client),app.TechnicalHeld)
        assert client.calls==0 and budget.db.execute("SELECT actual FROM charges WHERE id=?",(rid,)).fetchone()[0]is not None
        cache.close();budget.close();checks["ledger_return_before_cache_crash_holds"]=True

        for returned in(False,True):
            out,cache,budget=open_case("transient_"+str(returned));rid,_=cache.begin_request("prior429",payload,before_start=lambda rid:budget.reserve(rid,payload))
            cache.finish_request(rid,dict(status="failed",http_status=429,**({"raw_response":{}}if returned else{})))
            client=Client([]if returned else[good]);ask=native(cache,budget,client)
            if returned:expect(ask,app.TechnicalHeld);assert client.calls==0
            else:assert ask(stage,prompt,128)[1]=="Fixture complete answer is available."and client.calls==1
            cache.close();budget.close()
        checks["known429_recovery_only_without_returned_evidence"]=True

        out,cache,budget=open_case("provider400")
        client=Client([dict(status="failed",http_status=400,error="data_inspection_failed",requested_model=model)])
        expect(native(cache,budget,client),app.TechnicalHeld);assert client.calls==1
        cache.close();budget.close();out,cache,budget=open_case("provider400")
        client=Client([]);expect(native(cache,budget,client),app.TechnicalHeld);assert client.calls==0
        assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0]==1
        cache.close();budget.close();checks["provider400_held_no_promptbypass_or_repeat"]=True

        for name,changes,kind in[("length",{"finish_reason":"length"},app.AlgorithmTerminal),
                ("filter",{"finish_reason":"content_filter"},app.AlgorithmTerminal),("refusal",{"refusal":"declined"},app.AlgorithmTerminal),
                ("empty",{"answer":""},app.AlgorithmTerminal),("model",{"returned_model":"wrong"},app.TechnicalHeld),
                ("missingusage",{"usage":None},app.TechnicalHeld),
                ("boolusage",{"usage":dict(prompt_tokens=True,completion_tokens=2,total_tokens=3)},app.TechnicalHeld),
                ("wrongsum",{"usage":dict(prompt_tokens=7,completion_tokens=2,total_tokens=11)},app.TechnicalHeld)]:
            out,cache,budget=open_case("terminal_"+name);client=Client([{**good,**changes}]);expect(native(cache,budget,client),kind)
            assert client.calls==1;cache.close();budget.close();out,cache,budget=open_case("terminal_"+name)
            client=Client([]);expect(native(cache,budget,client),kind);assert client.calls==0
            cache.close();budget.close()
        checks["4_algorithm_terminals_and4_invalid_receipts_never_reroll"]=True

        # Exercise actual new run loop with16 fake model calls, then fully
        # completed reentry must not construct Cache/client/budget/network.
        out=root/"runner16";out.mkdir();fixture_protocol=root/"fixture_protocol.json";app.atomic_json(fixture_protocol,{"fixture":True})
        fixture_config={**config,"budget":{**config["budget"],"ledger":str(out/"ledger.sqlite3")}}
        Budget(out/"ledger.sqlite3",out).close()
        original_read=app.read
        def read(path):return fixture_config if Path(path)==app.PACKAGE/"config.json"else original_read(path)
        short={**good,"answer":"Paris"}
        tasks=app.rows(app.frozen.ORDER);first_ordinary=next(i for i,t in enumerate(tasks)if t["role"]=="ordinary")
        short_indices={0,1,first_ordinary,first_ordinary+1}
        answers=[];expected=[]
        for i,t in enumerate(tasks):
            item=next(r for r in items if r["probe_id"]==t["probe_id"])
            expected.append(app.build_payload(config["generation"],templates.prompt(item,t["arm"]),128))
            answers.append(short if i in short_indices else good)
            if i in short_indices:
                expected.append(app.build_payload(config["generation"],templates.formatter_prompt(item,t["arm"],"Paris"),128))
                answers.append(good)
        client=Client(answers);factory=lambda config,output:Budget(out/"ledger.sqlite3",output)
        from scripts import cli_support
        from experiments.query_decomposition.p02_formal1500_v1_20261021 import network
        with patch.object(app,"OUT",out),patch.object(app.frozen,"PROTOCOL",fixture_protocol),patch.object(app,"read",read), \
                patch.object(app.frozen,"check",lambda:None),patch.object(app,"authorize",lambda:None), \
                patch.object(app,"NoCapBudget",factory),patch.object(app.SubmissionAwareStrictAsk,"_client",lambda self:client), \
                patch.object(cli_support,"temporary_api_key",fake_context),patch.object(network,"provider_dns",fake_context),patch.object(app,"emit",lambda value:None):
            result=app.run()
            assert result["terminal"]==16 and result["normal_pairs"]==8 and client.calls==20
            assert sum(v["formatter_triggered"]for v in result["by_arm"].values())==4
            assert client.payloads==expected
            assert all("max_tokens"in p and p["max_tokens"]==128 for p in client.payloads)
            with patch.object(app,"Cache",side_effect=AssertionError("Cache on completed reentry")), \
                    patch.object(app,"NoCapBudget",side_effect=AssertionError("Budget on completed reentry")):
                assert app.run()["terminal"]==16
            # Report must preserve in_flight rather than reopening via Cache.
            with closing(sqlite3.connect(out/"checkpoint.sqlite3"))as db:
                db.execute("UPDATE requests SET status='in_flight' WHERE id=(SELECT id FROM requests LIMIT 1)")
                db.commit()
            app.report()
            with closing(sqlite3.connect(out/"checkpoint.sqlite3"))as db:assert db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]==1
        checks["actual16fresh_QA_plus4_formatters_exactpayloads_bothroles_andarms"]=True
        checks["completed_reentry_and_report_are_readonly_noCache_noNetwork"]=True
        for name,last,status in[("formatter_length",{**good,"finish_reason":"length"},"algorithm_failure"),
                ("formatter400",dict(status="failed",http_status=400,error="data_inspection_failed"),"held_technical")]:
            out=root/name;out.mkdir();fixture_config={**config,"budget":{**config["budget"],"ledger":str(out/"ledger.sqlite3")}}
            Budget(out/"ledger.sqlite3",out).close();client=Client([short,last]);factory=lambda config,output:Budget(out/"ledger.sqlite3",output)
            with patch.object(app,"OUT",out),patch.object(app.frozen,"PROTOCOL",fixture_protocol),patch.object(app,"read",read), \
                    patch.object(app.frozen,"check",lambda:None),patch.object(app,"authorize",lambda:None),patch.object(app,"NoCapBudget",factory), \
                    patch.object(app.SubmissionAwareStrictAsk,"_client",lambda self:client),patch.object(cli_support,"temporary_api_key",fake_context), \
                    patch.object(network,"provider_dns",fake_context),patch.object(app,"emit",lambda value:None):
                result=app.run(limit=1);assert client.calls==2
                if status=="algorithm_failure":
                    outcome=app.snapshot()[0]
                    assert outcome["status"]==status and outcome["initial_answer"]=="Paris"and outcome["formatted_answer"]is None
                    assert outcome["failure_stage"].startswith("legacy/sentence_format/")and outcome["usage"]["prompt_tokens"]==14
                else:
                    assert result["terminal"]==0 and result["held_technical_tasks"]==1
                    held=next((out/"held").glob("*.json"));h=app.read(held)
                    assert h["initial_answer"]=="Paris"and h["formatter_triggered"]and len(h["completed_stages"])==1
                    assert sum(v["held_completed_stage_prompt_tokens"]for v in result["by_arm"].values())==7
                    assert result["new_actual_returned_usage_CNY"]>0 and result["unresolved_reserved_CNY"]==.1
                if status=="held_technical":
                    client=Client([]);again=app.run(limit=1);assert client.calls==0
        checks["formatter_length_terminal_and_formatter400held_keep_QA_text_usage_and_no_reissue"]=True
        # Independent partial reporter fixture: technical missing notquality0,
        # algorithm terminal retained, unresolved/orphan funds not discarded.
        out,cache,budget=open_case("partial_report")
        fixture_config={**config,"budget":{**config["budget"],"ledger":str(out/"ledger.sqlite3")}}
        tasks=app.rows(app.frozen.ORDER)
        for t in tasks[:-1]:cache.put("outcome",t["task_id"],{**t,"status":"success","answer":"Fixture complete answer is available.","usage":good["usage"]})
        app.atomic_json(out/"held/fixture.json",{**tasks[-1],"status":"held_technical"});budget.reserve("orphan",payload)
        with patch.object(app,"OUT",out),patch.object(app,"read",read),patch.object(app.frozen,"check",lambda:None):
            result=app.report(cache)
            assert result["terminal"]==15 and result["held_technical_tasks"]==1 and result["reservations_without_current_or_retired_cache_record_ids"]==["orphan"]
            assert not any(k in result for k in("f1","em","F1","EM","accuracy"))
            t=tasks[-1];cache.put("outcome",t["task_id"],{**t,"status":"algorithm_failure","answer":"partial","usage":good["usage"]})
            result=app.report(cache);assert result["terminal"]==16 and result["held_technical_tasks"]==0
        cache.close();budget.close();checks["partial_missing_and_terminal_denominators_and_unknown_cost_preserved"]=True
    assert not(app.OUT/"checkpoint.sqlite3").exists()and not(app.HERE/"RUN_AUTHORIZATION.json").exists()
    result=dict(passed=True,checks=checks,real_API_SSH_GPU=0,global_ledger_unchanged=True,
        fixtures="TemporarySQLite+fakeclient/budget/gate;noactualmodelanswersorendpointsopened")
    app.atomic_json(app.OUT/"OFFLINE_CONTRACTS.json",result)
    return result


if __name__=="__main__":print(json.dumps(validate(),ensure_ascii=True))
