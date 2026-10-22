"""Fake provider + temporary SQLite fixtures; never opens a real provider client."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import tempfile
from unittest.mock import patch

from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import run as app
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as shared


@contextmanager
def fake_gate(*args,**kwargs):
    yield


class Budget:
    def __init__(self,path,output):
        self.db=sqlite3.connect(path)
        self.namespace="p02_prompt_research_v1_20261022/"+str(output.resolve())
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,namespace TEXT,reserved REAL,actual REAL,status TEXT)")
        self.db.commit()
    def reserve(self,rid,payload):
        with self.db:
            self.db.execute("INSERT INTO charges VALUES(?,?,.1,NULL,'reserved')",(rid,self.namespace))
    def finish(self,rid,result):
        if result.get("usage"):
            with self.db:
                self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?",(app.cost(result),rid))
    def close(self):
        self.db.close()


class Client:
    def __init__(self,answers):
        self.answers=deepcopy(answers)
        self.calls=0
    def request_once(self,payload):
        self.calls+=1
        assert self.answers,"Unexpected fake provider resubmission"
        return self.answers.pop(0)
    def close(self):
        pass


class Ask(app.SubmissionAwareStrictAsk):
    def __init__(self,cache,config,budget,client):
        super().__init__(cache,None,config,budget,lambda event:None,maximum_attempts=5,recover_unknown=True)
        self.fixture_client=client
    def _client(self):
        return self.fixture_client


def validate():
    app.check()
    config=app.read(app.PACKAGE/"config.json")
    model=config["generation"]["model"]
    good=dict(status="success",answer="Fixture answer",finish_reason="stop",requested_model=model,returned_model=model,
        usage=dict(prompt_tokens=7,completion_tokens=2,total_tokens=9))
    unknown=dict(status="unknown",requested_model=model)
    prompt="Offline fixture prompt"
    payload=app.build_payload(config["generation"],prompt,128)
    checks={}
    def held(ask):
        try:
            ask("fixture",prompt,128)
        except RuntimeError:
            return
        raise AssertionError("Expected held outcome")
    def open_case(root,name):
        out=root/name;out.mkdir(exist_ok=True)
        cache,budget=app.Cache(out/"checkpoint.sqlite3"),Budget(out/"ledger.sqlite3",out)
        resources.callback(cache.close)
        resources.callback(budget.close)
        return out,cache,budget
    with tempfile.TemporaryDirectory(prefix="runner_fixtures_",dir=app.OUT) as directory,ExitStack() as resources, \
            patch.object(shared,"provider_gate",fake_gate),patch.object(shared.time,"sleep",lambda seconds:None):
        root=Path(directory)
        # A pre-existing historical receipt is deliberately not supplied as seed.
        out,cache,budget=open_case(root,"fresh_resume")
        old=sqlite3.connect(out/"historical.sqlite3")
        old.execute("CREATE TABLE requests(payload TEXT,result TEXT)")
        old.execute("INSERT INTO requests VALUES(?,?)",(app.canonical(payload),json.dumps({**good,"answer":"Historical answer"})))
        old.commit();old.close()
        client=Client([good]);ask=Ask(cache,config,budget,client)
        assert ask("first",prompt,128)[1]=="Fixture answer" and client.calls==1
        cache.close();budget.close()
        out,cache,budget=open_case(root,"fresh_resume")
        client=Client([]);ask=Ask(cache,config,budget,client)
        assert ask("second",prompt,128)[1]=="Fixture answer" and client.calls==0
        assert cache.db.execute("SELECT count(*) FROM request_stages").fetchone()[0]==2
        cache.close();budget.close();checks["fresh_seed_None_and_success_resume_no_reroll"]=True

        out,cache,budget=open_case(root,"max5")
        client=Client([unknown]*5);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==5 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==4
        assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==5
        cache.close();budget.close()
        out,cache,budget=open_case(root,"max5")
        client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==0 and budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0]==5
        cache.close();budget.close();checks["unknown_max5_durable_and_all5_reservations_preserved"]=True

        out,cache,budget=open_case(root,"inflight")
        cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload))
        cache.close();budget.close()
        out,cache,budget=open_case(root,"inflight")
        client=Client([unknown]*4);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==4 and budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==5
        cache.close();budget.close();checks["inflight_restart_counts_original_attempt"]=True

        for field,value in [("response_id","fixtureid"),("usage",{}),("raw_response",{}),("returned_model",model),("answer","partial"),("content","partial")]:
            out,cache,budget=open_case(root,"returned_"+field)
            rid,_=cache.begin_request("recorded",payload,before_start=lambda rid:budget.reserve(rid,payload))
            cache.finish_request(rid,{**unknown,field:value})
            cache.close();budget.close()
            out,cache,budget=open_case(root,"returned_"+field)
            client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
            assert client.calls==0 and budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==1
            assert cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==0
            cache.close();budget.close()
        checks["unknown_with_each_returned_trace_never_resubmits_after_restart"]=True

        out,cache,budget=open_case(root,"ledger_window")
        rid,_=cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload))
        budget.finish(rid,good)  # Simulated crash before cache.finish_request.
        cache.close();budget.close()
        out,cache,budget=open_case(root,"ledger_window")
        client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==0 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==0
        assert budget.db.execute("SELECT actual FROM charges WHERE id=?",(rid,)).fetchone()[0] is not None
        cache.close();budget.close();checks["ledger_returned_usage_cache_crash_window_held"]=True

        for returned in (False,True):
            out,cache,budget=open_case(root,"transient_"+str(returned))
            rid,_=cache.begin_request("prior429",payload,before_start=lambda rid:budget.reserve(rid,payload))
            cache.finish_request(rid,dict(status="failed",http_status=429,**({"raw_response":{"parsed":"partial"}} if returned else {})))
            client=Client([] if returned else [good]);ask=Ask(cache,config,budget,client)
            if returned:
                held(ask);assert client.calls==0
            else:
                assert ask("recover429",prompt,128)[1]=="Fixture answer" and client.calls==1
            cache.close();budget.close()
        checks["known429_without_return_can_recover_but_returned_evidence_holds"]=True

        for name,changes,terminal in [
                ("length",{"finish_reason":"length"},"terminal_length"),
                ("filter",{"finish_reason":"content_filter"},"terminal_content_filter"),
                ("refusal",{"refusal":"declined"},"terminal_refusal"),
                ("empty",{"answer":""},"terminal_empty"),
                ("model",{"returned_model":"wrong"},None),
                ("missing_usage",{"usage":None},None),
                ("bool_usage",{"usage":dict(prompt_tokens=True,completion_tokens=2,total_tokens=3)},None),
                ("inconsistent_total",{"usage":dict(prompt_tokens=7,completion_tokens=2,total_tokens=10)},None)]:
            value={**good,**changes}
            assert not app.normal_complete(value,model)
            assert app.algorithm_failure(value,model)==terminal
            if terminal:
                out,cache,budget=open_case(root,"algorithm_"+name)
                client=Client([value]);ask=Ask(cache,config,budget,client);held(ask)
                assert client.calls==1
                cache.close();budget.close()
                out,cache,budget=open_case(root,"algorithm_"+name)
                client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
                assert client.calls==0
                cache.close();budget.close()
        checks["all4_algorithm_terminals_never_reroll_and_invalid_identity_usage_hold"]=True

        # Real temporary Cache/report with 15/16 synthetic outcomes and one held;
        # only its temporary ledger is visible to the reporting function.
        out,cache,budget=open_case(root,"report")
        original_read=app.read
        def fixture_read(path):
            if Path(path)==app.PACKAGE/"config.json":
                return {**config,"budget":{**config["budget"],"ledger":str(out/"ledger.sqlite3")}}
            return original_read(path)
        tasks=app.rows(app.HERE/"TASK_ORDER16.frozen.jsonl")
        inputs={r["probe_id"]:r for r in app.rows(app.frozen.INPUTS)}
        app.identity(cache)
        for t in tasks[:-1]:
            i=inputs[t["probe_id"]]
            cache.put("outcome",t["task_id"],{**t,"status":"success","answer":"Fixture answer","usage":good["usage"],"dataset":i["dataset"]})
        app.atomic_json(out/"held/fixture.json",{**tasks[-1],"status":"held_technical"})
        budget.reserve("orphan_fixture_reservation",payload)
        with patch.object(app,"OUT",out),patch.object(app,"read",fixture_read):
            partial=app.report(cache)
            assert partial["status"]=="partial" and partial["terminal"]==15 and partial["held_technical_tasks"]==1
            assert partial["reservations_without_current_or_retired_cache_record_ids"]==["orphan_fixture_reservation"]
            last=tasks[-1];i=inputs[last["probe_id"]]
            cache.put("outcome",last["task_id"],{**last,"status":"algorithm_failure","reason":"terminal_length","answer":"partial","usage":good["usage"],"dataset":i["dataset"]})
            full=app.report(cache)
            assert full["status"]=="all_conditions_terminal" and full["terminal"]==16 and full["held_technical_tasks"]==0
            assert sum(a["algorithm_terminal"] for a in full["by_arm"].values())==1
            assert all(k not in full for k in ("F1","EM","accuracy","f1","em"))
            app.identity(cache);app.identity(cache)
            original=cache.get("meta","identity")
            with patch.object(cache,"get",return_value={**original,"inputs_sha256":"changed"}):
                try:
                    app.identity(cache)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Changed checkpoint identity must hold")
        cache.close();budget.close()
        checks["report_separates15_plusheld_then16_terminal_no_fake_F1"]=True
        checks["orphan_reservation_preserved_and_reported_not_deleted"]=True
        checks["resume_identity_mismatch_rejected"]=True
    assert not(app.OUT/"checkpoint.sqlite3").exists() and not(app.HERE/"RUN_AUTHORIZATION.json").exists()
    result=dict(passed=True,checks=checks,new_real_provider_requests=0,real_provider_client_created=False,
        fixtures="Temporary local SQLite with fake client/budget/gate only; original inputs and global ledger untouched",
        execution_protocol_sha256=app.sha(app.EXECUTION))
    app.atomic_json(app.OUT/"OFFLINE_RUNNER_VALIDATION.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(validate(),ensure_ascii=True))
