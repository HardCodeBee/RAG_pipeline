"""Temporary SQLite/fake-provider contracts; never sends a provider request."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022 import run as app, prepare as frozen
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.runtime import CandidateAsk, receipt_integrity
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as shared
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.validate_runner_offline import Budget, Client, fake_gate


class Ask(CandidateAsk):
    def __init__(self,cache,config,budget,client):
        super().__init__(cache,None,config,budget,lambda value:None,maximum_attempts=5,recover_unknown=True)
        self.fixture_client=client
    def _client(self):
        return self.fixture_client


def validate():
    frozen.check()
    assert not(app.HERE/"RUN_AUTHORIZATION.json").exists() and not(app.OUT/"checkpoint.sqlite3").exists()
    app.OUT.mkdir(parents=True,exist_ok=True)
    config=app.read(app.PACKAGE/"config.json")
    model=config["generation"]["model"]
    good=dict(status="success",answer="Offline complete sentence.",finish_reason="stop",requested_model=model,returned_model=model,
        usage=dict(prompt_tokens=7,completion_tokens=2,total_tokens=9))
    unknown=dict(status="unknown",requested_model=model)
    prompt="Offline candidate fixture"
    payload=app.build_payload(config["generation"],prompt,128)
    checks={}
    def held(ask,p=prompt,stage="fixture"):
        try:ask(stage,p,128)
        except RuntimeError:return
        raise AssertionError("Expected held")
    with tempfile.TemporaryDirectory(prefix="B_fake_",dir=app.OUT) as directory,ExitStack() as resources, \
            patch.object(shared,"provider_gate",fake_gate),patch.object(shared.time,"sleep",lambda _:None):
        root=Path(directory)
        def open_case(name):
            out=root/name;out.mkdir(exist_ok=True)
            cache,budget=app.Cache(out/"checkpoint.sqlite3"),Budget(out/"ledger.sqlite3",out)
            resources.callback(cache.close);resources.callback(budget.close)
            return out,cache,budget
        # Missing root authorization fails before creation of paid-run state.
        with patch.object(app,"Cache",side_effect=AssertionError("Must not constructCache")), \
                patch.object(app,"NoCapBudget",side_effect=AssertionError("Must not open globalledger")):
            try:app.run(limit=1)
            except FileNotFoundError:pass
            else:raise AssertionError("Missing root authorization must prevent run")
        checks["no_authorization_blocks_before_Cache_ledger_key_network"]=True

        out,cache,budget=open_case("B0_import")
        items={r["probe_id"]:r for r in frozen.rows(frozen.INPUTS)}
        app.identity(cache);app.import_controls(cache,items);app.import_controls(cache,items)
        assert len(cache.all("outcome"))==8 and cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==0
        assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==0
        assert all(r["arm"]=="B0" and r["actual_new_CNY"]==0 for r in cache.all("outcome").values())
        cache.close();budget.close();checks["all8_B0_import_idempotent_no_request_or_charge"]=True

        out,cache,budget=open_case("dedup_resume")
        client=Client([good]);ask=Ask(cache,config,budget,client)
        first=ask("B1",prompt,128);second=ask("B2",prompt,128)
        assert first==second and client.calls==1 and cache.db.execute("SELECT count(*) FROM request_stages").fetchone()[0]==2
        cache.close();budget.close()
        out,cache,budget=open_case("dedup_resume")
        client=Client([]);ask=Ask(cache,config,budget,client);assert ask("restart",prompt,128)==first and client.calls==0
        cache.close();budget.close();checks["exact_payload_share_and_restart_never_reroll"]=True

        out,cache,budget=open_case("max5")
        client=Client([unknown]*5);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==5 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==4
        cache.close();budget.close()
        out,cache,budget=open_case("max5")
        client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==0 and budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0]==5
        cache.close();budget.close();checks["max5_durable_including_initial_preserves_reservations"]=True

        traces=[("response_id","id"),("raw_response",{}),("usage",{}),("returned_model",model),
            ("answer","partial"),("content","partial"),("refusal","declined"),("refusal",""),
            ("finish_reason","stop"),("finish_reason","length"),("finish_reason","")]
        for i,(field,value) in enumerate(traces):
            out,cache,budget=open_case("trace"+str(i))
            rid,_=cache.begin_request("interrupted",payload,before_start=lambda rid:budget.reserve(rid,payload))
            cache.finish_request(rid,{**unknown,field:value});cache.close();budget.close()
            out,cache,budget=open_case("trace"+str(i))
            client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
            assert client.calls==0 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==0
            assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==1
            cache.close();budget.close()
        checks["all11_return_traces_including_refusal_finish_only_hold_after_restart"]=True

        out,cache,budget=open_case("ledger_crash")
        rid,_=cache.begin_request("inflight",payload,before_start=lambda rid:budget.reserve(rid,payload))
        budget.finish(rid,good);cache.close();budget.close()
        out,cache,budget=open_case("ledger_crash")
        client=Client([]);ask=Ask(cache,config,budget,client);held(ask)
        assert client.calls==0 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0]==0
        cache.close();budget.close();checks["ledger_returned_usage_cache_crash_window_held"]=True

        for name,changes,terminal in [
                ("length",dict(finish_reason="length"),"terminal_length"),
                ("filter",dict(finish_reason="content_filter"),"terminal_content_filter"),
                ("refusal",dict(refusal="declined"),"terminal_refusal"),
                ("empty",dict(answer=""),"terminal_empty"),
                ("missingusage",dict(usage=None),None),
                ("badmodel",dict(returned_model="wrong"),None),
                ("boolusage",dict(usage=dict(prompt_tokens=True,completion_tokens=2,total_tokens=3)),None),
                ("badsum",dict(usage=dict(prompt_tokens=7,completion_tokens=2,total_tokens=10)),None)]:
            value={**deepcopy(good),**changes}
            assert not app.normal_complete(value,model) and app.algorithm_failure(value,model)==terminal
            if terminal:
                out,cache,budget=open_case("terminal"+name)
                client=Client([value]);ask=Ask(cache,config,budget,client);held(ask,stage="B1");held(ask,stage="B2")
                assert client.calls==1 and budget.db.execute("SELECT count(*) FROM charges").fetchone()[0]==1
                cache.close();budget.close()
        checks["algorithm_terminals_shared_no_reroll_invalid_model_usage_never_terminal"]=True

        out,cache,budget=open_case("report24")
        app.identity(cache);app.import_controls(cache,items)
        client=Client([good]*13);ask=Ask(cache,config,budget,client)
        for task in frozen.rows(frozen.ORDER):
            item=items[task["probe_id"]];text=frozen.templates.prompt(item,task["arm"])
            rid,answer,finish=ask(task["task_id"],text,128)
            cache.put("outcome",task["task_id"],{**task,"status":"success","answer":answer,"request_id":rid,
                "usage":good["usage"],"finish_reason":finish,"utc":app.utc()})
        assert client.calls==13 and cache.db.execute("SELECT count(*) FROM requests").fetchone()[0]==13
        budget.reserve("orphan_fixture",payload)
        original_read=app.read
        def fixture_read(path):
            if Path(path)==app.PACKAGE/"config.json":return {**config,"budget":{**config["budget"],"ledger":str(out/"ledger.sqlite3")}}
            return original_read(path)
        with patch.object(app,"OUT",out),patch.object(app,"read",fixture_read):
            result=app.report(cache)
        assert result["terminal"]==24 and result["status"]=="all_conditions_terminal"
        assert result["distinct_candidate_returned_receipts"]==13 and result["comparisons"]["B1_to_B2"]["same_actual_receipt_pairs"]==3
        assert result["B0_new_actual_CNY"]==0 and result["B0_attributed_already_charged_CNY"]>0
        assert result["reservations_without_cache_record_ids"]==["orphan_fixture"] and result["unresolved_reserved_CNY"]>0
        assert all(k not in result for k in ("f1","F1","em","EM","accuracy"))
        assert result["candidate_unique_input_tokens"]==13*7
        checks["24logical_13distinct_3shared_receipts_B0_fees_separate_orphan_preserved"]=True
        original=cache.get("meta","identity")
        with patch.object(cache,"get",return_value={**original,"inputs_sha256":"changed"}):
            try:app.identity(cache)
            except AssertionError:pass
            else:raise AssertionError("Identity change must hold")
        checks["changed_resume_identity_rejected"]=True
        cache.close();budget.close()
    assert not(app.OUT/"checkpoint.sqlite3").exists() and not(app.HERE/"RUN_AUTHORIZATION.json").exists()
    result=dict(passed=True,checks=checks,real_provider_requests=0,global_ledger_modified=False,
        production_Cache_created=False,fixture="TemporarySQLite, fakeclient/fakebudget/fakegate",protocol_sha256=app.sha(frozen.PROTOCOL))
    app.atomic_json(app.OUT/"OFFLINE_VALIDATION.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(validate(),ensure_ascii=True))
