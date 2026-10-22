"""Pure flow/driver integration checks; all providers and ledgers are temporary fakes."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline
from . import context,flow,run as app
from .message_transport import build_payload
from .native_ask import NativeAsk,AlgorithmTerminal,TechnicalHeld
from .validate_messages_offline import CONFIG,fixture,no_real_submission,strict_for,good


def packet(query,qid="synthetic"):
    docs=[]
    for i in range(10):
        title=f"Synthetic document {i}"
        text=f"{title}. Alpha and its candidate parent have a documented birthplace."
        docs.append(dict(doc_id=hashlib.sha256((query+str(i)).encode()).hexdigest(),source=title,
            text=text,rank=i+1,block=f"[Chunk {i+1} | source={title}, pages=1-1]\n{text}"))
    return dict(qid=qid,query=query,k=10,docs=docs)


class Provider:
    def __init__(self,short=False,length_first=False):
        self.payloads=[];self.short=short;self.length_first=length_first

    def request_once(self,payload):
        self.payloads.append(deepcopy(payload));text=payload["messages"][-1]["content"]
        if self.length_first and len(self.payloads)==1:return good("Partial content ends here.","length")
        if payload["max_tokens"]==256:
            person="Gamma"if "Gamma is the"in text else"Beta"
            return good(f"Where was {person} born?")
        if payload["max_tokens"]==64:return good("Synthetic City")
        if text.startswith("Express the candidate answer below"):
            return good("Alpha is the accepted formatted answer.")
        if self.short:return good("Alpha")
        sid=int(re.findall(r"Current subquestion #(\d+):",text)[-1])
        expanded=len(payload["messages"])>1 or text.startswith("User:\n")
        if sid==2:return good(("Gamma"if expanded else"Beta")+" is the candidate parent of Alpha.")
        return good("Alpha has an established complete factual answer.")

    def close(self):pass


CASE=dict(id="2wiki:synthetic",qid="synthetic",dataset="2wiki",question="Where was Alpha's parent born?",
    supporting_questions=["Who is Alpha?","Who is Alpha's parent?","Where was Alpha's parent born?"])


def fake_ask(provider,events):
    def ask(stage,value,maximum,allow_final_truncation=False):
        payload=build_payload(CONFIG["generation"],value,maximum)
        events.append((stage,deepcopy(value),maximum,allow_final_truncation))
        receipt=provider.request_once(payload)
        return app.digest(payload),receipt["answer"],receipt["finish_reason"]
    return ask


def flow_contracts():
    original_calls=[];reset_calls=[]
    original_queries=[];reset_queries=[]
    original=Pipeline().execute(CASE,lambda q:(original_queries.append(q)or packet(q)),fake_ask(Provider(),original_calls))
    reset=flow.execute_arm(CASE,lambda q:(reset_queries.append(q)or packet(q)),fake_ask(Provider(),reset_calls),"reset")
    assert original_calls==reset_calls and original_queries==reset_queries
    for key in original:
        if key!="nodes":assert original[key]==reset[key],key
    for old,new in zip(original["nodes"],reset["nodes"]):
        assert all(old[key]==new[key]for key in old)
    runs={}
    for arm in flow.ARMS:
        calls=[];queries=[]
        result=flow.execute_arm(CASE,lambda q:(queries.append(q)or packet(q)),fake_ask(Provider(),calls),arm)
        nodes=result["nodes"]
        for sid,node in enumerate(nodes,1):
            wire=node["shared_context"]["wire_input"]
            history=[{"role":role,"content":content}for old in nodes[:sid-1]
                     for role,content in(("user",old["child_prompt"]),("assistant",old["answer"]))]
            history.append(dict(role="user",content=node["child_prompt"]))
            if arm=="chat"and sid>1:assert wire==history
            elif arm=="flat"and sid>1:
                assert wire=="\n\n".join(("User:\n"if row["role"]=="user"else"Assistant:\n")+row["content"]for row in history)
            else:assert wire==node["child_prompt"]
        assert all(isinstance(value,str)for stage,value,_,_ in calls if not stage.startswith("legacy/answer/"))
        assert all(cap==128 for stage,_,cap,_ in calls if stage.startswith("legacy/answer/"))
        assert all(cap==256 for stage,_,cap,_ in calls if stage.startswith("legacy/rewrite/"))
        assert calls[-1][0]=="legacy/final"and calls[-1][2:]==(64,True)
        assert result["original_retrieval"]==nodes[-1]["original_retrieval"]
        assert result["final_context"].endswith("\n\n".join(d["block"]for d in nodes[-1]["original_retrieval"]["docs"]))
        runs[arm]=(result,queries,calls)
    assert "Where was Beta born?"in runs["reset"][1]
    assert "Where was Gamma born?"in runs["chat"][1]and"Where was Gamma born?"in runs["flat"][1]
    # Each archived prompt is the current native string, never an old flat/chat
    # wire. No instruction/example/history deduplication is silently introduced.
    assert all(node["raw_P02_child_prompt"]==node["child_prompt"]for value,_,_ in runs.values()for node in value["nodes"])
    return dict(reset_full_trajectory_and_every_wire_equal_core=True,
        flat_chat_exact_same_content_order_different_roles=True,
        every_ordinary_and_goal_uses_context_only_after_first=True,
        own_changed_answers_drive_real_subsequent_rewrite_and_retrieval=True,
        rewrite_formatter_final_string_interfaces_and_native_caps_root_unchanged=True,
        no_previous_expanded_wire_nested=True)


def formatting_and_isolation():
    for arm in flow.ARMS:
        calls=[];seen=[]
        result=flow.execute_arm(CASE,packet,fake_ask(Provider(short=True),calls),arm,record_node=seen.append)
        assert len(seen)==len(CASE["supporting_questions"])+1
        for node in result["nodes"]:
            assert node["initial_answer"]=="Alpha"
            assert node["answer"]=="Alpha is the accepted formatted answer."
            assert node["sentence_format_request_id"]
            sid=node["step_id"]
            fmt=next(value for stage,value,_,_ in calls if stage==f"legacy/sentence_format/{sid}")
            assert isinstance(fmt,str)and("Question:\n"+node["request"]+"\n\n")in fmt
            if arm=="chat"and sid>1:
                assert all(row["content"]==node["answer"]for row in node["shared_context"]["wire_input"][1::2])
        again=flow.execute_arm(CASE,packet,fake_ask(Provider(),[]),arm)
        assert isinstance(again["nodes"][0]["shared_context"]["wire_input"],str)
        goalonly={**CASE,"supporting_questions":[]}
        alonecalls=[];alone=flow.execute_arm(goalonly,packet,fake_ask(Provider(),alonecalls),arm)
        assert alone["nodes"][0]["role"]=="goal"and alone["nodes"][0]["shared_context"]["historical_turns"]==0
        assert isinstance(alonecalls[0][1],str)
    calls=[]
    def failed_record(node):raise RuntimeError("Synthetic recorder failure")
    try:flow.execute_arm(CASE,packet,fake_ask(Provider(),calls),"chat",record_node=failed_record)
    except RuntimeError:pass
    else:raise AssertionError("Recorder failure must stop before commit/next turn")
    assert [stage for stage,_,_,_ in calls]==["legacy/answer/1"]
    return dict(ordinary_and_goal_short_answer_formatter_preserved_allarms=True,
        accepted_post_formatter_assistant_not_raw_initial=True,
        record_node_precedes_commit_and_failure_prevents_next_turn=True,
        goal_only_single_user_equivalence=True,new_original_query_has_empty_context=True)


def gate_and_resume():
    tasks=[row for row in app.rows(app.ORDER)if row["engineering_initial"]]
    data={row["id"]:row for row in app.rows(app.INPUTS)}
    assert len(tasks)==18
    with fixture()as(cache,budget),no_real_submission():
        provider=Provider();strict=strict_for(cache,budget,provider)
        def execute(task):
            native=NativeAsk(cache,strict,CONFIG,task["task_id"])
            partial=[]
            def save_event(value):
                if value["event"]=="shared_context_request":
                    key=task["task_id"]+"/"+value["stage"]
                    previous=cache.get("qa_input",key);assert previous is None or previous==value
                    cache.put("qa_input",key,value)
            case=data[task["id"]]
            try:
                result=flow.execute_arm(case,lambda q:packet(q,case["qid"]),native,task["arm"],
                    record_node=partial.append,record_event=save_event)
                value={**task,"status":result["status"],"trajectory":result,"stages":native.stages}
            except AlgorithmTerminal as exc:
                value={**task,"status":"algorithm_failure","partial_nodes":partial,"stages":native.stages,
                    "terminal_stage":exc.value["stage"],"terminal_reason":exc.value["terminal_reason"]}
            cache.put("outcome",task["task_id"],value)
            return value
        assert not app.engineering_gate(cache,tasks,CONFIG)["passed"]
        for task in tasks:execute(task)
        gate=app.engineering_gate(cache,tasks,CONFIG)
        assert gate["passed"]and gate["planned_conditions"]==18 and not gate["quality_or_predictions_inspected"]
        count=len(provider.payloads)
        for task in tasks:execute(task)
        assert len(provider.payloads)==count,"Resume must rebuild identical full messages without another submission"
        assert app.engineering_gate(cache,tasks,CONFIG)==gate
        # A corrupted wire must stop the engineering gate. Only temporary fake
        # SQLite is mutated; no production cache or record is opened here.
        key,raw=cache.db.execute("SELECT key,value FROM data WHERE kind='qa_input' LIMIT 1").fetchone()
        value=json.loads(raw);value["wire_input"]="Another query must not leak here"
        with cache.db:cache.db.execute("UPDATE data SET value=? WHERE kind='qa_input'AND key=?",(json.dumps(value),key))
        assert not app.engineering_gate(cache,tasks,CONFIG)["passed"]
        with cache.db:cache.db.execute("UPDATE data SET value=? WHERE kind='qa_input'AND key=?",(raw,key))
        assert app.engineering_gate(cache,tasks,CONFIG)["passed"]
        strict.close()
    # Valid length is a retained algorithm outcome, not an efficacy gate.
    with fixture()as(cache,budget),no_real_submission():
        provider=Provider(length_first=True);strict=strict_for(cache,budget,provider)
        task=tasks[0];native=NativeAsk(cache,strict,CONFIG,task["task_id"]);case=data[task["id"]]
        def capture(value):
            if value["event"]=="shared_context_request":cache.put("qa_input",task["task_id"]+"/"+value["stage"],value)
        try:flow.execute_arm(case,packet,native,task["arm"],record_event=capture)
        except AlgorithmTerminal as exc:
            outcome={**task,"status":"algorithm_failure","partial_nodes":[],"stages":native.stages,
                "terminal_stage":exc.value["stage"],"terminal_reason":exc.value["terminal_reason"]}
            cache.put("outcome",task["task_id"],outcome)
        else:raise AssertionError("Expected synthetic length terminal")
        assert app.engineering_gate(cache,[task],CONFIG)["passed"]
        assert len(provider.payloads)==1
        strict.close()
    return dict(engineering18_contract_gate_without_gold_prediction_or_score=True,
        complete_stage_receipts_and_perquery_wire_reconstruction_verified=True,
        resume_same_messages_zero_new_submissions=True,
        technical_missing_and_context_corruption_prevent_expansion=True,
        algorithm_terminal_retained_not_quality_screened=True)


def terminal_gate_branches():
    task=next(row for row in app.rows(app.ORDER)if row["engineering_initial"])
    case=next(row for row in app.rows(app.INPUTS)if row["id"]==task["id"])
    assert case["supporting_questions"]
    goalsid=len(case["supporting_questions"])+1
    results=[]
    for branch in("ordinary_QA","goal_QA","ordinary_formatter","goal_formatter","rewrite"):
        class TerminalProvider(Provider):
            def request_once(self,payload):
                text=payload["messages"][-1]["content"]
                formatting=text.startswith("Express the candidate answer below")
                qa=payload["max_tokens"]==128 and not formatting
                sid=int(re.findall(r"Current subquestion #(\d+):",text)[-1])if qa else None
                fail=(branch=="ordinary_QA"and qa and sid==1
                    or branch=="goal_QA"and qa and sid==goalsid
                    or branch=="ordinary_formatter"and formatting and("Question:\n"+case["supporting_questions"][0]+"\n\n")in text
                    or branch=="goal_formatter"and formatting and("Question:\n"+case["question"]+"\n\n")in text
                    or branch=="rewrite"and payload["max_tokens"]==256)
                if fail:
                    self.payloads.append(deepcopy(payload));return good("Valid partial algorithm output.","length")
                return super().request_once(payload)
        with fixture()as(cache,budget),no_real_submission():
            provider=TerminalProvider(short="formatter"in branch);strict=strict_for(cache,budget,provider)
            def execute():
                native=NativeAsk(cache,strict,CONFIG,task["task_id"]);partial=[]
                def save(value):
                    if value["event"]=="shared_context_request":
                        key=task["task_id"]+"/"+value["stage"]
                        previous=cache.get("qa_input",key);assert previous is None or previous==value
                        cache.put("qa_input",key,value)
                try:flow.execute_arm(case,lambda q:packet(q,case["qid"]),native,task["arm"],record_node=partial.append,record_event=save)
                except AlgorithmTerminal as exc:
                    result={**task,"status":"algorithm_failure","partial_nodes":partial,"stages":native.stages,
                        "terminal_stage":exc.value["stage"],"terminal_reason":exc.value["terminal_reason"]}
                    cache.put("outcome",task["task_id"],result)
                    return result
                raise AssertionError("Expected synthetic algorithm failure: "+branch)
            result=execute();count=len(provider.payloads)
            assert app.engineering_gate(cache,[task],CONFIG)["passed"],branch
            again=execute();assert again==result and len(provider.payloads)==count
            assert app.engineering_gate(cache,[task],CONFIG)["passed"]
            results.append(dict(branch=branch,terminal_stage=result["terminal_stage"],accepted_by_engineering_gate=True,
                no_new_submission_on_resume=True))
            strict.close()
    return results


def preparation_and_expense():
    result=app.prepare();tasks=app.rows(app.ORDER);data=app.rows(app.INPUTS)
    assert result["conditions"]==270 and len(data)==90 and len(tasks)==270
    assert {row["id"]for row in tasks[:18]}==set(app.initial_ids())
    assert all(row["engineering_initial"]for row in tasks[:18])and not any(row["engineering_initial"]for row in tasks[18:])
    assert {arm:sum(row["arm"]==arm for row in tasks)for arm in flow.ARMS}=={arm:90 for arm in flow.ARMS}
    old={row["id"]:row for row in app.rows(app.HISTORY)}
    assert len(old)==90 and all(row["historical_quality_available"]for row in old.values())
    base=dict(actual_new_returned_usage_CNY=2.,historical_complete_query_CNY=1.,eligible_for_complete_paid_query_trend=True,utc="t")
    assert not app.expense_reasons(base,[base]*19)
    assert "recent20_complete_paid_queries_above1.5x_historical_references"in app.expense_reasons(base,[base]*20)
    high={**base,"actual_new_returned_usage_CNY":4.}
    assert "paid_query_above3x_historical_complete_query_reference"in app.expense_reasons(high,[])
    assert not(app.OUT/"checkpoint.sqlite3").exists()and not(app.HERE/"RUN_AUTHORIZATION.json").exists()
    return dict(same90_threebalancedarms_270_no_quality_selection=True,
        sixengineering_queries_predeclared_18_then_remaining252=True,
        original_native_historical_reference90_separate_from_freshreset=True,
        expense_recent20_counts_completequery_not_stages=True,
        no_authorization_checkpoint_or_real_submission_created=True)


def main():
    contracts=dict(preparation=preparation_and_expense(),flow=flow_contracts(),
        formatting_and_isolation=formatting_and_isolation(),engineering_resume=gate_and_resume(),
        legitimate_algorithm_terminals_do_not_block_gate=terminal_gate_branches())
    result=dict(passed=True,contracts=contracts,new_API_requests=0,new_SSH_requests=0,new_GPU_jobs=0,
        note="Only fake providers/temporary ledgers; no fresh endpoint or gold used by engineering gate")
    app.atomic_json(HERE/"OFFLINE_CONTRACTS.json",result)
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
