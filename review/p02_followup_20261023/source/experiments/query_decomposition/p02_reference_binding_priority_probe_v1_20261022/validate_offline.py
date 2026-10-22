"""Small new-interface contracts; all provider/network entry points are fake.

The frozen transport/native runtime's durable max-five/ledger and model/usage
contracts remain bound dependencies rather than reimplemented here.
"""
from contextlib import ExitStack,closing,contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy
import json
import sqlite3

from . import run as app,templates


@contextmanager
def fixture_db(path):
    with closing(sqlite3.connect(path))as db:
        with db:yield db


class FakeNative:
    def __init__(self,responses):self.responses=list(responses);self.stages=[];self.calls=[]
    def __call__(self,stage,prompt,maximum):
        self.calls.append((stage,prompt,maximum));response=self.responses.pop(0)
        if response=="technical":raise app.TechnicalHeld("fake technical hold; no provider")
        terminal=response=="terminal"
        value=dict(stage=stage,request_id="fake_"+str(len(self.calls)),answer="truncated"if terminal else response,
            finish_reason="length"if terminal else"stop",usage=dict(prompt_tokens=10,completion_tokens=4,total_tokens=14),
            terminal_reason="terminal_length"if terminal else None)
        self.stages.append(value)
        if terminal:raise app.AlgorithmTerminal(value)
        return value["request_id"],value["answer"],value["finish_reason"]


def mapping_contracts():
    def item(raw,rewrite,answer):return dict(current_question=raw,accepted_rewrite=rewrite,step_id=2,
        history=[dict(step_id=1,request="What is the first answer?",answer=answer)])
    value=item("Who is Liu E's spouse's father?","Who is Liu Cong's father?","Liu Cong was Liu E's spouse.")
    binding=templates.bindings(value)[0]
    assert binding["raw_text"]=="Liu E's spouse"and binding["candidate_text"]=="Liu Cong"
    value=item("Where was the writer born?","Where was MARIA Smith born?","The writer is Maria Smith.")
    assert templates.bindings(value)[0]["candidate_text"]=="Maria Smith"
    assert not templates.bindings(item("Where was he born?","Where was John born?","No name given."))
    assert not templates.bindings(item("Where was John born?","Where was of John born?","The full name is of John."))
    assert not templates.bindings(item("Where was he born?","Where was he born?","John was born there."))
    assert not templates.bindings(item("Is it?","Is it!","Yes!"))
    # A prior request/reader cannot supply the right-hand value, only ANSWER.
    value=item("Where was he born?","Where was John born?","No identity established.")
    value["history"][0]["request"]="John";value["reader_packet"]={"docs":[{"block":"John"}]}
    assert not templates.bindings(value)


def perform_contracts(items):
    item=items[0]
    for arm in templates.ARMS:
        native=FakeNative(["one two three","This is a complete factual sentence."])
        result=app.perform(item,arm,native)
        assert result["status"]=="success"and result["formatter_triggered"]and len(native.calls)==2
        assert result["initial_answer"]=="one two three"and result["formatted_answer"]=="This is a complete factual sentence."
        assert native.calls[0]==("legacy/answer/"+str(item["step_id"]),templates.prompt(item,arm),128)
        assert native.calls[1]==("legacy/sentence_format/"+str(item["step_id"]),templates.formatter_prompt(item,"one two three"),128)
        assert result["usage"]==dict(prompt_tokens=20,completion_tokens=8,total_tokens=28)
        native=FakeNative(["one two three four"]);result=app.perform(item,arm,native)
        assert not result["formatter_triggered"]and len(native.calls)==1
    native=FakeNative(["one two three","technical"]);result=app.perform(item,"C0",native)
    assert result["status"]=="held_technical"and result["initial_answer"]=="one two three"
    assert len(result["stages"])==1 and result["usage"]["total_tokens"]==14
    native=FakeNative(["one two three","terminal"]);result=app.perform(item,"C0",native)
    assert result["status"]=="algorithm_failure"and result["initial_answer"]=="one two three"
    assert result["failure_stage"].startswith("legacy/sentence_format/")and len(result["stages"])==2
    native=FakeNative(["terminal"]);result=app.perform(item,"C0",native)
    assert result["status"]=="algorithm_failure"and not result["formatter_triggered"]and len(native.calls)==1


def phase_contracts():
    with TemporaryDirectory(prefix="p02_reference_phase_fake_")as temporary:
        root=Path(temporary);context=root/"context";out=root/"out";context.mkdir();out.mkdir()
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf8")
        write(context/"PROTOCOL.frozen.json",{"fake":True});(context/"INPUTS90.jsonl").write_text("{}\n",encoding="utf8")
        planned=[dict(task_id="fake_"+str(index))for index in range(270)]
        app.jsonl(context/"ORDER270.jsonl",planned)
        protocol=app.sha(context/"PROTOCOL.frozen.json")
        expected=dict(protocol_sha256=protocol,inputs_sha256=app.sha(context/"INPUTS90.jsonl"),order_sha256=app.sha(context/"ORDER270.jsonl"))
        with fixture_db(out/"checkpoint.sqlite3")as db:
            db.executescript("CREATE TABLE data(kind TEXT,key TEXT,value TEXT);CREATE TABLE requests(status TEXT);")
            db.execute("INSERT INTO data VALUES('meta','identity',?)",(json.dumps(expected),))
            db.executemany("INSERT INTO data VALUES('outcome',?,'{}')",[(row["task_id"],)for row in planned[:-1]])
        write(out/"held/fake.json",dict(task_id=planned[-1]["task_id"]))
        job=dict(job_id="123",protocol_sha256=protocol,directory="/fake/context90_"+protocol[:16]);write(out/"REMOTE_JOB_ID.json",job)
        ready=dict(job_id="123",protocol_sha256=protocol,status="ready",gpu="Fake A6000",no_LLM_credentials_or_requests=True)
        write(out/"REMOTE_GPU_READY.json",ready)
        finished=dict(status="stopped",GPU="Fake A6000",retrieval_requests_processed=5,no_LLM_requests=True)
        with patch.object(app,"CONTEXT",context),patch.object(app,"CONTEXT_OUT",out):
            assert not app.context90_finished() # Missing FINISHED, despite all attempted.
            write(out/"REMOTE_GPU_FINISHED.json",finished)
            assert app.context90_finished() # 269 quality outcomes + one held is enough.
            for key in expected:
                bad={**expected,key:"wrong"}
                with fixture_db(out/"checkpoint.sqlite3")as db:db.execute("UPDATE data SET value=? WHERE kind='meta'",(json.dumps(bad),))
                assert not app.context90_finished()
            with fixture_db(out/"checkpoint.sqlite3")as db:db.execute("UPDATE data SET value=? WHERE kind='meta'",(json.dumps(expected),))
            for name,value in[("REMOTE_JOB_ID.json",{**job,"job_id":"999"}),
                ("REMOTE_GPU_READY.json",{**ready,"protocol_sha256":"wrong"}),
                ("REMOTE_GPU_FINISHED.json",{**finished,"GPU":"Wrong GPU"})]:
                original=app.read(out/name);write(out/name,value);assert not app.context90_finished();write(out/name,original)
            with fixture_db(out/"checkpoint.sqlite3")as db:db.execute("INSERT INTO requests VALUES('in_flight')")
            assert not app.context90_finished()
            with fixture_db(out/"checkpoint.sqlite3")as db:db.execute("DELETE FROM requests")
            assert app.context90_finished()


def authorization_and_queue_contracts(items):
    # Before client/Cache/key/network imports, default execution must return
    # after initial32 and deferred execution must enforce predecessor guard.
    planned=app.task_order(items);done=planned[:32]
    with ExitStack()as stack:
        stack.enter_context(patch.object(app,"authorized",return_value={}))
        stack.enter_context(patch.object(app,"rows",return_value=planned))
        stack.enter_context(patch.object(app,"outcomes",return_value=done))
        stack.enter_context(patch.object(app,"held_outcomes",return_value=[]))
        stack.enter_context(patch.object(app,"report",return_value={"fake":"read-only report"}))
        stack.enter_context(patch.object(app,"context90_finished",return_value=False))
        assert app.run()=={"fake":"read-only report"}
        assert app.run(deferred=True)=={"fake":"read-only report"}
    with patch.object(app,"authorized",return_value={}),patch.object(app,"rows",return_value=planned),\
        patch.object(app,"outcomes",return_value=planned[:-1]),\
        patch.object(app,"held_outcomes",return_value=[planned[-1]]),\
        patch.object(app,"report",return_value={"held":"read-only report"}):
        assert app.run(deferred=True)=={"held":"read-only report"}
    with patch.object(app,"check",return_value={}),patch.object(app,"read",return_value=dict(approved=True,
        experiment=app.HERE.name,protocol_sha256="fake",decision_id="fake")),patch.object(app,"sha",return_value="fake"),\
        patch.object(app,"rows",return_value=[dict(id="fake")]):
        app.authorized()
        try:app.authorized(deferred=True)
        except AssertionError:pass
        else:raise AssertionError("Initial32 AUTH must not release deferred120")
    with TemporaryDirectory(prefix="p02_reference_wait_fake_")as temporary,\
        patch.object(app,"OUT",Path(temporary)),patch.object(app,"sha",return_value="fake"),\
        patch.object(app,"context90_completion_status",side_effect=[dict(finished=False,reason="pending"),dict(finished=True,reason="complete")]),\
        patch.object(app,"atomic_json",side_effect=lambda path,value:path.write_text(json.dumps(value),encoding="utf8")),\
        patch.object(app.time,"sleep")as sleeper:
        assert app.wait_for_context90()
        sleeper.assert_called_once_with(30)
        assert app.read(Path(temporary)/"DEFERRED_WAIT_STATUS.json")["finished"]is True
    with TemporaryDirectory(prefix="p02_reference_late_held_fake_")as temporary,\
        patch.object(app,"OUT",Path(temporary)),patch.object(app,"sha",return_value="fake"),\
        patch.object(app,"atomic_json",side_effect=lambda path,value:path.write_text(json.dumps(value),encoding="utf8")),\
        patch.object(app,"context90_completion_status",side_effect=[dict(finished=False,reason="pending"),dict(finished=True,reason="complete")]):
        assert app.held_outcomes()==[]
        def later_hold(seconds):
            assert seconds==30
            folder=Path(temporary)/"held";folder.mkdir()
            (folder/"late.json").write_text(json.dumps(dict(task_id=planned[0]["task_id"])),encoding="utf8")
        with patch.object(app.time,"sleep",side_effect=later_hold):assert app.wait_for_context90()
        assert {row["task_id"]for row in app.held_outcomes()}=={planned[0]["task_id"]}


def validate():
    items=app.rows(app.INPUTS)
    mapping_contracts();perform_contracts(items);phase_contracts();authorization_and_queue_contracts(items)
    value=dict(passed=True,tests=["single_window_shared_name_boundary_literal_case_copy",
        "unsupported_request_reader_function_and_punctuation_rejected","four_arms_native128_formatter_raw_current_same_reader",
        "three_vs_four_word_trigger","formatter_technical_preserves_paid_QA","QA_and_formatter_terminal_not_rerolled",
        "exact_predecessor_protocol_inputs_order_job_READY_FINISHED_noinflight",
        "269_quality_plus1held_releases_only_after_finish","default_initial32_only_separate_deferred_AUTH_and_flag",
        "held_is_attempted_no_automatic_resubmission","durable_metadata_only_wait_no_real_sleep_or_provider"],
        inherited_guards="Frozen context90 NativeAsk/SubmissionAwareStrictAsk and TRANSPORT_CONTRACTS; no new retry implementation",
        real_API_SSH_GPU_requests=0,live_checkpoint_ledger_writes=0)
    app.atomic_json(app.HERE/"LOCAL_RUNTIME_CONTRACTS.json",value)
    return value


if __name__=="__main__":print(json.dumps(validate(),ensure_ascii=True))
