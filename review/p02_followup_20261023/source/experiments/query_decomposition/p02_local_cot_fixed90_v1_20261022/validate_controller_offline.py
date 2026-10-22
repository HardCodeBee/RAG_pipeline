"""Real cache/NativeAsk, synthetic provider and retrieval; no live service or ledger writes."""
from contextlib import contextmanager, closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
import tempfile
from unittest.mock import patch
from . import run as r
from . import remote_bridge, preparation as p
from .offline_checks import packet, SENTINEL
from p02_multigrain_timing_cachefix_v1.core.cache import Cache
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022 import message_transport as transport


@contextmanager
def fake_gate(*args, **kwargs):
    if kwargs.get("stop") and kwargs["stop"](): raise InterruptedError("Stopped")
    yield


class Budget:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE charges(id TEXT PRIMARY KEY, actual REAL, reserved REAL, status TEXT)")
    def reserve(self, rid, payload):
        with self.db: self.db.execute("INSERT INTO charges VALUES(?,NULL,1.0,'reserved')", (rid,))
    def finish(self, rid, value):
        if value.get("usage"):
            with self.db: self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?", (r.cost(value["usage"]), rid))
    def close(self): self.db.close()


class Client:
    def __init__(self, callback): self.callback = callback; self.calls = []; self.stage = None
    def request_once(self, payload):
        self.calls.append(deepcopy(payload)); return self.callback(payload, len(self.calls))
    def close(self): pass


def success(config, answer, finish="stop"):
    return dict(status="success", answer=answer, finish_reason=finish, response_id="fake",
        requested_model=config["generation"]["model"], returned_model=config["generation"]["model"],
        usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110))


@contextmanager
def harness(config, callback, directory):
    with Cache(directory / "checkpoint.sqlite3") as cache, closing(Budget()) as budget:
        client = Client(callback)
        with closing(r.SubmissionAwareStrictAsk(cache, None, config, budget, r.event,
                recover_unknown=True, maximum_attempts=5, stop=lambda: False)) as strict:
            strict.client = client
            with patch.object(transport, "provider_gate", fake_gate), patch.object(transport.time, "sleep", lambda *_: None):
                yield cache, budget, strict, client


def assert_held(native, prompt="fake prompt"):
    try: native("legacy/answer/1", prompt, 128)
    except r.TechnicalHeld: return
    raise AssertionError("Expected technical hold")


def check():
    assert not r.PROTOCOL.exists(), "Preparation checks run before root freezes"
    r.prepare(); r.validate(); verified = []
    config = r.read(r.PACKAGE / "config.json")
    tasks = r.rows(r.ORDER); inputs = {v["id"]: v for v in r.rows(r.INPUTS)}
    with tempfile.TemporaryDirectory(prefix="p02_cot90_fake_") as directory:
        temp = Path(directory)
        with patch.object(r, "OUT", temp / "events"):
            r.event(dict(event="unknown_retirement", utc="retained-time"))
            assert r.rows(r.OUT / "events.jsonl")[-1]["utc"] == "retained-time"
            with harness(config, lambda p, n: dict(status="unknown", error_type="ReadTimeout") if n == 1 else success(config, "A complete factual sentence is available."), temp / "retry") as (cache, budget, strict, client):
                r.NativeAsk(cache, strict, config, "fake/recovery", event=r.event)("legacy/answer/1", "recover unknown", 128)
                assert len(client.calls) == 2 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 1
                assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0] == 1
            verified.append("utc_safe_unknown_recovery_keeps_old_unknown_reservation")
            with harness(config, lambda p, n: dict(status="unknown"), temp / "exhaust") as (cache, budget, strict, client):
                assert_held(r.NativeAsk(cache, strict, config, "fake/exhaust"))
                assert_held(r.NativeAsk(cache, strict, config, "fake/resume"))
                assert len(client.calls) == 5 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 4
                assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0] == 5
            verified.append("durable5_attempts_no_sixth_on_resume_all_reservations_retained")
            def wrongmodel(payload, count):
                value = success(config, "Returned content prevents safe replay."); value["returned_model"] = "different-model"; return value
            with harness(config, wrongmodel, temp / "wrongmodel") as (cache, budget, strict, client):
                assert_held(r.NativeAsk(cache, strict, config, "fake/wrongmodel"))
                assert len(client.calls) == 1 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 0
            verified.append("returned_wrong_model_never_reissued")
            with harness(config, lambda p, n: dict(status="unknown"), temp / "ledger") as (cache, budget, strict, client):
                payload = r.build_payload(config["generation"], "ledger crash", 128)
                rid, _ = cache.begin_request("fake", payload, before_start=lambda i: budget.reserve(i, payload))
                cache.finish_request(rid, dict(status="unknown"))
                budget.db.execute("UPDATE charges SET actual=0.1,status='returned_usage' WHERE id=?", (rid,)); budget.db.commit()
                assert_held(r.NativeAsk(cache, strict, config, "fake/ledger"), "ledger crash")
                assert not client.calls
            verified.append("ledger_actual_before_cache_guard_blocks_returnless_retry")

            fake_protocol = temp / "PROTOCOL.json"; fake_protocol.write_text('{"fake":true}', encoding="utf8")
            failures = {"format": False, "formatter": False, "final": False}
            with patch.object(r, "PROTOCOL", fake_protocol), harness(config, lambda p, n: None, temp / "engineering") as (cache, budget, strict, client):
                cache.put("meta", "identity", r.expected_identity())
                for task in tasks[:24]:
                    case = inputs[task["id"]]; arm = task["arm"]; ordinary = len(case["supporting_questions"])
                    native = r.NativeAsk(cache, strict, config, task["task_id"], event=r.event)
                    city = {"N128": "Delta", "N256": "Gamma", "F256": "Echo", "R256": "Foxtrot"}[arm]
                    def callback(payload, count):
                        stage = client.stage
                        if stage.startswith("legacy/answer/"):
                            sid = int(stage.rsplit("/", 1)[1])
                            if sid <= ordinary and arm in ("F256", "R256"):
                                if arm == "F256" and not failures["format"]:
                                    failures["format"] = True; return success(config, "Malformed structured output.")
                                answer = "Support: " + SENTINEL + " is for audit.\nAnswer: " + city
                            else: answer = city if sid <= ordinary else "The final requested year is 2001."
                        elif stage.startswith("legacy/sentence_format/"):
                            assert SENTINEL not in payload["messages"][0]["content"]
                            if arm == "R256" and not failures["formatter"]:
                                failures["formatter"] = True; return success(config, "This incomplete formatter output", "length")
                            answer = "The requested target city is " + city + "."
                        elif stage.startswith("legacy/rewrite/"):
                            assert SENTINEL not in payload["messages"][0]["content"]
                            answer = "When did " + city + " open?"
                        else:
                            assert stage == "legacy/final" and SENTINEL not in payload["messages"][0]["content"]
                            if arm == "N128" and not failures["final"]:
                                failures["final"] = True; return success(config, "")
                            answer = "2001"
                        return success(config, answer)
                    client.callback = callback
                    def invoke(stage, prompt, maximum, allow_final_truncation=False):
                        client.stage = stage
                        return native(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
                    invoke.stages = native.stages
                    value = r.perform(task, case, packet, invoke, cache, r.Pipeline(config), [], [])
                    cache.put("outcome", task["task_id"], value)
                assert all(failures.values()), failures
                gate = r.engineering_gate(cache, tasks, config)
                assert gate["passed"] and gate["format_failures"] >= 1 and gate["algorithm_failures"] >= 3, gate
                assert any(v["status"] == "algorithm_failure" and v.get("terminal_stage") == "legacy/final" for v in cache.all("outcome").values())
                # A wire mutation must stop structural approval, regardless of quality.
                victim = next(v for v in cache.all("outcome").values() if v["candidate_outputs"])
                wrong = deepcopy(victim); wrong["candidate_outputs"][0]["parsed_answer"] = "CORRUPT"
                cache.db.execute("UPDATE data SET value=? WHERE kind='outcome' AND key=?", (json.dumps(wrong), victim["task_id"])); cache.db.commit()
                assert not r.engineering_gate(cache, tasks, config)["passed"]
            verified.append("first24_real_native_wire_gate_allarms_raw_parse_formatter_and_final_algorithm_failures_retained")
            verified.append("first24_gate_rejects_mutated_parsed_answer_without_quality_selection")

            class BrokenNative:
                stages = []
                def __call__(self, *args, **kwargs): raise TypeError("Local programming defect")
            with Cache(temp / "programming" / "checkpoint.sqlite3") as cache:
                try: r.perform(tasks[0], inputs[tasks[0]["id"]], packet, BrokenNative(), cache, r.Pipeline(config), [], [])
                except TypeError: pass
                else: raise AssertionError("Local defect became held or algorithm outcome")
            verified.append("unexpected_programming_exception_propagates_not_provider_hold")

            case = inputs[tasks[0]["id"]]; resources = r.read(r.HERE / "RESOURCE_CONTRACTS.json")
            class FakeRemote:
                count = 0
                def retrieve(self, qid, query, k):
                    self.count += 1; result = packet(query); result["qid"] = qid
                    for doc in result["docs"]: doc["score"] = 1.0
                    return result
            remote = FakeRemote()
            with Cache(temp / "retrieval" / "checkpoint.sqlite3") as cache:
                result, key, request, hit = r.retrieve_cached(cache, remote, resources, case, case["question"])
                assert not hit and r.retrieve_cached(cache, remote, resources, case, case["question"])[3] and remote.count == 1
                bad = deepcopy(result); bad["qid"] = "different-query"
                cache.db.execute("UPDATE data SET value=? WHERE kind='retrieval' AND key=?", (json.dumps(bad), key)); cache.db.commit()
                try: r.retrieve_cached(cache, remote, resources, case, case["question"])
                except ValueError: pass
                else: raise AssertionError("Corrupt cache was reused")
                assert remote.count == 1
            verified.append("retrieval_cache_hit_identity_and_complete_packet_checked_corruption_not_refetched")

        bridge = remote_bridge.bridge_class()(r.read(r.HERE / "RESOURCE_CONTRACTS.json"), temp / "bridge", dict(protocol_sha256="a" * 64))
        bridge.started = True; bridge.job_id = "12345"; bridge.exists = lambda *_: False; bridge.capture_receipts = lambda: None
        (bridge.output / "ALLOCATION_CLOCK.json").write_text(json.dumps(dict(submission_started_utc=datetime.now(timezone.utc).isoformat())), encoding="utf8")
        bridge.command = lambda args, timeout=45: "12345|RUNNING|p02-cot90\n"
        assert bridge.can_start_condition()["allowed"] and "/cot90_" in bridge.directory
        for reply in ("", "12345|COMPLETED|p02-cot90\n", "12345|RUNNING|p02-factor90\n", "12346|RUNNING|p02-cot90\n", "malformed"):
            bridge.command = lambda args, timeout=45, reply=reply: reply
            assert not bridge.can_start_condition()["allowed"]
        def unreachable(*args, **kwargs): raise OSError("unavailable")
        bridge.command = unreachable; assert not bridge.can_start_condition()["allowed"]
        bridge.command = lambda args, timeout=45: "12345|RUNNING|p02-cot90\n"
        (bridge.output / "ALLOCATION_CLOCK.json").write_text(json.dumps(dict(submission_started_utc=(datetime.now(timezone.utc) - timedelta(hours=6)).isoformat())), encoding="utf8")
        assert not bridge.can_start_condition()["allowed"]
        bridge.started = False; assert not bridge.can_start_condition()["allowed"]
        bridge.log("synthetic", utc="already-provided")
        assert r.rows(bridge.output / "REMOTE_EVENTS.jsonl")[-1]["utc"] == "already-provided"
        verified.append("actual_bridge_string_command_interface_owned_running_scheduler_margin_and_stopped_fail_closed")
        # Exercise the production startup waiter with normal pending->ready,
        # not a substitute implementation of scheduler startup.
        ticks = [0]
        ready = dict(status="ready", job_id=bridge.job_id,
            protocol_sha256=bridge.authorization["protocol_sha256"], gpu="NVIDIA RTX A6000",
            no_LLM_credentials_or_requests=True,
            resources={key: dict(identity=value["identity"]) for key, value in bridge.resources.items()})
        bridge.remote_json = lambda name: deepcopy(ready)
        bridge.exists = lambda path: path.endswith("/READY.json") and ticks[0] >= 2
        bridge.command = lambda args, timeout=45: "12345|" + ("PENDING" if not ticks[0] else "RUNNING") + "|p02-cot90\n"
        with patch.object(remote_bridge.time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + 1)):
            bridge.await_ready()
        assert ticks[0] == 2
        ready["job_id"] = "other-job"
        try: bridge.await_ready()
        except AssertionError: pass
        else: raise AssertionError("Foreign READY accepted")
        ready["job_id"] = bridge.job_id
        bridge.exists = lambda path: path.endswith("/WORKER_ERROR.json")
        try: bridge.await_ready()
        except RuntimeError: pass
        else: raise AssertionError("Worker error accepted")
        bridge.exists = lambda path: False
        with patch.object(remote_bridge.time, "monotonic", side_effect=[0, 601]):
            try: bridge.await_ready()
            except RuntimeError: pass
            else: raise AssertionError("Startup timeout ignored")
        (bridge.output / "STOP_REQUESTED").write_text("local stop", encoding="utf8")
        try: bridge.await_ready()
        except RuntimeError: pass
        else: raise AssertionError("Local stop ignored during startup")
        verified.append("production_startup_waits_pending_to_owned_ready_rejects_foreign_error_timeout_and_stop_without_resubmission")
    actual = r.predecessors.status()
    assert actual.get("ordinary_stage", {}).get("finished") is True, actual
    # This check never waits on or touches the running reference controller.
    with patch.object(r.predecessors, "metadata", lambda folder, stage=False: dict(finished=stage)):
        assert not r.predecessors.status()["finished"]
    with patch.object(r.predecessors, "metadata", lambda folder, stage=False: dict(finished=True)), patch.object(r.predecessors, "reference_workers_closed", lambda: dict(closed=False)):
        assert not r.predecessors.status()["finished"]
    with patch.object(r.predecessors, "metadata", side_effect=AssertionError("foreign identity")):
        assert not r.predecessors.status()["finished"]
    verified.append("direct_predecessor_partial_or_open_worker_or_foreign_identity_never_releases_services")
    assert not (r.OUT / "checkpoint.sqlite3").exists()
    result = dict(passed=True, checks=verified, controller_sha256=r.sha(r.HERE / "run.py"),
        flow_sha256=r.sha(r.HERE / "flow.py"), remote_bridge_sha256=r.sha(r.HERE / "remote_bridge.py"),
        retrieval_transport_sha256=r.sha(r.HERE / "retrieval_transport.py"),
        template_sha256=r.sha(p.STAGE / "templates.py"), inputs_sha256=r.sha(r.INPUTS),
        new_API_SSH_GPU_requests=0, production_checkpoint_or_ledger_writes=0)
    r.atomic_json(r.HERE / "CONTROLLER_OFFLINE_CHECKS.json", result); return result


if __name__ == "__main__": print(json.dumps(check(), ensure_ascii=True))
