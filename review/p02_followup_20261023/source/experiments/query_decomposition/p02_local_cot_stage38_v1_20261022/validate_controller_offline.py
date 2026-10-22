"""Zero-service contracts over temporary caches and fake provider receipts."""
from contextlib import contextmanager, closing
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import tempfile
from unittest.mock import patch

from . import run as r
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
        usage = value.get("usage")
        if usage:
            with self.db: self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?", (r.cost(usage), rid))
    def close(self): self.db.close()


class Client:
    def __init__(self, callback): self.callback = callback; self.calls = []
    def request_once(self, payload):
        self.calls.append(deepcopy(payload))
        return self.callback(payload, len(self.calls))
    def close(self): pass


def success(config, answer, **values):
    return dict(status="success", answer=answer, finish_reason="stop", response_id="fake",
        requested_model=config["generation"]["model"], returned_model=config["generation"]["model"],
        usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110), **values)


@contextmanager
def harness(config, callback, directory):
    with Cache(directory / "checkpoint.sqlite3") as cache, closing(Budget()) as budget:
        client = Client(callback)
        with closing(r.SubmissionAwareStrictAsk(cache, None, config, budget, r.event,
                recover_unknown=True, maximum_attempts=5, stop=lambda: False)) as strict:
            strict.client = client
            with patch.object(transport, "provider_gate", fake_gate), patch.object(transport.time, "sleep", lambda *_: None):
                yield cache, budget, strict, client


def assert_held(native, prompt, cap):
    try: native("legacy/answer/1", prompt, cap)
    except r.TechnicalHeld: pass
    else: raise AssertionError("Expected held receipt")


def check():
    assert not r.PROTOCOL.exists(), "Run preparation checks before freezing"
    r.prepare(); r.validate()
    config = r.read(r.PACKAGE / "config.json"); items = r.rows(r.INPUTS); item = items[0]
    verified = []
    with tempfile.TemporaryDirectory(prefix="p02_local_cot_fake_") as folder:
        temp = Path(folder)
        with patch.object(r, "OUT", temp / "events"):
            r.event(dict(utc="already-provided", event="bounded_technical_recovery"))
            assert r.rows(r.OUT / "events.jsonl")[-1]["utc"] == "already-provided"
            verified.append("event_accepts_existing_utc_without_masking_provider_state")
            # Use real NativeAsk/strict durability, with no HTTP client or real ledger.
            with harness(config, lambda p, n: (dict(status="unknown", error_type="ReadTimeout") if n == 1 else success(config, "The factual answer has five words.")), temp / "recovery") as (cache, budget, strict, client):
                native = r.NativeAsk(cache, strict, config, "fake/recovery", event=r.event)
                native("legacy/answer/1", "fake prompt", 128)
                assert len(client.calls) == 2 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 1
                assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0] == 1
                assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 2
            verified.append("returnless_unknown_durable_retry_retains_receipt_reservation_and_utc_event")
            with harness(config, lambda p, n: dict(status="unknown", error_type="ReadTimeout"), temp / "exhausted") as (cache, budget, strict, client):
                native = r.NativeAsk(cache, strict, config, "fake/exhausted", event=r.event)
                assert_held(native, "five attempts", 128)
                assert len(client.calls) == 5 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 4
                assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL").fetchone()[0] == 5
                assert_held(r.NativeAsk(cache, strict, config, "fake/restart", event=r.event), "five attempts", 128)
                assert len(client.calls) == 5
            verified.append("maximum5_across_resume_no_sixth_attempt_all5_reservations_preserved")
            def returned_invalid(payload, count):
                value = success(config, "Returned text cannot be reissued."); value["returned_model"] = "different-model"
                return value
            with harness(config, returned_invalid, temp / "wrongmodel") as (cache, budget, strict, client):
                assert_held(r.NativeAsk(cache, strict, config, "fake/model"), "wrong model", 128)
                assert len(client.calls) == 1 and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 0
            verified.append("returned_wrong_model_held_without_replay")
            with harness(config, lambda p, n: dict(status="unknown"), temp / "ledgerguard") as (cache, budget, strict, client):
                payload = r.build_payload(config["generation"], "ledger crash", 128)
                rid, _ = cache.begin_request("fake", payload, before_start=lambda i: budget.reserve(i, payload))
                cache.finish_request(rid, dict(status="unknown"))
                budget.db.execute("UPDATE charges SET actual=0.1,status='returned_usage' WHERE id=?", (rid,)); budget.db.commit()
                assert_held(r.NativeAsk(cache, strict, config, "fake/ledger"), "ledger crash", 128)
                assert not client.calls and cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 0
            verified.append("ledger_return_before_cache_guard_prevents_reissue")
            with harness(config, lambda p, n: success(config, "Support: A supplied fact.\nAnswer: Paris") if n == 1 else success(config, "Paris is the requested city."), temp / "formatter") as (cache, budget, strict, client):
                value = r.perform(item, "R256", r.NativeAsk(cache, strict, config, "fake/formatter", event=r.event))
                assert value["status"] == "success" and value["raw_output"].startswith("Support:")
                assert value["support"] == "A supplied fact." and value["parsed_answer"] == "Paris"
                assert value["accepted_answer"] == "Paris is the requested city." and value["formatter_triggered"]
                assert client.calls[0]["max_tokens"] == 256 and client.calls[1]["max_tokens"] == 128
                assert client.calls[1]["messages"][0]["content"] == r.formatter_prompt(item, "Paris")
                assert "A supplied fact." not in client.calls[1]["messages"][0]["content"]
            verified.append("raw_support_answer_accepted_separate_formatter_answer_only128")
            with harness(config, lambda p, n: success(config, "Malformed complete model output."), temp / "parsefailure") as (cache, budget, strict, client):
                value = r.perform(item, "F256", r.NativeAsk(cache, strict, config, "fake/parser"))
                assert value["status"] == "algorithm_failure" and value["parser_status"] == "format_failure"
                assert value["raw_output"] == "Malformed complete model output." and value["accepted_answer"] is None
                assert not value["formatter_triggered"] and len(client.calls) == 1
            verified.append("parser_failure_algorithm_outcome_no_formatter_no_reroll")
            def length(payload, count):
                value = success(config, "Support: A fact.\nAnswer: Paris" if count == 1 else "A truncated factual sentence")
                if count == 2: value["finish_reason"] = "length"
                return value
            with harness(config, length, temp / "formatterfailure") as (cache, budget, strict, client):
                value = r.perform(item, "R256", r.NativeAsk(cache, strict, config, "fake/length"))
                assert value["status"] == "algorithm_failure" and value["reason"] == "terminal_length"
                assert value["parsed_answer"] == "Paris" and value["support"] == "A fact."
                assert value["accepted_answer"] is None and len(client.calls) == 2
            verified.append("formatter_length_is_algorithm_failure_retains_parsed_answer_no_retry")
            # Logging/programming exceptions propagate rather than becoming a provider hold.
            def broken_native(*args): raise TypeError("local logging defect")
            broken_native.stages = []
            try: r.perform(item, "N128", broken_native)
            except TypeError: pass
            else: raise AssertionError("Local TypeError was mislabeled as technical provider hold")
            verified.append("unexpected_local_exception_not_hidden_as_technical_provider_failure")
            # Complete first32 synthetic outcomes, deliberately including a model format failure.
            structured_count = 0
            def engineering_reply(payload, count):
                nonlocal structured_count
                prompt = payload["messages"][0]["content"]
                if "Output the Support and Answer fields:" in prompt:
                    structured_count += 1
                    answer = "Malformed format" if structured_count == 1 else "Support: The supplied fact remains established.\nAnswer: The requested factual sentence is complete."
                else: answer = "The requested factual sentence is complete."
                return success(config, answer)
            fake_protocol = temp / "fake-protocol.json"; fake_protocol.write_text('{"fake":true}', encoding="utf8")
            with patch.object(r, "PROTOCOL", fake_protocol), harness(config, engineering_reply, temp / "engineering") as (cache, budget, strict, client):
                cache.put("meta", "identity", r.identity())
                for task in r.rows(r.ORDER)[:32]:
                    record = next(x for x in items if x["probe_id"] == task["probe_id"])
                    value = dict(**task, **r.perform(record, task["arm"], r.NativeAsk(cache, strict, config, task["task_id"])))
                    cache.put("outcome", task["task_id"], value)
                gate = r.engineering_gate(cache, {i["probe_id"]: i for i in items}, config)
                assert gate["passed"] is True and gate["planned_conditions"] == 32
                # Force a genuine parser failure independently of arm rotation for this assertion.
                assert any(v["status"] == "algorithm_failure" for v in cache.all("outcome").values()), "Ensure format failure exercised"
            verified.append("first32_engineering_gate_accepts_recorded_algorithm_format_failure")
    # Read-only live prerequisite state and failure-closed metadata variants.
    original_metadata = r._metadata
    actual = r.predecessor_status(); assert actual["finished"], actual
    for target in (r.STAGE.name, r.OPERATOR.name):
        def inflight(folder, kind, key, *, target=target):
            identity, done, active = original_metadata(folder, kind, key)
            return identity, done, 1 if folder.name == target else active
        with patch.object(r, "_metadata", inflight): assert not r.predecessor_status()["finished"]
    verified.append("readonly_predecessor152_346_and_verified_factor_pause_no_inflight_fail_closed")
    original_read = r.read
    def noauth(path):
        value = original_read(path)
        if Path(path).name == "FACTOR_TECHNICAL_PAUSE_AUTHORIZATION.json": value = {**value, "approved": False}
        return value
    with patch.object(r, "read", noauth): assert not r.predecessor_status()["finished"]
    verified.append("factor_technical_pause_requires_dedicated_root_authorization")
    assert not (r.OUT / "checkpoint.sqlite3").exists()
    result = dict(passed=True, checks=verified, controller_sha256=r.sha(r.HERE / "run.py"),
        templates_sha256=r.sha(r.HERE / "templates.py"), input_sha256=r.sha(r.INPUTS),
        new_API_SSH_GPU_requests=0, production_checkpoint_or_ledger_writes=0)
    r.atomic_json(r.HERE / "CONTROLLER_OFFLINE_CHECKS.json", result)
    return result


if __name__ == "__main__": print(json.dumps(check(), ensure_ascii=True))
