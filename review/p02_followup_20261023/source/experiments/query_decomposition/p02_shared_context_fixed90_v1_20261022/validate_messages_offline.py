"""Message transport contracts using only fake providers and temporary ledgers.

No real provider client, credential, remote connection, corpus or GPU is used.
This module owns no flow transcript: record_node/commit behavior is tested by
the independent flow contracts. The original core/shared runtime stays frozen.
"""
import ast
from contextlib import closing, contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload as original_payload
from . import message_transport as transport
from .native_ask import (
    AlgorithmTerminal, NativeAsk, SubmissionAwareStrictAsk, TechnicalHeld,
    integrity, stage_success,
)

SOURCE_NATIVE = HERE.parent / "p02_m4_wiki20_dynamic_dev_v1_20261022/native_ask.py"
SHARED = HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py"
CONFIG = dict(generation=dict(model="synthetic-model", base_url="https://invalid.invalid/v1",
    api_key_env="UNUSED_FAKE_KEY", timeout_seconds=1, temperature=0.0,
    enable_thinking=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def good(answer="The synthetic factual answer is complete.", finish="stop"):
    return dict(status="success", requested_model="synthetic-model",
        returned_model="synthetic-model", answer=answer, finish_reason=finish,
        usage=dict(prompt_tokens=7, completion_tokens=5, total_tokens=12))


class FakeBudget:
    """Persistent request IDs and unresolved liabilities; no global ledger."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,actual REAL,status TEXT,reserved REAL)")

    def reserve(self, rid, request):
        reservation = sum(len(m["content"].encode("utf8")) for m in request["messages"]) * .8e-6
        reservation += request["max_tokens"] * 2e-6
        with self.db:
            self.db.execute("INSERT INTO charges VALUES(?,NULL,'reserved',?)", (rid, reservation))

    def finish(self, rid, receipt):
        usage = receipt.get("usage")
        if usage and all(type(usage.get(k)) is int and usage[k] >= 0
                         for k in ("prompt_tokens", "completion_tokens")):
            actual = usage["prompt_tokens"] * .8e-6 + usage["completion_tokens"] * 2e-6
            with self.db:
                self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?", (actual, rid))

    def close(self):
        self.db.close()


class FakeClient:
    def __init__(self, replies=(), *, repeat=None):
        self.replies, self.repeat = list(replies), repeat
        self.payloads, self.closed = [], False

    def request_once(self, payload):
        self.payloads.append(deepcopy(payload))
        if self.replies:
            return deepcopy(self.replies.pop(0))
        if self.repeat is not None:
            return deepcopy(self.repeat)
        raise AssertionError("Unexpected fake submission")

    def close(self):
        self.closed = True


@contextmanager
def fixture():
    with TemporaryDirectory(prefix="messages-fake-", dir=HERE) as directory:
        folder = Path(directory)
        with Cache(folder / "cache.sqlite3") as cache, closing(FakeBudget(folder / "ledger.sqlite3")) as budget:
            yield cache, budget


@contextmanager
def no_real_submission():
    # Only local class globals are patched; payload factories are never patched.
    with patch.object(transport, "provider_gate", lambda *a, **k: nullcontext()), \
         patch.object(transport.time, "sleep", lambda seconds: None):
        yield


def strict_for(cache, budget, client, events=None):
    strict = SubmissionAwareStrictAsk(cache, None, CONFIG, budget,
        (events if events is not None else []).append, maximum_attempts=5)
    strict.client = client
    return strict


def rejected(callback, expected):
    try:
        callback()
    except expected as error:
        return error
    raise AssertionError("Expected " + expected.__name__)


def builder_contracts():
    settings = CONFIG["generation"]
    prompt = "  Frozen single-user text.\nDo not normalize.  "
    original = original_payload(settings, prompt, 128)
    assert transport.build_payload(settings, prompt, 128) == original
    assert transport.build_payload(settings, [dict(role="user", content=prompt)], 128) == original
    messages = [dict(role="user", content="First question"),
        dict(role="assistant", content=" Accepted formatted answer. "),
        dict(role="user", content="Current question")]
    payload = transport.build_payload(settings, messages, 128)
    assert payload["messages"] == messages
    assert {k: v for k, v in payload.items() if k != "messages"} == {k: v for k, v in original.items() if k != "messages"}
    messages[1]["content"] = "Mutated later"
    messages.append(dict(role="assistant", content="Must not leak"))
    assert len(payload["messages"]) == 3 and payload["messages"][1]["content"] == " Accepted formatted answer. "
    snapshot = deepcopy(payload)
    payload["messages"][0]["content"] = "Mutating output also detaches input"
    assert messages[0]["content"] == "First question" and snapshot["messages"][0]["content"] == "First question"
    # Role is part of the complete wire identity even with identical text.
    left = transport.build_payload(settings, [dict(role="user", content="same")], 128)
    right = deepcopy(left); right["messages"][0]["role"] = "assistant"
    assert canonical(left) != canonical(right)
    invalid = [[], [dict(role="assistant", content="x")],
        [dict(role="user", content="x"), dict(role="assistant", content="x")],
        [dict(role="user", content="x"), dict(role="user", content="y"), dict(role="user", content="z")],
        [dict(role="system", content="x")], [dict(role="user", content="")],
        [dict(role="user", content="  \n")], [dict(role="user", content=[])],
        [dict(role="user", content="x", name="extra")], ["not a message"], None, {"role": "user", "content": "x"}]
    for value in invalid:
        rejected(lambda: transport.build_payload(settings, value, 128), (ValueError if isinstance(value, list) else TypeError))
    return dict(native_string_and_single_user_wire_identical=True, real_ordered_role_messages=True,
        only_messages_field_changes=True, detached_deepcopy_preserves_verbatim_text=True,
        invalid_roles_order_fields_empty_or_nontext_rejected=True)


def envelope_and_preflight_contracts():
    settings = CONFIG["generation"]
    content = "x" * (transport.CONTEXT_UNIT_LIMIT - transport.MESSAGE_FRAMING_UNITS - 128)
    assert transport.build_payload(settings, content, 128)["messages"][0]["content"] == content
    error = rejected(lambda: transport.build_payload(settings, content + "x", 128), transport.ContextWindowHeld)
    assert error.estimated_units == 128001 and error.maximum_units == 128000
    prefix = [dict(role="user", content="A"), dict(role="assistant", content="B")]
    remaining = 128000 - (3 * 64) - 128 - 2
    multi = prefix + [dict(role="user", content="z" * remaining)]
    transport.build_payload(settings, multi, 128)
    multi[-1]["content"] += "z"
    rejected(lambda: transport.build_payload(settings, multi, 128), transport.ContextWindowHeld)
    unicode_content = "\u6c49" * ((128000 - 64 - 128) // 3)
    transport.build_payload(settings, unicode_content, 128)
    rejected(lambda: transport.build_payload(settings, unicode_content + "\u6c49", 128), transport.ContextWindowHeld)
    with fixture() as (cache, budget), no_real_submission():
        client = FakeClient(); strict = strict_for(cache, budget, client)
        try:
            rejected(lambda: strict("bad", [], 128), ValueError)
            rejected(lambda: strict("too-long", content + "x", 128), transport.ContextWindowHeld)
            rejected(lambda: NativeAsk(cache, strict, CONFIG, "query/chat")(
                "legacy/answer/1", content + "x", 128), transport.ContextWindowHeld)
            assert not client.payloads
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0
            assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 0
        finally:
            strict.close()
    return dict(str_and_list_exact_limit_pass_one_byte_over_held=True, unicode_UTF8_bytes_used=True,
        framing64_per_message_and_output_allowance_counted=True, no_truncation=True,
        invalid_or_oversized_input_before_client_reservation_request=True)


def submission_cache_and_isolation_contracts():
    messages = [dict(role="user", content="Initial A"), dict(role="assistant", content="Accepted A"),
        dict(role="user", content="Same current?")]
    changed = deepcopy(messages); changed[1]["content"] = "Accepted B"
    reordered = deepcopy(messages); reordered[0]["content"], reordered[-1]["content"] = reordered[-1]["content"], reordered[0]["content"]
    with fixture() as (cache, budget), no_real_submission():
        client = FakeClient([good(), good(), good(), good()]); strict = strict_for(cache, budget, client)
        try:
            a = NativeAsk(cache, strict, CONFIG, "queryA/chat")
            rid, _, _ = a("legacy/answer/2", messages, 128)
            assert client.payloads[0]["messages"] == messages
            assert [m["role"] for m in client.payloads[0]["messages"]] == ["user", "assistant", "user"]
            assert type(client.payloads[0]["messages"][0]["content"]) is str
            reused, _, _ = NativeAsk(cache, strict, CONFIG, "same-wire/flat")("legacy/answer/2", deepcopy(messages), 128)
            assert reused == rid and len(client.payloads) == 1
            b, _, _ = NativeAsk(cache, strict, CONFIG, "queryB/chat")("legacy/answer/2", changed, 128)
            order_id, _, _ = NativeAsk(cache, strict, CONFIG, "queryC/chat")("legacy/answer/2", reordered, 128)
            assert len({rid, b, order_id}) == 3 and len(client.payloads) == 3
            assert client.payloads[1]["messages"][1]["content"] == "Accepted B"
            assert client.payloads[0]["messages"][1]["content"] == "Accepted A"
            # A completed logical stage cannot be rebound to a changed transcript.
            rejected(lambda: a("legacy/answer/2", changed, 128), TechnicalHeld)
            assert len(client.payloads) == 3
            single, _, _ = NativeAsk(cache, strict, CONFIG, "first/reset")("legacy/answer/1", "First native QA", 128)
            chat_first, _, _ = NativeAsk(cache, strict, CONFIG, "first/chat")("legacy/answer/1", [dict(role="user", content="First native QA")], 128)
            assert single == chat_first and len(client.payloads) == 4
            assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 4
            assert cache.db.execute("SELECT count(*) FROM request_stages WHERE request_id=?", (rid,)).fetchone()[0] == 2
            assert cache.db.execute("SELECT stage FROM requests WHERE id=?", (rid,)).fetchone()[0] == "full8/queryA/chat/legacy/answer/2"
            assert a.stages[0]["payload_sha256"] == hashlib.sha256(canonical(transport.build_payload(CONFIG["generation"], messages, 128)).encode()).hexdigest()
        finally:
            strict.close()
    return dict(fake_provider_received_actual_chat_roles=True, exact_whole_payload_shared_once=True,
        changed_history_and_order_get_distinct_requests=True, role_part_of_wire_identity=True,
        native_stage_binding_immutable=True, task_and_arm_inputs_do_not_leak=True,
        first_single_user_compatible_with_reset=True, one_charge_first_owner_multiple_logical_uses=True)


def receipt_and_terminal_contracts():
    invalid = [{"usage": None}, {"usage": {"prompt_tokens": 7, "completion_tokens": 5}},
        {"usage": {"prompt_tokens": -1, "completion_tokens": 5, "total_tokens": 4}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 5, "total_tokens": 6}},
        {"usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 13}},
        {"requested_model": "other"}, {"returned_model": "other"}, {"status": "model_identity_mismatch"}]
    for stage in ("legacy/rewrite/2", "legacy/answer/2", "legacy/sentence_format/2", "legacy/final"):
        assert stage_success(good(), "synthetic-model", stage)
        assert stage_success(good(finish="length"), "synthetic-model", stage) == (stage == "legacy/final")
        for change in invalid:
            receipt = {**good(finish="length"), **change}
            assert not integrity(receipt, "synthetic-model")
            with fixture() as (cache, budget), no_real_submission():
                client = FakeClient(); strict = strict_for(cache, budget, client)
                cap = 64 if stage == "legacy/final" else 128
                prompt = "Existing invalid receipt"
                payload = transport.build_payload(CONFIG["generation"], prompt, cap)
                rid, _ = cache.begin_request("first-owner", payload)
                cache.finish_request(rid, receipt)
                try:
                    rejected(lambda: NativeAsk(cache, strict, CONFIG, "case/chat")(
                        stage, prompt, cap, allow_final_truncation=stage == "legacy/final"), TechnicalHeld)
                    assert not client.payloads and not cache.all("stage")
                finally:
                    strict.close()
    for receipt, reason in [(good(finish="length"), "terminal_length"),
            (good(answer=""), "terminal_empty"), ({**good(), "refusal": "declined"}, "terminal_refusal"),
            (good(finish="content_filter"), "terminal_content_filter")]:
        with fixture() as (cache, budget), no_real_submission():
            client = FakeClient([receipt]); strict = strict_for(cache, budget, client)
            try:
                for _ in range(2):
                    error = rejected(lambda: NativeAsk(cache, strict, CONFIG, "terminal/chat")(
                        "legacy/answer/2", "Terminal QA", 128), AlgorithmTerminal)
                    assert error.value["terminal_reason"] == reason
                assert len(client.payloads) == 1
            finally:
                strict.close()
    # Legitimate native final length is accepted once, including persistent restart.
    with TemporaryDirectory(prefix="messages-length-", dir=HERE) as directory, no_real_submission():
        folder = Path(directory); client = FakeClient([good(finish="length")])
        for _ in range(2):
            with Cache(folder / "cache.sqlite3") as cache, closing(FakeBudget(folder / "ledger.sqlite3")) as budget:
                strict = strict_for(cache, budget, client)
                try:
                    assert NativeAsk(cache, strict, CONFIG, "final/reset")(
                        "legacy/final", "Native final", 64, allow_final_truncation=True)[2] == "length"
                finally:
                    strict.close()
        assert len(client.payloads) == 1
    return dict(all_stages_require_model_three_nonnegative_int_usage_exact_sum=True,
        invalid_receipts_technical_held_not_zero_or_retried=True,
        length_empty_refusal_filter_algorithm_terminals_never_rerolled=True,
        final64_length_stage_specific_one_submission_persistent_reuse=True)


def unknown_and_attempt_contracts():
    signs = [dict(response_id="r"), dict(raw_response={}), dict(usage={}),
        dict(returned_model="synthetic-model"), dict(answer="partial"), dict(content="partial"),
        dict(refusal="declined"), dict(finish_reason="length")]
    for extra in signs + [{"ledger_only": True}]:
        with fixture() as (cache, budget), no_real_submission():
            client = FakeClient(); strict = strict_for(cache, budget, client)
            payload = transport.build_payload(CONFIG["generation"], "Unknown QA", 128)
            rid, _ = cache.begin_request("first-owner", payload, before_start=lambda r: budget.reserve(r, payload))
            receipt = dict(status="unknown", **{k: v for k, v in extra.items() if k != "ledger_only"})
            cache.finish_request(rid, receipt)
            if extra.get("ledger_only"):
                with budget.db:
                    budget.db.execute("UPDATE charges SET actual=.01,status='returned_usage' WHERE id=?", (rid,))
            try:
                rejected(lambda: NativeAsk(cache, strict, CONFIG, "held/chat")(
                    "legacy/answer/2", "Unknown QA", 128), TechnicalHeld)
                assert not client.payloads
                assert cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 0
                assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 1
                assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 1
            finally:
                strict.close()
    with TemporaryDirectory(prefix="messages-five-", dir=HERE) as directory, no_real_submission():
        folder = Path(directory); client = FakeClient(repeat=dict(status="unknown"))
        for _ in range(7):
            with Cache(folder / "cache.sqlite3") as cache, closing(FakeBudget(folder / "ledger.sqlite3")) as budget:
                strict = strict_for(cache, budget, client)
                try:
                    rejected(lambda: NativeAsk(cache, strict, CONFIG, "five/chat")(
                        "legacy/answer/2", "Bounded no-return unknown", 128), TechnicalHeld)
                    assert len(client.payloads) == 5
                    assert cache.db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 4
                    assert budget.db.execute("SELECT count(*) FROM charges WHERE actual IS NULL AND reserved>0").fetchone()[0] == 5
                finally:
                    strict.close()
    return dict(all_eight_return_signs_held_without_blind_retry=True,
        returned_ledger_before_cache_crash_window_held=True,
        no_return_unknown_at_most_five_actual_calls_across_seven_restarts=True,
        retired_receipts_and_all_unknown_liabilities_preserved=True)


def formatter_recovery_contract():
    with TemporaryDirectory(prefix="messages-formatter-", dir=HERE) as directory, no_real_submission():
        folder = Path(directory); client = FakeClient([good(answer="X"), dict(status="failed", http_status=400)])
        for _ in range(2):
            with Cache(folder / "cache.sqlite3") as cache, closing(FakeBudget(folder / "ledger.sqlite3")) as budget:
                strict = strict_for(cache, budget, client)
                try:
                    ask = NativeAsk(cache, strict, CONFIG, "format/chat")
                    assert ask("legacy/answer/1", [dict(role="user", content="QA for formatter")], 128)[1] == "X"
                    rejected(lambda: ask("legacy/sentence_format/1", "Native formatter remains string", 128), TechnicalHeld)
                    assert len(client.payloads) == 2
                    assert len(ask.stages) == 1 and ask.stages[0]["stage"] == "legacy/answer/1"
                    assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 2
                finally:
                    strict.close()
    return dict(successful_QA_not_resubmitted_after_formatter_held=True,
        formatter_string_interface_preserved=True, first_QA_charge_and_failed_formatter_reservation_retained=True)


def source_contracts():
    native_expected = SOURCE_NATIVE.read_text(encoding="utf-8").replace(
        "from p02_multigrain_timing_cachefix_v1.core.llm import build_payload", "from .message_transport import build_payload").replace(
        "from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import StrictAsk", "from .message_transport import StrictAsk")
    assert (HERE / "native_ask.py").read_text(encoding="utf-8") == native_expected
    def call_ast(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StrictAsk")
        return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"), include_attributes=False)
    assert call_ast(HERE / "message_transport.py") == call_ast(SHARED)
    assert transport.single_user_payload is original_payload
    return dict(native_adapter_only_two_imports_changed=True,
        StrictAsk_call_AST_exact_frozen_method=True, builder_binding_local_only=True,
        no_global_build_payload_monkeypatch=True)


def main():
    protected = [ROOT / "src/p02_multigrain_timing_cachefix_v1/core/llm.py",
        ROOT / "src/p02_multigrain_timing_cachefix_v1/core/cache.py", SHARED, SOURCE_NATIVE]
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in protected}
    contracts = dict(source=source_contracts(), builder=builder_contracts(),
        context_envelope=envelope_and_preflight_contracts(),
        actual_messages_cache_and_isolation=submission_cache_and_isolation_contracts(),
        receipt_and_terminal=receipt_and_terminal_contracts(),
        unknown_and_attempts=unknown_and_attempt_contracts(), formatter_recovery=formatter_recovery_contract())
    assert before == {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in protected}
    result = dict(passed=True, scope="New message transport only; fake client/gate/temporary Cache and ledger; no flow/prompt-effect result",
        contracts=contracts, protected_source_sha256=before,
        local_source_sha256={n: sha(HERE / n) for n in ("message_transport.py", "native_ask.py", "validate_messages_offline.py")},
        transcript_commit_owned_by_flow_not_transport=True,
        context_guard="Conservative sum of UTF8 content bytes plus64 per message plus max_tokens <=128000; technical held without truncation; not measured token usage or a money cap",
        no_global_monkeypatch=True, real_provider_requests=0, real_SSH_connections=0, real_GPU_jobs=0,
        credentials_read=False, new_experiment_checkpoint_created=False)
    target = HERE / "TRANSPORT_CONTRACTS.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, classes=len(contracts), output=str(target.relative_to(ROOT)),
        output_sha256=sha(target), sources=result["local_source_sha256"],
        real_provider_requests=0, real_SSH_connections=0, real_GPU_jobs=0), ensure_ascii=True))


if __name__ == "__main__":
    main()
