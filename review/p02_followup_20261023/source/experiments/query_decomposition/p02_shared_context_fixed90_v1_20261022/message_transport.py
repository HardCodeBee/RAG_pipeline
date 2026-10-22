"""Experiment-local support for complete string or multi-message payloads.

No provider/Cache/gate is opened on import. The frozen request factory and
shared runtime are unchanged. Only this module's builder binding differs in
the locally copied StrictAsk.__call__; all durable submission logic is exact.
This transport never owns, appends or commits a conversation transcript.
"""
from copy import deepcopy
import hashlib
import json
import time

from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload as single_user_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk as FrozenStrictAsk, complete_receipt, provider_gate,
)


CONTEXT_UNIT_LIMIT = 128000
MESSAGE_FRAMING_UNITS = 64


class ContextWindowHeld(ValueError):
    """Conservative pre-submission envelope exceeded; never truncate or send."""
    def __init__(self, estimated_units):
        self.estimated_units = estimated_units
        self.maximum_units = CONTEXT_UNIT_LIMIT
        super().__init__(f"Technical context hold: estimated_units={estimated_units}; limit={CONTEXT_UNIT_LIMIT}")


def checked_payload(payload):
    """A frozen conservative byte+framing envelope, not measured token usage."""
    maximum = payload["max_tokens"]
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Output allowance must be a positive integer")
    units = sum(len(m["content"].encode("utf8")) + MESSAGE_FRAMING_UNITS
                for m in payload["messages"]) + maximum
    if units > CONTEXT_UNIT_LIMIT:
        raise ContextWindowHeld(units)
    return payload


def build_payload(settings, prompt_or_messages, max_tokens):
    """Preserve native string wire or send a validated detached message list.

    Lists must be user/assistant/.../user with only role/content fields and
    nonempty text. Validation does not normalize whitespace, case, content or
    order. A caller's later append/mutation cannot change the returned payload.
    Existing native string inputs keep the exact original factory behavior.
    """
    if isinstance(prompt_or_messages, str):
        return checked_payload(single_user_payload(settings, prompt_or_messages, max_tokens))
    if not isinstance(prompt_or_messages, list):
        raise TypeError("Expected a prompt string or an explicit message list")
    messages = deepcopy(prompt_or_messages)
    if not messages or len(messages) % 2 != 1:
        raise ValueError("Messages must start/end with user and alternate roles")
    for index, message in enumerate(messages):
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain exactly role and content")
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Messages must start/end with user and alternate roles")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be nonempty text")
    payload = single_user_payload(settings, "", max_tokens)
    payload["messages"] = messages
    return checked_payload(payload)


class StrictAsk(FrozenStrictAsk):
    """Unchanged frozen submission method, resolving the local payload builder.

    __init__/_client/_retire/close are inherited; the experiment-local native
    adapter adds its existing return-evidence and ledger-before-cache guard.
    """
    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        if allow_final_truncation:
            raise ValueError("These probes require complete stop-ended outputs")
        request = build_payload(self.settings, prompt, maximum)
        payload = canonical(request)
        while True:
            row = self.cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
            if row:
                rid, status, raw = row
                result = {**(json.loads(raw) if raw else {}), "status": status}
                if complete_receipt(result):
                    self.cache.begin_request(stage, request)  # retain every logical use
                    self.events({"event": "model_call", "stage": stage, "request_id": rid,
                                 "source": "current_run", "reused": True, "usage": result["usage"]})
                    return rid, result["answer"].strip(), "stop"
                attempts = 1 + self.cache.db.execute(
                    "SELECT count(*) FROM retired_requests WHERE payload=?", (payload,)).fetchone()[0]
                transient = status == "failed" and result.get("http_status") in (408, 429, 500, 502, 503, 504)
                returned = any(result.get(k) is not None for k in
                               ("response_id", "raw_response", "usage", "returned_model"))
                returned = returned or bool((result.get("answer") or result.get("content") or "").strip())
                ambiguous = (status in ("unknown", "in_flight") and self.recover_unknown
                             and not returned)
                # A returned incomplete answer is terminal even if it has no usage.
                if attempts >= self.maximum_attempts or not (transient or ambiguous or result.get("proved_not_submitted")):
                    raise RuntimeError(f"Held terminal/exhausted request {rid}; attempts={attempts}; status={status}")
                self._retire(rid, stage, payload, attempts, "authorized_unknown_reissue" if ambiguous else "known_transient")
                time.sleep(min(3 * attempts, 15))
                continue
            if self.seed is not None:
                saved = self.seed.execute("SELECT request_id,result FROM requests WHERE payload=?", (payload,)).fetchone()
                result = json.loads(saved[1]) if saved else None
                if result is not None and complete_receipt(result):
                    rid, _ = self.cache.begin_request(stage, request)
                    self.cache.finish_request(rid, result)
                    self.events({"event": "model_call", "stage": stage, "request_id": rid,
                                 "source": "exact_frozen_replay/" + saved[0], "reused": True,
                                 "usage": result["usage"]})
                    return rid, result["answer"].strip(), "stop"
            client = self._client()  # Validate config/credential before reserving.
            with provider_gate(self.settings["base_url"], gate_root=self.gate_root, stop=self.stop):
                rid, existing = self.cache.begin_request(stage, request,
                    before_start=lambda request_id: self.budget.reserve(request_id, request))
                if existing is not None:
                    continue
                result = client.request_once(request)
                self.budget.finish(rid, result)
                self.cache.finish_request(rid, result)
            self.events({"event": "model_call", "stage": stage, "request_id": rid,
                         "source": "provider", "reused": False, "status": result.get("status"),
                         "finish_reason": result.get("finish_reason"), "usage": result.get("usage"),
                         "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()})
