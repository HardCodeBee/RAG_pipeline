"""Native stage-aware adapter over unchanged StrictAsk; no automatic execution."""

import hashlib
import json

from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from .message_transport import build_payload
from .message_transport import StrictAsk


def usage_ok(receipt):
    u = receipt.get("usage") or {}
    return (all(type(u.get(k)) is int and u[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"])


def integrity(receipt, model):
    return (receipt.get("status") == "success" and usage_ok(receipt)
        and receipt.get("requested_model") == receipt.get("returned_model") == model)


def stage_success(receipt, model, stage):
    return (integrity(receipt, model) and receipt.get("finish_reason") in
        (("stop", "length") if stage == "legacy/final" else ("stop",))
        and isinstance(receipt.get("answer"), str) and bool(receipt["answer"].strip())
        and not receipt.get("refusal"))


def terminal_reason(receipt, model):
    if not integrity(receipt, model):
        return None
    if receipt.get("refusal"):
        return "terminal_refusal"
    if receipt.get("finish_reason") in ("length", "content_filter"):
        return "terminal_" + receipt["finish_reason"]
    if isinstance(receipt.get("answer"), str) and not receipt["answer"].strip():
        return "terminal_empty"
    return None


def any_returned(receipt):
    if any(receipt.get(k) is not None for k in ("response_id", "raw_response", "usage", "returned_model", "refusal", "finish_reason")):
        return True
    value = receipt.get("answer") or receipt.get("content") or ""
    return bool(value.strip()) if isinstance(value, str) else bool(value)


class SubmissionAwareStrictAsk(StrictAsk):
    """Preserve all returned signs plus ledger-before-cache crash window."""
    def _retire(self, rid, stage, payload, attempts, reason):
        row = self.cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()
        receipt = json.loads(row[0]) if row and row[0] else {}
        charge = self.budget.db.execute("SELECT actual,status FROM charges WHERE id=?", (rid,)).fetchone()
        if any_returned(receipt) or (charge and (charge[0] is not None or charge[1] == "returned_usage")):
            raise RuntimeError("Held returned receipt/ledger; no blind replay")
        return super()._retire(rid, stage, payload, attempts, reason)


class AlgorithmTerminal(RuntimeError):
    def __init__(self, value):
        self.value = value
        super().__init__("Algorithm terminal at " + value["stage"] + ": " + value["terminal_reason"])


class TechnicalHeld(RuntimeError):
    pass


class NativeAsk:
    """Attach one condition's logical stages to persistent exact-payload calls.

    Shared StrictAsk always receives allow_final_truncation=False. An already
    persisted complete final length receipt is accepted here without a second
    request. Rewrite/ordinary/goal/formatter require stop. Every path validates
    exact model, all3 integer token fields and their sum before propagation.
    """
    def __init__(self, cache, strict, config, task_id, *, event=None):
        self.cache, self.strict, self.config, self.task_id = cache, strict, config, task_id
        self.event, self.stages = event or (lambda event: None), []

    def active(self, payload):
        row = self.cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (canonical(payload),)).fetchone()
        return (row[0], {**(json.loads(row[2]) if row[2] else {}), "status": row[1]}) if row else (None, {})

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        if bool(allow_final_truncation) != (stage == "legacy/final"):
            raise ValueError("Native final-only truncation flag contract")
        payload = build_payload(self.config["generation"], prompt, maximum)
        payload_sha = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
        stage_key = self.task_id + "/" + stage
        model = self.config["generation"]["model"]
        saved = self.cache.get("stage", stage_key)

        def classify(rid, receipt):
            success = stage_success(receipt, model, stage)
            reason = None if success else terminal_reason(receipt, model)
            if not success and reason is None:
                return None
            answer = receipt.get("answer")
            return dict(stage=stage, request_id=rid, payload_sha256=payload_sha,
                answer=answer.strip() if isinstance(answer, str) else "", finish_reason=receipt.get("finish_reason"),
                usage=receipt["usage"], terminal_reason=reason)

        rid, receipt = self.active(payload)
        if saved is not None:
            value = classify(rid, receipt)
            if value != saved:
                raise TechnicalHeld("Saved stage binding/receipt mismatch: " + stage)
        else:
            value = classify(rid, receipt)
            if value is None:
                try:
                    # NEVER forward native allow_final_truncation to StrictAsk.
                    self.strict("full8/" + stage_key, prompt, maximum)
                except (RuntimeError, InterruptedError, ValueError, TypeError, AttributeError) as exc:
                    rid, receipt = self.active(payload)
                    value = classify(rid, receipt)
                    if value is None:
                        raise TechnicalHeld("Held stage " + stage + "; " + type(exc).__name__) from None
                else:
                    rid, receipt = self.active(payload)
                    value = classify(rid, receipt)
            if value is None:
                raise TechnicalHeld("Incomplete exact model/usage receipt: " + stage)
            # Preserve logical stage use even on exact final-length reuse.
            self.cache.begin_request("full8/" + stage_key, payload)
            self.cache.put("stage", stage_key, value)
        self.stages.append(value)
        self.event(dict(event="logical_stage", task_id=self.task_id, **value))
        if value["terminal_reason"]:
            raise AlgorithmTerminal(value)
        return value["request_id"], value["answer"], value["finish_reason"]
