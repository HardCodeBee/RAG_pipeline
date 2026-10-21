"""Private copy of bounded transport; only input ceiling and tiered prices changed."""
from __future__ import annotations
import json,re,time
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import canonical,digest,utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json as write_json
from .common import OUT,MAX_INPUT,CONTEXT,IntegrityError,reserve_cost,usage_cost

class RawTransport:
    def __init__(self, store: Store, client, budget: Budget, model="qwen-plus-2025-12-01"):
        self.store, self.client, self.budget, self.model = store, client, budget, model
        self.last_start = 0.0
        self.new_requests = 0
        self.qid = None

    def ask(self, stage: str, prompt: str, max_tokens: int):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Empty prompt")
        # Conservative byte upper bound, not a claim of exact tokenizer count.
        byte_upper = len(prompt.encode("utf-8")) + 256
        if byte_upper > MAX_INPUT or byte_upper + max_tokens > CONTEXT:
            raise ValueError("Context byte safety limit exceeded; no silent truncation")
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": max_tokens, "extra_body": {"enable_thinking": False}}
        request_id = digest([self.store.namespace, payload])
        if self.qid is not None:
            shown = {}
            for (raw_packet,) in self.store.db.execute("SELECT value FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid') AND json_extract(value,'$.qid')=?", (self.qid,)):
                packet = json.loads(raw_packet)
                for doc in packet.get("docs", []):
                    if doc.get("text") and doc["text"] in prompt:
                        shown[doc["doc_id"]] = doc
            self.store.put("evidence", digest([self.qid, request_id, stage]),
                           {"qid": self.qid, "stage": stage, "request_id": request_id,
                            "doc_ids": sorted(shown), "source_match": "exact raw corpus text present in this call; generated answers are not source evidence"})
        db = self.store.db
        existing = db.execute("SELECT status,result FROM requests WHERE id=?", (request_id,)).fetchone()
        if existing:
            if existing[0] in {"unknown", "in_flight"}:
                raise RuntimeError("Unknown submission requires reconciliation")
            self.store.event("logical_api_call", qid=self.qid, stage=stage, request_id=request_id, cache_hit=True, max_tokens=max_tokens)
            return request_id, json.loads(existing[1])
        self.store.event("logical_api_call", qid=self.qid, stage=stage, request_id=request_id, cache_hit=False, max_tokens=max_tokens)
        network_retries = 0
        for attempt in range(4):
            if (OUT / 'STOP_REQUESTED').exists():
                raise IntegrityError('Stop requested before a new submission attempt')
            self.budget.wait_for_start()
            if (OUT / 'STOP_REQUESTED').exists():
                raise IntegrityError('Stop requested while waiting for the rate slot')
            liability_id = digest([request_id, attempt])
            self.budget.reserve(liability_id, self.store.namespace, reserve_cost(byte_upper,max_tokens))
            db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,'in_flight',NULL)", (request_id, stage, canonical(payload)))
            db.execute("UPDATE requests SET status='in_flight' WHERE id=?", (request_id,))
            cursor = db.execute("INSERT INTO attempts(request_id,started,status) VALUES(?,?,'in_flight')", (request_id, utc()))
            attempt_id = cursor.lastrowid
            db.commit()
            self.last_start = time.monotonic()
            started = time.perf_counter()
            retryable_network_failure = False
            try:
                response = self.client.chat.completions.create(**payload)
                raw = response.model_dump(mode="json")
                choice = response.choices[0]
                usage = raw.get("usage") or {}
                result = {"status": "success", "answer": choice.message.content or "", "finish_reason": choice.finish_reason,
                          "refusal": getattr(choice.message, "refusal", None), "returned_model": response.model,
                          "usage": usage, "raw_response": raw, "latency_seconds": time.perf_counter() - started}
                if usage:
                    self.budget.settle(liability_id, usage_cost(usage.get("prompt_tokens",0),usage.get("completion_tokens",0)), "returned_usage")
                if response.model != self.model:
                    result["status"] = "model_identity_mismatch"
                if usage.get("prompt_tokens", 0) > MAX_INPUT:
                    result["status"] = "input_token_limit_violation"
            except Exception as exc:
                import httpx
                from openai import APIConnectionError, APITimeoutError
                retryable_network_failure = isinstance(exc, (
                    APIConnectionError, APITimeoutError, httpx.ConnectError,
                    httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout,
                    httpx.WriteError, httpx.WriteTimeout, httpx.RemoteProtocolError,
                ))
                code = getattr(exc, "status_code", None)
                body = getattr(exc, "body", {})
                body = body.get("error", body) if isinstance(body, dict) else {}
                error_code = body.get("code") if isinstance(body, dict) else None
                error_code = error_code if isinstance(error_code, str) and re.fullmatch(r"[\w.-]{1,100}", error_code) else None
                causes, cause = [], exc
                while cause is not None and len(causes) < 6:
                    causes.append({"type": type(cause).__name__, "errno": getattr(cause, "errno", None)})
                    cause = cause.__cause__ or cause.__context__
                result = {"status": "failed" if code else "unknown", "http_status": code,
                          "failure_class": type(exc).__name__, "error_code": error_code,
                          "failure_causes": causes, "latency_seconds": time.perf_counter() - started}
                if error_code in {"data_inspection_failed", "DataInspectionFailed"} and code in {None, 400, 422}:
                    result["status"] = "provider_content_rejected"
                    result["retryable"] = False
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                try:
                    result["retry_after_seconds"] = max(0, float(retry_after)) if retry_after else None
                except (TypeError, ValueError):
                    result["retry_after_seconds"] = None
                if code in {400, 401, 403, 404, 422, 429} and result['status'] != 'provider_content_rejected':
                    self.budget.settle(liability_id, 0.0, "http_failed_not_returned_usage")
            db.execute("UPDATE attempts SET status=?,result=? WHERE id=?", (result["status"], canonical(result), attempt_id))
            db.execute("UPDATE requests SET status=?,result=? WHERE id=?", (result["status"], canonical(result), request_id))
            db.commit()
            self.store.event("api_attempt", stage=stage, request_id=request_id, attempt=attempt, status=result["status"])
            self.new_requests += 1
            self.store.export()
            write_json(self.budget.path.parent / "budget_status.json", self.budget.summary())
            if (result["status"] == "unknown" and retryable_network_failure
                    and network_retries < 2 and attempt < 3):
                delay = (5, 15)[network_retries]
                network_retries += 1
                self.store.event("unknown_network_retry_scheduled", qid=self.qid,
                                 stage=stage, request_id=request_id, attempt=attempt,
                                 network_retry=network_retries, delay_seconds=delay,
                                 original_liability_id=liability_id,
                                 original_reserve_kept=True,
                                 policy="new in-process requests only; first complete response")
                time.sleep(delay)
                continue
            rate_codes = {"Throttling.RateQuota", "Throttling.RequestQuota", "rate_limit_exceeded", "Throttling"}
            if result.get("http_status") == 429 and result.get("error_code") in rate_codes and attempt < 3:
                delay = max([30, 60, 60][attempt], result.get("retry_after_seconds") or 0)
                if delay > 300:
                    raise RuntimeError("Long provider cooldown; stop instead of busy retrying")
                remaining = delay
                while remaining > 0:
                    time.sleep(min(60, remaining))
                    remaining -= min(60, remaining)
                continue
            if result["status"] in {"unknown", "model_identity_mismatch", "input_token_limit_violation"} or result.get("http_status") in {401, 403, 404, 429}:
                raise RuntimeError("Transport stopped; inspect sanitized method status")
            return request_id, result
        raise AssertionError("Unreachable request state")
