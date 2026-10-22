"""Shared submission gate and durable accounting for approved small probes.

Importing this module does not open the provider or change campaign holds.
No SDK/HTTP retries are enabled: each durable attempt is one SDK call.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import LLM, build_payload
from p02_multigrain_timing_cachefix_v1.core.runtime import Budget

GATE_ROOT = ROOT / "outputs/query_decomposition/p02_prompt_research_v1_20261022/provider_gate"
INPUT_RATE = 0.8e-6
OUTPUT_RATE = 2e-6


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(temporary, path)


@contextmanager
def provider_gate(base_url, *, gate_root=GATE_ROOT, interval=3.0, stop=None):
    """Serialize real calls across cooperating processes and space their starts.

The lock remains held through the complete HTTP call. Legacy experiments do
not participate; their scheduling must be coordinated by the caller.
"""
    identity = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:20]
    directory = Path(gate_root) / identity
    while True:
        if stop and stop():
            raise InterruptedError("Stopped before provider submission")
        lock = run_lock(directory)
        try:
            lock.__enter__()
            break
        except (RuntimeError, BlockingIOError):
            time.sleep(0.25)
    try:
        state = directory / "last_start.json"
        previous = json.loads(state.read_text(encoding="utf8")) if state.exists() else {}
        while True:
            delay = previous.get("epoch", 0) + interval - time.time()
            if delay <= 0:
                break
            if stop and stop():
                raise InterruptedError("Stopped while waiting for submission interval")
            time.sleep(min(delay, 0.25))
        if stop and stop():
            raise InterruptedError("Stopped before provider submission")
        atomic_json(state, {"epoch": time.time(), "utc": utc(), "minimum_interval_seconds": interval})
        yield
    finally:
        lock.__exit__(None, None, None)


class NoCapBudget(Budget):
    """Use the existing charges ledger without historical fixed budget caps."""
    def __init__(self, config, output):
        ledger = Path(config["budget"]["ledger"])
        if not ledger.is_file():
            raise FileNotFoundError("Existing cumulative charges ledger is required")
        self.db = sqlite3.connect(ledger, timeout=30)
        self.namespace = "p02_prompt_research_v1_20261022/" + str(Path(output).resolve())

    def reserve(self, request_id, request):
        reserve = sum(len(m["content"].encode("utf8")) for m in request["messages"]) * INPUT_RATE
        reserve += request["max_tokens"] * OUTPUT_RATE
        with self.db:
            self.db.execute(
                "INSERT INTO charges(id,namespace,reserved,actual,status) VALUES(?,?,?,NULL,'reserved')",
                (request_id, self.namespace, reserve),
            )


def complete_receipt(result):
    """Only complete successful content with returned usage is reusable."""
    usage = result.get("usage") or {}
    return (
        result.get("status") == "success"
        and result.get("finish_reason") == "stop"
        and isinstance(result.get("answer"), str)
        and bool(result["answer"].strip())
        and not result.get("refusal")
        and all(isinstance(usage.get(k), int) and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens"))
    )


class StrictAsk:
    """Durable bounded recovery using native Cache and the shared provider gate.

All attempts and unresolved charge reservations survive restarts. A successful
but length-ended/empty/refused answer is an algorithm outcome, never rerolled.
Only technical failures can be recovered; unknown reissues are explicit in the
protocol/event log. One run_lock must surround the caller's whole execution.
"""
    def __init__(self, cache, seed, config, budget, events, *, recover_unknown=True,
                 maximum_attempts=5, stop=None, gate_root=GATE_ROOT):
        self.cache, self.seed, self.settings = cache, seed, config["generation"]
        self.budget, self.events = budget, events
        self.recover_unknown, self.maximum_attempts = recover_unknown, maximum_attempts
        self.stop, self.gate_root = stop, gate_root
        self.client = None
        cache.db.execute("CREATE TABLE IF NOT EXISTS retired_requests AS SELECT * FROM requests WHERE 0")
        cache.db.execute("CREATE TABLE IF NOT EXISTS retired_request_stages AS SELECT * FROM request_stages WHERE 0")
        cache.db.commit()

    def _retire(self, rid, stage, payload, attempts, reason):
        with self.cache.db:
            self.cache.db.execute("INSERT INTO retired_requests SELECT * FROM requests WHERE id=?", (rid,))
            self.cache.db.execute("INSERT INTO retired_request_stages SELECT * FROM request_stages WHERE request_id=?", (rid,))
            self.cache.db.execute("DELETE FROM request_stages WHERE request_id=?", (rid,))
            self.cache.db.execute("DELETE FROM requests WHERE id=?", (rid,))
        self.events({"event": "bounded_technical_recovery", "utc": utc(), "stage": stage,
                     "retired_request_id": rid, "attempts_used": attempts,
                     "maximum_attempts": self.maximum_attempts, "reason": reason,
                     "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                     "old_receipt_and_reservation_preserved": True})

    def _client(self):
        if self.client is None:
            import httpx
            from openai import OpenAI
            key = os.environ.get(self.settings["api_key_env"])
            if not key:
                raise ValueError("Required provider credential is absent from process environment")
            timeout = httpx.Timeout(self.settings["timeout_seconds"], connect=10)
            http = httpx.Client(transport=httpx.HTTPTransport(retries=0,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)),
                timeout=timeout, trust_env=False)
            api = OpenAI(api_key=key, base_url=self.settings["base_url"], timeout=timeout,
                         max_retries=0, http_client=http)
            self.client = LLM(self.settings, client=api)
            self.client.owns_client = True
        return self.client

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
            # Loop validates the recorded response or performs bounded technical recovery.

    def close(self):
        if self.client is not None:
            self.client.close()
