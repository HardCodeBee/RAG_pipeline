"""Bounded query-independent client preparation for the M6 transport amendment."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import threading
import time

import m6_preconnect_transport as v1

WORKERS = 4
PREPARATION_ATTEMPTS = 8
METADATA_TIMEOUT = 12.0
KEEPALIVE_EXPIRY = 120.0
MAX_PREPARED_AGE = 30.0
MAX_FIRST_USE_AGE = 60.0


def new_client(**sdk_kwargs):
    """Retain SDK defaults except the explicitly amended idle expiry."""
    from openai import OpenAI, DefaultHttpxClient
    import httpx

    if sdk_kwargs.get("timeout") != 60 or sdk_kwargs.get("max_retries") != 0:
        raise ValueError("Frozen generation timeout and retries required")
    if set(sdk_kwargs) - {"timeout", "max_retries", "api_key"}:
        raise ValueError("Unexpected SDK configuration")
    http_client = DefaultHttpxClient(timeout=sdk_kwargs["timeout"], limits=httpx.Limits(
        max_connections=1000, max_keepalive_connections=100,
        keepalive_expiry=KEEPALIVE_EXPIRY))
    try:
        return OpenAI(http_client=http_client, **sdk_kwargs)
    except BaseException:
        try:
            http_client.close()
        except Exception:
            pass
        raise


def close_clients(clients):
    """Close each owned initial client once, including unassigned clients."""
    for entry in clients:
        with entry["close_lock"]:
            if entry["closed"]:
                continue
            entry["closed"] = True
            try:
                entry["client"].close()
            except Exception:
                pass


def prepare_clients(engine, contract, readiness_path, amendment_sha):
    readiness_path = Path(readiness_path)
    if readiness_path.exists():
        raise ValueError("Preserve existing readiness receipt")
    generation = contract["generation"]
    if generation["timeout_seconds"] != 60 or generation["max_retries"] != 0:
        raise ValueError("Frozen generation timeout and retries required")
    started = time.time()
    opened, opened_lock = [], threading.Lock()
    transferred = False

    def worker(number):
        client_id = f"ready-{number}"
        calls = []
        for attempt in range(1, PREPARATION_ATTEMPTS + 1):
            entry = None
            call_started = time.monotonic()
            try:
                client = new_client(timeout=60, max_retries=0)
                entry = {"id": client_id, "client": client, "closed": False,
                         "close_lock": threading.Lock()}
                with opened_lock:
                    opened.append(entry)
                result = client.models.retrieve(generation["model"], timeout=METADATA_TIMEOUT)
                if result.id != generation["model"]:
                    raise ValueError("Metadata model identity mismatch")
            except Exception as error:
                metadata, retryable = engine.safe_failure(error)
                if entry is not None:
                    close_clients([entry])
                call = {"attempt": attempt, "status": "failure", "retryable": retryable,
                        "elapsed_seconds": time.monotonic() - call_started, **metadata}
                calls.append(call)
                print(json.dumps({"status": "client_preparation_attempt", "client_id": client_id, **call}), flush=True)
                if not retryable:
                    break
            else:
                entry["ready_at_monotonic"] = time.monotonic()
                entry["ready_at_unix"] = time.time()
                call = {"attempt": attempt, "status": "success",
                        "elapsed_seconds": entry["ready_at_monotonic"] - call_started}
                calls.append(call)
                print(json.dumps({"status": "client_preparation_attempt", "client_id": client_id, **call}), flush=True)
                return entry, {"client_id": client_id, "status": "ready", "metadata_calls": calls,
                               "ready_at_unix": entry["ready_at_unix"]}
        return None, {"client_id": client_id, "status": "failed", "metadata_calls": calls}

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            results = list(executor.map(worker, range(1, WORKERS + 1)))
        clients = [entry for entry, _receipt in results if entry is not None]
        finished_monotonic = time.monotonic()
        ages = [finished_monotonic - entry["ready_at_monotonic"] for entry in clients]
        all_ready = len(clients) == WORKERS
        fresh = all_ready and all(0 <= age <= MAX_PREPARED_AGE for age in ages)
        status = "ready" if fresh else ("ready_clients_expired" if all_ready else "incomplete_client_preparation")
        receipt = {"status": status, "transport_amendment_sha256": amendment_sha,
            "started_at_unix": started, "finished_at_unix": time.time(),
            "required_clients": WORKERS, "successful_clients": len(clients),
            "maximum_attempts_per_client": PREPARATION_ATTEMPTS, "metadata_timeout_seconds": METADATA_TIMEOUT,
            "keepalive_expiry_seconds": KEEPALIVE_EXPIRY, "maximum_prepared_age_seconds": MAX_PREPARED_AGE,
            "maximum_ready_age_seconds": max(ages) if ages else None,
            "clients": [item for _entry, item in results],
            "metadata_calls": sum(len(item["metadata_calls"]) for _entry, item in results),
            "generation_requests": 0, "ledger_attempts_created": 0,
            "query_or_answer_inputs_used": False, "partial_policy_effects_computed": False}
        for entry, item in results:
            if entry is not None:
                item["ready_age_at_preparation_end_seconds"] = finished_monotonic - entry["ready_at_monotonic"]
        if not fresh:
            close_clients(opened)
            clients = []
        with readiness_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": status, "successful_clients": receipt["successful_clients"],
                          "metadata_calls": receipt["metadata_calls"], "ledger_attempts_created": 0}), flush=True)
        transferred = fresh
        return clients, receipt
    finally:
        if not transferred:
            close_clients(opened)


def make_provider_factory(engine, contract, clients, amendment, readiness_sha):
    if len(clients) != WORKERS or len({entry["id"] for entry in clients}) != WORKERS:
        raise ValueError("Exactly four distinct prepared clients required")
    if len({id(entry["client"]) for entry in clients}) != WORKERS or any(entry["closed"] for entry in clients):
        raise ValueError("Prepared clients must be distinct and open")
    available = queue.Queue()
    for entry in clients:
        available.put(entry)
    base_factory = v1.provider_factory(engine, contract, amendment, client_factory=new_client)
    probe = base_factory()
    base = type(probe)
    probe.close()

    class ReadyProvider(base):
        def reset_client(self):
            entry = getattr(self, "_ready_entry", None)
            if entry is not None and self.generator._client is entry["client"]:
                self.generator._client = None
                close_clients([entry])
            else:
                super().reset_client()

        def __call__(self, row, repeat_id):
            trace = {"transport_amendment_sha256": amendment, "readiness_sha256": readiness_sha}
            if not hasattr(self, "_ready_entry") and not hasattr(self, "_assignment_failure"):
                try:
                    entry = available.get_nowait()
                except queue.Empty:
                    self._assignment_failure = {**trace, "stage": "prepared_client_assignment",
                        "generation_request_submitted": False, "error_chain": ["PreparedClientPoolExhausted"]}
                else:
                    self._ready_entry = entry
                    age = time.monotonic() - entry["ready_at_monotonic"]
                    self._first_use_age = age
                    if entry["closed"] or not 0 <= age <= MAX_FIRST_USE_AGE:
                        close_clients([entry])
                        self._assignment_failure = {**trace, "ready_client_id": entry["id"],
                            "ready_client_age_at_first_call_seconds": age, "stage": "prepared_client_age_check",
                            "generation_request_submitted": False, "error_chain": ["PreparedClientExpiredOrClosed"]}
                    else:
                        self.generator._client = entry["client"]
            if hasattr(self, "_assignment_failure"):
                raise engine.AttemptFailure(self._assignment_failure, False) from None
            entry = self._ready_entry
            trace.update(ready_client_id=entry["id"], ready_client_age_at_first_call_seconds=self._first_use_age,
                         ready_client_used_for_this_attempt=self.generator._client is entry["client"])
            try:
                result = super().__call__(row, repeat_id)
            except engine.AttemptFailure as error:
                raise engine.AttemptFailure({**error.metadata, **trace}, error.retryable) from None
            return {**result, **trace}

    return lambda: ReadyProvider(contract)
