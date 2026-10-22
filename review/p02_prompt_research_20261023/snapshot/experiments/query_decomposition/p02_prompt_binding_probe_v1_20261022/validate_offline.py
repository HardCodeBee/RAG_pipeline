"""Meaningful offline checks of bounded submission/restart behavior; no provider."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock
from . import shared_runtime as runtime

OUT = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1" / HERE.name
CONFIG = {"generation": {"model": "fake", "base_url": "https://fake.invalid/v1", "timeout_seconds": 1,
                          "api_key_env": "UNUSED_OFFLINE_TEST_ENV"}}
SUCCESS = {"status": "success", "finish_reason": "stop", "answer": "A complete factual sentence.",
           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
UNKNOWN = {"status": "unknown", "failure_class": "APIConnectionError"}


class Budget:
    def __init__(self):
        self.reservations, self.finished = [], []

    def reserve(self, rid, payload):
        self.reservations.append(rid)

    def finish(self, rid, result):
        self.finished.append((rid, result))


class Client:
    def __init__(self, results):
        self.results, self.calls = list(results), 0

    def request_once(self, payload):
        self.calls += 1
        if not self.results:
            raise AssertionError("Unexpected extra SDK submission")
        return self.results.pop(0)

    def close(self):
        pass


class FakeAsk(runtime.StrictAsk):
    def __init__(self, *args, client, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def _client(self):
        return self.client


@contextmanager
def no_gate(*args, **kwargs):
    yield


def invoke(path, client, budget):
    with Cache(path) as cache:
        ask = FakeAsk(cache, None, CONFIG, budget, lambda event: None, client=client)
        try:
            return ask("fake/case/node", "fixed test input", 128)
        finally:
            ask.close()


def held(path, client, budget):
    try:
        invoke(path, client, budget)
    except RuntimeError:
        return
    raise AssertionError("Expected a held outcome")


def validate():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    old_gate, old_sleep = runtime.provider_gate, runtime.time.sleep
    try:
        runtime.provider_gate, runtime.time.sleep = no_gate, lambda delay: None
        with TemporaryDirectory(prefix="offline_checks_", dir=OUT) as directory:
            root = Path(directory).resolve()
            assert root.is_relative_to(OUT.resolve()), "Validate cleanup target stays in isolated output"
            budget = Budget()
            first = Client([SUCCESS])
            invoke(root / "success.sqlite3", first, budget)
            again = Client([])
            invoke(root / "success.sqlite3", again, budget)
            assert first.calls == 1 and again.calls == 0
            checks.append("complete_success_reused_without_new_submission_after_restart")
            length = Client([{**SUCCESS, "finish_reason": "length"}])
            held(root / "length.sqlite3", length, budget)
            held(root / "length.sqlite3", Client([]), budget)
            assert length.calls == 1
            checks.append("length_is_terminal_and_never_retried")
            for key, evidence in [("response_id", "received-id"), ("usage", {}),
                                  ("raw_response", {}), ("answer", "returned content")]:
                client = Client([{**UNKNOWN, key: evidence}])
                held(root / (key + ".sqlite3"), client, budget)
                held(root / (key + ".sqlite3"), Client([]), budget)
                assert client.calls == 1
            checks.append("unknown_with_any_receipt_or_content_is_held_across_restarts")
            transient = Client([UNKNOWN] * 5)
            held(root / "bounded.sqlite3", transient, budget)
            assert transient.calls == 5
            restarted = Client([])
            held(root / "bounded.sqlite3", restarted, budget)
            assert restarted.calls == 0
            db = sqlite3.connect(root / "bounded.sqlite3")
            assert db.execute("SELECT count(*) FROM retired_requests").fetchone()[0] == 4
            assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 1
            db.close()
            checks.append("maximum_five_attempts_persists_across_restart_and_preserves_all_receipts")
            # Simulate an interrupted process after reserving and before a return.
            with Cache(root / "interrupted.sqlite3") as cache:
                payload = runtime.build_payload(CONFIG["generation"], "fixed test input", 128)
                cache.begin_request("fake/case/node", payload, before_start=lambda rid: budget.reserve(rid, payload))
            resumed = Client([UNKNOWN] * 4)
            held(root / "interrupted.sqlite3", resumed, budget)
            assert resumed.calls == 4
            checks.append("interrupted_inflight_counts_as_prior_attempt_and_retains_reserve")
            # Check the platform's actual nonblocking lock behavior.
            lockdir = root / "lock_probe"
            with run_lock(lockdir):
                caught = False
                try:
                    with run_lock(lockdir):
                        pass
                except (RuntimeError, BlockingIOError):
                    caught = True
                assert caught, "The provider gate must recognize the platform lock-conflict exception"
            checks.append("actual_platform_run_lock_conflict_matches_provider_gate_handler")
    finally:
        runtime.provider_gate, runtime.time.sleep = old_gate, old_sleep
    result = {"passed": True, "checks": checks, "new_provider_submissions": 0,
              "new_global_charges": 0, "new_retrieval_or_GPU_jobs": 0}
    runtime.atomic_json(OUT / "OFFLINE_RUNTIME_VALIDATION.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=True))
