"""Offline contracts for a fresh E0 and terminal algorithm outcomes; no network."""
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import json

from . import run
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as runtime


class FakeBudget:
    def __init__(self):
        self.reserved = []

    def reserve(self, rid, request):
        self.reserved.append(rid)

    def finish(self, rid, result):
        pass


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def request_once(self, payload):
        self.calls += 1
        return self.result

    def close(self):
        pass


class FakeAsk(runtime.StrictAsk):
    def __init__(self, *args, fake_client, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = fake_client

    def _client(self):
        return self.client


@contextmanager
def no_network_gate(*args, **kwargs):
    yield


def main():
    dry = run.dry_run()
    config = run.read(run.PACKAGE / "config.json")
    node = run.rows(run.INPUTS)[0]
    prompt = run.variant_prompt(node["control_prompt"], "E0")
    results = {"stop": {"status": "success", "finish_reason": "stop", "answer": "A fresh fake response.",
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
               "length": {"status": "success", "finish_reason": "length", "answer": "Incomplete",
                           "usage": {"prompt_tokens": 10, "completion_tokens": 128}},
               "empty": {"status": "success", "finish_reason": "stop", "answer": "",
                          "usage": {"prompt_tokens": 10, "completion_tokens": 0}},
               "refusal": {"status": "success", "finish_reason": "stop", "answer": "", "refusal": "fake refusal",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}}
    saved_gate = runtime.provider_gate
    runtime.provider_gate = no_network_gate
    checked = {}
    try:
        with TemporaryDirectory(prefix="p02_fewshot_offline_") as temporary:
            from pathlib import Path
            for label, receipt in results.items():
                fake = FakeClient(receipt)
                budget = FakeBudget()
                with run.Cache(Path(temporary) / (label + ".sqlite3")) as cache:
                    ask = FakeAsk(cache, None, config, budget, lambda value: None, fake_client=fake)
                    if label == "stop":
                        _, answer, _ = ask("fresh_E0", prompt, 128)
                        assert answer == "A fresh fake response." and fake.calls == 1
                        ask("fresh_E0_resume", prompt, 128)
                        assert fake.calls == 1 and len(budget.reserved) == 1
                    else:
                        for stage in ("first", "resume"):
                            try:
                                ask(stage, prompt, 128)
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError("Terminal algorithm receipt must not pass complete contract")
                        assert fake.calls == 1 and len(budget.reserved) == 1
                        assert run.algorithm_failure(receipt) == "terminal_" + label
                    checked[label] = dict(fake_actual_calls=fake.calls, reservations=len(budget.reserved),
                                          fresh_or_terminal_resume_does_not_repeat=True)
    finally:
        runtime.provider_gate = saved_gate
    assert run.algorithm_failure({"status": "unknown", "answer": ""}) is None
    assert run.algorithm_failure({"status": "success", "finish_reason": "stop", "answer": "Answer", "usage": None}) is None
    result = dict(passed=True, utc=run.utc(), dry_run_passed=dry["passed"],
                  shared_runtime_sha256=run.sha(run.PARENT / "shared_runtime.py"),
                  fresh_E0_verified_with_fake_submission=True, historical_receipt_not_used_as_fake_output=True,
                  per_receipt_contracts=checked, invalid_usage_remains_technical=True,
                  real_provider_requests=0, no_retrieval_or_GPU=True)
    run.atomic_json(run.OUT / "OFFLINE_CONTRACTS.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
