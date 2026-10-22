"""C8 pure-contract plus temporary SQLite/fake-client checks; no paid state."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from experiments.query_decomposition.p02_failure8_actual_target_probe_v1_20261022 import prepare as app
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.runtime import CandidateAsk, normal_complete, algorithm_failure
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import run as source_run
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.validate_runner_offline import Budget, Client, fake_gate
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as shared


class Ask(CandidateAsk):
    def __init__(self, cache, config, budget, client):
        super().__init__(cache, None, config, budget, lambda _: None, maximum_attempts=5, recover_unknown=True)
        self.fixture_client = client

    def _client(self):
        return self.fixture_client


def validate():
    app.dry_run()
    protocol_before = app.sha(app.PROTOCOL)
    items = {r["probe_id"]: r for r in app.rows(app.INPUTS)}
    controls = {r["probe_id"]: r for r in app.read(app.CONTROLS)["controls"]}
    uses = app.rows(app.HERE / "LOGICAL_USES16.frozen.jsonl")
    config = app.read(app.PACKAGE / "config.json")
    model = config["generation"]["model"]
    good = dict(status="success", answer="Offline complete response.", finish_reason="stop",
        requested_model=model, returned_model=model, usage=dict(prompt_tokens=7, completion_tokens=2, total_tokens=9))
    checks = {}
    outputs = {}
    app.OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="C_fake_", dir=app.OUT) as directory, \
            patch.object(shared, "provider_gate", fake_gate), patch.object(shared.time, "sleep", lambda _: None):
        root = Path(directory)
        cache = source_run.Cache(root / "checkpoint.sqlite3")
        budget = Budget(root / "ledger.sqlite3", root)
        try:
            client = Client([good] * 5)
            ask = Ask(cache, config, budget, client)
            for use in uses:
                item = items[use["probe_id"]]
                if use["reuse_already_paid_I0"]:
                    control = controls[use["probe_id"]]
                    assert app.digest(control["source_payload"]) == app.digest(
                        app.build_payload(config["generation"], app.templates.prompt(item, use["arm"]), 128))
                    outputs[use["task_id"]] = dict(request_id=control["source_request_id"],
                        answer=control["result"]["answer"], source_reuse=True, new_charge_CNY=0)
                else:
                    rid, answer, finish = ask(use["task_id"], app.templates.prompt(item, use["arm"]), 128)
                    assert finish == "stop"
                    outputs[use["task_id"]] = dict(request_id=rid, answer=answer, source_reuse=False)
            assert len(outputs) == 16 and client.calls == 5
            assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 5
            assert budget.db.execute("SELECT count(*) FROM charges").fetchone()[0] == 5
            assert sum(r["source_reuse"] for r in outputs.values()) == 11
            checks["16logical_11exact_source_reuses_only5new_fake_calls"] = True
            for item in items.values():
                if item["current_question"] == item["frozen_rewritten_query"]:
                    assert outputs[item["probe_id"] + "/C0"] == outputs[item["probe_id"] + "/C1"]
            checks["3unchanged_C1_exact_same_paid_I0_receipt_no_fake_call"] = True
        finally:
            cache.close()
            budget.close()
        cache = source_run.Cache(root / "checkpoint.sqlite3")
        budget = Budget(root / "ledger.sqlite3", root)
        try:
            client = Client([])
            ask = Ask(cache, config, budget, client)
            for use in uses:
                if not use["reuse_already_paid_I0"]:
                    item = items[use["probe_id"]]
                    rid, answer, finish = ask(use["task_id"], app.templates.prompt(item, use["arm"]), 128)
                    assert rid == outputs[use["task_id"]]["request_id"]
            assert client.calls == 0
            checks["all5_new_payloads_resume_without_reroll_or_new_charge"] = True
        finally:
            cache.close()
            budget.close()
    rendered = app.rows(app.HERE / "RENDERED_PAYLOADS16.frozen.jsonl")
    for r in rendered:
        item = items[r["probe_id"]]
        assert r["original_raw_current_evaluation_target"] == item["current_question"]
        assert r["actual_model_current"] == app.templates.model_current(item, r["arm"])
    checks["16evaluation_targets_stay_raw_including5_changed_actual_tasks"] = True
    # Deliberately damaged rewrite stays on the wire; no hidden gold/manual repair.
    example = deepcopy(next(iter(items.values())))
    example["frozen_rewritten_query"] = "Which deliberately different relation is asked?"
    text = app.templates.prompt(example, "C1")
    assert "Which deliberately different relation is asked?" in text.split("\n\nActual task:\n", 1)[1]
    assert app.templates.evaluation_target(example) == example["current_question"]
    checks["deviating_rewrite_not_silently_corrected_or_new_gold_target"] = True
    for changes in [dict(returned_model="wrong"), dict(usage=None),
                    dict(usage=dict(prompt_tokens=True, completion_tokens=2, total_tokens=3)),
                    dict(usage=dict(prompt_tokens=7, completion_tokens=2, total_tokens=10))]:
        value = {**deepcopy(good), **changes}
        assert not normal_complete(value, model) and algorithm_failure(value, model) is None
    assert algorithm_failure({**good, "finish_reason": "length"}, model) == "terminal_length"
    checks["immutable_B_runtime_strict_model_usage_and_terminal_rules"] = True
    assert app.sha(app.PROTOCOL) == protocol_before
    assert not (app.OUT / "checkpoint.sqlite3").exists()
    assert not (app.HERE / "RUN_AUTHORIZATION.json").exists() and not (app.HERE / "run.py").exists()
    result = dict(passed=True, checks=checks, fake_new_calls=5, real_provider_requests=0,
        production_Cache_created=False, global_ledger_modified=False, source_controls_modified=False,
        no_execution_entry=True, no_run_authorization=True, protocol_sha256=protocol_before)
    app.save(app.OUT / "OFFLINE_VALIDATION.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=True))
