"""Run the frozen M6 collector with an explicitly recorded transport amendment.

Preserves original code, six attempts per outcome, four workers and $15 ceiling.
The wrapper never computes partial policy effects. A high failure rate stops the
continuation for inspection instead of consuming the remaining budget blindly.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
DIRECTORY = RESEARCH / "m6_beir_validation_v1" / "answers_v1"
RECORDS = DIRECTORY / "recovery_20260915_v1"
PROTOCOL_SHA = "5b4faa41dc4a3ae83400d31a796354cdafb62cb461bfc6ba1a6f594b2b7bc785"
BATCH_ATTEMPTS = 512
MAX_BATCHES = 60


def state():
    con = sqlite3.connect((DIRECTORY / "attempts.sqlite3").as_uri()+"?mode=ro", uri=True)
    try:
        return {"attempts": con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                "successes": con.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0],
                "charged_or_reserved_micro_usd": con.execute("SELECT SUM(charged_micro_usd) FROM attempts").fetchone()[0],
                "statuses": dict(con.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status"))}
    finally:
        con.close()


def stop_rule(before, after, stop_reason):
    if after["successes"] == 31254 and stop_reason == "complete":
        return "complete"
    if stop_reason != "invocation_attempt_limit":
        return "collector_stopped_"+stop_reason
    attempts = after["attempts"]-before["attempts"]
    successes = after["successes"]-before["successes"]
    if attempts <= 0 or successes < 0 or successes > attempts:
        return "invalid_batch_accounting"
    if (attempts-successes)*10 > attempts:
        return "transport_failure_fraction_above_10_percent"
    return None


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def update_progress(value):
    path = RESEARCH / "validation_fresh4000_e03b_e1_v1" / "progress.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    node = document["m6_candidate_validation"]["beir_dev_validation"]
    node["status"] = value["status"]
    node["recovery_20260915"] = value
    node["answer_collection"]["status"] = value["status"]
    node["answer_collection"]["recovery_current_process_id"] = value.get("process_id")
    node["answer_collection"]["recovery_latest_ledger"] = value["ledger"]
    node["new_paid_calls"] = value["ledger"]["attempts"]
    node["new_paid_calls_semantics"] = "Reserved ledger attempts; may include preconnect-only failures. Metadata GETs are separately recorded per attempt."
    node["new_successful_answers"] = value["ledger"]["successes"]
    node["new_quality_results"] = 0
    node["after_collection_analysis"]["status"] = "awaiting_complete_recovered_collection; previous waiter no longer active"
    document["last_goal_continuation_audit"] = {
        "classification": "progress", "evidence": "four interrupted attempts reconciled with retained charges; frozen collection resumed",
        "observed_at_unix": time.time(), "goal_achieved": False,
    }
    temporary = path.with_suffix(".recovery.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def run(pilot=False):
    if not os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip():
        raise ValueError("Existing credential and standard endpoint required")
    sys.path.insert(0, str(RESEARCH))
    import m6_beir_answer_collection as collection
    import m6_preconnect_transport as transport

    prefix = "preconnect_pilot" if pilot else "continuation"
    if (RECORDS / (prefix + "_plan.json")).exists():
        raise ValueError("Continuation already started; inspect process and ledger before restarting")
    recovery = json.loads((RECORDS / "recovery_ledger_checks.json").read_text(encoding="utf-8"))
    assert recovery["status"] == "passed_collection_integrity" and recovery["successful_generations"] == 21686
    rows, contract, binding = collection.load_prepared(PROTOCOL_SHA)
    budget = collection.common.load_budget(DIRECTORY / "budget.json", binding, contract["generation"]["model"])
    assert budget["hard_budget_usd"] == 15.0 and budget["maximum_workers"] == 4 and budget["maximum_attempts_per_outcome"] == 6
    assert contract["generation"]["timeout_seconds"] == 60 and contract["generation"]["max_retries"] == 0
    assert collection.DIRECTORY.resolve() == DIRECTORY.resolve()
    initial = state()
    assert not (set(initial["statuses"])-{"success", "failure"})
    assert initial["successes"] >= 21686 and initial["charged_or_reserved_micro_usd"] < budget["ceiling_micro_usd"]
    source_hashes = {Path(p).name: collection.common.sha256(p) for p in (__file__, transport.__file__)}
    if not pilot:
        pilot_plan = json.loads((RECORDS / "preconnect_pilot_plan.json").read_text(encoding="utf-8"))
        pilot_result = json.loads((RECORDS / "preconnect_pilot_terminal.json").read_text(encoding="utf-8"))
        assert pilot_plan["source_hashes"] == source_hashes
        assert pilot_result["stop_reason"] == "pilot_passed_6_of_6"
        assert pilot_result["ledger"] == initial
    batch_attempts, maximum_batches = (6, 1) if pilot else (BATCH_ATTEMPTS, MAX_BATCHES)
    plan = {"status": "frozen_before_continuation_batches", "protocol_sha256": PROTOCOL_SHA,
            "mode": prefix, "source_hashes": source_hashes,
            "dependencies": {p: importlib.metadata.version(p) for p in ("openai", "httpx", "httpcore")},
            "first_applicable_attempt_id": initial["attempts"] + 1,
            "wrapper_sha256": collection.common.sha256(__file__), "frozen_collector_identity": collection.code_identity(),
            "budget_sha256": budget["sha256"], "hard_budget_usd": 15.0,
            "max_attempts_per_batch": batch_attempts, "maximum_batches": maximum_batches,
            "clients": "One client per original worker; preconnect before first generation; close on failure or batch end",
            "preconnect": {"maximum_metadata_calls": transport.METADATA_ATTEMPTS, "metadata_timeout": transport.METADATA_TIMEOUT,
                "endpoint": "GET /v1/models/gpt-4.1-mini-2025-04-14", "only_retryable_errors_retried": True,
                "generation_timeout": 60, "sdk_retries": 0, "generation_retries_unchanged": True,
                "failed_preconnect_keeps_full_reserve_and_attempt": True, "tls_verification": "default enabled",
                "proxy_changes": "none", "pilot_required": "6 successes from exactly 6 ledger attempts"},
            "failure_stop": "Stop if more than 10 percent of attempts in a completed batch lack successful outcomes",
            "resource_changes": "Bounded metadata GETs added; generation budget and caps unchanged", "scientific_protocol_changes": "none", "initial": initial,
            "partial_effects": "never computed", "started_at_unix": time.time(), "process_id": os.getpid()}
    plan_path = RECORDS / (prefix + "_plan.json")
    write_new(plan_path, plan)
    amendment_sha256 = collection.common.sha256(plan_path)
    status = {"status": prefix + "_running", "process_id": os.getpid(),
              "transport_amendment_sha256": amendment_sha256,
              "started_at_unix": plan["started_at_unix"], "batch": 0, "ledger": initial,
              "required_successes": 31254, "partial_policy_effects_computed": False}
    update_progress(status)
    print(json.dumps(status), flush=True)
    terminal = "maximum_batches_reached"
    for index in range(1, maximum_batches+1):
        before = state()
        manifest = collection.engine.collect(rows, contract, binding, budget, DIRECTORY,
            transport.provider_factory(collection.engine, contract, amendment_sha256), max_new_attempts=batch_attempts)
        after = state()
        terminal = stop_rule(before, after, manifest["stop_reason"])
        if pilot:
            passed = (after["attempts"] - before["attempts"] == 6 and
                      after["successes"] - before["successes"] == 6 and
                      manifest["stop_reason"] == "invocation_attempt_limit")
            terminal = "pilot_passed_6_of_6" if passed else "pilot_failed_reliability_gate"
        receipt = {"batch": index, "before": before, "after": after, "collector_stop_reason": manifest["stop_reason"],
                   "continuation_stop_reason": terminal, "finished_at_unix": time.time(), "partial_policy_effects_computed": False}
        write_new(RECORDS / f"{prefix}_batch_{index:03d}.json", receipt)
        status.update(batch=index, ledger=after, last_batch_finished_at_unix=receipt["finished_at_unix"])
        if terminal:
            status["status"] = "complete_pending_full_ledger_audit" if terminal == "complete" else "recovered_collection_stopped_for_inspection"
            status["stop_reason"] = terminal
        update_progress(status)
        print(json.dumps({"status": status["status"], **receipt}), flush=True)
        if terminal:
            break
    else:
        terminal = "maximum_batches_reached"
        status["status"] = "recovered_collection_stopped_at_batch_limit"
    status.update(stop_reason=terminal, finished_at_unix=time.time(), process_id=None)
    write_new(RECORDS / (prefix + "_terminal.json"), status)
    update_progress(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        base = {"attempts": 100, "successes": 90}
        assert stop_rule(base, {"attempts": 612, "successes": 602}, "invocation_attempt_limit") is None
        assert stop_rule(base, {"attempts": 612, "successes": 550}, "invocation_attempt_limit") == "transport_failure_fraction_above_10_percent"
        assert stop_rule(base, {"attempts": 612, "successes": 602}, "budget_reservation_limit") == "collector_stopped_budget_reservation_limit"
        assert stop_rule(base, {"attempts": 100, "successes": 90}, "invocation_attempt_limit") == "invalid_batch_accounting"
        assert stop_rule(base, {"attempts": 40000, "successes": 31254}, "complete") == "complete"
        print(json.dumps({"status": "passed_continuation_stop_checks", "network_calls": 0}))
    else:
        run(pilot=args.pilot)
