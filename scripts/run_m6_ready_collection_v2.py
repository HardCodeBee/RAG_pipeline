"""Fixed readiness, 6-answer pilot, 128-attempt check, then bounded collection.

No inference attempt is reserved until all four clients pass metadata readiness.
Scientific inputs and the original answer collector remain unchanged. This run
has its own immutable operational amendment; the failed v1 pilot stays failed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
ANSWERS = RESEARCH / "m6_beir_validation_v1" / "answers_v1"
RECORDS = ANSWERS / "recovery_20260915_v1"
PROTOCOL = "5b4faa41dc4a3ae83400d31a796354cdafb62cb461bfc6ba1a6f594b2b7bc785"
PLAN = RECORDS / "ready_v2_plan.json"
SCHEDULE = (6, 128) + (512,) * 30


def decision(index, before, after, reason):
    if reason == "complete" and after["successes"] == 31254:
        return "complete"
    attempts = after["attempts"] - before["attempts"]
    successes = after["successes"] - before["successes"]
    if reason != "invocation_attempt_limit":
        return "collector_stopped_" + str(reason)
    if attempts != SCHEDULE[index] or not 0 <= successes <= attempts:
        return "invalid_step_accounting"
    failures = attempts - successes
    if index == 0 and failures:
        return "pilot_failed_6_of_6_gate"
    if index == 1 and failures > 2:
        return "128_attempt_check_exceeded_two_failures"
    if index >= 2 and failures * 10 > attempts:
        return "batch_exceeded_10_percent_failures"
    return None


def write_progress(record):
    path = RESEARCH / "validation_fresh4000_e03b_e1_v1" / "progress.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    node = document["m6_candidate_validation"]["beir_dev_validation"]
    node["status"] = record["status"]
    node["transport_ready_v2"] = record
    node["answer_collection"]["status"] = record["status"]
    node["answer_collection"]["recovery_current_process_id"] = record.get("process_id")
    node["answer_collection"]["recovery_latest_ledger"] = record["ledger"]
    node["new_paid_calls"] = record["ledger"]["attempts"]
    node["new_paid_calls_semantics"] = "Reserved answer attempts; readiness metadata calls are recorded separately."
    node["new_successful_answers"] = record["ledger"]["successes"]
    node["after_collection_analysis"]["status"] = (
        "complete_evaluation_and_separate_check" if record.get("analysis_complete")
        else "guarded_by_all31254_outcomes_in_v2_controller; old waiter inactive")
    active = record.get("process_id") is not None
    document["next"] = (
        "Observe the current v2 controller process; do not launch duplicate collection. "
        "Fixed sequence: four-client readiness, 6/6 pilot, 128 attempts with at most two failures, "
        "512-attempt batches with at most10% failures. On all31254 outcomes, perform frozen audit/evaluation/separate check."
        if active else
        "Inspect the terminal v2 receipt and authoritative ledger. Preserve the original $15 budget and frozen scientific protocol. "
        "Do not restart a failed pilot or relax its gate. No partial effects before all31254 outcomes.")
    if record.get("analysis_complete"):
        document["next"] = "Review the complete frozen BEIR evaluation and separate numerical check; decide the next research step from both fixed-baseline comparisons."
    document["last_goal_continuation_audit"] = {
        "classification": "progress", "observed_at_unix": time.time(), "goal_achieved": False,
        "evidence": "Explicit v2 transport amendment tests readiness before answer reservations; the previous failed pilot remains unchanged."}
    temporary = path.with_suffix(".ready_v2.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run():
    if not os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip():
        raise ValueError("Existing credential and standard official endpoint required")
    if PLAN.exists():
        raise ValueError("This immutable v2 run already started; inspect its terminal or live process")
    sys.path.insert(0, str(RESEARCH))
    import m6_beir_answer_collection as collection
    import finish_m6_beir_validation as finish
    import continue_m6_beir_collection as previous
    import m6_preconnect_transport as original_transport
    import m6_ready_clients_v2 as readiness

    print(json.dumps({"status": "verifying_frozen_inputs_before_v2_network"}), flush=True)
    frozen_analysis = finish.verify_analysis(PROTOCOL)
    rows, contract, binding = collection.load_prepared(PROTOCOL)
    budget = collection.common.load_budget(ANSWERS / "budget.json", binding, contract["generation"]["model"])
    assert budget["hard_budget_usd"] == 15 and budget["maximum_workers"] == 4
    assert budget["maximum_attempts_per_outcome"] == 6
    assert contract["generation"]["timeout_seconds"] == 60 and contract["generation"]["max_retries"] == 0
    assert (readiness.WORKERS, readiness.PREPARATION_ATTEMPTS, readiness.METADATA_TIMEOUT,
            readiness.KEEPALIVE_EXPIRY, readiness.MAX_PREPARED_AGE, readiness.MAX_FIRST_USE_AGE) == (4, 8, 12, 120, 30, 60)
    initial = previous.state()
    assert initial == {"attempts": 26089, "successes": 21692, "charged_or_reserved_micro_usd": 10087821,
                       "statuses": {"failure": 4397, "success": 21692}}
    old_check = json.loads((RECORDS / "preconnect_pilot_ledger_checks.json").read_text(encoding="utf-8"))
    assert old_check["independent_review_status"] == "passed_stopped_pilot_full_ledger_integrity"
    assert old_check["reliability_gate_passed"] is False and old_check["continuation_batch_started"] is False
    assert old_check["outcomes_jsonl_sha256"] == collection.common.sha256(ANSWERS / "outcomes.jsonl")
    assert old_check["outcomes_manifest_sha256"] == collection.common.sha256(ANSWERS / "outcomes_manifest.json")
    sources = {Path(p).name: collection.common.sha256(p) for p in
               (__file__, readiness.__file__, original_transport.__file__, previous.__file__)}
    plan = {"status": "frozen_operational_readiness_v2_before_network", "protocol_sha256": PROTOCOL,
        "source_hashes": sources, "frozen_collector_identity": collection.code_identity(),
        "frozen_analysis_hashes": frozen_analysis["source_sha256"], "budget_sha256": budget["sha256"],
        "dependencies": {p: importlib.metadata.version(p) for p in ("openai", "httpx", "httpcore")},
        "initial": initial, "first_applicable_answer_attempt_id": initial["attempts"] + 1,
        "rationale": "Previous pilot had five successful generations and one preconnect-only failure. Establish four clients before reserving answers; retain failed v1 gate and all charges.",
        "schedule_maximum_attempts": list(SCHEDULE), "scientific_protocol_changes": "none",
        "readiness": {"clients": 4, "metadata_attempts_per_client": 8, "metadata_operation_timeout_seconds": 12,
            "endpoint": "GET /v1/models/gpt-4.1-mini-2025-04-14", "sdk_retries": 0,
            "only_retryable_errors_retried": True, "keepalive_expiry_seconds": 120,
            "max_connections": 1000, "max_keepalive_connections": 100,
            "maximum_ready_age_at_admission_seconds": 30, "first_use_age_guard_seconds": 60,
            "timeout_is_total_wall_clock_bound": False, "failed_readiness_new_answer_reservations": 0,
            "on_any_readiness_failure": "close all clients and terminate; no same-run readiness retry",
            "tls_verification": "default enabled", "proxy_changes": "none"},
        "answer_collection": {"original_engine_unchanged": True, "maximum_workers": 4,
            "maximum_attempts_per_outcome": 6, "hard_budget_usd": 15, "generation_timeout_seconds": 60,
            "after_generation_failure": "Original v1 bounded preconnection on the next ledger attempt; retain original charges and caps"},
        "release_gates": ["6 of exactly6 first answer attempts succeed",
            "At most2 failures among the next128 attempts", "At most10 percent failures in every subsequent completed batch"],
        "all_failure_records_retained": True, "partial_policy_effects": "never inspected",
        "complete_analysis": "Original audit-complete then evaluate then check-evaluation, only after31254 successes",
        "started_at_unix": time.time(), "process_id": os.getpid()}
    previous.write_new(PLAN, plan)
    amendment = collection.common.sha256(PLAN)
    record = {"status": "ready_v2_running", "process_id": os.getpid(), "ledger": initial,
        "transport_amendment_sha256": amendment, "started_at_unix": plan["started_at_unix"],
        "step": 0, "analysis_complete": False, "partial_policy_effects_computed": False}
    write_progress(record)
    print(json.dumps(record), flush=True)
    terminal = "schedule_exhausted"
    try:
        for index, maximum in enumerate(SCHEDULE):
            record["step"] = index + 1
            before = previous.state()
            readiness_path = RECORDS / f"ready_v2_step_{index+1:03d}_connections.json"
            clients, proof = readiness.prepare_clients(collection.engine, contract, readiness_path, amendment)
            try:
                after_readiness = previous.state()
                assert after_readiness == before, "Readiness changed the generation ledger"
                if not clients:
                    terminal = "readiness_failed_before_answer_reservation"
                    previous.write_new(RECORDS / f"ready_v2_step_{index+1:03d}.json", {
                        "step": index+1, "before": before, "after": after_readiness,
                        "readiness_sha256": collection.common.sha256(readiness_path), "stop_reason": terminal})
                    break
                factory = readiness.make_provider_factory(collection.engine, contract, clients, amendment,
                    collection.common.sha256(readiness_path))
                manifest = collection.engine.collect(rows, contract, binding, budget, ANSWERS,
                    factory, max_new_attempts=maximum)
            finally:
                readiness.close_clients(clients)
            after = previous.state()
            terminal = decision(index, before, after, manifest["stop_reason"])
            receipt = {"step": index+1, "before": before, "after": after,
                "readiness_sha256": collection.common.sha256(readiness_path),
                "collector_stop_reason": manifest["stop_reason"], "stop_reason": terminal,
                "finished_at_unix": time.time(), "partial_policy_effects_computed": False}
            previous.write_new(RECORDS / f"ready_v2_step_{index+1:03d}.json", receipt)
            record.update(ledger=after, last_step=receipt)
            write_progress(record)
            print(json.dumps(receipt), flush=True)
            if terminal:
                break
        else:
            terminal = "schedule_exhausted"
        if terminal == "complete":
            record["status"] = "ready_v2_complete_running_frozen_analysis"
            write_progress(record)
            finish.audit(PROTOCOL, False)
            finish.evaluate(PROTOCOL)
            finish.check_evaluation(PROTOCOL)
            record["analysis_complete"] = True
            terminal = "complete_evaluated_and_separately_checked"
    except KeyboardInterrupt:
        terminal = "operator_interrupt_requires_ledger_inspection"
        print(json.dumps({"status": terminal}), flush=True)
        raise
    except Exception as error:
        terminal = "execution_exception_requires_inspection"
        record["safe_exception_class"] = type(error).__name__
        print(json.dumps({"status": terminal, "safe_exception_class": type(error).__name__}), flush=True)
        raise
    finally:
        record.update(status="ready_v2_terminal", stop_reason=terminal, process_id=None,
            ledger=previous.state(), finished_at_unix=time.time())
        previous.write_new(RECORDS / "ready_v2_terminal.json", record)
        write_progress(record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        base = {"attempts": 100, "successes": 90}
        assert decision(0, base, {"attempts": 106, "successes": 96}, "invocation_attempt_limit") is None
        assert decision(0, base, {"attempts": 106, "successes": 95}, "invocation_attempt_limit") == "pilot_failed_6_of_6_gate"
        assert decision(1, base, {"attempts": 228, "successes": 216}, "invocation_attempt_limit") is None
        assert decision(1, base, {"attempts": 228, "successes": 215}, "invocation_attempt_limit") == "128_attempt_check_exceeded_two_failures"
        assert decision(2, base, {"attempts": 612, "successes": 550}, "invocation_attempt_limit") == "batch_exceeded_10_percent_failures"
        assert decision(2, base, {"attempts": 102, "successes": 92}, "budget_reservation_limit") == "collector_stopped_budget_reservation_limit"
        assert decision(3, base, {"attempts": 110, "successes": 31254}, "complete") == "complete"
        print(json.dumps({"status": "passed_v2_fixed_gate_checks", "network_calls": 0}))
    else:
        run()
