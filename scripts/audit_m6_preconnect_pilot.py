"""Audit the stopped M6 pilot and retained ledger; never calculate policy effects."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
ANSWERS = RESEARCH / "m6_beir_validation_v1" / "answers_v1"
RECORDS = ANSWERS / "recovery_20260915_v1"
OUTPUT = RECORDS / "preconnect_pilot_ledger_checks.json"
PROTOCOL = "5b4faa41dc4a3ae83400d31a796354cdafb62cb461bfc6ba1a6f594b2b7bc785"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def readonly(path):
    path = Path(path).resolve()
    suffix = "?mode=ro" if Path(str(path) + "-wal").exists() else "?mode=ro&immutable=1"
    connection = sqlite3.connect(path.as_uri() + suffix, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def main():
    assert not OUTPUT.exists(), "Preserve previous independent audit"
    sys.path.insert(0, str(RESEARCH))
    import m6_beir_answer_collection as collection
    import audit_pool_answer_ledger as original_audit
    import finish_m6_beir_validation as analysis

    rows, contract, binding = collection.load_prepared(PROTOCOL)
    budget = collection.common.load_budget(ANSWERS / "budget.json", binding, contract["generation"]["model"])
    frozen_analysis = analysis.verify_analysis(PROTOCOL)
    plan = read(RECORDS / "preconnect_pilot_plan.json")
    terminal = read(RECORDS / "preconnect_pilot_terminal.json")
    batch = read(RECORDS / "preconnect_pilot_batch_001.json")
    amendment = sha(RECORDS / "preconnect_pilot_plan.json")
    assert plan["protocol_sha256"] == PROTOCOL and plan["budget_sha256"] == budget["sha256"]
    assert plan["frozen_collector_identity"] == collection.code_identity()
    assert all(sha(PROJECT / "scripts" / name) == value for name, value in plan["source_hashes"].items())
    assert all(importlib.metadata.version(name) == value for name, value in plan["dependencies"].items())
    assert terminal["transport_amendment_sha256"] == amendment
    assert terminal["stop_reason"] == "pilot_failed_reliability_gate" and terminal["process_id"] is None
    assert not (RECORDS / "continuation_plan.json").exists(), "A full batch was started unexpectedly"

    with collection.common.exclusive_run(ANSWERS):
        con = readonly(ANSWERS / "attempts.sqlite3")
        prior = readonly(RECORDS / "before.sqlite3")
        try:
            attempts = [dict(row) for row in con.execute("SELECT * FROM attempts ORDER BY id")]
            raw_outcomes = [tuple(row) for row in con.execute("SELECT * FROM outcomes ORDER BY query_id,action,repeat_id")]
            outcomes = [json.loads(row[3]) for row in raw_outcomes]
            identity = con.execute("SELECT payload FROM identity WHERE id=1").fetchone()[0]
            assert json.loads(identity) == {**binding, "budget_sha256": budget["sha256"]}
            assert identity == prior.execute("SELECT payload FROM identity WHERE id=1").fetchone()[0]
            checked = original_audit.check_records(rows, attempts, outcomes, budget,
                contract["generation"]["model"], 31254, False)
            assert checked["successful_generations"] == 21692 and checked["attempts"] == 26089
            assert checked["charged_or_reserved_micro_usd"] == 10087821
            assert checked["attempt_status_counts"] == {"success": 21692, "failure": 4397}
            assert checked["complete"] is False and checked["policy_effects_computed"] is False

            manifest = read(ANSWERS / "outcomes_manifest.json")
            exported_path = ANSWERS / manifest["outcomes"]["path"]
            assert sha(exported_path) == manifest["outcomes"]["sha256"]
            exported = [json.loads(line) for line in exported_path.read_text(encoding="utf-8").splitlines()]
            assert exported == outcomes
            assert manifest["status"] == "incomplete" and manifest["stop_reason"] == "invocation_attempt_limit"
            assert manifest["global_attempts"] == checked["attempts"]
            for key in ("successful_generations", "required_successful_generations", "charged_or_reserved_micro_usd"):
                assert manifest[key] == checked[key]
            for key in ("actions_freeze_sha256", "execution_contract_sha256"):
                assert manifest[key] == binding[key]

            current_by_key = {row[:3]: row for row in raw_outcomes}
            old_outcomes = [tuple(row) for row in prior.execute("SELECT * FROM outcomes ORDER BY query_id,action,repeat_id")]
            assert len(old_outcomes) == 21686
            assert all(current_by_key[row[:3]] == row for row in old_outcomes)

            pilot = [row for row in attempts if row["id"] >= plan["first_applicable_attempt_id"]]
            assert [row["id"] for row in pilot] == list(range(26084, 26090))
            summaries = []
            for row in pilot:
                payload = json.loads(row["payload"])
                assert payload["transport_amendment_sha256"] == amendment
                calls = payload["metadata_calls"]
                assert len(calls) <= 3
                if row["status"] == "success":
                    assert payload["stage"] == "generation" and "generation_request_submitted" not in payload
                    assert not calls or calls[-1]["status"] == "success"
                else:
                    assert row["id"] == 26085 and payload["stage"] == "preconnect"
                    assert payload["generation_request_submitted"] is False
                    assert len(calls) == 3 and all(c["status"] == "failure" and "SSLEOFError" in c["error_chain"] for c in calls)
                    assert row["charged_micro_usd"] == row["reserve_micro_usd"] == 1179
                summaries.append({"id": row["id"], "status": row["status"], "stage": payload["stage"],
                    "metadata_calls": len(calls), "metadata_failures": sum(c["status"] == "failure" for c in calls),
                    "charged_micro_usd": row["charged_micro_usd"]})
            assert sum(r["status"] == "success" for r in pilot) == 5
            assert sum(r["charged_micro_usd"] for r in pilot) == 2468
            assert batch["before"] == plan["initial"] and batch["after"] == terminal["ledger"]
            assert terminal["ledger"] == {"attempts": 26089, "successes": 21692,
                "charged_or_reserved_micro_usd": 10087821, "statuses": {"failure": 4397, "success": 21692}}
        finally:
            con.close()
            prior.close()

        result = {**checked, "independent_review_status": "passed_stopped_pilot_full_ledger_integrity",
            "checked_at_unix": time.time(), "checker_sha256": sha(__file__),
            "original_ledger_checker_sha256": sha(RESEARCH / "audit_pool_answer_ledger.py"),
            "protocol_sha256": PROTOCOL, "budget_sha256": budget["sha256"],
            "transport_amendment_sha256": amendment, "analysis_source_hashes_verified": frozen_analysis["source_sha256"],
            "original_collector_source_hashes_verified": collection.code_identity(),
            "wrapper_source_hashes_verified": plan["source_hashes"], "dependencies_verified": plan["dependencies"],
            "before_backup_sha256": sha(RECORDS / "before.sqlite3"),
            "outcomes_manifest_sha256": sha(ANSWERS / "outcomes_manifest.json"),
            "outcomes_jsonl_sha256": sha(exported_path), "all_exported_outcomes_equal_ledger": True,
            "old_21686_outcomes_retained_exactly": True, "ledger_identity_unchanged": True,
            "pilot_attempts": summaries, "pilot_successes": 5, "pilot_failures": 1,
            "pilot_charged_micro_usd": 2468, "failed_preconnect_full_reserve_micro_usd": 1179,
            "remaining_budget_micro_usd": 4912179, "reliability_gate_passed": False,
            "continuation_batch_started": False, "new_network_calls": 0,
            "scope": "Complete existing-ledger and metric-record integrity; no partial policy effects or provider-invoice reconciliation"}
        with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: result[key] for key in ("independent_review_status", "attempts",
            "successful_generations", "charged_or_reserved_micro_usd", "remaining_budget_micro_usd",
            "pilot_successes", "pilot_failures", "reliability_gate_passed", "new_network_calls")}))
        print(json.dumps({"audit_record": str(OUTPUT), "sha256": sha(OUTPUT)}))


if __name__ == "__main__":
    main()
