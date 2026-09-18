"""Reconcile four interrupted M6 attempts without changing the frozen experiment.

Default: read-only metadata. --apply: locked SQLite backup, explicit local failure
disposition with unknown remote outcome, unchanged charges, and integrity audit.
No mode makes network calls. --self-check exercises rollback and preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
DIRECTORY = RESEARCH / "m6_beir_validation_v1" / "answers_v1"
RECOVERY = DIRECTORY / "recovery_20260915_v1"
TARGETS = (26068, 26069, 26070, 26071)
PROTOCOL_SHA = "5b4faa41dc4a3ae83400d31a796354cdafb62cb461bfc6ba1a6f594b2b7bc785"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metadata(connection):
    return {
        "status_counts": dict(connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")),
        "successful_outcomes": connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0],
        "attempts": connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        "charged_or_reserved_micro_usd": connection.execute("SELECT SUM(charged_micro_usd) FROM attempts").fetchone()[0],
    }


def capture_targets(connection, targets):
    connection.row_factory = sqlite3.Row
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM attempts WHERE status NOT IN ('success','failure') ORDER BY id")]
    require(tuple(row["id"] for row in pending) == tuple(targets), "Unresolved attempts differ from the reviewed set")
    for row in pending:
        require(row["status"] == "reserved", "Unexpected nonterminal state")
        require(row["charged_micro_usd"] == row["reserve_micro_usd"] > 0, "Full reservation must be retained")
        key = row["query_id"], row["action"], row["repeat_id"]
        require(connection.execute("SELECT COUNT(*) FROM outcomes WHERE query_id=? AND action=? AND repeat_id=?", key).fetchone()[0] == 0,
                "A result already exists for an interrupted attempt")
        require(connection.execute("SELECT COUNT(*) FROM attempts WHERE query_id=? AND action=? AND repeat_id=?", key).fetchone()[0] == 1,
                "Unexpected attempt history for the interrupted key")
        require(set(json.loads(row["payload"])) == {"started_at_unix", "prompt_sha256", "tiktoken_prompt_tokens", "reserved_input_tokens"},
                "Interrupted record contains response information or unexpected metadata")
    return pending


def logical_digest(connection, restore=None):
    """Hash all existing records, optionally undoing just the reviewed dispositions."""
    restore = restore or {}
    result = hashlib.sha256()
    for table, order in (("identity", "id"), ("attempts", "id"), ("outcomes", "query_id,action,repeat_id"), ("sqlite_sequence", "name")):
        result.update(table.encode())
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            value = list(row)
            if table == "attempts" and value[0] in restore:
                original = restore[value[0]]
                value[4], value[7] = original["status"], original["payload"]
            result.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            result.update(b"\n")
    return result.hexdigest()


def disposition(connection, originals, evidence_sha, before_digest):
    require(not connection.in_transaction, "Unexpected active transaction")
    connection.execute("BEGIN IMMEDIATE")
    try:
        require(capture_targets(connection, [r["id"] for r in originals]) == originals, "Records changed since backup")
        timestamp = time.time()
        for row in originals:
            payload = json.loads(row["payload"])
            payload.update(
                retryable=True,
                unknown_usage_charged_full_reserve=True,
                reconciliation_reason="interrupted_unknown_remote_outcome",
                original_status="reserved",
                reconciled_at_unix=timestamp,
                reconciliation_plan_sha256=evidence_sha,
                remote_execution_response_and_billing="unknown; no provider failure or nonexecution inferred",
                local_disposition="no result was persisted; original reservation and attempt count retained",
            )
            updated = connection.execute(
                "UPDATE attempts SET status='failure',payload=? WHERE id=? AND status='reserved' AND payload=?",
                (json.dumps(payload, ensure_ascii=False), row["id"], row["payload"]),
            )
            require(updated.rowcount == 1, "Conditional update failed")
        restored = {row["id"]: row for row in originals}
        require(logical_digest(connection, restored) == before_digest, "Unreviewed ledger fields changed")
        require(not connection.execute("SELECT 1 FROM attempts WHERE status NOT IN ('success','failure') LIMIT 1").fetchone(),
                "Unresolved records remain")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def apply():
    sys.path.insert(0, str(RESEARCH))
    import m6_beir_answer_collection as collection
    import audit_pool_answer_ledger as audit

    require(collection.DIRECTORY.resolve() == DIRECTORY.resolve(), "Collection directory differs")
    require(not RECOVERY.exists(), "Recovery record already exists; inspect it before taking another action")
    rows, contract, binding = collection.load_prepared(PROTOCOL_SHA)
    budget = collection.common.load_budget(DIRECTORY / "budget.json", binding, contract["generation"]["model"])
    require(budget["hard_budget_usd"] == 15.0, "Budget differs from the previously approved ceiling")
    with collection.common.exclusive_run(DIRECTORY):
        connection = sqlite3.connect(DIRECTORY / "attempts.sqlite3")
        try:
            before = metadata(connection)
            require(before == {"status_counts": {"failure": 4381, "reserved": 4, "success": 21686},
                               "successful_outcomes": 21686, "attempts": 26071,
                               "charged_or_reserved_micro_usd": 10072023}, "Live ledger moved; re-review the snapshot")
            require(json.loads(connection.execute("SELECT payload FROM identity WHERE id=1").fetchone()[0]) ==
                    {**binding, "budget_sha256": budget["sha256"]}, "Ledger identity differs")
            originals = capture_targets(connection, TARGETS)
            require(sum(row["charged_micro_usd"] for row in originals) == 4916, "Unresolved reservation differs")
            before_digest = logical_digest(connection)
            RECOVERY.mkdir()
            snapshot_path = RECOVERY / "before.sqlite3"
            snapshot = sqlite3.connect(snapshot_path)
            try:
                connection.backup(snapshot)
                require(logical_digest(snapshot) == before_digest, "SQLite backup differs")
            finally:
                snapshot.close()
            for name in ("outcomes.jsonl", "outcomes_manifest.json", "budget.json", "collection_protocol.json"):
                (RECOVERY / name).write_bytes((DIRECTORY / name).read_bytes())
            plan = {
                "status": "prepared_before_disposition", "prepared_at_unix": time.time(),
                "reason": "Collector no longer owns the OS lock; four local attempts have no persisted response",
                "unknown_remote_result": True, "remote_invoice_reconciled": False,
                "backup_sha256": digest(snapshot_path), "logical_digest_before": before_digest,
                "protocol_sha256": PROTOCOL_SHA, "budget_sha256": budget["sha256"],
                "recovery_script_sha256": digest(__file__), "before": before,
                "original_attempt_records": originals,
                "disposition": "Mark local attempts failure/retryable with original full charges; retry only within existing six-attempt and 15 USD limits",
                "scientific_protocol_change": False, "outcomes_changed": False,
            }
            plan_path = RECOVERY / "reconciliation_plan.json"
            save(plan_path, plan)
            disposition(connection, originals, digest(plan_path), before_digest)
            after = metadata(connection)
            require(after["charged_or_reserved_micro_usd"] == before["charged_or_reserved_micro_usd"] and
                    after["attempts"] == before["attempts"] and after["successful_outcomes"] == before["successful_outcomes"],
                    "Accounting or completed outcomes changed")
            save(RECOVERY / "reconciliation_complete.json", {
                "status": "complete_local_disposition_remote_outcome_unknown", "after": after,
                "reconciliation_plan_sha256": digest(plan_path), "changed_attempt_ids": list(TARGETS),
                "only_reviewed_status_and_payload_changed": True, "fees_released_micro_usd": 0,
                "new_network_calls": 0, "partial_policy_effects_computed": False,
            })
            manifest = collection.common.export_results(DIRECTORY, connection, binding, "reconciled_interruption_incomplete")
            attempts = [dict(row) for row in connection.execute("SELECT * FROM attempts ORDER BY id")]
            outcomes = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM outcomes ORDER BY query_id,action,repeat_id")]
            checked = audit.check_records(rows, attempts, outcomes, budget, contract["generation"]["model"], 31254, False)
            require(checked["successful_generations"] == manifest["successful_generations"] == 21686,
                    "Recovery audit coverage changed")
            checked.update(recovery_plan_sha256=digest(plan_path),
                           auditor_sha256=digest(RESEARCH / "audit_pool_answer_ledger.py"))
            save(RECOVERY / "recovery_ledger_checks.json", checked)
            print(json.dumps({"status": "reconciled_and_integrity_checked_no_network", **after,
                              "remaining_budget_micro_usd": budget["ceiling_micro_usd"]-after["charged_or_reserved_micro_usd"]}))
        finally:
            connection.close()


def self_check():
    with tempfile.TemporaryDirectory(prefix="m6_reconcile_") as temporary:
        con = sqlite3.connect(Path(temporary) / "test.sqlite3")
        try:
            con.executescript("CREATE TABLE identity(id INTEGER PRIMARY KEY,payload TEXT);"
                "CREATE TABLE attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,query_id TEXT,action TEXT,repeat_id INTEGER,status TEXT,reserve_micro_usd INTEGER,charged_micro_usd INTEGER,payload TEXT);"
                "CREATE TABLE outcomes(query_id TEXT,action TEXT,repeat_id INTEGER,payload TEXT);")
            con.execute("INSERT INTO identity VALUES(1,?)", ('{"binding":"unchanged"}',))
            con.execute("INSERT INTO attempts VALUES(1,'done','dense',0,'success',100,20,?)", ('{"answer":"preserve these exact bytes"}',))
            con.execute("INSERT INTO outcomes VALUES('done','dense',0,?)", ('{"kept":"unchanged"}',))
            for number in (2, 3, 4, 5):
                payload = json.dumps({"started_at_unix": 1, "prompt_sha256": "synthetic", "tiktoken_prompt_tokens": 10, "reserved_input_tokens": 100})
                con.execute("INSERT INTO attempts VALUES(?,?,'bm25',0,'reserved',100,100,?)", (number, str(number), payload))
            con.commit()
            targets = capture_targets(con, (2, 3, 4, 5))
            before = logical_digest(con)
            try:
                disposition(con, targets, "synthetic-evidence", "deliberately-wrong-digest")
            except ValueError:
                require(logical_digest(con) == before, "Rejected transaction did not roll back")
            else:
                raise AssertionError("Corruption check accepted wrong digest")
            disposition(con, targets, "synthetic-evidence", before)
            require(logical_digest(con, {r["id"]: r for r in targets}) == before, "Original records not preserved")
            require(metadata(con)["charged_or_reserved_micro_usd"] == 420, "Charges changed")
            try:
                disposition(con, targets, "synthetic-evidence", before)
            except ValueError:
                pass
            else:
                raise AssertionError("Repeated reconciliation accepted")
            print(json.dumps({"status": "passed", "checks": ["rollback_on_mismatch", "all_records_preserved_except_disposition", "charges_retained", "repeat_refused"], "network_calls": 0}))
        finally:
            con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.apply:
        apply()
    else:
        con = sqlite3.connect((DIRECTORY / "attempts.sqlite3").as_uri()+"?mode=ro", uri=True)
        try:
            print(json.dumps(metadata(con)))
        finally:
            con.close()
