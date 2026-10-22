"""Explicit user resume from preserved paused checkpoints, with attempt counts intact."""
from contextlib import closing
from pathlib import Path
import json
import shutil
import sqlite3

from . import run as r
from . import recover as previous

SOURCE = r.OUT / "approved_recovery001"
TARGET = r.OUT / "approved_resume002_20261022"


class ResumeAsk(previous.AutomaticAsk):
    def __call__(self, *args, **kwargs):
        if (r.OUT / "STOP_REQUESTED").exists() or (TARGET / "STOP_REQUESTED").exists():
            r.engine.STOP.set()
            raise InterruptedError("Explicit user pause marker; no new submission")
        return super().__call__(*args, **kwargs)


def prepare():
    r.check()
    assert not TARGET.exists(), "Do not reset attempts or overwrite a resumed run"
    assert r.read(r.OUT / "EXTERNAL_PAYLOAD_APPROVED.json")["usage_charges_approved"]
    previous_authority = r.read(SOURCE / "RECOVERY_AUTHORITY.json")
    assert previous_authority["maximum_total_actual_attempts_per_complete_payload"] == 5
    paused = r.read(SOURCE / "PAUSED_BY_USER.json")
    assert paused["status"] == "paused_by_user"
    TARGET.mkdir()
    authority = {"utc": r.utc(), "authority": "User: 恢复评估", "scope": r.HERE.name,
        "source": str(SOURCE.relative_to(r.ROOT)), "target": str(TARGET.relative_to(r.ROOT)),
        "prior_payload_and_fee_approval_retained": True,
        "maximum_total_actual_attempts_per_complete_payload": 5,
        "prior_attempts_including_pause_interrupted_submissions_counted": True,
        "original_checkpoints_and_billing_reserves_preserved": True,
        "prompt_sha256": r.sha(r.HERE / "ordinary_child_prompt.frozen.txt"),
        "runner_sha256": r.sha(Path(__file__)), "main_evaluation_remains_paused": True}
    unresolved = []
    for name in ["control90.jsonl", "OFFLINE_CHECK.json", "EXTERNAL_PAYLOAD_APPROVED.json"]:
        shutil.copyfile(SOURCE / name, TARGET / name)
    for ds in r.engine.DATASETS:
        old = SOURCE / "candidate" / ds
        new = TARGET / "candidate" / ds
        new.mkdir(parents=True)
        with closing(sqlite3.connect((old / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as source, closing(sqlite3.connect(new / "checkpoint.sqlite3")) as copied:
            source.backup(copied)
            for rid, stage, payload, status, raw in copied.execute("SELECT id,stage,payload,status,result FROM requests WHERE status!='success'").fetchall():
                result = json.loads(raw) if raw else {}
                assert status in {"in_flight", "unknown", "failed"}
                attempts = 1 + copied.execute("SELECT count(*) FROM retired_requests WHERE payload=?", (payload,)).fetchone()[0]
                if attempts >= 5:
                    continue
                if status == "in_flight":
                    result = {"status": "unknown", "failure_class": "InterruptedByUserPause",
                        "reconciliation": "No provider completion response or response ID persisted; preserve original billing reserve."}
                    copied.execute("UPDATE requests SET status='unknown',result=? WHERE id=?", (json.dumps(result), rid))
                    copied.commit()
                r.retire(copied, rid, "User explicitly resumed the same evaluation; prior maximum5 technical-recovery authorization retained")
                unresolved.append({"dataset": ds, "request_id": rid, "stage": stage,
                    "previous_state": status, "attempts_before_resume": attempts,
                    "billing_reserve_preserved": True})
        if (old / "trace.jsonl").exists():
            shutil.copyfile(old / "trace.jsonl", new / "trace.jsonl")
    authority["reconciled_unresolved_requests"] = unresolved
    r.save(TARGET / "RESUME_AUTHORITY.json", authority)
    r.save(r.OUT / "RESUMED_BY_USER.json", authority)
    marker = r.OUT / "STOP_REQUESTED"
    if marker.exists():
        marker.rename(r.OUT / "STOP_REQUESTED.paused_20261021")
    with (r.PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"id": "P02_CHILDPROMPT_MINIMAL4_QUICK90_RESUME002_APPROVED", **authority}, ensure_ascii=False) + "\n")
    return authority


def main():
    authority = prepare()
    r.emit({"status": "resuming", "unresolved_requests": len(authority["reconciled_unresolved_requests"]),
        "attempts_preserved": True, "main_evaluation_remains_paused": True})
    r.configure()
    r.engine.OUT = TARGET
    r.engine.ScopedAsk = ResumeAsk
    r.engine.rows = previous.filtered_rows
    while True:
        r.engine.STOP.clear()
        try:
            result = r.engine.run()
            break
        except RuntimeError:
            if (r.OUT / "STOP_REQUESTED").exists() or (TARGET / "STOP_REQUESTED").exists():
                r.save(TARGET / "PAUSED_BY_USER.json", {"utc": r.utc(), "status": "paused_by_user", "receipts_preserved": True})
                return
            failures = {r.read(p)["id"] for p in TARGET.glob("candidate/*/failure_*.json")}
            new_held = failures - previous.HELD
            if not new_held:
                raise
            previous.HELD.update(new_held)
            r.save(TARGET / "HELD_AFTER_RECOVERY_LIMIT.json", {"utc": r.utc(), "ids": sorted(previous.HELD), "full_denominator": 90})
    r.emit(result)


if __name__ == "__main__":
    main()
