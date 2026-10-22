"""Only the two direct predecessors; metadata and owned worker closure, never scores."""
from contextlib import closing
import json
import re
import sqlite3
from .preparation import ROOT, HERE, STAGE, REFERENCE, read, rows, sha, ro

TERMINAL = {"COMPLETED", "TIMEOUT", "CANCELLED", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
PARENT = "/home/projects/u7830488/p02_prompt_small_evals_20261022"


def metadata(folder, *, stage=False):
    out = ROOT / "outputs/query_decomposition" / folder.name
    ordered = rows(folder / ("TASK_ORDER.frozen.jsonl" if stage else "ORDER360.jsonl"))
    planned = {t["task_id"] for t in ordered}; count = 152 if stage else 360
    assert len(ordered) == len(planned) == count
    expected = dict(protocol_sha256=sha(folder / "PROTOCOL.frozen.json"),
        inputs_sha256=sha(folder / ("INPUTS.frozen.jsonl" if stage else "INPUTS90.jsonl")),
        order_sha256=sha(folder / ("TASK_ORDER.frozen.jsonl" if stage else "ORDER360.jsonl")))
    expected.update(dict(historical_seed=None) if stage else dict(resources=read(folder / "RESOURCE_CONTRACTS.json"), no_historical_LLM_seed=True))
    with closing(ro(out / "checkpoint.sqlite3")) as db:
        db.execute("BEGIN")
        row = db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
        assert row and json.loads(row[0]) == expected
        done = dict(db.execute("SELECT key,json_extract(value,'$.status') FROM data WHERE kind='outcome'"))
        active = db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
    held = {read(p)["task_id"] for p in (out / "held").glob("*.json")}
    assert set(done) <= planned and held <= planned and not set(done).intersection(held)
    assert set(done.values()) <= {"success", "success_truncated", "algorithm_failure"}
    return dict(finished=(set(done) | held) == planned and active == 0, planned=count,
        attempted=len(set(done) | held), in_flight=active, protocol_sha256=expected["protocol_sha256"])


def scheduler_closed(job, receipt):
    assert receipt["job_id"] == job["job_id"] and receipt["protocol_sha256"] == job["protocol_sha256"]
    assert receipt["remote_directory"] == job["directory"] and receipt["experiment"] == REFERENCE.name
    assert receipt["read_only_remote"] is True and receipt["API_requests"] == receipt["new_jobs"] == 0
    terminal = receipt["terminal"]
    assert terminal["verified"] is True and terminal["job_id"] == job["job_id"] and terminal["remote_directory"] == job["directory"]
    commands = receipt["commands"]; jid = job["job_id"]
    assert commands["squeue"]["argv"] == ["squeue", "-h", "-j", jid, "-o", "%i|%T|%j"]
    assert commands["sacct"]["argv"] == ["sacct", "-n", "-P", "-j", jid, "--format=JobIDRaw,JobName,State,ExitCode,Start,End,Timelimit,NodeList"]
    assert commands["scontrol"]["argv"] == ["scontrol", "show", "job", "-o", jid]
    assert all(c["code"] == 0 and not c["stderr"].strip() for c in commands.values())
    assert not commands["squeue"]["stdout"].strip()
    accounting = [s.split("|") for s in commands["sacct"]["stdout"].splitlines() if s.strip()]
    assert accounting and sum(row[0] == jid for row in accounting) == 1
    assert all(len(row) >= 6 and (row[0] == jid or row[0].startswith(jid + "."))
        and row[2].split()[0].rstrip("+") in TERMINAL and row[5] not in ("", "Unknown") for row in accounting)
    main = next(row for row in accounting if row[0] == jid)
    state = main[2].split()[0].rstrip("+"); assert terminal["state"] == state
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=([^\s]*)", commands["scontrol"]["stdout"]))
    assert fields.get("JobId") == jid and fields.get("JobName") == "p02-factor90"
    assert fields.get("UserId", "").split("(", 1)[0] == "u7830488" and fields.get("WorkDir") == job["directory"]
    assert fields.get("JobState") == state
    return dict(closed=True, mode="verified_scheduler_terminal", job_id=jid, state=state)


def reference_workers_closed():
    out = ROOT / "outputs/query_decomposition" / REFERENCE.name
    folders = [out] + sorted(p for p in (out / "continuations").glob("*") if p.is_dir())
    protocol = sha(REFERENCE / "PROTOCOL.frozen.json"); checked = []
    for folder in folders:
        path = folder / "REMOTE_JOB_ID.json"
        if (folder / "REMOTE_SUBMISSION_ATTEMPT.json").exists() and not path.exists():
            return dict(closed=False, reason="unreconciled_reference_allocation")
        if not path.exists(): continue
        job = read(path); jid = job["job_id"]
        assert isinstance(jid, str) and jid.isdigit() and job["protocol_sha256"] == protocol
        assert re.fullmatch(re.escape(PARENT + "/factor90_" + protocol[:16]) + r"(?:_recovery\d+)?", job["directory"])
        finished = folder / "REMOTE_GPU_FINISHED.json"; ready = folder / "REMOTE_GPU_READY.json"
        if finished.exists() and ready.exists():
            value = read(finished); start = read(ready)
            assert start["job_id"] == jid and start["protocol_sha256"] == protocol and start["status"] == "ready"
            assert start["no_LLM_credentials_or_requests"] is True and "A6000" in start["gpu"]
            assert {k: v["identity"] for k, v in start["resources"].items()} == {k: v["identity"] for k, v in read(REFERENCE / "RESOURCE_CONTRACTS.json").items()}
            assert value["status"] == "stopped" and value["no_LLM_requests"] is True
            assert type(value["retrieval_requests_processed"]) is int and value["retrieval_requests_processed"] >= 0
            checked.append(dict(closed=True, mode="owned_worker_finished", job_id=jid))
        elif (folder / "SCHEDULER_TERMINAL.json").exists():
            checked.append(scheduler_closed(job, read(folder / "SCHEDULER_TERMINAL.json")))
        else: return dict(closed=False, reason="reference_worker_still_open_or_closure_unconfirmed", job_id=jid)
    return dict(closed=bool(checked), allocations=checked)


def status():
    result = {}
    try:
        for group in read(HERE / "PREDECESSORS.prepared.json").values():
            for name, expected in group.items(): assert sha(ROOT / name) == expected
        result["ordinary_stage"] = metadata(STAGE, stage=True)
        result["reference90"] = metadata(REFERENCE)
        if not result["reference90"]["finished"]:
            return dict(finished=False, **result)
        result["reference_workers"] = reference_workers_closed()
        return dict(finished=result["ordinary_stage"]["finished"] and result["reference_workers"]["closed"], **result)
    except (OSError, sqlite3.Error, AssertionError, ValueError, TypeError, KeyError) as exc:
        return dict(finished=False, reason="predecessor_metadata_unconfirmed", error_type=type(exc).__name__, **result)
