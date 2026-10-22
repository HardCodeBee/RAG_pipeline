"""Explicitly authorized max-five recovery; original checkpoints remain intact."""
from contextlib import closing
from pathlib import Path
import json
import os
import shutil
import sqlite3
import time

from . import run as r
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r002 import RecordingClient

TARGET = r.OUT / "approved_recovery001"
HELD = set()
ORIGINAL_ROWS = r.engine.rows


class AutomaticAsk(r.ScopedAsk):
    def __init__(self, *args, **kwargs):
        import httpx
        from openai import OpenAI
        super().__init__(*args, **kwargs)
        self.client.close()
        timeout = httpx.Timeout(self.settings["timeout_seconds"], connect=10)
        http = httpx.Client(transport=httpx.HTTPTransport(retries=2,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)),
            timeout=timeout, trust_env=False)
        api = OpenAI(api_key=os.environ[self.settings["api_key_env"]], base_url=self.settings["base_url"],
                     timeout=timeout, max_retries=0, http_client=http)
        self.client = r.SerialLLM(self.settings, client=RecordingClient(api))
        self.client.owns_client = True

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = r.canonical(r.build_payload(self.settings, prompt, maximum))
        while True:
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
                result = json.loads(row[2]) if row and row[2] else {}
                transient = row and (row[1] == "unknown" and result.get("failure_class") in {"APIConnectionError", "APITimeoutError"}
                    or row[1] == "failed" and result.get("http_status") in {408, 429, 500, 502, 503, 504})
                attempts = 1 + self.cache.db.execute("SELECT count(*) FROM retired_requests WHERE payload=?", (payload,)).fetchone()[0]
                if not transient or attempts >= 5:
                    raise
                r.retire(self.cache.db, row[0], "explicit_user_approval_automatic_transient_recovery_maximum5_total")
                self.events({"event": "approved_automatic_recovery", "stage": stage,
                    "retired_request_id": row[0], "attempts_used": attempts, "maximum": 5})
                time.sleep(min(5 * attempts, 30))


def filtered_rows(path):
    values = ORIGINAL_ROWS(path)
    if Path(path).resolve() == (r.COHORT / "runtime90.frozen.jsonl").resolve():
        return [v for v in values if v["id"] not in HELD]
    return values


def prepare():
    r.check()
    assert not TARGET.exists()
    assert r.read(r.OUT / "EXTERNAL_PAYLOAD_APPROVED.json")["usage_charges_approved"]
    TARGET.mkdir()
    authority = {"utc": r.utc(), "authority": "User explicit reply: 允许上述有限自动恢复",
        "scope": r.HERE.name, "maximum_total_actual_attempts_per_complete_payload": 5,
        "original_receipts_and_reserves_preserved": True, "successful_results_not_rerun": True,
        "algorithm_failures_not_retried": True, "provider_concurrency": 1,
        "minimum_submission_interval_seconds": 3, "SDK_POST_retries": 0,
        "transport_connect_before_send_retries": 2, "keepalive_connections": 0,
        "main_evaluation_remains_paused": True, "runner_sha256": r.sha(Path(__file__))}
    retired = []
    for name in ["control90.jsonl", "OFFLINE_CHECK.json", "EXTERNAL_PAYLOAD_APPROVED.json"]:
        shutil.copyfile(r.OUT / name, TARGET / name)
    for ds in r.engine.DATASETS:
        old = r.OUT / "candidate" / ds
        new = TARGET / "candidate" / ds
        new.mkdir(parents=True)
        with closing(sqlite3.connect((old / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as source, closing(sqlite3.connect(new / "checkpoint.sqlite3")) as copied:
            source.backup(copied)
            for rid, status, raw in copied.execute("SELECT id,status,result FROM requests WHERE status!='success'").fetchall():
                result = json.loads(raw) if raw else {}
                assert status == "unknown" and result.get("failure_class") in {"APIConnectionError", "APITimeoutError"}
                r.retire(copied, rid, authority["authority"])
                retired.append({"dataset": ds, "request_id": rid})
        if (old / "trace.jsonl").exists():
            shutil.copyfile(old / "trace.jsonl", new / "trace.jsonl")
    authority["initial_retired_requests"] = retired
    r.save(TARGET / "RECOVERY_AUTHORITY.json", authority)
    with (r.PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"id": "P02_CHILDPROMPT_MINIMAL4_QUICK90_AUTOMATIC_RECOVERY_APPROVED", **authority}, ensure_ascii=False) + "\n")


def main():
    prepare()
    r.configure()
    r.engine.OUT = TARGET
    r.engine.ScopedAsk = AutomaticAsk
    r.engine.rows = filtered_rows
    while True:
        r.engine.STOP.clear()
        try:
            result = r.engine.run()
            break
        except RuntimeError:
            failures = {r.read(p)["id"] for p in TARGET.glob("candidate/*/failure_*.json")}
            new_held = failures - HELD
            if not new_held:
                raise
            HELD.update(new_held)
            r.save(TARGET / "HELD_AFTER_RECOVERY_LIMIT.json", {"utc": r.utc(), "ids": sorted(HELD), "full_denominator": 90})
    r.emit(result)


if __name__ == "__main__":
    main()
