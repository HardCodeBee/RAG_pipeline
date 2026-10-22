"""Injected engineering/metadata failures; sockets and process creation blocked."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
from unittest.mock import patch
from unittest.mock import Mock
from . import run, preparation as prep, remote_bridge
from .offline_checks import FakeServices


def blocked(*args, **kwargs):
    raise AssertionError("Offline validation attempted a network or process service")


class FakeCache:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE requests(id TEXT PRIMARY KEY,payload TEXT,status TEXT,result TEXT)")
        self.outcomes = {}

    def all(self, kind):
        assert kind == "outcome"
        return self.outcomes


def engineering_checks():
    config = prep.read(run.PACKAGE / "config.json")
    inputs = {r["id"]: r for r in prep.rows(prep.INPUTS)}
    tasks = prep.rows(prep.ORDER); cache = FakeCache()
    for task in tasks[:24]:
        case = inputs[task["id"]]; services = FakeServices(case, final_length=task["index"] % 2 == 0); stages = []
        def ask(stage, prompt, maximum, allow_final_truncation=False):
            rid, answer, finish = services.ask(stage, prompt, maximum, allow_final_truncation)
            payload = run.build_payload(config["generation"], prompt, maximum)
            usage = dict(prompt_tokens=20, completion_tokens=4, total_tokens=24)
            receipt = dict(answer=answer, finish_reason=finish, requested_model=config["generation"]["model"],
                returned_model=config["generation"]["model"], response_id="offline:"+rid, usage=usage)
            cache.db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,?)", (rid, json.dumps(payload), "success", json.dumps(receipt)))
            stages.append(dict(stage=stage, request_id=rid, payload_sha256=run.digest(payload), usage=usage))
            return rid, answer, finish
        result = run.flow.execute_arm(case, services.retrieve, ask, task["arm"], pipeline=run.Pipeline(config))
        cache.outcomes[task["task_id"]] = dict(**task, status=result["status"], trajectory=result, stages=stages)
    passed = run.engineering_gate(cache, tasks, config)
    assert passed["passed"] and passed["planned_conditions"] == 24
    key = tasks[0]["task_id"]; saved = cache.outcomes.pop(key)
    assert not run.engineering_gate(cache, tasks, config)["passed"]
    cache.outcomes[key] = saved
    rid, original = cache.db.execute("SELECT id,payload FROM requests LIMIT 1").fetchone()
    bad = json.loads(original); bad["messages"].append(dict(role="assistant", content="Wrong shared context"))
    cache.db.execute("UPDATE requests SET payload=? WHERE id=?", (json.dumps(bad), rid))
    assert not run.engineering_gate(cache, tasks, config)["passed"]
    cache.db.execute("UPDATE requests SET payload=? WHERE id=?", (original, rid))
    node = saved["trajectory"]["nodes"][0]; prior = node["raw_P02_child_prompt"]; node["raw_P02_child_prompt"] = prior + " tamper"
    assert not run.engineering_gate(cache, tasks, config)["passed"]
    node["raw_P02_child_prompt"] = prior
    # Valid algorithm failure still counts as an engineering-complete condition.
    sid = "offline-terminal"; payload = run.build_payload(config["generation"], "Terminal fixture", 128)
    receipt = dict(answer="", finish_reason="length", requested_model=config["generation"]["model"],
        returned_model=config["generation"]["model"], response_id=sid, usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    cache.db.execute("INSERT INTO requests VALUES(?,?,?,?)", (sid, json.dumps(payload), "success", json.dumps(receipt)))
    cache.outcomes[key] = dict(**tasks[0], status="algorithm_failure", terminal_stage="legacy/answer/1", terminal_reason="terminal_length",
        partial_nodes=[], stages=[dict(stage="legacy/answer/1", request_id=sid, payload_sha256=run.digest(payload))])
    assert run.engineering_gate(cache, tasks, config)["passed"]
    cache.db.close()
    return dict(**passed, missing_technical_blocks=True, multirole_wire_blocks=True,
                tampered_raw_prompt_blocks=True, valid_algorithm_terminal_kept=True)


def predecessor_checks():
    with tempfile.TemporaryDirectory(prefix="_offline_factor90_", dir=prep.HERE) as directory:
        root = Path(directory); stage = root / "stage"; stage.mkdir()
        output = root / "outputs/query_decomposition/stage"; (output / "held").mkdir(parents=True)
        for name in ("PROTOCOL.frozen.json", "INPUTS.frozen.jsonl"):
            (stage / name).write_text("{}\n", encoding="utf8")
        tasks = [dict(task_id=str(i)) for i in range(152)]
        prep.jsonl(stage / "TASK_ORDER.frozen.jsonl", tasks)
        identity = dict(protocol_sha256=prep.sha(stage / "PROTOCOL.frozen.json"), inputs_sha256=prep.sha(stage / "INPUTS.frozen.jsonl"),
                        order_sha256=prep.sha(stage / "TASK_ORDER.frozen.jsonl"), historical_seed=None)
        with closing(sqlite3.connect(output / "checkpoint.sqlite3")) as db:
            db.execute("CREATE TABLE data(kind TEXT,key TEXT,value TEXT)"); db.execute("CREATE TABLE requests(status TEXT)")
            db.execute("INSERT INTO data VALUES('meta','identity',?)", (json.dumps(identity),))
            db.executemany("INSERT INTO data VALUES('outcome',?,'{}')", ((str(i),) for i in range(151)))
            db.commit()
        prep.atomic_json(output / "held/151.json", dict(task_id="151"))
        with patch.object(run, "ROOT", root), patch.object(run, "STAGE", stage):
            assert run.stage_completion_status()["finished"]
            with closing(sqlite3.connect(output / "checkpoint.sqlite3")) as db:
                db.execute("INSERT INTO requests VALUES('in_flight')"); db.commit()
            assert not run.stage_completion_status()["finished"]
            with closing(sqlite3.connect(output / "checkpoint.sqlite3")) as db:
                db.execute("DELETE FROM requests"); db.execute("UPDATE data SET value='{}' WHERE kind='meta'"); db.commit()
            assert not run.stage_completion_status()["finished"]
            with closing(sqlite3.connect(output / "checkpoint.sqlite3")) as db:
                db.execute("UPDATE data SET value=? WHERE kind='meta'", (json.dumps(identity),)); db.commit()
            (output / "held/151.json").unlink()
            assert not run.stage_completion_status()["finished"]
    # Wait path stops before runtime services when the exact predecessor is pending.
    with patch.object(run, "check", return_value={}), patch.object(run, "predecessor_status", return_value=dict(finished=False)), \
         patch.object(run, "atomic_json"), patch.object(run, "sha", return_value="offline-protocol"), patch.object(run, "authorized", blocked):
        assert run.await_predecessors(False) is False
    return dict(all152_attempted_with_hold_releases=True, inflight_blocks=True, wrong_identity_blocks=True,
                incomplete_attempts_block=True, metadata_wait_opens_no_runtime=True)


def main():
    prep.prepare()
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), \
         patch.object(socket, "create_connection", blocked), patch.object(subprocess, "Popen", blocked):
        engineering = engineering_checks(); predecessors = predecessor_checks()
        from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022 import remote_bridge as original
        old_here = original.HERE
        with tempfile.TemporaryDirectory(prefix="_offline_bridge90_", dir=prep.HERE) as folder:
            bridge = remote_bridge.bridge_class()({}, folder, dict(protocol_sha256="a"*64))
            assert bridge.directory.endswith("/factor90_"+"a"*16)
            assert original.HERE == old_here == prep.PRIOR
            assert bridge.can_start_condition() == dict(allowed=False, reason="worker_not_running")
            prep.atomic_json(Path(folder) / "ALLOCATION_CLOCK.json", dict(submission_started_utc="2000-01-01T00:00:00+00:00"))
            bridge.started = True
            with patch.object(bridge, "exists", return_value=False):
                assert bridge.can_start_condition()["reason"] == "allocation_time_margin"
            prep.atomic_json(Path(folder) / "ALLOCATION_CLOCK.json", dict(submission_started_utc=run.utc()))
            with patch.object(bridge, "exists", return_value=False):
                assert bridge.can_start_condition()["allowed"]
            for marker in ("FINISHED.json", "WORKER_ERROR.json", "STOP_REQUESTED"):
                with patch.object(bridge, "exists", side_effect=lambda path: path.endswith("/" + marker)), \
                     patch.object(bridge, "capture_receipts") as capture:
                    value = bridge.can_start_condition()
                    assert not value["allowed"] and value["worker_markers"] == [marker]
                    capture.assert_called_once()
            with patch.object(bridge, "exists", side_effect=OSError("offline connection ended")):
                assert bridge.can_start_condition()["reason"] == "worker_lifecycle_unconfirmed"
            client = Mock(); connection = Mock()
            original_connect = client.connect
            bridge.client = client
            with patch.object(remote_bridge.socket, "create_connection", return_value=connection) as connector:
                bridge.client.connect("cluster1.comp.anu.edu.au", timeout=15)
            connector.assert_called_once_with(("150.203.213.5", 22), timeout=15)
            original_connect.assert_called_once_with("cluster1.comp.anu.edu.au", sock=connection, timeout=15)
    result = dict(passed=True, engineering=engineering, predecessors=predecessors,
        isolated_private_bridge_no_parent_global_mutation=True,
        expired_allocation_blocks_new_condition=True,
        reused_stopped_worker_and_all_terminal_markers_block_new_dispatch=True,
        unreachable_worker_does_not_cascade_held_conditions=True,
        verified_address_socket_preserves_original_host_name=True,
        order_conditions=len(prep.rows(prep.ORDER)), exact_C0_S1_B1_P1=CounterArms(),
        no_API_SSH_GPU_or_provider_clients=True, production_cache_or_ledger_created=False,
        checked_hashes={prep.rel(path): prep.sha(path) for path in (Path(__file__), prep.HERE / "run.py", prep.HERE / "preparation.py",
            prep.HERE / "remote_bridge.py", prep.HERE / "score.py", prep.HERE / "ORDER360.jsonl", prep.HERE / "PREDECESSORS.frozen.json")})
    prep.atomic_json(prep.HERE / "CONTROLLER_OFFLINE_CHECKS.json", result)
    print(json.dumps(result, ensure_ascii=True))


def CounterArms():
    from collections import Counter
    return dict(Counter(task["arm"] for task in prep.rows(prep.ORDER)))


if __name__ == "__main__": main()
