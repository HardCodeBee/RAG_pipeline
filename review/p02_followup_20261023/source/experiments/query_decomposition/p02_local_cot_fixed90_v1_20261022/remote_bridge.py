"""Independent local transport subclass; no predecessor module globals are patched."""
from contextlib import contextmanager
import importlib.util
from datetime import datetime, timezone
import os
import socket
import sys
import time
from .preparation import HERE, OUT, read


def bridge_class():
    from . import retrieval_transport as module

    class CotBridge(module.RemoteBridge):
        @property
        def client(self):
            return self._client

        @client.setter
        def client(self, client):
            self._client = client
            address = read(HERE / "EXECUTION_SETTINGS.json").get("verified_host_address")
            if client is None or not address:
                return
            original_connect = client.connect
            def connect_verified(hostname, *args, **kwargs):
                assert hostname == "cluster1.comp.anu.edu.au" and "sock" not in kwargs
                connection = socket.create_connection((address, kwargs.get("port", 22)), timeout=kwargs.get("timeout", 15))
                try:
                    # Paramiko still validates the original hostname under RejectPolicy.
                    return original_connect(hostname, *args, sock=connection, **kwargs)
                except BaseException:
                    connection.close()
                    raise
            client.connect = connect_verified

        def __init__(self, resources, output, authorization):
            super().__init__(resources, output, authorization)
            self.directory = module.REMOTE_PARENT + "/cot90_" + authorization["protocol_sha256"][:16]

        def start(self):
            assert "RAG_EVAL_REMOTE_PASSWORD" not in os.environ, "Use ephemeral getpass, not an environment credential"
            result = super().start()
            if self.started:
                self.await_ready()
            return result

        def await_ready(self):
            # A successful sbatch receipt may still be PENDING. Do not close a
            # healthy new allocation before the scheduler starts its worker.
            deadline = time.monotonic() + 600
            while True:
                if (self.output / "STOP_REQUESTED").exists():
                    raise RuntimeError("Local stop during worker startup")
                if any(self.exists(self.directory + "/" + name) for name in
                       ("FINISHED.json", "WORKER_ERROR.json", "STOP_REQUESTED")):
                    self.capture_receipts()
                    raise RuntimeError("Owned worker closed during startup")
                if self.exists(self.directory + "/READY.json"):
                    ready = self.remote_json("READY.json")
                    assert ready["status"] == "ready" and ready["job_id"] == self.job_id
                    assert ready["protocol_sha256"] == self.authorization["protocol_sha256"]
                    assert ready["no_LLM_credentials_or_requests"] is True and "A6000" in ready["gpu"]
                    assert {key: value["identity"] for key, value in ready["resources"].items()} == {
                        key: value["identity"] for key, value in self.resources.items()}
                    self.capture_receipts()
                    return
                state = self.command(["squeue", "-h", "-j", self.job_id, "-o", "%i|%T|%j"])
                active = [line.split("|") for line in state.splitlines() if line.strip()]
                assert len(active) == 1 and len(active[0]) == 3
                assert active[0][0] == self.job_id and active[0][2] == "p02-cot90"
                assert active[0][1] in ("PENDING", "CONFIGURING", "RUNNING")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Worker startup exceeded ten minutes; retain allocation, no resubmission")
                time.sleep(2)

        def command(self, args, timeout=45):
            settings = read(HERE / "EXECUTION_SETTINGS.json")
            args = ["--job-name=p02-cot90" if arg == "--job-name=p02-cot90" else
                    "--time=" + settings["job_time_limit"] if arg == "--time=04:00:00" else arg for arg in args]
            if args[0] == "sbatch":
                from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json
                clock = self.output / "ALLOCATION_CLOCK.json"
                assert not clock.exists(), "Keep recorded allocation; no automatic resubmission"
                atomic_json(clock, dict(submission_started_utc=datetime.now(timezone.utc).isoformat(),
                    job_time_limit_seconds=settings["job_time_limit_seconds"],
                    timing="Conservative submission time includes queueing; not measured GPU runtime"))
            return super().command(args, timeout)

        def can_start_condition(self):
            # Reusing a closed allocation must leave remaining tasks unstarted.
            # Check again before every condition to avoid cascading retrieval holds.
            try:
                if not self.started:
                    if self.sftp is not None:
                        self.capture_receipts()
                    return dict(allowed=False, reason="worker_not_running")
                terminal = [name for name in ("FINISHED.json", "WORKER_ERROR.json", "STOP_REQUESTED")
                            if self.exists(self.directory + "/" + name)]
                if terminal:
                    self.capture_receipts()
                    return dict(allowed=False, reason="worker_stop_or_error_marker", worker_markers=terminal)
                state = self.command(["squeue", "-h", "-j", self.job_id, "-o", "%i|%T|%j"])
                active = [line.split("|") for line in state.splitlines() if line.strip()]
                if len(active) != 1 or len(active[0]) != 3 or active[0][0] != self.job_id or active[0][2] != "p02-cot90" or active[0][1] not in ("RUNNING", "CONFIGURING"):
                    self.capture_receipts()
                    return dict(allowed=False, reason="owned_scheduler_allocation_not_confirmed_running")
            except Exception as error:
                return dict(allowed=False, reason="worker_lifecycle_unconfirmed", error_type=type(error).__name__)
            settings = read(HERE / "EXECUTION_SETTINGS.json"); clock = read(self.output / "ALLOCATION_CLOCK.json")
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(clock["submission_started_utc"])).total_seconds()
            remaining = settings["job_time_limit_seconds"] - elapsed
            return dict(allowed=remaining >= settings["minimum_remaining_seconds_for_new_condition"],
                        reason="ready" if remaining >= settings["minimum_remaining_seconds_for_new_condition"] else "allocation_time_margin",
                        conservative_remaining_seconds=remaining,
                        minimum_remaining_seconds=settings["minimum_remaining_seconds_for_new_condition"])

        def capture_receipts(self):
            from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json
            for remote, local in (("READY.json", "REMOTE_GPU_READY.json"), ("FINISHED.json", "REMOTE_GPU_FINISHED.json"),
                                  ("WORKER_ERROR.json", "REMOTE_WORKER_ERROR.json")):
                if self.exists(self.directory + "/" + remote):
                    atomic_json(self.output / local, self.remote_json(remote))

        def fetch(self, request):
            result = super().fetch(request)
            self.capture_receipts()
            return result

        def close(self):
            if self.sftp is not None and self.started:
                try:
                    self.upload("STOP_REQUESTED", b"cot90 local orchestrator exited\n")
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and not self.exists(self.directory + "/FINISHED.json"):
                        time.sleep(.5)
                    self.capture_receipts()
                    self.started = False
                except Exception as error:
                    self.log("stop_receipt_capture_pending", error_type=type(error).__name__)
            super().close()

    return CotBridge


@contextmanager
def open_bridge(*, resources, output, authorization):
    from p02_multigrain_timing_cachefix_v1.core.cache import run_lock
    with run_lock(OUT / "remote_deployment_lock"):
        bridge = bridge_class()(resources, output, authorization)
        try:
            yield bridge.start()
        finally:
            bridge.close()
