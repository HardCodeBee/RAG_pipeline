"""Authorized isolated retrieval job; no API key or LLM message crosses SSH."""
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import time
import uuid

from .retrieval_bridge import canonical, identity, key, check_packet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = "/home/projects/u7830488/robustness_models_retrievers_v1_20261017"
REMOTE_PARENT = "/home/projects/u7830488/p02_prompt_small_evals_20261022"
PORTABLE = ROOT / "experiments/query_decomposition/robustness_models_retrievers_v1_20261017/remote_resources.py"
DEPENDENCIES = ("src/indexes/streaming_flat_index.py", "src/embedders/text_embedder.py",
                "src/persistence/artifact_io.py", "src/provenance.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RemoteBridge:
    def __init__(self, resources, output, authorization):
        self.resources, self.output, self.authorization = resources, Path(output), authorization
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = self.sftp = None
        self.directory = REMOTE_PARENT + "/balanced20_" + authorization["protocol_sha256"][:16]
        assert PurePosixPath(self.directory).parent == PurePosixPath(REMOTE_PARENT)
        self.started = False

    def log(self, event, **data):
        with (self.output / "REMOTE_EVENTS.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **data), ensure_ascii=False) + "\n")

    def command(self, args, timeout=45):
        _, stdout, stderr = self.client.exec_command(shlex.join(args), timeout=timeout)
        value, error = stdout.read().decode(), stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if code:
            # Arguments contain only fixed paths and resource/job configuration.
            self.log("remote_command_failed", program=args[0], returncode=code, stderr=error[-1200:])
            raise RuntimeError("Remote preflight/job command failed: " + args[0])
        return value

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def remote_json(self, name):
        with self.sftp.open(self.directory + "/" + name, "rb") as source:
            return json.loads(source.read())

    def upload(self, relative, content):
        assert not relative.startswith("/") and ".." not in PurePosixPath(relative).parts
        target = self.directory + "/" + relative
        if self.exists(target):
            with self.sftp.open(target, "rb") as source:
                assert source.read() == content, "Existing remote artifact differs; never overwrite"
            return
        temporary = target + "." + uuid.uuid4().hex + ".tmp"
        with self.sftp.open(temporary, "wb") as destination:
            destination.write(content)
        self.sftp.rename(temporary, target)

    def start(self):
        auth = self.authorization
        assert auth.get("approved") is True and auth["experiment"] == HERE.name
        assert auth["protocol_sha256"] == sha(HERE / "PROTOCOL.frozen.json")
        assert any(json.loads(line).get("id") == auth["decision_id"] for line in
            (HERE.parent / "unified_rag_eval_v1_20261021/DECISIONS.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip())
        # Imports/connect happen only after the independent full8 authorization.
        import paramiko
        password = os.environ.pop("RAG_EVAL_REMOTE_PASSWORD", None) or getpass.getpass("Remote password: ")
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            self.client.connect("cluster1.comp.anu.edu.au", username="u7830488", password=password,
                look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=20, banner_timeout=15)
        finally:
            password = None
        self.client.get_transport().set_keepalive(25)
        self.sftp = self.client.open_sftp()
        self.sftp.get_channel().settimeout(45)
        for path in (REMOTE_PARENT, self.directory, self.directory + "/requests", self.directory + "/responses"):
            if not self.exists(path):
                self.sftp.mkdir(path)
        files = {"remote_worker.py": HERE / "remote_worker.py", "resource_contract.py": HERE / "resource_contract.py",
            "retrieval_bridge.py": HERE / "retrieval_bridge.py", "portable_resources.py": PORTABLE}
        inputs = [json.loads(line) for line in (HERE / "INPUTS20.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        config = dict(remote_directory=self.directory, bundle_root=BUNDLE,
            protocol_sha256=auth["protocol_sha256"],
            query_model=read(HERE / "QUERY_MODEL.frozen.json"),
            resource_contracts=self.resources, uploaded_hashes={name: sha(path) for name, path in files.items()},
            dependency_hashes={name: sha(ROOT / name) for name in DEPENDENCIES},
            allowed_qids={ds: [item["qid"] for item in inputs if item["dataset"] == ds] for ds in self.resources},
            idle_seconds=1800)
        for name, source in files.items():
            self.upload(name, source.read_bytes())
        self.upload("REMOTE_JOB.frozen.json", (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode())
        job_file = self.output / "REMOTE_JOB_ID.json"
        if job_file.exists():
            old = read(job_file)
            assert old["directory"] == self.directory
            self.job_id = old["job_id"]
            self.log("reuse_recorded_job_without_resubmission", job_id=self.job_id)
            self.started = not (self.exists(self.directory + "/FINISHED.json") or self.exists(self.directory + "/STOP_REQUESTED"))
            # Retrieval results remain reusable after a finished job. A new miss
            # is held explicitly; this helper never resubmits a GPU job silently.
            return self
        submission = self.output / "REMOTE_SUBMISSION_ATTEMPT.json"
        if submission.exists():
            raise RuntimeError("Unreconciled remote submission; inspect scheduler before any new GPU job")
        # Recheck the small source/manifest pins before allocating a GPU.
        for relative, expected in config["dependency_hashes"].items():
            with self.sftp.open(BUNDLE + "/" + relative, "rb") as source:
                assert hashlib.sha256(source.read()).hexdigest() == expected, "Remote retrieval dependency differs"
        for dataset, record in self.resources.items():
            with self.sftp.open(BUNDLE + "/resources/" + dataset + "/manifest.json", "rb") as source:
                assert hashlib.sha256(source.read()).hexdigest() == record["contract"]["portable_manifest_sha256"]
        gpu = self.command(["sinfo", "-h", "-p", "db4ai", "-o", "%P|%a|%G"])
        assert "up" in gpu and "a6000" in gpu.lower(), "Expected remote GPU partition unavailable"
        self.log("current_partition_checked", sinfo=gpu.strip())
        python = BUNDLE + "/.remote_env/robustness_models_retrievers_v1_20261017/venv/bin/python"
        wrapped = shlex.join(["env", "HF_HOME=" + BUNDLE + "/.remote_cache/huggingface", "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
            "OPENBLAS_NUM_THREADS=4", python, self.directory + "/remote_worker.py", "--directory", self.directory])
        submission.write_text(json.dumps(dict(status="submission_unknown_until_receipt", directory=self.directory,
            protocol_sha256=auth["protocol_sha256"], utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n", encoding="utf8")
        response = self.command(["sbatch", "--parsable", "--partition=db4ai", "--account=db4ai", "--qos=db4ai",
            "--gres=gpu:a6000:1", "--cpus-per-task=4", "--mem=32G", "--time=02:00:00", "--export=NONE",
            "--job-name=p02-target-balanced20", "--chdir=" + self.directory,
            "--output=" + self.directory + "/gpu-%j.out", "--error=" + self.directory + "/gpu-%j.err", "--wrap=" + wrapped]).strip()
        assert re.fullmatch(r"\d+(;[^\s]+)?", response), "Unknown submission reply; hold without retry"
        self.job_id = response.split(";")[0]
        job_file.write_text(json.dumps(dict(job_id=self.job_id, directory=self.directory, protocol_sha256=auth["protocol_sha256"]), indent=2) + "\n", encoding="utf8")
        self.started = True
        self.log("retrieval_job_submitted", job_id=self.job_id)
        return self

    def for_dataset(self, dataset):
        if dataset not in self.resources:
            raise ValueError("Dataset outside the fixed full8 cohort")
        bridge = self

        class Retriever:
            def retrieve(self, qid, query, k=10):
                return bridge.fetch(identity(bridge.resources[dataset]["identity"], dataset, qid, query, k))

        return Retriever()

    def fetch(self, request):
        request_id = key(request)
        response_name = "responses/" + request_id + ".json"
        self.upload("requests/" + request_id + ".json", (canonical(request) + "\n").encode())
        started = time.monotonic()
        while not self.exists(self.directory + "/" + response_name):
            if self.exists(self.directory + "/WORKER_ERROR.json") or self.exists(self.directory + "/FINISHED.json"):
                raise RuntimeError("Retrieval worker stopped before this query; keep completed LLM stages")
            if time.monotonic() - started > 600:
                state = self.command(["squeue", "-h", "-j", self.job_id, "-o", "%T|%R"]).strip()
                self.log("retrieval_wait_held", request_id=request_id, job_state=state)
                raise RuntimeError("Retrieval wait exceeded ten minutes; no automatic job resubmission")
            time.sleep(.5)
        result = self.remote_json(response_name)
        assert result.get("request_id") == request_id and result.get("identity") == request
        if result.get("status") != "success":
            raise RuntimeError("Remote retrieval error; no fabricated empty packet")
        packet = check_packet(request, result["packet"])
        self.log("actual_retrieval_or_exact_worker_reuse", request_id=request_id, dataset=request["dataset"],
            qid=request["qid"], query=request["query"], doc_ids=[doc["doc_id"] for doc in packet["docs"]])
        return packet

    def close(self):
        try:
            if self.sftp is not None:
                try:
                    # This STOP belongs only to our isolated new retrieval job.
                    if self.started:
                        try:
                            self.upload("STOP_REQUESTED", b"balanced20 local orchestrator exited\n")
                        except Exception as error:
                            self.log("retrieval_stop_unconfirmed", error_type=type(error).__name__, automatic_worker_idle_limit_seconds=1800)
                finally:
                    try:
                        self.sftp.close()
                    except Exception as error:
                        self.log("sftp_close_error", error_type=type(error).__name__)
        finally:
            if self.client is not None:
                self.client.close()


@contextmanager
def open_bridge(*, resources, output, authorization):
    # The main Cache lock is acquired later by the caller. This separate lock
    # must cover deployment as well, so concurrent entrypoints cannot submit
    # two allocations before one reaches the local checkpoint.
    from p02_multigrain_timing_cachefix_v1.core.cache import run_lock
    with run_lock(Path(output) / "remote_deployment_lock"):
        bridge = RemoteBridge(resources, output, authorization)
        try:
            yield bridge.start()
        finally:
            bridge.close()
