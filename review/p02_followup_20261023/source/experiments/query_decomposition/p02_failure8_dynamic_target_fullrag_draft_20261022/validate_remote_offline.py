"""Exercise the real remote classes using only temporary files and fake services.

No real SSH client, provider, CUDA implementation, scheduler or corpus is opened.
The real fixed-eight input contains no references; only its identity fields are
used to check the production bridge's allowed-query configuration.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import io
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from . import remote_bridge as bridge, resource_contract, retrieval_bridge


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def fake_packet(request):
    docs = []
    for rank in range(1, 11):
        text = "Synthetic document; no corpus or model was consulted."
        docs.append(dict(rank=rank, doc_id=f"fake-{rank}", score=1-rank/100,
            source="Synthetic source", text=text,
            block=f"[Chunk {rank} | source=Synthetic source, pages=1-1]\n{text}"))
    return {**{k: request[k] for k in ("qid", "query", "k")}, "docs": docs}


def fake_manifest(dataset):
    files = {"chunks.jsonl": b"synthetic chunks\n", "offsets.bin": b"synthetic offsets\n"}
    manifest = dict(schema_version=1, dataset=dataset,
        corpus=dict(chunks="chunks.jsonl", chunks_sha256=sha_bytes(files["chunks.jsonl"]),
            offsets="offsets.bin", documents=10),
        dense=dict(bge=dict(query_model="BAAI/bge-small-en-v1.5", query_revision="fake-revision",
            space=dict(dimension=384, normalized=True))),
        files=[dict(path=name, sha256=sha_bytes(body), bytes=len(body)) for name, body in files.items()])
    body = encoded(manifest)
    record = dict(identity=resource_contract.identity(manifest, sha_bytes(body)),
        contract=resource_contract.contract(manifest, sha_bytes(body)))
    return manifest, body, files, record


class Writable(io.BytesIO):
    def __init__(self, server, path):
        super().__init__()
        self.server, self.path = server, path

    def close(self):
        if not self.closed:
            self.server.files[self.path] = self.getvalue()
            self.server.writes.append(self.path)
        super().close()


class FakeSFTP:
    def __init__(self, server):
        self.server, self.closed = server, False

    def get_channel(self):
        return SimpleNamespace(settimeout=lambda value: None)

    def stat(self, path):
        if path not in self.server.files and path not in self.server.directories:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_size=len(self.server.files.get(path, b"")))

    def mkdir(self, path):
        self.server.directories.add(path)
        self.server.mkdirs.append(path)

    def open(self, path, mode):
        if mode == "rb":
            if path not in self.server.files:
                raise FileNotFoundError(path)
            return io.BytesIO(self.server.files[path])
        assert mode == "wb", mode
        return Writable(self.server, path)

    def rename(self, source, target):
        assert source in self.server.files and target not in self.server.files
        self.server.files[target] = self.server.files.pop(source)
        self.server.writes.append(target)

    def close(self):
        self.closed = True


class FakeSSH:
    def __init__(self, server):
        self.server, self.closed = server, False
        self.sftp = FakeSFTP(server)
        server.clients.append(self)

    def load_system_host_keys(self):
        pass

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, host, **kwargs):
        assert host == "cluster1.comp.anu.edu.au"
        assert kwargs["username"] == "u7830488"
        assert kwargs["look_for_keys"] is False and kwargs["allow_agent"] is False
        assert kwargs["password"] == "synthetic-credential-not-real"
        self.server.connections += 1

    def get_transport(self):
        return SimpleNamespace(set_keepalive=lambda value: None)

    def open_sftp(self):
        return self.sftp

    def exec_command(self, command, timeout):
        args = shlex.split(command)
        self.server.commands.append(args)
        if args[0] == "sinfo":
            output = b"db4ai|up|gpu:a6000:1\n"
        elif args[0] == "sbatch":
            self.server.submissions += 1
            # The production sentinel must already be durable at submit time.
            state = json.loads((self.server.output / "REMOTE_SUBMISSION_ATTEMPT.json").read_text())
            assert state["status"] == "submission_unknown_until_receipt"
            if self.server.unknown_submission:
                raise TimeoutError("Synthetic missing scheduler receipt")
            output = b"7654321\n"
        elif args[0] == "squeue":
            output = b"COMPLETED|None\n"
        else:
            raise AssertionError("Unexpected remote command: " + args[0])
        stdout, stderr = io.BytesIO(output), io.BytesIO()
        stdout.channel = SimpleNamespace(recv_exit_status=lambda: 0)
        return None, stdout, stderr

    def close(self):
        self.closed = True


class Server:
    def __init__(self, output):
        self.output = output
        self.files, self.directories = {}, set()
        self.writes, self.mkdirs, self.commands, self.clients = [], [], [], []
        self.connections = self.submissions = 0
        self.unknown_submission = False


@contextmanager
def fixture():
    actual_inputs = [json.loads(line) for line in (HERE / "INPUTS8.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(actual_inputs) == 8 and len({item["id"] for item in actual_inputs}) == 8
    assert all(set(item) == {"id", "qid", "dataset", "question", "supporting_questions"} for item in actual_inputs)
    with TemporaryDirectory(prefix="p02-remote-fake-", dir=HERE) as temporary:
        root = Path(temporary)
        here = root / "experiments/query_decomposition" / HERE.name
        here.mkdir(parents=True)
        output = root / "isolated-output"
        output.mkdir()
        plan = here.parent / "unified_rag_eval_v1_20261021"
        plan.mkdir()
        (plan / "DECISIONS.jsonl").write_bytes(encoded(dict(id="fake-approved-decision")))
        (here / "PROTOCOL.frozen.json").write_bytes(encoded(dict(experiment=HERE.name, fake=True)))
        (here / "INPUTS8.jsonl").write_text("".join(json.dumps(item)+"\n" for item in actual_inputs), encoding="utf-8")
        (here / "QUERY_MODEL.frozen.json").write_bytes((HERE / "QUERY_MODEL.frozen.json").read_bytes())
        for name in ("remote_worker.py", "resource_contract.py", "retrieval_bridge.py"):
            (here / name).write_bytes((HERE / name).read_bytes())
        portable = root / "portable_resources.py"
        portable.write_bytes(b"# Synthetic remote module; never executed.\n")
        resources, server = {}, Server(output)
        for relative in bridge.DEPENDENCIES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            body = ("fake dependency " + relative).encode()
            path.write_bytes(body)
            server.files[bridge.BUNDLE + "/" + relative] = body
        for dataset in ("hotpotqa", "2wiki"):
            _, body, _, record = fake_manifest(dataset)
            server.files[bridge.BUNDLE + "/resources/" + dataset + "/manifest.json"] = body
            resources[dataset] = record
        authorization = dict(approved=True, experiment=HERE.name,
            protocol_sha256=sha_bytes((here / "PROTOCOL.frozen.json").read_bytes()), decision_id="fake-approved-decision")
        paramiko = ModuleType("paramiko")
        paramiko.SSHClient = lambda: FakeSSH(server)
        paramiko.RejectPolicy = lambda: object()
        with patch.object(bridge, "HERE", here), patch.object(bridge, "ROOT", root), \
            patch.object(bridge, "PORTABLE", portable), patch.dict(sys.modules, {"paramiko": paramiko}), \
            patch.dict(bridge.os.environ, {"RAG_EVAL_REMOTE_PASSWORD": "synthetic-credential-not-real"}):
            yield SimpleNamespace(root=root, here=here, output=output, resources=resources,
                inputs=actual_inputs, server=server, auth=authorization)


def new_bridge(f, authorization=None):
    # Each fake SSH start receives a synthetic password after production pop().
    bridge.os.environ["RAG_EVAL_REMOTE_PASSWORD"] = "synthetic-credential-not-real"
    return bridge.RemoteBridge(f.resources, f.output, authorization or f.auth)


def rejected(callback, errors=(AssertionError, RuntimeError, ValueError)):
    try:
        callback()
    except errors:
        return
    raise AssertionError("Invalid operation was accepted")


def authorization_contract():
    with fixture() as f:
        for change in (dict(approved=False), dict(protocol_sha256="wrong"), dict(decision_id="unknown")):
            b = new_bridge(f, {**f.auth, **change})
            try:
                rejected(b.start)
            finally:
                b.close()
        assert f.server.connections == f.server.submissions == 0
    return dict(approved_protocol_and_decision_before_SSH=True, real_connections=0)


def start_and_scope_contract():
    with fixture() as f:
        b = new_bridge(f)
        try:
            b.start()
            config = json.loads(f.server.files[b.directory + "/REMOTE_JOB.frozen.json"])
            expected = {ds: [item["qid"] for item in f.inputs if item["dataset"] == ds] for ds in f.resources}
            assert config["allowed_qids"] == expected
            assert sum(map(len, config["allowed_qids"].values())) == 8
            assert config["resource_contracts"] == f.resources
            uploaded = set(config["uploaded_hashes"])
            assert uploaded == {"remote_worker.py", "resource_contract.py", "retrieval_bridge.py", "portable_resources.py"}
            assert all(sha_bytes(f.server.files[b.directory+"/"+name]) == value for name,value in config["uploaded_hashes"].items())
            commands = [args for args in f.server.commands if args[0] == "sbatch"]
            assert len(commands) == f.server.submissions == 1
            args = commands[0]
            assert "--gres=gpu:a6000:1" in args and "--cpus-per-task=4" in args
            assert "--mem=32G" in args and "--time=02:00:00" in args and "--export=NONE" in args
            wrapped = args[args.index("--wrap=")+1] if "--wrap=" in args else next(arg[len("--wrap="):] for arg in args if arg.startswith("--wrap="))
            wire = shlex.split(wrapped)
            assert wire[-3:] == [b.directory + "/remote_worker.py", "--directory", b.directory]
            assert "HF_HUB_OFFLINE=1" in wire and "TRANSFORMERS_OFFLINE=1" in wire
            assert any(arg.startswith("HF_HOME="+bridge.BUNDLE) for arg in wire)
            assert not any(term in wrapped for term in ("run_gpu", "supervisor", "api_code_qwen", "synthetic-credential-not-real"))
            rejected(lambda: b.for_dataset("musique"), (ValueError,))
        finally:
            b.close()
        assert f.server.files[b.directory + "/STOP_REQUESTED"] == b"full8 local orchestrator exited\n"
        assert all(path.startswith(b.directory + "/") for path in f.server.writes)
        assert all(path == bridge.REMOTE_PARENT or path == b.directory or path.startswith(b.directory+"/") for path in f.server.mkdirs)
        assert all(c.closed and c.sftp.closed for c in f.server.clients)
        assert not any(b"synthetic-credential-not-real" in body for path,body in f.server.files.items() if path.startswith(b.directory+"/"))
    return dict(exact_allowed_8_qids=True, only_four_source_uploads_plus_configuration=True,
        new_isolated_worker_only=True, one_A6000_four_CPU_32G_two_hour_limits=True,
        no_provider_credential_or_LLM_message_uploaded=True, writes_only_new_job_directory=True,
        close_only_own_STOP_and_all_connections_closed=True)


def submission_unknown_contract():
    with fixture() as f:
        f.server.unknown_submission = True
        b = new_bridge(f)
        try:
            rejected(b.start, (TimeoutError,))
        finally:
            b.close()
        assert (f.output / "REMOTE_SUBMISSION_ATTEMPT.json").exists()
        assert not (f.output / "REMOTE_JOB_ID.json").exists()
        b2 = new_bridge(f)
        try:
            rejected(b2.start, (RuntimeError,))
        finally:
            b2.close()
        assert f.server.submissions == 1
        assert b.directory + "/STOP_REQUESTED" not in f.server.files
    return dict(submission_sentinel_durable_before_sbatch=True, unknown_submission_restart_no_resubmit=True,
        unconfirmed_job_not_silently_cancelled_or_replaced=True)


def finished_restart_and_response_contract():
    with fixture() as f:
        b = new_bridge(f)
        try:
            b.start()
        finally:
            b.close()
        f.server.files[b.directory + "/FINISHED.json"] = encoded(dict(status="stopped"))
        b2 = new_bridge(f)
        try:
            b2.start()
            assert b2.started is False and f.server.submissions == 1
            item = f.inputs[0]
            request = retrieval_bridge.identity(f.resources[item["dataset"]]["identity"], item["dataset"], item["qid"], "Synthetic current question?", 10)
            rejected(lambda: b2.fetch(request), (RuntimeError,))
            request_id = retrieval_bridge.key(request)
            response_path = b2.directory + "/responses/" + request_id + ".json"
            valid = dict(status="success", request_id=request_id, identity=request, packet=fake_packet(request))
            f.server.files[response_path] = encoded(valid)
            assert b2.fetch(request) == valid["packet"]
            for change in (dict(request_id="wrong"), dict(identity={**request, "query":"wrong"}), dict(status="held_retrieval")):
                f.server.files[response_path] = encoded({**valid, **change})
                rejected(lambda: b2.fetch(request))
            f.server.files[response_path] = encoded({**valid, "packet":{**valid["packet"], "qid":"wrong"}})
            rejected(lambda: b2.fetch(request), (ValueError,))
            rejected(lambda: b2.for_dataset(item["dataset"]).retrieve(item["qid"], "Synthetic current question?", 9), (ValueError,))
            assert f.server.submissions == 1
        finally:
            b2.close()
    return dict(recorded_finished_job_never_resubmitted=True, missing_packet_held=True,
        existing_exact_packet_reusable_after_finish=True, wrong_response_identity_status_or_packet_rejected=True,
        wrong_k_rejected_before_remote_upload=True)


def lock_and_cleanup_contract():
    with fixture() as f:
        with bridge.open_bridge(resources=f.resources, output=f.output, authorization=f.auth) as b:
            before = f.server.connections
            def second_entry():
                with bridge.open_bridge(resources=f.resources, output=f.output, authorization=f.auth):
                    raise AssertionError("Second deployment lock unexpectedly opened")
            rejected(second_entry, (RuntimeError, BlockingIOError))
            assert f.server.connections == before and f.server.submissions == 1
            def unavailable(*args, **kwargs):
                raise OSError("Synthetic SFTP outage during STOP")
            b.upload = unavailable
        assert f.server.clients[-1].closed and f.server.clients[-1].sftp.closed
        events = (f.output / "REMOTE_EVENTS.jsonl").read_text(encoding="utf-8")
        assert "retrieval_stop_unconfirmed" in events
        assert b.directory + "/STOP_REQUESTED" not in f.server.files
    return dict(deployment_lock_prevents_second_connection_and_submit=True,
        STOP_upload_failure_recorded_without_masking_context_exit=True, SSH_and_SFTP_closed_on_upload_failure=True)


def worker_contracts():
    """Run actual worker validation/dispatch with entirely fake CUDA/retrieval."""
    actual = [json.loads(line) for line in (HERE / "INPUTS8.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    module_name = "p02_fake_only_worker_under_review"
    fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args:None)
    spec = importlib.util.spec_from_file_location(module_name, HERE / "remote_worker.py")
    worker = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"fcntl": fake_fcntl}):
        spec.loader.exec_module(worker)
    checks = {}
    for fault in ("none", "wrong_resource", "wrong_qid", "wrong_request_hash", "wrong_contract", "changed_stamp", "changed_model_bytes"):
        with TemporaryDirectory(prefix="p02-worker-fake-", dir=HERE) as temporary:
            base = Path(temporary)
            directory, bundle = base / "new-job", base / "readonly-bundle"
            directory.mkdir(); bundle.mkdir()
            (directory / "requests").mkdir(); (directory / "responses").mkdir()
            resources = {}
            for dataset in ("hotpotqa", "2wiki"):
                manifest, body, files, record = fake_manifest(dataset)
                folder = bundle / "resources" / dataset
                folder.mkdir(parents=True)
                (folder / "manifest.json").write_bytes(body)
                for name, contents in files.items():
                    (folder / name).write_bytes(contents)
                stamps = {name:[(folder/name).stat().st_size, (folder/name).stat().st_mtime_ns] for name in files}
                (folder / "VALIDATED.json").write_bytes(encoded(dict(fully_verified=True, manifest_sha256=sha_bytes(body), files=stamps)))
                resources[dataset] = record
            dependency = bundle / "src/fake_dependency.py"
            dependency.parent.mkdir(); dependency.write_bytes(b"# Fake pinned dependency\n")
            portable_path = directory / "portable_resources.py"
            portable_path.write_bytes(b"# Fake loader is injected; this is never executed.\n")
            model_file = bundle / ".remote_cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/fake-revision/model.safetensors"
            model_file.parent.mkdir(parents=True)
            model_file.write_bytes(b"fake model bytes; not a loadable tensor\n")
            model = dict(model="BAAI/bge-small-en-v1.5", revision="fake-revision",
                files={"model.safetensors":sha_bytes(model_file.read_bytes())})
            config = dict(remote_directory=str(directory.resolve()), bundle_root=str(bundle.resolve()),
                protocol_sha256="fake-protocol", query_model=model,
                uploaded_hashes={"portable_resources.py":sha_bytes(portable_path.read_bytes())},
                dependency_hashes={"src/fake_dependency.py":sha_bytes(dependency.read_bytes())},
                resource_contracts=resources,
                allowed_qids={ds:[item["qid"] for item in actual if item["dataset"]==ds] for ds in resources},
                idle_seconds=1)
            item = actual[0]
            request = retrieval_bridge.identity(resources[item["dataset"]]["identity"], item["dataset"], item["qid"], "Synthetic question?", 10)
            if fault == "wrong_resource":request["resource_identity"] = "wrong"
            if fault == "wrong_qid":request["qid"] = "not-one-of-the-eight"
            request_id = retrieval_bridge.key(request)
            if fault == "wrong_request_hash":request_id = "wrong-file-name"
            (directory / "requests" / (request_id+".json")).write_bytes(encoded(request))
            if fault == "wrong_contract":config["resource_contracts"]["2wiki"]["identity"] = "wrong"
            if fault == "changed_stamp":(bundle / "resources/2wiki/chunks.jsonl").write_bytes(b"changed synthetic source\n")
            if fault == "changed_model_bytes":model_file.write_bytes(b"changed fake model bytes\n")
            (directory / "REMOTE_JOB.frozen.json").write_bytes(encoded(config))
            calls, closes = [], []
            class FakeRetrieval:
                def __init__(self, store, manifest, **kwargs):
                    assert kwargs == dict(backend="bge", device="cuda", threads=4, max_batch=1)
                def retrieve(self, qid, query, k):
                    calls.append((qid,query,k))
                    (directory / "STOP_REQUESTED").write_bytes(b"fake worker test completed\n")
                    return fake_packet(request)
                def _kernel(self, backend):
                    assert backend == "bge"
                    return SimpleNamespace(close=lambda:closes.append(1))
            class Loader:
                def create_module(self, spec):return None
                def exec_module(self, module):module.PortableRetrieval=FakeRetrieval
            portable_spec = ModuleSpec("fake_portable_only", Loader())
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True, get_device_name=lambda index:"Synthetic A6000"))
            modules = {"torch":fake_torch,"fcntl":fake_fcntl,"resource_contract":resource_contract,"retrieval_bridge":retrieval_bridge}
            with patch.dict(sys.modules, modules), patch.object(sys,"path",list(sys.path)), \
                patch.object(worker.importlib.util,"spec_from_file_location",lambda *args,**kwargs:portable_spec):
                if fault == "none":
                    worker.main(directory)
                    response = json.loads((directory / "responses" / (request_id+".json")).read_text())
                    assert response["status"]=="success" and response["identity"]==request
                    assert len(calls)==1 and len(closes)==2
                    assert (directory / "READY.json").exists() and (directory / "FINISHED.json").exists()
                else:
                    rejected(lambda:worker.main(directory), (AssertionError, ValueError))
                    assert not calls and not list((directory / "responses").glob("*.json"))
            checks[fault] = True
    return dict(actual_worker_validation_and_dispatch_exercised=True,
        fake_CUDA_and_PortableRetrieval_only=True, no_real_torch_model_import=True,
        valid_request_one_fake_search=True, wrong_resource_qid_hash_or_contract_rejected_before_search=True,
        changed_resource_stamp_rejected_before_search=True, changed_model_bytes_rejected_before_search=True, cases=checks)


def main():
    result = dict(passed=True, scope="Actual bridge/worker methods; entirely fake SSH/CUDA/scheduler/retriever and temporary files",
        authorization=authorization_contract(), deployment=start_and_scope_contract(),
        unknown_submission=submission_unknown_contract(), restart=finished_restart_and_response_contract(),
        lock_and_cleanup=lock_and_cleanup_contract(), worker=worker_contracts(),
        real_SSH_connections=0, real_GPU_jobs=0, real_provider_requests=0,
        test_code_sha256=sha_bytes(Path(__file__).read_bytes()),
        reviewed_source_sha256={name:sha_bytes((HERE/name).read_bytes()) for name in
            ("remote_bridge.py","remote_worker.py","resource_contract.py","retrieval_bridge.py")})
    path = HERE / "REMOTE_OFFLINE_CONTRACTS.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, classes=6, output=str(path.relative_to(ROOT)),
        output_sha256=sha_bytes(path.read_bytes()), source_sha256=result["reviewed_source_sha256"],
        real_SSH_connections=0, real_GPU_jobs=0, real_provider_requests=0), ensure_ascii=True))


if __name__ == "__main__":
    main()
