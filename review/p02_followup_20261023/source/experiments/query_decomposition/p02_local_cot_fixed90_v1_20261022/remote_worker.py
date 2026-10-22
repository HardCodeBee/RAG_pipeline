"""Single allocated GPU, retrieval only; immutable requests/results via files.

This file and its two pure contract modules are copied to a NEW job directory.
Old evaluation resources are read only. No LLM credentials or messages are used.
"""
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


class MemoryStore:
    def __init__(self):
        self.data = {}

    def get(self, kind, key):
        return self.data.get((kind, key))

    def put(self, kind, key, value):
        self.data.setdefault((kind, key), value)

    def event(self, *args, **kwargs):
        pass

    def export(self):
        pass


def main(directory):
    import fcntl
    folder = Path(directory).resolve()
    config = read(folder / "REMOTE_JOB.frozen.json")
    assert str(folder) == config["remote_directory"]
    for filename, expected in config["uploaded_hashes"].items():
        assert Path(filename).name == filename and sha(folder / filename) == expected
    bundle = Path(config["bundle_root"]).resolve()
    # The existing query encoder/index implementations must match local pins.
    for relative, expected in config["dependency_hashes"].items():
        file = (bundle / relative).resolve()
        assert file.is_relative_to(bundle) and sha(file) == expected
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(folder))
    from resource_contract import contract, identity
    from retrieval_bridge import identity as request_identity, key, check_packet
    query_model = config["query_model"]
    assert query_model["model"] == "BAAI/bge-small-en-v1.5"
    snapshot = bundle / ".remote_cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots" / query_model["revision"]
    for relative, expected in query_model["files"].items():
        artifact = (snapshot / relative).resolve()
        # Snapshot files may be content-addressed HF symlinks into sibling blobs.
        assert artifact.is_relative_to((bundle / ".remote_cache/huggingface").resolve())
        assert sha(artifact) == expected, "Pinned BGE/tokenizer bytes differ"
    import torch
    assert torch.cuda.is_available(), "An allocated remote CUDA device is required"
    device = torch.cuda.get_device_name(0)
    assert "A6000" in device, "Unexpected allocated GPU type"
    # Use the exact frozen PortableRetrieval implementation, without its CLI.
    spec = importlib.util.spec_from_file_location("p02_frozen_portable_resources", folder / "portable_resources.py")
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    retrievers = {}
    verification = {}
    for dataset, expected in config["resource_contracts"].items():
        manifest = bundle / "resources" / dataset / "manifest.json"
        actual = contract(read(manifest), sha(manifest))
        assert actual == expected["contract"] and identity(read(manifest), sha(manifest)) == expected["identity"]
        assert actual["query_model"] == query_model["model"] and actual["query_revision"] == query_model["revision"]
        validated = read(manifest.parent / "VALIDATED.json")
        assert validated.get("fully_verified") is True
        assert validated["manifest_sha256"] == sha(manifest)
        assert set(validated["files"]) == {entry["path"] for entry in read(manifest)["files"]}
        for relative, stamp in validated["files"].items():
            file = (manifest.parent / relative).resolve()
            assert file.is_relative_to(manifest.parent.resolve())
            stat = file.stat()
            assert [stat.st_size, stat.st_mtime_ns] == stamp, "Previously verified resource changed"
        verification[dataset] = dict(identity=expected["identity"], manifest_sha256=sha(manifest),
            verified_file_stamps=len(validated["files"]), fresh_full_byte_hash_not_claimed=True)
        retrievers[dataset] = portable.PortableRetrieval(MemoryStore(), str(manifest),
            backend="bge", device="cuda", threads=4, max_batch=1)
    requests, responses = folder / "requests", folder / "responses"
    requests.mkdir(exist_ok=True)
    responses.mkdir(exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(folder / "READY.json", dict(status="ready", gpu=device, pid=os.getpid(),
            protocol_sha256=config["protocol_sha256"],
            job_id=os.environ.get("SLURM_JOB_ID"), resources=verification,
            model_load="lazy_on_first_actual_query", no_LLM_credentials_or_requests=True))
        last = time.monotonic()
        processed = 0
        try:
            while not (folder / "STOP_REQUESTED").exists():
                pending = [p for p in sorted(requests.glob("*.json")) if not (responses / p.name).exists()]
                if not pending:
                    if time.monotonic() - last > config["idle_seconds"]:
                        break
                    time.sleep(.2)
                    continue
                for path in pending:
                    request = read(path)
                    expected = request_identity(request["resource_identity"], request["dataset"],
                        request["qid"], request["query"], request["k"])
                    assert request == expected and path.stem == key(request)
                    assert request["qid"] in config["allowed_qids"][request["dataset"]]
                    assert request["resource_identity"] == config["resource_contracts"][request["dataset"]]["identity"]
                    try:
                        packet = retrievers[request["dataset"]].retrieve(request["qid"], request["query"], 10)
                        packet = check_packet(request, packet)
                        response = dict(status="success", request_id=key(request), identity=request, packet=packet)
                    except Exception as error:
                        response = dict(status="held_retrieval", request_id=key(request), identity=request,
                            error_type=type(error).__name__)
                    write(responses / path.name, response)
                    processed += 1
                    last = time.monotonic()
        finally:
            for retriever in retrievers.values():
                retriever._kernel("bge").close()
            write(folder / "FINISHED.json", dict(status="stopped", retrieval_requests_processed=processed,
                GPU=device, no_LLM_requests=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    try:
        main(args.directory)
    except Exception as error:
        path = Path(args.directory) / "WORKER_ERROR.json"
        write(path, dict(status="held", error_type=type(error).__name__))
        raise
