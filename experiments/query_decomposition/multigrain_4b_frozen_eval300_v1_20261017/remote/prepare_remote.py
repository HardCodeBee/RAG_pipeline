"""Verify and stage the read-only base-model sweep; submit once or export results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "code"))
from contracts import (ARMS, BASE_MANIFEST, REPO, REVISION, queries, read_cases,
                       read_json, save_json, sha, summary, validate_config, verify_package)


def once(path, value):
    if path.exists():
        assert read_json(path) == value, "Existing staged configuration changed: " + str(path)
    else:
        save_json(path, value)


def stage(args):
    verify_package(PACKAGE)
    config = read_json(PACKAGE / "base_runtime.json")
    route = args.route_root.resolve() if args.route_root else Path(config.pop("default_route_root"))
    config.pop("default_route_root", None)
    python = args.python.resolve() if args.python else route / "environment/venv/bin/python"
    model = args.model_path.resolve() if args.model_path else route / "models/qwen3_4b_instruct_2507"
    out = PACKAGE / "results"
    out.mkdir(exist_ok=True)
    expected = read_json(PACKAGE / "BASE_MODEL_IDENTITY.expected.json")
    assert expected["repo_id"] == REPO and expected["revision"] == REVISION
    assert expected["manifest_sha256"] == BASE_MANIFEST
    assert not (model / "adapter_config.json").exists(), "Model folder contains an adapter"
    for r in expected["files"]:
        p = model / r["path"]
        assert p.is_file() and p.stat().st_size == r["bytes"], "Existing base file missing/size changed: " + str(p)
    assert python.is_file(), "Existing environment python missing: " + str(python)
    # Only deployment path changes. Full content hashes are checked in the allocation.
    expected["local_path"] = str(model)
    once(out / "BASE_MODEL_IDENTITY.runtime.json", expected)
    config.update(model_path=str(model), base_identity_file=str(out / "BASE_MODEL_IDENTITY.runtime.json"), output_dir=str(out))
    validate_config(config)
    queries(PACKAGE / "queries300.query_only.jsonl")
    once(out / "runtime_config.json", config)
    return route, python, out, config


def submit(args, python, out):
    import fcntl
    with (out / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = out / "JOB_SUBMISSION.json"
        pending = out / "SUBMISSION_PENDING.json"
        history = read_json(receipt) if receipt.exists() else {"jobs": []}
        if pending.exists():
            raise RuntimeError("An unconfirmed sbatch submission exists; reconcile scheduler before submitting again")
        if history["jobs"]:
            last = history["jobs"][-1]["job_id"]
            if not args.resume:
                print(json.dumps(history, indent=2)); return
            active = subprocess.run(["squeue", "--noheader", "--jobs", last, "--format=%i %T"], check=True, capture_output=True, text=True)
            assert not active.stdout.strip(), "Previous job remains active"
            completed = subprocess.run(["sacct", "--noheader", "--parsable2", "--jobs", last, "--format=JobIDRaw,State"], check=True, capture_output=True, text=True)
            states = [s.split("|")[1].split()[0].rstrip("+") for s in completed.stdout.splitlines() if s.split("|")[0] == last]
            assert states and states[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}, "Scheduler has no verified terminal state"
        args_list = ["sbatch", "--parsable", "--job-name=base4b-eval300", "--chdir="+str(PACKAGE),
                     "--output="+str(out / "slurm-%j.out"), "--error="+str(out / "slurm-%j.err"),
                     str(PACKAGE / "runtime/run_stage.slurm"), str(python), str(PACKAGE / "code/run_remote.py"),
                     "--config", str(out / "runtime_config.json")]
        save_json(pending, {"command": args_list, "created_utc": datetime.now(timezone.utc).isoformat()})
        result = subprocess.run(args_list, capture_output=True, text=True)
        if result.returncode:
            save_json(out / "SUBMISSION_REJECTED.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            pending.rename(out / ("SUBMISSION_REJECTED_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json"))
            raise RuntimeError("sbatch rejected; see retained SUBMISSION_REJECTED.json")
        job_id = result.stdout.strip().split(";")[0]
        assert job_id.isdigit(), "Unknown submission outcome: preserve pending record and reconcile scheduler"
        history["jobs"].append({"job_id": job_id, "command": args_list, "utc": datetime.now(timezone.utc).isoformat()})
        save_json(receipt, history)
        pending.rename(out / ("SUBMISSION_CONFIRMED_"+job_id+".json"))
        print(json.dumps(history, indent=2))


def export_return():
    verify_package(PACKAGE)
    out = PACKAGE / "results"
    rows = queries(PACKAGE / "queries300.query_only.jsonl")
    # Lock out active GPU writing while creating a consistent snapshot.
    import fcntl
    out.mkdir(exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        values = read_cases(out, rows)
        save_json(out / "SUMMARY.json", summary(values))
        (out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in values), encoding="utf-8")
        paths = {str(p.relative_to(out)):p for p in out.rglob("*") if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".tmp")}
        for name in ("queries300.query_only.jsonl", "PROTOCOL_FROZEN.json", "PACKAGE_MANIFEST.json", "base_runtime.json"):
            paths["package/"+name] = PACKAGE / name
        for p in (PACKAGE / "prompts").glob("*.txt"):
            paths["package/prompts/"+p.name] = p
        manifest = {"files": {n:sha(p) for n,p in sorted(paths.items())}, "completed_attempts": len(values), "expected": 300,
                    "adapter_loaded": False, "api_calls": 0, "partial_return_allowed": True}
        archive = PACKAGE / "base4b_eval300_return.zip"
        temporary = archive.with_suffix(".zip.tmp")
        with ZipFile(temporary, "w", ZIP_DEFLATED) as z:
            for name,path in sorted(paths.items()):
                z.write(path, name)
            z.writestr("RETURN_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
        temporary.replace(archive)
    print(json.dumps({"archive": str(archive), "completed_attempts": len(values), "expected": 300}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check-package", action="store_true")
    p.add_argument("--route-root", type=Path)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--python", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-return", action="store_true")
    args = p.parse_args()
    if args.check_package:
        verify_package(PACKAGE)
        queries(PACKAGE / "queries300.query_only.jsonl")
        validate_config({k:v for k,v in read_json(PACKAGE / "base_runtime.json").items() if k != "default_route_root"})
        print("Package contract passed: original4B only,300 queries,1 frozen prompt,300 attempts,no API.")
    elif args.export_return:
        export_return()
    else:
        _, python, out, config = stage(args)
        if args.submit:
            submit(args, python, out)
        else:
            print(json.dumps({"status": "prepared_not_submitted", "python": str(python), "config": config}, indent=2))
