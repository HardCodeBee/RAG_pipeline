"""Remote CPU-only staging and optional Slurm submission for the approved200."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Existing run file differs: " + str(path))
    else:
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        if sha(package / filename) != expected:
            raise ValueError("Changed package file: " + filename)
    cfg = json.loads((package / "predict200.proposed.json").read_text(encoding="utf-8"))
    original_root = Path(cfg["adapter_path"]).parents[2]
    root = args.route_root.resolve() if args.route_root else original_root
    # Only deployment paths may change; retain all instructions and inference settings.
    for field in ("adapter_path", "model_path", "base_identity_file", "input_file", "output_dir"):
        cfg[field] = str(root / Path(cfg[field]).relative_to(original_root))
    python = root / "environment/venv/bin/python"
    launcher, predictor = root / "work/run_stage.slurm", root / "work/predict.py"
    adapter = Path(cfg["adapter_path"])
    for item in (python, launcher, predictor, adapter / "adapter_model.safetensors", adapter / "manifest.json",
                 Path(cfg["base_identity_file"]), Path(cfg["model_path"]) / "model.safetensors.index.json"):
        if not item.is_file():
            raise ValueError("Required existing remote file missing: " + str(item))
    checkpoint = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    if checkpoint["manifest_sha256"] != manifest["adapter_manifest_sha256"]:
        raise ValueError("Not the approved checkpoint-000148 manifest")
    base = json.loads(Path(cfg["base_identity_file"]).read_text(encoding="utf-8"))
    if base["manifest_sha256"] != manifest["base_manifest_sha256"]:
        raise ValueError("Not the approved base-model manifest")
    combined = hashlib.sha256(json.dumps({"base_model_manifest": base["manifest_sha256"],
        "adapter_manifest": checkpoint["manifest_sha256"]}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if combined != manifest["expected_model_manifest"]:
        raise ValueError("Not the approved combined model identity")
    expected_weight = next(r for r in checkpoint["files"] if r["path"] == "adapter_model.safetensors")
    if (adapter / "adapter_model.safetensors").stat().st_size != expected_weight["bytes"]:
        raise ValueError("Adapter size differs; the existing predictor also verifies full hashes")
    rows = [json.loads(s) for s in (package / "queries200.query_only.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows) != 300 or len({r["id"] for r in rows}) != 300 or any(set(r) != {"id", "query"} for r in rows):
        raise ValueError("Expected exactly300 unique query-only records")
    if cfg["max_new_tokens"] != 1024 or cfg["num_beams"] != 1 or cfg["seed"] != 1729:
        raise ValueError("Approved decoding settings changed")
    run = Path(cfg["input_file"]).parent
    run.mkdir(parents=True, exist_ok=True)
    save_once(Path(cfg["input_file"]), (package / "queries200.query_only.jsonl").read_bytes())
    config_file = run / "predict200.json"
    save_once(config_file, (json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    submission_file = run / "job_submission.json"
    info = {"status": "prepared", "route_root": str(root), "run": str(run), "config": str(config_file),
            "input_sha256": sha(Path(cfg["input_file"])), "inputs": 300}
    if args.submit:
        if submission_file.exists():
            print(submission_file.read_text(encoding="utf-8"))
            return
        command = ["sbatch", "--parsable", "--job-name=qdmr4b-eval300", "--chdir=" + str(run),
                   "--output=" + str(run / "slurm-%j.out"), "--error=" + str(run / "slurm-%j.err"),
                   str(launcher), str(python), str(predictor), "--config", str(config_file)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError("Unexpected sbatch response; reconcile scheduler before resubmitting")
        info.update(status="submitted", job_id=job_id, command=command)
        save_once(submission_file, (json.dumps(info, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
