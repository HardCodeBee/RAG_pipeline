"""Validate original4B return against locally frozen100 and four prompts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = HERE / "remote"
sys.path.insert(0,str(PACKAGE / "code"))
from contracts import (ARMS, BASE_MANIFEST, REPO, REVISION, digest, parse_array,
                       queries, read_json, save_json, sha, summary, validate_config)


def validate(archive, allow_partial=False):
    expected_rows = queries(PACKAGE / "queries100.query_only.jsonl")
    expected = {(a,r["id"]):r for r in expected_rows for a in ARMS}
    with ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) <= 2000
        assert sum(r.file_size for r in z.infolist()) < 100 * 1024 * 1024
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and "\\" not in n and ":" not in n for n in names)
        read = lambda name: json.loads(z.read(name))
        manifest = read("RETURN_MANIFEST.json")
        assert set(names) == set(manifest["files"]) | {"RETURN_MANIFEST.json"}
        for name,h in manifest["files"].items(): assert hashlib.sha256(z.read(name)).hexdigest() == h
        for name in ("queries100.query_only.jsonl","PROTOCOL_FROZEN.json","PACKAGE_MANIFEST.json","base_runtime.json"):
            assert z.read("package/"+name) == (PACKAGE/name).read_bytes(), name
        for a in ARMS: assert z.read("package/prompts/"+a+".txt") == (PACKAGE/"prompts"/(a+".txt")).read_bytes()
        config = read("runtime_config.json")
        validate_config(config)
        template = read_json(PACKAGE/"base_runtime.json")
        for k,v in template.items():
            if k != "default_route_root": assert config[k] == v, k
        identity = read("IDENTITY.json")
        assert identity["adapter_loaded"] is False and identity["model_manifest"] == BASE_MANIFEST
        assert identity["runtime_config_sha256"] == digest(config)
        assert identity["input_sha256"] == sha(PACKAGE/"queries100.query_only.jsonl")
        assert identity["package_manifest_sha256"] == sha(PACKAGE/"PACKAGE_MANIFEST.json")
        assert identity["prompt_sha256"] == {a:sha(PACKAGE/"prompts"/(a+".txt")) for a in ARMS}
        report = read("model_report.json")
        assert report["adapter"] is None and report["adapter_parameters"] == 0
        assert report["unique_parameters"] == report["base_unique_parameters"] == 4022468096
        assert report["base_identity"]["repo_id"] == REPO and report["base_identity"]["revision"] == REVISION
        assert report["base_identity"]["manifest_sha256"] == BASE_MANIFEST
        assert report["template_policy"] == "qwen3_instruct_native_nonthinking"
        predictions = [json.loads(s) for s in z.read("predictions.jsonl").decode("utf-8").splitlines() if s.strip()]
        seen = set()
        for r in predictions:
            key = (r["arm"],r["id"])
            assert key in expected and key not in seen
            seen.add(key)
            assert r["query"] == expected[key]["query"] and r["index"] == next(i for i,q in enumerate(expected_rows,1) if q["id"] == r["id"])
            assert r["identity_sha256"] == digest(identity) and r["model_manifest"] == BASE_MANIFEST and r["adapter_loaded"] is False
            assert r["prompt_sha256"] == identity["prompt_sha256"][r["arm"]]
            for k,v in parse_array(r["raw_output"]).items(): assert r[k] == v, (key,k)
            assert read(f"cases/{r['arm']}/{r['index']:03d}.json") == r
            if not r["generation_error"]:
                assert len(r["input_token_ids"]) == r["input_tokens"]
                assert len(r["output_token_ids"]) == r["output_tokens"]
                d = r["decoding_config"]
                assert d["do_sample"] is False and d["num_beams"] == 1 and d["max_new_tokens"] == 1024
                assert d["use_cache"] is True
        assert read("SUMMARY.json") == summary(predictions)
        assert {n for n in names if n.startswith("cases/") and n.endswith(".json")} == {f"cases/{r['arm']}/{r['index']:03d}.json" for r in predictions}
        assert manifest["completed_attempts"] == len(predictions)
        assert allow_partial or len(predictions) == 400, "Partial return: pass --allow-partial for diagnostic intake"
        return {"valid":True,"completed":len(predictions),"expected":400,"summary":summary(predictions),"semantic_review":"pending_current_Codex","archive_sha256":sha(archive),"api_calls":0}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args()
    result=validate(a.archive,a.allow_partial)
    out=ROOT/"outputs/query_decomposition"/HERE.name/("returned_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True,exist_ok=False)
    with ZipFile(a.archive) as z: z.extractall(out)
    save_json(out/"LOCAL_ACCEPTANCE.json",result)
    print(json.dumps({"directory":str(out),**result},ensure_ascii=False,indent=2))
