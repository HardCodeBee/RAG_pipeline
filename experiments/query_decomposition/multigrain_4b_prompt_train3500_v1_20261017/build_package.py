"""Freeze fresh100 and create an API-free base4B remote package."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import unicodedata
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = HERE / "remote"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
RETURNED = ROOT / "outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer"
SEED = "20261017-original4b-prompt100-01"
QUOTAS = {"hotpotqa": {"bridge":24,"comparison":10}, "2wiki": {"comparison":9,"inference":8,"compositional":8,"bridge_comparison":8}, "musique": {"2-hop":11,"3-hop":11,"4-hop":11}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip(" \t\n?!？！.")


def freeze(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode()
    if path.exists():
        assert path.read_bytes() == data, str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare_samples():
    frozen = HERE / "questions100.frozen.jsonl"
    if frozen.exists():
        return rows(frozen)
    previous = ROOT / "experiments/query_decomposition/multigrain_v8_simplify_v1_20261017/SELECTION_FROZEN.json"
    previous_metadata = json.loads(previous.read_text(encoding="utf-8"))
    paths = {ROOT/r["path"] for r in previous_metadata["excluded_query_files"]}
    scope = ROOT / "experiments/query_decomposition"
    paths.update(scope.rglob("*questions*.jsonl"))
    paths.update(scope.rglob("cases.frozen.jsonl"))
    paths.add(RETURNED / "data/accepted_master.jsonl")
    ids, texts, sources = set(), set(), []
    for p in sorted(paths):
        if not p.exists(): continue
        sources.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p)})
        for r in rows(p):
            rid = r.get("qid",r.get("id"))
            if rid: ids.add(rid)
            question = r.get("question",r.get("query",""))
            if isinstance(question,str) and question: texts.add(norm(question))
            if isinstance(r.get("source"),dict) and r["source"].get("id"): ids.add(r["source"]["id"])
    manifest = json.loads((RETURNED / "data/split_manifest.json").read_text(encoding="utf-8"))
    for r in manifest["records"]:
        if r["id"] in ids or r["project_assignment"] in {"project_dev","project_test","reserved_test_family","reserved_dev_family"}:
            ids.add(r.get("source",{}).get("id",""))
    screen = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_FROZEN.json"
    ids.update(json.loads(screen.read_text(encoding="utf-8"))["ids"])
    native = ROOT / "data/multihop_comprehensive_v1_20261015"
    native_files = {"hotpotqa":ROOT/"analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl", "2wiki":native/"2wiki/native/dev.json", "musique":native/"musique/native/musique_ans_v1.0_dev.jsonl"}
    pool, seen = [], set()
    for dataset,p in native_files.items():
        records = json.loads(p.read_text(encoding="utf-8")) if dataset == "2wiki" else rows(p)
        for r in records:
            qid, query = r.get("_id",r.get("id")),r["question"]
            key = norm(query)
            if qid in ids or key in texts or key in seen: continue
            seen.add(key)
            kind = str(len(r["question_decomposition"]))+"-hop" if dataset == "musique" else r["type"]
            pool.append({"id":dataset+":"+qid,"qid":qid,"dataset":dataset,"query":query,"type":kind,"source_split":"dev"})
    rank = lambda value: hashlib.sha256((SEED+value).encode()).hexdigest()
    selected = []
    for dataset, quotas in QUOTAS.items():
        for kind,n in quotas.items():
            choices = sorted((r for r in pool if r["dataset"] == dataset and r["type"] == kind),key=lambda r:rank(r["id"]))
            assert len(choices) >= n
            selected.extend(choices[:n])
    selected.sort(key=lambda r:rank("order:"+r["id"]))
    for i,r in enumerate(selected,1): r["index"] = i
    assert len(selected) == len({r["id"] for r in selected}) == len({norm(r["query"]) for r in selected}) == 100
    frozen.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected),encoding="utf-8")
    freeze(HERE/"SELECTION_FROZEN.json",{"seed":SEED,"quotas":QUOTAS,"sample_role":"prompt_selection_and_pretraining_baseline_not_final_unseen_test","selection_uses_answers_or_outcomes":False,"excluded_files":sources,"excluded_ids":len(ids),"excluded_exact_normalized_texts":len(texts),"sources":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in native_files.values()],"limitations":"Known ID and normalized-text exclusion, not complete paraphrase/pretraining decontamination."})
    (HERE/"EXCLUDE_FROM_TRAINING.jsonl").write_text("".join(json.dumps({"dataset":r["dataset"],"qid":r["qid"],"query":r["query"],"role":"prompt_selection100_do_not_train"},ensure_ascii=False)+"\n" for r in selected),encoding="utf-8")
    return selected


def main():
    selected = prepare_samples()
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE/"queries100.query_only.jsonl").write_text("".join(json.dumps({"id":r["id"],"query":r["query"]},ensure_ascii=False)+"\n" for r in selected),encoding="utf-8")
    prompt_sources = {"v6":HERE.parent/"multigrain_decomposer_fresh_eval_v1_20261017/TRAINING_PROMPT_v6.txt"}
    prompt_sources.update({a:HERE.parent/"multigrain_v8_simplify_v1_20261017"/("TRAINING_PROMPT_"+a+".txt") for a in ("v8","v11","v12")})
    (PACKAGE/"prompts").mkdir(exist_ok=True)
    for a,p in prompt_sources.items(): shutil.copyfile(p,PACKAGE/"prompts"/(a+".txt"))
    (PACKAGE/"runtime").mkdir(exist_ok=True)
    for name in ("model_runtime.py","adapter_runtime.py","run_stage.slurm"):
        shutil.copyfile(RETURNED/"work"/name,PACKAGE/"runtime"/name)
    shutil.copyfile(RETURNED/"BASE_MODEL_IDENTITY.json",PACKAGE/"BASE_MODEL_IDENTITY.expected.json")
    old = json.loads((RETURNED/"configs/predict_initial.json").read_text(encoding="utf-8"))
    cfg = {"default_route_root":str(PurePosixPath(old["model_path"]).parent.parent),
           "repo_id":"Qwen/Qwen3-4B-Instruct-2507","revision":"cdbee75f17c01a7cc42f958dc650907174af0554",
           "adapter_path":None,"expected_base_parameters":4022468096,"expected_adapter_parameters":0,"allow_upstream_checkpoint":True,
           "template_policy":"qwen3_instruct_native_nonthinking","dtype":"bfloat16","runtime_device":"cuda","seed":1729,
           "max_new_tokens":1024,"num_beams":1,"allow_tf32":False,"attn_implementation":"sdpa","arms":["v6","v8","v11","v12"]}
    freeze(PACKAGE/"base_runtime.json",cfg)
    protocol = {"task":"original4B_shared100_four_prompt_selection","queries":100,"planned_outputs":400,"arms":cfg["arms"],"api_calls":0,"adapter_loaded":False,"training_started":False,
                "query_sha256":sha(PACKAGE/"queries100.query_only.jsonl"),"selection_sha256":sha(HERE/"SELECTION_FROZEN.json"),
                "prompts":{a:{"source":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for a,p in prompt_sources.items()},
                "input":"system=exact frozen prompt,user=original query only; no source answers/passages/reference decompositions",
                "review":"Current Codex reviews400 literal outputs under unchanged F/S/C/U; not human gold. Whole-query pass includes format compliance; uncertain/generation failures/truncation retain full100 denominator.",
                "selection_order":["whole-query semantic pass count","minimum dataset pass rate","fewer distinct cases with F or S failure","fewer mean input tokens"],
                "model":"base Qwen3-4B-Instruct-2507 only; original weights, no adapter",
                "no_postprocessing":"No inferred rewrites, reference replacement, JSON repair, format-constrained generation or extra editorial calls.",
                "same_queries_all_arms":True,"queries_excluded_from_training":True,"remote_scope":"inference_only_user_manual_upload_return"}
    freeze(PACKAGE/"PROTOCOL_FROZEN.json",protocol)
    files = {str(p.relative_to(PACKAGE)).replace("\\","/"):sha(p) for p in sorted(PACKAGE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and "results" not in p.parts and p.name != "PACKAGE_MANIFEST.json" and not p.name.endswith((".zip",".pyc"))}
    freeze(PACKAGE/"PACKAGE_MANIFEST.json",{"task":protocol["task"],"api_calls":0,"adapter_loaded":False,"expected_outputs":400,"files":files})
    OUT.mkdir(parents=True,exist_ok=True)
    archive = OUT/"original4b_prompt100_remote_package.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        for name in [*files,"PACKAGE_MANIFEST.json"]: z.write(PACKAGE/name,name)
    with ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist()) == {*files,"PACKAGE_MANIFEST.json"}
        for name,expected in files.items(): assert hashlib.sha256(z.read(name)).hexdigest() == expected
    receipt = {"utc":datetime.now(timezone.utc).isoformat(),"archive":str(archive),"sha256":sha(archive),"bytes":archive.stat().st_size,"queries":100,"arms":4,"outputs_expected":400,"datasets":dict(Counter(r["dataset"] for r in selected)),"api_calls":0,"weights_included":False,"adapter_loaded":False,"status":"built_not_remote_executed"}
    (OUT/"PACKAGE_BUILD.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
