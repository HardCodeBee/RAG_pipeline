"""Offline only: freeze8 inputs and prove16 single-sentence payload contracts."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.child_prompts import CHILD_INSTRUCTIONS, child
from p02_multigrain_timing_cachefix_v1.core.pipeline import passages
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import templates

PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
MECHANISMS = ROOT / "analysis/p02_prompt_research_followup_20261022/ORDINARY_FAILURE_MECHANISMS.json"
FORMAL = HERE.parent / "p02_formal1500_v1_20261021"
FORMAL_OUT = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021"
QUERIES = FORMAL / "queries1500.id_query.frozen.jsonl"
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PRIOR = HERE.parent / "unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "INPUTS8.frozen.jsonl"
CAP, SEED = 128, 20261022


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False)+"\n" for v in values), encoding="utf8")


def order(probe_id):
    offset = int(hashlib.sha256(f"{SEED}|{probe_id}|ordinary_failure_order".encode()).hexdigest(),16) % 2
    return templates.ARMS[offset:] + templates.ARMS[:offset]


def public_index():
    prior = read(PRIOR)
    raw = MANIFEST.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    assert blob == prior["manifest_git_blob_sha"]
    assert prior["remote_git_blobs_live_verified_by_related_prompt_mechanisms_agent"] is True
    manifest = json.loads(raw)
    index = {r["source"].replace("\\", "/"): {**r, "container":r.get("bundle")} for r in manifest["source_files"]}
    for archive in manifest["archives"]:
        for member in archive.get("members", []):
            index[member["source"].replace("\\", "/")] = {**member, "container":archive["bundle"]}
    return index, dict(public_repository=prior["public_repository"], verified_public_commit=prior["verified_public_commit"],
        public_manifest=prior["public_manifest"], manifest_git_blob_sha=blob,
        verification="Offline recheck against previously live-verified public manifest; no new network verification")


def assemble():
    selected = read(MECHANISMS)["examples"]
    assert len(selected)==8 and Counter(x["eligible_for_same_reader_qa_challenge"] for x in selected)=={True:6,False:2}
    config = read(PACKAGE/"config.json")["generation"]
    examples = read(PACKAGE/"prompts/child_examples.json")
    qindex = {r["id"]:(i+1,r) for i,r in enumerate(rows(QUERIES))}
    public, publicproof = public_index()
    items, proofs, receipts, decomposition = [], [], [], []
    for x in selected:
        sourcepath = ROOT/x["source_path"]
        source = read(sourcepath)
        assert sha(sourcepath)==x["source_sha256"]==public[x["source_path"]]["sha256"]
        node = next(n for n in source["nodes"] if n["step_id"]==x["step_id"])
        assert node["role"]=="ordinary" and not node.get("sentence_format_request_id")
        assert source["id"]==x["id"] and source["question"]==x["original_question"]
        assert node["request"]==x["actual_current_question"]
        assert node["prior_complete_answers"]==x["history"]
        prompt = node["child_prompt"]
        assert hashlib.sha256(prompt.encode()).hexdigest()==x["actual_child_prompt_sha256"]
        assert prompt.startswith(CHILD_INSTRUCTIONS+templates.DELIMITER)
        assert child(source["question"],node["step_id"],node["request"],node["prior_complete_answers"],passages(node["reader_packet"]),examples,role="ordinary")==prompt
        actual = prompt.split("\n\nActual task:\n",1)[1]
        assert actual.split(f"Current subquestion #{node['step_id']}:\n",1)[1].split("\n\nRetrieved passages:",1)[0]==node["request"]
        payload = build_payload(config,prompt,CAP)
        dbpath = sourcepath.parents[3]/"methods"/x["dataset"]/"p02_multigrain_timing_cachefix_v1/checkpoint.sqlite3"
        with closing(sqlite3.connect(dbpath.resolve().as_uri()+"?mode=ro",uri=True)) as db:
            row = db.execute("SELECT payload,status,result FROM requests WHERE id=?",(node["answer_request_id"],)).fetchone()
        assert row and json.loads(row[0])==payload and row[0]==canonical(payload) and row[1]=="success"
        result = json.loads(row[2]);usage=result.get("usage") or {}
        assert result["status"]=="success" and result["finish_reason"]=="stop" and not result.get("refusal")
        assert result["returned_model"]==payload["model"]
        assert all(type(usage.get(k)) is int and usage[k]>=0 for k in ["prompt_tokens","completion_tokens","total_tokens"])
        assert usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
        assert result["answer"].strip()==node["initial_answer"]==node["answer"]==x["initial_answer"]
        position,qr = qindex[source["id"]]
        rawpath=FORMAL_OUT/"remote_raw"/f"{position:03d}.json"
        raw=read(rawpath)
        assert qr["query"]==source["question"]==raw["query"] and raw["id"]==source["id"] and raw["index"]==position
        assert raw["parsed"]==source["supporting_questions"] and raw["format_valid"] and not raw["truncated"] and not raw["generation_error"]
        assert raw["actual_prompt"].startswith((FORMAL/"v12.txt").read_text(encoding="utf8").rstrip())
        probe_id=source["id"]+"/ordinary/"+str(node["step_id"])
        item=dict(probe_id=probe_id,id=source["id"],dataset=x["dataset"],step_id=node["step_id"],role="ordinary",
            source=x["source_path"],source_sha256=x["source_sha256"],original_question=source["question"],
            current_question=node["request"],history=node["prior_complete_answers"],reader_packet=node["reader_packet"],
            supporting_questions=source["supporting_questions"],control_prompt=prompt,control_payload=payload,
            historical_request_id=node["answer_request_id"])
        items.append(item)
        receipts.append(dict(probe_id=probe_id,source_db=rel(dbpath),request_id=node["answer_request_id"],payload_sha256=digest(payload),
            result_sha256=digest(result),result=result,legacy_requested_model_bound_via_exact_payload=payload["model"]))
        dec=dict(probe_id=probe_id,formal1500_position_one_based=position,raw_file=rel(rawpath),raw_sha256=sha(rawpath),
            source_sha256=x["source_sha256"],actual_raw_identity_sha256=raw["identity_sha256"],supporting_questions_exact=True,
            actual_R02_array=raw["parsed"],semantic_repairs_applied=False)
        decomposition.append(dec)
        proofs.append(dict(probe_id=probe_id,source=x["source_path"],source_sha256=x["source_sha256"],
            public_container=public[x["source_path"]]["container"],public_manifest_exact=True,original_wire_payload_exact=True,
            legacy_complete_stop_usage=True,renderer_exact=True,current_from_actual_child_prompt=True,history_reader_examples_unchanged=True,
            decomposition_actual_R02_array_exact=True))
    return items,proofs,receipts,decomposition,{**publicproof,"verified_nodes":8,"cases":proofs}


def make_contracts(items):
    generation=read(PACKAGE/"config.json")["generation"]
    contracts=[]
    for item in items:
        original=item["control_prompt"]
        for arm in templates.ARMS:
            prompt=templates.prompt(original,arm)
            payload=build_payload(generation,prompt,CAP)
            if arm=="I0":
                assert prompt==original and payload==item["control_payload"]
            else:
                assert prompt.replace("\n"+templates.SENTENCE+templates.DELIMITER,templates.DELIMITER,1)==original
                assert prompt.split(templates.DELIMITER,1)[1]==original.split(templates.DELIMITER,1)[1]
            assert {k:v for k,v in payload.items() if k!="messages"}=={k:v for k,v in item["control_payload"].items() if k!="messages"}
            contracts.append(dict(probe_id=item["probe_id"],arm=arm,payload_sha256=digest(payload),max_tokens=CAP,
                exact_original_I0=(arm=="I0"),only_confirmed_sentence_I1=(arm=="I1"),
                actual_current_history_reader_fewshots_all_exact=True,supporting_questions_exact=True))
    assert len(contracts)==len({r["payload_sha256"] for r in contracts})==16
    return contracts


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    items,proofs,receipts,dec,public=assemble()
    jsonl(INPUTS,items)
    jsonl(HERE/"SOURCE_PROOF8.frozen.jsonl",proofs)
    jsonl(HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl",receipts)
    jsonl(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl",dec)
    save(HERE/"PUBLIC_PAYLOAD_PROOF.json",public)
    jsonl(HERE/"CONTRACTS16.frozen.jsonl",make_contracts(items))
    selection=[{k:x[k] for k in ["category","dataset","review_index","id","step_id","eligible_for_same_reader_qa_challenge","limits"]} for x in read(MECHANISMS)["examples"]]
    jsonl(HERE/"SELECTION8.frozen.jsonl",selection)
    jsonl(HERE/"TASK_ORDER16.frozen.jsonl",[dict(index=i*2+j,task_id=item["probe_id"]+"/"+arm,probe_id=item["probe_id"],arm=arm) for i,item in enumerate(items) for j,arm in enumerate(order(item["probe_id"]))])
    save(HERE/"GENERATION.frozen.json",read(PACKAGE/"config.json")["generation"])
    (HERE/"I1_SENTENCE.frozen.txt").write_text(templates.SENTENCE+"\n",encoding="utf8")
    for arm in templates.ARMS:
        (HERE/(arm+"_EXAMPLE_PROMPT.txt")).write_text(templates.prompt(items[0]["control_prompt"],arm),encoding="utf8")
    pins={rel(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    for p in [MECHANISMS,MANIFEST,PRIOR,QUERIES,FORMAL/"REMOTE_INPUTS.frozen.json",FORMAL/"v12.txt",PACKAGE/"config.json",
              PACKAGE/"core/child_prompts.py",PACKAGE/"core/llm.py",PACKAGE/"core/pipeline.py",PACKAGE/"prompts/child_examples.json"]:
        pins[rel(p)]=sha(p)
    pins.update({r["source"]:r["source_sha256"] for r in items})
    pins.update({r["raw_file"]:r["raw_sha256"] for r in dec})
    save(PROTOCOL,dict(id=HERE.name,state="offline_prepared_no_runner_or_authorization",nodes=8,conditions=16,arms=list(templates.ARMS),
        cap=CAP,sentence=templates.SENTENCE,groups={"six_reader_local_QA_failures":6,"two_reasonable_abstention_boundaries":2},
        selection="All8 existing ORDINARY_FAILURE_MECHANISMS examples in same order; no new-output/final-score filtering",
        control="I0 exact original full payload, fresh contemporaneous generation if later authorized; never historical response seeding",
        intervention="I1 only append confirmed sentence at original main instruction end before Worked examples; no gold/entity fill or decomposition repair",
        fixed="Original R02/v12 raw array,current question,history,reader,examples,model,128cap; no formatter/upstream/downstream rerun",
        task_order="SHA256(20261022|probe_id|ordinary_failure_order) modulo2 swaps within-node arm order; all outputs retained",
        source_R02=read(FORMAL/"REMOTE_INPUTS.frozen.json"),
        limits="Exposed selected mechanism challenge; no population accuracy, causal attribution or completeRAG F1 claim;371 decomposition extra generation and122 downstream recovery remain",
        execution="Only root may decide afterW90; no RUN_AUTHORIZATION or API entrypoint in this offline package",
        main_evaluation_remains_paused=True,new_API_requests=0,new_retrieval=0,new_GPU_jobs=0,pins=pins))
    return dry_run()


def dry_run():
    protocol=read(PROTOCOL)
    for path,expected in protocol["pins"].items():
        assert sha(ROOT/path)==expected,path
    items,proofs,receipts,dec,public=assemble()
    assert items==rows(INPUTS)
    assert proofs==rows(HERE/"SOURCE_PROOF8.frozen.jsonl")
    assert receipts==rows(HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl")
    assert dec==rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public==read(HERE/"PUBLIC_PAYLOAD_PROOF.json")
    assert make_contracts(items)==rows(HERE/"CONTRACTS16.frozen.jsonl")
    tasks=rows(HERE/"TASK_ORDER16.frozen.jsonl")
    assert len(tasks)==len({t["task_id"] for t in tasks})==16
    for item in items:
        assert [t["arm"] for t in tasks if t["probe_id"]==item["probe_id"]]==list(order(item["probe_id"]))
    assert not (HERE/"RUN_AUTHORIZATION.json").exists() and not (OUT/"checkpoint.sqlite3").exists()
    assert read(HERE.parent/"unified_rag_eval_v1_20261021/PLAN.json")["execution_approval"]["approved"] is False
    result=dict(passed=True,nodes=8,conditions=16,source_public_hashes_exact=8,old_wire_payloads_complete_stop_usage=8,
        current_renderer_matches=8,actual_current_from_child_prompt=8,R02_raw_arrays_exact=8,only_sentence_addition_contracts=8,
        original_history_reader_fewshots_decomposition_unchanged=True,groups={"local_QA_failure":6,"reasonable_abstention_boundary":2},
        model_payloads_contain_no_new_gold_or_expected_answer=True,no_client_constructed=True,no_Cache_constructed=True,
        new_API_requests=0,no_execution_entrypoint=True,main_evaluation_remains_paused=True,protocol_sha256=sha(PROTOCOL))
    save(OUT/"DRYRUN.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","dry-run"])
    args=p.parse_args()
    print(json.dumps(prepare() if args.action=="prepare" else dry_run(),ensure_ascii=True))
