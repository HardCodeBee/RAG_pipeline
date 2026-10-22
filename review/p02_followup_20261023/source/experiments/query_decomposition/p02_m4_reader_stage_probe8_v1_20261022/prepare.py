"""Select before inspecting local QA outcomes; verify actual rawD/public sources/wire."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.child_prompts import child,GOAL_RULE
from p02_multigrain_timing_cachefix_v1.core.pipeline import passages
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import prepare as source
from experiments.query_decomposition.p02_live_target_balanced20_dev_v1_20261022.native_ask import integrity,stage_success
from . import templates

PACKAGE=ROOT/"src/p02_multigrain_timing_cachefix_v1";PLAN=HERE.parent/"unified_rag_eval_v1_20261021"
OUT=ROOT/"outputs/query_decomposition"/HERE.name
INPUTS=HERE/"INPUTS8.frozen.jsonl";ORDER=HERE/"ORDER16.frozen.jsonl";PROTOCOL=HERE/"PROTOCOL.frozen.json"
EVIDENCE=PLAN/"reviews/20261022_prompt_binding_goal_synthesis/evidence_selection_readonly"
INVENTORY=EVIDENCE/"NODE_POOL_INVENTORY.jsonl"
CAP=128;SEED=20261022
read,rows,sha,rel,jsonl=source.read,source.rows,source.sha,source.rel,source.jsonl
TARGET=("2wiki:d8ffd42a0bda11eba7f7acde48001122",2)


def digest(value):return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def selection_hash(row):
    return hashlib.sha256(f"p02-m4-reader-stage8-v1|{SEED}|{row['id']}|{row['step_id']}".encode()).hexdigest()


def select():
    pool=[r for r in rows(INVENTORY)if r["dataset"]=="2wiki"and r["changed_membership"]["M4_format_duplicate_fill"]]
    assert len(pool)==138 and Counter(r["role"]for r in pool)=={"ordinary":43,"goal":95}
    target=next(r for r in pool if(r["id"],r["step_id"])==TARGET)
    assert target["role"]=="goal"
    selected=[{**target,"selection_reason":"mandatory_exposed_Wiki721_candidate_reader_loss"}];used={target["id"]}
    strata=[("goal","seeded_fullscore_control",3),("ordinary","all_nonfullscore",2),("ordinary","seeded_fullscore_control",2)]
    strata_proof=[]
    for role,prior,n in strata:
        group=sorted([r for r in pool if r["role"]==role and r["selection"]==prior],key=lambda r:(selection_hash(r),r["id"],r["step_id"]))
        accepted=[]
        for row in group:
            if row["id"]in used:continue
            selected.append({**row,"selection_reason":"hashed_"+role+"_"+prior});accepted.append([row["id"],row["step_id"]]);used.add(row["id"])
            if len(accepted)==n:break
        assert len(accepted)==n,"Insufficient distinct queries; report rather than relax"
        strata_proof.append(dict(role=role,prior_selection=prior,available_nodes=len(group),planned=n,accepted=accepted))
    assert len(selected)==len(used)==8 and Counter(r["role"]for r in selected)=={"ordinary":4,"goal":4}
    manifest=dict(rule="One fixed Wiki721; then3successgoal,2nonfullordinary,2successordinary;hashascending;globaldistinctquerybeforeanyQAread",
        seed=SEED,hash_expression="SHA256(p02-m4-reader-stage8-v1|20261022|canonicalid|step_id)",
        inventory_sha256=sha(INVENTORY),eligible_changed_nodes=138,all_selected_prior_semantic_exposure=True,
        no_local_QA_truth_or_answer_or_new_outcome_selection=True,strata=strata_proof,
        selected=[dict(id=r["id"],step_id=r["step_id"],role=r["role"],prior_selection=r["selection"],selection_reason=r["selection_reason"],selection_sha256=selection_hash(r),source=r["source"],source_sha256=r["source_sha256"])for r in selected])
    return selected,manifest


def source_receipt(db,rid,payload):
    row=db.execute("SELECT payload,status,result FROM requests WHERE id=?",(rid,)).fetchone()
    assert row and row[0]==canonical(payload) and row[1]=="success","Missing/mismatched historicalwire"
    result={**json.loads(row[2]),"status":row[1]}
    # Older wire may omit requested_model; bind it only through exact payload.
    assert result.get("requested_model")in(None,payload["model"])
    bound={**result,"requested_model":payload["model"]}
    assert stage_success(bound,payload["model"],"legacy/answer/1")
    return dict(request_id=rid,payload_sha256=digest(payload),result_sha256=digest(result),result=result,
        legacy_requested_model_bound_via_exact_payload=payload["model"])


def assemble():
    selected,manifest=select()  # This is complete before opening raw case/QA.
    config=read(PACKAGE/"config.json")["generation"];examples=read(PACKAGE/"prompts/child_examples.json")
    qindex={r["id"]:(i+1,r)for i,r in enumerate(rows(source.QUERIES))}
    public,publicproof=source.public_index()
    items=[];proofs=[];receipts=[];decomposition=[]
    for row in selected:
        path=ROOT/row["source"];case=read(path)
        assert sha(path)==row["source_sha256"]==public[row["source"]]["sha256"]
        node=next(n for n in case["nodes"]if n["step_id"]==row["step_id"])
        assert node["role"]==row["role"] and case["id"]==row["id"] and case["dataset"]=="2wiki"
        pid=case["id"]+"/"+node["role"]+"/"+str(node["step_id"])
        value=dict(probe_id=pid,id=case["id"],dataset=case["dataset"],step_id=node["step_id"],role=node["role"],
            original_question=case["question"],current_question=node["request"],history=node["prior_complete_answers"],
            supporting_questions=case["supporting_questions"],original_retrieval=node["original_retrieval"],rewritten_retrieval=node["rewritten_retrieval"],
            rewritten_query=node["rewritten_query"],reader_packet=node["reader_packet"],control_prompt=node["child_prompt"],
            source=row["source"],source_sha256=row["source_sha256"],historical_request_id=node["answer_request_id"])
        rendered=child(case["question"],node["step_id"],node["request"],node["prior_complete_answers"],passages(node["reader_packet"]),examples,role=node["role"])
        if node["role"]=="goal":rendered=rendered[:-len("Answer:")]+GOAL_RULE+"\n\nAnswer:"
        assert rendered==node["child_prompt"]==templates.prompt(value,"M0")
        assert templates.reader(value,"M0")==node["reader_packet"]
        assert [d["doc_id"]for d in templates.reader(value,"M4")["docs"]]==row["selected_doc_ids"]["M4_format_duplicate_fill"]
        assert templates.prompt(value,"M4")!=rendered
        payload=build_payload(config,rendered,CAP);value["control_payload"]=payload
        dbpath=path.parents[3]/"methods"/case["dataset"]/"p02_multigrain_timing_cachefix_v1/checkpoint.sqlite3"
        with closing(sqlite3.connect(dbpath.resolve().as_uri()+"?mode=ro",uri=True))as db:
            receipt=source_receipt(db,node["answer_request_id"],payload)
            assert receipt["result"]["answer"].strip()==node["initial_answer"]
            fmt=None
            assert templates.needs_formatter(node["initial_answer"])==bool(node.get("sentence_format_request_id"))
            if node.get("sentence_format_request_id"):
                fmt_payload=build_payload(config,templates.formatter_prompt(value,"M0",node["initial_answer"]),CAP)
                fmt=source_receipt(db,node["sentence_format_request_id"],fmt_payload)
                assert fmt["result"]["answer"].strip()==node["answer"]
            else:assert node["initial_answer"]==node["answer"]
        value["historical_stage_usage_reference"]=receipt["result"]["usage"]
        value["historical_formatter_usage_reference"]=fmt["result"]["usage"]if fmt else None
        position,qr=qindex[case["id"]];rawpath=source.FORMAL_OUT/"remote_raw"/f"{position:03d}.json";raw=read(rawpath)
        assert qr["query"]==case["question"]==raw["query"] and raw["id"]==case["id"] and raw["index"]==position
        assert raw["parsed"]==case["supporting_questions"] and raw["format_valid"] and not raw["truncated"] and not raw["generation_error"]
        assert raw["actual_prompt"].startswith((source.FORMAL/"v12.txt").read_text(encoding="utf8").rstrip())
        assert not({"gold","reference","prediction","f1","em","selection_reason"}&set(value))
        items.append(value)
        receipts.append(dict(probe_id=pid,source_db=rel(dbpath),answer=receipt,formatter=fmt))
        proofs.append(dict(probe_id=pid,source=row["source"],source_sha256=row["source_sha256"],public_manifest_exact=True,
            public_container=public[row["source"]]["container"],original_wire_payload_exact=True,source_stop_model_all3_usage=True,
            original_formatter_triggered=bool(fmt),M0_renderer_and_reader_exact=True,M4_only_actual_reader_changed=True))
        decomposition.append(dict(probe_id=pid,raw_file=rel(rawpath),raw_sha256=sha(rawpath),formal1500_position_one_based=position,
            actual_raw_identity_sha256=raw["identity_sha256"],actual_R02_array=raw["parsed"],array_exact=True,manual_repairs_applied=False))
    return items,manifest,proofs,receipts,decomposition,{**publicproof,"verified_nodes":8,"cases":proofs}


def tasks(items):
    return [dict(index=i*2+j,id=r["id"],dataset=r["dataset"],probe_id=r["probe_id"],step_id=r["step_id"],role=r["role"],arm=arm,task_id=r["probe_id"]+"/"+arm)
        for i,r in enumerate(items)for j,arm in enumerate(templates.ARMS[i%2:]+templates.ARMS[:i%2])]


def contracts(items):
    config=read(PACKAGE/"config.json")["generation"];result=[]
    for item in items:
        base=item["control_prompt"];before,after=base.rsplit(templates.READER,1)
        for arm in templates.ARMS:
            text=templates.prompt(item,arm);b,newreader=text.rsplit(templates.READER,1)
            assert b==before and newreader.endswith(templates.suffix(item))
            assert newreader==passages(templates.reader(item,arm))+templates.suffix(item)
            payload=build_payload(config,text,CAP)
            assert {k:v for k,v in payload.items()if k!="messages"}=={k:v for k,v in item["control_payload"].items()if k!="messages"}
            if arm=="M0":assert payload==item["control_payload"]
            assert len(templates.reader(item,arm)["docs"])==10
            result.append(dict(task_id=item["probe_id"]+"/"+arm,arm=arm,role=item["role"],payload_sha256=digest(payload),
                only_actual_reader_membership_and_chunk_numbers_change=True,original_question_rawCurrent_history_instructions_examples_goalrule_fixed=True,
                cap=CAP,formatter_native_128_preserved=True))
    assert len(result)==len({r["payload_sha256"]for r in result})==16
    return result


def prepare():
    if PROTOCOL.exists():return dry_run()
    assert not(HERE/"RUN_AUTHORIZATION.json").exists()and not(OUT/"checkpoint.sqlite3").exists()
    items,manifest,proofs,receipts,decomp,public=assemble()
    jsonl(INPUTS,items);jsonl(ORDER,tasks(items));jsonl(HERE/"CONTRACTS16.frozen.jsonl",contracts(items))
    for name,value in[("SELECTION.frozen.json",manifest),("PUBLIC_PAYLOAD_PROOF.json",public),("GENERATION.frozen.json",read(PACKAGE/"config.json")["generation"])]:atomic_json(HERE/name,value)
    for name,data in[("SOURCE_PROOF8.frozen.jsonl",proofs),("HISTORICAL_RECEIPTS8.local.frozen.jsonl",receipts),("DECOMPOSITION_PROOF8.frozen.jsonl",decomp)]:jsonl(HERE/name,data)
    return dict(state="offline_prepared_not_authorized",cases=8,conditions=16,roles=dict(Counter(r["role"]for r in items)),API_GPU=0)


def check():
    p=read(PROTOCOL)
    for path,expected in{**p["pins"],**p["protected_holds"]}.items():assert sha(ROOT/path)==expected,path
    assert read(PLAN/"PLAN.json")["execution_approval"]["approved"]is False
    return p


def dry_run():
    check();items,manifest,proofs,receipts,decomp,public=assemble()
    assert rows(INPUTS)==items and rows(ORDER)==tasks(items)and rows(HERE/"CONTRACTS16.frozen.jsonl")==contracts(items)
    assert read(HERE/"SELECTION.frozen.json")==manifest and read(HERE/"PUBLIC_PAYLOAD_PROOF.json")==public
    assert rows(HERE/"SOURCE_PROOF8.frozen.jsonl")==proofs and rows(HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl")==receipts
    assert rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")==decomp
    value=dict(passed=True,conditions=16,distinct_queries=8,ordinary=4,goal=4,source_wire_public_actual_R02_exact=True,
        reader_only_contracts=True,native_formatter128=True,stage_only_no_RAG_F1=True,seed=None,API_GPU=0,protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT/"DRYRUN.json",value);return value
