"""Read-only source verification plus local preparation; no model client."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import prepare as source
from . import templates

PACKAGE=ROOT/"src/p02_multigrain_timing_cachefix_v1"
PLAN=HERE.parent/"unified_rag_eval_v1_20261021"
OUT=ROOT/"outputs/query_decomposition"/HERE.name
INPUTS=HERE/"INPUTS8.frozen.jsonl"
ORDER=HERE/"ORDER16.frozen.jsonl"
PROTOCOL=HERE/"PROTOCOL.frozen.json"
CAP=128
SEED=20261022
read,rows,sha,rel,jsonl=source.read,source.rows,source.sha,source.rel,source.jsonl


def digest(value):return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def assemble():
    items,proofs,receipts,decomposition,public=source.assemble()
    assert items==rows(source.INPUTS)
    assert proofs==rows(source.HERE/"SOURCE_PROOF8.frozen.jsonl")
    assert receipts==rows(source.HERE/"HISTORICAL_RECEIPTS8.local.frozen.jsonl")
    assert decomposition==rows(source.HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public==read(source.HERE/"PUBLIC_PAYLOAD_PROOF.json")
    assert len(items)==8 and Counter(r["dataset"]for r in items)=={"hotpotqa":5,"2wiki":3}
    assert all(r["role"]=="ordinary"for r in items)
    usage={r["probe_id"]:r["result"]["usage"]for r in receipts}
    records=[]
    for item in items:
        # Every message is already rebuilt from actual source/history/reader
        # and checked against its exact public, successful historical wirebody.
        assert templates.prompt(item,"O0")==item["control_prompt"]
        value={**item,"historical_stage_usage_reference":usage[item["probe_id"]]}
        assert not({"gold","reference","prediction","f1","em"}&set(value))
        records.append(value)
    return records,proofs,decomposition,public


def tasks(items):
    return [dict(index=i*2+j,id=r["id"],dataset=r["dataset"],probe_id=r["probe_id"],
        step_id=r["step_id"],arm=arm,task_id=r["probe_id"]+"/"+arm)
        for i,r in enumerate(items)for j,arm in enumerate(templates.ARMS[i%2:]+templates.ARMS[:i%2])]


def contracts(items):
    settings=read(PACKAGE/"config.json")["generation"]
    result=[]
    for item in items:
        original=item["control_payload"]
        for arm in templates.ARMS:
            text=templates.prompt(item,arm);payload=build_payload(settings,text,CAP)
            prefix,actual=item["control_prompt"].split(templates.ACTUAL,1)
            changed_prefix,changed_actual=text.split(templates.ACTUAL,1)
            assert changed_prefix==prefix
            block=templates.removed_block(item)
            if arm=="O0":assert payload==original and changed_actual==actual
            else:
                assert changed_actual==actual[len(block):] and block+changed_actual==actual
                restored={**payload,"messages":[{**payload["messages"][0],"content":prefix+templates.ACTUAL+block+changed_actual}]}
                assert restored==original
            assert f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\nRetrieved passages:'in changed_actual
            result.append(dict(task_id=item["probe_id"]+"/"+arm,arm=arm,payload_sha256=digest(payload),
                max_tokens=CAP,original_payload_exact=arm=="O0",only_actual_original_block_deleted=arm=="O1",
                instructions_examples_history_current_reader_unchanged=True,scope_instruction_retained=True,
                raw_Current_evaluation_target_unchanged=True,source_sha256=item["source_sha256"]))
    assert len(result)==16 and len({r["payload_sha256"]for r in result})==16
    return result


def prepare():
    if PROTOCOL.exists():return dry_run()
    assert not(OUT/"checkpoint.sqlite3").exists()and not(HERE/"RUN_AUTHORIZATION.json").exists()
    items,proofs,decomposition,public=assemble()
    jsonl(INPUTS,items);jsonl(ORDER,tasks(items));jsonl(HERE/"CONTRACTS16.frozen.jsonl",contracts(items))
    jsonl(HERE/"SOURCE_PROOF8.frozen.jsonl",proofs);jsonl(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl",decomposition)
    atomic_json(HERE/"PUBLIC_PAYLOAD_PROOF.json",public)
    atomic_json(HERE/"GENERATION.frozen.json",read(PACKAGE/"config.json")["generation"])
    return dict(state="offline_prepared_not_authorized",planned_cases=len(items),planned_conditions=len(tasks(items)),new_API_GPU=0)


def check():
    value=read(PROTOCOL)
    for path,expected in {**value["pins"],**value["protected_holds"]}.items():assert sha(ROOT/path)==expected,path
    assert read(PLAN/"PLAN.json")["execution_approval"]["approved"]is False
    return value


def dry_run():
    check()
    items,proofs,decomposition,public=assemble()
    assert rows(INPUTS)==items and rows(ORDER)==tasks(items)
    assert rows(HERE/"CONTRACTS16.frozen.jsonl")==contracts(items)
    assert rows(HERE/"SOURCE_PROOF8.frozen.jsonl")==proofs and rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")==decomposition
    assert read(HERE/"PUBLIC_PAYLOAD_PROOF.json")==public
    assert read(HERE/"GENERATION.frozen.json")==read(PACKAGE/"config.json")["generation"]
    result=dict(passed=True,planned_conditions=16,source_and_message_byte_contracts_exact=True,
        stage_only_no_formatter_retrieval_goal_final_or_scorer=True,history_seed=None,new_API_GPU=0,
        checkpoint_created=False,protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT/"DRYRUN.json",result)
    return result
