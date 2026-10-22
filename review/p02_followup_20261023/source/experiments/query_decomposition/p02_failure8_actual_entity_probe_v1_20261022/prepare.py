"""Offline snapshot/render contracts; no provider clients or Cache instances."""
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
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import prepare as source, run as source_runner
from experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022 import templates as reviewed_templates
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022 import templates
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.runtime import normal_complete

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE = source.PACKAGE
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "INPUTS8.frozen.jsonl"
CONTROLS = HERE / "B0_FRESH_I0_RECEIPTS8.local.frozen.json"
ORDER = HERE / "TASK_ORDER16.frozen.jsonl"
CAP, SEED = 128, 20261022
read, rows, sha, digest, rel, save, jsonl = source.read, source.rows, source.sha, source.digest, source.rel, source.save, source.jsonl


def arm_order(case_id):
    # Identical ordering rule to reviewed R15; only the arm names change.
    offset = int(hashlib.sha256(f"{SEED}|{case_id}|actual_child_binding_order".encode()).hexdigest(), 16) % 2
    return ("B1", "B2")[offset:] + ("B1", "B2")[:offset]


def assemble():
    source_runner.check()
    originals, proofs, _, raw_proofs, public = source.assemble()
    assert originals == rows(source.INPUTS) and len(originals) == 8
    assert proofs == rows(source.HERE / "SOURCE_PROOF8.frozen.jsonl")
    assert raw_proofs == rows(source.HERE / "DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public == read(source.HERE / "PUBLIC_PAYLOAD_PROOF.json")
    selection = rows(source.HERE / "SELECTION8.frozen.jsonl")
    assert [r["id"] + "/ordinary/" + str(r["step_id"]) for r in selection] == [r["probe_id"] for r in originals]
    items = []
    for original, selected in zip(originals, selection):
        frozen_source = read(ROOT / original["source"])
        node = next(n for n in frozen_source["nodes"] if n["step_id"] == original["step_id"])
        assert node["role"] == "ordinary" and node["child_prompt"] == original["control_prompt"]
        rewrite = node["rewritten_query"]
        assert isinstance(rewrite, str) and rewrite.strip()
        items.append({**original, "frozen_rewritten_query": rewrite,
            "interface_group": "identical_candidate_control" if rewrite == original["current_question"] else "rewrite_text_differs",
            "mechanism_group": "local_QA_failure" if selected["eligible_for_same_reader_qa_challenge"] else "reasonable_abstention_boundary"})
    assert sum(r["interface_group"] == "rewrite_text_differs" for r in items) == 5
    return items, proofs, raw_proofs, public


def snapshot_i0(items):
    """One read-only source transaction: all eight exact fresh I0 receipts or fail."""
    config = read(PACKAGE / "config.json")
    path = source.OUT / "checkpoint.sqlite3"
    controls = []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db, \
            closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as ledger:
        db.execute("BEGIN")
        ledger.execute("BEGIN")
        for item in items:
            key = item["probe_id"] + "/I0"
            row = db.execute("SELECT value FROM data WHERE kind='outcome' AND key=?", (key,)).fetchone()
            assert row, "Missing fresh I0 blocks all B0 freezing; no fallback: " + key
            outcome = json.loads(row[0])
            assert outcome["arm"] == "I0" and outcome["task_id"] == key and outcome["fresh_experiment"] is True
            assert outcome["probe_id"] == item["probe_id"] and outcome["status"] == "success"
            result_row = db.execute("SELECT id,payload,status,result FROM requests WHERE id=?", (outcome["request_id"],)).fetchone()
            assert result_row and result_row[2] == "success"
            rid, payload, status, raw = result_row
            result = {**json.loads(raw), "status": status}
            assert payload == canonical(item["control_payload"])
            assert normal_complete(result, config["generation"]["model"])
            assert result["answer"].strip() == outcome["answer"] and result["usage"] == outcome["usage"]
            assert outcome["finish_reason"] == result["finish_reason"] == "stop"
            charge = ledger.execute("SELECT namespace,reserved,actual,status FROM charges WHERE id=?", (rid,)).fetchone()
            assert charge and charge[0] == "p02_prompt_research_v1_20261022/" + str(source.OUT.resolve())
            assert charge[2] is not None and charge[3] == "returned_usage"
            controls.append(dict(probe_id=item["probe_id"], source_task_id=key, source_request_id=rid,
                source_database=rel(path), source_payload=json.loads(payload), payload_sha256=digest(json.loads(payload)),
                source_outcome=outcome, outcome_sha256=digest(outcome), result=result, receipt_sha256=digest(result),
                source_charge=dict(namespace=charge[0], reserved=charge[1], actual_CNY=charge[2], status=charge[3]),
                source_charge_already_recorded=True, B0_new_actual_CNY=0))
    return dict(source_experiment=source.HERE.name, source_execution_protocol_sha256=sha(source_runner.EXECUTION),
        source_inputs_sha256=sha(source.INPUTS), count=8, controls=controls, no_I1_controls=True,
        exact_full_payload_model_stop_all3_usage_required=True, no_B0_submission=True, new_provider_requests=0)


def contracts(items):
    generation = read(PACKAGE / "config.json")["generation"]
    checks, rendered = [], []
    for item in items:
        original = item["control_prompt"]
        for arm in templates.ARMS:
            prompt = templates.prompt(item, arm)
            payload = build_payload(generation, prompt, CAP)
            reviewed_arm = {"B0": "R0", "B1": "R1", "B2": "R2"}[arm]
            assert prompt == reviewed_templates.prompt(item, reviewed_arm)
            assert {k:v for k,v in payload.items() if k != "messages"} == {k:v for k,v in item["control_payload"].items() if k != "messages"}
            if arm == "B0":
                assert prompt == original and payload == item["control_payload"]
            else:
                candidate = item["current_question"] if arm == "B1" else item["frozen_rewritten_query"]
                addition = "\n\n" + templates.block(candidate)
                assert prompt.count(addition) == 1 and prompt.replace(addition, "", 1) == original
                assert f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\n' in prompt.split("\n\nActual task:\n",1)[1]
                assert prompt.split("\n\nActual task:\n",1)[0] == original.split("\n\nActual task:\n",1)[0]
            value = dict(probe_id=item["probe_id"], arm=arm, payload_sha256=digest(payload),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), max_tokens=CAP,
                raw_current_authoritative_and_unchanged=True, original_history_reader_examples_raw_D_unchanged=True,
                exact_R15_block_renderer=True, interface_group=item["interface_group"], mechanism_group=item["mechanism_group"])
            checks.append(value)
            rendered.append({**value, "payload":payload})
        a,b = templates.prompt(item,"B1"),templates.prompt(item,"B2")
        # Delete only each fixed block; original prompt is recovered in both arms.
        assert a.replace(templates.block(item["current_question"]),"",1) == b.replace(templates.block(item["frozen_rewritten_query"]),"",1)
    assert len(checks) == 24 and len({c["payload_sha256"] for c in checks}) == 21
    assert len({c["payload_sha256"] for c in checks if c["arm"] != "B0"}) == 13
    return checks, rendered


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    assert not (HERE / "RUN_AUTHORIZATION.json").exists() and not (OUT / "checkpoint.sqlite3").exists()
    items, proofs, raw_proofs, public = assemble()
    jsonl(INPUTS,items)
    save(CONTROLS,snapshot_i0(items))
    jsonl(HERE / "SOURCE_PROOF8.frozen.jsonl",proofs)
    jsonl(HERE / "DECOMPOSITION_PROOF8.frozen.jsonl",raw_proofs)
    save(HERE / "PUBLIC_PAYLOAD_PROOF.json",public)
    save(HERE / "GENERATION.frozen.json",read(PACKAGE / "config.json")["generation"])
    checks,rendered=contracts(items)
    jsonl(HERE / "CONTRACTS24.frozen.jsonl",checks)
    jsonl(HERE / "RENDERED_PAYLOADS24.frozen.jsonl",rendered)
    jsonl(ORDER,[dict(index=i*2+j,task_id=r["probe_id"]+"/"+arm,probe_id=r["probe_id"],arm=arm)
        for i,r in enumerate(items) for j,arm in enumerate(arm_order(r["id"]))])
    dependencies=[source.PROTOCOL,source_runner.EXECUTION,source.INPUTS,source.HERE/"run.py",source.HERE/"prepare.py",
        source.HERE/"SOURCE_PROOF8.frozen.jsonl",source.HERE/"DECOMPOSITION_PROOF8.frozen.jsonl",source.HERE/"PUBLIC_PAYLOAD_PROOF.json",
        source.HERE/"SELECTION8.frozen.jsonl",HERE.parent/"p02_actual_child_binding_probe_v1_20261022/templates.py",
        HERE.parent/"p02_prompt_binding_probe_v1_20261022/shared_runtime.py",PACKAGE/"config.json",PACKAGE/"core/cache.py",
        PACKAGE/"core/runtime.py",PACKAGE/"core/llm.py",ROOT/"scripts/cli_support.py",HERE.parent/"p02_formal1500_v1_20261021/network.py"]
    pins={rel(p):sha(p) for p in list(HERE.iterdir())+dependencies if p.is_file() and p != PROTOCOL and p.name != "RUN_AUTHORIZATION.json"}
    pins.update({r["source"]:r["source_sha256"] for r in items})
    pins.update({r["raw_file"]:r["raw_sha256"] for r in raw_proofs})
    save(PROTOCOL,dict(id=HERE.name,prepared_utc=utc(),state="offline_prepared_root_authorization_required",nodes=8,
        logical_conditions=24,arms=list(templates.ARMS),B0_reused_fresh_I0_controls=8,new_candidate_conditions=16,
        distinct_new_payloads=13,identical_B1_B2_nodes=3,rewrite_different_nodes=5,
        source_controls_sha256=sha(CONTROLS),cap=CAP,groups=dict(local_QA_failure=6,reasonable_abstention_boundary=2),
        selection="All same eight I8 nodes, original order; no gold/new-output filtering, replacement or decomposition repair",
        B0="Only exact fresh I8 I0 snapshots; no request, no fallback, no I1, no historical formal response substitution",
        B1="Append reviewed R15 block; candidate repeats original current question",
        B2="Identical block and location; only candidate changes to same-source existing rewritten_query",
        fixed="Raw current remains authoritative; original question,raw D,history,reader,examples,other instructions,model,128cap unchanged; no formatter or goal/final rerun",
        payload_reuse="Same full candidate payload shares same local receipt including algorithm terminal; seed=None; no cross-experiment candidate response seeding",
        returned_guard="I8 strict receipt and ledger guard plus non-null refusal/finish_reason-only traces hold; I8/shared files unchanged",
        recovery="Five durable actual technical attempts including initial,SDK/HTTP0,shared serial gate3sec; terminal valid-usage algorithm failures never rerolled; invalid model/usage stays held",
        costs="B0 source charge already recorded; attributed reuse separate from new ledger fees/reservations; duplicate logical uses do not dilute unique-payload expense trend",
        authorization="run requires root-created approved RUN_AUTHORIZATION with exact protocol/controls hashes and decision log entry; preparation never creates it",
        no_native_or_final_F1=True,new_provider_requests_at_prepare=0,main_evaluation_remains_paused=True,pins=pins))
    return dry_run()


def check():
    protocol=read(PROTOCOL)
    for path,expected in protocol["pins"].items():
        assert sha(ROOT/path)==expected,path
    assert read(PLAN/"PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def dry_run():
    protocol=check()
    items,proofs,raw_proofs,public=assemble()
    assert items==rows(INPUTS) and snapshot_i0(items)==read(CONTROLS)
    assert proofs==rows(HERE/"SOURCE_PROOF8.frozen.jsonl") and raw_proofs==rows(HERE/"DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public==read(HERE/"PUBLIC_PAYLOAD_PROOF.json")
    checks,rendered=contracts(items)
    assert checks==rows(HERE/"CONTRACTS24.frozen.jsonl") and rendered==rows(HERE/"RENDERED_PAYLOADS24.frozen.jsonl")
    assert len(rows(ORDER))==16 and len({r["task_id"] for r in rows(ORDER)})==16
    for item in items:
        assert [r["arm"] for r in rows(ORDER) if r["probe_id"]==item["probe_id"]]==list(arm_order(item["id"]))
    value=dict(passed=True,nodes=8,logical_conditions=24,full_B0_fresh_I0_receipts=8,matching_model_stop_all3_usage_and_source_charge=8,
        candidate_conditions=16,distinct_new_payloads=13,identical_B1_B2_nodes=3,source_public_hashes_exact=8,raw_R02_proofs_exact=8,
        pure_rendered_payloads=24,R15_exact_renderer_contracts=24,original_current_history_reader_examples_raw_D_unchanged=True,
        no_I1_prompt_or_response_imported=True,no_client_created=True,no_Cache_created=True,new_API_requests=0,
        run_authorization_present=(HERE/"RUN_AUTHORIZATION.json").exists(),main_evaluation_remains_paused=True,
        protocol_sha256=sha(PROTOCOL),controls_sha256=sha(CONTROLS),new_checkpoint_present=(OUT/"checkpoint.sqlite3").exists())
    save(OUT/"DRYRUN.json",value)
    return value


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","dry-run"))
    a=p.parse_args()
    print(json.dumps(prepare() if a.action=="prepare" else dry_run(),ensure_ascii=True))
