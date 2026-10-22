"""Freeze C8 input contracts only; no execution, client or production Cache."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022 import prepare as source
from experiments.query_decomposition.p02_failure8_actual_target_probe_v1_20261022 import templates

OUT = ROOT / "outputs/query_decomposition" / HERE.name
PACKAGE, PLAN, CAP = source.PACKAGE, source.PLAN, 128
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "INPUTS8.frozen.jsonl"
CONTROLS = HERE / "FRESH_I0_RECEIPTS8.local.frozen.json"
ORDER = HERE / "NEW_PAYLOAD_ORDER5.frozen.jsonl"
read, rows, sha, digest, rel, save, jsonl = source.read, source.rows, source.sha, source.digest, source.rel, source.save, source.jsonl


def assemble():
    source.check()
    items, proofs, raw_proofs, public = source.assemble()
    assert items == rows(source.INPUTS) and len(items) == 8
    controls = source.snapshot_i0(items)
    assert controls == read(source.CONTROLS)
    # Do not edit historical proof: retain its old B0 attribution keys as provenance.
    return items, proofs, raw_proofs, public, controls


def contracts(items, controls):
    generation = read(PACKAGE / "config.json")["generation"]
    control_by_id = {r["probe_id"]: r for r in controls["controls"]}
    checks, rendered, uses, order = [], [], [], []
    for item in items:
        raw, rewrite = item["current_question"], item["frozen_rewritten_query"]
        original = item["control_prompt"]
        before, actual = original.split("\n\nActual task:\n", 1)
        head = f'Current subquestion #{item["step_id"]}:\n'
        tail = "\n\nRetrieved passages:"
        left, rest = actual.split(head, 1)
        current, right = rest.split(tail, 1)
        assert current == raw
        control = control_by_id[item["probe_id"]]
        assert control["source_payload"] == item["control_payload"]
        for arm in templates.ARMS:
            text = templates.prompt(item, arm)
            model_current = templates.model_current(item, arm)
            payload = build_payload(generation, text, CAP)
            expected = before + "\n\nActual task:\n" + left + head + model_current + tail + right
            assert text == expected
            assert text.split("\n\nActual task:\n", 1)[0] == before
            assert {k: v for k, v in payload.items() if k != "messages"} == {k: v for k, v in item["control_payload"].items() if k != "messages"}
            assert payload["messages"] == [{"role": "user", "content": text}]
            assert templates.evaluation_target(item) == raw
            reuse = arm == "C0" or rewrite == raw
            if reuse:
                assert text == original and payload == control["source_payload"]
            else:
                # Reverse exactly the one value replacement; every other byte survives.
                newer_actual = text.split("\n\nActual task:\n", 1)[1]
                assert newer_actual == left + head + rewrite + tail + right
                assert before + "\n\nActual task:\n" + left + head + raw + tail + right == original
            value = dict(probe_id=item["probe_id"], arm=arm, payload_sha256=digest(payload),
                prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), max_tokens=CAP,
                original_raw_current_evaluation_target=raw, actual_model_current=model_current,
                changed_actual_task_value_only=(arm == "C1" and rewrite != raw),
                original_question_history_reader_examples_raw_D_unchanged=True,
                reuse_already_paid_I0=reuse, interface_group=item["interface_group"], mechanism_group=item["mechanism_group"])
            checks.append(value)
            rendered.append({**value, "payload": payload})
            uses.append(dict(task_id=item["probe_id"] + "/" + arm, probe_id=item["probe_id"], arm=arm,
                reuse_already_paid_I0=reuse,
                source_request_id=control["source_request_id"] if reuse else None,
                new_payload=not reuse, evaluation_target=raw, model_current=model_current))
            if not reuse:
                order.append(dict(index=len(order), task_id=item["probe_id"] + "/" + arm,
                    probe_id=item["probe_id"], arm=arm, payload_sha256=digest(payload)))
    assert len(checks) == len(rendered) == len(uses) == 16
    assert len(order) == 5 and len({r["payload_sha256"] for r in order}) == 5
    assert len({r["payload_sha256"] for r in checks}) == 13
    assert sum(r["reuse_already_paid_I0"] for r in uses) == 11
    assert len({r["source_request_id"] for r in uses if r["reuse_already_paid_I0"]}) == 8
    assert not ({r["payload_sha256"] for r in order} & {digest(c["source_payload"]) for c in controls["controls"]})
    return checks, rendered, uses, order


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    assert not (HERE / "RUN_AUTHORIZATION.json").exists() and not (OUT / "checkpoint.sqlite3").exists()
    items, proofs, raw_proofs, public, controls = assemble()
    jsonl(INPUTS, items)
    save(CONTROLS, controls)
    jsonl(HERE / "SOURCE_PROOF8.frozen.jsonl", proofs)
    jsonl(HERE / "DECOMPOSITION_PROOF8.frozen.jsonl", raw_proofs)
    save(HERE / "PUBLIC_PAYLOAD_PROOF.json", public)
    save(HERE / "GENERATION.frozen.json", read(PACKAGE / "config.json")["generation"])
    checks, rendered, uses, order = contracts(items, controls)
    jsonl(HERE / "CONTRACTS16.frozen.jsonl", checks)
    jsonl(HERE / "RENDERED_PAYLOADS16.frozen.jsonl", rendered)
    jsonl(HERE / "LOGICAL_USES16.frozen.jsonl", uses)
    jsonl(ORDER, order)
    pins = {rel(p): sha(p) for p in HERE.iterdir() if p.is_file() and p != PROTOCOL}
    # Transitive frozen B/I source pins are retained without editing those files.
    pins.update(read(source.PROTOCOL)["pins"])
    pins[rel(source.PROTOCOL)] = sha(source.PROTOCOL)
    save(PROTOCOL, dict(id=HERE.name, prepared_utc=utc(), state="offline_prepared_no_execution_entry",
        nodes=8, logical_conditions=16, arms=list(templates.ARMS), cap=CAP,
        source_decomposition="Exact source R02/v12 actual arrays from original I8 proof; no manual60 replacement or new training",
        groups=dict(local_QA_failure=6, reasonable_abstention_boundary=2),
        selection="All original I8 nodes in original order, no post-output reselection",
        C0="Exact already-paid fresh I0 wire payload and result; never a new request",
        C1="Replace only the actual-task Current subquestion value by same-source frozen rewritten_query",
        unchanged="All other prompt bytes, original question, raw D, history, reader, examples, model/settings and128cap; ordinary only",
        evaluation_target="Original raw current is unchanged for both arms, separately from model_current; off-target rewrite harms count",
        reuse=dict(source_unique_paid_receipts=8, C0_logical=8, C1_identical_logical=3, source_logical_total=11,
            new_distinct_payloads=5, new_logical_conditions=5, new_C0_cost_CNY=0,
            historical_controls_not_new_synchronous_draws=True),
        stop_rule="If actual-task replacement fails, stop this entity-prompt branch without more same-case tuning",
        interpretation="Selected exposed mechanism challenge, not overall accuracy, independent confirmation or full-RAG F1",
        controls_sha256=sha(CONTROLS), authorization="No runner/RUN_AUTH/client/productionCache; root must review a separate execution entry before API",
        future_runtime="New independent checkpoint for five new payloads only; immutable B CandidateAsk guards/gate/ledger/max5; strict model+all3 integer usage; algorithm terminals no reroll, technical held separate",
        main_evaluation_remains_paused=True, new_provider_requests_at_prepare=0, pins=pins))
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    assert not (HERE / "RUN_AUTHORIZATION.json").exists()
    assert not (HERE / "run.py").exists() and not (OUT / "checkpoint.sqlite3").exists()
    return protocol


def dry_run():
    protocol = check()
    items, proofs, raw_proofs, public, controls = assemble()
    assert items == rows(INPUTS) and controls == read(CONTROLS)
    assert proofs == rows(HERE / "SOURCE_PROOF8.frozen.jsonl")
    assert raw_proofs == rows(HERE / "DECOMPOSITION_PROOF8.frozen.jsonl")
    assert public == read(HERE / "PUBLIC_PAYLOAD_PROOF.json")
    checks, rendered, uses, order = contracts(items, controls)
    for path, expected in [("CONTRACTS16.frozen.jsonl", checks), ("RENDERED_PAYLOADS16.frozen.jsonl", rendered),
                           ("LOGICAL_USES16.frozen.jsonl", uses), (ORDER.name, order)]:
        assert rows(HERE / path) == expected
    result = dict(passed=True, nodes=8, logical_conditions=16, contracts_exact=16,
        original_raw_current_evaluation_targets_fixed=16, actual_current_value_replacements=5,
        source_unique_paid_I0_receipts=8, reuse_logical_uses=11, C1_identical_I0_reuse=3,
        new_distinct_payloads=5, source_public_hashes_exact=8, raw_R02_proofs_exact=8,
        all_source_receipts_matching_model_stop_three_integer_usage=True,
        no_goal_formatter_final_retrieval_or_decomposition_run=True, no_client_created=True,
        no_production_Cache_created=True, new_API_requests=0, main_evaluation_remains_paused=True,
        no_run_authorization=True, no_execution_entry=True, protocol_sha256=sha(PROTOCOL),
        controls_sha256=sha(CONTROLS), pins_verified=len(protocol["pins"]))
    save(OUT / "DRYRUN.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == "prepare" else dry_run(), ensure_ascii=True))
