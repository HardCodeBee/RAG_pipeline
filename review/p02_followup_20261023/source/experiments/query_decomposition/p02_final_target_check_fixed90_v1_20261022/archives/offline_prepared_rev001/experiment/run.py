"""Frozen-upstream full-RAG final-output contrast. Prepare/dry-run never submit."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import render_final
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE,
)

PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
BASE = HERE.parent / "p02_noformatter_quick90_v1_20261021"
CONTROL = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_frozen_no_refs_quick90_v1_20261021/control90.jsonl"
BINDINGS = BASE / "SOURCE_BINDINGS.frozen.jsonl"
REFERENCES = BASE / "references90.local.frozen.jsonl"
REPLAY = PACKAGE / "data/replay.sqlite3"
RUNTIME = HERE.parent / "p02_prompt_binding_probe_v1_20261022/shared_runtime.py"
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PRIOR_PUBLIC = PLAN / "reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
INPUTS = HERE / "INPUTS90.frozen.jsonl"
ORDER = HERE / "TASK_ORDER180.frozen.jsonl"
CONTRACTS = HERE / "CONTRACTS180.frozen.jsonl"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
ARMS = ("W0", "W1")
DATASETS = ("hotpotqa", "2wiki", "musique")
SEED = 20261022
SENTENCE = "Before answering, check that your answer matches the entity, attribute or relation asked in the Original question and applies its requested comparison or combination, without adding conditions that the question does not state."
DELIMITER = "\n\nWorked examples:"
SCORERS = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf8")


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def variant(prompt, arm):
    assert prompt.count(DELIMITER) == 1
    assert SENTENCE not in prompt
    if arm == "W0":
        return prompt
    if arm != "W1":
        raise ValueError(arm)
    prefix, remainder = prompt.split(DELIMITER)
    return prefix + "\n" + SENTENCE + DELIMITER + remainder


def arm_order(case_id):
    rotate = int(hashlib.sha256(f"{SEED}|{case_id}|final_target_order".encode()).hexdigest(), 16) % 2
    return ARMS[rotate:] + ARMS[:rotate]


def usage_ok(result):
    u = result.get("usage") or {}
    return (all(type(u.get(k)) is int and u[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"])


def legacy_success(result, payload):
    # Old receipts lack requested_model; bind it through the exact source payload.
    return (result.get("status") == "success" and result.get("finish_reason") == "stop"
            and result.get("returned_model") == payload["model"] and usage_ok(result)
            and isinstance(result.get("answer"), str) and bool(result["answer"].strip())
            and not result.get("refusal"))


def receipt_integrity(result, model):
    return (result.get("status") == "success" and usage_ok(result)
            and result.get("requested_model") == result.get("returned_model") == model)


def algorithm_failure(result, model):
    if not receipt_integrity(result, model):
        return None
    if result.get("finish_reason") in ("length", "content_filter"):
        return "terminal_" + result["finish_reason"]
    if result.get("refusal"):
        return "terminal_refusal"
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        return "terminal_empty"
    return None


def cost(value):
    u = value.get("usage") or {}
    return u.get("prompt_tokens", 0) * INPUT_RATE + u.get("completion_tokens", 0) * OUTPUT_RATE


def public_index():
    prior = read(PRIOR_PUBLIC)
    raw = MANIFEST.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert blob == prior["manifest_git_blob_sha"]
    assert prior["remote_git_blobs_live_verified_by_related_prompt_mechanisms_agent"] is True
    data = json.loads(raw)
    index = {r["source"].replace("\\", "/"): {**r, "container": r.get("bundle")} for r in data["source_files"]}
    for archive in data["archives"]:
        for member in archive.get("members", []):
            index[member["source"].replace("\\", "/")] = {**member, "container": archive["bundle"]}
    return index, dict(public_repository=prior["public_repository"], verified_public_commit=prior["verified_public_commit"],
                      public_manifest=prior["public_manifest"], manifest_git_blob_sha=blob,
                      verification="Offline match to previously live-verified public manifest; no new remote fetch",
                      credentials_gold_diagnostic_labels_global_ledger_excluded=True)


def build_inputs():
    """Only recorded inference inputs/receipts. No reference/gold loader here."""
    controls = rows(CONTROL)
    bindings = rows(BINDINGS)
    assert len(controls) == len(bindings) == 90
    assert [r["id"] for r in controls] == [r["id"] for r in bindings]
    assert Counter(r["dataset"] for r in controls) == {d: 30 for d in DATASETS}
    config = read(PACKAGE / "config.json")
    examples = read(PACKAGE / "prompts/final_examples.json")
    public, proof = public_index()
    inputs, sourceproof = [], []
    with closing(sqlite3.connect(REPLAY.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        for c, b in zip(controls, bindings):
            sourcepath = b["source"].replace("\\", "/")
            assert sha(ROOT / sourcepath) == b["source_sha256"] == public[sourcepath]["sha256"]
            source = read(ROOT / sourcepath)
            t = c["trajectory"]
            assert source["id"] == t["id"] == c["id"]
            prompt, context, helper = render_final(t["question"], t["nodes"], t["original_retrieval"], config["readout"], examples)
            assert prompt == t["final_prompt"] == source["final_prompt"]
            assert context == t["final_context"] == source["final_context"] and helper == t["helper"]
            payload = build_payload(config["generation"], prompt, 64)
            match = db.execute("SELECT payload,request_id,result FROM requests WHERE payload=?", (canonical(payload),)).fetchone()
            assert match and match[1] == b["final_request_id"] == source["request_id"] == t["request_id"]
            result = json.loads(match[2])
            assert legacy_success(result, payload)
            assert result["answer"].strip() == source["prediction"] == t["prediction"] == c["prediction"]
            final = [r for r in c["calls"] if r["stage"] == "legacy/final"]
            assert len(final) == 1 and final[0]["request_id"] == match[1] and final[0]["usage"] == result["usage"]
            prefix = [r for r in c["calls"] if r["stage"] != "legacy/final"]
            unique = {r["request_id"]: r for r in prefix}
            assert len(unique) == len(prefix), "Ambiguous duplicated upstream usage"
            for r in prefix:
                assert usage_ok(r)
            pu = {k: sum(r["usage"][k] for r in prefix) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            item = dict(id=c["id"], dataset=c["dataset"], qid=t["qid"], source=sourcepath, source_sha256=b["source_sha256"],
                        historical_final_request_id=match[1], historical_final_payload=payload, historical_final_result=result,
                        upstream_usage=pu, upstream_request_ids=[r["request_id"] for r in prefix],
                        upstream_state_sha256=digest({k: t[k] for k in ("question", "supporting_questions", "nodes", "history_all", "helper", "original_retrieval", "final_context")}))
            inputs.append(item)
            sourceproof.append(dict(id=c["id"], source=sourcepath, source_sha256=b["source_sha256"], public_container=public[sourcepath]["container"],
                                    historical_final_request_id=match[1], payload_sha256=digest(payload),
                                    exact_full_wire_payload=True, legacy_requested_model_bound_by_payload=payload["model"], returned_model_matches=True,
                                    complete_stop_usage=True, rendered_prompt_context_helper_exact=True,
                                    upstream_state_sha256=item["upstream_state_sha256"]))
    return inputs, sourceproof, {**proof, "source_cases_checked": 90, "source_cases": sourceproof}


def contracts(items):
    config = read(PACKAGE / "config.json")
    result = []
    for item in items:
        old = item["historical_final_payload"]
        for arm in ARMS:
            new = build_payload(config["generation"], variant(old["messages"][0]["content"], arm), 64)
            assert {k: v for k, v in old.items() if k != "messages"} == {k: v for k, v in new.items() if k != "messages"}
            assert len(old["messages"]) == len(new["messages"]) == 1
            if arm == "W0":
                assert new == old
            else:
                assert new["messages"][0]["content"].replace("\n" + SENTENCE + DELIMITER, DELIMITER, 1) == old["messages"][0]["content"]
            result.append(dict(id=item["id"], dataset=item["dataset"], arm=arm, payload_sha256=digest(new),
                               upstream_state_sha256=item["upstream_state_sha256"], max_tokens=64,
                               original_payload_exact=(arm == "W0"), only_confirmed_single_sentence=(arm == "W1")))
    assert len(result) == len({r["payload_sha256"] for r in result}) == 180
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        return dry_run()
    items, sourceproof, publicproof = build_inputs()
    jsonl(INPUTS, items)
    jsonl(HERE / "SOURCE_PROOF90.frozen.jsonl", sourceproof)
    atomic_json(HERE / "PUBLIC_PAYLOAD_PROOF.json", publicproof)
    jsonl(CONTRACTS, contracts(items))
    tasks = [dict(index=i * 2 + j, task_id=r["id"] + "/" + arm, id=r["id"], dataset=r["dataset"], arm=arm)
             for i, r in enumerate(items) for j, arm in enumerate(arm_order(r["id"]))]
    jsonl(ORDER, tasks)
    (HERE / "W1_SENTENCE.frozen.txt").write_text(SENTENCE + "\n", encoding="utf8")
    first = items[0]["historical_final_payload"]["messages"][0]["content"]
    for arm in ARMS:
        (HERE / (arm + "_EXAMPLE_PROMPT.txt")).write_text(variant(first, arm), encoding="utf8")
    paths = [Path(__file__), HERE / "__init__.py", HERE / "PROTOCOL.md", INPUTS, ORDER, CONTRACTS,
             HERE / "SOURCE_PROOF90.frozen.jsonl", HERE / "PUBLIC_PAYLOAD_PROOF.json", HERE / "W1_SENTENCE.frozen.txt",
             HERE / "W0_EXAMPLE_PROMPT.txt", HERE / "W1_EXAMPLE_PROMPT.txt", RUNTIME, CONTROL, BINDINGS, REFERENCES,
             MANIFEST, PRIOR_PUBLIC, PACKAGE / "config.json", PACKAGE / "core/llm.py", PACKAGE / "core/cache.py",
             PACKAGE / "core/pipeline.py", PACKAGE / "core/runtime.py", PACKAGE / "core/final_prompts.py",
             PACKAGE / "prompts/final_examples.json", ROOT / "scripts/cli_support.py",
             HERE.parent / "p02_formal1500_v1_20261021/network.py",
             HERE.parent / "comprehensive_eval_v1_20261015/scoring.py"]
    pins = {relative(p): sha(p) for p in paths}
    pins.update({r["source"]: r["source_sha256"] for r in items})
    config = read(PACKAGE / "config.json")
    protocol = dict(id=HERE.name, prepared_utc=utc(), state="offline_prepared_not_run", run_authorized=False,
        cases=90, cases_per_dataset=30, conditions=180, arms=list(ARMS), sentence=SENTENCE, output_cap=64,
        inference_scope="Final only, all original actual R02/v12 decomposition and upstream QA/goal/retrieval frozen; full-RAG final quality endpoint",
        control="Fresh W0 exact original payload, isolated new checkpoint, StrictAsk seed=None; resume same checkpoint only",
        treatment="W1 appends only confirmed sentence at instruction prefix end before Worked examples",
        primary="Native final Answer F1/EM W1 vs fresh W0, fixed90 paired; historical .7668887868810469/.6666666666666666 independent reference",
        development_limit="Repeatedly exposed fixed90, no independent confirmatory or official-test claim",
        task_order="Original source case order; SHA256(20261022|case_id|final_target_order) modulo2 swaps within-case arm order",
        request_policy=dict(maximum_actual_attempts=5, durable_across_restarts=True, SDK_retries=0, HTTP_retries=0,
                            serialized_shared_gate=True, minimum_start_interval_seconds=3, no_historical_seed=True,
                            unknown_no_attributable_receipt="Existing bounded recovery preserves retired reservations",
                            unknown_returned_receipt="Hold without blind resubmission"),
        quality_policy="Valid identity/complete integer usage length,refusal,empty,filter terminal=0. Missingusage/modelmismatch/technical exhausted held separately. No full F1 until90 resolved per arm",
        tokens_cost="Original upstream unique logical usage plus new final; fresh actual final ledger and unresolved reservations separate; no LLM call metric or timing claim",
        expense_alerts="Existing single stage3x or recent20 stage1.5x paired historical stage policy; no budget cap",
        model=config["generation"]["model"], endpoint=config["generation"]["base_url"],
        scoring="Gold only loaded by report; instantiate official/native metric once per dataset per report",
        main_evaluation_remains_paused=True, provider_submissions_at_prepare=0, pins=pins)
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, "Frozen file changed: " + path
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def dry_run():
    check()
    items, proofs, publicproof = build_inputs()
    assert items == rows(INPUTS)
    assert proofs == rows(HERE / "SOURCE_PROOF90.frozen.jsonl")
    assert publicproof == read(HERE / "PUBLIC_PAYLOAD_PROOF.json")
    assert contracts(items) == rows(CONTRACTS)
    tasks = rows(ORDER)
    assert len(tasks) == 180 and len({r["task_id"] for r in tasks}) == 180
    for item in items:
        assert [t["arm"] for t in tasks if t["id"] == item["id"]] == list(arm_order(item["id"]))
    assert set(t["task_id"] for t in tasks) == {i["id"] + "/" + arm for i in items for arm in ARMS}
    value = dict(utc=utc(), passed=True, source_cases=90, source_public_hashes=90, legacy_stop_usage=90,
                 exact_W0_full_payloads=90, exact_current_renderer_reproductions=90, single_sentence_W1_contracts=90,
                 unique_condition_payloads=180, cases_per_dataset=dict(Counter(r["dataset"] for r in items)),
                 first_arm_counts=dict(Counter(arm_order(i["id"])[0] for i in items)),
                 historical_requested_model_from_exact_payload_not_nonexistent_receipt_field=True,
                 no_gold_loaded=True, no_client_created=True, no_provider_requests=0, no_checkpoint_created=True,
                 run_authorization_present=(HERE / "RUN_AUTHORIZATION.json").exists(), main_evaluation_remains_paused=True,
                 protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT / "DRYRUN.json", value)
    return value


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def checkpoint_identity(cache):
    identity = dict(protocol_sha256=sha(PROTOCOL), inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER))
    old = cache.get("meta", "identity")
    if old is not None:
        assert old == identity, "Checkpoint cannot be reused with changed experiment identity"
    else:
        assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0
        assert cache.all("outcome") == {}
        cache.put("meta", "identity", identity)


def report(cache=None):
    """Reporting opens existing checkpoints read-only; never alters request states."""
    check()
    if cache is None and (OUT / "checkpoint.sqlite3").exists():
        with closing(sqlite3.connect((OUT / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            outcomes = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    elif cache is not None:
        outcomes = list(cache.all("outcome").values())
    else:
        outcomes = []
    items = {r["id"]: r for r in rows(INPUTS)}
    expected_tasks = {r["task_id"] for r in rows(ORDER)}
    assert len({r["task_id"] for r in outcomes}) == len(outcomes) and all(r["task_id"] in expected_tasks for r in outcomes)
    refs = {r["id"]: r for r in rows(REFERENCES)}
    global SCORERS
    if SCORERS is None:
        from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
        SCORERS = {ds: metric(ds) for ds in DATASETS}  # Once per dataset/process.
    scorers = SCORERS
    details = []
    for r in outcomes:
        assert r["status"] in ("success", "algorithm_failure")
        s = scorers[r["dataset"]](r["answer"], refs[r["id"]]) if r["status"] == "success" else dict(f1=0.0, em=0.0)
        prefix = items[r["id"]]["upstream_usage"]
        flow = {k: prefix[k] + r["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        details.append({**r, **s, "full_flow_logical_usage": flow,
                        "full_flow_logical_estimated_CNY": cost({"usage": flow}), "new_final_returned_usage_CNY": cost(r)})
    index = {(r["id"], r["arm"]): r for r in details}
    def summarize(arm, ds=None):
        subset = [r for r in details if r["arm"] == arm and (ds is None or r["dataset"] == ds)]
        n = 90 if ds is None else 30
        missing = n - len(subset)
        return dict(planned=n, quality_terminal=len(subset), missing_or_held=missing,
                    algorithm_failures=sum(r["status"] == "algorithm_failure" for r in subset),
                    F1=sum(r["f1"] for r in subset)/n if missing == 0 else None,
                    EM=sum(r["em"] for r in subset)/n if missing == 0 else None,
                    F1_fixed_denominator_bounds=[sum(r["f1"] for r in subset)/n, (sum(r["f1"] for r in subset)+missing)/n],
                    EM_fixed_denominator_bounds=[sum(r["em"] for r in subset)/n, (sum(r["em"] for r in subset)+missing)/n],
                    full_flow_logical_prompt_tokens=sum(r["full_flow_logical_usage"]["prompt_tokens"] for r in subset),
                    full_flow_logical_completion_tokens=sum(r["full_flow_logical_usage"]["completion_tokens"] for r in subset),
                    full_flow_logical_estimated_CNY=sum(r["full_flow_logical_estimated_CNY"] for r in subset),
                    new_final_returned_usage_CNY=sum(r["new_final_returned_usage_CNY"] for r in subset))
    paired_ids = [i for i in items if (i, "W0") in index and (i, "W1") in index]
    pairs = [dict(id=i, dataset=items[i]["dataset"], W0=index[i,"W0"]["f1"], W1=index[i,"W1"]["f1"],
                  delta=index[i,"W1"]["f1"]-index[i,"W0"]["f1"]) for i in paired_ids]
    historic = [scorers[i["dataset"]](i["historical_final_result"]["answer"].strip(), refs[i["id"]]) for i in items.values()]
    historical = dict(F1=sum(s["f1"] for s in historic)/90, EM=sum(s["em"] for s in historic)/90, independent_reference_only=True)
    assert abs(historical["F1"] - .7668887868810469) < 1e-12 and abs(historical["EM"] - 2/3) < 1e-12
    heldfiles = [read(p) for p in (OUT / "held").glob("*.json")] if (OUT / "held").exists() else []
    held = [r for r in heldfiles if r["task_id"] not in {o["task_id"] for o in outcomes}]
    value = dict(utc=utc(), status="completed" if len(outcomes)==180 else "partial", scope="Frozen-upstream full-RAG final-output development contrast",
                 fixed_cases=90, fixed_conditions=180, quality_terminal=len(outcomes), held_technical_tasks=len(held),
                 held_task_ids=[r["task_id"] for r in held], arms={a:summarize(a) for a in ARMS},
                 by_dataset={d:{a:summarize(a,d) for a in ARMS} for d in DATASETS}, historical_original_P02=historical,
                 paired=dict(complete_pairs=len(pairs), delta_F1=sum(p["delta"] for p in pairs)/90 if len(pairs)==90 else None,
                             improved=sum(p["delta"]>0 for p in pairs), tied=sum(p["delta"]==0 for p in pairs), harmed=sum(p["delta"]<0 for p in pairs),
                             counts_scope="Resolved pairs only until complete90; never a complete-score subset estimate"),
                 gold_only_loaded_in_report=True, scorers_initialized_once_per_dataset=3, no_LLM_call_count_metric=True,
                 no_complete_flow_timing_claim=True, main_evaluation_remains_paused=True)
    ledgerpath = Path(read(PACKAGE / "config.json")["budget"]["ledger"])
    with closing(sqlite3.connect(ledgerpath.resolve().as_uri()+"?mode=ro", uri=True)) as db:
        actual, reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", ("p02_prompt_research_v1_20261022/"+str(OUT.resolve()),)).fetchone()
    value.update(new_experiment_actual_returned_usage_CNY=actual, unresolved_reserved_CNY=reserve)
    jsonl(OUT / "DETAILS.jsonl", details)
    jsonl(OUT / "PAIRED.jsonl", pairs)
    atomic_json(OUT / "REPORT.json", value)
    return value


def run(limit=None):
    check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert authority.get("protocol_sha256") == sha(PROTOCOL)
    assert any(r.get("id") == authority.get("decision_id") for r in rows(PLAN / "DECISIONS.jsonl"))
    config = read(PACKAGE / "config.json")
    items = {r["id"]:r for r in rows(INPUTS)}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self, name, **fields): event(dict(event=name, **fields))
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config, OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
        checkpoint_identity(cache)
        stop = lambda: (OUT / "STOP_REQUESTED").exists()
        recent = [(cost(items[r["id"]]["historical_final_result"]), cost(r))
                  for r in sorted(cache.all("outcome").values(),key=lambda r:r["utc"]) if r.get("usage")]
        with closing(StrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as ask:
            processed = 0
            for task in rows(ORDER):
                key = task["task_id"]
                if cache.get("outcome", key) is not None:
                    continue
                if stop() or (limit is not None and processed >= limit):
                    return report(cache)
                processed += 1
                item = items[task["id"]]
                prompt = variant(item["historical_final_payload"]["messages"][0]["content"], task["arm"])
                base = {**task, "historical_final_request_id":item["historical_final_request_id"], "upstream_state_sha256":item["upstream_state_sha256"], "fresh_experiment":True}
                try:
                    rid, answer, finish = ask("final_target/"+key, prompt, 64)
                    result = json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?",(rid,)).fetchone()[0])
                    if not receipt_integrity(result, config["generation"]["model"]):
                        raise RuntimeError("Held receipt: matching model and complete integer usage required")
                    outcome = {**base, "status":"success", "answer":answer, "request_id":rid, "finish_reason":finish, "usage":result["usage"], "utc":utc()}
                except (RuntimeError, InterruptedError) as exc:
                    payload = canonical(build_payload(config["generation"], prompt, 64))
                    row = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?",(payload,)).fetchone()
                    result = {**(json.loads(row[2]) if row and row[2] else {}), "status":row[1] if row else None}
                    reason = algorithm_failure(result, config["generation"]["model"])
                    if reason:
                        outcome = {**base, "status":"algorithm_failure", "answer":result.get("answer", ""), "request_id":row[0],
                                   "finish_reason":result.get("finish_reason"), "usage":result["usage"], "reason":reason, "utc":utc()}
                        event(dict(event="terminal_algorithm_outcome",task_id=key,reason=reason))
                    else:
                        hold = {**base, "status":"held_technical", "reason":str(exc), "request_id":row[0] if row else None, "utc":utc()}
                        atomic_json(OUT / "held" / (digest(key)+".json"),hold)
                        event(dict(event="held_technical", task_id=key, reason=str(exc)))
                        continue
                cache.put("outcome",key,outcome)
                reference, actual = cost(item["historical_final_result"]), cost(outcome)
                recent.append((reference,actual))
                reasons = []
                if reference and actual > 3*reference:
                    reasons.append("final_stage_above3x_paired_historical")
                if len(recent)>=20 and sum(c for _,c in recent[-20:])>1.5*sum(b for b,_ in recent[-20:]):
                    reasons.append("recent20_final_stage_mean_above1.5x_paired_historical")
                if reasons:
                    alert = dict(event="expense_alert",task_id=key,reasons=reasons,historical_final_CNY=reference,actual_final_CNY=actual,scope="Final stage cost, full query logical cost reported separately")
                    event(alert); emit(alert)
                emit(dict(event="condition_terminal", task_id=key,status=outcome["status"]))
                report(cache)
    return report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","dry-run","report","run"))
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    emit({"prepare":prepare,"dry-run":dry_run,"report":report,"run":lambda:run(args.limit)}[args.action]())


if __name__ == "__main__":
    main()
