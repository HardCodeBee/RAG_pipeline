"""Offline prepare/check/report; only explicit authorized run submits requests.

No retrieval, decomposition, final-answer generation, scorer or gold loader.
"""
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
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, complete_receipt, atomic_json, INPUT_RATE, OUTPUT_RATE,
)

PARENT = HERE.parent / "p02_prompt_binding_probe_v1_20261022"
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
INPUTS = HERE / "INPUTS.frozen.jsonl"
ORDER = HERE / "TASK_ORDER.frozen.jsonl"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PRIOR_PROOF = PLAN / "reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
ARMS = ("E0", "E1", "E2")
SEED = 20261022
RUNTIME_SHA = "5c6c5c4914e995bce4a995ebb60ff9acb29517c5e5d77118bd2ac2413091b98e"
ORIGINAL = (
    "Current subquestion #2:\nWhat is the length of #1?\n",
    "Current subquestion #2:\nHow many species does #1 contain other than Callicoma serratifolia?\n",
)
REFERENTS = {
    "E1": ("Colorado River", "Callicoma"),
    "E2": ("the river into which Pack Creek runs after it goes through the Spanish Valley",
           "the genus that contains Callicoma serratifolia"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def variant_prompt(prompt, arm):
    """Change only two #1 tokens in worked-example current questions."""
    prefix, actual = prompt.split("\n\nActual task:\n", 1)
    assert all(prefix.count(old) == 1 for old in ORIGINAL)
    assert "Answer: Pack Creek flows into the Colorado River." in prefix
    assert "Answer: Callicoma contains Callicoma serratifolia." in prefix
    if arm == "E0":
        return prompt
    if arm not in REFERENTS:
        raise ValueError(arm)
    for old, referent in zip(ORIGINAL, REFERENTS[arm]):
        prefix = prefix.replace(old, old.replace("#1", referent), 1)
    return prefix + "\n\nActual task:\n" + actual


def arm_order(case_id):
    offset = int(hashlib.sha256(f"{SEED}|{case_id}|ordinary_fewshot_order".encode()).hexdigest(), 16) % 3
    return list(ARMS[offset:] + ARMS[:offset])


def source_proofs(items):
    prior = read(PRIOR_PROOF)
    raw = MANIFEST.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert git_blob == prior["manifest_git_blob_sha"]
    assert prior["remote_git_blobs_live_verified_by_related_prompt_mechanisms_agent"] is True
    manifest = json.loads(raw)
    index = {r["source"].replace("\\", "/"): {**r, "container": r.get("bundle")} for r in manifest["source_files"]}
    for archive in manifest["archives"]:
        for member in archive.get("members", []):
            index[member["source"].replace("\\", "/")] = {**member, "container": archive["bundle"]}
    cases = []
    for item in items:
        entry = index[item["source"].replace("\\", "/")]
        assert entry["sha256"] == item["source_sha256"] == sha(ROOT / item["source"])
        raw_case = read(ROOT / item["source"])
        assert raw_case["id"] == item["case_id"]
        node = next(n for n in raw_case["nodes"] if n["step_id"] == item["step_id"])
        assert node["role"] == item["role"] == "ordinary"
        assert raw_case["question"] == item["original_question"]
        assert node["request"] == item["current_question"]
        assert node["reader_packet"] == item["reader_packet"]
        assert node["prior_complete_answers"] == item["history"]
        assert node["child_prompt"] == item["control_prompt"]
        assert node["answer_request_id"] == item["control_source_request_id"]
        assert node["initial_answer"].strip() == item["control_result"]["answer"].strip()
        assert not node.get("sentence_format_request_id")
        cases.append(dict(probe_id=item["probe_id"], source=item["source"], source_sha256=item["source_sha256"],
                          public_container=entry["container"], same_source_case_node_request=True,
                          original_question_history_reader_and_prompt_match=True))
    return dict(public_repository=prior["public_repository"], verified_public_commit=prior["verified_public_commit"],
                public_manifest=prior["public_manifest"], manifest_git_blob_sha=git_blob,
                verification="Reused previously live-verified public manifest identity; all30 local source hashes and source-node fields rechecked offline. No claim of a new remote fetch.",
                verified_nodes=len(cases), cases=cases,
                payload_excludes=["credentials", "references/gold", "native scores", "diagnostic labels", "global accounting ledger"],
                candidate_changes="Two nonsensitive demonstration referent substitutions only", new_provider_requests=0)


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    assert sha(PARENT / "shared_runtime.py") == RUNTIME_SHA
    with_history = [r for r in rows(PARENT / "PROBES.frozen.jsonl") if r["role"] == "ordinary"]
    no_history = rows(PARENT / "first_node_followup/PROBES.frozen.jsonl")
    assert len(with_history) == len(no_history) == 15
    assert all(r["history"] for r in with_history)
    assert all(r["history"] == [] and r["step_id"] == 1 for r in no_history)
    items = []
    for a, b in zip(with_history, no_history):
        items.extend(({**a, "group": "with_history"}, {**b, "group": "no_history"}))
    assert len({r["probe_id"] for r in items}) == 30
    proof = source_proofs(items)
    jsonl(INPUTS, items)
    tasks = [dict(index=i * 3 + j, task_id=r["probe_id"] + "/" + arm,
                  probe_id=r["probe_id"], case_id=r["case_id"], group=r["group"], arm=arm)
             for i, r in enumerate(items) for j, arm in enumerate(arm_order(r["case_id"]))]
    jsonl(ORDER, tasks)
    atomic_json(HERE / "PUBLIC_PAYLOAD_PROVENANCE.json", proof)
    config = read(PACKAGE / "config.json")
    frozen_contracts = [dict(probe_id=item["probe_id"], group=item["group"], arm=arm,
                             payload_sha256=digest(build_payload(config["generation"], variant_prompt(item["control_prompt"], arm), 128)),
                             historical_source_request_id=item["control_source_request_id"],
                             original_question_and_actual_task_unchanged=True, all_other_prompt_bytes_unchanged=True)
                        for item in items for arm in ARMS]
    jsonl(HERE / "CONTRACTS.frozen.jsonl", frozen_contracts)
    paths = [Path(__file__).resolve(), HERE / "__init__.py", HERE / "validate_offline.py", HERE / "PROTOCOL.md", INPUTS, ORDER,
             HERE / "CONTRACTS.frozen.jsonl",
             HERE / "PUBLIC_PAYLOAD_PROVENANCE.json", PARENT / "shared_runtime.py", PARENT / "PROBES.frozen.jsonl",
             PARENT / "PROTOCOL.frozen.json", PARENT / "first_node_followup/PROBES.frozen.jsonl",
             PARENT / "first_node_followup/PROTOCOL.frozen.json", PARENT / "first_node_followup/PUBLIC_PAYLOAD_PROVENANCE.json",
             PACKAGE / "config.json", PACKAGE / "core/llm.py", PACKAGE / "core/cache.py", PACKAGE / "core/runtime.py",
             ROOT / "scripts/cli_support.py", HERE.parent / "p02_formal1500_v1_20261021/network.py", MANIFEST, PRIOR_PROOF]
    pins = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}
    pins.update({r["source"].replace("\\", "/"): r["source_sha256"] for r in items})
    protocol = dict(id=HERE.name, prepared_utc=utc(), authority="P02_PROMPT_RESEARCH_AUTONOMOUS_SMALL_EVALUATIONS_APPROVED_20261022",
        state="prepared_only_no_provider_submissions", scope="Ordinary fixed-reader component development probe; no final F1/EM or complete-RAG result",
        nodes=30, group_counts={"with_history": 15, "no_history": 15}, planned_conditions=90, arms=list(ARMS),
        sample="Exactly the previous15 with-history ordinary nodes plus previous15 no-history first nodes; original within-group order, groups interleaved. No gold, native scores, new model outputs, or label-based resampling.",
        E0="Fresh synchronized original prompt in a new isolated checkpoint, StrictAsk seed=None; exact same historical payload but no historical response seeding. Resumes reuse only this experiment's saved receipts; no reroll or best-of selection.",
        E1="Only replace #1 in the two worked-example CURRENT subquestions by Colorado River and Callicoma, already in each example's prior answer.",
        E2="Only replace the same two #1 tokens by the full descriptions of their same referents.",
        exact_referents=REFERENTS, fixed="Actual original/current questions, history, reader contents/order, all demo original questions/answers/labels/order and main rules, full model payload and128 output cap. No formatter.",
        order="Per-case SHA256(20261022|case_id|ordinary_fewshot_order) modulo3 rotates E0/E1/E2; repeated nodes of the same case share order; all three outputs retained.",
        historical_control="Complete same-node original request_id/full payload/stop+usage, independent reference only, never substituted for E0 fresh inference.",
        requests=dict(SDK_retries=0, HTTP_transport_retries=0, maximum_actual_attempts=5, count_across_restarts=True,
                      frozen_shared_runtime_sha256=RUNTIME_SHA, concurrent_requests=1, minimum_start_interval_seconds=3,
                      unknown_without_receipt="Bounded authorized recovery; retired receipts and charge reservations retained",
                      unknown_with_response_id_usage_content_or_raw_response="Hold; never silently resubmit",
                      algorithm_failure="Length/empty/refusal/content-filter are terminal recorded outcomes, no reroll; missing/invalid usage or identity mismatch remain technical holds",
                      gate="outputs/query_decomposition/p02_prompt_research_v1_20261022/provider_gate"),
        cost="Existing no-cap ledger; all actual costs and unknown reservations separate; stage expense alerts at single3x or recent20 mean1.5x paired historical component cost, not full-query cost.",
        terminal_policy="Preserve all90 conditions. Technical held tasks remain missing and separately enumerated; algorithm failures remain in planned/terminal denominators. Do not remove inconvenient outcomes.",
        endpoint=config["generation"]["base_url"], model=config["generation"]["model"], report="Coverage/statuses, paired text changes, tokens/estimated cost; later separate semantic review; no final F1 or judge",
        main_evaluation_remains_paused=True, new_provider_submissions_at_prepare=0, pins=pins)
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    assert sha(PARENT / "shared_runtime.py") == RUNTIME_SHA
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def dry_run():
    check()
    items = rows(INPUTS)
    config = read(PACKAGE / "config.json")
    public = source_proofs(items)
    assert public == read(HERE / "PUBLIC_PAYLOAD_PROVENANCE.json")
    assert Counter((r["group"], r["dataset"]) for r in items) == {
        (group, ds): 5 for group in ("with_history", "no_history") for ds in ("hotpotqa", "2wiki", "musique")}
    contracts = []
    for item in items:
        assert complete_receipt(item["control_result"])
        assert digest(item["control_payload"]) == item["control_payload_sha256"]
        for arm in ARMS:
            prompt = variant_prompt(item["control_prompt"], arm)
            payload = build_payload(config["generation"], prompt, 128)
            assert payload["max_tokens"] == 128
            assert {k: v for k, v in payload.items() if k != "messages"} == {k: v for k, v in item["control_payload"].items() if k != "messages"}
            if arm == "E0":
                assert payload == item["control_payload"]
            else:
                original_prefix, original_actual = item["control_prompt"].split("\n\nActual task:\n", 1)
                candidate_prefix, candidate_actual = prompt.split("\n\nActual task:\n", 1)
                assert original_actual == candidate_actual
                # Invert exactly the two current-question changes. Everything
                # else, including both demonstration original questions, matches.
                for old, referent in zip(ORIGINAL, REFERENTS[arm]):
                    new = old.replace("#1", referent)
                    assert candidate_prefix.count(new) == 1
                    candidate_prefix = candidate_prefix.replace(new, old, 1)
                assert candidate_prefix == original_prefix
            contracts.append(dict(probe_id=item["probe_id"], group=item["group"], arm=arm,
                                  payload_sha256=digest(payload), historical_source_request_id=item["control_source_request_id"],
                                  original_question_and_actual_task_unchanged=True, all_other_prompt_bytes_unchanged=True))
    assert len(contracts) == len({r["payload_sha256"] for r in contracts}) == 90
    assert contracts == rows(HERE / "CONTRACTS.frozen.jsonl")
    tasks = rows(ORDER)
    assert len(tasks) == 90 and {t["task_id"] for t in tasks} == {r["probe_id"] + "/" + r["arm"] for r in contracts}
    for item in items:
        assert [t["arm"] for t in tasks if t["probe_id"] == item["probe_id"]] == arm_order(item["case_id"])
    jsonl(OUT / "DRYRUN_CONTRACTS.jsonl", contracts)
    result = dict(passed=True, utc=utc(), nodes=30, conditions=90, fresh_E0_conditions=30,
                  public_source_nodes_checked=30, only_two_demonstration_current_referents_changed=True,
                  all_demo_original_questions_other_words_rules_actual_task_fixed=True,
                  all_payloads_unique_within_new_checkpoint=True, no_historical_seed=True,
                  first_arm_counts=dict(Counter(arm_order(r["case_id"])[0] for r in items)),
                  protocol_sha256=sha(PROTOCOL), new_provider_submissions=0, main_evaluation_remains_paused=True)
    atomic_json(OUT / "DRYRUN.json", result)
    return result


def cost(result):
    usage = result.get("usage") or {}
    return usage.get("prompt_tokens", 0) * INPUT_RATE + usage.get("completion_tokens", 0) * OUTPUT_RATE


def algorithm_failure(receipt):
    if receipt.get("status") != "success":
        return None
    if receipt.get("finish_reason") in ("length", "content_filter"):
        return "terminal_" + receipt["finish_reason"]
    if receipt.get("refusal"):
        return "terminal_refusal"
    if isinstance(receipt.get("answer"), str) and not receipt["answer"].strip():
        return "terminal_empty"
    return None


def event(value):
    with (OUT / "events.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def report(cache=None):
    check()
    dbpath = OUT / "checkpoint.sqlite3"
    if cache is None and not dbpath.exists():
        return dict(planned_conditions=90, recorded_outcomes=0, status="prepared_not_started", new_provider_submissions=0)
    db = cache.db if cache else sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        outcomes = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        receipts = [json.loads(v) for (v,) in db.execute("SELECT result FROM requests WHERE result IS NOT NULL")]
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_requests'").fetchone():
            receipts += [json.loads(v) for (v,) in db.execute("SELECT result FROM retired_requests WHERE result IS NOT NULL")]
    finally:
        if cache is None:
            db.close()
    index = {(r["probe_id"], r["arm"]): r for r in outcomes}
    done = {r["task_id"] for r in outcomes}
    held = [read(p) for p in (OUT / "held").glob("*.json")]
    held = [r for r in held if r["task_id"] not in done]
    groups = {}
    for group in ("with_history", "no_history"):
        for arm in ARMS:
            rr = [r for r in outcomes if r["group"] == group and r["arm"] == arm]
            good = [r for r in rr if r["status"] == "success"]
            groups[group + "/" + arm] = dict(planned=15, terminal=len(rr), complete=len(good),
                algorithm_failures=sum(r["status"] == "algorithm_failure" for r in rr),
                historical_text_changed=sum(r["answer"] != r["historical_answer"] for r in good),
                input_tokens=sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rr),
                output_tokens=sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rr),
                missing_usage=sum(not r.get("usage") for r in rr))
    paired = {}
    for group in ("with_history", "no_history"):
        probes = [r["probe_id"] for r in rows(INPUTS) if r["group"] == group]
        for a, b in (("E0", "E1"), ("E0", "E2"), ("E1", "E2")):
            eligible = [p for p in probes if (p, a) in index and (p, b) in index
                        and index[p, a]["status"] == index[p, b]["status"] == "success"]
            paired[group + "/" + a + "_to_" + b] = dict(complete_pairs=len(eligible),
                text_changed_ids=[p for p in eligible if index[p, a]["answer"] != index[p, b]["answer"]])
    value = dict(utc=utc(), scope="Ordinary fixed-stage component probe; no final F1/EM", planned_conditions=90,
        recorded_outcomes=len(outcomes), complete=sum(r["status"] == "success" for r in outcomes),
        terminal_algorithm_failures=sum(r["status"] == "algorithm_failure" for r in outcomes), held_technical_tasks=len(held),
        status="all_conditions_terminal" if len(outcomes) == 90 else "partial", sections=groups, paired_text=paired,
        held_task_ids=[r["task_id"] for r in held], known_returned_usage_CNY=sum(cost(r) for r in receipts),
        historical_is_independent_reference=True, fresh_E0_not_historical_replay=True, main_evaluation_remains_paused=True)
    config = read(PACKAGE / "config.json")
    with closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as ledger:
        known, reserved = ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",
            ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),)).fetchone()
    value.update(actual_returned_usage_CNY=known, unresolved_reserved_CNY=reserved)
    atomic_json(OUT / "REPORT.json", value)
    return value


def run(limit=None):
    check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert any(r.get("id") == authority.get("decision_id") for r in rows(PLAN / "DECISIONS.jsonl"))
    config = read(PACKAGE / "config.json")
    items = {r["probe_id"]: r for r in rows(INPUTS)}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self, name, **fields): event({"event": name, **fields})
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config, OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
        cache.put("meta", "identity", dict(protocol_sha256=sha(PROTOCOL), inputs_sha256=sha(INPUTS), order_sha256=sha(ORDER)))
        stop = lambda: (OUT / "STOP_REQUESTED").exists()
        recent = [(cost(items[r["probe_id"]]["control_result"]), cost(r))
                  for r in sorted(cache.all("outcome").values(), key=lambda r: r["utc"]) if r.get("usage")]
        with closing(StrictAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as ask:
            processed = 0
            for task in rows(ORDER):
                key = task["task_id"]
                if cache.get("outcome", key) is not None:
                    continue
                if stop() or (limit is not None and processed >= limit):
                    return report(cache)
                processed += 1
                item, arm = items[task["probe_id"]], task["arm"]
                prompt = variant_prompt(item["control_prompt"], arm)
                base = dict(task_id=key, probe_id=item["probe_id"], case_id=item["case_id"], group=item["group"],
                            dataset=item["dataset"], role="ordinary", step_id=item["step_id"], arm=arm,
                            historical_source_request_id=item["control_source_request_id"],
                            historical_answer=item["control_result"]["answer"].strip(), fresh_experiment=True)
                try:
                    rid, answer, finish = ask("ordinary_fewshot/" + key, prompt, 128)
                    result = json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()[0])
                    outcome = {**base, "status": "success", "answer": answer, "request_id": rid,
                               "finish_reason": finish, "usage": result["usage"], "utc": utc()}
                except (RuntimeError, InterruptedError) as exc:
                    payload = canonical(build_payload(config["generation"], prompt, 128))
                    row = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
                    result = {**(json.loads(row[2]) if row and row[2] else {}), "status": row[1] if row else None}
                    reason = algorithm_failure(result)
                    if reason:
                        outcome = {**base, "status": "algorithm_failure", "answer": result.get("answer", ""),
                                   "request_id": row[0], "finish_reason": result.get("finish_reason"),
                                   "refusal": result.get("refusal"), "usage": result.get("usage"), "reason": reason, "utc": utc()}
                        event(dict(event="terminal_algorithm_outcome", task_id=key, reason=reason))
                    else:
                        hold = {**base, "status": "held_technical", "reason": str(exc), "request_id": row[0] if row else None, "utc": utc()}
                        atomic_json(OUT / "held" / (digest(key) + ".json"), hold)
                        event(dict(event="held_technical", task_id=key, reason=str(exc)))
                        continue
                cache.put("outcome", key, outcome)
                if outcome.get("usage"):
                    reference, actual = cost(item["control_result"]), cost(outcome)
                    recent.append((reference, actual))
                    reasons = []
                    if reference and actual > 3 * reference:
                        reasons.append("component_above_3x_paired_historical_stage")
                    if len(recent) >= 20 and sum(c for _, c in recent[-20:]) > 1.5 * sum(b for b, _ in recent[-20:]):
                        reasons.append("recent20_component_mean_above_1.5x_paired_historical_stages")
                    if reasons:
                        alert = dict(event="expense_alert", task_id=key, reasons=reasons, historical_stage_CNY=reference,
                                     actual_stage_CNY=actual, scope="Component costs; not a complete-query benchmark")
                        event(alert); emit(alert)
                emit(dict(event="condition_terminal", task_id=key, status=outcome["status"]))
                report(cache)
    return report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "run", "report"))
    parser.add_argument("--limit", type=int, help="Maximum newly processed conditions, including held; not a reset of technical attempt counts")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.action == "prepare":
        OUT.mkdir(parents=True, exist_ok=True)
        result = prepare()
    elif args.action == "dry-run":
        result = dry_run()
    elif args.action == "report":
        result = report()
    else:
        result = run(args.limit)
    emit(result)


if __name__ == "__main__":
    main()
