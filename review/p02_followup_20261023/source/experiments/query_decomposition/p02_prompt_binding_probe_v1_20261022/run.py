"""Prepare/dry-run fixed-input probes; only the explicit run action can submit.

This component experiment never runs retrieval, decomposition, or final QA.
No reference-answer or diagnostic correctness-label file is read.
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
from p02_multigrain_timing_cachefix_v1.core import child_prompts, pipeline
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from .shared_runtime import NoCapBudget, StrictAsk, atomic_json, complete_receipt, INPUT_RATE, OUTPUT_RATE

PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
COHORT = HERE.parent / "p02_noformatter_quick90_v1_20261021"
OUT = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1" / HERE.name
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "PROBES.frozen.jsonl"
DATASETS = ("hotpotqa", "2wiki", "musique")
ROLES = ("ordinary", "goal")
SEED = 20261022
RESOLVE = "Resolve each #N from its numbered prior question and answer. "
DESCRIPTIVE = ("Use relevant prior answers to resolve descriptive references while preserving "
               "the current question's entity roles, relation, conditions and scope. ")
VARIANTS = ("delete_resolve", "describe_binding")
AUTHORITY = "User: 按照你的计划 自主进行推进 对于一些小实验的评估不要过问我"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def prompt_variant(prompt, variant):
    assert prompt.count(RESOLVE) == 1, "Only the single frozen instruction may change"
    if variant == "delete_resolve":
        return prompt.replace(RESOLVE, "", 1)
    if variant == "describe_binding":
        return prompt.replace(RESOLVE, DESCRIPTIVE, 1)
    raise ValueError(variant)


def render(case, node):
    examples = read(PACKAGE / "prompts/child_examples.json")
    text = child_prompts.child(case["question"], node["step_id"], node["request"],
        node["prior_complete_answers"], pipeline.passages(node["reader_packet"]), examples, role=node["role"])
    if node["role"] == "goal":
        text = text[:-len("Answer:")] + child_prompts.GOAL_RULE + "\n\nAnswer:"
    return text


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    settings = read(PACKAGE / "config.json")["generation"]
    runtime = {r["id"]: r for r in rows(COHORT / "runtime90.frozen.jsonl")}
    bindings = rows(COHORT / "SOURCE_BINDINGS.frozen.jsonl")
    eligible = []
    pins = {str((COHORT / n).relative_to(ROOT)): sha(COHORT / n)
            for n in ("runtime90.frozen.jsonl", "SOURCE_BINDINGS.frozen.jsonl")}
    seed_path = PACKAGE / "data/replay.sqlite3"
    with closing(sqlite3.connect(seed_path.resolve().as_uri() + "?mode=ro", uri=True)) as seed:
        for binding in bindings:
            source = ROOT / binding["source"]
            assert sha(source) == binding["source_sha256"], source
            case = read(source)
            assert case["id"] in runtime and case["id"] == binding["id"]
            for node in case["nodes"]:
                if not node["prior_complete_answers"] or node["role"] not in ROLES:
                    continue
                prompt = render(case, node)
                assert prompt == node["child_prompt"], "Frozen prompt mismatch"
                payload = build_payload(settings, prompt, 128)
                # Never obtain a control by searching unrelated queries for a matching prompt.
                source_rid = node["answer_request_id"]
                receipt = seed.execute("SELECT result FROM requests WHERE request_id=? AND payload=?",
                    (source_rid, canonical(payload))).fetchone()
                assert receipt is not None, (case["id"], source_rid)
                result = json.loads(receipt[0])
                assert complete_receipt(result), (case["id"], source_rid)
                assert result["answer"].strip() == node["initial_answer"].strip()
                assert not node.get("sentence_format_request_id"), "Do not mix formatter outputs with local reader response"
                probe_id = f"{case['id']}/{node['role']}/{node['step_id']}"
                key = hashlib.sha256(f"{SEED}|{probe_id}".encode()).hexdigest()
                eligible.append({"probe_id": probe_id, "case_id": case["id"], "qid": case["qid"],
                    "dataset": case["dataset"], "role": node["role"], "step_id": node["step_id"],
                    "selection_sha256": key, "source": binding["source"],
                    "source_sha256": binding["source_sha256"], "original_question": case["question"],
                    "current_question": node["request"], "history": node["prior_complete_answers"],
                    "reader_packet": node["reader_packet"], "control_prompt": prompt,
                    "control_payload": payload, "control_payload_sha256": digest(payload),
                    "control_source_request_id": source_rid, "control_result": result})
    selected = []
    groups = {(role, ds): sorted((r for r in eligible if r["role"] == role and r["dataset"] == ds),
                                key=lambda r: r["selection_sha256"])[:5]
              for role in ROLES for ds in DATASETS}
    assert all(len(g) == 5 for g in groups.values())
    # Stable interleaving makes small --limit runs cover both roles early.
    for index in range(5):
        for ds in DATASETS:
            for role in ROLES:
                selected.append(groups[(role, ds)][index])
    for item in selected:
        pins[item["source"]] = item["source_sha256"]
    for p in [HERE / "run.py", HERE / "shared_runtime.py", HERE / "validate_offline.py", PACKAGE / "config.json",
              PACKAGE / "core/child_prompts.py", PACKAGE / "core/pipeline.py",
              PACKAGE / "core/llm.py", PACKAGE / "core/cache.py", PACKAGE / "core/runtime.py",
              PACKAGE / "prompts/child_examples.json", ROOT / "scripts/cli_support.py",
              HERE.parent / "p02_formal1500_v1_20261021/network.py"]:
        pins[str(p.relative_to(ROOT))] = sha(p)
    INPUTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf8")
    pins[str(INPUTS.relative_to(ROOT))] = sha(INPUTS)
    protocol = {"id": HERE.name, "created_utc": utc(), "authority": AUTHORITY,
        "scope": "Fixed-input local stage development probe; not full RAG or formal quality evaluation",
        "planned_nodes": 30, "per_role": 15, "per_role_dataset": 5, "candidate_variants": list(VARIANTS),
        "sample_seed": SEED, "selection": "Ascending SHA256(seed|case_id/role/step_id), first5 per role/dataset; no scores or gold read; nodes may share an original query",
        "control": "Same original frozen fixed90 case/node answer_request_id AND complete payload, stop+usage; never cross-query replacement",
        "control_replay_only": True, "candidate_seed_reuse": False,
        "changes": {"delete_resolve": {"from": RESOLVE, "to": ""},
                    "describe_binding": {"from": RESOLVE, "to": DESCRIPTIVE}},
        "frozen": "All examples, IDs, history, questions, reader text/order, model and original128 output allowance; no formatter, retrieval or downstream generation",
        "requests": {"SDK_retries": 0, "HTTP_transport_retries": 0, "maximum_actual_attempts": 5,
            "include_prior_attempts_in_same_persistent_namespace": True,
            "unknown_without_any_returned_receipt": "bounded reissue under delegated small-experiment authority; preserve old receipt and charge reservation",
            "unknown_with_response_id_usage_raw_response_or_content": "hold for receipt review, do not reissue",
            "length_empty_refusal": "terminal algorithm outcome; never rerun for scores",
            "provider_concurrency": 1, "minimum_submission_interval_seconds": 3,
            "shared_gate": "outputs/query_decomposition/p02_prompt_research_v1_20261022/provider_gate"},
        "cost": "Existing global charges ledger, no old fixed cap; all actual/unknown costs retained. Component cost alerts use paired historical stage costs, not full-RAG claims.",
        "measurement": "Coverage, exact output changes, stop/usage, input/output tokens and estimated expense; semantic review separate; no native final F1/EM computed",
        "destination": settings["base_url"], "main_evaluation_remains_paused": True,
        "new_gold_or_external_judge_usage": False, "pins": pins}
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    return protocol


def dry_run():
    check()
    probes = rows(INPUTS)
    config = read(PACKAGE / "config.json")
    counts = Counter((r["role"], r["dataset"]) for r in probes)
    assert len(probes) == 30 and all(v == 5 for v in counts.values())
    tasks = []
    for item in probes:
        assert complete_receipt(item["control_result"])
        assert digest(item["control_payload"]) == item["control_payload_sha256"]
        # The literal replacement is the sole candidate input change.
        for variant in VARIANTS:
            candidate = prompt_variant(item["control_prompt"], variant)
            request = build_payload(config["generation"], candidate, 128)
            unchanged_fields = {k: v for k, v in request.items() if k != "messages"}
            assert unchanged_fields == {k: v for k, v in item["control_payload"].items() if k != "messages"}
            assert request["messages"] == [{"role": "user", "content": candidate}]
            tasks.append({"probe_id": item["probe_id"], "variant": variant,
                          "control_source_request_id": item["control_source_request_id"],
                          "candidate_payload_sha256": digest(request)})
    assert len(tasks) == 60
    output = {"passed": True, "utc": utc(), "planned_nodes": 30,
              "planned_candidate_tasks": 60, "control_receipts_exact_same_node": 30,
              "counts": {f"{role}/{ds}": n for (role, ds), n in counts.items()},
              "selection_does_not_read_gold_or_scores": True,
              "candidate_changes_only_one_sentence": True, "new_provider_submissions": 0,
              "new_retrieval_or_GPU_jobs": 0, "main_evaluation_remains_paused": True,
              "protocol_sha256": sha(PROTOCOL), "tasks": tasks}
    atomic_json(OUT / "DRY_RUN.json", output)
    return {k: v for k, v in output.items() if k != "tasks"}


def emit_event(value):
    with (OUT / "events.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def cost(result):
    usage = result.get("usage") or {}
    return usage.get("prompt_tokens", 0) * INPUT_RATE + usage.get("completion_tokens", 0) * OUTPUT_RATE


def report(cache=None):
    check()
    if cache is None:
        database = OUT / "checkpoint.sqlite3"
        if not database.exists():
            return {"planned_candidate_tasks": 60, "complete": 0}
        db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        db = cache.db
    try:
        outcomes = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        receipts = [json.loads(r[0]) for r in db.execute("SELECT result FROM requests WHERE result IS NOT NULL")]
        receipts += [json.loads(r[0]) for r in db.execute("SELECT result FROM retired_requests WHERE result IS NOT NULL")]
    finally:
        if cache is None:
            db.close()
    good = [r for r in outcomes if r["status"] == "success"]
    completed_keys = {r["task_id"] for r in outcomes}
    held = [read(path) for path in (OUT / "held").glob("*.json")]
    held = [r for r in held if r["task_id"] not in completed_keys]
    sections = {}
    for role in ROLES:
        for variant in VARIANTS:
            group = [r for r in good if r["role"] == role and r["variant"] == variant]
            sections[role + "/" + variant] = {"planned": 15, "complete": len(group),
                "text_changed": sum(r["answer"] != r["control_answer"] for r in group),
                "input_tokens": sum(r["usage"]["prompt_tokens"] for r in group),
                "output_tokens": sum(r["usage"]["completion_tokens"] for r in group)}
    value = {"utc": utc(), "scope": "Fixed-input stage probe; no final F1/EM",
             "planned_candidate_tasks": 60, "complete": len(good), "recorded_outcomes": len(outcomes),
             "terminal_algorithm_failures": sum(r["status"] == "algorithm_failure" for r in outcomes),
             "held_technical_tasks": len(held),
             "sections": sections, "known_returned_usage_CNY": sum(cost(r) for r in receipts),
             "outcome_statuses": dict(Counter(r["status"] for r in outcomes)),
             "main_evaluation_remains_paused": True}
    # Preserve billable unknown liabilities in the common ledger; do not report them as zero.
    config = read(PACKAGE / "config.json")
    ledger = sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        namespace = "p02_prompt_research_v1_20261022/" + str(OUT.resolve())
        known, reserved = ledger.execute(
            "SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", (namespace,)).fetchone()
        value.update(actual_returned_usage_CNY=known, unresolved_reserved_CNY=reserved)
    finally:
        ledger.close()
    atomic_json(OUT / "REPORT.json", value)
    return value


def run(limit=None):
    check()
    config = read(PACKAGE / "config.json")
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self, name, **fields):
            emit_event({"event": name, **fields})
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config, OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
        cache.put("meta", "identity", {"protocol_sha256": sha(PROTOCOL), "inputs_sha256": sha(INPUTS)})
        stopping = lambda: (OUT / "STOP_REQUESTED").exists()
        with closing(StrictAsk(cache, None, config, budget, emit_event, stop=stopping)) as ask:
            attempted = 0
            probes = rows(INPUTS)
            controls = {r["probe_id"]: r for r in probes}
            prior = sorted(cache.all("outcome").values(), key=lambda r: r["utc"])
            recent = [(cost(controls[r["probe_id"]]["control_result"]), cost(r))
                      for r in prior if r["status"] == "success"]
            for item in probes:
                for variant in VARIANTS:
                    key = item["probe_id"] + "/" + variant
                    previous = cache.get("outcome", key)
                    if previous is not None:
                        continue
                    if stopping() or (limit is not None and attempted >= limit):
                        return report(cache)
                    attempted += 1
                    base = {"task_id": key, "probe_id": item["probe_id"], "case_id": item["case_id"],
                            "dataset": item["dataset"], "step_id": item["step_id"], "role": item["role"],
                            "variant": variant, "control_source_request_id": item["control_source_request_id"],
                            "control_answer": item["control_result"]["answer"].strip()}
                    try:
                        rid, answer, finish = ask("local_probe/" + key,
                            prompt_variant(item["control_prompt"], variant), 128)
                    except (RuntimeError, InterruptedError) as exc:
                        payload = canonical(build_payload(config["generation"],
                            prompt_variant(item["control_prompt"], variant), 128))
                        row = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
                        receipt = json.loads(row[2]) if row and row[2] else {}
                        if row and row[1] == "success":
                            cache.put("outcome", key, {**base, "status": "algorithm_failure",
                                "answer": receipt.get("answer", ""), "request_id": row[0],
                                "finish_reason": receipt.get("finish_reason"), "usage": receipt.get("usage"),
                                "reason": "Returned response failed complete-stop/content/usage contract; never rerun for scores",
                                "utc": utc()})
                        emit_event({"event": "task_held", "task_id": key, "failure": str(exc),
                                    "main_evaluation_remains_paused": True})
                        atomic_json(OUT / "held" / (digest(key) + ".json"), {**base, "reason": str(exc), "utc": utc()})
                        continue
                    result = json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()[0])
                    outcome = {**base, "status": "success", "answer": answer, "request_id": rid,
                               "finish_reason": finish, "usage": result["usage"], "utc": utc()}
                    cache.put("outcome", key, outcome)
                    baseline_cost = cost(item["control_result"])
                    fresh_cost = cost(result)
                    recent.append((baseline_cost, fresh_cost))
                    reasons = []
                    if baseline_cost and fresh_cost > 3 * baseline_cost:
                        reasons.append("component_cost_above_3x_paired_historical_stage")
                    if len(recent) >= 20 and sum(c for _, c in recent[-20:]) > 1.5 * sum(b for b, _ in recent[-20:]):
                        reasons.append("recent20_component_mean_above_1.5x_paired_historical_stages")
                    if reasons:
                        alert = {"event": "expense_alert", "task_id": key, "reasons": reasons,
                                 "baseline_stage_CNY": baseline_cost, "candidate_stage_CNY": fresh_cost,
                                 "scope": "Component cost only; not a complete-query RAG cost benchmark"}
                        emit_event(alert)
                        emit(alert)
                    emit({"event": "task_complete", "task_id": key,
                          "text_changed": answer != base["control_answer"]})
                    report(cache)
            return report(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "run", "report"))
    parser.add_argument("--limit", type=int, help="Maximum newly processed candidate tasks in this invocation, including held tasks; technical attempts remain bounded independently")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.action == "prepare":
        emit(prepare())
    elif args.action == "dry-run":
        emit(dry_run())
    elif args.action == "report":
        emit(report())
    else:
        emit(run(args.limit))


if __name__ == "__main__":
    main()
