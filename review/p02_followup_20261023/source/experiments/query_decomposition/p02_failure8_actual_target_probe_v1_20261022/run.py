"""Root-authorized C8 local screen: reuse11 exact I0 uses, submit only5 new tasks."""
from contextlib import closing
from pathlib import Path
import argparse
import json
import sqlite3
from urllib.parse import urlsplit
from experiments.query_decomposition.p02_failure8_actual_target_probe_v1_20261022 import prepare as frozen, templates
from experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.runtime import CandidateAsk, normal_complete, algorithm_failure
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, atomic_json, INPUT_RATE, OUTPUT_RATE
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload

HERE, ROOT, OUT, PACKAGE = frozen.HERE, frozen.ROOT, frozen.OUT, frozen.PACKAGE
read, rows, sha, digest = frozen.read, frozen.rows, frozen.sha, frozen.digest


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"utc": utc(), **value}, ensure_ascii=False) + "\n")


def cost(result):
    usage = result["usage"]
    return usage["prompt_tokens"] * INPUT_RATE + usage["completion_tokens"] * OUTPUT_RATE


def identity(cache):
    expected = dict(protocol_sha256=sha(frozen.PROTOCOL), controls_sha256=sha(frozen.CONTROLS),
        inputs_sha256=sha(frozen.INPUTS), order_sha256=sha(frozen.ORDER))
    previous = cache.get("meta", "identity")
    if previous is None:
        assert cache.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0 and not cache.all("outcome")
        cache.put("meta", "identity", expected)
    else:
        assert previous == expected, "Resume identity mismatch"


def import_controls(cache, items):
    controls = {r["probe_id"]: r for r in read(frozen.CONTROLS)["controls"]}
    for use in rows(HERE / "LOGICAL_USES16.frozen.jsonl"):
        if not use["reuse_already_paid_I0"]:
            continue
        item, control = items[use["probe_id"]], controls[use["probe_id"]]
        payload = build_payload(read(PACKAGE / "config.json")["generation"], templates.prompt(item, use["arm"]), frozen.CAP)
        assert payload == control["source_payload"] and control["source_request_id"] == use["source_request_id"]
        assert normal_complete(control["result"], payload["model"])
        value = dict(**use, id=item["id"], dataset=item["dataset"], step_id=item["step_id"],
            status="success", answer=control["result"]["answer"].strip(), request_id=control["source_request_id"],
            usage=control["result"]["usage"], finish_reason="stop", payload_sha256=digest(payload),
            interface_group=item["interface_group"], mechanism_group=item["mechanism_group"],
            original_raw_current_evaluation_target=item["current_question"], actual_model_current=templates.model_current(item, use["arm"]),
            origin="reused_exact_paid_fresh_I0_nonconcurrent", source_database=control["source_database"],
            source_task_id=control["source_task_id"], actual_new_CNY=0,
            attributed_already_charged_CNY=control["source_charge"]["actual_CNY"], utc=control["source_outcome"]["utc"])
        previous = cache.get("outcome", value["task_id"])
        assert previous is None or previous == value
        if previous is None:
            cache.put("outcome", value["task_id"], value)


def new_unique(outcomes):
    return list({r["request_id"]: r for r in outcomes if r.get("new_payload")}.values())


def report(cache=None):
    frozen.check()
    if cache is not None:
        outcomes = list(cache.all("outcome").values())
        known_ids = {r[0] for r in cache.db.execute("SELECT id FROM requests UNION SELECT id FROM retired_requests")}
    elif (OUT / "checkpoint.sqlite3").exists():
        with closing(sqlite3.connect((OUT / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            outcomes = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='outcome'")]
            known_ids = {r[0] for r in db.execute("SELECT id FROM requests UNION SELECT id FROM retired_requests")}
    else:
        return dict(status="prepared_not_started", planned_nodes=8, planned_logical_conditions=16,
            source_logical_reuses=11, source_unique_paid_receipts=8, distinct_planned_new_payloads=5, main_evaluation_remains_paused=True)
    uses = rows(HERE / "LOGICAL_USES16.frozen.jsonl")
    planned = {r["task_id"] for r in uses}
    done = {r["task_id"] for r in outcomes}
    assert len(outcomes) == len(done) and done <= planned
    held = [read(p) for p in (OUT / "held").glob("*.json") if read(p)["task_id"] not in done]
    with closing(sqlite3.connect(Path(read(PACKAGE / "config.json")["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        charges = [dict(id=i, actual_CNY=a, reserved_CNY=r, status=s) for i, a, r, s in db.execute(
            "SELECT id,actual,reserved,status FROM charges WHERE namespace=?", ("p02_prompt_research_v1_20261022/" + str(OUT.resolve()),))]
    pending = {r["id"] for r in charges if r["actual_CNY"] is None}
    index = {(r["probe_id"], r["arm"]): r for r in outcomes}
    pairs = [r["probe_id"] for r in rows(frozen.INPUTS) if (r["probe_id"], "C0") in index and (r["probe_id"], "C1") in index]
    reused = {r["request_id"]: r for r in outcomes if r.get("reuse_already_paid_I0")}
    unique = new_unique(outcomes)
    sections = {}
    for arm in templates.ARMS:
        data = [r for r in outcomes if r["arm"] == arm]
        sections[arm] = dict(planned=8, terminal=len(data), normal_complete=sum(r["status"] == "success" for r in data),
            algorithm_terminal=sum(r["status"] == "algorithm_failure" for r in data),
            logical_input_tokens=sum(r["usage"]["prompt_tokens"] for r in data), logical_output_tokens=sum(r["usage"]["completion_tokens"] for r in data))
    value = dict(status="all_conditions_terminal" if len(done) == 16 else "partial", planned_nodes=8, planned_logical_conditions=16,
        terminal=len(done), sections=sections, pending_task_ids=sorted(planned - done), held_technical=held,
        comparison=dict(terminal_pairs=len(pairs), changed_text_ids=[p for p in pairs if index[p, "C0"]["answer"] != index[p, "C1"]["answer"]],
            same_actual_receipt_pairs=sum(index[p, "C0"]["request_id"] == index[p, "C1"]["request_id"] for p in pairs)),
        distinct_new_returned_receipts=len(unique), distinct_planned_new_payloads=5,
        new_unique_input_tokens=sum(r["usage"]["prompt_tokens"] for r in unique), new_unique_output_tokens=sum(r["usage"]["completion_tokens"] for r in unique),
        actual_new_known_returned_usage_CNY=sum(r["actual_CNY"] or 0 for r in charges),
        unresolved_reserved_CNY=sum(r["reserved_CNY"] for r in charges if r["actual_CNY"] is None),
        unresolved_reservation_ids=sorted(pending), reservations_without_cache_record_ids=sorted(pending - known_ids),
        source_logical_reuses=sum(r.get("reuse_already_paid_I0", False) for r in outcomes), source_unique_paid_receipts=len(reused),
        source_unique_attributed_already_charged_CNY=sum(r["attributed_already_charged_CNY"] for r in reused.values()), source_new_CNY=0,
        original_raw_current_evaluation_target_fixed=True, no_native_or_final_F1=True, semantic_labels_not_automatically_assigned=True,
        scope="Exposed eight-node local screen with nonconcurrent I0 reuse; not causal C benefit or fullRAG",
        any_apparent_repair_requires_fresh_synchronous_all8_validation=True, main_evaluation_remains_paused=True)
    frozen.jsonl(OUT / "DETAILS.local.jsonl", outcomes)
    atomic_json(OUT / "REPORT.json", value)
    return value


def run(limit=None):
    frozen.check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert authority.get("protocol_sha256") == sha(frozen.PROTOCOL) and authority.get("controls_sha256") == sha(frozen.CONTROLS)
    assert any(r.get("id") == authority.get("decision_id") for r in rows(frozen.PLAN / "DECISIONS.jsonl"))
    config = read(PACKAGE / "config.json")
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    items = {r["probe_id"]: r for r in rows(frozen.INPUTS)}
    controls = {r["probe_id"]: r for r in read(frozen.CONTROLS)["controls"]}
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class NetworkEvents:
        def event(self, name, **fields):
            event(dict(event=name, **fields))
    with run_lock(OUT), Cache(OUT / "checkpoint.sqlite3") as cache, closing(NoCapBudget(config, OUT)) as budget, \
            temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"), \
            provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()):
        identity(cache)
        import_controls(cache, items)
        stop = lambda: (OUT / "STOP_REQUESTED").exists()
        previous = new_unique(list(cache.all("outcome").values()))
        recent = [(cost(controls[r["probe_id"]]["result"]), cost(r)) for r in previous]
        seen = {r["request_id"] for r in previous}
        with closing(CandidateAsk(cache, None, config, budget, event, maximum_attempts=5, recover_unknown=True, stop=stop)) as ask:
            processed = 0
            for task in rows(frozen.ORDER):
                key = task["task_id"]
                if cache.get("outcome", key) is not None:
                    continue
                if stop() or (limit is not None and processed >= limit):
                    return report(cache)
                processed += 1
                item = items[task["probe_id"]]
                text = templates.prompt(item, task["arm"])
                payload = build_payload(config["generation"], text, frozen.CAP)
                assert task["arm"] == "C1" and digest(payload) == task["payload_sha256"]
                base = dict(**task, id=item["id"], dataset=item["dataset"], step_id=item["step_id"], new_payload=True,
                    reuse_already_paid_I0=False, interface_group=item["interface_group"], mechanism_group=item["mechanism_group"],
                    original_raw_current_evaluation_target=item["current_question"], actual_model_current=templates.model_current(item, task["arm"]),
                    origin="new_actual_current_replacement_provider_or_exact_local_receipt")
                try:
                    rid, answer, finish = ask("failure8_actual_target/" + key, text, frozen.CAP)
                    result = json.loads(cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()[0])
                    if not normal_complete(result, config["generation"]["model"]):
                        raise RuntimeError("Held strict receipt: matching model, stop/nonempty/no refusal, all3 integer usage required")
                    outcome = dict(**base, status="success", answer=answer, request_id=rid, finish_reason=finish, usage=result["usage"], utc=utc())
                except (RuntimeError, InterruptedError) as exc:
                    row = cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (canonical(payload),)).fetchone()
                    result = {**(json.loads(row[2]) if row and row[2] else {}), "status": row[1] if row else None}
                    reason = algorithm_failure(result, config["generation"]["model"])
                    if reason:
                        outcome = dict(**base, status="algorithm_failure", answer=result.get("answer", ""), request_id=row[0],
                            finish_reason=result.get("finish_reason"), refusal=result.get("refusal"), usage=result["usage"], reason=reason, utc=utc())
                    else:
                        atomic_json(OUT / "held" / (digest(key) + ".json"), dict(**base, status="held_technical", reason=str(exc), request_id=row[0] if row else None, utc=utc()))
                        event(dict(event="held_technical", task_id=key, reason=str(exc)))
                        report(cache)
                        continue
                cache.put("outcome", key, outcome)
                if outcome["request_id"] not in seen:
                    seen.add(outcome["request_id"])
                    reference, actual = cost(controls[item["probe_id"]]["result"]), cost(outcome)
                    recent.append((reference, actual))
                    reasons = []
                    if reference and actual > 3 * reference:
                        reasons.append("unique_stage_above3x_same_node_I0")
                    if len(recent) >= 20 and sum(c for _, c in recent[-20:]) > 1.5 * sum(c for c, _ in recent[-20:]):
                        reasons.append("recent20_unique_stage_above1.5x_same_node_I0")
                    if reasons:
                        alert = dict(event="expense_alert", task_id=key, reasons=reasons, I0_stage_CNY=reference, actual_stage_CNY=actual,
                            scope="Unique new ordinary stage, not wholequery cost")
                        event(alert)
                        emit(alert)
                emit(dict(event="condition_terminal", task_id=key, status=outcome["status"]))
                report(cache)
    return report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "report", "run"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    emit(frozen.dry_run() if args.action == "dry-run" else report() if args.action == "report" else run(args.limit))
