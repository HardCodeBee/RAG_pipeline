"""Final output/protocol audit, without retrieval or API calls."""

import ast
from collections import Counter
import hashlib
import json
import sqlite3

from .prepare import HERE, OUT, ROOT, PILOT_SPEC, PILOT_OUT, rows, load, write


def function_body(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    result = load(OUT / "RESULTS.json")
    records = rows(OUT / "CASE_RESULTS.jsonl")
    old = {(r["dataset"], r["qid"]): r for r in rows(PILOT_OUT / "CASE_RESULTS.jsonl")}
    assert len(records) == 60 and len({(r["dataset"], r["qid"]) for r in records}) == 60
    assert Counter(r["dataset"] for r in records) == {"hotpotqa": 20, "2wiki": 20, "musique": 20}
    assert all(result["frozen_hash_checks"].values())
    assert function_body(HERE / "run.py", "run_case") == function_body(PILOT_SPEC / "run.py", "run_case")
    for r in records:
        if r["sample_group"].startswith("old24"):
            before = old[(r["dataset"], r["qid"])]
            assert r["conditions"] == before["conditions"]
            assert r["historical_references"] == before["historical_references"]
    n_outputs = 0
    for r in records:
        if not r["sample_group"].startswith("fresh36"):
            continue
        for arm in ("single_original", "dual_original_rewritten"):
            path = OUT / "cases" / r["dataset"] / arm / (r["qid"] + ".json")
            if not path.exists():
                assert r["conditions"][arm]["f1"] is None
                assert r["conditions"][arm]["status"] == "provider_or_transport_missing"
                continue
            raw = load(path)
            assert raw["nodes"][-1]["role"] == "goal"
            assert raw["nodes"][-1]["request"] == r["question"]
            assert len(raw["original_top10_doc_ids"]) == 10
            assert all(d["block"] in raw["final_context"] for d in raw["nodes"][-1]["original_retrieval"]["docs"])
            assert all(n["answer"] in raw["helper"] for n in raw["nodes"])
            assert all(len(n["reader_packet"]["docs"]) == len({d["doc_id"] for d in n["reader_packet"]["docs"]}) == 10 for n in raw["nodes"])
            n_outputs += 1
    active = {}
    nonterminal_finish = {}
    recovered = load(OUT / "OFFLINE_FINAL_RECOVERY.json") if (OUT / "OFFLINE_FINAL_RECOVERY.json").exists() else None
    allowed_length = {recovered["final_request_id"]} if recovered else set()
    for path in OUT.glob("**/checkpoint.sqlite3"):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            active[str(path.parent.relative_to(OUT))] = db.execute("SELECT count(*) FROM requests WHERE status IN ('in_flight','unknown')").fetchone()[0]
            if "/api/" in path.as_posix():
                incomplete = {r[0] for r in db.execute("SELECT id FROM requests WHERE status='success' AND json_extract(result,'$.finish_reason')!='stop'")}
                assert incomplete.issubset(allowed_length)
                nonterminal_finish[str(path.parent.relative_to(OUT))] = len(incomplete)
    assert not any(active.values())
    assert result["global_returned_usage_list_price_cny"] + result["global_unresolved_reserved_cny"] < 200
    assert result["expansion_returned_usage_list_price_cny"] + result["expansion_unresolved_reserved_cny"] < 5
    audit = {"planned_total": 60, "per_dataset": dict(Counter(r["dataset"] for r in records)), "old24_outputs_and_scores_preserved": True,
        "new36_answer_outputs": n_outputs, "new36_missing_condition_outputs": 72 - n_outputs,
        "same_generation_function_as_pilot": True, "same_prompts_and_frozen_protocol": True,
        "all_successful_cases_execute_goal_and_retain_all_original10": True,
        "all_complete_answers_preserved_in_auxiliary": True, "reader_packets_10_unique_documents": True,
        "active_or_unknown_logical_requests": active, "length_limited_raw_finals_scored_without_retry": nonterminal_finish,
        "native_order_controls": {ds: load(OUT / f"NATIVE_EQUIVALENCE_{ds}.json") for ds in ("2wiki", "musique")},
        "frozen_hash_checks": result["frozen_hash_checks"],
        "fresh36_four_way_common": result["fresh36"]["four_way_common"], "total60_four_way_common": result["total60"]["four_way_common"],
        "budget_under_both_caps": True}
    write(OUT / "COMPLETION_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
