"""Expose literal outputs and aggregate explicitly authored semantic judgments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1" / HERE.name
OLD = HERE.parent / "multigrain_selfcontained_decomposer_v1_20261017"

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def raw(case, phase, arm):
    p = OUT / (phase + "_" + arm) / "cases" / arm / case["dataset"] / (case["qid"] + ".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def show(phase, arms, start, end):
    for case in rows(HERE / (phase + ".questions.frozen.jsonl"))[start-1:end]:
        print(f"{case['index']}. {case['dataset']}/{case['type']} Q: {case['question']}")
        seen = {}
        for arm in arms:
            r = raw(case, phase, arm)
            if not r:
                value = "UNSUBMITTED"
            elif r["status"] != "success" or r.get("finish_reason") != "stop":
                value = "UNAVAILABLE:" + r["status"] + ":" + str(r.get("finish_reason"))
            else:
                value = json.dumps(r["mechanical"].get("questions"), ensure_ascii=False)
            print(arm + " " + ("IDENTICAL_TO_" + seen[value] if value in seen and not value.startswith("UN") else value))
            seen.setdefault(value, arm)

def summarize(phase, arms):
    cases = rows(HERE / (phase + ".questions.frozen.jsonl"))
    document = json.loads((HERE / ("AUTHORED_" + phase + ".json")).read_text(encoding="utf-8"))
    judgments = {r["index"]: r for r in document["cases"]}
    assert len(judgments) == 60
    records, stats, audit = [], {}, {}
    for c in cases:
        rec = dict(c)
        j = judgments[c["index"]]
        for arm in arms:
            r = raw(c, phase, arm)
            assert r is not None
            code = j[arm]
            available = r["status"] == "success" and r.get("finish_reason") == "stop" and r.get("mechanical", {}).get("array_of_nonempty_strings")
            assert (code == "X") == (not available), (c["index"], arm, code, r["status"])
            assert code == "P" or code == "X" or (set(code.lstrip("?")) <= set("FSCU") and code.lstrip("?"))
            verdict = "unavailable" if code == "X" else "pass" if code == "P" else "uncertain" if code.startswith("?") else "fail"
            reason = j.get("reasons", {}).get(arm, j.get("reason", ""))
            assert reason
            rec[arm] = {"code": code, "verdict": verdict, "reason": reason, "questions": r.get("mechanical", {}).get("questions"), "literal_answer": r.get("answer"), "request_id": r["request_id"]}
        records.append(rec)
    for arm in arms:
        stats[arm] = {"planned": 60, "verdicts": dict(Counter(r[arm]["verdict"] for r in records)),
                      "pass_rate_full": sum(r[arm]["code"] == "P" for r in records)/60,
                      "failed_checks": {ch: sum(ch in r[arm]["code"] and not r[arm]["code"].startswith("?") for r in records) for ch in "FSCU"},
                      "by_dataset": {ds: dict(Counter(r[arm]["verdict"] for r in records if r["dataset"] == ds)) for ds in ("hotpotqa", "2wiki", "musique")}}
        root = OUT / (phase + "_" + arm)
        protocol = json.loads((root / "PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
        prompt = ROOT / protocol["prompt_path"]
        assert sha(prompt) == protocol["prompt_sha256"]
        assert sha(HERE / (phase + ".questions.frozen.jsonl")) == protocol["cases_sha256"]
        requests = [r for p in (root / "api" / arm).glob("*/*/requests.jsonl") for r in rows(p)]
        assert len(requests) == protocol.get("expected_requests", 60)
        for req in requests:
            p = req["payload"]
            assert p["model"] == protocol["model"] and p["temperature"] == 0 and p["max_tokens"] == 1024 and p["extra_body"] == {"enable_thinking": False}
            assert [m["role"] for m in p["messages"]] == ["system", "user"]
            assert p["messages"][0]["content"] == prompt.read_text(encoding="utf-8")
        if arm == "v9":
            expected = set()
            for c in cases:
                draft = raw(c, phase, protocol["upstream_version"])
                if draft["status"] == "success" and draft.get("finish_reason") == "stop" and draft.get("mechanical", {}).get("array_of_nonempty_strings"):
                    expected.add("Original question:\n" + c["question"] + "\n\nCandidate prerequisite questions:\n" + draft["answer"])
            assert {r["payload"]["messages"][1]["content"] for r in requests} == expected
        else:
            assert {r["payload"]["messages"][1]["content"] for r in requests} == {r["question"] for r in cases}
        values = [raw(c, phase, arm) for c in cases]
        complete = [r for r in values if r["status"] == "success" and r.get("finish_reason") == "stop"]
        stats[arm]["mean_input_tokens"] = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in complete)/max(1,len(complete))
        stats[arm]["mean_output_tokens"] = sum(r.get("usage", {}).get("completion_tokens", 0) for r in complete)/max(1,len(complete))
        audit[arm] = {"requests": len(requests), "statuses": dict(Counter(r["status"] for r in requests)), "payload_roles_model_prompt_and_allowed_inputs": True, "input_mode": protocol.get("mode", "single_pass_original_query_only"), "prompt_sha256": sha(prompt)}
    common = [r for r in records if all(r[a]["code"] != "X" for a in arms)]
    pairs = []
    for b in arms:
        for a in arms:
            if arms.index(a) <= arms.index(b):
                continue
            matched = [r for r in records if r[b]["code"] != "X" and r[a]["code"] != "X"]
            pairs.append({"before": b, "after": a, "n": len(matched), "before_pass": sum(r[b]["code"] == "P" for r in matched), "after_pass": sum(r[a]["code"] == "P" for r in matched), "gained": [r["index"] for r in matched if r[b]["code"] != "P" and r[a]["code"] == "P"], "lost": [r["index"] for r in matched if r[b]["code"] == "P" and r[a]["code"] != "P"]})
    with sqlite3.connect((OUT.parent / "budget.sqlite3").resolve().as_uri()+"?mode=ro", uri=True) as db:
        actual, reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUT.parent))+"%",)).fetchone()
        global_actual, global_reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) FROM charges").fetchone()
    result = {"phase": phase, "scope": "query-only type-stratified subset, single current-Codex review; not human-gold, trained4B or RAG-F1", "stats": stats, "all_common": {"n": len(common), "passes": {a: sum(r[a]["code"] == "P" for r in common) for a in arms}}, "pairs": pairs, "audit": audit, "study_usage_list_price_cny": actual, "study_unknown_reserved_cny": reserve, "global_usage_list_price_cny": global_actual, "global_unknown_reserved_cny": global_reserve, "actual_bill_verified": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (phase + ".RESULTS.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / (phase + ".REVIEW_ALL_ARMS.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("dev", "holdout"), default="dev")
    p.add_argument("--versions", nargs="+", default=["v2", "v3", "v4", "v5"])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=60)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.summarize:
        summarize(args.phase, args.versions)
    else:
        show(args.phase, args.versions, args.start, args.end)
