"""Select on development only, before unsealing confirmation queries."""
import json

from .study import HERE, OUT, api, previous


def freeze():
    result = json.loads((OUT / "dev.RESULTS.json").read_text(encoding="utf-8"))
    stats = result["stats"]
    base_bytes = len((HERE / "TRAINING_PROMPT_v8.txt").read_bytes())
    candidates = []
    for arm in ("v10", "v11", "v12"):
        pair = next(p for p in result["pairs"] if p["before"] == "v8" and p["after"] == arm)
        prompt = HERE / ("TRAINING_PROMPT_" + arm + ".txt")
        eligible = len(prompt.read_bytes()) < base_bytes and stats[arm]["verdicts"].get("pass", 0) >= stats["v8"]["verdicts"].get("pass", 0) and pair["after_pass"] >= pair["before_pass"]
        candidates.append({"version": arm, "eligible": eligible, "pass_count": stats[arm]["verdicts"].get("pass", 0), "bytes": len(prompt.read_bytes()), "pair": pair})
    eligible = [c for c in candidates if c["eligible"]]
    selected = min(eligible, key=lambda c: (-c["pass_count"], c["bytes"]))["version"] if eligible else "v8"
    prompt = HERE / ("TRAINING_PROMPT_" + selected + ".txt")
    receipt = {"utc": previous.utc(), "version": selected, "prompt_sha256": api.sha(prompt),
               "candidates": candidates, "development_result_sha256": api.sha(OUT / "dev.RESULTS.json"),
               "authored_review_sha256": api.sha(HERE / "AUTHORED_dev.json"),
               "selection_sha256": api.sha(HERE / "SELECTION_FROZEN.json"),
               "selection_amendment_sha256": api.sha(HERE / "SELECTION_AMENDMENT.json"),
               "holdout_questions_sha256": api.sha(HERE / "holdout.questions.frozen.jsonl"),
               "holdout_text_or_output_inspected": False, "holdout_tuning_allowed": False,
               "rule": "shorter and not lower than v8 on full60 or pairwise common pass count; maximize full pass count, tie shorter prompt",
               "status": "candidate frozen for fresh confirmation; not an established 90% or downstream-RAG result"}
    api.immutable_json(HERE / "FINAL_CANDIDATE_FROZEN.json", receipt)
    (HERE / "BEST_PROMPT.txt").write_bytes(prompt.read_bytes())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    freeze()
