"""Reserve repeatedly failed exposed queries for future4B labeling, not training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from .review import ROOT, HERE, OUT, OLD, rows, raw, sha

DEST = HERE / "hard_cases_for_4b"
CHECKS = {"F": "faithfulness_and_unknowns", "S": "self_containment",
          "C": "supporting_information_coverage", "U": "distinct_usefulness"}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    if (DEST / "SUMMARY.json").exists():
        print((DEST / "SUMMARY.json").read_text(encoding="utf-8"))
        return
    cohorts = []
    historical = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/multigrain_decomposer_prompt_followup_20261017/AUTHORED_REVIEW_ALL_ARMS.jsonl"
    for c in rows(historical):
        reviews = {}
        for v in ("v2", "v3", "v4", "v5"):
            j = c[v]
            folder = "multigrain_decomposer_prompt_pilot_v1_20261017" if v == "v2" else f"multigrain_decomposer_prompt_{v}_20261017"
            p = OUT.parent / folder / "cases" / v / c["dataset"] / (c["qid"] + ".json")
            reviews[v] = {"verdict": j["verdict"], "failed_checks": [k for k, name in CHECKS.items() if j.get("checks", {}).get(name) == "fail"],
                          "reason": j["reason"], "literal_output": j.get("literal_answer"), "raw_result_file": str(p.relative_to(ROOT))}
        cohorts.append((c, "old_exposed36", reviews, "v5"))
    authored = {r["index"]: r for r in json.loads((HERE / "AUTHORED_dev.json").read_text(encoding="utf-8"))["cases"]}
    for c in rows(HERE / "dev.questions.frozen.jsonl"):
        reviews = {}
        j = authored[c["index"]]
        for v in ("v2", "v3", "v4", "v5", "v6", "v7", "v8"):
            code = j[v]
            verdict = "pass" if code == "P" else "unavailable" if code == "X" else "uncertain" if code.startswith("?") else "fail"
            r = raw(c, "dev", v)
            p = OUT / ("dev_" + v) / "cases" / v / c["dataset"] / (c["qid"] + ".json")
            reviews[v] = {"verdict": verdict, "failed_checks": list(code) if verdict == "fail" else [],
                          "reason": j.get("reasons", {}).get(v, j["reason"]), "literal_output": r.get("answer"), "raw_result_file": str(p.relative_to(ROOT))}
        cohorts.append((c, "fresh_development60", reviews, "v8"))
    # Protected held-out bodies are not opened. Verify against metadata hashes.
    manifest = ROOT / "outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data/split_manifest.json"
    protected = {r["query_sha256"] for r in json.loads(manifest.read_text(encoding="utf-8"))["records"] if r["project_assignment"] in {"project_test", "project_dev", "reserved_test_family", "reserved_dev_family"}}
    selected, seen = [], set()
    for c, cohort, reviews, latest in cohorts:
        failures = [v for v, j in reviews.items() if j["verdict"] == "fail"]
        if len(failures) < 2:
            continue
        key = (c["dataset"], c["qid"])
        assert key not in seen
        seen.add(key)
        query_hash = hashlib.sha256(normalize(c["question"]).encode()).hexdigest()
        protected_overlap = query_hash in protected
        ambiguous = any(j["verdict"] == "uncertain" for j in reviews.values())
        current = reviews[latest]["verdict"]
        priority = {"fail": "latest_still_fails", "pass": "previously_hard_now_fixed", "uncertain": "latest_uncertain", "unavailable": "latest_unavailable"}[current]
        selected.append({
            "id": "hard_case:" + c["dataset"] + ":" + c["qid"], "query": c["question"],
            "source": {"dataset": c["dataset"], "native_id": c["qid"], "native_split": "dev", "query_type": c["type"], "cohort": cohort},
            "normalized_query_sha256": query_hash,
            "flag": {"repeated_semantic_failure": True, "failed_versions": failures,
                     "complete_reviewed_versions": [v for v, j in reviews.items() if j["verdict"] != "unavailable"],
                     "latest_version": latest, "latest_verdict": current, "priority": priority,
                     "failure_modes": sorted({k for v in failures for k in reviews[v]["failed_checks"]})},
            "failure_history": [{"version": v, **reviews[v]} for v in failures],
            "latest_output_reference_not_gold": {"version": latest, **reviews[latest]},
            "training_preparation": {"reserved_for_future_4b_training": not protected_overlap,
                                     "admitted_to_training": False, "target_output": None,
                                     "label_status": "protected_split_conflict" if protected_overlap else "needs_original_review" if ambiguous else "needs_gold_decomposition",
                                     "has_uncertain_review": ambiguous, "protected_metadata_overlap": protected_overlap,
                                     "target_format": "JSON array of self-contained supporting questions, no #N or explicit goal; program appends original query",
                                     "exclude_from_future_4b_unseen_evaluation": True,
                                     "reason": "Repeated failures are counterexamples, not gold labels. Correct target must be authored and checked before SFT admission."}})
    rank = {"latest_still_fails": 0, "latest_uncertain": 1, "latest_unavailable": 2, "previously_hard_now_fixed": 3}
    selected.sort(key=lambda r: (rank[r["flag"]["priority"]], -len(r["flag"]["failed_versions"]), r["id"]))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "CANDIDATES.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    exclusions = [{"dataset": r["source"]["dataset"], "native_id": r["source"]["native_id"], "normalized_query_sha256": r["normalized_query_sha256"], "normalization": "NFKC,casefold,collapse_whitespace,preserve_punctuation", "status": "reserved_for_future_training_labeling", "must_exclude_future_4b_unseen_eval": True} for r in selected]
    (DEST / "FUTURE_EVAL_EXCLUSIONS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exclusions), encoding="utf-8")
    text = ["4B未来训练困难样本候选：仅标记，尚无金标、未并入SFT", "",
            "规则：同题至少两个单次生成版本明确语义失败；不把API缺失或不确定计为失败。", "范围：已观察旧36题+本轮开发60题；不读取/筛选本轮保留60题。", "最新版修复不等于人工金标；原句有疑义的条目先澄清。", ""]
    for i, r in enumerate(selected, 1):
        text += [f"{i}. {r['source']['dataset']} | {r['source']['native_id']}", "Query: " + r["query"],
                 "失败版本: " + ", ".join(r["flag"]["failed_versions"]), "最新版状态: " + r["flag"]["latest_version"] + ":" + r["flag"]["latest_verdict"],
                 "错误类型: " + ", ".join(r["flag"]["failure_modes"]), "标签状态: " + r["training_preparation"]["label_status"]]
        for j in r["failure_history"]:
            text += [j["version"] + " 原始输出: " + str(j["literal_output"]), "判定: " + j["reason"]]
        text += [""]
    (DEST / "CASES.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    summary = {"selection_rule": "at_least_two_explicit_single_pass_semantic_failures", "old36_versions": ["v2", "v3", "v4", "v5"],
               "fresh60_versions": ["v2", "v3", "v4", "v5", "v6", "v7", "v8"], "excluded_from_failure_count": ["API unavailable", "uncertain", "v9 two-stage editor"],
               "queries_screened": len(cohorts), "candidates": len(selected),
               "by_dataset": dict(Counter(r["source"]["dataset"] for r in selected)), "by_cohort": dict(Counter(r["source"]["cohort"] for r in selected)),
               "by_priority": dict(Counter(r["flag"]["priority"] for r in selected)),
               "by_label_status": dict(Counter(r["training_preparation"]["label_status"] for r in selected)),
               "failure_mode_counts_overlap": dict(Counter(k for r in selected for k in r["flag"]["failure_modes"])),
               "protected_metadata_overlaps": sum(r["training_preparation"]["protected_metadata_overlap"] for r in selected),
               "holdout60_used_for_hard_case_selection": False, "actual_sft_files_changed": False,
               "gold_labels_authored": 0, "review_quality": "current Codex nonblind rubric judgments, not human gold",
               "source_review_sha256": {"old36": sha(historical), "fresh60": sha(HERE / "AUTHORED_dev.json")}}
    (DEST / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
