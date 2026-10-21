"""Score all matched conditions, preserving fresh and regression denominators."""

import json
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write
from .prepare import HERE, OUT, ROOT, OUTPUT, OLD_OUT, S02, sha


def mean(values):
    return sum(values) / len(values) if values else None


def paired(records, left, right):
    available = [r for r in records if all(r["conditions"][c].get("f1") is not None for c in (left, right))]
    differences = [r["conditions"][left]["f1"] - r["conditions"][right]["f1"] for r in available]
    return {"pairs": len(available), "planned": len(records), "mean_difference": mean(differences),
        "gains": sum(d > 0 for d in differences), "ties": sum(d == 0 for d in differences),
        "losses": sum(d < 0 for d in differences),
        "loss_qids": [r["qid"] for r, d in zip(available, differences) if d < 0]}


def main():
    protocol = load(HERE / "protocol.json")
    cases = rows(HERE / "cases.frozen.jsonl")
    references, scorers = {}, {}
    for dataset in {c["dataset"] for c in cases}:
        references[dataset] = {r["qid"]: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.references.local.jsonl")}
        scorers[dataset] = metric(dataset)
    details = []
    for case in cases:
        reference = references[case["dataset"]][case["qid"]]
        gold_docs = set(reference["positive_doc_ids"])
        row = {k: case[k] for k in ("dataset", "qid", "question", "evaluation_group", "historical_f1", "historical_group")}
        row["gold_answer"] = reference["answer"]
        row["conditions"] = {}
        for condition in protocol["conditions"]:
            path = OUT / "cases" / case["dataset"] / condition / (case["qid"] + ".json")
            if not path.exists():
                row["conditions"][condition] = {"status": "unstarted_or_incomplete", "f1": None}
                continue
            result = load(path)
            shown = set(result["original_top10_doc_ids"])
            rewrites = []
            displaced_gold = []
            for node in result["nodes"]:
                actual_docs = {d["doc_id"] for d in node["reader_packet"]["docs"]}
                original_docs = {d["doc_id"] for d in node["original_retrieval"]["docs"]}
                shown.update(actual_docs)
                if node["rewritten_retrieval"] is not None:
                    rewrites.append({"step_id": node["step_id"], "question": node["request"], "rewrite": node["rewritten_query"],
                        "prior_complete_answers": node["prior_complete_answers"], "answer": node["answer"]})
                lost = (original_docs - actual_docs) & gold_docs
                if lost:
                    displaced_gold.append({"step_id": node["step_id"], "doc_ids": sorted(lost)})
            with sqlite3.connect((OUT / "api" / case["dataset"] / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
                raw = db.execute("SELECT result FROM requests WHERE id=?", (result["request_id"],)).fetchone()[0]
                usage = json.loads(raw).get("usage", {})
            score = scorers[case["dataset"]](result["prediction"], reference)
            original = result["nodes"][-1]["original_retrieval"]["docs"]
            row["conditions"][condition] = {"status": result["status"], "prediction": result["prediction"], **score,
                "all_gold_docs_seen_by_some_reader": gold_docs.issubset(shown),
                "gold_document_recall": len(gold_docs & shown) / len(gold_docs),
                "rewrites": rewrites, "displaced_gold": displaced_gold,
                "all_original_top10_preserved_in_final": all(d["block"] in result["final_context"] for d in original),
                "final_prompt_tokens": usage.get("prompt_tokens")}
        details.append(row)
    def summary(records):
        conditions = {}
        for condition in protocol["conditions"]:
            available = [r["conditions"][condition] for r in records if r["conditions"][condition].get("f1") is not None]
            conditions[condition] = {"available": len(available), "missing": len(records) - len(available),
                "mean_f1_available": mean([r["f1"] for r in available]),
                "mean_final_prompt_tokens": mean([r["final_prompt_tokens"] for r in available]),
                "all_gold_docs_seen_questions": sum(r["all_gold_docs_seen_by_some_reader"] for r in available),
                "changed_rewrites": sum(len(r["rewrites"]) for r in available),
                "steps_with_displaced_gold": sum(len(r["displaced_gold"]) for r in available),
                "original_top10_preserved": all(r["all_original_top10_preserved_in_final"] for r in available)}
        return {"planned": len(records), "conditions": conditions,
            "new_dual_minus_single": paired(records, "c03_both", "c04_single"),
            "old_rewrite_new_final_minus_single": paired(records, "c01_final_only", "c04_single"),
            "final_effect_with_old_rewrite": paired(records, "c01_final_only", "c00_old_prompts"),
            "final_effect_with_new_rewrite": paired(records, "c03_both", "c02_rewrite_only"),
            "rewrite_effect_with_old_final": paired(records, "c02_rewrite_only", "c00_old_prompts"),
            "rewrite_effect_with_new_final": paired(records, "c03_both", "c01_final_only")}
    overall = summary(details)
    complete = all(v["missing"] == 0 for v in overall["conditions"].values())
    comparison = overall["new_dual_minus_single"]
    passed = complete and comparison["losses"] == 0 and comparison["mean_difference"] > 0
    with sqlite3.connect((OUTPUT / "budget.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        actual, reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()
        global_actual, global_reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges").fetchone()
    api = {}
    for dataset in references:
        path = OUT / "api" / dataset / "checkpoint.sqlite3"
        if not path.exists():
            continue
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            api[dataset] = {"request_statuses_including_imported_receipts": dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status")),
                "new_attempt_statuses": dict(db.execute("SELECT status,count(*) FROM attempts GROUP BY status")),
                "imported_exact_success_receipts": db.execute("SELECT count(*) FROM data WHERE kind='imported_success_receipt'").fetchone()[0]}
    result = {"scope": protocol["scope"], "complete": complete, "passing_gate_met": passed,
        "passing_gate": protocol["passing_gate"], "overall": overall,
        "by_evaluation_group": {g: summary([r for r in details if r["evaluation_group"] == g]) for g in ("fresh", "regression")},
        "by_dataset": {ds: summary([r for r in details if r["dataset"] == ds]) for ds in references},
        "new_api_usage_list_price_cny": actual, "unresolved_reservations_cny": reserved,
        "global_api_usage_list_price_cny": global_actual, "global_reservations_cny": global_reserve,
        "api": api,
        "frozen_hash_checks": {"cases": sha(HERE / "cases.frozen.jsonl") == protocol["cases_sha256"],
            "prompts": sha(HERE / "prompts.py") == protocol["prompt_source_sha256"],
            "examples": sha(HERE / "final_examples.json") == protocol["final_examples_sha256"],
            "old_results": sha(OLD_OUT / "CASE_RESULTS.jsonl") == protocol["old_results_sha256"],
            "child_prompts": sha(S02 / "prompts.py") == protocol["child_prompt_source_sha256"]}}
    write(OUT / "RESULTS.json", result)
    (OUT / "CASE_RESULTS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    regressions = [r for r in details if all(r["conditions"][c].get("f1") is not None for c in ("c03_both", "c04_single")) and r["conditions"]["c03_both"]["f1"] < r["conditions"]["c04_single"]["f1"]]
    write(OUT / "REGRESSIONS_FOR_NEXT_ROUND.json", [{**c, "evaluation_group": "regression"} for c in cases if any(r["qid"] == c["qid"] for r in regressions)])
    names = {"c00_old_prompts": "旧改写＋旧最终prompt", "c01_final_only": "旧改写＋新最终prompt", "c02_rewrite_only": "新改写＋旧最终prompt", "c03_both": "新改写＋新最终prompt", "c04_single": "单路＋新最终prompt"}
    text = ["双路prompt优化第1轮：新题＋下降题回归", "状态：" + ("全部完成" if complete else "执行中"),
        "本轮12道新题（三数据集各4题，历史成功/失败各2）＋1道上一轮下降题。新题不与上轮24题重合。全部5组使用同一批题；所有生成完成前不查看标准答案进行运行时选择。",
        "问题分解在生成前按query-only人工固定，假定分解正确，不评估自动分解器。本轮是小样本开发诊断。",
        "单路与新双路共用新最终prompt、完整句传递、原query全部Top10、goal执行和辅助位置。双路按原始与改写两路交替去重截取10篇；检索、子回答prompt和模型保持冻结。",
        "", "方案\t全部F1%\t新题F1%\t回归题F1%"]
    pct = lambda x: "未完成" if x is None else f"{100*x:.2f}"
    for label, name in names.items():
        text.append("\t".join([name, pct(overall["conditions"][label]["mean_f1_available"]),
            pct(result["by_evaluation_group"]["fresh"]["conditions"][label]["mean_f1_available"]),
            pct(result["by_evaluation_group"]["regression"]["conditions"][label]["mean_f1_available"])]))
    text.extend(["", "新双路对单路：" + json.dumps(comparison, ensure_ascii=False), "通过本轮门槛：" + str(passed),
        "逐题无下降仅适用于本批题，不保证未来任意问题；没有根据标准答案选择哪组输出。", "", "逐题结果"])
    for r in details:
        text.extend(["", r["dataset"] + " / " + r["evaluation_group"] + " / " + r["qid"], "原题：" + r["question"], "标准答案（离线评分）：" + str(r["gold_answer"])])
        for label, name in names.items():
            a = r["conditions"][label]
            text.append(f"{name}：{a.get('prediction')}；F1={a.get('f1')}")
        for label in ("c00_old_prompts", "c03_both", "c04_single"):
            p = OUT / "cases" / r["dataset"] / label / (r["qid"] + ".json")
            if p.exists():
                text.append(names[label] + "完整问答及改写：")
                for n in load(p)["nodes"]:
                    text.extend([f"  #{n['step_id']} {n['request']}", "  检索改写：" + n["rewritten_query"], "  完整回答：" + n["answer"]])
    text.extend(["", "新增请求按返回用量估算CNY：" + str(actual), "未结预留CNY：" + str(reserved),
        "精确成功payload复用有旧响应凭据，不代表端到端独立运行零费用。API成功逻辑响应与异常尝试分别记录。",
        "改动与用途：实验目录DECISIONS.txt、REWRITE_TEMPLATE.txt、FINAL_TEMPLATE.txt；运行痕迹：本目录cases/与api/。",
        "冻结检查：" + json.dumps(result["frozen_hash_checks"], ensure_ascii=False)])
    (OUT / "REPORT.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "gate": passed, "paired": comparison, "conditions": overall["conditions"], "new_cost": actual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
