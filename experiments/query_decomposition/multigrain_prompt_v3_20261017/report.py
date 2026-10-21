"""Round2 matched final-prompt comparison and a strict zero-regression gate."""

import json
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write
from experiments.query_decomposition.multigrain_prompt_v2_20261017.report import paired, mean
from .prepare import HERE, OUT, ROOT, OUTPUT, PREVIOUS, PREVIOUS_OUT, S02, sha


LABELS = {"dual_final_v2": "双路＋上一版最终prompt", "dual_final_v3": "双路＋姓名修订prompt",
    "single_final_v2": "单路＋上一版最终prompt", "single_final_v3": "单路＋姓名修订prompt"}


def main():
    protocol = load(HERE / "protocol.json")
    cases = rows(HERE / "cases.frozen.jsonl")
    references, scorers = {}, {}
    for dataset in ("hotpotqa", "2wiki", "musique"):
        references[dataset] = {r["qid"]: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.references.local.jsonl")}
        scorers[dataset] = metric(dataset)
    details = []
    for case in cases:
        ref = references[case["dataset"]][case["qid"]]
        row = {k: case[k] for k in ("qid", "dataset", "question", "evaluation_group", "historical_f1")}
        row["gold_answer"] = ref["answer"]
        row["conditions"] = {}
        for label in LABELS:
            path = OUT / "cases" / case["dataset"] / label / (case["qid"] + ".json")
            if not path.exists():
                row["conditions"][label] = {"status": "unstarted_or_incomplete", "f1": None}
                continue
            result = load(path)
            with sqlite3.connect((OUT / "api" / case["dataset"] / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
                raw = db.execute("SELECT result FROM requests WHERE id=?", (result["request_id"],)).fetchone()[0]
                usage = json.loads(raw).get("usage", {})
            original = result["nodes"][-1]["original_retrieval"]["docs"]
            row["conditions"][label] = {"status": result["status"], "prediction": result["prediction"],
                **scorers[case["dataset"]](result["prediction"], ref), "final_prompt_tokens": usage.get("prompt_tokens"),
                "original_top10_preserved": all(d["block"] in result["final_context"] for d in original),
                "changed_rewrites": sum(n["rewritten_retrieval"] is not None for n in result["nodes"])}
        for left, right in (("dual_final_v2", "dual_final_v3"), ("single_final_v2", "single_final_v3")):
            a = OUT / "cases" / case["dataset"] / left / (case["qid"] + ".json")
            b = OUT / "cases" / case["dataset"] / right / (case["qid"] + ".json")
            if a.exists() and b.exists():
                assert load(a)["final_context"] == load(b)["final_context"]
                assert load(a)["nodes"] == load(b)["nodes"]
        details.append(row)
    def summarize(records):
        conditions = {}
        for label in LABELS:
            available = [r["conditions"][label] for r in records if r["conditions"][label].get("f1") is not None]
            conditions[label] = {"available": len(available), "missing": len(records) - len(available),
                "mean_f1_available": mean([r["f1"] for r in available]), "mean_em_available": mean([r["em"] for r in available]),
                "mean_final_prompt_tokens": mean([r["final_prompt_tokens"] for r in available]),
                "original_top10_preserved": all(r["original_top10_preserved"] for r in available)}
        return {"planned": len(records), "conditions": conditions,
            "dual_minus_single_v3": paired(records, "dual_final_v3", "single_final_v3"),
            "dual_minus_single_v2": paired(records, "dual_final_v2", "single_final_v2"),
            "final_change_on_dual": paired(records, "dual_final_v3", "dual_final_v2"),
            "final_change_on_single": paired(records, "single_final_v3", "single_final_v2")}
    overall = summarize(details)
    complete = all(v["missing"] == 0 for v in overall["conditions"].values())
    pair = overall["dual_minus_single_v3"]
    passed = complete and pair["losses"] == 0 and pair["mean_difference"] > 0
    with sqlite3.connect((OUTPUT / "budget.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        actual, reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()
        global_actual, global_reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges").fetchone()
    result = {"scope": protocol["scope"], "complete": complete, "passing_gate_met": passed, "overall": overall,
        "by_evaluation_group": {g: summarize([r for r in details if r["evaluation_group"] == g]) for g in ("fresh", "regression")},
        "by_dataset": {ds: summarize([r for r in details if r["dataset"] == ds]) for ds in references},
        "new_api_usage_list_price_cny": actual, "unresolved_reservations_cny": reserved,
        "global_api_usage_list_price_cny": global_actual, "global_reservations_cny": global_reserved,
        "frozen_hash_checks": {"prompts": sha(HERE / "prompts.py") == protocol["new_prompt_sha256"],
            "examples": sha(HERE / "final_examples.json") == protocol["new_examples_sha256"],
            "v2_prompts": sha(PREVIOUS / "prompts.py") == protocol["v2_prompts_sha256"],
            "v2_examples": sha(PREVIOUS / "final_examples.json") == protocol["v2_examples_sha256"],
            "cases": sha(HERE / "cases.frozen.jsonl") == protocol["cases_sha256"],
            "previous_results": sha(PREVIOUS_OUT / "CASE_RESULTS.jsonl") == protocol["previous_results_sha256"],
            "child_prompts": sha(S02 / "prompts.py") == protocol["child_prompts_sha256"]}}
    write(OUT / "RESULTS.json", result)
    (OUT / "CASE_RESULTS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    losses = set(pair["loss_qids"])
    write(OUT / "REGRESSIONS_FOR_NEXT_ROUND.json", [{**r, "evaluation_group": "regression"} for r in cases if r["qid"] in losses])
    pct = lambda value: "未完成" if value is None else f"{100*value:.2f}"
    text = ["prompt优化第2轮：固定改写器，只修订最终姓名选择", "状态：" + ("全部完成" if complete else "执行中"),
        "12道新题＋上一轮1道下降题。新题不与此前24＋12题重合；部分Mu题与旧题共享语料事实/模板，不能当独立泛化测试。",
        "Mu历史分组含2道F1=1、1道F1=0、1道F1=0.6667；未把部分分伪装成零分。各组使用相同query-only参考分解，假定正确而非独立人工金标。",
        "改写器/检索/子回答/完整句传递/goal执行/原始Top10/辅助位置全部沿用。每条单路或双路历史分别送入旧v2与新v3最终prompt，直接隔离最终prompt效果。",
        "", "方案\t全部F1%\t新题F1%\t回归题F1%"]
    for label, name in LABELS.items():
        text.append("\t".join([name, pct(overall["conditions"][label]["mean_f1_available"]),
            pct(result["by_evaluation_group"]["fresh"]["conditions"][label]["mean_f1_available"]),
            pct(result["by_evaluation_group"]["regression"]["conditions"][label]["mean_f1_available"])]))
    text.extend(["", "同一最终prompt下双路减单路：" + json.dumps(pair, ensure_ascii=False),
        "最终prompt改动在双路上的效果：" + json.dumps(overall["final_change_on_dual"], ensure_ascii=False),
        "最终prompt改动在单路上的效果：" + json.dumps(overall["final_change_on_single"], ensure_ascii=False),
        "通过本批门槛：" + str(passed), "零下降只适用于已评估样本，不保证未来任意题。没有用标准答案选择运行时输出。", "", "逐题完整执行："])
    for row in details:
        text.extend(["", row["dataset"] + "/" + row["evaluation_group"] + "/" + row["qid"], "原题：" + row["question"], "标准答案（仅离线评分）：" + str(row["gold_answer"])])
        for label, name in LABELS.items():
            a = row["conditions"][label]
            text.append(f"{name}：{a.get('prediction')}；F1={a.get('f1')}")
        for label in ("dual_final_v3", "single_final_v3"):
            p = OUT / "cases" / row["dataset"] / label / (row["qid"] + ".json")
            if p.exists():
                text.append(LABELS[label] + "：")
                for n in load(p)["nodes"]:
                    text.extend([f"#{n['step_id']} {n['request']}", "检索改写：" + n["rewritten_query"], "完整回答：" + n["answer"]])
    text.extend(["", "新增返回用量估算CNY：" + str(actual), "未结预留CNY：" + str(reserved),
        "所有修改及用途见实验目录DECISIONS.txt；API请求、原始输出和缓存凭据保存在本目录api/，逐题材料在cases/。",
        "冻结检查：" + json.dumps(result["frozen_hash_checks"], ensure_ascii=False)])
    (OUT / "REPORT.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "gate": passed, "paired": pair, "conditions": overall["conditions"], "cost": actual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
