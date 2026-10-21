"""Read-only native-metric and resource comparison to the frozen local ReAct baseline."""

import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import ROOT, OUTPUT, CAMPAIGN, load, rows, write


def average(values):
    return sum(values) / len(values) if values else None


def logical_usage(request_ids, checkpoint):
    usage, returned_model = [], set()
    with sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for rid in request_ids:
            row = db.execute("SELECT payload,result FROM requests WHERE id=?", (rid,)).fetchone()
            if row is None or row[1] is None:
                return {"complete_usage": False}
            payload, response = json.loads(row[0]), json.loads(row[1])
            if response.get("status") != "success":
                return {"complete_usage": False}
            returned_model.add(response.get("returned_model"))
            usage.append(response.get("usage", {}))
    prompt = sum(r.get("prompt_tokens", 0) for r in usage)
    completion = sum(r.get("completion_tokens", 0) for r in usage)
    return {"complete_usage": True, "logical_calls": len(request_ids), "prompt_tokens": prompt,
        "completion_tokens": completion, "logical_standalone_list_price_cny": prompt * .8e-6 + completion * 2e-6,
        "returned_models": sorted(returned_model)}


def compare(round_name, primary, single):
    spec = ROOT / "experiments/query_decomposition" / round_name
    out = OUTPUT / round_name
    cases = rows(spec / "cases.frozen.jsonl")
    details = []
    config = load(CAMPAIGN / "CAMPAIGN_FROZEN.json")["protocol"]["methods"]["b03_react_dense"]["default"]
    for dataset in ("hotpotqa", "2wiki", "musique"):
        references = {r["qid"]: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.references.local.jsonl")}
        scorer = metric(dataset)
        source = CAMPAIGN / dataset / "formal/b03_react_dense/k10_s5"
        checkpoint = source / "checkpoint.sqlite3"
        call_ids = {}
        for event in rows(source / "trace.jsonl"):
            if event.get("event") == "logical_api_call":
                call_ids.setdefault(event.get("qid"), {})[(event["stage"], event["request_id"])] = event["request_id"]
        with sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for case in [c for c in cases if c["dataset"] == dataset]:
                row = {k: case[k] for k in ("dataset", "qid", "question", "evaluation_group")}
                record = db.execute("SELECT value FROM data WHERE kind='prediction' AND key=?", (case["qid"],)).fetchone()
                prediction = json.loads(record[0]) if record else None
                if prediction:
                    assert prediction["question"] == case["question"]
                valid = prediction and prediction.get("status") in {"success", "algorithm_failure"} and isinstance(prediction.get("prediction"), str)
                row["react"] = {"status": prediction.get("status") if prediction else "unstarted", "prediction": prediction.get("prediction") if prediction else None,
                    **(scorer(prediction["prediction"], references[case["qid"]]) if valid else {"f1": None, "em": None})}
                if valid:
                    row["react"].update(logical_usage(list(call_ids.get(case["qid"], {}).values()), checkpoint))
                    row["react"]["search_actions"] = prediction.get("searches")
                for name, condition in (("ours", primary), ("single", single)):
                    path = out / "cases" / dataset / condition / (case["qid"] + ".json")
                    if not path.exists():
                        row[name] = {"status": "incomplete", "f1": None, "em": None}
                        continue
                    result = load(path)
                    ids = []
                    for node in result["nodes"]:
                        ids.extend(node[key] for key in ("rewrite_request_id", "answer_request_id", "sentence_format_request_id") if node.get(key))
                    ids.append(result["request_id"])
                    row[name] = {"status": result["status"], "prediction": result["prediction"], **scorer(result["prediction"], references[case["qid"]]),
                        **logical_usage(ids, out / "api" / dataset / "checkpoint.sqlite3"),
                        "unique_retrieval_questions": len({case["question"]} | {n["request"] for n in result["nodes"]} | {n["rewritten_query"] for n in result["nodes"]})}
                details.append(row)
    def summarize(records):
        paired = [r for r in records if r["react"]["f1"] is not None and r["ours"]["f1"] is not None]
        differences = [r["ours"]["f1"] - r["react"]["f1"] for r in paired]
        methods = {}
        for name in ("ours", "single", "react"):
            values = [r[name] for r in records if r[name]["f1"] is not None]
            usage = [r for r in values if r.get("complete_usage")]
            methods[name] = {"available": len(values), "missing": len(records) - len(values),
                "mean_f1_available": average([r["f1"] for r in values]), "mean_em_available": average([r["em"] for r in values]),
                "usage_available": len(usage), "mean_logical_api_calls": average([r["logical_calls"] for r in usage]),
                "mean_logical_standalone_cny": average([r["logical_standalone_list_price_cny"] for r in usage])}
        return {"planned": len(records), "methods": methods, "paired_ours_minus_react": {"pairs": len(paired),
            "mean_difference": average(differences), "gains": sum(d > 0 for d in differences),
            "ties": sum(d == 0 for d in differences), "losses": sum(d < 0 for d in differences),
            "loss_qids": [r["qid"] for r, d in zip(paired, differences) if d < 0]}}
    report = {"round": round_name, "our_condition": primary, "single_condition": single, "react_config": config,
        "comparison": "same questions, frozen shared corpora/BGE retriever and qwen-plus-2025-12-01; each method keeps its own prompts/search policy",
        "scope_limit": "ours uses manually fixed query-only reference plans in a diagnostic pilot; no claim about an automatic decomposer, equal search/token budget, official ReAct reproduction, SOTA, or broad dominance",
        "cost_limit": "logical standalone API cost uses all per-case calls including reused successes, not newly paid experiment cost; excludes local retrieval wall time and plan authoring",
        "overall": summarize(details), "by_dataset": {ds: summarize([r for r in details if r["dataset"] == ds]) for ds in ("hotpotqa", "2wiki", "musique")},
        "by_evaluation_group": {g: summarize([r for r in details if r["evaluation_group"] == g]) for g in ("fresh", "regression")}}
    write(out / "REACT_COMPARISON.json", report)
    (out / "REACT_CASE_RESULTS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    pct = lambda x: "未完成" if x is None else f"{100*x:.2f}"
    text = ["与冻结本地ReAct适配版的同题对照", round_name,
        "这次为人工固定参考分解条件下的诊断实验。ReAct配置k=10,max_search=5；两者使用共同语料/BGE/API模型，但prompt、搜索策略及计算预算不同。不能据此认定自动方法或官方全数据集SOTA。",
        "范围\t题数\t我们F1%\t单路F1%\tReAct F1%\t对ReAct上升/持平/下降"]
    for name, value in [("全部", report["overall"]), *report["by_dataset"].items(), *report["by_evaluation_group"].items()]:
        pair = value["paired_ours_minus_react"]
        text.append("\t".join([name, str(value["planned"]), *[pct(value["methods"][k]["mean_f1_available"]) for k in ("ours", "single", "react")], f"{pair['gains']}/{pair['ties']}/{pair['losses']}"]))
    text.extend(["", "成本为完整逻辑调用按既有单价估算，不把缓存复用视为独立方法免费运行。", json.dumps(report["overall"]["methods"], ensure_ascii=False),
        "", "逐题："])
    for r in details:
        text.append(json.dumps(r, ensure_ascii=False))
    (out / "REACT_COMPARISON.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"round": round_name, "overall": report["overall"], "by_dataset": report["by_dataset"]}, ensure_ascii=False))


if __name__ == "__main__":
    compare("multigrain_prompt_v2_20261017", "c03_both", "c04_single")
    compare("multigrain_prompt_v3_20261017", "dual_final_v3", "single_final_v3")
