"""Offline native scoring of fresh36 and the mixed60 cohort; no API calls."""

from collections import Counter, defaultdict
import hashlib
import json
import sqlite3

import numpy as np

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from .prepare import HERE, OUT, PILOT_OUT, PILOT_SPEC, INPUTS, ROOT, OUTPUT, rows, load, write


ARMS = ("single_original", "dual_original_rewritten")


def bootstrap_pairs(records, reference):
    paired = [r for r in records if r["conditions"][ARMS[1]].get("f1") is not None and (
        r["conditions"][ARMS[0]].get("f1") if reference == "single_original" else r["historical_references"][reference].get("f1")) is not None]
    diffs = [r["conditions"][ARMS[1]]["f1"] - (r["conditions"][ARMS[0]]["f1"] if reference == "single_original" else r["historical_references"][reference]["f1"]) for r in paired]
    result = {"pairs": len(paired), "missing_pairs": len(records) - len(paired), "mean_difference": float(np.mean(diffs)) if diffs else None,
        "gains": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
        "dual_mean_f1_common": float(np.mean([r["conditions"][ARMS[1]]["f1"] for r in paired])) if paired else None,
        "reference_mean_f1_common": float(np.mean([r["conditions"][ARMS[0]]["f1"] if reference == "single_original" else r["historical_references"][reference]["f1"] for r in paired])) if paired else None}
    if paired and all(r["sample_group"] == "fresh36_type_stratified_random" for r in paired):
        groups = defaultdict(list)
        for r, d in zip(paired, diffs):
            groups[(r["dataset"], r["type"])].append(d)
        rng = np.random.default_rng(20261017)
        totals = np.zeros(10000)
        for group in groups.values():
            values = np.asarray(group)
            totals += rng.choice(values, size=(10000, len(values)), replace=True).sum(axis=1)
        result["exploratory_paired95_ci"] = np.quantile(totals / len(paired), [0.025, 0.975]).tolist()
        result["ci_note"] = "10000 within-dataset/type bootstrap resamples; conditional on these small reference-plan strata, not a proof of general superiority"
    return result


def summarize(records):
    result = {"planned": len(records), "conditions": {}, "historical_references": {}}
    for key in ARMS + ("frozen_s02", "top10_v1", "react"):
        values = [r["conditions"][key] if key in ARMS else r["historical_references"][key] for r in records]
        valid = [v for v in values if v.get("f1") is not None]
        n = len(records)
        total = sum(v["f1"] for v in valid)
        item = {"available": len(valid), "missing": n - len(valid), "mean_f1_available": total / len(valid) if valid else None,
            "all_planned_f1_bounds": [total / n, (total + n - len(valid)) / n] if n else None,
            "statuses": dict(Counter(v.get("status", "unstarted") for v in values))}
        if key in ARMS:
            nodes = [node for v in valid for node in v["node_diagnostics"]]
            item.update({"all_gold_docs_shown": sum(v["all_gold_docs_shown_to_some_reader"] for v in valid),
                "rewrites": sum(n["rewrite_changed"] for n in nodes),
                "steps_with_added_gold_docs": sum(bool(n["added_gold_docs_actually_shown"]) for n in nodes),
                "steps_with_displaced_gold_docs": sum(bool(n["original_gold_docs_displaced"]) for n in nodes),
                "mean_final_prompt_tokens": float(np.mean([v["final_prompt_tokens"] for v in valid])) if valid else None,
                "all_original_top10_preserved": all(v["original_top10_preserved_in_final_prompt"] for v in valid)})
        result["conditions" if key in ARMS else "historical_references"][key] = item
    result["dual_minus_single"] = bootstrap_pairs(records, "single_original")
    result["dual_minus_frozen_s02"] = bootstrap_pairs(records, "frozen_s02")
    result["dual_minus_react"] = bootstrap_pairs(records, "react")
    common = [r for r in records if all(r["conditions"][a].get("f1") is not None for a in ARMS) and all(r["historical_references"][h].get("f1") is not None for h in ("frozen_s02", "react"))]
    result["four_way_common"] = {"pairs": len(common), "missing": len(records) - len(common), "mean_f1": {
        **{arm: float(np.mean([r["conditions"][arm]["f1"] for r in common])) if common else None for arm in ARMS},
        **{h: float(np.mean([r["historical_references"][h]["f1"] for r in common])) if common else None for h in ("frozen_s02", "react")}}}
    return result


def main():
    old = rows(PILOT_OUT / "CASE_RESULTS.jsonl")
    for row in old:
        row["sample_group"] = "old24_outcome_selected_diagnostic"
    historic = {(r["dataset"], r["qid"], r["method"]): r for r in rows(HERE / "historical_references_new36.frozen.jsonl")}
    mu_path = OUT / "BASELINE_MUSIQUE_RESULTS.json"
    if mu_path.exists():
        for row in load(mu_path)["results"]:
            historic[("musique", row["qid"], "frozen_s02")] = row
    react_path = OUT / "REACT_COMPLETION_RESULTS.json"
    if react_path.exists():
        for row in load(react_path)["results"]:
            historic[("musique", row["qid"], "react")] = row
    mapping = load(OUT / "API_STORES.json") if (OUT / "API_STORES.json").exists() else {ds: "api/" + ds for ds in ("hotpotqa", "2wiki", "musique")}
    failures = load(OUT / "TERMINAL_FAILURES.json") if (OUT / "TERMINAL_FAILURES.json").exists() else []
    refs = {ds: {r["qid"]: r for r in rows(INPUTS / ds / "formal.references.local.jsonl")} for ds in ("hotpotqa", "2wiki", "musique")}
    scorers = {ds: metric(ds) for ds in refs}
    fresh = []
    for case in rows(HERE / "cases.frozen.jsonl"):
        ds, qid = case["dataset"], case["qid"]
        ref = refs[ds][qid]
        gold = set(ref["positive_doc_ids"])
        row = {k: case[k] for k in ("dataset", "qid", "question", "type", "sample_group")}
        row["gold_answer"] = ref["answer"]
        row["historical_references"] = {key: historic.get((ds, qid, key), {"f1": None, "prediction": None, "status": "unstarted"}) for key in ("frozen_s02", "top10_v1", "react")}
        row["conditions"] = {}
        for arm in ARMS:
            path = OUT / "cases" / ds / arm / (qid + ".json")
            if not path.exists():
                known = next((r for r in failures if r["dataset"] == ds and r["qid"] == qid and arm in r["missing_conditions"]), None)
                row["conditions"][arm] = {"status": known["status"] if known else "unstarted_or_incomplete", "f1": None, "prediction": None}
                continue
            raw = load(path)
            with sqlite3.connect((OUT / mapping[ds] / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
                returned = db.execute("SELECT result FROM requests WHERE id=?", (raw["request_id"],)).fetchone()
                tokens = json.loads(returned[0])["usage"]["prompt_tokens"]
            shown = set(raw["original_top10_doc_ids"])
            node_details = []
            for node in raw["nodes"]:
                original = {d["doc_id"] for d in node["original_retrieval"]["docs"]}
                reader = {d["doc_id"] for d in node["reader_packet"]["docs"]}
                shown.update(reader)
                node_details.append({"step_id": node["step_id"], "question": node["request"], "rewritten_query": node["rewritten_query"], "answer": node["answer"],
                    "rewrite_changed": node["rewritten_retrieval"] is not None,
                    "added_gold_docs_actually_shown": sorted((reader - original) & gold),
                    "original_gold_docs_displaced": sorted((original - reader) & gold)})
            row["conditions"][arm] = {"status": raw["status"], "prediction": raw["prediction"], **scorers[ds](raw["prediction"], ref),
                "gold_document_recall": len(gold & shown) / len(gold) if gold else None,
                "all_gold_docs_shown_to_some_reader": gold.issubset(shown),
                "original_top10_preserved_in_final_prompt": all(d["block"] in raw["final_context"] for d in raw["nodes"][-1]["original_retrieval"]["docs"]),
                "final_prompt_tokens": tokens, "node_diagnostics": node_details}
        fresh.append(row)
    combined = old + fresh
    with sqlite3.connect((OUTPUT / "budget.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        local_actual, local_reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()
        global_actual, global_reserve = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges").fetchone()
    api = {}
    for checkpoint in OUT.glob("**/checkpoint.sqlite3"):
        with sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True) as db:
            api[str(checkpoint.parent.relative_to(OUT))] = {"requests": dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status")), "attempts": dict(db.execute("SELECT status,count(*) FROM attempts GROUP BY status"))}
    frozen = load(OUT / "EXECUTION_FROZEN.json")
    checks = {"cases": hashlib.sha256((HERE / "cases.frozen.jsonl").read_bytes()).hexdigest() == frozen["cases_sha256"],
        "runner": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest() == frozen["code_sha256"],
        "protocol": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"],
        "prompts": hashlib.sha256((ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/prompts.py").read_bytes()).hexdigest() == frozen["child_and_final_prompts_sha256"]}
    if (OUT / "NATIVE_CONTINUATION_FROZEN.json").exists():
        continued = load(OUT / "NATIVE_CONTINUATION_FROZEN.json")
        checks["native_continuation"] = hashlib.sha256((HERE / "resume_native.py").read_bytes()).hexdigest() == continued["continuation_sha256"]
    report = {"scope": "old24 purposive diagnostics plus36 fresh type-stratified random cases;60 total,20 per dataset",
        "fresh36": summarize(fresh), "old24": summarize(old), "total60": summarize(combined),
        "fresh_by_dataset": {ds: summarize([r for r in fresh if r["dataset"] == ds]) for ds in refs},
        "total_by_dataset": {ds: summarize([r for r in combined if r["dataset"] == ds]) for ds in refs},
        "fresh_by_type": {ds + "/" + kind: summarize([r for r in fresh if r["dataset"] == ds and r["type"] == kind]) for ds, kind in sorted({(r["dataset"], r["type"]) for r in fresh})},
        "expansion_returned_usage_list_price_cny": local_actual, "expansion_unresolved_reserved_cny": local_reserve,
        "global_returned_usage_list_price_cny": global_actual, "global_unresolved_reserved_cny": global_reserve,
        "api_execution": api, "frozen_hash_checks": checks,
        "limitations": ["Question-only reference decompositions are an assumed ideal condition, not independent human oracle labels.", "Frozen S02 uses actual4B outputs; comparison includes decomposition and execution changes.", "Matched single/dual comparison isolates the optional rewriting-and-extra-retrieval flow, with added compute.", "No independent complete-sentence or mandatory-binding ablation.", "Report available denominators and full-planned bounds; API missingness is not F1 zero.", "Gold-document visibility does not prove factual entailment or final evidence coverage."]}
    write(OUT / "RESULTS.json", report)
    (OUT / "CASE_RESULTS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    render_report(report, combined)
    print(json.dumps({k: report[k] for k in ("fresh36", "total60", "expansion_returned_usage_list_price_cny", "frozen_hash_checks")}, ensure_ascii=False))


def render_report(result, details):
    def pct(value):
        return "缺失" if value is None else f"{value * 100:.2f}"
    complete = (OUT / "GENERATION_TERMINAL.json").exists() and (OUT / "REACT_COMPLETION_RESULTS.json").exists()
    lines = ["多粒度子问题＋完整句＋双路检索：扩展至60题", "状态：" + ("全部计划执行已结束；服务缺答见分母" if complete else "执行中或存在未完成请求，结果尚不完整"), "", "设置：每数据集保留旧8题，新增12题；新36题按类型分层固定随机抽取，不按历史成败选题。旧24题依然是历史F1端点各半的诊断样本。", "新增参考分解在API调用前按原query编写和冻结；自包含问句、完整句向后传递、goal检索回答、另答最终原query。提示词、检索资源、两路交替去重Top10和最终原query全部Top10均沿用原试验。", "仅执行新增36题；旧24题原样复用。Hotpot固定原始子问题调用既有retrieve_many预取，两条控制问句Top10顺序一致。2Wiki批量校验发现两个近似同分文档顺序互换，在发送该数据集任何候选回答请求前停止；归档批量结果，2Wiki与Mu改回上一轮逐条检索，新命名空间内控制顺序均通过。动态改写始终在线。", "主对照为冻结s02_child4与ReAct(k10,max_search5)。Mu新12题补跑原S02；ReAct复用52条已完成成绩，仅补跑剩余8题；不重新选参数或改提示词。", "已知改写越界等问题本轮不修复。问题式参考分解假定正确，不是独立人工金标准。歧义原题保持原句，未根据标准答案改写。", "", "Answer F1（%，四组共同可评分题目）", "样本\t共同题数/计划\t冻结S02\tReAct\t单路\t双路"]
    for name, summary in [("新增36", result["fresh36"]), ("旧24", result["old24"]), ("合计60", result["total60"]), *[("新增/" + ds, s) for ds, s in result["fresh_by_dataset"].items()], *[("合计/" + ds, s) for ds, s in result["total_by_dataset"].items()]]:
        vals = [summary["historical_references"]["frozen_s02"], summary["historical_references"]["react"], *[summary["conditions"][a] for a in ARMS]]
        common = summary["four_way_common"]
        lines.append("\t".join([name, f"{common['pairs']}/{summary['planned']}", *[pct(common["mean_f1"][key]) for key in ("frozen_s02", "react", *ARMS)]]))
        lines.append("  各组实际可评分题数(S02/ReAct/单路/双路)：" + "/".join(str(v["available"]) for v in vals) + ";各自可用均值和全计划上下界保存在RESULTS.json。")
        for ref in ("single", "frozen_s02", "react"):
            p = summary["dual_minus_" + ref]
            interval = "不对混合诊断样本作总体区间声明" if "exploratory_paired95_ci" not in p else str([round(100 * v, 2) for v in p["exploratory_paired95_ci"]]) + "个百分点"
            lines.append(f"  双路vs{ref}：配对{p['pairs']}题，共同题均值={pct(p['dual_mean_f1_common'])} vs {pct(p['reference_mean_f1_common'])}；F1差{pct(p['mean_difference'])}个百分点，上/平/降={p['gains']}/{p['ties']}/{p['losses']}；探索性95%区间=" + interval)
    if complete:
        lines.extend(["", "主要结果解释：",
            "新增36题的共同可评分35题：双路63.84%，冻结S02为66.65%，ReAct为62.99%，单路43.92%。双路减S02为-2.80个百分点，减ReAct为+0.85个百分点，两个探索性区间均跨0。没有证据支持将冻结最佳版本更新为新方案。",
            "双路减单路为+19.92个百分点，9题上升、26题不变、0题下降，探索性95%区间[11.36,29.46]个百分点。可确认本轮参考分解下，改写加额外检索改善了独立子问题单路方案；不能把该收益等同于对S02/ReAct的优势。",
            "合计60题共同59题的双路均值65.27%，高于S02的59.87%和ReAct的63.35%；其中含旧24题的成败端点定向抽样，不能仅用这一混合均值声称稳定总体优势。",
            "一题2hop__124412_402641在S02、单路与双路遇到提供方内容审核拒绝，保留计划分母，不计为F1零。ReAct对这题为动作协议失败，按其原协议计0；四组共同比较仍排除此题。",
            "4hop1__151650_5274_458768_33637的双路最终响应返回了64token并因length截断。原包装器要求stop而未保存题级文件，现仅从相同payload的已返回记录离线重建，按原始文本计算F1=0，保留success_truncated状态；没有新调用、补写答案或修改提示词。恢复证据见OFFLINE_FINAL_RECOVERY.json，原包装器异常也保留。",
            "置信区间为新增可评分题在数据集/类型层内的一万次配对bootstrap，属于小样本、条件性的探索性区间。"])
    lines.extend(["", "缺失答案保留分母，不计为0；RESULTS.json另存全计划F1上下界。主比较应优先看新增36题的同题配对结果，不能用不同可用题数的均值直接相减。", "S02使用真实4B分解；新两组使用预先编写的参考分解，差异并非仅一个执行组件。双路vs单路也是增加调用和检索预算后的比较。", "", "已有额外历史参照（仅可用题，不与完整60直接排名）"])
    for label in ("top10_v1",):
        s = result["total60"]["historical_references"][label]
        lines.append(f"{label}: F1={pct(s['mean_f1_available'])}%，可用{s['available']}/60，缺失{s['missing']}")
    notes = OUT / "DIAGNOSTIC_NOTES.jsonl"
    if notes.exists():
        lines.extend(["", "实际痕迹核查："])
        for note in rows(notes):
            lines.append(json.dumps(note, ensure_ascii=False, indent=2))
    lines.extend(["", "逐题输出（完整提示词、文档、前置问答和响应见各自cases与api目录；旧24题见原pilot目录）"])
    for row in details:
        lines.extend(["", row["dataset"] + " / " + row["qid"] + " / " + row["sample_group"], "原query：" + row["question"], "标准答案（只用于离线评分）：" + str(row["gold_answer"])])
        baseline = row["historical_references"]["frozen_s02"]
        lines.append(f"S02：{baseline.get('prediction')}；F1={baseline.get('f1')}；状态={baseline.get('status')}")
        react = row["historical_references"]["react"]
        lines.append(f"ReAct：{react.get('prediction')}；F1={react.get('f1')}；状态={react.get('status')}")
        for arm in ARMS:
            a = row["conditions"][arm]
            lines.append(f"{arm}：{a.get('prediction')}；F1={a.get('f1')}；状态={a.get('status')}")
            for n in a.get("node_diagnostics", []):
                lines.extend([f"  #{n['step_id']} {n['question']}", "  改写：" + n["rewritten_query"], "  完整句：" + n["answer"], "  新增/丢失标注文档：" + str(n["added_gold_docs_actually_shown"]) + " / " + str(n["original_gold_docs_displaced"])])
    lines.extend(["", "扩展部分返回用量估算费用CNY：" + str(result["expansion_returned_usage_list_price_cny"]), "扩展未结预留CNY：" + str(result["expansion_unresolved_reserved_cny"]), "全项目返回用量估算/未结预留CNY：" + str([result["global_returned_usage_list_price_cny"], result["global_unresolved_reserved_cny"]]), "冻结检查：" + str(result["frozen_hash_checks"]), "调用与异常状态：" + json.dumps(result["api_execution"], ensure_ascii=False), "本报告不将词级F1错误直接解释为幻觉，不作SOTA或总体优势声明。"])
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
