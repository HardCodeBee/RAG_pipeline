"""Reconcile literal authored reviews, frozen controls and exact API payloads."""

from collections import Counter
import json
from pathlib import Path

from .run_api_pilot import HERE, OUT as CONTROL, rows, sha, summarize
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc, write_json


DESTINATION = OUTPUT / "multigrain_decomposer_prompt_followup_20261017"
CHECKS = {"F": "faithfulness_and_unknowns", "S": "self_containment", "C": "supporting_information_coverage", "U": "distinct_usefulness"}
ARMS = ("v2", "v3", "v4", "v5")


def authored_record(case, arm):
    root = OUTPUT / f"multigrain_decomposer_prompt_{arm}_20261017"
    document = json.loads((HERE / f"AUTHORED_{arm}.json").read_text(encoding="utf-8"))
    mapping = {r["qid"]: r for r in document["cases"]}
    assert len(mapping) == 36
    judgment = mapping[case["qid"]]
    path = root / "cases" / arm / case["dataset"] / (case["qid"] + ".json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["question"] == case["question"] and raw["qid"] == case["qid"]
    code = judgment["code"]
    if code == "X":
        assert raw["status"] != "success"
        verdict = "unavailable"
        checks = {name: "not_scored" for name in CHECKS.values()}
    else:
        assert raw["status"] == "success" and raw["finish_reason"] == "stop" and raw["mechanical"]["array_of_nonempty_strings"]
        verdict = "pass" if code == "P" else "uncertain" if code.startswith("?") else "fail"
        checks = {name: "pass" for name in CHECKS.values()}
        if code != "P":
            for letter in code.lstrip("?"):
                checks[CHECKS[letter]] = "uncertain" if code.startswith("?") else "fail"
    return {"verdict": verdict, "checks": checks, "reason": judgment["reason"],
            "literal_answer": raw.get("answer"), "request_id": raw["request_id"],
            "raw_result_sha256": sha(path), "raw_file": str(path.relative_to(ROOT))}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    cases = rows(CONTROL / "cases.frozen.jsonl")
    frozen = {r["qid"]: r for r in rows(CONTROL / "AUTHORED_SEMANTIC_REVIEW.jsonl")}
    pairs = []
    for case in cases:
        pair = {k: case[k] for k in ("qid", "dataset", "question", "type")}
        pair["v2"] = frozen[case["qid"]]["v2"]
        for arm in ARMS[1:]:
            pair[arm] = authored_record(case, arm)
        pairs.append(pair)
    audit = {}
    expenses = {}
    statistics = {}
    for arm in ARMS:
        root = CONTROL if arm == "v2" else OUTPUT / f"multigrain_decomposer_prompt_{arm}_20261017"
        protocol = json.loads((root / "PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
        assert sha(root / "cases.frozen.jsonl") == sha(CONTROL / "cases.frozen.jsonl")
        if arm != "v2":
            assert protocol["old_control_review_sha256"] == sha(CONTROL / "AUTHORED_SEMANTIC_REVIEW.jsonl")
            assert protocol["semantic_rubric_sha256"] == sha(HERE / "PILOT_REVIEW_RULES.txt")
        request_rows = []
        for path in (root / "api" / arm).glob("*/*/requests.jsonl"):
            request_rows += rows(path)
        assert len(request_rows) == 36
        seen = set()
        for request in request_rows:
            payload = request["payload"]
            assert payload["model"] == "qwen-plus-2025-12-01" and payload["temperature"] == 0 and payload["max_tokens"] == 1024
            assert payload["extra_body"] == {"enable_thinking": False}
            assert [m["role"] for m in payload["messages"]] == ["system", "user"]
            assert payload["messages"][0]["content"] == (HERE / f"TRAINING_PROMPT_{arm}.txt").read_text(encoding="utf-8")
            seen.add(payload["messages"][1]["content"])
        assert seen == {r["question"] for r in cases}
        audit[arm] = {"stored_requests": len(request_rows), "statuses": dict(Counter(r["status"] for r in request_rows)),
                      "same_model_decoding_roles_original_questions": True,
                      "prompt_file": str((HERE / f"TRAINING_PROMPT_{arm}.txt").relative_to(ROOT)),
                      "prompt_sha256": sha(HERE / f"TRAINING_PROMPT_{arm}.txt")}
        if arm == "v2":
            values = [json.loads((root / "cases/v2" / r["dataset"] / (r["qid"] + ".json")).read_text(encoding="utf-8")) for r in cases]
            expenses[arm] = {"usage_list_price_cny": sum(r.get("usage", {}).get("prompt_tokens",0)*.8e-6 + r.get("usage", {}).get("completion_tokens",0)*2e-6 for r in values), "scope": "reused control; not new spending"}
            mechanically = {"input_tokens": sum(r["usage"]["prompt_tokens"] for r in values), "output_tokens": sum(r["usage"]["completion_tokens"] for r in values), "complete_success": 36}
        else:
            state = summarize(pilot_out=root, arm_prompts={arm: f"TRAINING_PROMPT_{arm}.txt"})
            mechanically = state["metrics"][arm]
            expenses[arm] = {"usage_list_price_cny": state["pilot_usage_list_price_cny"], "unknown_reserved_cny": state["unknown_reserved_cny"]}
        statistics[arm] = {"planned": 36, "complete": mechanically["complete_success"],
                           "verdicts": dict(Counter(r[arm]["verdict"] for r in pairs)),
                           "pass_rate_full_denominator": sum(r[arm]["verdict"] == "pass" for r in pairs)/36,
                           "failed_checks": {name: sum(r[arm]["checks"][name] == "fail" for r in pairs) for name in CHECKS.values()},
                           "mean_input_tokens_returned": mechanically["input_tokens"]/mechanically["complete_success"],
                           "mean_output_tokens_returned": mechanically["output_tokens"]/mechanically["complete_success"],
                           "by_dataset": {ds: dict(Counter(r[arm]["verdict"] for r in pairs if r["dataset"] == ds)) for ds in ("hotpotqa", "2wiki", "musique")}}
    common = [r for r in pairs if all(r[a]["verdict"] != "unavailable" for a in ARMS)]
    common_summary = {"n": len(common), "passes": {a: sum(r[a]["verdict"] == "pass" for r in common) for a in ARMS},
                      "self_containment_failures": {a: sum(r[a]["checks"]["self_containment"] == "fail" for r in common) for a in ARMS}}
    comparisons = []
    for before, after in (("v2","v3"),("v3","v4"),("v3","v5"),("v2","v5")):
        subset = [r for r in pairs if all(r[a]["verdict"] != "unavailable" for a in (before,after))]
        comparisons.append({"before": before, "after": after, "n": len(subset),
                            "before_pass": sum(r[before]["verdict"] == "pass" for r in subset),
                            "after_pass": sum(r[after]["verdict"] == "pass" for r in subset),
                            "gained_pass": [r["qid"] for r in subset if r[before]["verdict"] != "pass" and r[after]["verdict"] == "pass"],
                            "lost_pass": [r["qid"] for r in subset if r[before]["verdict"] == "pass" and r[after]["verdict"] != "pass"],
                            "fixed_self_containment": [r["qid"] for r in subset if r[before]["checks"]["self_containment"] == "fail" and r[after]["checks"]["self_containment"] == "pass"],
                            "new_self_containment_failure": [r["qid"] for r in subset if r[before]["checks"]["self_containment"] == "pass" and r[after]["checks"]["self_containment"] == "fail"]})
    recommendation = max(ARMS, key=lambda a: (common_summary["passes"][a], -common_summary["self_containment_failures"][a], -statistics[a]["mean_input_tokens_returned"]))
    result = {"utc": utc(), "scope": "same36 exposed-set prompt development diagnostic", "reviewer": "one shared nonblind Codex context; not independent human gold",
              "unchanged_rubric": str((HERE / "PILOT_REVIEW_RULES.txt").relative_to(ROOT)),
              "statistics": statistics, "all_arm_common": common_summary, "comparisons": comparisons,
              "candidate_by_common_passes_then_self_containment_then_length": recommendation,
              "selection_limit": "Within-set candidate only; not a confirmed 90-percent decomposer or downstream RAG gain.",
              "expenses": expenses, "new_usage_list_price_cny": sum(expenses[a]["usage_list_price_cny"] for a in ARMS[1:]),
              "new_unknown_reserved_cny": sum(expenses[a]["unknown_reserved_cny"] for a in ARMS[1:]),
              "audit": audit}
    write_json(DESTINATION / "RESULTS.json", result)
    (DESTINATION / "AUTHORED_REVIEW_ALL_ARMS.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in pairs),encoding="utf-8")
    case_lines = ["原v2失败及不确定的12题：真实query、完整输出与逐版本审阅", "",
                  "所有判断均为同一非盲Codex上下文初审；没有读取标准答案、支持文档或下游F1。",
                  "保留网络缺失，不补写模型分解；按同一四项语义口径对照。", ""]
    for index, pair in enumerate([r for r in pairs if r["v2"]["verdict"] != "pass"], 1):
        case_lines += [f"{index}. {pair['dataset']} | {pair['qid']}", "原始query：", pair["question"], ""]
        for arm in ARMS:
            r = pair[arm]
            case_lines += [f"{arm}：{r['verdict']}", "原始模型输出：", r["literal_answer"] if r["literal_answer"] is not None else "[无完整模型回答；状态不明请求保留，未重发]",
                           "逐题判断：" + r["reason"], ""]
        case_lines += ["=" * 60, ""]
    (DESTINATION / "V2_FAILURE_CASES_BEFORE_AFTER.txt").write_text("\n".join(case_lines)+"\n",encoding="utf-8")
    lines = ["自包含多粒度分解prompt：v2失败分析与v3/v4/v5同36题对照", "",
             "v2为冻结对照；所有版本使用相同36题（HotpotQA/2Wiki/MuSiQue各12），qwen-plus-2025-12-01、temperature0、thinking关闭、max_tokens1024。",
             "v3只加强逐项独立理解及描述性绑定；v4在v3上只扩展用途/输入已知事实规则；v5从v3增加三个通用示例，不沿用v4的长规则。",
             "问题、模型、解码、system/user角色、同一四项语义口径均已逐payload核对；旧v2判断未修改。",
             "v2失败主要为6题跨题指代、4题无效支持问题，另有2题不确定；错误维度可能重叠。详见V2_FAILURE_ANALYSIS.txt。",
             "当前Codex同一非盲上下文初审，非人工金标。样本被用来选择prompt修改；不作为独立测试结果或4B/下游F1结论。", "",
             "版本\t完整返回\t通过/36\t失败\t不确定\t缺失\t自包含失败\t平均输入token(完整回应)"]
    for arm in ARMS:
        r = statistics[arm]; c=r["verdicts"]
        lines.append(f"{arm}\t{r['complete']}/36\t{c.get('pass',0)}/36 ({r['pass_rate_full_denominator']:.2%})\t{c.get('fail',0)}\t{c.get('uncertain',0)}\t{c.get('unavailable',0)}\t{r['failed_checks']['self_containment']}\t{r['mean_input_tokens_returned']:.1f}")
    lines += ["", f"四版共同有效{len(common)}题，通过数："+json.dumps(common_summary["passes"],ensure_ascii=False), ""]
    for c in comparisons:
        lines.append(f"{c['before']}→{c['after']}，共同{c['n']}题：通过{c['before_pass']}→{c['after_pass']}；转为通过{len(c['gained_pass'])}，失去通过{len(c['lost_pass'])}；修复自包含{len(c['fixed_self_containment'])}，新增自包含失败{len(c['new_self_containment_failure'])}。")
    lines += ["", "分数据集（各版均保留12题分母）："]
    for ds in ("hotpotqa","2wiki","musique"):
        lines.append(ds+": "+", ".join(f"{a} {statistics[a]['by_dataset'][ds].get('pass',0)}/12" for a in ARMS))
    lines += ["", f"按共同题通过数、较少自包含失败、较短输入依次比较的本轮候选：{recommendation}。仅作后续开发起点，不声明显著优势或90%泛化正确率。",
              f"本轮新增用量按既有单价计{result['new_usage_list_price_cny']:.7f}元，另有{result['new_unknown_reserved_cny']:.7f}元状态不明预留；未核对实际账单。",
              "所有未知请求保留状态和预算，不重发；没有把缺失当作回答错误，也未填补分解或最终答案。",
              "API输入token增加与规则/示例长度相关。后续4B训练必须另做原生tokenizer的输入+目标长度检查，不能据此假定旧1024窗口足够。",
              "保留文件：各版本prompt、精确diff、DECISIONS.txt、各版本协议与原始API请求、AUTHORED_v3/v4/v5.json、逐题合并审阅和RESULTS.json。"]
    (DESTINATION / "REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"statistics":statistics,"common":common_summary,"comparisons":comparisons,"candidate":recommendation,
                      "new_cost":result['new_usage_list_price_cny'],"new_unknown_reserved":result['new_unknown_reserved_cny']},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
