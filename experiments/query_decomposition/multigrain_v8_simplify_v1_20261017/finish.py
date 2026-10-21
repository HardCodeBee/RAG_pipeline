"""Reconcile literal reviews, paired outcomes, prompt lengths and API liabilities."""
from collections import Counter
import json
import sqlite3

from .study import HERE, OUT, ROOT, api, previous


def finish():
    results = {phase: json.loads((OUT / (phase + ".RESULTS.json")).read_text(encoding="utf-8")) for phase in ("dev", "holdout")}
    reviews = {phase: api.rows(OUT / (phase + ".REVIEW_ALL_ARMS.jsonl")) for phase in results}
    frozen = json.loads((HERE / "FINAL_CANDIDATE_FROZEN.json").read_text(encoding="utf-8"))
    candidate = frozen["version"]
    assert api.sha(HERE / "AUTHORED_dev.json") == frozen["authored_review_sha256"]
    assert api.sha(OUT / "dev.RESULTS.json") == frozen["development_result_sha256"]
    assert api.sha(HERE / "holdout.questions.frozen.jsonl") == frozen["holdout_questions_sha256"]
    assert api.sha(HERE / ("TRAINING_PROMPT_" + candidate + ".txt")) == frozen["prompt_sha256"]
    # Freeze integrity and payload checks come from the original review aggregator.
    requests = []
    for phase in results:
        for arm, audit in results[phase]["audit"].items():
            assert audit["payload_roles_model_prompt_and_allowed_inputs"]
            root = OUT / (phase + "_" + arm)
            protocol = json.loads((root / "PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
            assert api.sha(HERE / ("TRAINING_PROMPT_" + arm + ".txt")) == protocol["prompt_sha256"]
            for p in (root / "api" / arm).glob("*/*/requests.jsonl"):
                requests.extend(api.rows(p))
            for p in (root / "api" / arm).glob("*/*/checkpoint.sqlite3"):
                with sqlite3.connect(p) as db:
                    assert all(n == 1 for (n,) in db.execute("SELECT count(*) FROM attempts GROUP BY request_id"))
                    if phase == "holdout":
                        assert all(started > frozen["utc"] for (started,) in db.execute("SELECT started FROM attempts"))
    source_hash = json.loads((HERE / "CHANGE_v10.json").read_text(encoding="utf-8"))["parent_sha256"]
    assert api.sha(HERE / "TRAINING_PROMPT_v8.txt") == source_hash
    assert api.sha(HERE / "TRAINING_PROMPT_v8.txt") == api.sha(HERE.parent / "multigrain_decomposer_fresh_eval_v1_20261017/TRAINING_PROMPT_v8.txt")
    dev_ids = {r["qid"] for r in api.rows(HERE / "dev.questions.frozen.jsonl")}
    hold_ids = {r["qid"] for r in api.rows(HERE / "holdout.questions.frozen.jsonl")}
    assert len(dev_ids) == len(hold_ids) == 60 and not dev_ids & hold_ids
    observations = {}
    remaining = []
    for phase, records in reviews.items():
        observations[phase] = {}
        for arm in results[phase]["stats"]:
            failures = [r for r in records if r[arm]["verdict"] == "fail"]
            observations[phase][arm] = {
                "failures": len(failures),
                "failed_checks_overlapping": {key: sum(key in r[arm]["code"] for r in failures) for key in "FSCU"},
                "mean_questions_returned": sum(len(r[arm]["questions"]) for r in records if r[arm]["questions"] is not None) / max(1, sum(r[arm]["questions"] is not None for r in records)),
                "prompt_utf8_bytes": len((HERE / ("TRAINING_PROMPT_" + arm + ".txt")).read_bytes()),
            }
            if arm != "v8":
                common = [r for r in records if r["v8"]["code"] != "X" and r[arm]["code"] != "X"]
                observations[phase][arm]["common_with_v8"] = {
                    "n": len(common),
                    "failed_checks_overlapping": {a: {ch: sum(r[a]["verdict"] == "fail" and ch in r[a]["code"] for r in common) for ch in "FSCU"} for a in ("v8", arm)},
                }
        for r in records:
            if r[candidate]["verdict"] != "pass":
                remaining.append({"phase": phase, "dataset": r["dataset"], "qid": r["qid"], "question": r["question"], "candidate": candidate, "review": r[candidate], "use": "diagnostic review only, not an SFT target or untouched future test"})
    with sqlite3.connect(OUT.parent / "budget.sqlite3") as db:
        local = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUT.parent))+"%",)).fetchone()
        total = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) FROM charges").fetchone()
    audit = {
        "utc": previous.utc(),
        "v8_unchanged": True, "request_attempts": len(requests),
        "request_states": dict(Counter(r["status"] for r in requests)),
        "one_attempt_per_request": True, "unknown_requests_replayed": False,
        "query_only_payloads_passed": True, "frozen_split_no_overlap": True,
        "holdout_tuning": False, "trained_model_or_frozen_RAG_changed": False,
        "review_type": "same-context Codex authored review, nonblind, not independent human gold",
        "candidate": candidate, "target_met_holdout": results["holdout"]["stats"][candidate]["pass_rate_full"] >= .9,
        "study_estimated_cny": local[0], "study_unknown_reserved_cny": local[1],
        "global_estimated_cny": total[0], "global_unknown_reserved_cny": total[1],
        "actual_bill_verified": False, "observations": observations,
    }
    api.immutable_json(OUT / "EXECUTION_AUDIT.json", audit)
    (OUT / "REMAINING_CASES.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in remaining), encoding="utf-8")
    hold = results["holdout"]
    pair = next(p for p in hold["pairs"] if p["before"] == "v8" and p["after"] == candidate)
    conclusion = {"utc": previous.utc(), "development_selected_candidate": candidate,
                  "baseline_retained": "v8", "default_RAG_or_training_updated": False,
                  "prompt_bytes": {a:observations["holdout"][a]["prompt_utf8_bytes"] for a in ("v8", candidate)},
                  "holdout_full60_passes": {a:hold["stats"][a]["verdicts"].get("pass", 0) for a in ("v8", candidate)},
                  "common_pair": pair, "target90_met": audit["target_met_holdout"],
                  "interpretation": "Prompt substantially shorter, full planned pass counts tied; common58 loses one net pass and anaphora increases. Semantic improvement/noninferiority not established. Keep v8 as main version, v11 as frozen compact research candidate; no confirmation retuning."}
    api.immutable_json(OUT / "CONFIRMATION_CONCLUSION.json", conclusion)
    lines = ["v8精简与语义保持优化：完整实验报告", "", "结论：v11从3987字节缩短到2412字节（减少39.50%），新确认集与v8同为41/60通过。共同有效58题中，v8通过41题，v11通过40题；修复2题、损害3题。尚未证明语义质量改善或统计不劣，未达到90%。v8继续作为主版本，v11仅作为冻结的精简候选。", "", "尤其需要保留的负结果：共同有效确认题中，跨题指代S从3题增至11题。开发集上的指代减少没有泛化到确认集。语义F、覆盖C、冗余U计数互有重叠，不能相加作为因果贡献。", "", "范围：全新调优60题＋预留确认60题，各含HotpotQA/2Wiki/MuSiQue各20题。按照旧配额分层、固定种子抽样，排除已知旧题ID及规范化精确文本；不宣称消除所有同义改写或预训练泄漏。", "模型：qwen-plus-2025-12-01，temperature=0，thinking关闭，每题每版本单次调用，max_tokens=1024。system为完整版本提示词，user只有原query；没有检索原文、标准答案、参考分解或下游分数。", "评审：沿用F/S/C/U四项，当前Codex同上下文非盲逐题初审；不是独立人工金标。完全相同的字面问题列表可沿用已审判定，不同输出逐条审阅。", "分母：不确定及API不可用保留在计划60题分母；另外报告共同有效配对结果，不把API返回更多称为语义收益。", "这不是训练后4B正确率，也不是RAG Answer F1。训练集与原冻结RAG配置均未改动。", "", "每次改动：", "v10：合并重复禁止项与自查段落，保留两类构造操作及四个示例，3987→2065字节。共同50题30→27，新增退步3题，无新增通过；不采用。", "v11：改为直接从原题构造、由内向外展开未知对象，再独立恢复完整描述；改用三层嵌套及精确布尔谓词等示例。2412字节。开发集共同48题30→33，修复4题/损害1题，按记录规则选为确认候选。", "v12：把构造进一步压成两类提取操作＋对原句最小改写，明确前提与最终目标边界，替换为四层桥接和作品作者—任职公司示例；2379字节。指代下降但新增冗余身份/方向/位置问，完整开发集38题通过，低于v11的41题；未采用。", "", "记录修订：最初冻结规则以完整通过数为主；v10显示API返回差异可产生伪增益后，在v11/v12提交及确认集解封之前，增加共同有效配对不得下降的候选资格要求（SELECTION_AMENDMENT.json）。", "评审修订：dev第15题战争短语附着有原始歧义，统一更正v8/v10判定；无新增通过，保留旧汇总快照和REVIEW_AMENDMENT_001.json，不隐藏修订。", "", "协议与所有版本：" + str(HERE.relative_to(ROOT)), "开发集最优候选：" + candidate, ""]
    for phase, result in results.items():
        lines.append(phase + " (full planned denominator60):")
        for arm, stats in result["stats"].items():
            lines.append(f"  {arm}: {stats['verdicts']}; pass={stats['pass_rate_full']:.2%}; input tokens={stats['mean_input_tokens']:.1f}; output tokens={stats['mean_output_tokens']:.1f}; bytes={observations[phase][arm]['prompt_utf8_bytes']}")
        lines.append("Paired comparisons: " + json.dumps(result["pairs"], ensure_ascii=False))
        lines.append("By dataset: " + json.dumps({a:s["by_dataset"] for a,s in result["stats"].items()}, ensure_ascii=False))
        lines.append("")
    lines.extend(["确认集实际改善与退步例子（完整原query与原样输出）："])
    for index in pair["gained"] + pair["lost"]:
        r = next(r for r in reviews["holdout"] if r["index"] == index)
        lines.extend(["", f"{index}. {r['dataset']}: {r['question']}"])
        for a in ("v8", candidate):
            lines.append(a + " " + json.dumps(r[a], ensure_ascii=False))
    lines.extend(["", "正式审计：" + json.dumps(audit, ensure_ascii=False, indent=2), "", "费用为返回token用量按既有价格估算，不是核实账单；未知请求预留单列，不重发。完整分母、共同分母及各数据集分布均保留。"])
    (OUT / "REPORT.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    finish()
