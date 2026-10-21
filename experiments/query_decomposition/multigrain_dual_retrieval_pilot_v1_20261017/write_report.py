"""Render the completed diagnostic pilot, with exact plans and answer traces."""

import json

from .screen_musique import HERE, OUT, rows, load


def percentage(value):
    return "未完成" if value is None else f"{100 * value:.2f}"


def main():
    result = load(OUT / "RESULTS.json")
    details = rows(OUT / "CASE_RESULTS.jsonl")
    plans = {r["qid"]: r for r in rows(HERE / "cases.frozen.jsonl")}
    notes_path = OUT / "DIAGNOSTIC_NOTES.jsonl"
    notes = rows(notes_path) if notes_path.exists() else []
    complete = all(
        summary["available"] == 24
        for summary in result["overall"]["conditions"].values()
    )
    lines = [
        "不同粒度独立子问题＋完整句传递＋可选双路检索：诊断实验",
        f"状态：{'24题两组全部完成' if complete else '执行中，以下结果尚不完整'}",
        "",
        "一、固定设计",
        "三数据集各8题：历史S02 F1=0与F1=1各4题，共24题。按历史结果和问题类型有目的地选取，非随机总体评估；历史S02在这24题的50%由抽样规则决定。",
        "MuSiQue原先没有S02成绩，因此先用原冻结S02在正式样本顺序前24题运行筛选：9题F1=1、7题F1=0、8题为部分分；从两个端点各取4题。该筛选没有使用新方法。",
        "人工编写并在新方法调用前冻结的query-only自包含子问题，无强制#N绑定。假定分解正确是本实验条件，不能将该参考方案称为独立人工金标准。没有把事实答案或支持文档标注加入生成输入。",
        "末节点等于原query，也检索并生成完整句回答；随后另一次调用回答最终原query。所有历史问答完整句作为可选上下文向后传递；前置缺答不阻断原始检索路线。",
        "单路组：每个自包含子问题仅检索原问句Top10。双路组：结合前置完整问答调用同一API生成可选改写，保留原问句检索，同时检索改写问句Top10。两路按名次交替、原始路先、doc_id去重，截取10篇。改写不变时不重复检索。",
        "子问题回答的是原始子问题，输入两路合并后的10篇；完整句不再抽成裸实体。少于4词时仅追加一次完整句格式化调用，实际次数单独记录。这一词数规则只是格式启发式。",
        "最终输入保留历史原query的全部Top10，按原冻结位置插入问答辅助块，故不删除第10篇。子问题新取得的证据通过生成问答传递，未直接追加其原文至最终输入。",
        "检索器、语料、生成模型qwen-plus-2025-12-01、S02子问题4-shot与最终prompt均冻结；temperature=0。双路组仅增加基于各自完整历史的改写与相应检索；两组历史独立，相同完整API payload可复用缓存。",
        "本实验不训练模型，不作ReAct式行动循环，不恢复训练标签修订，不修改历史成绩。",
        "",
        "二、最终Answer F1（%）",
        "数据集\t题数\t历史S02\t历史Top10-v1\t历史ReAct\t新单路\t新双路",
    ]
    for name, summary in [("全部", result["overall"]), *result["by_dataset"].items()]:
        history = summary["historical_references"]
        conditions = summary["conditions"]
        lines.append("\t".join([
            name, str(summary["planned"]),
            *[percentage(history[key]["mean_f1_available"]) for key in ("frozen_s02", "top10_v1", "react")],
            *[percentage(conditions[key]["mean_f1_available"]) for key in ("single_original", "dual_original_rewritten")],
        ]))
    paired = result["overall"]["paired_dual_minus_single"]
    lines.extend([
        f"主要配对结果：双路减单路 {100 * paired['mean_difference']:.2f} 个F1百分点；{paired['gains']}题上升、{paired['ties']}题持平、{paired['losses']}题下降。",
        "三题由F1=0升至1：Firefly影片年份、Sibylla祖母、Raiders Of Red Gap导演死亡时间比较；另有Genesis比较由0.1818升至0.5。",
        "唯一下降题：Jeanne de Laval-Tinténiac缩短成Jeanne de Laval，F1由1降至0.6667；完整姓名证据仍保留，不能说是相关文档被挤出或事实性错误已被证实。",
        "本结果没有单独验证完整句优于短答案，也没有单独隔离取消强制引用的收益；两组均已采用这两项设计。原先成功的12题中双路保留11题F1=1，另1题为姓名缩短的部分分。",
        "尚未专门构造并核查‘第一跳实体名称没有任何证据，但更粗问题可直接回答’这一语料条件，因此本轮不能声称已验证这一特定机制。",
    ])
    lines.extend(["", "三、配对与覆盖诊断", "范围\t配对题数\t双路减单路F1百分点\t上升/持平/下降\t全部标注文档曾被reader看到(单路/双路)"])
    for name, summary in [("全部", result["overall"]), *result["by_dataset"].items(), *result["by_historical_group"].items()]:
        paired = summary["paired_dual_minus_single"]
        single = summary["conditions"]["single_original"]
        dual = summary["conditions"]["dual_original_rewritten"]
        lines.append("\t".join([
            name, str(paired["pairs"]), percentage(paired["mean_difference"]),
            f"{paired['gains']}/{paired['ties']}/{paired['losses']}",
            f"{single['questions_with_all_gold_docs_shown_to_some_reader']}/{dual['questions_with_all_gold_docs_shown_to_some_reader']}",
        ]))
    single = result["overall"]["conditions"]["single_original"]
    dual = result["overall"]["conditions"]["dual_original_rewritten"]
    lines.extend([
        f"历史F1=0的12题：单路平均F1={percentage(result['by_historical_group']['failure']['conditions']['single_original']['mean_f1_available'])}%，双路={percentage(result['by_historical_group']['failure']['conditions']['dual_original_rewritten']['mean_f1_available'])}%。",
        f"最终prompt平均token：单路{single['mean_final_prompt_tokens']:.2f}，双路{dual['mean_final_prompt_tokens']:.2f}。",
        f"双路发生实际改写{dual['rewrites']}次，{dual['steps_with_new_gold_docs_shown']}个步骤向reader新增了标注相关文档，{dual['steps_with_original_gold_docs_displaced']}个步骤排除了原始路的一篇标注相关文档。",
        f"完整句格式化追加调用：单路{single['sentence_format_calls']}次，双路{dual['sentence_format_calls']}次。因单行问句格式不合格被拒绝的改写：{dual['rejected_rewrite_formats']}次。格式合法不等于语义保持。",
    ])
    lines.extend([
        "覆盖含义：标注的相关文档曾出现在任一子问题reader或原query最终reader中，不等于文档已经足以回答，更不等于完整事实已经传入最终prompt。",
        "双路最多仍只读10篇，故可能排除原始路线部分尾部文档；这一风险与最终保留原query全部Top10是两个不同环节。",
        "历史S02/Top10/ReAct是描述性参照，尤其S02对比还混合了分解、goal执行与信息传递变化；双路减单路才是本轮匹配的主要比较。双路增加改写调用和检索次数，未作等计算预算比较。",
        "",
        "四、逐题核查观察（仅陈述实际痕迹，不把相关性当成因果贡献）",
    ])
    for note in notes:
        lines.append(json.dumps(note, ensure_ascii=False, indent=2))
    audit_path = OUT / "REWRITE_SCOPE_AUDIT.json"
    if audit_path.exists():
        audit = load(audit_path)
        lines.append(f"逐条查看{audit['changed_rewrites']}条发生变化的改写，{audit['flagged_target_changes']}条改变了当前子问题所求；完整逐条记录见REWRITE_SCOPE_AUDIT.json。这是本次Codex定性审阅，不是独立人工评审。")
    lines.extend(["", "五、全部24题完整分解与实际执行"])
    for index, row in enumerate(details, 1):
        lines.extend([
            "", f"{index}. {row['dataset']} / {row['qid']} / 历史{row['historical_group']}",
            "原query：" + row["question"],
            "标准答案（仅用于离线打分）：" + str(row["gold_answer"]),
            "参考分解：",
        ])
        plan = plans[row["qid"]]["qdmr"]
        for step in plan["steps"]:
            lines.append(f"  #{step['id']} {step['request']}")
        lines.append(f"  goal = {plan['goal']}")
        for label, reference in row["historical_references"].items():
            lines.append(f"历史{label}：{reference['prediction']}；F1={reference['f1']}")
        for label, arm in row["conditions"].items():
            lines.append(f"{label}：{arm.get('prediction')}；F1={arm.get('f1')}；状态={arm['status']}")
            for node in arm.get("node_diagnostics", []):
                lines.extend([
                    f"  #{node['step_id']} 原问句：{node['question']}",
                    "  检索改写：" + node["rewritten_query"],
                    "  传递的完整回答：" + node["answer"],
                    "  新增且实际呈现的标注相关文档：" + str(node["added_gold_docs_actually_shown"]),
                    "  被挤出的原路标注相关文档：" + str(node["original_gold_docs_displaced"]),
                ])
    lines.extend([
        "", "六、用量、可回溯记录与局限",
        "本轮返回用量按既有单价记账（含Mu筛选）CNY：" + str(result["api_usage_list_price_cny"]),
        "未结预留CNY：" + str(result["unresolved_liability_cny"]) + "；包括已留档的提交前DNS失败及其他无返回用量的连接异常，不能称作已确认消费。不是已核对账单。",
        "网络恢复沿用共享Transport已有策略：新请求在同一进程内的可重试网络异常最多重试两次，保留未知尝试的费用预留，采用第一个完整响应；重启后未知请求不会自动重发。本轮未修改这一既有策略，最终逻辑请求成功不代表所有网络尝试成功。",
        "主试验API状态与用量：" + json.dumps(result["pilot_api_execution"], ensure_ascii=False),
        "冻结哈希检查：" + json.dumps(result["frozen_hash_checks"], ensure_ascii=False),
        "详细记录：本目录RESULTS.json、CASE_RESULTS.jsonl、cases/、api/、EXECUTION_FROZEN.json；实验目录cases.frozen.jsonl、protocol.json、DECISIONS.txt。每次请求的完整prompt、输出、用量和异常保存在独立SQLite。",
        "24题是小规模机制诊断；不能据此推断总体提升、统计显著性、自动分解器质量或SOTA。F1是官方词级指标，职业同义表达等情况需要与事实性分开看。没有把本轮F1=0自动计为幻觉。",
    ])
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUT / "REPORT.txt"))


if __name__ == "__main__":
    main()
