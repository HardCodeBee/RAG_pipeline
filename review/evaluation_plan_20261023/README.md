# 评估计划修订007：六行主表、三组已有分析、P02 QA/raw

**2026-10-23 更新；仅提交审核，全部评估保持暂停。** 本轮没有模型生成、检索、GPU、计时或训练。用户已授权本方案整理与公开推送；未授权恢复执行。

当前版本是 [PLAN.txt](PLAN.txt) 与 [PLAN.json](PLAN.json)。它们取代修订006的十方法主矩阵、追逐论文最高F1配置政策和必做计时要求。旧方案原文保存在 [revision006 快照](history/revision006_before_narrative_20261023/PLAN.txt)；原有 `reviews/20261023_preexecution_plan_budget/`、参数来源和 `.frozen` 执行细节继续保留为历史，不能据旧15000/1500计数启动任务。原snapshot的相对路径以当时评估计划根目录为基准。

论文围绕“**分解组织信息需求，分步求解形成最终回答可用事实**”展开。六行按论证职责选择；不以预期对手强弱决定取舍，不承诺全面最佳或各组件必要。负结果应收紧论点。

| 内容 | 具体范围 | 本轮执行状态 |
|---|---|---|
| 六行主表 | P02、Top10、Top10+CoT3、QD+Rerank、Self-Ask、DecomP；每数据集500，9000条件 | 设计更新，未执行 |
| 三组已有分析 | 正确性与充分性；union20/shared-context；信息传递案例与后续负结果 | 复用旧协议证据，0新增API预算 |
| P02 QA/raw | 固定100题/集，共300题；同一上游与中性final提示，两臂600个final条件 | 名单和prompt已准备，未执行 |
| 可选计时 | 沿用50题/集固定名单；六方法900流程 | 单独预算，不阻挡主结论 |

ReAct、IRCoT、Iter-RetGen、Hybrid退出本轮必做范围，旧结果和参数来源保留。Qwen+BGE配置不变。Self-Ask k5/max5；DecomP k2、内外8步、CoT；QD原题+最多5子查询、各k10、重排到10。均标为有来源的统一资源适配，不声称当前底座最优或原论文最佳复现。

| 审核内容 | 文件 |
|---|---|
| 当前全部约定与统计/失败规则 | [PLAN.txt](PLAN.txt)、[机器可读方案](PLAN.json) |
| 六行职责、固定代表性参数 | [METHODS.json](reviews/20261023_narrative_revision007/METHODS.json) |
| 旧结论、分母、负结果与可支持边界 | [EVIDENCE_MAP.md](reviews/20261023_narrative_revision007/EVIDENCE_MAP.md) |
| P02 QA/raw具体输入、去重、缺失和费用归属 | [PROTOCOL.md](reviews/20261023_narrative_revision007/qa_raw/PROTOCOL.md)、[JSON](reviews/20261023_narrative_revision007/qa_raw/PROTOCOL.json) |
| 共同中性final提示 | [PROMPT.txt](reviews/20261023_narrative_revision007/qa_raw/PROMPT.txt) |
| 固定300题（只有ID/题型/组号） | [COHORT](reviews/20261023_narrative_revision007/qa_raw/COHORT.qid_only.json)、[可复现抽样脚本](reviews/20261023_narrative_revision007/qa_raw/select_cohort.py) |
| 每单元旧结果候选、代理单价及预算公式 | [BUDGET.md](reviews/20261023_narrative_revision007/budget/BUDGET.md)、[BUDGET.json](reviews/20261023_narrative_revision007/budget/BUDGET.json) |

新增API预算按历史usage和冻结公开价格计算，人民币：

| 预算情景 | 估值 |
|---|---:|
| 主表，候选整题缓存全部通过严格核验 | ¥30.58–34.10 |
| QA/raw，600个新final（共同上游已由主表承担） | ¥1.43 |
| 必要合计，条件复用 | **¥32.01–35.53** |
| 必要合计，不预抵扣旧整题复用 | **¥44.18–47.70** |
| 主表费用再增50%并提高QA/raw输入/输出假设 | 约¥53.36（条件复用）或¥71.62（不抵扣复用） |
| 可选计时，另列 | ¥4.28–4.63 |

这两个端点是历史均值与Self-Ask轮次敏感性情景，**不是统计区间、费用上下界或硬预算帽**。GPU没有已核实机时价格，未编造人民币费用；历史未知负债保留，不能记0或自动重发。

旧六方法范围共有6365个可评分记录，其中2499属于本次参数改变，3866只是整题复用候选。**本修订严格迁移已通过0条**，不能把候选称为新主表已完成。条件复用情景需5134个新完整流程当量；未预抵扣参数改变单元的可能前缀复用。MuSiQue P02成本采用同方法dev100代理；缺失基线单元用明确标注的跨数据集同方法代理。

QA/raw只比较固定P02轨迹后的最终材料接口；QA与Raw并非信息等量或token匹配。共同中性prompt意味着QA臂也不是原生P02 final。Raw涵盖全部ordinary和goal实际reader文段，保留问题关联；不能读取被丢弃候选。历史S02 QA/raw不能顶替，旧P02轨迹也必须经过C1或由新主表产生。CI跨0不意味着等价。

三组旧分析保留各自协议，包括goal512及其他干预臂；不笼统称为原生P02，也不并入新共享层主表。原始材料已在[前次实验续包](../p02_followup_20261023/README.md)、[诊断包](../p02_diagnostic_20261021/README.md)和[原P02包](../p02_20261021/README.md)提供，本次按论证任务索引，不重推旧原始轨迹。

请重点审核：五个参照能否承担各自比较任务；固定适配参数是否合理；旧分析的结论范围是否准确；共同prompt与实际reader Raw是否构成适当对照；预算的候选复用和代理假设是否透明。未完成probe不补跑，大规模judge/更多基线/四项必要性消融/等预算及多模型扩展不排入本轮。

本包不包含凭据、原始账本、题目/答案数据、模型权重或本地docs。只有计划、已有证据摘要、参数来源引用、预算汇总及qid元数据。准备完成不等于执行器就绪；完整执行器、300题上游/C1与payload核验、未知回执核对和明确恢复授权仍是启动条件。
