# M6：BEIR 条件来源完整验证结论

日期：2026-09-15。完整采集、原账本审计、正式评价及另一数值实现复核均已完成。

## 1. 研究问题

仅凭检索前 query 表示，冻结的 M6 能否选择 BM25 或 Dense，使最终答案 F1 稳定优于两个固定动作，并达到预先要求的至少 +0.01 实际收益？

M6 使用冻结 BGE 的第六层普通有效 token 均值、L2 归一化和线性头；logit>0 选择 BM25，否则选择 Dense。训练监督来自旧数据的两动作答案 F1 差，在线输入不含检索结果、参考答案或答案效用。[候选与历史回顾](research_resumption_20260915.md#4-当前候选-m6是什么有什么证据)

## 2. 设计与完成状态

- 样本：从 BEIR dev 中排除与已使用来源共享的信息需求 group，冻结 **5209 个 query、5206 个 group**。这是一个有明确纳入条件的来源子集。
- 先冻结候选、样本和全部动作，再进行验证；本轮未据新效用重新训练、校准或筛选样本。
- 对全部 query 采集 BM25、Dense 各三个成功生成，共 **31254 个答案**；主指标为 normalized-token answer F1。固定生成器为 `gpt-4.1-mini-2025-04-14`。
- 两主比较共享预定的 20000 次 group bootstrap，各报告双侧 **97.5% 区间**；在单项区间有效的条件下，Bonferroni 联合覆盖率至少为 95%。
- 通过标准：两个主比较的下界都大于零，且相对最佳固定动作的点增益至少为 **+0.01**。更强的“下界也至少 +0.01”另外报告。

上述设计以[原冻结协议](../../../work/router_research/m6_beir_validation_v1/protocol.json)和[原预算](../../../work/router_research/m6_beir_validation_v1/answers_v1/budget.json)为准。

采集终态为 **35657 次 attempt、31254 次 success、4403 次 failure、0 条 reserved**。原有 21686 个成功结果保留，恢复过程补齐 9568 个成功结果。账本累计记账/保留额度为 **$12.364080**，仍在原 **$15** 上限内；这是本地账本口径，不是服务方账单核销。未删除缺失样本或将失败填为平局。[完整采集清单](../../../work/router_research/m6_beir_validation_v1/answers_v1/outcomes_manifest.json)、[终批状态](../../../work/router_research/m6_beir_validation_v1/answers_v1/recovery_20260915_v1/ready_v2_step_021.json)、[既有成功记录保留核验](../../../work/router_research/m6_beir_validation_v1/answers_v1/recovery_20260915_v1/preconnect_pilot_ledger_checks.json)

## 3. 正式结果

以下均为原始 F1 尺度（0–1），不是百分比相对提升。

| 预定比较 | 平均 F1 差 | 双侧 97.5% 区间 |
|---|---:|---:|
| M6 − Dense | +0.0037037958 | [−0.0013788826, +0.0089715660] |
| M6 − BM25 | +0.0318053738 | [+0.0218100681, +0.0420525210] |

本样本中较强固定动作为 Dense。相对最佳固定动作的点增益为 **+0.0037037958**，原分析给出的联合区间为 **[−0.0013788826, +0.0089715660]**。[完整正式评价](../../../work/router_research/m6_beir_validation_v1/evaluation.json)

描述性平均 F1：M6 **0.6392923773**、Dense **0.6355885815**、BM25 **0.6074870035**。这些绝对值用于说明量级，不增加新的检验。[完整正式评价](../../../work/router_research/m6_beir_validation_v1/evaluation.json)

原账本审计为 `passed_collection_integrity`；另一数值实现复核为 `passed_separate_complete_pair_effects_and_group_intervals`，覆盖全部 31254 个答案。正式字段 `statistical_and_point_magnitude_standard_met` 和 `core_goal_achieved` 均为 **false**。[原完整审计](../../../work/router_research/m6_beir_validation_v1/answers_v1/complete_ledger_checks.json)、[独立数值复核](../../../work/router_research/m6_beir_validation_v1/evaluation_separate_checks.json)

## 4. 当前结论

**本次完整验证未通过预定门槛。** M6 优于 BM25 的证据明确；相对较强固定动作 Dense 的区间跨零，且点增益不足 +0.01，因此不能宣布已实现相对最佳固定动作的目标收益。

在原条件来源与区间估计前提下，上界 **+0.0089715660** 已低于 **+0.01**，支持排除这个冻结候选在该条件总体上至少 +0.01 的收益；**小幅正收益仍与数据相容**。这也不是“已经证明收益恰为零”，更不是对所有未来检索前路由模型的否定。

旧内部结果 M6−Dense 约 **+0.004796**，本次约 **+0.003704**。数值接近不能证明稳定性：旧池参与过反复选择，两次来源、估计对象及区间条件不同；本次外部区间跨零。不能把两次点值并列当成新的稳定性检验。[旧内部结果及选择边界](research_resumption_20260915.md#4-当前候选-m6是什么有什么证据)

M6 的六层实现有本地重放与耗时证据，但 **M6 路由尚未集成主 runtime**。完整研究验证结束，不等于已经部署，也不等于端到端延迟或线上收益已经验证。[工程状态回顾](research_resumption_20260915.md#4-当前候选-m6是什么有什么证据)

## 5. 完整结果后的预设描述诊断

四组按事先固定的分数边界划分；gap 为 BM25−Dense 三重复平均 F1 差，near/far 不代表校准后的正确概率。

| 固定组 | D_far | D_near | B_near | B_far |
|---|---:|---:|---:|---:|
| 组内平均 gap | −0.059047829 | −0.022248486 | −0.003269531 | +0.037301880 |

固定选择与效用存在描述性关联：选择关联项为 **+0.009826908**；但所选 BM25 的有益贡献 **+0.019501216** 与有害贡献幅度 **0.015797420** 大幅抵消。B_near 的总体负贡献仅 **−0.000356516**，不能据此声称调整阈值就能达到 +0.01。关联分解不是因果归因，也未产生新阈值的独立收益证据。[完整诊断](m6_post_validation_diagnostics_20260915.md)

三个 repeat 的 M6−Dense 增益落在 **+0.003491426 至 +0.003857019**；仅 **45/5209** 个 query 的三个 gap 同时出现正值和负值。这削弱了“主要是单次生成噪声”的解释，但三个重复共用同批 query，不能视为三次独立来源复制。[完整诊断数值](m6_post_validation_diagnostics_20260915.json)

诊断入口曾把 `M6` 特征矩阵误作分数，被形状检查拦截；在诊断计算和输出开始前，另建入口显式改读 `M6_logits`。原冻结四文件保留，修正入口经独立审阅；数值公式、四组边界及动作不变，原正式评价不受影响。修正有[独立收据](m6_post_validation_diagnostics_correction_20260915.json)，完整诊断已通过另一数值实现复核，与原正式逐 query 贡献最大差为 **0**。

## 6. 下一研究问题

固定 M6 表示，区分“动作边界位置不合适”与“现有读出不能有效分开有益和有害切换”两种解释。普通校准及固定系数偏移此前已经试过，若再涉及只能作为机制对照，不能称为尚未尝试的收益方案。目前仅确定研究问题，**尚未执行新的实验**；本批诊断不能用来验证据其选择的新规则。[已有校准边界](research_resumption_20260915.md#4-当前候选-m6是什么有什么证据)

## 7. 证据边界与入口

- 本次是一个冻结条件来源的完整验证；不能外推为跨语料、跨生成器或线上稳定收益。区间是近似的条件来源 group 抽样推断，不包含所有历史模型选择和重训不确定性。
- 原来源随机拆分算法未独立复现；来源搜索曾展示五条 query 及相关文档 ID，该偏离已披露，未用于修改此前冻结的候选或纳入规则。
- F1/EM 不等于独立语义 Answer Correctness；数值复核是同一完整样本的另一求和路径，并非新样本复制。
- 该来源已用于完整效果评价和机制判断，现为已消费证据；以后按这些结果调整候选、阈值或特征，不能再用它声称新策略获得独立确认。

完整证据入口：[正式结果](../../../work/router_research/m6_beir_validation_v1/evaluation.json)、[原完整账本审计](../../../work/router_research/m6_beir_validation_v1/answers_v1/complete_ledger_checks.json)、[独立数值复核](../../../work/router_research/m6_beir_validation_v1/evaluation_separate_checks.json)、[完整采集终态](../../../work/router_research/m6_beir_validation_v1/answers_v1/outcomes_manifest.json)、[完整描述诊断](m6_post_validation_diagnostics_20260915.md)、[研究与恢复回顾](research_resumption_20260915.md)。
