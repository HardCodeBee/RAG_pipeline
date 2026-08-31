# Phase 2.8b：独立 holdout router 候选选择

实验日期：2026-08-29。

## 1. 结论先行

本轮 2,000-query 独立 holdout 已完成，但 **9 个候选中没有任何一个达到部署门槛，也没有任何一个达到预注册的放宽研究门槛**。因此当前不能选择一个 router 替换固定 Dense。

后续只保留以下两个 **诊断候选**，用于研究下一版训练目标或安全切换机制；它们不是已验证可部署模型：

1. `T0_tiecap__M4_oof_late_fusion`：预注册稳健排序第一；holdout F1 0.639927，相对固定 Dense 增益 +0.003587，95% CI [-0.001053, 0.008401]。
2. `T0_tiecap__M4_pca32_oof_late_fusion`：预注册稳健排序第二；holdout F1 0.640082，增益 +0.003742，95% CI [-0.001443, 0.009284]。

`T0_tiecap__M3_pca32_structured_ridge` 的点估计最高（F1 0.640460，增益 +0.004120），但置信区间下界更差、伤害/收益质量比更高，未进入预注册 diagnostic top 2。它应作为更简单的对照基线保留，不作为主候选。

当前部署决策仍应是 **固定 Dense**。正式 final holdout 继续保持未开启状态。

## 2. 实验目的与边界

本实验回答：Phase 2.8 在扩充训练 query 上显示潜力的 M3、M4 和 M4-PCA，能否泛化到一个事先预留、与训练 group 完全独立的 HotpotQA query 集合，并据此确认后续保留哪些 router。

本轮 holdout 的角色是 **候选选择集**，不是最终测试集。它在 Phase 2.8 retrieval 和 generation 前已经预留；本轮打开后已被消费，后续不得用它反复调模型、阈值或 gate 后再声称是独立验证。

评价目标仍是端到端答案效用：

$$
y(q)=\overline{F1}_{BM25}(q)-\overline{F1}_{Dense}(q).
$$

正值表示 BM25 generation 的答案 F1 更高，负值表示 Dense 更高。检索 hit、qrels、supporting facts、retrieval score、检索结果与生成答案都没有作为部署时 router 特征。

## 3. 独立性与冻结证据

- Holdout：2,000 queries、2,000 groups，每个 group 一条 query。
- 与 Phase 2.8 acquisition groups 交集：0。
- 与 T2 最大训练视图 groups 交集：0。
- Holdout 样本在 retrieval 前冻结；run sample SHA-256：`91d3c71b5b82eab336e3abc2c6b2862497e5e045cb6bf72f61a30d85efbbe58b`。
- 协议配置 SHA-256：`2394ca0408426d6b3d8cfd5cbeef998171d1c7252e237a3f76c398ad1870bfcc`。
- 9 个候选的 2,000-query 预测在 generation 标签出现前全部冻结；预测 NPZ SHA-256：`1601abf8e94af95a8710cb4a27e30d2e7672347c3433bafdfbdf629a819f914e`。
- 模型拟合时读取 holdout outcome：0 行。
- 本轮读取正式 final holdout：0 行。

预测冻结证据见 [`results/prediction_freeze.json`](results/prediction_freeze.json)，样本独立性见 [`results/holdout_open_manifest.json`](results/holdout_open_manifest.json)。

## 4. 实验配置

### 4.1 候选网格

三个训练视图与三个模型结构形成 9 个固定候选：

- T0：BM25 胜 1,196、Dense 胜 1,564、tie 6,000，总计 8,760。
- T1：BM25 胜 2,000、Dense 胜 2,000、tie 6,000，总计 10,000。
- T2：BM25 胜 3,000、Dense 胜 3,000、tie 6,000，总计 12,000。
- M3：structured features + fold-internal PCA32 query embedding，ridge 预测连续 F1 gap。
- M4：structured 分支和 384D raw query embedding 分支独立训练，再由 inner-OOF prediction 做 late fusion。
- M4-PCA：与 M4 相同，但 embedding 分支改为 fold-internal PCA32。

每个候选使用 Phase 2.8 对应训练视图全量拟合，采用三个固定 replica seed；M4/M4-PCA 的 base model、meta fusion 与 calibration 只使用训练集内 4-fold OOF。三个 replica 的 holdout 连续预测取平均。

### 4.2 Retrieval 与 generation

- Retriever：BM25 与 BGE Dense。
- 每种 retriever 保留 top 50；generation 实际使用 top 5 context。
- 每块 context 上限：1,800。
- Prompt：`hotpot_short_answer_v1`。
- Generator：`gpt-4.1-mini-2025-04-14`。
- Temperature：0；最大输出：64 tokens。
- 每个 query/action 生成 3 次；query/action 分数取三次 normalized token F1 均值。
- 并发 workers：4；费用硬上限：5 USD。
- 实际估算 generation cost：2.842166 USD。

完整冻结配置见 [`config.yaml`](config.yaml) 与 [`results/frozen_config.yaml`](results/frozen_config.yaml)。

### 4.3 预注册 gate

部署 gate 同时要求：

- 相对 holdout 最佳固定 retriever 的 F1 gain 至少 0.01；
- 配对 group bootstrap 95% CI 下界大于 0；
- harmful/beneficial mass ratio 不超过 0.5；
- switch coverage 在 0.01 至 0.50；
- calibration slope 在 0.5 至 1.5；
- 预测最高 decile 的实际 gap 大于 0。

若无部署候选，放宽研究 gate 仍要求：gain 至少 0.01、CI 下界大于 0、harmful/beneficial ratio 不超过 0.75、coverage 不超过 0.60、non-tie AUC 至少 0.55。

## 5. 实验流程

1. 冻结 2,000-query candidate-selection holdout、9 个候选、生成配置、gate 与选择规则。
2. 验证 holdout 与 acquisition/training group 零交集，再打开该候选选择集。
3. 对 2,000 queries 分别运行 BM25 与 BGE Dense top-50 retrieval。
4. 从 query text 提取 17D lexical、13D corpus-static 与 384D BGE query embedding；不读取 query 的 retrieval/generation outcome。
5. 用每个 Phase 2.8 冻结训练视图全量拟合 9 个候选，每个候选运行 3 个固定 replica；冻结 9×2,000 个预测及哈希。
6. 确认预测冻结时成功 generation outcome 为 0，再启动答案生成。
7. 对 BM25 与 Dense 各完成 2,000×3=6,000 次生成，共 12,000 次；确认 pending/failure 为 0。
8. 按 query 汇总三次 F1，计算固定 retriever、oracle 与每个 router 的端到端 F1、安全指标和校准指标。
9. 使用配对 group bootstrap 计算 95% CI，应用预注册部署 gate、研究 gate和 diagnostic top-2 规则。
10. 标记该 2,000-query holdout 已消费，并保持正式 final holdout 未开启。

## 6. Holdout 数据概况

- BM25 胜：267（13.35%）。
- Dense 胜：337（16.85%）。
- 精确 tie：1,396（69.80%）。
- High-margin BM25 胜：238。
- High-margin Dense 胜：311。
- 固定 BM25 mean F1：0.604848。
- 固定 Dense mean F1：0.636340。
- 最佳固定动作：Dense，领先 BM25 0.031492 F1。
- Query-wise oracle mean F1：0.707624。
- Oracle 相对固定 Dense 的理论 headroom：0.071284 F1。

Oracle 只表示事后逐 query 选择两种已生成答案的上界，不是可部署 router。

## 7. 九个候选结果

| 排序 | 训练视图 | 模型 | Router F1 | 对最佳固定增益 | 95% CI | 切换率 | 收益/伤害 query | 伤害/收益质量比 | AUC | 部署 / 研究 gate |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
| 1 | T0 | M4 | 0.639927 | +0.003587 | [-0.001053, 0.008401] | 11.70% | 37 / 21 | 0.621 | 0.568 | fail / fail |
| 2 | T0 | M4-PCA | 0.640082 | +0.003742 | [-0.001443, 0.009284] | 16.25% | 38 / 32 | 0.675 | 0.581 | fail / fail |
| 3 | T0 | M3 | 0.640460 | +0.004120 | [-0.002031, 0.010758] | 17.80% | 48 / 38 | 0.728 | 0.580 | fail / fail |
| 4 | T2 | M3 | 0.634777 | -0.001563 | [-0.012615, 0.009853] | 53.05% | 144 / 145 | 1.039 | 0.575 | fail / fail |
| 5 | T1 | M3 | 0.633650 | -0.002690 | [-0.012883, 0.007574] | 46.20% | 121 / 121 | 1.081 | 0.570 | fail / fail |
| 6 | T1 | M4-PCA | 0.633236 | -0.003104 | [-0.013753, 0.007649] | 48.50% | 134 / 133 | 1.086 | 0.578 | fail / fail |
| 7 | T2 | M4 | 0.633411 | -0.002928 | [-0.014108, 0.008384] | 53.60% | 150 / 152 | 1.073 | 0.562 | fail / fail |
| 8 | T2 | M4-PCA | 0.632529 | -0.003811 | [-0.015363, 0.007803] | 55.10% | 153 / 155 | 1.090 | 0.578 | fail / fail |
| 9 | T1 | M4 | 0.631149 | -0.005191 | [-0.015885, 0.005622] | 48.45% | 131 / 141 | 1.146 | 0.561 | fail / fail |

表中排序采用预注册的稳健 diagnostic 规则，优先考虑 CI 下界和安全性，不按 Router F1 点估计单独排序。完整机器可读表见 [`results/holdout_candidate_comparison.csv`](results/holdout_candidate_comparison.csv)，全部指标与 gate 失败原因见 [`results/holdout_selection_results.json`](results/holdout_selection_results.json)。

## 8. 主要观察

### 8.1 T0 的小幅正增益方向可以复现，但证据不足

Phase 2.8 T0 OOF 的三个候选增益为 +0.00256 至 +0.00300；独立 holdout 为 +0.00359 至 +0.00412，方向一致。但三个独立 CI 都跨 0，实际增益均低于 0.01，且伤害/收益比均未通过部署阈值。

因此可说 T0 存在弱的、可重复方向性信号，但不能说它已经带来确定或有实际意义的提升。

### 8.2 T1/T2 的 OOF 大幅改善没有泛化

- T1：Phase 2.8 OOF gain 为 +0.01959 至 +0.02290；独立 holdout 变为 -0.00269 至 -0.00519。
- T2：Phase 2.8 OOF gain 为 +0.04024 至 +0.04274；独立 holdout 变为 -0.00156 至 -0.00381。
- T1/T2 在 holdout 上切换率约 46% 至 55%，harmful/beneficial mass ratio 全部大于 1。

这表明 winner-balanced 训练视图确实增强了同视图 OOF 的可学习信号，但同时改变了动作先验和决策阈值。模型在自然分布 holdout 上过度切向 BM25，收益 query 与伤害 query 大致相抵，最终不如固定 Dense。不能把 T1/T2 OOF 曲线解释为自然 HotpotQA 分布上的泛化提升。

### 8.3 模型结构差异小于训练视图差异

T0 下 M3、M4、M4-PCA 的 F1 只相差 0.00053；M4 与 M4-PCA 的配对差异为 -0.000155，95% CI [-0.005156, 0.004624]。M4 与 M3 的配对差异也跨 0。

因此本轮不支持“384D raw embedding 明显优于 PCA32”，也不支持“PCA32 late fusion 明显优于 raw M4”。PCA32 替换在本轮没有损害点估计，但尚未形成结构性优势。

### 8.4 当前 query-only 信号仍偏弱

9 个候选的 non-tie AUC 约 0.561 至 0.581；最好也只是弱排序信号。T0 的 top decile realized gap 为正且 calibration slope 合理，但不足以把总体增益 CI 推到 0 以上。问题不只是模型是否训练充分，更核心的是：当前可部署 query-only/corpus-static 特征对 BM25/Dense 答案效用差的可预测信息有限，而配平训练又引入了先验/阈值偏移。

## 9. 候选选择与后续含义

### 9.1 现在应该选择什么

- **线上/部署候选：无。** 继续使用固定 Dense。
- **主要诊断候选：** `T0_tiecap__M4_oof_late_fusion`。
- **次要诊断候选：** `T0_tiecap__M4_pca32_oof_late_fusion`。
- **简化对照：** `T0_tiecap__M3_pca32_structured_ridge`，只作为低复杂度基线。
- **停止推进：** 全部 T1/T2 候选；它们在独立 holdout 上出现过度切换和负端到端增益。

选择两个 M4 版本不是因为它们显著优于 M3，而是因为预注册 diagnostic 排序优先选择 CI 下界更好、伤害质量比更低的候选，同时保留 raw embedding 与 PCA32 两种表示用于后续判断。三者在统计上仍不可区分。

### 9.2 不应立即做什么

- 不应把本轮最高点估计当作成功模型。
- 不应再用这 2,000 条 holdout 调阈值、重新筛特征或挑新模型后重复报告独立结果。
- 不应立即打开正式 final holdout；当前连研究 gate 都未通过，继续消费 final holdout 不能解决训练目标偏移。
- 不应继续单纯扩大 winner-balanced T1/T2 数据并期待自然分布收益自动增加。

### 9.3 下一步研究焦点

下一阶段若继续，应以 T0 M4/M4-PCA 为诊断起点，在训练数据内部研究“表示学习”和“部署动作校准”分离：winner-rich query 可用于学习相对排序，但动作阈值必须在独立、接近自然分布的 development 数据上冻结。任何新方案需要新的独立验证集，不能复用本 holdout 作为未见数据。

## 10. 费用、状态与验证

- Retrieval：2,000 BM25 + 2,000 Dense，全部完成。
- Generation：BM25 6,000 success + Dense 6,000 success；pending/failure 为 0。
- 估算新增 provider cost：2.842166 USD，低于 5 USD 硬上限。
- 预测冻结：9 strategies × 2,000 queries。
- 最终状态：features complete、predictions frozen、evaluation complete。
- 正式 final holdout rows read：0。
- 聚焦测试：17 passed，1 个既有 pandas/numexpr 版本 warning。
- 仓库 HEAD：`522f0cd4fc289488c4b0d3c31f6f2090600a41ef`；本实验脚本与配置在该工作树中尚未提交，因此 HEAD 不能单独重建本轮结果，需同时保留本记录中的冻结配置和 source run。

逐 query 的两种动作三重复 F1、实际 gap 与 9 个预测 gap 见 [`results/holdout_query_metrics.jsonl.gz`](results/holdout_query_metrics.jsonl.gz)。本地核心产物路径见 [`../../registry.yaml`](../../registry.yaml)。
