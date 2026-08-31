# Phase 2.10：Scope 对 M3 的最小增量诊断计划

## 1. 本阶段只回答一个问题

Phase 2.9 已经确认，`c_scope_log_at_least_2_docs` 是唯一通过发现集、old/new 稳定性和 natural-2000 复现的单个原始 query 特质。Phase 2.10 不再搜索新特征，也不重新比较模型家族，只回答：

> 在完全保持历史 M3 的训练目标、训练视图、折分、PCA、Ridge 和 policy 不变时，加入这一维 scope 特征，能否带来稳定且实际有意义的增量 policy utility？

本阶段属于 **post-selection incremental diagnostic**。T2 已用于 Phase 2.9 的特征发现，natural-2000 也已经用于 Phase 2.8 的候选/训练视图选择和 Phase 2.9 的单特质确认。因此，即使 Phase 2.10 全部通过，也不能将结果称为新的独立泛化证据。

## 2. 数据边界

唯一训练视图为 `T2_winner3000`：

- 12,000 queries；
- 11,964 个 `group_id`；
- BM25 winner 3,000；
- Dense winner 3,000；
- exact tie 6,000；
- 不加权、不重采样；
- tie 的回归目标保留为 0；
- 不测试 T0、T1 或其他 query 比例。

主目标仍是每个 query 的三次 generation 平均 answer-F1 差：

\[
y_i=\overline{F1}_{BM25,i}-\overline{F1}_{Dense,i}.
\]

检索 evidence recall 只能作机制描述，不能作为训练目标、模型输入或通过门槛。

## 3. 只允许三个候选

### S0：scope-only

- 输入只有 `c_scope_log_at_least_2_docs`；
- fold 内 StandardScaler；
- Ridge，`alpha=10`。

它用于回答 scope 是否能够单独形成 policy，不是 natural 结果出来后可临时替换主候选的后备模型。

### M3：历史基线

- 原 structured 30D；
- BGE query embedding 384D；
- embedding 在 fold 内中心化并做 PCA32；
- 30D structured 与 32D PCA 拼接；
- Ridge，`alpha=10`。

### M3S：M3 + scope

- 原 structured 30D 加一维 scope；
- BGE embedding 仍只做 PCA32；
- 拼接为 63D；
- Ridge，`alpha=10`。

M3S 是唯一主增量候选。禁止新增 XGBoost、M4、其他 Phase 2.9 特征、交互项、不同 PCA 维数、不同 alpha、不同训练视图或 learned threshold。

## 4. 先补齐并验证 T2 的 scope

Phase 2.9 的 discovery matrix 只保存了 6,000 个非 tie query；模型训练需要 T2 全部 12,000 个 query。因此必须使用已经修正并冻结的 `corpus_static_cache_v2.npz` 与同一 scope extractor，为全部 T2 query 重新提取该特征。

提取阶段不能读取 natural outcome。完成后必须验证：

1. 12,000 行 query/group 顺序与 T2 snapshot 完全一致；
2. missing fraction 恰好为 0；
3. 不能对 tie query 插值、均值填充或设任意默认值；
4. 其中 6,000 个非 tie 值与 Phase 2.9 discovery matrix 逐元素完全一致；
5. natural-2000 的该列与 Phase 2.9 natural matrix 逐元素完全一致；
6. cache、extractor、输入和输出 hash 与 manifest 一致。

任何一项失败都停止，不能进入训练。

## 5. 严格同折训练

沿用三个历史 split seed：`20260901`、`20260917`、`20261003`。每个 seed 使用 5-fold `StratifiedGroupKFold`，strata 为 BM25 winner / tie / Dense winner，group 为 `group_id`。

三个候选必须共享完全相同的 outer folds 和 inner folds。每个 outer fold 内，再用 4-fold group-stratified inner OOF prediction 拟合非负 affine calibrator。

所有处理严格限制在当前训练 fold：

- scope scaler 只 fit training fold；
- 原 30D scaler 只 fit training fold；
- embedding mean 只由 training fold 计算；
- PCA32 只 fit training fold；
- Ridge 只 fit training fold；
- calibrator 只使用 outer-train 内的 inner-OOF prediction 与 target；
- outer-validation 只能 transform 和 predict。

同一 fold 中，M3 和 M3S 必须共享同一个 embedding mean 与 PCA projection，使两者差异只来自新增的一维 scope。不能先在 12,000 query 上统一 fit PCA 后再交叉验证。

GroupKFold 可以防止同一 HotpotQA group 跨折，但不能消除“scope 已经在整个 T2 上被选中”的全局特征选择偏差。因此内部 OOF 仍是 post-selection diagnostic，而不是无偏外部验证。

## 6. 先复现历史 M3

正式比较前，新的执行器必须复现冻结的历史 M3：

- fold id 完全一致；
- calibrated OOF prediction 在 `rtol=1e-10, atol=1e-12` 内一致；
- 三个 seed 的核心 policy metrics 与历史工件一致。

如果无法复现，优先判定为实现路径差异，停止本阶段，不能把差异解释成 scope 的效果。

## 7. 固定 policy 与 OOF 汇总

三个模型都输出 calibrated predicted gap。policy 固定为：

- 默认使用 Dense；
- predicted gap `> 0` 时切换到 BM25；
- predicted gap `= 0` 时保持 Dense；
- 不扫描阈值。

每个 split seed 产生一份完整 outer-OOF prediction；对三个 seed 的 calibrated gap 按 query 求平均，形成 consensus OOF score。

需要同时报告：

- fixed BM25、fixed Dense、best fixed、oracle mean F1；
- router mean F1；
- gain over Dense 与 gain over best fixed；
- group-bootstrap 95% CI；
- beneficial / neutral / harmful switch；
- beneficial gain mass 与 harmful loss mass；
- harmful-to-beneficial mass ratio；
- switch coverage；
- high-margin BM25 winner recall；
- oracle recovery 与 policy regret；
- non-tie AUC、gap Spearman、MAE、RMSE；
- calibration slope、deciles、top-decile realized gap；
- 每个 outer fold 和每个 split seed 的结果；
- M3S 相对 M3 的逐 query policy utility 差，以及 paired grouped-bootstrap CI。

bootstrap 以 `group_id` 为单位，固定 10,000 次；三个候选和成对比较共享相同 resample，以降低无关的 Monte Carlo 差异。

## 8. 内部门槛与第一停止点

M3S 只有同时满足以下条件，才允许进入 natural diagnostic：

1. gain over best fixed 至少 `0.01`；
2. 该 gain 的 grouped-bootstrap 95% CI lower `> 0`；
3. 三个 split seed 的 gain over best fixed 都 `> 0`；
4. M3S − M3 的 paired policy-utility point estimate `> 0`；
5. M3S − M3 的 paired grouped-bootstrap CI lower `> 0`；
6. 三个 split seed 的 M3S − M3 delta 都 `> 0`；
7. harmful-to-beneficial mass ratio `<= 0.5`；
8. switch coverage 在 `[0.01, 0.50]`；
9. calibration slope 在 `[0.5, 1.5]`；
10. top predicted-gap decile 的 realized gap `> 0`。

scope-only 用相同指标完整报告，但不能挽救主增量门槛失败。

任何内部条件失败，立即记录 `STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL`，并且：

- Phase 2.10 路径不生成 natural prediction；
- Phase 2.10 路径不得读取 natural outcome 文件；
- 不继续其他特征组合；
- 不继续其他模型；
- 不调整 threshold。

这一区分非常重要：单特质统计关联可以存在，但如果无法转化成相对 M3 的稳定 policy 增量，就没有理由继续扩大模型实验。

## 9. natural-2000 只作已消费诊断

natural-2000 包含 2,000 个不同 group：BM25 winner 267、Dense winner 337、tie 1,396。它已经被两次使用：

1. Phase 2.8 用于候选与训练视图选择；
2. Phase 2.9 用于确认 scope 单特质。

因此它必须标记为 `consumed post-selection diagnostic`，不是 holdout，也不是 final test。

只有内部 M3S 门槛全部通过后，才能执行以下顺序：

1. 仅在 T2 上做三个模型的 full fit；
2. 每个 replica 的 calibrator 仍只由 T2 内部 4-fold group OOF prediction 拟合；
3. scaler、embedding mean、PCA、Ridge 都不得 fit natural；
4. 对三个 replica 的 calibrated natural score 求平均；
5. 先冻结并哈希三个模型的 natural prediction；
6. 再由 Phase 2.10 evaluation 路径读取现有 natural outcomes；
7. 不允许根据结果更改候选、训练视图、alpha、PCA 或 threshold。

natural continuation screen 要求 M3S：

- gain over best fixed `>= 0.01`，且 CI lower `> 0`；
- M3S − M3 paired delta `> 0`，且 paired CI lower `> 0`；
- harmful-to-beneficial ratio `<= 0.5`；
- coverage 在 `[0.01, 0.50]`；
- calibration slope 在 `[0.5, 1.5]`；
- top decile realized gap `> 0`；
- non-tie AUC `>= 0.55`。

失败则记录 `STOP_NO_POST_SELECTION_NATURAL_STABILITY`。通过也只能记录 `ELIGIBLE_FOR_SEPARATELY_FROZEN_OFFICIAL_FINAL_PROTOCOL`，不能宣称 generalization、不能部署，也不能在 Phase 2.10 打开 official final holdout。

## 10. 执行顺序

1. Preflight：校验配置、输入、历史工件和实现 hash；
2. 使用 v2 cache 提取全部 T2 scope，并完成逐元素一致性检查；
3. 冻结 Phase 2.10 split manifest 和实现 hash；
4. 复现历史 M3；
5. 在严格同折条件下运行 scope-only、M3、M3S 的内部 OOF；
6. 计算三者 policy metrics 与唯一主 paired comparison；
7. 应用内部硬门槛；
8. 若失败，在此停止，natural outcome reads = 0；
9. 若通过，先 full-fit 并冻结 natural prediction；
10. 再进行 consumed natural diagnostic；
11. 根据冻结门槛停止，或仅标记为未来独立 final 协议的候选。

## 11. 当前状态

本文件与对应 YAML 只冻结实验设计和现有输入。当前尚未创建 Phase 2.10 runner、尚未生成 Phase 2.10 模型 prediction、尚未读取 Phase 2.10 natural outcome、尚未运行任何正式实验，official final holdout 读取数为 0。
