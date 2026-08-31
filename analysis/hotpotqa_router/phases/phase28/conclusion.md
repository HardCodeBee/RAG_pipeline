# Phase 2.8：HotpotQA BM25/Dense query 扩充与候选重训

实验时间：2026-08-27 至 2026-08-29。

## 1. 实验目的与边界

本阶段只回答两个问题：

1. 能否在不使用 sample weight 的前提下扩充 BM25 胜者与 Dense 胜者 query；
2. 将扩充后的 query 按冻结流程重新训练上一阶段有潜力的 M3、M4、M4-PCA，是否出现稳定改善。

本实验的监督目标是每个 query 上 BM25 与 Dense 的三次生成答案平均 normalized token F1 差：

$$
y(q)=\overline{F1}_{BM25}(q)-\overline{F1}_{Dense}(q).
$$

正值表示 BM25 的答案效用更高，负值表示 Dense 更高，精确 tie 作为零保留。本文报告的是答案效用 router 的严格 OOF 结果，不把 retrieval-only 指标当作最终 RAG 效果，也不把 oracle、qrels、检索结果或生成答案作为部署时特征。

## 2. 实验设计与冻结配置

- 数据集：HotpotQA train；排除原 Phase 2.7 已使用 group。
- 路由动作：BM25 或 BGE dense；默认动作是 Dense。
- 每个 retriever 先取 top 50，generation 实际使用 top 5 context，context 上限 1,800 tokens。
- Prompt：`hotpot_short_answer_v1`。
- Generator：`gpt-4.1-mini-2025-04-14`，temperature 0，最多 64 output tokens。
- 每个 query/action 生成 3 次；标签为三次答案 F1 均值之差。
- Query 扩充 hard budget：20 USD；实际估算成本 11.3226844 USD。
- 特征边界：仅 query 文本变换、冻结 BGE query embedding、静态语料 lexical statistics、冻结 corpus prototype summary。
- 禁止特征：当前 query 的检索文档、检索分数/排名、qrels、gold supporting facts、acquisition stratum、生成答案和答案分数。
- 训练：无 sample weight；3 个 split seed；每个 seed 5-fold repeated stratified group CV；4-fold inner CV；所有 scaler、PCA、late fusion 和 calibration 均只在 fold 内拟合。
- Gate：平均 gain 至少 0.01、group bootstrap CI 下界大于 0、三个 split gain 全正、harmful/beneficial mass ratio 不超过 0.5、switch coverage 在 0.01 至 0.5、calibration slope 在 0.5 至 1.5、预测最高 decile 的实际 gap 为正。

有效冻结配置见 [`config.yaml`](config.yaml)、[`results/training_views/T0_tiecap.yaml`](results/training_views/T0_tiecap.yaml)、[`results/training_views/T1_winner2000.yaml`](results/training_views/T1_winner2000.yaml) 和 [`results/training_views/T2_winner3000.yaml`](results/training_views/T2_winner3000.yaml)。

## 3. 实验流程

1. **冻结协议与数据边界**：固定 HotpotQA partition、BM25/BGE 动作、top 50 retrieval、top 5 generation context、prompt、generator、三次重复、答案 F1 gap 标签、无权重训练和所有 gate；同时排除旧 Phase 2.7 group。
2. **预留独立 confirmation**：在任何新 retrieval 和 generation 前，从未使用的新 train group 中每组最多取一个 query，再用冻结 seed 随机预留 2,000 条 confirmation；它们不进入 acquisition、训练或本次正式 OOF。
3. **定向 retrieval 筛选**：对 acquisition pool 运行 BM25 与 BGE retrieval，用 evidence page recall 差划分 `bm25_plus`、`dense_plus` 和多个 equal lane；按用户要求优先补充 BM25 表现更好的 query，再补充 Dense 胜者。
4. **生成与答案效用标注**：对选中 query 分别使用 BM25 top 5 与 Dense top 5 context，各运行三次固定 generator；计算 normalized token F1/EM，并以两种动作的三次平均 F1 差产生 BM25 胜、Dense 胜或 tie 标签。
5. **配额检查与冻结导出**：持续分批生成，直到 BM25/Dense 总胜者均不少于 3,000、两侧 high-margin 胜者均不少于 2,500；随后冻结 7,885 条新增 label 和 47,310 条 outcome，并记录哈希、成本与 partition 读取数。
6. **构建嵌套训练视图**：从旧数据保留固定 6,000 个 tie，并按冻结 acquisition 顺序逐步加入胜者，形成 T0、T1、T2；只为 T2 实际使用的 3,240 条新增胜者提取 query-only/corpus-static 特征。
7. **Preflight 与泄漏检查**：逐个验证快照行数、哈希、特征维度、group 隔离、三次重复完整性和 fresh-dev/final-holdout 零读取。
8. **正式嵌套 OOF 训练**：对 M3、M4、M4-PCA 分别运行 3 个 split seed × 5 outer folds；每个 outer train 内使用 4-fold inner CV 完成 preprocessing、PCA、late fusion 与 calibration，不让 outer validation 参与选择或拟合。
9. **共识评估与停止判断**：对每个候选平均三个 split 的连续 OOF 分数，运行 10,000 次 group bootstrap，并逐项应用 7 个冻结 gate。由于没有候选全部通过，停止在 `DO_NOT_OPEN_CONFIRMATION_POOL`，没有开始独立 holdout。

## 4. Query 扩充实施与结果

先在新 HotpotQA train group 上运行 BM25/BGE retrieval，并按 evidence page recall 差定向选择 `bm25_plus` 与 `dense_plus` lane；随后对选中 query 的两个动作各生成 3 次答案。最终：

- 已完成配对 retrieval：48,866 queries；尚未检索的备用池：6,069 queries。
- 进入生成：7,885 queries，对应 47,310 条成功 outcome；provider 失败数为 0。
- 新标签：BM25 胜 2,020、Dense 胜 1,502、tie 4,363。
- 合并旧数据后：BM25 胜 3,216、Dense 胜 3,066、tie 11,203。
- 合并后 high-margin 胜者：BM25 2,986、Dense 2,840；四项冻结 quota 全部满足。
- `bm25_plus` lane：4,185 queries 中得到 BM25 胜 1,806、Dense 胜 219、tie 2,160。
- `dense_plus` lane：3,700 queries 中得到 Dense 胜 1,283、BM25 胜 214、tie 2,203。

扩充不是把 retrieval 指标直接当 router 标签：retrieval 差只用于提高采样效率，最终标签仍由两种动作的答案 F1 决定。完整导出状态见 [`results/export_manifest.json`](results/export_manifest.json)。

## 5. 训练视图与候选

三个训练视图嵌套构建，tie 均只从旧 Phase 2.7 数据中按冻结 seed 无放回选取 6,000 条：

| 视图 | BM25 胜 | Dense 胜 | tie | 总 query |
|---|---:|---:|---:|---:|
| T0 | 1,196 | 1,564 | 6,000 | 8,760 |
| T1 | 2,000 | 2,000 | 6,000 | 10,000 |
| T2 | 3,000 | 3,000 | 6,000 | 12,000 |

T2 共使用 3,240 条新增胜者 query 的新特征；其余新增 tie 和超出 3,000/3,000 quota 的胜者不进入冻结训练视图。

候选均在扩充 outcome 生成前冻结：

- M3：30D structured features 与 fold-internal PCA32 query embedding 拼接，使用 ridge 回归预测 F1 gap。
- M4：structured 分支与 384D raw query embedding 分支分别训练，再用 inner-OOF 连续预测进行 late fusion。
- M4-PCA：与 M4 相同，但 embedding 分支改为 fold-internal PCA32，检验 PCA32 能否替代 384D raw embedding。

## 6. 正式 OOF 结果

下表使用三个 split 的连续预测取平均后形成共识策略；CI 是按 `group_id` 重采样 10,000 次的 95% bootstrap CI。完整精度数据见 [`results/training_scale_comparison.csv`](results/training_scale_comparison.csv)，各 split 原始正式指标见 [`results/formal/`](results/formal/)。

| 视图 | 候选 | 共识 gain vs Dense | 95% CI | coverage | harmful/beneficial | AUC | Spearman | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T0 | M3 | +0.00262 | [-0.00019, 0.00543] | 17.17% | 0.803 | 0.585 | 0.096 | fail |
| T0 | M4 | +0.00256 | [-0.00034, 0.00548] | 15.26% | 0.809 | 0.567 | 0.078 | fail |
| T0 | M4-PCA | +0.00300 | [-0.00001, 0.00599] | 17.55% | 0.786 | 0.584 | 0.094 | fail |
| T1 | M3 | +0.01959 | [0.01398, 0.02493] | 44.80% | 0.664 | 0.594 | 0.108 | fail |
| T1 | M4 | +0.02150 | [0.01554, 0.02740] | 49.60% | 0.681 | 0.599 | 0.116 | fail |
| T1 | M4-PCA | +0.02290 | [0.01695, 0.02860] | 47.95% | 0.647 | 0.598 | 0.113 | fail |
| T2 | M3 | +0.04024 | [0.03356, 0.04672] | 52.50% | 0.608 | 0.638 | 0.176 | fail |
| T2 | M4 | +0.04228 | [0.03542, 0.04900] | 55.87% | 0.618 | 0.639 | 0.185 | fail |
| T2 | M4-PCA | +0.04274 | [0.03590, 0.04943] | 55.62% | 0.610 | 0.637 | 0.178 | fail |

## 7. 观察与解释

### 7.1 扩充胜者后出现了真实的 OOF 可学习信号

T0 的三个候选 gain 都低于 0.01 且 CI 跨零；T1 的 gain 提升到约 0.020–0.023，三个 split 均为正且 CI 明显高于零；T2 进一步提升到约 0.040–0.043。AUC 与 Spearman 也随视图扩充总体上升。因此“原来胜者 query 太少，router 很难学到有用信号”获得了支持，不能再把失败简单归因于所有 query-only 特征完全无效。

### 7.2 该曲线不是自然分布上的泛化曲线

T0/T1/T2 不只增加样本数，也有意改变 BM25 胜、Dense 胜、tie 的构成。T2 上 fixed BM25 mean F1 已略高于 fixed Dense，而 T0 上 Dense 明显更高；同时 router coverage 从约 15%–18% 上升到 52%–56%。因此更大的同视图 OOF gain 一部分来自更多可切换 BM25 胜者和更大 oracle headroom，不能解释成“部署到原始 HotpotQA 分布必然提升 0.043 F1”。

### 7.3 M4-PCA 值得保留，但尚不能宣称稳定胜过 M4

M4-PCA 在 T0、T1 和 T2 的共识策略 gain 都略高于 raw-embedding M4；T1 中还同时具有更低 coverage 和更低 harmful/beneficial ratio。T2 按三个 split gain 的简单均值则是 M4 略高于 M4-PCA。证据支持 PCA32 late fusion 是有效且更紧凑的候选，但差距较小，不能据此宣称它已经稳定优于 M4。

### 7.4 当前主要失败机制转为错误切换损失，而不是“完全学不到”

T1 三个模型均只因 harmful/beneficial mass ratio 大于 0.5 而失败；其中最接近 gate 的 M4-PCA ratio 为 0.647。T2 的 ratio 虽下降到约 0.608–0.618，但 coverage 全部超过 0.5，且 ratio 仍未达标。模型能够提高平均 F1，却仍做出太多代价较大的错误 BM25 切换，说明下一问题更接近决策安全性/阈值与可迁移性，而不是继续简单扩大 query 数量。

## 8. 冻结决策

9 个“视图 × 候选”组合中没有任何一个通过全部 7 项 gate。正式决策为 `DO_NOT_OPEN_CONFIRMATION_POOL`：

- 预留的 2,000-query confirmation pool 未打开，读取数为 0；
- fresh-dev 读取数为 0；
- final holdout 读取数为 0；
- 本阶段训练与汇总没有额外 provider 调用。

决策与每项 gate 的机器可读结果见 [`results/training_decision.json`](results/training_decision.json)。

## 9. 局限、排除项与后续方向

- 结论只适用于本次人为构造的 HotpotQA train 训练视图；尚无 sealed natural-distribution 验证。
- 扩充阶段以 retrieval evidence 差做定向采样，虽然该字段没有进入模型特征，但会改变样本构成。
- 所有候选仍只观察 query 与静态语料特征，未证明 query-only router 已达到足够部署价值。
- 一次误触发的旧 Phase 2.7 `screen` 会尝试 17 个旧候选，已在完成一个无关 M0 基线后中断；该 partial screen 不进入任何表格、候选选择或结论。
- 讨论中的 2,000-query 独立 holdout 评估尚未执行。若之后建立 Phase 2.8b 协议修订，必须先冻结唯一候选、full-fit 模型、calibration、阈值、指标和停止规则；本记录不能把该提案写成本次已完成结果。
- 当前证据不支持继续无条件扩充 query。若开展下一阶段，应先冻结一个不利用 holdout 的安全切换方案（例如只作为独立研究问题检查阈值/选择性约束），再决定是否需要新的 sealed 验证；不能用本阶段 OOF 反复调 gate 后再声称通过原 gate。

## 10. 验证状态

- Phase 2.8 聚焦测试：5 passed。
- 三个训练视图 preflight：全部 passed。
- 三个视图正式训练：每个 3 candidates × 3 split seeds × 5 outer folds，全部完成。
- Query/label/outcome 与训练快照使用哈希和行数校验；没有 fresh-dev/final holdout 泄漏。
