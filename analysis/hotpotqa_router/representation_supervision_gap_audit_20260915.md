# Query 表征与监督覆盖审计：先重读已有全层模型

2026-09-15。有界只读审计 `../work/router_research` 的顶层训练源码、报告和完成收据，并接续本日已完成的 M6 读出与 Student 结果。**不再把全层微调、LP→FT、支持监督或 query 关系监督作为尚未尝试的方向。当前最小缺口是：已完成全层监督的 encoder，在统一线性头下，从 M6 而非末层 CLS 读取，是否留下尚未被测到的效用增量。** 本文只提出具体设计，没有进行表示提取或新拟合。[证据 JSON](representation_supervision_gap_audit_20260915.json)保存来源 SHA256、行号、已完成状态与读取边界。

## 1. 哪些早期“尚未做”已经过时

- [容量覆盖审计](../../../work/router_research/encoder_capacity_coverage_audit.md)，5、50 行，曾未找到全层更新；随后 [encoder scope 正式结果](../../../work/router_research/encoder_scope_results.md)，15–27、55–60 行，已完成 embedding＋12 blocks 的全层 F。
- [初始化核查](../../../work/router_research/encoder_head_initialization_review.md)，27 行，曾未找到继承 LP 头再解冻；随后 [LP→FT v2 结果](../../../work/router_research/lp_ft_v2_results.md)，15–27、31–38 行，已完成 H。
- [读取层核查](../../../work/router_research/encoder_layer_readout_review.md)，13 行，曾未找到中间层读取；随后 [四表示结果](../../../work/router_research/layer_pooling_results.md)，7–26 行，已完成 C6/M6/C12/M12。
- [query 关系覆盖](../../../work/router_research/query_relationship_coverage.md)是早期快照；后续 query-rank、几何/初始值诊断与 NZ/Z 监督分配修复均已完成。

这些历史审计无需改写；当前覆盖必须结合后续完成产物判断，不能只读旧设计文档的“未执行”状态。

## 2. 已完成的关键对照

以下 F1 差均采用 0–1 标度，均为已消费旧 9600 的开发结果；不同阶段的区间校正范围不同，不能以是否跨零直接比较两阶段效应大小。

| 已回答过的问题 | 关键结果 | 证据位置 |
|---|---|---|
| 能否拟合监督；增加训练步数或选时刻是否足够 | E11A 真实/置乱微集均可拟合；E11C 五折达到操作性 fit 门，native−Dense −.003706；E13 cal 效用选点 U−Dense −.001075，均未确认策略收益 | [容量审计](../../../work/router_research/encoder_capacity_coverage_audit.md)，19–25 行 |
| 从 gap MSE 改为效用加权 BCE 并更新 encoder | E10 native−Dense +.000814，97.5% 区间 [−.000630,+.002337]；并非第一次做 cost-sensitive 监督 | [E10](../../../work/router_research/e10_report.md)，9–35、41–44 行 |
| 加入 coverage、检索统计或共同答案效用辅助监督 | E02 已有两动作 coverage；E09 probe8 相对 F1-only +.000750、相对 coverage −.001259，两区间跨零；E14 共同效用辅助相对匹配 N −.000358，也未过门 | [E09](../../../work/router_research/e09_report.md)，9–33、39–44 行；[E14](../../../work/router_research/e14_report.md)，9–24 行 |
| 扩大 encoder 更新范围 | F−Dense −.002490 [−.007815,+.002995]；F−末两层 C +.001216 [−.004043,+.006613]，97.5% 区间。全层真实更新，fit 达标更快 | [scope](../../../work/router_research/encoder_scope_results.md)，15–39、55–60 行 |
| LP 初始化后再全层微调 | H−Dense −.001909；H−F +.000581；H−P +.000069，三项 98.75% 区间均跨零。H 五折已训练并独立复核 | [LP→FT](../../../work/router_research/lp_ft_v2_results.md)，15–38、53–61 行 |
| 冻结编码器的层位与池化 | 仅 M6 通过本轮候选门：M6−Dense +.004796 [+.000115,+.009460]；M6−C12 +.006775 [+.001920,+.011513]，99 又 1/6% 条件区间 | [layer pooling](../../../work/router_research/layer_pooling_results.md)，7–26、47–49 行 |
| 词项与 IDF 对应、换一组划分 | E16 I−U +.000747817，区间跨零；E18 I−N +.000922314，区间跨零。均匀池化、IDF 置乱与重新划分都已经做过 | [E16](../../../work/router_research/e16_report.md)，11–38 行；[E18](../../../work/router_research/e18_report.md)，9–31 行 |
| 用真实支持文档监督 query encoder | G 的保留折支持匹配从 90.1150% 到 91.1412%；但 G−Dense +.002401772，97.5% 区间 [−.000288402,+.005081891]。监督学习发生，但完整 Router 门未过 | [G 结果](../../../work/router_research/support_encoder_results.md)，7–24 行；[支持迁移诊断](../../../work/router_research/support_transfer_results.md)，5–28 行 |
| query 间连续效用关系监督 | MSE+.1×关系损失的 R−M +.00062160，97.5% 区间 [−.00046490,+.00173646]；NZ/Z 追加 10 次 encoder 拟合后，NZ−All +.000185083，区间仍跨零 | [query rank](../../../work/router_research/query_rank_results.md)，9–39 行；[分配修复](../../../work/router_research/query_rank_allocation_results.md)，9–39 行 |

支持监督之后的 full384/PCA32、树/Ridge、PLS、new35 替换和 old+new union 也已执行，均未确认所需新增策略收益；见[完整维度](../../../work/router_research/support_full_readout_results.md)、[树交互](../../../work/router_research/support_readout_interaction_results.md)、[监督子空间](../../../work/router_research/supervised_subspace_results.md)、[new35](../../../work/router_research/new35_support_results.md)、[union](../../../work/router_research/feature_union_results.md)。不宜再提出泛泛的“去掉 PCA”“加非线性”或“增加支持关系监督”。本日固定 M6 的 Ridge/Fourier、严格 OOF Student 和单点正则补偿也已关闭，见[当前总报告](research_progress_report_20260915.md)。

## 3. 当前输入、目标和推断边界

- 早期多数 encoder 训练只更新 BGE-small 的 block10/11，主表示是末层 CLS，经 L2 归一化；F/H 则更新 embedding 和全部 12 个 block。两者都不是在线使用检索结果的 Router。
- M6 是原始、**未微调** BGE 完成第 6 个 block 后的普通 WordPiece 均值，排除 padding/special token，保留标点和停用词；FP32 求均值再 L2 归一化。[实现](../../../work/router_research/encoder_layer_pooling.py)，9–23 行；[原模型提取入口](../../../work/router_research/run_layer_pooling.py)，156–173 行。
- 支持/qrels、probe、生成答案效用可以作为离线训练监督；它们不是当前 query 推理时可用的输入。支持匹配准确率、关系次序和 BCE 都不是最终答案 F1，也不能单独确认 Router 收益。
- 当前 weighted BCE 用三重复平均 `d=F1(BM25)−F1(Dense)` 的符号为标签，以 `|d|·1[|d|>1e−12]` 加权。其理想 sigmoid 是效用加权偏好，不是普通 winner 概率或 gap；MSE 的理想 Bayes 动作也正确，不能把 BCE 历史解释为修正了 MSE 的理论错误。[E10 推导](../../../work/router_research/e10_report.md)，11–23 行；[当前凸头目标](../../../work/router_research/weighted_linear_probe.py)，28–62 行。
- 所有新推断仍须以全部 query 的配对答案效用为分母；三重复不能扩大独立 query 数。旧池、多次外折和已消费验证集不会因重读表示恢复独立确认资格。M6 后续独立验证也未达到预定稳定增量门，见[验证结论](m6_beir_validation_conclusion_20260915.md)。

## 4. 唯一优先缺口：训练 checkpoint × 读取视图

**具体假设：全层 H 的效用监督已在中间表示中留下可用变化，但原末层 CLS/native 决策路径没有利用它。** 这不同于再次扩大 encoder 或改 Student 的 λ，也不预设假设成立。现有 layer-pooling 只提取原始 BGE；H 的正式预测只读取末层 CLS。当前有界源码/报告搜索未找到这两者的交叉，不等于全项目所有历史的绝对不存在。

尤其不要从旧支持 G 或 query-rank checkpoint 提取 M6 后声称得到新的支持/关系监督表示：它们只更新 block10/11，上游 embedding 与第 1–6 层冻结；在同一 eval/token/精度路径下，第 6 层不会因此学习变化。只有 F/H 这类真实全层更新的既有模型，适合本次零新增 encoder 训练的重读。

### 推荐最小设计，尚未执行

固定选择已有 H，保留 F 作为历史背景，不同时试 F/H 后选较好者。H 是已经完成 LP 初始化、全层更新和复核的明确训练路径；选择这个问题已经使用历史开发结果，应如实披露。

| encoder | M6：mean layer6 | C12：CLS layer12 |
|---|---|---|
| 原始 BGE | 复用原 5 个 M6 头和 OOF；五折 λ 均为 .001 | 新拟合 5 个 λ=.001 头 |
| 同折已训练 H | 新拟合 5 个 λ=.001 头 | 新拟合 5 个 λ=.001 头 |

共 **15 个新凸线性头，0 新 encoder 训练**。四格共享固定 λ=.001、相同效用权重、未惩罚截距、原生数值路径和零动作边界；不加 cal、不再做 inner 选参。原 C12 五折曾选 [.01,.001,.001,.001,.001]，不能把旧 C12 原样当作全格 λ=.001 的匹配控制。原 M6 则必须先精确重放，才可复用。

最有区分力的比较是 **H_M6−base_M6**，以及交互 **(H_M6−base_M6)−(H_C12−base_C12)**。前者询问既有全层训练对当前有用读取视图是否产生净增量；后者区分 checkpoint 的变化是否依赖读取视图。这里 M6/C12 同时区别层位和 token 汇聚，只能称“视图交互”，不能单独归因读取层。若以后要作为候选，还须超过两固定动作并满足另行冻结的实际效应门；几何变化、fit BCE 或交互为正均不能代替净收益。

### 保存格式、隔离与精度已可核对

- **checkpoint 路径：** H 为 `../work/router_research/lp_ft_v2/fold{0..4}.pt`；F 为 `../work/router_research/encoder_scope_v1/fold{0..4}.pt`，10 份均存在，约 132.9 MB/份。此次只查路径、文件大小、保存源码和完成收据，没有重复加载 checkpoint 张量。
- **保存 schema：** `trained_parameters` 加 `protocol_id/protocol_sha256/runner_sha256/fold`。full 模型保存 199 个可训练 tensor，包含 encoder 和 `answer.weight/bias`；原 pooler 两参数冻结，恢复须从绑定的原 BGE 载入再覆盖完整训练参数。[保存入口](../../../work/router_research/run_e11c.py)，220–224 行；[恢复函数](../../../work/router_research/encoder_scope_training.py)，94 行起。不能将该文件当成完整 Hugging Face `state_dict` 直接盲载。
- **每折自己的 H：** `run_lp_ft_v2.py`，206–218 行，只传本折 fit 与 `gap[fit]`；父训练入口 `run_e11c.py`，115–139 行，在其他行不填训练标签。H 按 fit-only 门停止，cal 未使用，test 在全部端点固定后统一预测。H 五折停止于 7/5/5/4/5 轮；不能把其他四折的 H ensemble 用于本折 test，因为那些模型可能见过本折查询标签。
- **不能伪造 nested CV：** H encoder 已学习整个 outer-fit 的标签。在这些固定 H 特征上再分 inner folds 选头 λ，不会重新获得 encoder 级的内层隔离。固定 λ 避开这项选择问题；outer-test 仍只能用本折 checkpoint。若需要重新内选 encoder/头组合，就必须重做严格内层 encoder 训练，超出本方案。
- **无需读 query 正文：** `layer_pooling_v1/tokens.npz` 已保存 `input_ids/attention_mask/token_type_ids/special_tokens_mask`；提取入口见 `run_layer_pooling.py`，91–97、201 行。绑定原 tokens、fold indices、query/group 身份后，可直接沿用 max128、batch8、eval、CUDA BF16 autocast。autocast 不表示所有中间值都是 BF16，不能额外量化已经 FP32 的归一化表示；M6 仍按 FP32 mean/L2。当前原生桥见 `lp_ft_training.py`，51–90 行。
- **小试只验实现：** 恢复 H 后先用旧 `answer` 头重放原 fit/test 固定少量分数，另验证 hidden-state 输出不改变旧 C12/native 路径；确认后才丢开旧 answer 头拟合新统一凸头。保存模型全部冻结后统一评价，不凭新效果切换 F、H、层数、λ或阈值。

如果 H_M6 未获得所需增量，结果只关闭“现有 H 端点换读取即可恢复收益”这一解释，不能证明以 M6 为训练出口的未来 encoder 适配无效。如果 H_M6 改善而 H_C12 不改善，才支持进一步研究训练出口与读取视图的匹配；它仍是已消费数据上的候选机制证据，不能预言独立收益。

## 5. 本次审计完成范围

已读取命名历史报告、相关保存/恢复/训练/提取源码，以及 F/H/layer-pooling 的独立检查与完成收据；当前三组均为已完成并复核状态。未重跑历史 checker，没有加载模型、读取 query/gold/answer 正文、计算新效应、训练模型或调用外部 API。未搜索原始会话日志，也未改写历史冻结文件。下一项实验仍需另行固定实现、数值检查、主比较和继续／停止门；本文本身不表示已执行。
