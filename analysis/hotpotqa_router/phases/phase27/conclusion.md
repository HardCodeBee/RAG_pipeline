# HotpotQA BM25/Dense Router Phase 2.7 实验记录

**实验时间范围：** 2026-08-26 至 2026-08-27  
**最终状态：** 已完成 0–12 阶段；Stage 10–11 因前置 Gate 失败而不适用  
**Query-only 决策：** `STOP_QUERY_ONLY_V1`  
**Stage 12 诊断：** `GOLD_EVIDENCE_SIGNAL_SUFFICIENT`  
**可部署 Router：** 无

## 1. 实验摘要

Phase 2.6 的最佳 query-only Router 仅比固定 Dense 提高 `+0.002149` Answer F1，
grouped-bootstrap 95% CI 为 `[-0.000585, +0.004876]`，未达到预设的 `+0.01`
实用门槛。Phase 2.7 因此不再把问题简单归因于“XGBoost 没有训练好”，而是通过
受控实验分别验证：

1. 结构化 query/corpus-static 特征是否包含可泛化信号；
2. 384 维 BGE query embedding 是否包含可解码的 Retriever 偏好信号；
3. 直接拼接 structured 与 raw embedding 是否适合 XGBoost；
4. PCA、线性模型和 late fusion 是否提供更合适的 inductive bias；
5. continuous gap、robust regression、weighted sign classification 和双 utility head
   哪一种训练目标更合理；
6. 增加 query-only 训练数据能否继续提升；
7. query-only 失败后，信号是否主要存在于检索结果和 evidence sufficiency 中。

最终结果是：最强 query-only 候选 `M3_pca32_structured_ridge` 的 consensus gain
只有 `+0.002455`，CI95 为 `[-0.000083, +0.005069]`，harmful/beneficial utility
mass ratio 为 `0.789`，因此没有模型可以冻结或部署。Stage 12 的 qrels-free
post-retrieval probe XGBoost 则达到 `+0.014834`，CI95 为
`[+0.010703, +0.018926]`。这将主要缺失信号定位到检索后的质量、重叠和证据充分性，
而不是 query wording 本身。

## 2. 实验范围与信息边界

### 2.1 正式研究问题

> 只使用部署时可获得的 query 特征和 corpus-static summaries，能否预测每个 query
> 的 BM25-minus-Dense Answer-F1 gap，并稳定、显著地优于固定 Dense？

### 2.2 Query-only 阶段允许的输入

- query 文本变换；
- 冻结的 BGE query embedding；
- corpus-static lexical/IDF/DF 统计；
- 冻结的 corpus prototype 相似度 summaries。

### 2.3 Query-only 阶段禁止的输入

- 当前 query 的 retrieved documents、scores 或 ranks；
- qrels、gold supporting facts 或 evidence coverage；
- generated answers、Answer F1 或 Answer Correctness；
- 当前 query 的真实 BM25/Dense utility。

Stage 12 只有在所有 query-only 候选失败后才允许读取 post-retrieval probe 和 gold
evidence 特征。Stage 12 是离线机制诊断，不能报告为严格的 pre-retrieval Router。

## 3. 数据、生成与监督标签

| 项目 | 实际设置 |
|---|---|
| Dataset / partition | HotpotQA / `train_only` |
| Query 数 | 9,600 |
| Actions | BM25、Dense/BGE |
| 每个 action 的生成重复数 | 3 |
| 完整 outcome rows | 57,600 = 9,600 × 2 × 3 |
| Retriever candidate K | 50 |
| Generation context K | 5 |
| Context 上限 | 1,800 tokens |
| Prompt | `hotpot_short_answer_v1` |
| Generator | `gpt-4.1-mini-2025-04-14` |
| Temperature | 0 |
| Max output tokens | 64 |
| Utility | 三次生成的 normalized token F1 均值 |
| 默认 action | Dense |
| 主监督目标 | `g(q) = mean_F1_BM25(q) - mean_F1_Dense(q)` |

有效 query-only 配置见
[`config.yaml`](config.yaml)，Stage 12 协议见
[`results/privileged_teacher_protocol.json`](results/privileged_teacher_protocol.json)。

新版 `hotpot_multihop_short_answer_v2` 结果没有混入本实验。本实验必须保留
Phase 2.6 的旧 prompt、Top-5 context、generator 和三次重复，以保证监督标签含义不变。

补齐到 9,600 query 时新完成了 13,934 个缺失 generation cells，失败 0 次；输入
`7,940,094` tokens，输出 `80,045` tokens，估算新增费用 `$3.3041`。冻结 snapshot
建立后，query-only 模型审计和 Stage 12 均为零外部调用。

### 3.1 Answer utility 基线

| 策略 | Mean Answer F1 |
|---|---:|
| Fixed BM25 | 0.620175 |
| Fixed Dense | 0.648209 |
| Per-query B/D oracle | 0.717565 |
| Oracle headroom over Dense | +0.069356 |

Winner strata 为：BM25 winner `1,196`、Dense winner `1,564`、exact tie `6,840`。
`tie` 表示两个 action 的三次平均 F1 完全相同，不代表两个检索结果列表相同。71.25%
的 query 为 tie，说明真正可产生 utility 差异的机会集中在较少的 non-tie query 上。

## 4. 特征配置

严格 query-only 特征共 414 维：

- **17D lexical compatibility：** corpus vocabulary coverage、OOV ratio、IDF 的
  mean/std/min/max/p90/max-share、log DF mean/std、DF≤10/100/1000 比例，以及
  numeric/year/capitalized/quoted anchor 的最大 IDF；
- **13D corpus prototype summaries：** Top-1 至 Top-8 prototype similarity、
  margin、mean、std、entropy 和 local density；
- **384D query embedding：** `BAAI/bge-small-en-v1.5`。

其中 30D lexical+prototype 被称为 structured block，384D embedding 可原样使用，
也可以在每个训练 fold 内做 PCA-16/32/64。raw concatenation 为 414D。

## 5. 训练与防泄漏协议

正式确认采用 repeated `StratifiedGroupKFold`：

- 3 个 split seeds：`20260901`、`20260917`、`20261003`；
- 每个 split 5 个 outer folds；
- 每个 outer-train 内 4 个 inner folds；
- 按 BM25 winner / exact tie / Dense winner 分层；
- 按既有 `group_id` 分组，相关 query 不跨 train/validation；
- XGBoost model seeds：`11, 23, 37, 53, 71`。

StandardScaler、embedding centering、PCA、超参数选择、early stopping、late-fusion
meta model 和 affine calibrator 都只能在 outer-train 内拟合。Outer validation 只接受
一次最终预测和评价。XGBoost 最多 2,000 trees，early stopping patience 为 75；
参数按 inner-OOF policy gain 的 lower-one-standard-error 规则选择。

集成与决策规则为：先平均各 model seed 的连续 predicted gap，再使用 inner-OOF
拟合非负斜率 affine calibrator，最终固定：

```text
calibrated_gap > 0  -> BM25
calibrated_gap <= 0 -> Dense
```

主策略没有进行自由阈值搜索，也没有使用 isotonic calibration。

## 6. Stage 0–1：协议与切分验证

Stage 0 验证了 57,600 个完整 outcome rows、9,600 个 feature rows、action/repeat
完整性、F1 有限性、输入哈希与 query/group 对齐。第一批 4,800 query 的 outcomes、
summary 和 features 与早先冻结池保持一致。数据库 integrity 为 `ok`。

Stage 1 为每套 split 记录 fold、group 和 winner-strata 分布，并验证 train/validation
group 零交叉。Fresh-dev 和 final holdout 均读取 0 行。

## 7. Stage 2：特征块与模型结构消融

4,800-query quick screen 共测试 17 个封闭候选。该 screen 只用于筛选正式候选，
不能单独发布为正式结论。完整紧凑数据见
[`results/stage2_screen_results.csv`](results/stage2_screen_results.csv)。

| 代表候选 | 输入与模型 | Screen gain |
|---|---|---:|
| M0 | 30D structured → XGBoost MSE | +0.002411 |
| M1a | 384D embedding → Ridge | -0.001194 |
| M1b | 384D embedding → Elastic Net | -0.000289 |
| M2 | raw 414D concatenation → XGBoost | -0.001651 |
| M3 | PCA32 embedding + structured → Ridge | +0.003579 |
| M4 | structured/embedding inner-OOF late fusion | +0.003622 |
| M5 | 分别预测 BM25/Dense utility 后取差 | -0.000284 |

主要观察：

1. Structured block 含有弱但可检测信号；
2. Raw embedding 单独使用 Ridge/Elastic Net 时没有实用收益；
3. Raw 414D concatenation + XGBoost 是表现最差的候选之一；
4. Fold-local PCA32 + Ridge 明显优于 raw embedding tree；
5. Late fusion 在单 split screen 最好，但这个优势后来没有随样本量保持；
6. 分别预测两个 action 的 absolute utility 没有优于直接预测 gap。

## 8. Stage 3：训练目标实验

目标实验结果见 [`results/target_results.csv`](results/target_results.csv)。

| 对照 | MSE/continuous target | Robust/替代 target |
|---|---:|---:|
| Structured XGBoost | +0.002411 | +0.001354 |
| Raw 414D XGBoost | -0.001651 | -0.002337 |
| Weighted sign XGBoost | — | +0.000700，CI 跨 0 |

PCA+Huber、weighted sign classification 和 dual-utility head 均未稳定改善。Continuous
BM25-minus-Dense gap 因而继续作为正式目标。Exact ties 保留在 regression 中；
weighted sign classifier 跳过 exact ties。

重复生成诊断显示：

- 4,271/4,800 个 query 的两个 action 都具有零重复方差；
- gap standard error 中位数为 0，p90 为 0.031596；
- 3×3 cross-repeat soft preference 对 non-tie winner 的方向准确率为 93.4%；
- paired-repeat winner 与三次均值 winner 的平均一致率为 85.9%。

标签不是完全无噪声，但噪声不足以解释 Router 接近零的收益。由于每个 action 只有
三次重复，没有采用不稳定的 inverse-variance weighting。

## 9. Stage 4：随机旋转诊断

用三个固定 seeds 对 embedding 施加正交旋转。旋转保持点积、cosine 和 Euclidean
distance 不变，只改变坐标轴。结果见
[`results/rotation_results.csv`](results/rotation_results.csv)。

| 模型 | 未旋转 gain | 三次旋转 gain |
|---|---:|---:|
| Raw embedding XGBoost | -0.000420 | -0.000409 / -0.001787 / -0.001490 |
| Embedding Ridge | -0.001194 | 三次完全相同 |
| PCA32 + structured Ridge | +0.003579 | +0.002580 / +0.002770 / +0.003872 |

Raw embedding XGBoost 的 non-tie AUC 从 `0.548` 降到约 `0.509–0.519`；Ridge
对正交旋转保持完全稳定；PCA32+Ridge 也相对稳定。该结果说明树模型容易依赖
embedding 的任意坐标切分，raw embedding + tree 的 inductive bias 不合适。旋转是
机制诊断，不参与冠军选择，也没有在看到 9,600 新标签后重新选候选。

## 10. Stage 5–7：校准、评价和正式候选冻结

Stage 5 将旧的“每个 seed 先二值化再 majority vote”改成连续分数平均、inner-OOF
affine calibration 和统一零阈值。五-seed M2 在 4,800 screen 上达到 `+0.000817`，
但 CI 为 `[-0.002640,+0.004415]`，harmful/beneficial ratio 为 `0.913`，说明校准
修正不能单独解决信号不足。

Stage 6 同时报告 policy gain、beneficial/neutral/harmful switch、gain/loss mass、
missed BM25-beneficial switch、routing regret、oracle recovery、AUC、Spearman、
coverage 和 calibration。Retrieval Recall/Hit 不用于替代 Answer F1 utility。

在 9,600 新标签可见之前冻结四个正式候选：

- `M2_raw_full_xgb_mse`：原始 414D + XGBoost 复现基线；
- `M3_pca32_structured_ridge`：PCA32 + structured + Ridge；
- `M4_oof_late_fusion`：structured 与 embedding 父模型的 inner-OOF late fusion；
- `M0_structured_xgb_mse`：structured-only XGBoost。

之后没有根据 9,600-query 结果重新选择候选。

## 11. Stage 7.2：9,600-query 正式确认

执行了 4 candidates × 3 split seeds，共 12 个 candidate×split 任务；每个任务包含
5 个 grouped outer folds。正式比较及 4,800→9,600 变化见
[`results/comparison_4800_9600.csv`](results/comparison_4800_9600.csv)。

| 候选 | 三个 split gain | Mean split gain | Consensus gain / CI95 |
|---|---|---:|---:|
| M2 raw 414D XGBoost | +0.000828 / -0.000732 / +0.000931 | +0.000342 | -0.000690 `[-0.002778,+0.001387]` |
| M3 PCA32 + Ridge | +0.002792 / +0.002193 / +0.002248 | +0.002411 | +0.002455 `[-0.000083,+0.005069]` |
| M4 late fusion | +0.002244 / +0.001126 / +0.000582 | +0.001317 | +0.001602 `[-0.001212,+0.004468]` |
| M0 structured XGBoost | +0.000104 / -0.000250 / -0.000875 | -0.000340 | -0.000349 `[-0.002580,+0.001857]` |

Mean split gain 是三个独立 split policy 的 gain 平均；consensus gain 是合并三套
连续 OOF prediction 后形成的共识策略结果。

最强 M3 的 consensus policy：

- Router mean F1：`0.650664`；
- switch queries：`1,558/9,600`，coverage `16.23%`；
- beneficial / neutral / harmful：`186 / 1,210 / 162`；
- missed BM25-beneficial queries：`1,010`；
- beneficial gain mass：`0.011645`；
- harmful loss mass：`0.009190`；
- harmful/beneficial ratio：`0.789`；
- non-tie AUC：`0.580`；
- gap Spearman：`0.086`；
- oracle headroom recovery：`3.54%`。

该模型检测到弱信号，但绝大多数切换落在 utility tie 上，而且 harmful loss 仍接近
beneficial gain 的 79%。

## 12. Stage 8：训练规模学习曲线

完整 48-row 曲线见 [`results/learning_curve.csv`](results/learning_curve.csv)，宽表摘要见
[`results/learning_curve_summary.csv`](results/learning_curve_summary.csv)。

| 候选 | 1,200 | 2,400 | 4,800 | 9,600 | 9,600−4,800 |
|---|---:|---:|---:|---:|---:|
| M2 raw XGBoost | -0.001211 | -0.000229 | +0.000360 | +0.000342 | -0.000018 |
| M3 PCA32 Ridge | -0.000432 | +0.000207 | +0.001091 | +0.002411 | +0.001320 |
| M4 late fusion | 0.000000 | +0.000275 | +0.001827 | +0.001317 | -0.000510 |
| M0 structured XGBoost | -0.001181 | +0.000384 | +0.000341 | -0.000340 | -0.000681 |

只有 M3 从 4,800 到 9,600 继续改善。M2 已平台化，M4 和 M0 下降。M3 在数据翻倍
后仍只有 `+0.002411`，约为预设 `+0.01` 门槛的四分之一。因此结果足以否定
“继续增加同类型 query-only 数据即可解决”的假设，但不能否定所有未来架构。

## 13. Stage 9：冻结 Gate 与 query-only 决策

候选必须同时满足：

1. mean OOF gain ≥ `+0.01`；
2. grouped-bootstrap CI95 lower > 0；
3. 三个 split gains 全部为正；
4. harmful/beneficial mass ratio ≤ 0.5；
5. switch coverage 在 1%–50%；
6. calibration slope 在 0.5–1.5；
7. top predicted-gap decile 的 realized gap > 0。

M3 失败于 practical gain、CI lower 和 harmful-loss 三项；其他候选失败更多。最终决策：

```text
STOP_QUERY_ONLY_V1
```

没有 candidate 被冻结为可部署 Router。

## 14. Stage 10–11：未进入的阶段

Stage 10 fresh-dev 和 Stage 11 Answer Correctness 不是尚未完成，而是因为 Stage 9
失败而判定为不适用：

- 未创建 `model_freeze.json`；
- fresh-dev 读取 0 行；
- final holdout 读取 0 行；
- 未购买 Answer Correctness labels；
- 未进行 fresh-dev 调参或结果选择。

## 15. Stage 12：post-retrieval / privileged-teacher 诊断

Stage 12 的结果见 [`results/teacher_results.csv`](results/teacher_results.csv)，完整冻结协议见
[`results/privileged_teacher_protocol.json`](results/privileged_teacher_protocol.json)。

### 15.1 Qrels-free probe 特征

50 个 post-retrieval probe 特征不使用 qrels 或 generation outcome，包括：

- BM25/Dense Top-1/2/3/5 scores；
- score mean/std/range、Top-1/2 和 Top-1/5 margin；
- relative margin、coefficient of variation、rank slope 和 normalized entropy；
- context token count、truncation、query-context Jaccard；
- two-list Top-1 equality、Top-1/3/5 overlap、Jaccard、reciprocal-rank overlap；
- 两种 Retriever 的统计差值。

这些特征需要先运行两个 Retriever，因此不属于严格 pre-retrieval Router，但可能用于
低成本 shallow probe/cascade。

### 15.2 Gold evidence 特征

34 个 privileged gold 特征使用 qrels/supporting-document information，包括两个
Retriever 的 coverage@1/2/3/5、complete coverage、hit@5、first gold rank、MRR 及
两者差值。Gold 特征永远不能作为在线部署输入。

### 15.3 训练结果

| 候选 | 信息块 | Consensus gain / CI95 | Coverage | Harmful/beneficial | Non-tie AUC |
|---|---|---:|---:|---:|---:|
| P0 Probe Ridge | probe | +0.013439 `[+0.009464,+0.017491]` | 35.1% | 0.580 | 0.656 |
| P1 Probe XGBoost | probe | +0.014834 `[+0.010703,+0.018926]` | 34.6% | 0.572 | 0.667 |
| G0 Gold Ridge | gold | +0.038390 `[+0.033399,+0.043573]` | 65.6% | 0.372 | 0.805 |
| G1 Gold XGBoost | gold | +0.039265 `[+0.034225,+0.044398]` | 62.6% | 0.358 | 0.801 |
| G2 Probe+Gold XGBoost | probe+gold | +0.038447 `[+0.033538,+0.043442]` | 58.2% | 0.347 | 0.809 |

P1 已通过 practical gain、CI、三 split 稳定性、coverage、calibration 和 ranking
检查，只因 harmful/beneficial ratio `0.572 > 0.5` 未通过完整安全 Gate。其 oracle
headroom recovery 为 `21.4%`。

预先定义的 gold coverage@5 rule——仅当 BM25 的 gold coverage@5 高于 Dense 时切换——
取得 `+0.026574` gain，CI95 `[+0.023418,+0.029841]`，harmful/beneficial ratio
`0.053`。Gold XGBoost 的 oracle recovery 为 `56.6%`。

配对策略增量为：

- Probe XGBoost − query-only M3：`+0.012380`，CI95
  `[+0.008471,+0.016380]`；
- Gold XGBoost − Probe XGBoost：`+0.024431`，CI95
  `[+0.020462,+0.028516]`。

Stage 12 因此输出 `GOLD_EVIDENCE_SIGNAL_SUFFICIENT`。该名称表示 gold/evidence
信息足以解释失败机制，不表示 gold teacher 可部署。

## 16. 跨实验观察与结论

### 16.1 特征是否有效

Structured query/corpus-static 特征和 PCA-compressed embedding 中存在弱信号，M3
在三个 split 上均为正，AUC/Spearman 也高于随机。但其实际 utility gain、CI 和
harmful-loss 均不满足部署标准。因此结论不是“所有特征无效”，而是“query-only
特征信号太弱，无法形成可靠策略”。

### 16.2 直接使用 embedding 大矩阵是否合适

对当前树模型不合适。Raw 384D embedding 单独建模几乎无收益，raw 414D 拼接给
XGBoost 也平台化；随机旋转进一步显示 raw tree 对任意 embedding 坐标敏感。

### 16.3 直接拼接是否实现错误

实现和 fold-local 训练过程没有发现技术错误，但受控结果否定了它作为最佳 inductive
bias。有效的语义几何不会因为把 384 个坐标直接附加到 30 个 structured features
后就自动变得适合 axis-aligned trees。

### 16.4 XGBoost 是否是正确的默认模型

不是。最强 query-only 候选是 fold-local PCA32 后的线性 Ridge。Structured-only
XGBoost 和 raw-full XGBoost 均接近零或为负；late fusion 在小样本 screen 有优势，
但在 9,600 上下降。

### 16.5 训练目标是否合理

Continuous F1 gap 是当前最合理且相对稳定的主目标。Robust loss、weighted sign
classification 和 dual-utility heads 均未改善正式 policy utility。问题不能主要归因于
目标形式选择错误。

### 16.6 训练是否充分

9,600-query 学习曲线足以拒绝“继续扩大相同 query-only 数据和相同四个候选”的方向。
这不证明所有未来模型都不可行，但不支持继续扩展 raw 414D XGBoost 或无边界的
query-only model search。

### 16.7 主要缺失信号在哪里

Post-retrieval probe 相比 M3 的配对增量为 `+0.012380`，gold evidence 又在 probe
之上增加 `+0.024431`。因此主要缺失信号位于两种 Retriever 实际返回的排名、分数、
列表重叠、context 以及 evidence coverage，而不是 query wording。

## 17. 最终研究结论

1. 当前严格 pre-retrieval query-only Router 不可部署；
2. 最强 M3 只恢复约 3.5% oracle headroom，且 harmful switch 未受控；
3. Raw embedding + XGBoost、直接 414D 拼接和单纯增加同类训练数据均没有得到支持；
4. Continuous utility-gap 训练协议和 leakage-safe OOF 评价足以揭示模型不稳定性；
5. Qrels-free post-retrieval probe 显著优于 query-only，说明下一步应研究 shallow
   BM25+Dense probe/cascade 与保守的 harmful-switch control；
6. Gold/evidence teacher 只用于定位上界和机制，不能作为 online Router；
7. 本实验没有验证 Answer Correctness，也没有打开 fresh-dev/final holdout。

## 18. 局限与排除项

- 结果仅适用于 HotpotQA、旧 prompt、Top-5 generation context、GPT-4.1-mini 和
  normalized token F1 协议；
- 没有测试新版 prompt 下重新生成的完整 9,600-query 标签；
- 没有测试 MLP、PLS、encoder fine-tuning 或新的 embedding；
- Stage 12 probe 需要先运行两个 Retriever，不能直接等同于节省完整 Retriever 成本的
  pre-retrieval Router；
- 没有 Answer Correctness 结果，不能把 F1 gain 表述为已验证的语义正确性提升；
- evidence code HEAD 为 `a422e39398f456316f8166e13dcbd7d496be9ca5`，但执行时
  worktree 为 dirty；Phase 2.7 runner 和协议文件的当前精确版本仍依赖本地工作区，
  这是现阶段的代码 provenance 限制；
- 被中断、部分完成或被 9,600 run 替代的中间 formal outputs 不进入最终指标。

## 19. 验证状态

- Formal validation：`passed`；
- Stage 12 validation：`passed`；
- Database integrity：`ok`；
- Formal tasks：12；
- Formal prediction arrays：24；
- Learning-curve rows：48；
- Stage 12 trained candidate×split tasks：15；
- Fresh-dev / final-holdout rows read：0 / 0；
- 当前专项测试：`12 passed, 1 warning`；warning 为本地 `numexpr` 版本提示，不影响
  两组 Phase 2.7 测试通过。

来源、原始运行路径、哈希和排除项见
本地核心产物路径见 [`../../registry.yaml`](../../registry.yaml)，归档验证摘要见
[`results/validation.json`](results/validation.json)。
