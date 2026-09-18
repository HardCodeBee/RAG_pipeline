# Phase 2.11：旧 30D 与新 35D 特征的直接 Router 对照

执行日期：2026-09-10。状态：已完成；证据性质：特征筛选后的内部诊断。

本次实验未得到新版优于旧版的明确证据。三次划分的 OOF 分数取均值后，新减旧的平均答案 F1 为 **+0.000879**，配对 group bootstrap 95% CI 为 **[-0.001380, +0.003178]**。

## 研究问题

将 `features_new.md` 已确定的 35 个字段整体替换旧 30 维结构化特征，在保留相同 PCA32 和 M3 训练流程时，能否提升检索策略选择后的答案 F1？本轮只比较两个固定候选，不重新筛选字段或搜索阈值。

## 实验设计

| 项目 | 固定设置 |
|---|---|
| 样本 | 原始 Phase 2.7 的 9,600 条 train-only query，9,559 个 group；保留全部 exact ties |
| 监督目标 | 每个 query/action 的 3 次答案 F1 均值；预测 BM25 减 Dense 的效用差 |
| 旧版输入 | 17D lexical + 13D prototype summary + PCA32，共 62D |
| 新版输入 | 文档所列完整 35D + 同一 PCA32，共 67D |
| 模型 | M3 Ridge，alpha=10，model seed=11 |
| 数据划分 | 5 outer / 4 inner StratifiedGroupKFold；split seeds 为 20260901、20260917、20261003 |
| 预处理 | StandardScaler、embedding 中心化、未白化 randomized PCA 均只拟合当前训练折 |
| 校准 | outer-train 内的 inner-OOF 预测拟合非负斜率 affine calibration |
| 决策 | 校准后的预测 gap > 0 时选 BM25，否则选 Dense |
| 主比较 | 三次 calibrated OOF 分数均值对应策略的新减旧平均 F1 |
| 不确定性 | 按 group_id 配对重采样 10,000 次，seed=20260902；三个效用对照使用同一批抽样 |

样本类别：BM25 winner 1,196，Dense winner 1,564，exact tie 6,840。这批原始样本用于恢复历史对照；本轮没有改用 winner-balanced T2。

新 35D 使用 query 文本统计、冻结语料统计、固定样本文档上的无排序聚合及冻结 prototype 摘要。保留的 384D BGE embedding 完全相同。模型输入不包含当前检索返回的文档、分数、qrels 或答案 F1；已有答案 F1 仅用于监督和评价。

35D 的形成参考过之前的特征筛选结果，其中 discovery 与本次样本重叠 2,760 条。因此，虽然训练、PCA 和校准均遵守折内拟合，本次 OOF 也不能消除历史特征选择带来的偏差。置信区间表示这组已选择候选和既有 OOF 预测下的内部差异，不能视为未接触数据上的确认结果，也不包含重新训练或重新筛选特征的不确定性。

## 特征与流程检验

旧缓存缺少目标 query/词形变体所需的 3,806 个词项。本次沿用原 500,000 文档样本和 seed=2026083001 重建缓存；74,021 个共同词项的统计和 postings、整份样本的文档编号与长度均逐项一致。缺缓存项没有被默认为 OOV。

重新提取的 2,760 条重叠样本，在 query/group 对齐后，35 列与原 discovery 特征逐值完全一致；其余样本采用相同公式计算。所有 9,600 × 35 个值均有限。

常量字段数：0；非零比例低于 1% 的字段数：0。完整字段检查见 [feature_checks.csv](results/feature_checks.csv)。

| 旧版复现 split seed | 最大预测差 | 划分相同 | 决策相同 |
|---|---:|---|---|
| 20260901 | 0 | True | True |
| 20260917 | 0 | True | True |
| 20261003 | 0 | True | True |

旧版历史复现通过后才运行新版。新旧所有 outer fold 编号一致。字段分布统计仅作检查，没有据此删列或调参。

## 实验结果

| 策略 | 平均答案 F1 | 相对固定 Dense | 95% CI |
|---|---:|---:|---|
| 固定 Dense | 0.648209 | 0 | — |
| 固定 BM25 | 0.620175 | -0.028034 | — |
| 旧 30D + PCA32 | 0.650664 | +0.002455 | [-0.000083, +0.005069] |
| 新 35D + PCA32 | 0.651543 | +0.003334 | [+0.000617, +0.006070] |

新减旧的 **+0.000879 F1** 相当于 **+0.0879 个 F1 百分点**。这里的 F1 是答案质量指标，不是 Router 分类准确率，也不是 Answer Correctness。

| split seed | 旧版相对 Dense | 新版相对 Dense | 新减旧 |
|---|---:|---:|---:|
| 20260901 | +0.002792 | +0.003478 | +0.000686 |
| 20260917 | +0.002193 | +0.002210 | +0.000017 |
| 20261003 | +0.002248 | +0.003474 | +0.001226 |

上表每行使用同一批 9,600 条样本的一次重新划分；三行并非三份独立数据。主结果是连续 OOF 分数平均后执行策略，不是三行 F1 的简单平均。

| 决策诊断 | 旧版 | 新版 |
|---|---:|---:|
| BM25 切换数 | 1558 | 1693 |
| 有益切换数 | 186 | 210 |
| 有害切换数 | 162 | 196 |
| 中性切换数 | 1210 | 1287 |
| 遗漏 BM25 优势 query | 1010 | 986 |
| 切换覆盖率 | 0.162292 | 0.176354 |
| 有害损失 / 有益收益 | 0.789190 | 0.752067 |
| non-tie AUC | 0.580132 | 0.571558 |

新旧在 1123 条 query 上选择不同：133 条改善、143 条变差、847 条 F1 不变。

| 旧决策 → 新决策 | query 数 | 对新减旧整体 F1 的贡献 |
|---|---:|---:|
| dense → dense | 7413 | +0.000000 |
| dense → bm25 | 629 | +0.002014 |
| bm25 → dense | 494 | -0.001135 |
| bm25 → bm25 | 1064 | +0.000000 |

### 原有推进门槛：仅作背景对照

原有门槛要求跨 split 的平均 Dense 增益至少 +0.01、consensus CI 下界大于 0、所有 split 增益为正，并满足损失收益比、切换覆盖率和校准要求。它与本次主问题“新特征是否优于旧特征”分别记录。

| 原有门槛 | 旧版 | 新版 |
|---|---|---|
| practical_gain | 未通过 | 未通过 |
| grouped_bootstrap_ci_lower | 未通过 | 通过 |
| all_split_seed_gains_positive | 通过 | 通过 |
| harmful_to_beneficial_mass_ratio | 未通过 | 未通过 |
| switch_coverage | 通过 | 通过 |
| calibration_slope | 通过 | 通过 |
| top_decile_realized_gap | 通过 | 通过 |

## 结论与适用范围

本次实验未得到新版优于旧版的明确证据。

三个 split 的新减旧点估计均为正，显示小幅正向趋势；但主比较的置信区间跨过 0。新版相对固定 Dense 的区间下界转为正，也不能替代新减旧的直接检验。

新版有害损失 / 有益收益仍为 0.752，高于原门槛 0.5；效用增益也低于 +0.01 的实际收益要求。因此，本轮既不能确认新版优于旧版，也没有通过原有推进门槛。

本次结论只针对完整新 35D 在固定 M3 recipe 下的整体替换；不能据此确定单个字段是否有效、证明新特征完整覆盖旧信息，也不能把结果推广为所有模型下的特征优劣。训练方式、PCA 维度和阈值均未同时改动。

本轮已完成内部比较；没有打开 natural-2000 outcomes、fresh natural 或官方 final holdout，也没有新增检索/生成调用、线上 Router 或部署。后续若进行更强的确认，需要新的独立样本和事先固定的验证协议。

## 复现与文件

- [固定配置](config.yaml)
- [紧凑结果 JSON](results/summary.json) 与 [结果 CSV](results/summary.csv)
- [生效配置及环境](../../../../outputs/router/hotpotqa_bd_router_v1/runs/phase31_feature_comparison_v1/effective_config.json)
- [35D 字段定义、分布与提取检查](../../../../outputs/router/hotpotqa_bd_router_v1/runs/phase31_feature_comparison_v1/feature_extraction.json)
- [新 35D 矩阵](../../../../outputs/router/hotpotqa_bd_router_v1/runs/phase31_feature_comparison_v1/new35_features.npz) 与 [新旧 OOF 预测](../../../../outputs/router/hotpotqa_bd_router_v1/runs/phase31_feature_comparison_v1/predictions.npz)
- [逐 query 对照数据](../../../../outputs/router/hotpotqa_bd_router_v1/runs/phase31_feature_comparison_v1/query_comparison.csv.gz)
- [运行入口](../../../../scripts/run_router_phase31_feature_comparison.py) 与 [特征提取薄接口](../../../../scripts/router_phase31_features.py)

在仓库根目录运行：

```powershell
C:\Users\12442\anaconda3\python.exe -B -X utf8 scripts\run_router_phase31_feature_comparison.py
C:\Users\12442\anaconda3\python.exe -B -X utf8 scripts\router_workspace.py verify phase31
```

逐 query 对照表包含 query/group ID、问题文本、两种固定动作的 mean F1、真实 gap、新旧 consensus 预测 gap、新旧动作及其实际 F1、新减旧效用差。它是分析数据，不能作为推理输入。

逐 seed 检查点保留在本轮 run 目录；配置或脚本变化会拒绝沿用该运行目录。原 `features_old.md`、`features_new.md` 未修改。复用训练模块的本地测试为 9 passed；独立核验重新计算了全部 split/consensus 策略效用、三个配对置信区间及决策变化，均通过。已有 pandas/numexpr 版本提示不影响测试完成。测试继续保留在本地忽略目录，没有改动 `.gitignore`。
