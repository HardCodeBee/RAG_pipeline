# 已有评估与相关实验

本页从既有报告抄录，不产生新的模型输出或评分。所有分数均为百分制；原始JSON/JSONL仍使用0—1。历史结果按数据集和协议分列，不混入未来主评。

## P02：历史完整RAG开发300题

三个数据集各100题，固定且已多轮查看/使用的开发题集。`timing_only` 与冻结P02采用相同记录，精确回放不是额外一轮独立验证。

| 数据集 | 题数 | Answer F1 | EM | 状态 |
| --- | ---: | ---: | ---: | --- |
| hotpotqa | 100 | 78.9908 | 66.0000 | 历史完整流程，已冻结 |
| 2wiki | 100 | 93.3474 | 87.0000 | 历史完整流程，已冻结 |
| musique | 100 | 39.8343 | 31.0000 | 历史完整流程，已冻结 |
| 三集等权/300题均值 | 300 | 70.7242 | 61.3333 | 不是正式500题结果 |

完整逻辑流程记录的输入tokens合计 2,837,258、输出tokens合计 28,223；按当时冻结费率估算生成费用 ¥2.326252。该值包含复用输入对应的逻辑开销，不等于这次整理产生的实际支出；不含GPU机时。已有客户端耗时混有缓存与并发，不能当作新计时方案的冷运行均值。

来源：[缓存修复后的完整报告](evidence/outputs/query_decomposition/hotpot_formal_eval_v1/p02_merge_cachefix_v1_20261021/reports/REPORT.json)、[逐题开发结果](results/p02_development300.jsonl)、[独立包冻结身份](../../src/p02_multigrain_timing_cachefix_v1/FROZEN.json)。

## 直接简化与缓存修复实验

以下版本均是同一开发300题上的完整流程历史结果，保留未采用和下降的实验。仅有 `timing_only` 被采用；各版本之间不是新的独立数据集。

| 版本 | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 | 均值F1 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| timing_only | 78.9908 | 93.3474 | 39.8343 | 70.7242 | 当前采用，300题精确等价回放 |
| union20_legacy | 78.2813 | 92.2855 | 40.3705 | 70.3124 | 未采用；存在缓存执行混杂 |
| discount20 | 79.4086 | 91.1188 | 39.5478 | 70.0251 | 未采用 |
| single_final | 74.4786 | 89.5172 | 39.0327 | 67.6761 | 未采用，质量下降 |
| union20_cachefix | 78.4242 | 92.2855 | 40.3705 | 70.3601 | 缓存修复后的对照，未采用 |
| terminal_goal | 72.8198 | 92.7043 | 37.8371 | 67.7871 | 未采用，质量下降 |

完整变动、费用、回执和因果解释边界见 [SIMPLIFICATION_ARCHIVE.json](../../src/p02_multigrain_timing_cachefix_v1/history/SIMPLIFICATION_ARCHIVE.json)。旧union与修复union不能拼接择优，缓存问题也不能归因于方法机制。对应原始轨迹与报告一并保留。

## 早期完整流程比较与路线参考

S02按当前约定仅作历史路线参考，不重跑。早期B0、角色头、goal/readout等探索及同300题基线报告分别保留其原始样本范围、协议、逐题配对和费用。它们并非修订006所要求的论文配置主评。

| 历史记录 | 入口 |
| --- | --- |
| S02与多粒度路线的开发300题完整流程比较 | [REPORT.txt](evidence/outputs/query_decomposition/hotpot_formal_eval_v1/complete_pipeline_comparison_v1_20261020/REPORT.txt) |
| 同300题的早期基线对照及失败核查 | [REPORT.txt](evidence/outputs/query_decomposition/hotpot_formal_eval_v1/baselines_same300_v1_20261020/REPORT.txt) |
| P02角色头直接父版本及探索结果 | [FINAL_RESULTS.frozen.json](evidence/outputs/query_decomposition/hotpot_formal_eval_v1/b0_strategy_round2_v1_20261020/FINAL_RESULTS.frozen.json) |
| R02分解器训练与120题分解质量确认 | [REPORT.txt](evidence/outputs/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018/REPORT.txt) |

分解器、prompt与检索组件实验是P02来源说明，不进入完整RAG主排名。报告中的同一Codex评审不包装成独立人工金标。

## 正式500题 × 三数据集：当前仍未完成

这是**旧P02协议**的执行库存；新统一层P02没有结果。500题仍可恢复，因此本页不发布部分正式F1/EM作为主成绩。以下状态数量只用于说明剩余工作，不是新增主表指标。

| 数据集 | 已成功完整RAG | 已处理但不完整 | 尚未完成流程 | 预生成分解 |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | 283 | 38 | 179 | 500 |
| 2wiki | 498 | 2 | 0 | 500 |
| musique | 0 | 0 | 500 | 500 |

旧原始 `REPORT.txt`/`SCORES.json` 含当时按可用子集聚合的分数，为保证历史证据完整在 `evidence/` 中原样保留；这些分数不能改名为完整500题成绩。全部1500题逐题明细保留未完成行、空分值与原始来源。

[逐题明细](evidence/outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/SCORES.jsonl) · [冻结正式协议](../../experiments/query_decomposition/p02_formal1500_v1_20261021/PROTOCOL.frozen.json) · [复用核查](../../experiments/query_decomposition/unified_rag_eval_v1_20261021/reviews/20261021T051154Z/REUSE_AUDIT.json)

## 尚不能从这些结果得出的结论

- 历史开发300题领先不构成正式500题或隐藏测试上的优势证明。
- 精确回放与独立目录核验检验打包/执行一致性，不是新样本推理。
- 未完成的正式流程不能按已有答案的子集排名；未知usage/请求不能补成零。
- 修订006调整了基线配置来源，旧10533条v2结果、781条P02候选和946条原始基线候选是迁移前库存，不能直接当作新协议的已完成数量。

## 已有时间与token汇总

另外保留了两份在本轮计划统一前生成的离线分析：

- [历史阶段计时汇总](evidence/analysis/qa_flow_time_20261021/reports/20261021T034522581426Z/REPORT.txt)：已有组件或跨阶段耗时重建，缓存/排队/API边界与当前计时协议不同。
- [历史P02与基线token汇总](evidence/analysis/query_tokens_p02_baselines_20261021/reports/20261021T041654095199Z/REPORT.md)：保留原始样本覆盖、完整usage要求和同题比较，旧Iter使用k10/T5。

这些报告中的“当前正式”指各自历史生成时刻，不是修订006的新主评。其部分样本均值不能替代未来固定50题时间补测，也不能改称500题统一结果。逐题明细无损压缩，分析脚本只收录、不执行。
