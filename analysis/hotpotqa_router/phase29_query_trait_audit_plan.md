# Phase 2.9：HotpotQA BM25/BGE 原始 Query 单特质审计

## 目标

本阶段只回答一个问题：在不改写 query、不查看本次 top-k、不使用 qrels、支持事实或生成结果作为特征的前提下，是否存在某一个原始 query 特质，能够稳定区分 BM25 与 BGE 的 answer-F1 优势。

本阶段不是特征组合实验，也不是 Router 训练实验。单特质证据过差时，不再尝试组合；全部严格验证后没有信号时，不进入模型诊断及之后阶段。

## 数据边界

1. 发现集：Phase 2.8 T2 中的 3,000 个 BM25 winner 与 3,000 个 Dense winner。6000 个 exact ties 不进入主二分类。
2. 来源稳健性：在发现集内部，分别重算旧 9,600-query cohort 和 Phase 2.8 新增 cohort 的效应，不允许方向翻转或明显塌缩。
3. 自然复现：此前按 group 均匀抽取的 2,000-query natural confirmation。它包含 BM25 267、Dense 337、tie 1,396；已用于旧候选选择，因此只能称预注册自然分布复现，不是新的最终测试集。
4. 官方 final holdout 保持未开启。

answer target 固定为：

\[
g(q)=\overline{F1}_{BM25}(q)-\overline{F1}_{BGE}(q)
\]

winner endpoint 只使用非 tie；continuous-gap endpoint 保留真实 gap。检索 evidence-page recall 只作机制辅助，不能救回 answer-F1 未通过的特征。

## 候选特质

候选集合在读取 outcome 关联前由代码生成清单、公式、类型、信息边界和 SHA-256，并冻结到 run directory：

- 历史 17D lexical 特征逐维；
- 历史 13D frozen corpus-prototype summary 逐维；
- 与实际 BM25 unique-term contract 对齐的修正版 17D；
- Stage C：collection frequency/selectivity、query scope、document-level co-occurrence/coherence、global term-impact、document-length compatibility、title/body compatibility、deterministic suffix-alternative competition（不是词典或 lemmatizer）；
- 语言学特质：tokenizer/surface-suffix、relation lexicalization、entity/anchor、lexical economy、answer-type/constraint、ambiguity、Hotpot multi-hop、一般 style controls。

Stage C 无条件执行，不以较早特征是否出现信号为前提。Stage C 的 scope、coherence、title/body 与 global impact 使用一个在结果分析前固定的 500,000-document uniform corpus sample；它会对这个冻结样本索引执行无排序的 query-conditioned 集合聚合，但不执行主 BM25/BGE top-k，不产生 ranked list，也不读取 qrels。exact DF/IDF/CF 仍来自完整 BM25 index。报告不得把 sample scope/title coverage 写成全库精确值，也不得把 title/body compatibility 写成 BM25 field contribution。

不纳入单特质集合：384D embedding 坐标、PCA、特征乘积、交互、学习出的组合分数、当前 query 的 top-k/rank/score、qrels、gold evidence、生成答案、F1、acquisition stratum。

## 执行顺序

1. `preflight`：先冻结 feature catalog 与 config/runner/extractor SHA-256，再读取 discovery 标签验证行数、类别数、ID 对齐和旧/新来源计数；自然 outcome 只验证文件哈希，不在 shortlist 前解析；官方 final holdout 零读取。
2. `build-corpus-cache`：用固定 seed 构造 500k 全库静态样本倒排缓存及 exact term stats；不读取任何 outcome 标签。
3. `extract-discovery`：为 6,000 条非 tie query 提取所有冻结单特质；历史 30D 与 snapshot 逐行对齐。
4. `analyze-discovery`：逐特质计算 winner AUC/OR、gap Spearman、Q4-Q1 gap 或二元 gap difference、group permutation、group bootstrap 和 Holm 校正。
5. `old/new stability`：使用完整发现集冻结的方向分别检查旧/新增 cohort，不重新翻转方向。
6. `freeze-shortlist`：只冻结通过完整 discovery 与 old/new 门槛的单特质。若为零，写出正式 STOP 决策并结束。
7. `natural confirmation`：仅在 shortlist 非空时提取并评估自然 2,000；公式、方向、阈值、缺失规则不再改变。若复现为零，正式 STOP。
8. 只有至少一个 `supported_single_feature` 时，才允许另立受控协议考虑组合或模型；本阶段本身不训练 Router。

## 单特质门控

连续特质在发现集必须同时满足：directional AUC ≥ 0.56 且 CI lower > 0.50；directional Spearman ≥ 0.10 且 CI 不含 0；directional Q4-Q1 mean gap ≥ 0.020 且 CI lower > 0；winner 与 gap 两组 Holm-adjusted p 都 ≤ 0.05。

二元特质必须同时满足：directional OR ≥ 1.30 且 CI lower > 1；directional mean-gap difference ≥ 0.020 且 CI lower > 0；两组 Holm-adjusted p 都 ≤ 0.05。

old/new 两个来源不得反向，并须保留至少 50% 的完整发现效应。自然复现阈值固定为 AUC ≥ 0.55、Spearman ≥ 0.08、gap effect ≥ 0.015；二元 OR ≥ 1.25、gap effect ≥ 0.015，并继续要求校正后显著和 CI 门槛。

本版推断要求缺失率严格等于 0；连续值不足 20 个/IQR 为零/单值占比超过 90%，或者二元 level 数量不足时，直接记为不可操作或自然变异不足。不插补，也不临时增加 missing indicator。自然集会分别检查全部 2,000 条 gap endpoint 和 604 条 non-tie winner endpoint。

## 结论边界

通过只能说明该单特质在原始 query 上具有稳定、可复现的单变量 answer-utility 关联；不能自动说明组合必然更强，也不能说明 Router 已经获得实际 policy gain。

零通过的正式结论是：“冻结的原始-query单特质中，没有一项在当前 HotpotQA BM25/BGE answer-F1 目标上通过完整门控。”届时禁止在同一发现集/自然复现集上改阈值、试交互、拼接特征或训练模型来挽救结果。
