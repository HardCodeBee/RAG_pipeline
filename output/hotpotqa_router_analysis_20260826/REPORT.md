# HotpotQA 预检索 Retriever Router：最新实验数据与分析报告

日期：2026-08-26

代码版本：[`64642100638c0a39f43d26cd9e43c8c78e281f40`](https://github.com/HardCodeBee/RAG_pipeline/commit/64642100638c0a39f43d26cd9e43c8c78e281f40)

分支：`codex/hotpotqa-router-analysis`

研究范围：严格 leakage-free、query-only/corpus-static 的 BM25 与 Dense/Contriever 预检索路由；最终效用为生成答案质量。

## 1. 执行摘要

最新证据支持两个同时成立的结论：

1. **Retriever 选择确实存在答案级别的逐 query headroom。** 在重复生成后的 600-query B/D 数据中，Dense 是最佳固定动作，但逐 query 事后 oracle 的 F1 比固定 Dense 高 `+0.05052`；在 4,800-query F1 screen 中，B/D oracle headroom 为 `+0.07071`。
2. **现有严格预检索特征无法稳定识别应该切换到 BM25 的 query。** 414 维完整特征下最好的 continuous-gain XGBoost 只比固定 Dense 高 `+0.002149`，95% CI 为 `[-0.000585, +0.004876]`，没有达到预注册的 `+0.01` 实用门槛。因此本轮结论为 `STOP_BEFORE_AC_LABELS`。

Contriever 修订也没有解决该问题：BM25/Contriever pair 本身有 `+0.06982` oracle headroom，但最强 Router 只比最佳 pair-fixed BM25 高 `+0.001208`，同时比固定 BGE 低 `-0.04329`。这说明当前瓶颈不是简单地换一个 Dense retriever，而更像是**严格预检索输入对“何时切换”缺乏足够可观测性**。

本包没有打开 fresh-dev 或 final holdout，没有新增 Answer Correctness 调用，也没有导出未完成的 9,600-query 状态。

## 2. 研究问题与边界

目标是在检索发生前，仅用当前 query 与静态 corpus 统计选择动作：

- `bm25`：SQLite/Lucene BM25；
- `dense` / `bge`：`BAAI/bge-small-en-v1.5` Dense retrieval；
- Contriever 修订中的 `contriever`：`facebook/contriever`。

严格预检索模型允许使用：

- 17 维 query/corpus lexical compatibility；
- 13 维冻结 corpus prototype summary；
- BGE 384 维或 Contriever 768 维 query embedding。

以下数据只允许作为离线 label 或诊断，禁止进入 Router 输入：当前检索结果、分数/rank、qrels、gold evidence、生成答案、F1、EM、AC。

本报告严格区分：

- retrieval-only 指标：nDCG、Recall、Hit；
- 答案指标：normalized token F1、EM、Answer Correctness；
- query oracle：事后逐 query 选择，属于不可部署上界；
- Router：只能用部署时可获得的预检索特征。

## 3. 冻结实验配置

### 3.1 数据与分组

- Dataset：HotpotQA，eligible source split 为 `train`，共 85,000 queries；
- Official dev 不参与训练；official test 只作为历史 retrieval evidence；
- Group：`normalized_answer_plus_supporting_page_titles`，防止相同信息需求跨 fold；
- Phase 2：train 300 + historical dev 300；
- Phase 2.6：冻结 train prefix 4,800；
- Contriever 修订：同一 train prefix 的前 1,200；
- fresh-dev：未打开；final holdout：未打开。

### 3.2 Retrieval 与生成

- BM25：Lucene BM25，`k1=1.5`、`b=0.75`、`english_regex_casefold_v1`；
- BGE：`BAAI/bge-small-en-v1.5`，revision `5c38ec7...`，normalized inner product；
- Contriever：`facebook/contriever`，revision `2bd46a2...`，masked mean，不归一化；
- Dense index：`streaming_flat_ip` exact shard-wise scan，不是 ANN；
- Candidate depth 50，最终 context Top-5，max context tokens 1,800；
- Generator：`gpt-4.1-mini-2025-04-14`，temperature `0.0`，每个 query/action 生成三次；
- AC judge：`gpt-5.4-mini-2026-03-17`，blind to action/retriever/context/retrieval metrics。

完整配置见：

- [`configs/hotpotqa_bd_router_config.yaml`](configs/hotpotqa_bd_router_config.yaml)
- [`configs/hotpotqa_contriever_router_config.yaml`](configs/hotpotqa_contriever_router_config.yaml)
- [`configs/requirements_experiment.txt`](configs/requirements_experiment.txt)
- [`configs/verified_constraints.txt`](configs/verified_constraints.txt)

## 4. 实验结果

### 4.1 Phase 2 revision：600-query 重复生成稳定性与 B/D headroom

每个 query/action 生成三次，先在 query/action 内取均值，再计算固定动作与 oracle。结果完整、无失败，gate 为 `GO`。

| Partition | Action | F1 | EM | AC |
|---|---|---:|---:|---:|
| Train, 300 | BM25 | 0.637989 | 0.557778 | 0.692778 |
| Train, 300 | Dense | 0.687148 | 0.597778 | 0.743333 |
| Dev, 300 | BM25 | 0.632704 | 0.545556 | 0.685556 |
| Dev, 300 | Dense | 0.695528 | 0.594444 | 0.758333 |

Dev 的 oracle 结果：

- best fixed：Dense，F1 `0.695528`；
- F1 oracle：`0.746050`；
- F1 headroom：`+0.050521`，95% CI `[0.031298, 0.073820]`；
- 按 F1-oracle 动作评估 AC：相对固定 Dense `+0.042778`；
- 独立 AC oracle headroom：`+0.046667`。

重复稳定性：

- normalized answer pair agreement：`0.90444`；
- F1 pair agreement：`0.95444`；
- EM pair agreement：`0.97556`；
- AC pair agreement：`0.96722`；
- leave-one-repeat-out 三个旋转均通过。

600-query pooled 描述中，F1 winner 为 BM25 `60`、Dense `106`、tie `434`；AC winner 为 BM25 `46`、Dense `86`、tie `468`。这说明可利用 headroom 集中在少量非 tie query，而不是大范围动作反转。

数据：

- [`data/phase2/phase2_bd_headroom_repeated_summary.json`](data/phase2/phase2_bd_headroom_repeated_summary.json)
- [`data/phase2/phase2_bd_600_outcomes.jsonl.gz`](data/phase2/phase2_bd_600_outcomes.jsonl.gz)：3,600 rows
- [`data/phase2/phase2_bd_600_query_summary.csv.gz`](data/phase2/phase2_bd_600_query_summary.csv.gz)：600 rows

### 4.2 Phase 2.5：F1 label 与 AC utility 的一致性

该阶段没有外部调用，使用已存在的 600-query、B/D、三重复结果审计 F1 是否可以承担低成本 Router screen。

| 指标 | Train | Historical dev |
|---|---:|---:|
| F1 winner：BM25 / tie / Dense | 29 / 222 / 49 | 31 / 212 / 57 |
| AC winner：BM25 / tie / Dense | 26 / 235 / 39 | 20 / 233 / 47 |
| AC non-tie 上 F1/AC sign agreement | 0.90769 | 0.94030 |
| AC-gap weighted sign agreement | 0.94576 | 0.97324 |
| F1 selector 的 AC gain over Dense | +0.05056 | +0.04278 |
| 上述 gain 的 95% CI | [0.02944, 0.07389] | [0.02278, 0.06556] |
| F1 selector switch coverage | 9.67% | 10.33% |
| F1 selector switch precision | 75.86% | 61.29% |

Cross-repeat F1 selector 的 mean AC gain 为：

- Train：`+0.046667`，CI `[0.025556, 0.069444]`；
- Historical dev：`+0.033889`，CI `[0.013333, 0.057222]`。

结论为 `GO_AC_LABEL_EXPANSION`，但其含义是“F1 可保留为辅助/screening label”，不是“已经有可部署 Router”。后续 Phase 2.6 的预检索预测 gate 仍必须独立通过。

数据：[`data/phase25/`](data/phase25/summary.json)。

### 4.3 Phase 2.6：4,800-query B/D 严格预检索 Router

冻结数据为 4,800 queries × BM25/Dense × 3 repeats，共 28,800 successful rows。模型使用 nested GroupKFold：5 outer folds、4 inner folds、5 model seeds；实用门槛为相对固定 Dense `+0.01`，且置信区间下界必须高于 0。

固定动作与 oracle：

| 项目 | F1 |
|---|---:|
| Fixed BM25 | 0.618936 |
| Fixed Dense | 0.647134 |
| B/D query oracle | 0.717841 |
| Oracle headroom over Dense | +0.070707 |

逐 query winner：BM25 `596`、Dense `797`、tie `3,407`。存在明显 oracle headroom，但正向切换 query 稀疏且类别高度不平衡。

| Candidate | Router F1 | Gain over Dense | 95% CI | Switch coverage | Switch precision | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|
| Full pairwise XGBoost | 0.646419 | -0.000715 | [-0.003756, 0.002293] | 9.81% | 11.68% | 2/5 |
| **Full gain XGBoost** | **0.649283** | **+0.002149** | **[-0.000585, 0.004876]** | 6.92% | 15.36% | 4/5 |
| Tier-A pairwise XGBoost | 0.645317 | -0.001817 | [-0.006099, 0.002583] | 17.98% | 13.33% | 0/5 |

最强 candidate 的 seed median gain 仅 `+0.000511`，seed range `[-0.000605, +0.002768]`。没有 candidate 达到 `+0.01`，也没有 CI 下界高于 0。因此冻结结论为：

`STOP_BEFORE_AC_LABELS`

这不是“BM25 没有价值”，而是现有 414 维预检索特征无法高精度识别少量 BM25-beneficial query。低 switch precision 是当前最直接的失败机制。

数据与特征：

- [`data/phase26/summary.json`](data/phase26/summary.json)
- [`data/phase26/validation.json`](data/phase26/validation.json)
- [`data/phase26/phase26_bd_4800_outcomes.jsonl.gz`](data/phase26/phase26_bd_4800_outcomes.jsonl.gz)：28,800 rows
- [`data/phase26/phase26_bd_4800_query_summary.csv.gz`](data/phase26/phase26_bd_4800_query_summary.csv.gz)：4,800 rows
- [`features/phase26/phase26_features_4800.npz`](features/phase26/phase26_features_4800.npz)
- [`features/phase26/feature_schema.json`](features/phase26/feature_schema.json)

### 4.4 Contriever 修订：检验 retriever pair 是否是主要问题

该实验在前 1,200 train queries 上比较 BM25、Contriever，并保留 BGE 为外部固定参考。每个 action 三次生成，共 10,800 successful rows；不读取 fresh-dev/final holdout，不调用 AC。

固定动作：

| Action | F1 |
|---|---:|
| BM25 | 0.611435 |
| BGE | **0.655932** |
| Contriever | 0.577866 |

BM25/Contriever pair oracle 为 `0.681258`，比最佳 pair-fixed BM25 高 `+0.069823`。但 Router 结果为：

| Candidate | Gain over pair-fixed BM25 | 95% CI | Gain over fixed BGE | Switch precision |
|---|---:|---:|---:|---:|
| Full pairwise XGBoost | -0.000153 | [-0.003889, 0.003638] | -0.044649 | 14.29% |
| **Full gain XGBoost** | **+0.001208** | **[-0.002219, 0.004823]** | **-0.043288** | 16.00% |
| Tier-A pairwise XGBoost | -0.003561 | [-0.007688, -0.000070] | -0.048058 | 10.81% |

结论为 `ARCHITECTURE_DIAGNOSIS_REQUIRED`。换用 Contriever 后仍出现“oracle 很高、Router 几乎不能兑现”的结构；而且全局性能显著低于固定 BGE，因此不能把 Contriever 当作新的默认动作。

数据与特征：

- [`data/contriever/summary.json`](data/contriever/summary.json)
- [`data/contriever/phase3_contriever_1200_outcomes.jsonl.gz`](data/contriever/phase3_contriever_1200_outcomes.jsonl.gz)：10,800 rows
- [`data/contriever/phase3_contriever_1200_query_summary.csv.gz`](data/contriever/phase3_contriever_1200_query_summary.csv.gz)：1,200 rows
- [`features/contriever/contriever_features_1200.npz`](features/contriever/contriever_features_1200.npz)

### 4.5 NQ DPR 表征诊断

该实验是离线 qrels-labeled representation diagnostic，不运行 retrieval、reranking 或 generation，也不训练部署 Router。

- NQ total queries：3,452；
- BM25-only：260；BGE-only：1,049；exclusive set：1,309；
- DPR embedding：768 维，`facebook/dpr-question_encoder-multiset-base`；
- DPR-only linear probe OOF ROC-AUC：`0.6774`，95% CI `[0.6429, 0.7120]`；
- surface features ROC-AUC：`0.6012`；surface+DPR：`0.6631`；
- kNN same-label purity @10/@20/@50：`0.5330/0.5280/0.5176`；
- cosine silhouette：`0.0037`。

结论：存在方向性可解码信号，但没有形成稳定的局部聚类或自然 decision boundary，未达到预注册的 practical geometry threshold。完整数据位于 [`data/nq_dpr_probe/`](data/nq_dpr_probe/summary.json)。

## 5. 综合判断

当前证据最支持以下解释：

1. **Utility headroom 不是主要瓶颈。** B/D 和 BM25/Contriever 都有约 `0.07` 的 query-oracle headroom。
2. **动作标签稀疏且 tie 很多。** 4,800-query B/D 中 `70.98%` 为 F1 tie；有利于 BM25 的 query 只有 `12.42%`。
3. **现有模型会切换，但切换精度不足。** 最强 Phase 2.6 Router 的 switch precision 只有 `15.36%`；错误切换抵消了正确切换收益。
4. **增加 embedding 维度或更换 Dense retriever 没有解决 observability gap。** 414 维 BGE feature 与 798 维 Contriever feature 均未过 gate。
5. **不应继续扩大昂贵 AC labels。** F1 screen 没有达到 `+0.01`，按冻结协议应停止在 AC expansion 之前。

因此，下一步更值得分析的是：

- BM25-beneficial query 是否能被 evidence relevance、candidate overlap、retrieval rank 等特权信号稳定识别；
- 如果特权 teacher 可以识别，能否蒸馏回严格预检索 student；
- 若不能，是否需要改变信息边界，引入一个低成本 retrieval probe，而不是继续堆叠 query-only 特征；
- 按 utility gap、evidence recall gap、问题类型、答案类型和 group density 分层后，是否存在稳定的可路由子群。

这些检索后/特权变量只能用于机制诊断，不能直接被描述为已部署 Router 输入。

## 6. 建议对方优先执行的分析

1. 用 `phase26_bd_4800_query_summary.csv.gz` 分析 BM25 winner、Dense winner、tie 三类的 feature distributions 和效应量；
2. 检查 17 lexical、13 corpus summary、384 embedding 各 block 的单变量 AUC、互信息、稳定性和 group-OOF calibration；
3. 以连续 `BM25 mean F1 − Dense mean F1` 为目标，分析 label noise、repeat variance 与 margin sensitivity；
4. 将 Router 失败拆成 false switch 与 missed switch，分别看 evidence recall、retrieval hit、query pattern；
5. 对 5 folds × 5 seeds 检查 threshold drift、switch coverage drift 和 subgroup consistency；
6. 用 privileged post-retrieval teacher 测试可观测性上限，但与 deployable student 结果分开报告；
7. 对 Contriever 实验判断 pair oracle 来自哪些 query，并核对这些 query 是否同时被固定 BGE 正确解决。

## 7. 数据包结构

```text
hotpotqa_router_analysis_20260826/
├── REPORT.md
├── DATA_DICTIONARY.md
├── configs/                  # 实际运行配置与依赖版本
├── data/
│   ├── phase2/              # 600-query 重复稳定性与 headroom
│   ├── phase25/             # F1/AC label audit
│   ├── phase26/             # 4,800-query B/D screen
│   ├── contriever/          # 1,200-query mechanism revision
│   ├── nq_dpr_probe/        # NQ DPR representation diagnostic
│   └── retrieval_context/   # 仅保留小型 Selected-7 背景汇总
├── features/                # 与冻结 query 顺序绑定的 NPZ 特征
└── code/                    # 本包数据导出脚本
```

字段定义见 [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)，整体导出统计见 [`data/derived_metrics.json`](data/derived_metrics.json)。

## 8. 代码与复现入口

实验代码已经推送到 GitHub，本包不重复复制仓库源代码。应从冻结 commit 读取：

- [`audit_router_phase25.py`](https://github.com/HardCodeBee/RAG_pipeline/blob/64642100638c0a39f43d26cd9e43c8c78e281f40/scripts/audit_router_phase25.py)
- [`run_router_phase26_bd_f1_gate.py`](https://github.com/HardCodeBee/RAG_pipeline/blob/64642100638c0a39f43d26cd9e43c8c78e281f40/scripts/run_router_phase26_bd_f1_gate.py)
- [`run_router_phase3_contriever.py`](https://github.com/HardCodeBee/RAG_pipeline/blob/64642100638c0a39f43d26cd9e43c8c78e281f40/scripts/run_router_phase3_contriever.py)
- [`run_router_phase3_train.py`](https://github.com/HardCodeBee/RAG_pipeline/blob/64642100638c0a39f43d26cd9e43c8c78e281f40/scripts/run_router_phase3_train.py)
- [`analyze_dpr_retriever_preference_space.py`](https://github.com/HardCodeBee/RAG_pipeline/blob/64642100638c0a39f43d26cd9e43c8c78e281f40/scripts/analyze_dpr_retriever_preference_space.py)

本包自身新增一份辅助代码：

- [`code/export_compact_data.py`](code/export_compact_data.py)：从原始 JSONL/SQLite 导出冻结、去上下文的分析数据。

## 9. 未包含内容

为避免泄漏、冗余和无效传输，本包不包含：

- 5.23M HotpotQA corpus、完整 queries/qrels、Dense vectors、BM25 index；
- 285 MB 的 Phase 3 SQLite state、49.8 MB 的 Contriever SQLite state、WAL/SHM；
- 未完成的 9,600-query outcomes（65,500 success + 20,900 pending），它们不进入任何正式指标；
- fresh-dev/final-holdout outcomes；
- checkpoints、joblib model、provider checkpoints；
- context 全文、provider request/response、judge 原始回复、API 凭据。

原始 context 和完整运行状态仍保留在本机；如确有需要，应在明确的内部访问边界下另行提供，不能与当前可分享分析包混合。

## 10. 数据范围确认

导出时检查：

- Phase 2：3,600 outcome rows / 600 query rows；
- Phase 2.6：28,800 outcome rows / 4,800 query rows；
- Contriever：10,800 outcome rows / 1,200 query rows；
- fresh-dev rows：0；final-holdout rows：0；
- context text/provider payload/API credentials：0；
- Phase 2.6 feature shape：`4800 × (17 + 13 + 384)`；
- Contriever feature shape：`1200 × (17 + 13 + 768)`。
