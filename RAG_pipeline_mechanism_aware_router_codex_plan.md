# Mechanism-Aware Retriever Router：本地审计后的精简执行计划

> 修订日期：2026-08-21  
> 状态：Phase 0–2 已完成；修订后 Phase 3 完成 1,200/2,400/4,800 档后提前停止，未执行 9,600、fresh dev 或 AC 复核  
> 当前主线：低成本、pre-retrieval 的 No-retrieval/BM25/Dense 路由，以 Answer Correctness 为主要效用，Answer F1 仅作参考

## 0. 本计划的解释规则

### 0.1 权威顺序

研究意图冲突时，按以下顺序解释：

1. 已记录的用户约束：最终答案效用优先、严格区分证据层级、模型契约需确认、代码简洁轻量；
2. 最新反思：`topic.md` 中的核心问题与 `../work/thinking01.md:22-42` 的“修正后的研究方向”；
3. 当前 checkout、ignored 本地产物、manifest 与实际结果；
4. 旧 `thinking.md`、ChatGPT 提议、论文启发和本计划中的候选设计。

仓库内没有名为 `think.md` 的文件。本版将用户本轮所称的 `think.md` 解释为最新的 `topic.md`，并用相邻目录的 `thinking01.md` 校正主线；旧 `thinking.md` 中的 history-aware 方向不回到当前 MVP。

事实判断另以当前本地文件、manifest 和可复核输出为准。GitHub 中未跟踪不等于本地不存在。

### 0.2 证据标签

- **已实现**：当前代码能执行，但不等于已经得到研究结论。
- **已运行**：有本地结果、协议与 provenance；只支持其原始指标范围。
- **待验证**：研究假设或候选方法，不得写成已证明机制。
- **待确认**：涉及模型、输入格式、评估或成本契约，执行前必须由用户确认。

### 0.3 总原则

- 先证明答案效用标签可信，再证明存在 action gap，再证明该 gap 可由部署前信息预测，最后才验证机制监督。
- 第一版 action set 是 $\mathcal A_0=\{N,B,D\}$，其中 $N$ 表示不检索、使用相同 prompt 与空 context；当前不研究 Hybrid 或其他 Dense retriever。
- retrieval、evidence、answer、latency 和 token/cost 分层报告。retrieval gain 不能冒充 RAG gain。
- oracle、post-retrieval teacher、test-qrels 结果只作非部署上界或诊断。
- 优先复用现有 loader、artifact registry、candidate cache、runner、pipeline 和 evaluator；只读取现有运行元数据，不把 registry/metadata 内容复制进 Router 数据，也不增加新的元数据层。没有第二个真实消费者时，不抽象公共框架。
- 未经明确要求，不改 `.gitignore`，不创建 branch，不 commit，不 push。

### 0.4 AC-primary / N-B-D 协议修订

- 最终研究结论、正式 oracle/regret 和 Phase 4–5 Gate 统一以三档盲评 Answer Correctness 为 primary utility；`incorrect/partially_correct/correct` 映射为 $0/0.5/1$，同一 query/action 的多次生成先取均值。
- Phase 3 按用户确认采用 normalized token F1 作为低成本早筛标签和 Gate。F1 只决定是否值得新增 AC judgment；即使通过也不能替代 AC 结论，未通过也不能推导 AC Router 在理论上必然失败。EM 始终只作 reference/diagnostic，不与 AC 或 F1 加权合成任意 scalar。
- Phase 3 不训练 R3AG 式 encoder；只借用 query–retriever compatibility 的问题表述，实际采用 LTRR 风格的轻量 pairwise Router。
- 2026-08-20 已完成的双 action、F1 regression Phase 3 结果保留为历史 pilot，不作为修订后 Phase 3 的结论；修订后使用 N/B/D action table 和 F1 pairwise 早筛协议，只有早筛通过并再次获得用户授权后才补做 AC 确认。

---

## 1. 对旧计划的审计结论

| 旧计划问题 | 修正 |
|---|---|
| 一开始固定 BM25/Dense/Hybrid 三动作 | 当前只研究 No-retrieval/BM25/Dense；Hybrid 与其他 retriever 不进入本轮 action set |
| 用 Selected-7 nDCG headroom 决定 RAG 研究是否继续 | 以 answer-level oracle headroom 为主要 gate；retrieval 只作诊断 |
| answer/evidence 被判断为本地缺失 | HotpotQA query 已含 answer/supporting facts；真正缺口是 evaluator-side adapter、Answer EM/F1 与经校准的 Answer Correctness（AC） |
| 把 ignored 的 tests/docs/artifacts 当成不存在 | 先审计和复用本地内容；Git 跟踪状态不是研究 readiness gate |
| 重建 RRF、oracle、matrix、records、persistence | 复用现有分析脚本与持久化；只保留一个实验 config、一个 split 文件和一个扁平 action table，不新增 Router 元数据层 |
| 为尚未验证的方法预建 `src/routing/` 大目录 | 先用 1 个 evaluator 模块和 1 个 pilot 脚本；有信号后最多增加单文件 router |
| 预设三分支、多头损失、KMeans prototypes | 不使用 MHA 或多头机制网络；Phase 3 只允许一个冻结的 64-centroid corpus summary 作为 Tier B 描述符，并必须单列成本与消融边界，不把它写成机制证据 |
| 固定 BGE、HF generator、样本量和通用数值阈值 | 均改为待确认或由 pilot 方差/power 决定，不静默选择 |
| 每阶段强制 stage JSON、设计文档、branch 和三次 commit | 直接复用现有 runner 的 metadata/results/summary；不新增阶段元数据文件，只在用户授权时提交 |
| 把 uncertainty 与 complementarity 混成第三类 | 分开检验：低置信度不等于证据互补，Hybrid gain 也不自动证明互补机制 |

旧计划中应继续保留的边界包括：固定实验环境、train/dev/test 隔离、在线特征禁止读取实际检索结果、连续 utility/margin/regret、gold-context sanity、paired inference、反事实分组以及 Teacher–Student 后置。

---

## 2. 当前本地基线与复用边界

### 2.1 已实现并可复用

| 能力 | 当前位置 | 本计划中的用途 |
|---|---|---|
| 固定 BM25/Dense runtime | `src/query_runtime_factory.py`、`src/query_plan.py` | 保持基线，不提前改成在线 router |
| 逐 query 执行、checkpoint、resume | `src/evaluation_runner.py` | 复用正式/试运行流程 |
| candidate 与 reranker cache | `src/evaluators/beir_suite.py` | 避免重复 retrieval/reranking |
| retrieval per-query metrics | `src/evaluators/beir_metrics.py` | 作为诊断层，不另写第二套 |
| weighted RRF、overlap、oracle 分析 | `scripts/analyze_beir_retriever_policy.py` | 复用计算逻辑；先恢复其已清理的 Top-50 source map/派生 rows，不先实现在线 Hybrid |
| Top-k cache replay 与完整性校验 | `scripts/recompute_beir_suite_topk.py` | 缓存深度足够时只做离线 replay |
| context、prompt、generation、timing/token 日志 | `src/pipeline.py` | 构造公平的 RAG utility |
| OpenAI/extractive generator | `src/generators/answer_generator.py` | extractive 只作 smoke；正式 generator 待确认 |
| SQLite BM25 analyzer 与 term DF/IDF | `src/retrievers/sqlite_bm25.py` | Phase 4 才按需暴露少量 lexical features |
| metadata/output writer | `src/persistence/` | 复用现有写入器，不建 routing persistence 或 Router 元数据层 |

本地 `tests/` 有 27 个 `test_*.py` 和 1 个 `conftest.py`，只是被 `.gitignore` 忽略；不得据此再造一套 config、resume、artifact 或 BEIR cache 测试。

### 2.2 HotpotQA 已有事实

- `data/beir/hotpotqa/manifest.json` 已登记 5,233,329 篇 corpus 文档和 97,852 条 query，并提供 train/dev/test qrels。
- `queries/queries.jsonl` 的 97,852 条 query 已包含 `metadata.answer` 与 `metadata.supporting_facts`；后者是 `(页面标题, 句子序号)`，不是 corpus doc ID。
- `artifacts/_registries/beir_offline_v1.json` 已登记 HotpotQA build、encoded corpus 与 SQLite BM25 artifact；不重建 corpus、不重新编码、不重建 BM25 index。
- `outputs/beir_hotpotqa_four_condition_full_v2/` 已有 BM25/Dense Top-50 first-stage candidate NPZ；现存 run JSONL 是四种 Top-5 final condition，不是 Top-50 result JSONL。
- 这些 candidate manifest 明确属于 **test split，共 7,405 query**。它们可复核历史结果，但必须封存，不能用于选择模型、阈值、特征或 RRF 权重。
- dev/train 若没有匹配 cache，只运行冻结协议所需的 query retrieval；这不是重建 artifact，也不能用 test cache 代替。
- 该 official test 的 retrieval outcomes 已在历史工作中被分析过，因此它不是“从未接触”的 retrieval test。确认性 final holdout 必须在生成 label 前从预先声明的、未用于方法选择的 eligible pool 中按 group 冻结；official test 只能作为额外的 historically exposed 结果，不能隐去这一边界。

### 2.3 已运行的 retrieval-only 证据

本地 Selected-7、26,428-query、18-unit、equal-50 离线分析已经得到：

| 指标 | nDCG@10 |
|---|---:|
| best fixed single | 0.434724 |
| fixed RRF $\alpha=0.5$ | 0.439180 |
| test-qrels post-hoc best fixed $\alpha$ | 0.449512 |
| per-query best single oracle | 0.505046 |
| per-query single-or-Hybrid oracle | 0.536715 |

Hybrid strict-gain query fraction 为 22.53%。这些结果只说明 retrieval 层存在描述性 headroom；它们使用 test qrels 和离线 rank-list fusion，没有 answer utility，也没有在线 Hybrid 成本，因此不能作为可部署 Router 或 RAG 改善结论。

`docs/assets/history_aware_retriever_policy/analysis/source_files.csv` 指向的 36 个派生 Top-50 JSONL 已被清理，RRF 脚本默认依赖的 `tmp/beir_retriever_dataset_interactions_v3/candidate_source_result_files.csv` 也不存在；验证表、派生分析和基础 candidate NPZ 仍在。需要新增派生分析时，先用现有 replay 路径恢复 Top-50 rows 与 source map，或显式传入恢复后的 source map；禁止因此重跑 retrieval/reranker。

### 2.4 真正缺口

1. Hotpot answer/evidence metadata 目前没有进入独立 RAG evaluator；
2. 仓库没有正式 Answer EM/F1、盲评 AC 与 answer-level utility 汇总；
3. 正式 generator、prompt、context budget 和 primary answer metric 尚未冻结；
4. dev/train 的 B/D action outcomes 需要按冻结协议生成；
5. 没有经验证的 pre-retrieval predictability 结果；
6. 本地 NQ BEIR query metadata 不含 answer；若纳入 NQ，exact source-ID mapping 仍是阻塞项；
7. 没有在线 Hybrid 或 router；在离线证据通过前也不应实现。

---

## 3. 研究问题与边界

固定 corpus $C$、BM25 $B$、Dense retriever $D$、generator $G$ 和 prompt/context protocol $P$。对 action $a$ 定义：

$$
U_a^{rag}(q)=\mathbb E_{s\sim\mathcal S}
M\!\left(G(q,\operatorname{Context}(R_a(q,C);P);s),y_q\right),
$$

其中 $M$ 固定为盲评 Answer Correctness，三档标签映射为 $0/0.5/1$；$s$ 表示 generation seed/随机性。当前每个 action 统一生成 3 次，并以重复均值估计 $U_a^{rag}$，同时报告 Monte Carlo 方差。不能对每个 action 只随机生成一次再取最大值，否则 oracle headroom 会产生 winner's-curse 偏差。

固定 AC judge 只读取 question、reference answers 与 predicted answer，不读取 action、retriever、retrieved context 或 retrieval metrics。normalized token F1 与 EM 平行计算并单列报告，但不进入 primary utility。AC 与 EM/F1 的分歧、重复稳定性和人工盲审校准必须单列报告。第一版目标是：

$$
\pi(q,x_C,x_R)\in\{N,B,D\},\qquad
\max_\pi\;\mathbb E[U_{\pi(q)}^{rag}(q)].
$$

必须依次回答：

1. **标签有效性**：固定 generator/evaluator 是否真的从正确证据中获益，生成噪声是否可接受？
2. **答案效用 headroom**：在固定 $(C,B,D,G,P)$ 下，N/B/D 的 query-level utility gap 是否稳定且有实际意义？
3. **可观测性/可预测性**：仅凭 retrieval 前可用的低成本信息，能否恢复一部分 headroom？
4. **机制监督增量**：在输入、容量和训练预算匹配时，lexical/semantic 辅助监督是否优于普通 utility regressor？
5. **当前范围边界**：本轮不研究 Hybrid 或其他 retriever，先证明 N/B/D 的 AC-primary pairwise Router 是否有效。

`Lexical Discriminativeness`、`Semantic Bridgeability` 和 `Complementarity` 均是 operational hypotheses，不是真实可直接观测的机制。给分支命名、加入 auxiliary loss 或观察到 ablation drop，都不能单独证明模型学到了机制。

### 3.1 信息角色

| 角色 | 可包含 | 禁止用途 |
|---|---|---|
| `ONLINE_PRE_RETRIEVAL` | raw query、固定 corpus DF/IDF、已部署 retriever 的固定配置、可选 query embedding、预计算 corpus summary | 任何当前 query 的结果、score、rank、overlap、gold answer/evidence |
| `OFFLINE_TRAIN_TARGET` | train 的 qrels、gold evidence、实际 Answer EM/F1/AC、action utility、机制 proxy | 直接进入 online feature 或 post-retrieval Teacher input |
| `POST_RETRIEVAL_DIAGNOSTIC` | retrieved docs、scores/ranks、margin、entropy、结果 overlap、teacher 表示 | 使用 qrels/gold/实际 EM/F1/AC/utility 作输入，或冒充低成本在线 Router 输入 |
| `EVALUATOR_ONLY` | sealed final-holdout labels/oracle、post-hoc fixed action | 调参、选特征、选阈值、选 alpha |

固定 corpus 上的 DF/IDF 或 centroid 属于 corpus-conditioned/transductive descriptor，只有部署时同一 corpus 已知才可使用，并必须计入构建与查询成本。

### 3.2 当前非目标

- history-aware routing；
- dynamic top-k、ANN 参数或动态 no-retrieval threshold/gating；固定 $N$ action 已纳入当前 action set；
- query rewriting、proposition decomposition；
- 多 Dense action、reranker action；
- dynamic Hybrid weight；
- 大型 MHA、多专家或通用 plugin/registry 架构。

---

## 4. 执行前必须冻结的协议

以下选择不得由 Codex 静默决定：

| 项目 | 最小要求 |
|---|---|
| Primary dataset | HotpotQA 可先进入；NQ 只有 exact answer-ID mapping 通过才加入 |
| Dense contract | checkpoint/revision、query/document encoder、pooling、normalization、prefix/input format |
| Generator contract | provider/model/revision、prompt、decoding、context budget、确定性或配对重复、复用边界、最大调用数/预算与停止规则；只有 provider 真正支持 seed 时才使用配对 seed |
| Retrieval contract | B/D 的固定配置、candidate/final $k$、相同 context builder、是否使用 reranker |
| Answer endpoint | HotpotQA 的 scalar primary utility 为三档盲评 AC；normalized token F1 与 EM 为 reference/diagnostic |
| AC judge contract | 固定 judge snapshot、盲评输入、三档 rubric、重复稳定性、人工校准样本、调用/预算与失败规则 |
| Split/group | 先声明唯一 eligible source pool，再固定 `query_id → pilot/train/dev/final_holdout`；四者互斥，同 information need/variants 同组，不能只检查 query ID；Hotpot official test 的历史 exposure 单列处理 |
| Practical effect | 在看 final holdout 前定义 minimum meaningful effect；不沿用通用 0.02/10%/200-query 阈值 |
| Feature cost tier | 是否允许为 Router 计算 dense query embedding，以及模型常驻/编码成本如何计入 |

当前冻结：优先复用已有 Dense artifact；HotpotQA 以 AC 为 primary、normalized token F1/EM 为 reference，action set 为 $\{N,B,D\}$；eligible pool 只使用 official train，并按 group 固定 pilot/train/dev/final holdout。若契约再次变更，必须新建 revision run 并保留历史结果。

---

## 5. 最小数据与产物契约

只建立一个扁平的 per-query action table；不建立 `StrategyId/Spec/Outcome/UtilityVector/OracleRecord` 等多层对象。

每行至少包含：

```text
dataset_id, split, query_id, group_id, question
reference_answers, supporting_fact_refs
gold_qrel_doc_ids, gold_context_type
action[N|B|D].retrieval_metrics / evidence_metrics
action[N|B|D].answer_em / answer_f1 / answer_ac
action[N|B|D].retrieval_ms / generation_ms / tokens
online_features
row_status / failure_reason
```

`supporting_fact_refs` 保留原始 `(title, sentence_index)`；`gold_qrel_doc_ids` 来自对应 split qrels。首轮 sanity 使用完整 qrel 页面的 `gold_page_context`。只有原始句子边界能够精确恢复并验证后，才允许另报 `gold_support_sentence_context`；不得把 `(title, sentence_index)` 静默当 doc ID。所有 label 与 online feature 虽可在同一离线表中保存，但训练代码必须通过显式 whitelist 构造 feature matrix；测试负责拒绝 gold、outcome 和 post-retrieval 字段。

首轮输出保持单一目录：

```text
outputs/router/<study_id>/
  config.yaml
  split.json
  runs/<run_id>/        # 直接使用现有 runner 的 metadata/results/summary
  action_table.jsonl
  summary.json
  predictions.jsonl      # 训练 Router 后才出现
```

`config.yaml` 与 `split.json` 在 Phase 0 后冻结；每次执行使用新的 `runs/<run_id>/`，不跨 Phase 覆盖已有 run。完整 answer/context 留在现有 result/checkpoint 中，action table 只用已经存在的 `dataset_id/split/query_id/action` 关联，不复制运行元数据，也不新增独立 generation cache。需要重试时直接复用现有 checkpoint/resume。只有性能或体积确有需要时，才把 feature matrix 另存 NPZ。

---

## 6. 必需执行阶段

### Phase 0：只读复用审计与协议冻结

**动作**

1. 通过现有 registry/metadata 验证目标 dataset/split 的 query、qrels、corpus、candidate cache、build、BM25 与 Dense；验证结果写入阶段 summary，不复制到 action table；
2. 列出每个所需 outcome 是“已缓存、可离线 replay、必须新运行”中的哪一种；
3. 在任何 pilot/action label 生成前写死 `split.json`，记录 eligible-set 原始分母、group 和 `query_id → pilot/train/dev/final_holdout`；四个 partition 互斥；
4. final holdout 不生成 retrieval 或 answer action outcomes；historically exposed test cache 只作既有 retrieval 证据，不参与方法选择。

**产物**：一个实验 config 和一个后续只读的 `split.json`。不新增 readiness 框架、Router 元数据文件、stage report 或 Git 分支。

**Gate**：任一模型、数据或 metric 契约未确认，或 split 存在交叉，则暂停对应实验；不得用猜测补齐。

**同步结果（2026-08-20）**：Phase 0 仍为 `GO`。HotpotQA train eligible query 为 85,000；冻结分区为 pilot 100、train 67,920、dev 8,490、final holdout 8,490，按 normalized answer + supporting-page titles 分组。改为 AC-primary 并加入 $N$ 不改变 corpus、split、BM25/BGE artifact 或检索配置，无需重建索引；final holdout outcomes 仍为 0。

### Phase 1：答案评估与 generator sanity pilot

**数据**

- 使用 Phase 0 从 HotpotQA official train 预先划出的约 100 条 pilot/calibration query；dev、final holdout 和现有 7,405 条 test cache 不参与 pilot。
- 若 NQ 能完成 exact mapping 与独立 split，再从其 train 划出约 100 条互斥 pilot/calibration query；不能 fuzzy text join，也不能从 qrels 文档猜答案。

**动作**

1. 用 evaluator-side adapter 联合读取 answer、原始 supporting-fact refs 与 split qrels，不重写核心 BEIR loader；
2. 实现并单测 normalized EM、token F1、alias max、必要的 no-answer 规则，以及盲评 AC 的三档 rubric、结构化输出解析和 failure handling；
3. 在完全相同 prompt/context builder 下比较 no-context、`gold_page_context`、BM25 context、Dense context，并同时报告 F1、EM、AC 及其分歧；sentence-level gold context 只有在句子边界验证通过后才作为附加条件；
4. 使用现有 artifact 对缺失的 pilot subset 做 B/D retrieval，不重建索引或 corpus embedding；
5. generation 与 AC 结果直接写入各自 run 的现有 result/checkpoint；model、prompt、context、decoding 和 judge rubric 只在 config 中记录，结果行仅在确有多次生成或 judge 重复时增加 `repeat_id`，不重复保存 config，也不增加新的 cache key 或元数据层。先在小子集重复生成/判分；若输出或 judge 不稳定，正式实验对每个 action 使用相同重复数，且只有 provider 支持时才使用配对 seed；
6. 对预注册的 50 条预测做人工盲审校准，覆盖 EM/F1/AC 一致与分歧情形；报告三档混淆、weighted agreement 和逐类失败，不用 AC 自证 AC。

**主要 Gate**

- 纳入实验的 query answer/evidence mapping 可证明且覆盖完整；同时报告 eligible-set 原始分母、映射失败数、排除数与逐类原因；
- gold-context 相对 no-context 的 primary AC paired effect 方向正确并达到预注册实际意义；F1 仅作参考；
- AC 重复稳定性与人工盲审校准可接受；EM/F1/AC 分歧可解释；
- metric、judge 与 generation failures 不形成 outcome-dependent 缺失。

若 pilot 不确定，按 power 需求扩样；若在有把握的样本上 gold context 仍无正效用，先修 generator/prompt/evaluator，不进入 Router。

**同步结果（2026-08-20）**：Phase 1 仍为 `GO`，无需新增调用。100 条 pilot 上 no-context AC 为 `0.4700`，gold-page-context AC 为 `0.9400`，paired gain `+0.4700`，95% CI `[0.3737, 0.5644]`；参考 F1 gain 为 `+0.3900`。50 条人工盲审 exact agreement 为 `0.86`，linear weighted kappa 为 `0.7546`，既有校准结论继续支持 AC 作为 primary utility。

### Phase 2：N/B/D action table 与 AC headroom

**动作**

1. 只为 train/dev 生成 action outcomes；final holdout 保持无 outcome，直到 Phase 5 的配置完全冻结；
2. 对 $N,B,D$ 使用相同 prompt、generator、重复和 evaluator；$N$ 使用空 context 且 retrieval latency 为 0，$B,D$ 使用相同 final $k$ 与 context budget；
3. answer 层以 AC 为 primary，F1/EM 为 reference；检索 action 继续记录 retrieval/evidence/latency/token，$N$ 的检索指标固定为空或 0；
4. 以连续 utility 和 margin 为主，硬 winner 只作 high-margin 诊断；
5. 当前不加入 Hybrid、RRF 或其他 Dense retriever；
6. 主估计使用完整或与 outcome 无关的随机样本。disagreement/high-margin oversampling 只能作探索，并保留无偏控制集与采样权重。

在 dev 上定义：

$$
a_{fixed}^{dev}=\arg\max_{a\in\{N,B,D\}}\frac{1}{|Q_{dev}|}\sum_q U_a^{rag}(q),
$$

$$
H_{NBD}^{rag}=\frac{1}{|Q_{dev}|}\sum_q\max(U_N^{rag}(q),U_B^{rag}(q),U_D^{rag}(q))
-\frac{1}{|Q_{dev}|}\sum_q U_{a_{fixed}^{dev}}^{rag}(q).
$$

**Gate**

- $H_{NBD}^{rag}$ 的 AC paired/group+repeat bootstrap 与 point estimate 支持预注册的实际意义；dev bootstrap 的每个重采样内都重新选择 best fixed；
- F1/EM 只报告 AC oracle 对应的参考变化；若方向明显冲突，审计分歧但不把 F1 变成第二个 Gate；
- 分别报告三组 pairwise non-tie/tie、三 action unique winner 和 tied-best，不能把 AC tie 强行变成可靠负例；
- headroom 不由极少异常 query、失败行或生成随机性主导；非确定性生成的 Monte Carlo 方差进入 CI/敏感性分析。

若答案层 headroom 不足，停止复杂 Router；Selected-7 retrieval oracle 不能替代这个 Gate。

**同步结果（2026-08-20）**：`GO`。复用 600 query × B/D × 3 repeats，只补齐 600 query × $N$ × 3 repeats；5,400 个 query/action/repeat 单元全部成功，final holdout outcomes 为 0。Dev 最佳固定 action 仍为 Dense，AC `0.7583`；N/B/D AC oracle 为 `0.8333`，headroom `+0.0750`，95% CI `[0.0489, 0.1050]`，三组 leave-one-repeat-out 全部通过。$N$ 的 dev 平均 AC 为 `0.4994`，不适合作为 fixed action，但有 10 条独占 AC winner；它把 B/D AC oracle 约 `0.8050` 再提高 `+0.0283`。AC oracle 对应参考 F1 相对固定 Dense 为 `+0.0615`。

### Phase 3：部署前信号的最小可预测性（先 F1 筛选）

修订后只训练轻量 Router，不训练 query/retriever encoder，不使用 MHA。R3AG 只提供“query 与 action 的兼容性共同决定选择”的问题表述；训练方式采用 LTRR 风格的 pairwise preference。

每个 query 构造三个固定方向的无序比较：$(N,B)$、$(N,D)$、$(B,D)$。为避免在尚未证明可学习性前支付大规模 AC judge 成本，本阶段先用三重复平均 F1 构造筛选标签：

$$
y_{ij}(q)=\mathbb 1[U_i^{F1}(q)>U_j^{F1}(q)],\qquad
w_{ij}(q)=|U_i^{F1}(q)-U_j^{F1}(q)|.
$$

严格 F1 tie 不伪装成可靠正负类：主 pairwise loss 跳过 tie，并按 $w_{ij}$ 降低近似平局样本的影响；同一 query 的三对比较始终按 group 进入同一 CV fold，不能把 pair rows 随机拆分。模型只读取 shared query 特征、query–corpus lexical/semantic descriptors 与固定 action-specific compatibility blocks；不读取当前 retrieval outcomes、qrels、gold evidence 或生成结果。主模型为浅层 XGBoost binary preference classifier（`binary:logistic`），正则化 logistic 为容量对照；这是 LTRR 风格的成对监督，不等同于直接使用 `rank:pairwise` 排序目标。

对每个 action 汇总 pairwise preference probability：

$$
S_i(q)=\sum_{j\ne i}P(i\succ j\mid q),\qquad
\pi(q)=\arg\max_{i\in\{N,B,D\}}S_i(q),
$$

平分时默认 Dense。Borda 概率求和允许三组 pair probability 存在不传递或循环偏好，因此只把它视为固定聚合规则，不把分数解释成一致的绝对 utility。Full Router 联合使用所有 compatibility 信号；若进入正式机制比较，正确消融方式是保持同一模型并移除一个 block，而不是部署多个独立 Router。修订后 Phase 3 因在 4,800 档提前停止，只完成 Full XGBoost、Full logistic 和 Tier A XGBoost 三个容量/成本对照，没有执行 lexical、dense-summary、raw-embedding 的完整 remove-one-block 消融，因此不能归因某个 feature block 的机制作用。

feature cost 分两档报告：

- **Tier A**：17 维 query–corpus lexical compatibility，包括词表覆盖/OOV、IDF/DF 分布、rare-term 比例和数字、年份、大写、引号词法锚点；已排除长度、词数、疑问词数量等通用 surface block。
- **Tier B / Full**：Tier A + 384 维冻结 query embedding + 13 维 query–corpus prototype 相似度摘要；prototype 是从固定 corpus embedding 抽样后得到的 64 个静态中心。即使最终选择 BM25，也必须计入 encoder latency、模型内存、prototype 构建和查询成本。

可加入一个只用于诊断的 post-retrieval privileged baseline，以量化“学生输入中缺少多少信息”，但它只能读取 retrieved docs/scores/ranks 等 post features；qrels、gold evidence、实际 EM/F1/AC 和 utility 始终只是 target/evaluator 字段。它不是部署基线。

原协议计划使用嵌套的 `1,200 / 2,400 / 4,800 / 9,600` query 学习曲线，1,200 只是起点，9,600 原为正式训练规模。实际完整执行至 4,800；9,600 只留下未完成的断点数据，按用户决定取消且不得进入训练或结论。已反复查看的旧 300 dev 降级为 analysis set；原计划从剩余 dev 冻结 4,000 条 fresh dev，但因提前停止从未打开。final holdout 继续封存。

**F1 筛选 Gate**：strongest pre-retrieval Router 在 fresh dev 上相对 fixed Dense 的 mean F1 gain 至少 `+0.01`，paired bootstrap 95% CI 下界大于 `0`，每个训练 seed 的 gain 均为正且中位数至少 `+0.01`。Phase 3 必须在 F1 结果汇报后停止；即使 Gate 通过，也只能建议后续 AC 复核，未经用户再次明确授权不得自动新增 AC judgment。本阶段 AC 调用数固定为 0，AC 仍是最终确认指标，F1 通过不能直接替代 AC 结论。

本次未执行 fresh dev，因此没有形成正式 Gate 判定；当前状态是依据训练池 group OOF 学习曲线作出的 `EARLY STOP`。这足以停止继续投入，但不能写成 fresh-dev Gate 已失败。

**执行依赖**：当前 Python 3.11 环境使用 XGBoost `3.0.5`，并以正则化 logistic 作为容量对照。该依赖不影响 Phase 0–2 的结论。

**历史 pilot（2026-08-20）**：旧双 action、F1 absolute-regression Phase 3 为 `STOP`；该结果只否定当时测试的 300-train 混合特征/回归协议。它不评价新的 AC-primary、N/B/D pairwise Router，也不能被新实验覆盖或改写。

**修订后 Phase 3 同步结果（2026-08-21）**：1,200、2,400、4,800 三档均完整完成 5-fold group OOF 与 5 个训练 seed。Full XGBoost 相对 fixed Dense 的中位 F1 gain 分别为 `-0.00696 / -0.00312 / -0.00281`，正提升 seed 数为 `0/5、0/5、1/5`；4,800 档最好单 seed 仅 `+0.00183`，低于预注册 `+0.01`。约 60% pair 为严格 F1 tie。9,600 未完成且已取消，fresh dev、model freeze、AC 与 final holdout 均未执行；Phase 4 不启动。

### Phase 4：机制 proxy 的增量价值

只在 Phase 3 通过后执行，采用相同输入、相近参数量和相同优化预算的对照：

1. AC pairwise Router only；
2. `+ lexical proxy loss`；
3. `+ semantic proxy loss`；
4. 两者组合。

首版用共享小 MLP 加可选 auxiliary heads；不使用 MHA，不强制独立“机制分支”。

训练期 proxy 可包括：

- lexical：gold-evidence lexical coverage、BM25 rank/utility、高 IDF exact-match；
- semantic：query-positive similarity、Dense rank/utility、低 lexical coverage 下的 Dense success。

它们只能来自 train，并明确标为 outcome-correlated proxy。Teacher 不能创造在线输入中原本不存在的信息。

**机制证据要求**

- 相对 input/capacity-matched baseline，primary $\Delta U^{rag}$ 有增量；regret 只作同一结果的解释；
- 预注册 lexical-anchor 与 low-overlap stress subset 的变化方向合理；
- 移除某 proxy 后的变化与其假设一致；
- 无增益的 head/feature 立即删除，不为架构完整性保留。

### Phase 5：冻结后的 test、反事实与泛化

模型、feature schema、tie margin、阈值和所有超参数在 dev 上冻结后，才首次为 `split.json` 中的 final holdout 生成 B/D answer/action outcomes 并评估一次；official dev/test 不自动替代该 holdout，若作为附加结果使用，必须单列数据来源、重复组处理与历史 exposure。

**必须报告**

- primary AC utility、参考 F1、两者的 paired CI 与 disagreement；
- `Regret_NBD/Recovery_NBD`；
- retrieval/evidence/answer 分层诊断；
- Router feature、retrieval、generation latency 与 token/cost；
- 多 seed 训练方差和 query/group sampling uncertainty；
- 失败样例与 action-selection 分布。

counterfactual query variants 第一轮只作 evaluation：同一 information need 保持同组，重新运行真实 B/D retrieval 与 generation，不继承 parent winner label。

若只在当前 corpus/retriever 内有效，结论只能写成 `environment-conditioned predictor`。只有 matched ablation、counterfactual 行为和至少一种预先定义的 OOD 测试共同支持时，才使用 `mechanism-aware` 或 `generalizable` 的强表述。

---

## 7. 条件扩展

### H：Hybrid / complementarity（当前不执行）

Phase 2 可计算固定 RRF，但只有同时满足下列条件才把 $H$ 加入 action set：

$$
g_H^{rag}(q)=U_H^{rag}(q)-\max(U_B^{rag}(q),U_D^{rag}(q)).
$$

晋级统计量是 dev 上的平均增益，而不是挑选单个正例：

$$
\bar g_H^{rag}=\frac{1}{|Q_{dev}|}\sum_{q\in Q_{dev}}g_H^{rag}(q).
$$

- dev 上 $\bar g_H^{rag}$ 的 group-paired CI 与 point estimate 支持预注册的实际意义；
- gain 不是由大量微小 tie 或少数异常样本造成；
- 双检索、fusion 与可能的额外 context 成本处在可接受 quality–cost Pareto 前沿；
- offline fusion 明确标注为 cost proxy，未实测前不声称在线节省。

之后才比较 B/D router 与 B/D/H router，并分别预测 Hybrid gain 与不确定性。低置信度选择 Hybrid 是 selective-fusion policy，不等同于证据 complementarity。

### T：Teacher–Student

只有 post-retrieval teacher 明显优于直接监督的 pre-retrieval Student，且验证集显示可蒸馏信息时才执行。Teacher 可读 retrieved docs/scores/ranks 等 post features，但 qrels、gold evidence、实际 Answer EM/F1/AC 和 $U_a^{rag}$ 只能是训练 target/evaluator 字段，不能成为 Teacher 输入。必须比较 direct soft-target Student 与 distilled Student；若 Student 输入不含可识别信号，distillation 不应继续。

### R：Runtime 与 dynamic alpha

- N/B/D runtime integration：离线 policy 冻结并通过 replay 后，才最小修改 `QueryPlan`/factory；不先建通用 runtime registry。
- Dynamic alpha、Hybrid、history-aware、dynamic $k$ 与 ANN tuning 均不进入当前实验。

---

## 8. 指标与统计规则

对 policy $\pi$ 与预先声明的 baseline $b$，唯一 primary 比较量为：

$$
\Delta U^{rag}(\pi,b)=U_\pi-U_b.
$$

其余量用于解释。对明确标注的 action set $\mathcal A\in\{BD,BDH\}$：

$$
U_\pi=\frac{1}{N}\sum_q U_{\pi(q)}^{rag}(q),
$$

$$
Regret_{\mathcal A}(\pi)=\frac{1}{N}\sum_q\left[\max_{a\in\mathcal A}U_a^{rag}(q)-U_{\pi(q)}^{rag}(q)\right].
$$

若 oracle headroom 大于稳定的数值容差，才解释：

$$
Recovery_{\mathcal A}(\pi)=
\frac{U_\pi-U_{best\ fixed}}
{U_{oracle}-U_{best\ fixed}}.
$$

规则：

- `best_fixed_selected_on_dev_then_final` 是部署基线；`posthoc_best_fixed_on_final` 只能作描述性上界。
- dev headroom 的每个 bootstrap resample 内重新计算 best fixed；final holdout 始终使用 dev 已冻结的 fixed action，不得重选。
- final-holdout oracle 只由 evaluator 计算和报告，不参与任何选择。
- 在固定 action table 上，utility、regret 与 recovery 是确定性相关量，不能作为三份独立支持证据；主结论只由预注册的 $\Delta U^{rag}$ 判定。
- paired bootstrap 按 information-need/group 重采样；多数据集用分层重采样并报告等权 macro。
- 非确定性 generation 的重复先在 query/action 内求均值；需要传播生成方差时使用 group + repeat 的两层分析，不能把同一 query 的重复当作独立 query。
- query sampling CI 与训练 seed 方差分别报告，不只报告最好 seed。
- 样本量由 pilot 的 paired variance、目标实际效应和 power 决定，不预设 5k/1k/1k 或每 action 200 条。
- secondary/subset 结果标明多重比较或探索性，不用事后阈值把 STOP 改成 GO。
- 若要定义 $U_{cost}=U_{rag}-\lambda Cost$，必须在看 final holdout 前用真实 profile 冻结 cost 定义与 $\lambda$；首轮也应完整记录 cost vector。

---

## 9. 最小代码改动图

### Phase 0–2 首轮允许的新增

1. `configs/router_v1.yaml`：协议确认后才创建，一个 config 足够；
2. `src/evaluators/rag_utility.py`：薄的 Hotpot/NQ eval adapter、Answer EM/F1/AC、utility/regret 汇总；AC 复用同一模块，不另建 judge 框架；
3. `scripts/run_router_pilot.py`：复用现有 cache、runner、pipeline 和 output writer，生成 action table；
4. 在现有 `tests/` 中增加少量高价值测试，最多新增 1–2 个测试文件。

### Phase 3–4 有信号后才允许

1. `scripts/run_router_experiment.py`：用 config 切换 baseline、feature set 与 ablation，训练和评估不拆成多脚本；
2. `src/router.py`：只有训练逻辑确需复用/上线时才新增；
3. `src/retrievers/sqlite_bm25.py`：仅在 lexical feature 实际进入实验时暴露稳定 analyzer/term-stat 接口，并做 token 等价回归。

### 当前禁止预建

- `src/routing/` 多级包及 records/config/dataset/oracle/statistics/persistence 子模块；
- 单独的 RRF runtime、strategy registry、`RoutedRAGPipeline`；
- 新增或调优第二套 KMeans/prototype artifact；已冻结的 Phase 3 64-centroid corpus summary 只保留为已运行 Tier B 输入，不继续扩建；
- `hf_causal_generator.py`（generator checkpoint 与契约未确认前）；
- 一概念一脚本的 build/train/evaluate/ood/stage-gate 工具链；
- `requirements/routing.txt`、`docs/research/` 或新的 tests 目录树。

若同一逻辑出现第二个真实消费者，再从现有脚本提取公共函数；不得为未来猜测建立抽象层。

---

## 10. 执行与复用纪律

每个阶段只需遵循：

1. 读取冻结的 config、`split.json` 与上一阶段 summary；
2. 通过现有 registry/metadata 验证源和 split，不复制其内容；
3. 标记 `reused / offline_replayed / newly_executed`；
4. 只实现当前 Gate 所需的最小改动；
5. 运行 focused tests，涉及公共路径时再运行完整本地回归；
6. 输出结果、CI、失败行、限制与 `GO / REVISE / STOP`；
7. 等待下一阶段授权。

禁止：

- 因派生 JSONL 被清理而重跑已有 retrieval/reranker；
- 用 test 选择 action、alpha、threshold、feature 或模型；
- 用实际检索 score/overlap 构造所谓 pre-retrieval feature；
- 把 offline RRF 称为当前在线 Hybrid；
- 把 retrieval-only gain 写成 answer gain；
- 让 synthetic query 继承 parent label；
- 隐藏 generator failure、选择性缺失、负结果或 seed 方差；
- 为未通过 Gate 的阶段留下 skeleton code。
- 为 Router 另建元数据记录，或重复保存现有 artifact metadata。

`.gitignore`、ignored tests/docs 是否版本化是独立的仓库治理决定，不是科学实验的 Phase 0 Gate；本计划不擅自修改。

---

## 11. 决策表与完成标准

| 问题 | 通过证据 | 不通过时 |
|---|---|---|
| evaluator/generator 有效吗 | gold-context 对 no-context 有可靠、实际的 AC gain，且 AC 通过人工校准 | 修 protocol；不训练 Router |
| N/B/D 值得路由吗 | AC oracle 对 dev-selected best fixed 有稳定 headroom，F1 作为参考不显示难以解释的反向变化 | 停止复杂 Router或审计指标分歧 |
| headroom 可预测吗 | 廉价 pre-retrieval baseline 提高 answer utility、降低 regret | 重新限定环境/RQ；不做机制网络 |
| mechanism proxy 有增量吗 | matched baseline 上有增量且 stress/ablation 方向一致 | 删除无效 head/feature，保留简单模型 |
| 当前 N/B/D 方法有效吗 | AC Router 稳定超过固定 Dense | 若不通过，停止扩展 action set |
| claim 可泛化吗 | counterfactual 与预定义 OOD 支持 | 降级为 environment-conditioned |

### 第一可交付里程碑

完成 Phase 0–2：在至少一个 answer-bearing dev benchmark 上，以复用 artifact 的方式建立可信的 N/B/D AC utility table，并明确给出“有 headroom / 无 headroom / 证据不足”。这比实现 Router 更优先。

### 最小可证伪研究完成

完成 Phase 0–3：在不访问 retrieval outcomes 的前提下，使用冻结模型和一次性 fresh dev 确认廉价部署前信息能否超过 dev-selected best fixed。结果为负也属于有效结论，并立即阻止多分支过度建设。本次在 4,800 档后提前停止，虽足以作出“不继续投入”的决策，但因未打开 fresh dev，不满足这里的正式完成定义。

### 完整机制主张完成

只有 Phase 4–5 的 primary $\Delta U^{rag}$ 通过 matched baseline 比较，并有与之相容的 regret 诊断、counterfactual、成本和至少一种 OOD 证据，才可声称 mechanism-aware routing。Hybrid、Teacher、runtime 和 dynamic alpha 均不是这一主张的默认前置条件。

---

## 12. 修订后的推进顺序

```text
协议与复用审计
      ↓
Answer evaluator + generator sanity
      ↓
N/B/D AC utility table + headroom
      ↓
Query-only / low-cost predictability baseline
      ↓
Matched mechanism-proxy ablation
      ↓
Frozen test + counterfactual + OOD
      ├── Hybrid（仅 answer+cost gate 通过）
      ├── Teacher–Student（仅存在可蒸馏增益）
      └── Runtime / dynamic alpha（最后）
```

本计划的第一个核心问题不是“如何实现三分支 Router”，而是：

> 在可信的最终答案效用下，BM25 与 Dense 是否存在足够、稳定的 query-level headroom；而部署前廉价信息是否真的能够预测这部分 headroom？
