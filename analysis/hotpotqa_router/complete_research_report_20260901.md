# HotpotQA BM25 与 Dense 检索器路由研究完整实验报告

- **报告版本日期：** 2026-09-01
- **研究证据时间范围：** 2026-08-16 至 2026-08-30；Selected-7 的源 Dense matrix 于 2026-08-16 开始形成，HotpotQA Router 主线实验为 2026-08-20 至 2026-08-30，其中 Phase 2.7–2.10 集中执行于 2026-08-26 至 2026-08-30
- **研究对象：** HotpotQA 上 BM25 与 BAAI/bge-small-en-v1.5 Dense 检索器之间的逐查询路由
- **最终部署状态：** 当前没有通过冻结门槛的可部署自适应路由器；当前本地 pipeline 运行时只能按配置实例化单一 BM25 或 Dense 检索器，本报告不对外部生产系统状态作推断
- **内部最终留出集状态：** 从 HotpotQA source train 内部切出的 8,490-query final holdout 未打开，读取结果行数为 0；它不是官方 HotpotQA test split

## 目录

1. 报告范围、证据层级与状态口径
2. 研究背景与贯穿全部阶段的核心问题
3. 术语、指标、动作、监督目标与统一实验契约
4. 前置检索层证据：Selected-7、候选池互补性与离线融合上界
5. Phase 0：数据、分组、动作与评估协议冻结
6. Phase 1：生成链路、上下文价值与 Answer Correctness 评审校准
7. Phase 2：固定动作、逐查询 oracle 与重复生成稳定性
8. Phase 2.5：Answer Correctness 跨重复稳定性与 normalized token F1 辅助一致性审计
9. Phase 2.6：4,800-query 严格预检索 Router
10. 辅助实验一：Contriever 是否可以解决检索器配对问题
11. 辅助实验二：Natural Questions 上的 Dense Passage Retrieval 表征诊断
12. Phase 2.7：9,600-query query-only 模型、目标、表示与规模审计
13. Phase 2.8：winner query 扩充、T0/T1/T2 训练视图与候选重训
14. Phase 2.8b：natural-2000 独立候选选择
15. Phase 2.9：207 个原始 query 与 query–collection 单特质审计
16. Phase 2.10：scope 对冻结 M3 的最小条件增量诊断
17. Phase 2.10 追加 natural-2000 诊断
18. 跨阶段研究链、已取得成果与被排除的解释
19. 当前瓶颈、运行时能力边界与下一项研究问题
20. 局限、外推范围、数据消费状态与复现产物索引

---

## 1. 报告范围、证据层级与状态口径

本报告将 HotpotQA BM25 与 Dense 检索器路由研究从前置检索层分析、Phase 0–2.6、Phase 2.7、Phase 2.8、Phase 2.8b、Phase 2.9、Phase 2.10 主实验及其追加 natural-2000 诊断串成一条完整研究链。报告逐阶段给出研究问题、实验设计、数据、特征、训练与验证协议、具体结果、冻结门槛、停止决策、局限和后续含义。

证据按以下四个层级解释，全文不混用这些层级：

1. **正式确认结果。** 实验协议、数据角色、候选、指标和门槛在结果可见之前被冻结；结果用于做预先规定的继续或停止决策。
2. **候选选择结果。** 数据可以用于选择后续诊断候选，但一经打开即被消费，不能再作为未见验证集。
3. **离线机制诊断。** 可以使用检索后信号、qrels 或 gold evidence 定位信息上界和失败机制，但不能报告为部署时可用的预检索路由能力。
4. **oracle 或反事实上界。** 使用当前查询真实结果事后挑选动作，只回答“如果知道正确动作，最多可以获得多少收益”，不回答“部署时能否预测正确动作”。

全研究的阶段决策索引如下；每一行的具体设计、数据和解释均在后文展开：

| 阶段 | 核心数据 | 主要问题 | 冻结结果/决策 | 部署含义 |
|---|---|---|---|---|
| Selected-7 | 7 families、18 units、26,428 queries | Retrieval pools 是否互补 | Oracle/hybrid headroom 存在 | 仅 retrieval/qrels 背景 |
| Phase 0 | HotpotQA 85,000 queries、83,167 groups | 能否冻结无泄漏协议 | Split/config frozen | 无模型结论 |
| Phase 1 | Pilot 100 | 生成与评审链是否可信 | `GO` | 只证明评估链可用 |
| Phase 2 initial | 300 train + 300 historical development | 两动作 headroom 和重复稳定性 | `REVISE` | 重复不足，禁止直接放行 |
| Phase 2 revisions | 同 600、三重复、二/三动作 | 稳定 headroom 是否存在 | `GO` | Oracle 可用性成立，未证明可预测 |
| Phase 2.5 | 已有 3,600 条 BM25/Dense outcome rows | Answer Correctness preference 能否跨重复稳定；F1 是否只保留为辅助诊断 | `GO_AC_LABEL_EXPANSION` | Train cross-repeat Answer Correctness selector 支持扩展 Answer Correctness labels；F1 角色仍为 auxiliary or diagnostic |
| Historical three-action learning curve | 1,200/2,400/4,800 | 增加 query-only data 是否解决 | 9,600 canceled under old protocol | 收缩到二动作 selective gate |
| Phase 2.6 | 4,800、28,800 rows | 严格 pre-retrieval Router | `STOP_BEFORE_AC_LABELS` | 无可部署候选 |
| Contriever | 1,200、10,800 rows | 更换 Dense pair 是否解决 | `ARCHITECTURE_DIAGNOSIS_REQUIRED` | Contriever 不推进 |
| Natural Questions DPR | 1,309 exclusive queries | 新 encoder 是否形成自然边界 | Geometry threshold 未通过 | 仅表征诊断 |
| Phase 2.7 | 9,600、57,600 rows | 目标/表示/模型/规模系统审计 | `STOP_QUERY_ONLY_V1` | Query-only V1 停止 |
| Phase 2.7 Stage 12 | 同 9,600 grouped OOF | 缺失信息位于哪里 | `GOLD_EVIDENCE_SIGNAL_SUFFICIENT` | Probe/gold 仅诊断；未蒸馏 |
| Phase 2.8 | 7,885 新 labels；T0/T1/T2 | 扩充 winners 是否改善内部学习 | `DO_NOT_OPEN_CONFIRMATION_POOL` | 内部信号有、safety gate fail |
| Phase 2.8b | Natural 2,000 | Winner-balanced gain 是否迁移 | `NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY` | Fixed Dense 保持；数据已消费 |
| Phase 2.9 | 6,000 discovery、207 traits、natural 2,000 | 是否有稳定单一 trait | `SINGLE_FEATURE_GATE_PASS` | 只允许最小增量诊断 |
| Phase 2.10 | T2 12,000 | Scope 是否给 M3 条件增量 | `STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL` | Post-selection 诊断未发现稳定条件增量 |
| Phase 2.10 appendix | Consumed natural 2,000 | Post-selection natural 描述 | `STOP_NO_POST_SELECTION_NATURAL_STABILITY` | 不改变主停止决定 |

当前本地全局注册表正式登记 Phase 0–2.8b。Phase 2.9 和 Phase 2.10 已有冻结配置、完成决策、验证文件与归档结论，因此可以称为“已执行并完成的诊断实验”；但它们尚未进入全局注册表，不能称为注册表层面的 `result_status: complete`。这一状态差异属于实验组织状态，不改变对应决策文件中的科学结论。

正文完整报告所有阶段级、候选级和关键 split 级结果，以及全部决策所依赖的统计量。逐查询、逐生成重复和逐 outer-fold 原始记录保存在文末列出的冻结产物中；这些原始记录构成本报告的证据包，而不是被重新复制成数万行正文。

## 2. 研究背景与贯穿全部阶段的核心问题

研究的出发点是：BM25 词法检索与 BGE Dense 语义检索在不同查询上具有互补性。即使某个固定检索器在总体平均上更强，另一个检索器仍可能在一部分查询上生成更好的最终答案。因此，全局问题不是“BM25 还是 Dense 哪一个平均更好”，而是：

> 能否在检索发生之前，仅根据部署时可获得的查询信息和静态语料统计，可靠判断当前查询应使用 BM25 还是 Dense，并在端到端答案效用上显著、稳定且安全地超过最佳固定检索器？

这一总问题被拆成六个连续子问题：

1. 两种检索动作在最终答案层面是否确实存在逐查询互补性，还是检索指标上的差异并不会传递到答案质量？
2. normalized token F1 是否足以作为大规模低成本筛选标签，并且与 Answer Correctness 语义评审保持一致？
3. 严格 query-only 与 corpus-static 特征能否预测 BM25 相对 Dense 的答案效用差？
4. 如果预测失败，失败来自模型结构、表示方式、训练目标、样本量、类别不平衡，还是来自检索前信息边界本身？
5. 如果定向增加 BM25 winner 和 Dense winner 可以提高内部可学习性，这种信号能否迁移到自然查询分布，并控制错误切换损失？
6. 如果某个可解释单特征与动作优势显著相关，它是否在控制既有模型后仍提供条件增量，并真正改变路由策略的答案效用？

整个研究的最终判据始终是端到端答案质量，而不是单独的检索命中、召回率、归一化折损累计增益或证据页召回率。检索指标用于解释机制、定向采样或构造离线诊断，不替代最终 Router 的答案效用验证。

## 3. 术语、指标、动作、监督目标与统一实验契约

### 3.1 名称、统计缩写、模型代号与数据视图全称

正文为了与冻结配置、结果表和机器决策文件逐字对应，会保留其中的英文名称与代号；本表先把所有反复出现的缩写和候选代号完整展开。后文出现代号时，含义始终以本表和对应阶段的精确定义为准。

| 缩写或代号 | 完整名称 | 本研究中的精确定义 |
|---|---|---|
| RAG | Retrieval-Augmented Generation，检索增强生成 | 先检索外部文档，再把选定上下文交给生成模型回答问题的完整系统 |
| QA | Question Answering，问题回答 | 本研究最终评价的是 HotpotQA 答案质量，而不是只评价检索排名 |
| BM25 | Best Matching 25 概率词法排序函数 | 本地实现为 `sqlite_bm25_v1`，使用冻结 analyzer、`k1=1.5`、`b=0.75` |
| Dense | Dense vector retrieval，稠密向量检索 | 主要指 `BAAI/bge-small-en-v1.5` 查询向量与冻结文档向量的归一化内积检索 |
| BAAI | Beijing Academy of Artificial Intelligence，北京智源人工智能研究院 | `BAAI/bge-small-en-v1.5` 模型仓库名称中的机构前缀 |
| BGE | BAAI General Embedding | 主要 Dense Retriever 使用的 embedding 模型家族；本研究冻结 small English version 1.5 |
| DPR | Dense Passage Retrieval，稠密段落检索 | Natural Questions 辅助实验所用的 query encoder 表征 |
| NQ | Natural Questions | 用于辅助表征诊断的数据集，不是 HotpotQA 答案 Router 的独立确认集 |
| BEIR | Benchmarking Information Retrieval，信息检索基准套件 | Selected-7 前置检索层分析的数据来源框架 |
| qrels | Query relevance judgments，查询相关性判断 | 标明 query 与文档是否相关的监督数据；只允许用于检索评价、采样或 privileged diagnostic，不允许作为在线预检索输入 |
| F1 | Precision 与 recall 的调和平均 | 先规范化预测答案与参考答案，再计算 token-level precision、recall 和调和平均；是后期大规模实验的主要答案效用 |
| EM | Exact Match，完全匹配 | 规范化预测答案是否与任一规范化参考答案完全一致；正文通常写作 normalized exact match |
| AC | Answer Correctness，答案正确性 | 冻结评审模型给出的语义正确性评分，取值为 0、0.5 或 1 |
| nDCG@10 | Normalized Discounted Cumulative Gain at rank 10，前十位归一化折损累计增益 | Selected-7 中使用 qrels 计算的检索排序指标，不等于生成答案效用 |
| Recall@K | Recall at rank K，前 K 位召回率 | 前 K 个检索结果覆盖全部相关文档的比例；`K` 表示截断深度 |
| MRR | Mean Reciprocal Rank，平均倒数排名 | 相关文档第一次出现位置的倒数再对 query 求平均；Stage 12 gold block 使用相关变体 |
| RRF | Reciprocal Rank Fusion，倒数排名融合 | Selected-7 离线混合实验用的无训练排名融合；冻结常数为 `rrf_k=60` |
| OOF | Out-of-fold，折外预测 | 每条 query 只由没有见过其所属 outer validation fold 的模型产生预测 |
| CI | Confidence Interval，置信区间 | 正式策略增益通常按 information-need group 做 10,000 次配对 bootstrap 得到 95% 区间 |
| ROC-AUC 或 AUC | Area Under the Receiver Operating Characteristic Curve，接收者操作特征曲线下面积 | 只在非 tie 的 BM25 winner 与 Dense winner 上衡量连续分数的排序区分能力；它不是策略效用 |
| PCA、PCA16、PCA32、PCA64 | Principal Component Analysis，主成分分析；分别保留 16、32、64 个主成分 | 必须只在 outer-train fold 内拟合；PCA32 是后期 M3 的冻结 embedding 压缩方式 |
| XGBoost | Extreme Gradient Boosting，极端梯度提升树 | 用于 structured、raw-full、post-retrieval probe 或 gold diagnostic 的树模型家族 |
| Ridge | L2-regularized linear regression，二范数正则化线性回归 | 后期 M3 的冻结回归器；主结果使用 `alpha=10` |
| Elastic Net | L1 与 L2 联合正则化线性回归 | Phase 2.7 的 embedding-only 候选之一 |
| L1、L2 | 一范数正则化、二范数正则化 | L1 倾向产生稀疏系数；L2 对全部系数作平方惩罚 |
| PMI | Pointwise Mutual Information，逐点互信息 | Phase 2.9 原始单特质家族中的词项关联统计 |
| UMAP | Uniform Manifold Approximation and Projection，统一流形逼近与投影 | 辅助表征几何可视化或诊断方法，不参与正式策略 gate |
| t-SNE | t-distributed Stochastic Neighbor Embedding，t 分布随机邻域嵌入 | 辅助表征几何可视化方法，不参与正式策略 gate |
| SHA-256 | Secure Hash Algorithm 256-bit，256 位安全散列算法 | 用于冻结配置、输入、预测或产物的内容身份校验 |
| NPZ | NumPy compressed archive，NumPy 压缩数组归档 | 保存冻结矩阵或数组的本地文件格式 |
| D | Dimensions，维数 | 例如 414D 表示 414 维输入，PCA32 表示保留 32 个主成分 |
| M0 | Structured-only XGBoost model | 只接收 30 维 structured block 的极端梯度提升树候选 |
| M1a、M1b | Embedding-only Ridge、Embedding-only Elastic Net | 只接收 raw 384 维 BGE query embedding 的两个线性候选 |
| M2、M2R | Raw-full XGBoost squared-error、Raw-full XGBoost pseudo-Huber | 把 30 维 structured block 与 raw 384 维 embedding 直接拼成 414 维；M2R 是 pseudo-Huber 目标变体 |
| M3 | Principal-component embedding plus structured model | 把 fold-local PCA embedding 与 30 维 structured block 拼接；后期冻结版本专指 PCA32 plus structured Ridge |
| M4 | Inner-out-of-fold late-fusion model | 分别训练 structured 与 embedding 父模型，再只用 inner-out-of-fold 父预测训练 Ridge 元模型 |
| M4-PCA | PCA32 inner-out-of-fold late-fusion model | Phase 2.8 中把 embedding-related branch 改成 fold-local PCA32 plus structured Ridge 的 late-fusion 变体 |
| M5 | Dual-utility Huber model | 分别预测 BM25 与 Dense 的绝对效用，再取差并加入 gap consistency loss 的双头模型 |
| S0 | Scope-only Ridge model | Phase 2.10 只使用 `c_scope_log_at_least_2_docs` 单一特征的机制对照 |
| M3S | M3 plus scope model | 在冻结 M3 的 30 维 structured block 中只增加唯一通过 Phase 2.9 的 scope 特征，其他训练设计保持不变 |
| P0、P1 | Qrels-free post-retrieval probe Ridge、Qrels-free post-retrieval probe XGBoost | 使用两路 Retriever 实际分数、排名形状、重叠和 context statistics 的检索后诊断模型；不使用 qrels，但必须先运行两路检索 |
| G0、G1、G2 | Gold Ridge、Gold XGBoost、Probe-plus-gold XGBoost | 使用 gold evidence 特征或 probe 与 gold 联合特征的 privileged diagnostic；永远不是可部署输入 |
| T0 | Tie-capped original training view | 1,196 个 BM25 winners、1,564 个 Dense winners、6,000 个 ties，共 8,760 queries |
| T1 | Winner-2,000 training view | 2,000 个 BM25 winners、2,000 个 Dense winners、6,000 个 ties，共 10,000 queries |
| T2 | Winner-3,000 training view | 3,000 个 BM25 winners、3,000 个 Dense winners、6,000 个 ties，共 12,000 queries |
| B/N/H | Beneficial, neutral, harmful switches | 相对默认 Dense，切到 BM25 后分别带来正效用、零效用和负效用的查询计数 |
| V1 | Version 1，第一版 | 指 Phase 2.7 所冻结的严格 query-only/corpus-static 特征与候选空间，不代表任何已部署版本 |

机器决策字符串的完整操作含义如下。这里的“继续”都只指进入协议明确指定的下一阶段，不代表模型已经可部署：

| 机器决策字符串 | 完整操作含义 |
|---|---|
| `GO` | 当前阶段的冻结检查通过，允许执行协议中已经定义的下一阶段；不等于 Router 通过最终部署门槛 |
| `REVISE` | 当前协议有明确失败项，必须修订实验设计并重新验证，不能按原设计直接继续 |
| `GO_AC_LABEL_EXPANSION` | Primary train gate 上，使用两个 repeats 的 Answer Correctness preference 选择动作并在第三个 repeat 评价时仍稳定提高 Answer Correctness，因此允许扩大 Answer Correctness 标签；F1 只保留为 auxiliary or diagnostic，不等于被晋升为 primary training label |
| `STOP_BEFORE_AC_LABELS` | F1 层 Router gate 未通过，因此在购买或生成更多 Answer Correctness 标签之前停止 |
| `ARCHITECTURE_DIAGNOSIS_REQUIRED` | 更换 Retriever 配对没有解决可预测性问题，后续应定位表示、信息边界或模型结构，而不是继续扩大当前 Contriever 分支 |
| `STOP_QUERY_ONLY_V1` | 第一版严格 query-only/corpus-static 特征、目标和候选空间没有达到冻结实用门槛，该研究分支停止 |
| `GOLD_EVIDENCE_SIGNAL_SUFFICIENT` | 检索行为与 gold evidence 诊断能解释显著动作效用差；只支持信息上界判断，不表示 gold 特征可部署或已经完成 Teacher-to-Student distillation |
| `DO_NOT_OPEN_CONFIRMATION_POOL` | Phase 2.8 的九个内部候选没有同时通过全部 gate，不得按该协议打开预留 confirmation pool |
| `NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY` | Natural-2,000 上没有合格候选；只能保留预先规定的前两个候选做有限诊断，不能晋级正式确认 |
| `SINGLE_FEATURE_GATE_PASS` | 207 个单特质中仅一个通过完整 discovery、来源稳定性、多重检验与 natural replication gate；只允许对它做唯一一次最小条件增量实验 |
| `STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL` | Scope 加入冻结 M3 后没有显著、安全且实用的配对策略增量，主 Phase 2.10 在内部 gate 停止 |
| `STOP_NO_POST_SELECTION_NATURAL_STABILITY` | 经明确授权的 consumed natural post-selection appendix 仍没有稳定正收益，不得据此恢复或继续该候选 |

### 3.2 动作与固定基线

- **BM25 动作：** 使用版本化的 `sqlite_bm25_v1` 词法检索器。
- **Dense 动作：** 使用 `BAAI/bge-small-en-v1.5` 查询编码器和 Dense 文档向量进行内积检索。
- **No-context 动作：** 不提供检索上下文，只在早期 sanity check 和三动作分析中使用，不属于后续 BM25/Dense 二动作 Router 的部署动作。
- **固定 BM25：** 所有查询都执行 BM25。
- **固定 Dense：** 所有查询都执行 Dense；在主要自然分布实验中通常是最佳固定动作，也是默认不切换动作。
- **逐查询 oracle：** 对每个查询事后选择真实答案 F1 更高的动作。它是不可部署的上界。

### 3.3 答案指标

- **Normalized token F1：** 对预测答案和参考答案进行规范化分词后计算 token-level precision、recall 与调和平均。本研究的大规模 Router 筛选和监督目标使用该指标。
- **Normalized exact match：** 规范化后预测答案是否与任一参考答案完全一致。
- **Answer Correctness：** 由冻结评审模型对答案语义正确性给出的评分。Phase 2.5 的 primary gate 用它验证动作 preference 的跨重复稳定性，并只把 F1 保留为辅助诊断；后续 Phase 3 另行采用 F1 作为低成本 screen，且只有该 F1 policy gate 通过时才允许扩展新的 Answer Correctness 评价。

### 3.4 监督目标与策略

对于查询 (q)，先对每个动作的三次生成答案 F1 取平均，再定义：

$$
g(q)=\overline{F1}_{BM25}(q)-\overline{F1}_{Dense}(q).
$$

- (g(q)>0)：BM25 在该查询上具有正的真实答案效用优势。
- (g(q)<0)：Dense 在该查询上具有正的真实答案效用优势。
- (g(q)=0)：两种动作在三重复平均 F1 上精确相同。

冻结的二动作策略是：模型输出 calibrated predicted gap；当预测值严格大于 0 时切换到 BM25，否则保留 Dense。等于 0 时不切换。除明确另立协议外，不在验证数据上扫描阈值。

### 3.5 Retrieval、context 与 generation 公共配置

- BM25 backend 为 `sqlite_bm25_v1`，使用 Lucene BM25 公式，`k1=1.5`、`b=0.75`、`english_regex_casefold_v1` analyzer。
- Dense model 为 `BAAI/bge-small-en-v1.5`，冻结 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`；文档编码输入为 `title + newline + text`，向量归一化后用 inner product。
- Dense index 为 `streaming_flat_ip` exact shard-wise scan，不是 approximate nearest neighbor index。
- First-stage retrieval 保留 Top 50 候选。
- 最终生成上下文使用排序最高的 Top 5 文档。
- 上下文总上限为 1,800 tokens。
- 上下文按 retrieval rank 排列，并按 document identifier 去重。
- 不使用 reranker。
- 生成模型为 `gpt-4.1-mini-2025-04-14`。
- Temperature 为 0。
- 最大生成长度为 64 output tokens。
- 每个 query/action 独立生成 3 次。
- Router 监督标签使用三次 normalized token F1 的动作内均值差。
- Answer Correctness judge 为 `gpt-5.4-mini-2026-03-17`，评分映射为 incorrect=`0`、partially correct=`0.5`、correct=`1`；judge 看不到 action、Retriever、retrieved context 或 retrieval metrics。

### 3.6 数据分组和泄漏边界

HotpotQA train 中的查询按 `normalized_answer_plus_supporting_page_titles` 分组。具有相同规范化答案和 supporting page titles 的查询不得跨训练、验证或留出分区，以防止同一信息需求的近重复版本泄漏。

严格预检索 Router 允许使用：

- 当前 query 的文本变换和词法统计；
- 基于冻结 analyzer、document frequency 和 inverse document frequency 的 query–collection 静态兼容性统计；
- 冻结的 query embedding；
- 从冻结语料原型得到的 corpus-static summaries。

严格预检索 Router 禁止使用：

- 当前查询实际返回的文档、document identifiers、scores、ranks 或列表重叠；
- qrels、gold supporting facts、gold document identifiers 或 evidence coverage；
- 生成答案、normalized token F1、exact match 或 Answer Correctness；
- oracle 动作、acquisition stratum 或任何从当前查询结果派生的标签。

检索后 probe 可以在单独的离线诊断阶段使用实际检索列表和分数，但必须明确标记为 post-retrieval；gold evidence 特征只能用于 privileged diagnostic。二者都不能被写成已经实现的预检索部署输入。

### 3.7 统一决策指标

- **Router mean F1：** Router 对每个查询所选动作的真实平均答案 F1。
- **Gain over best fixed：** Router mean F1 减去同一评估集上最佳固定动作的 mean F1。
- **95% confidence interval：** 按 `group_id` 做配对 bootstrap 得到的增益区间；主要实验使用 10,000 次 resamples。
- **Switch coverage：** 从默认 Dense 切换到 BM25 的查询比例。
- **Switch precision：** 被切换查询中真实 BM25 优于 Dense 的比例。
- **Beneficial gain mass：** 正确切换带来的总正效用质量。
- **Harmful loss mass：** 错误切换带来的总负效用质量绝对值。
- **Harmful-to-beneficial mass ratio：** harmful loss mass 除以 beneficial gain mass；越低越安全。
- **Non-tie area under the receiver operating characteristic curve：** 只在 BM25 winner 与 Dense winner 上衡量连续预测区分动作的排序能力。
- **Spearman rank correlation：** 连续预测分数与真实 (g(q)) 的秩相关。
- **Calibration slope：** 预测 gap 与真实 gap 的校准斜率。
- **Top-decile realized gap：** 预测最偏向 BM25 的最高十分位查询的真实平均 gap。

### 3.8 正式安全与实用门槛

从 Phase 2.7 起，候选通常需要同时满足：

1. 平均或 consensus gain 至少为 `+0.01 F1`；
2. 按 group 配对 bootstrap 的 95% confidence interval 下界严格大于 0；
3. 三个冻结 split seed 的 gain 全部为正；
4. harmful-to-beneficial mass ratio 不高于 `0.5`；
5. switch coverage 位于 `1%` 至 `50%`；
6. calibration slope 位于 `0.5` 至 `1.5`；
7. 预测最高十分位的真实 gap 大于 0。

某个候选只满足其中一部分，不等于通过部署 gate。统计显著的单特征关联、较高的 oracle headroom、内部 winner-balanced OOF gain 或检索后 probe gain，都不能替代完整 gate。

## 4. 前置检索层证据：Selected-7、候选池互补性与离线融合上界

### 4.1 研究问题

在进入昂贵的 HotpotQA 生成答案实验之前，前置检索分析先回答三个问题：

1. BM25 与 Dense 在多个数据集上的平均检索质量是否不同？
2. 两个检索器的 Top 50 候选池是否包含互补的相关文档？
3. 如果允许事后使用 qrels 选择单一检索器或加权 Reciprocal Rank Fusion，理论上存在多大检索层 headroom？

该分析只涉及 first-stage retrieval 结果的离线重放。它没有重新运行 retrieval 或模型推理，也没有运行 HotpotQA generation。它使用归一化折损累计增益 `nDCG@10` 和 qrels，因此只能提供检索层背景与 oracle 上界，不能直接证明答案 F1 会提高。

### 4.2 数据与设计

分析覆盖七个 BEIR family、18 个数据单元、26,428 条测试查询和 36 个既有结果文件：

| Dataset family | 数据单元 | Corpus documents | Test queries | 平均 whitespace query terms |
|---|---:|---:|---:|---:|
| ArguAna | 1 | 8,674 | 1,406 | 193.5533 |
| FiQA-2018 | 1 | 57,600 | 648 | 10.9398 |
| HotpotQA | 1 | 5,233,329 | 7,405 | 15.7214 |
| NFCorpus | 1 | 3,633 | 323 | 3.2941 |
| Natural Questions | 1 | 2,681,468 | 3,452 | 9.1585 |
| Touché-2020 | 1 | 382,545 | 49 | 6.5510 |
| CQADupStack | 12 个 forum，描述统计按 forum 等权 | 457,199 | 13,145 | 8.6443 |

补充 profile：ArguAna median terms/mean characters/question-mark fraction/mean positive qrels/multi-positive fraction=`174/1192.720/0.163585/1.000/0`；FiQA=`10/62.704/0.736111/2.632716/0.660494`；HotpotQA=`15/92.171/0.971236/2.000/1.000`；NFCorpus=`2/21.765/0.130031/38.185759/0.928793`；Natural Questions=`9/48.179/0/1.216976/0.192932`；Touché=`6/43.429/1.000/19.020408/1.000`；CQADupStack forum-equal profile=`8.25/51.062/0.511460/1.820741/0.240567`。

离线加权 Reciprocal Rank Fusion 使用 `rrf_k=60`。BM25 权重 `alpha` 从 0 到 1，以 0.05 为步长，共 21 个值；`alpha=0` 表示 Dense-only，`alpha=1` 表示 BM25-only。相同 fused score 时依次按最佳来源排名、来源排名和、document identifier 打破平局。

定义两个候选预算协议：

- **equal50：** 单一检索器使用 Top 50；混合检索使用 BM25 Top 25 加 Dense Top 25，融合输出不超过 50。虽然最终候选数相近，混合方案仍需要两次 Retriever 调用。
- **full100：** BM25 Top 50 与 Dense Top 50 取 union，再融合输出 Top 50；上游最多评分 100 个候选，因此与单路 Top 50 不是等预算比较。

### 4.3 固定检索器与单路 oracle 结果

| Dataset | BM25 nDCG@10 | Dense nDCG@10 | Dense−BM25 | Dense query-win fraction | BM25 query-win fraction | Tie fraction | 最佳固定动作 | 单路 query oracle nDCG@10 | Oracle gain |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ArguAna | 0.441039 | 0.595715 | +0.154676 | 0.512091 | 0.206259 | 0.281650 | Dense | 0.666817 | +0.071103 |
| FiQA-2018 | 0.223656 | 0.384833 | +0.161176 | 0.484568 | 0.128086 | 0.387346 | Dense | 0.420703 | +0.035870 |
| HotpotQA | 0.565940 | 0.694001 | +0.128062 | 0.486968 | 0.187171 | 0.325861 | Dense | 0.745992 | +0.051991 |
| NFCorpus | 0.312194 | 0.337535 | +0.025342 | 0.377709 | 0.291022 | 0.331269 | Dense | 0.386249 | +0.048714 |
| Natural Questions | 0.263108 | 0.435520 | +0.172411 | 0.495365 | 0.168888 | 0.335747 | Dense | 0.504039 | +0.068520 |
| Touché-2020 | 0.346551 | 0.223442 | -0.123108 | 0.224490 | 0.693878 | 0.081633 | BM25 | 0.373219 | +0.026669 |
| CQADupStack | 0.302907 | 0.372019 | +0.069112 | 0.297604 | 0.162419 | 0.539977 | Dense | 0.438305 | +0.066285 |

该表说明：Dense 在六个 family 上是最佳固定动作，BM25 只在 Touché-2020 上成为最佳固定动作；但每个 family 都存在非零单路 oracle headroom。检索器互补性不是 HotpotQA 独有现象。

Seven-family equal macro：BM25 `0.350771`、Dense `0.434724`、oracle `0.505046`、oracle gain over family-wise best fixed `0.052736`；Dense/BM25/tie query fractions 为 `0.411256/0.262532/0.326212`。

### 4.4 Top-50 候选池互补性

| Dataset | BM25/Dense candidate Jaccard | BM25 recall@50 | Dense recall@50 | Union recall up to 100 | BM25-only relevant fraction | Dense-only relevant fraction | Union headroom over better fixed pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | 0.238168 | 0.909673 | 0.972262 | 0.985064 | 0.012802 | 0.075391 | 0.012802 |
| FiQA-2018 | 0.114093 | 0.452821 | 0.607665 | 0.670822 | 0.063157 | 0.218001 | 0.063157 |
| HotpotQA | 0.078015 | 0.718771 | 0.813437 | 0.860095 | 0.046658 | 0.141323 | 0.046658 |
| NFCorpus | 0.143992 | 0.211568 | 0.259132 | 0.299999 | 0.040868 | 0.088431 | 0.040868 |
| Natural Questions | 0.141541 | 0.637963 | 0.824860 | 0.878742 | 0.053882 | 0.240778 | 0.053882 |
| Touché-2020 | 0.231039 | 0.454204 | 0.350239 | 0.531097 | 0.180858 | 0.076893 | 0.076893 |
| CQADupStack | 0.137709 | 0.514929 | 0.656459 | 0.706394 | 0.049934 | 0.191465 | 0.049934 |

HotpotQA 的 candidate Jaccard 只有 `0.078015`，union recall 比较强的 Dense pool 高 `0.046658`。这证明候选池内容互补，但不证明查询级 Router 能在检索前识别应该使用哪一池，也不证明将两池完整运行后的成本可接受。

Relevant-document decomposition 的 shared/BM25-only/Dense-only/neither fractions 分别为：ArguAna `0.896871/0.012802/0.075391/0.014936`；FiQA `0.389664/0.063157/0.218001/0.329178`；HotpotQA `0.672113/0.046658/0.141323/0.139905`；NFCorpus `0.170701/0.040868/0.088431/0.700001`；Natural Questions `0.584082/0.053882/0.240778/0.121258`；Touché `0.273346/0.180858/0.076893/0.468903`；CQADupStack `0.464994/0.049934/0.191465/0.293606`。Seven-family equal macro candidate Jaccard/BM25 recall/Dense recall/union recall/union headroom=`0.154937/0.557133/0.640579/0.704602/0.049171`。

### 4.5 离线混合与 query oracle

在 equal50 协议下，Selected-7 汇总的最佳单路 query oracle nDCG@10 为 `0.505046`，最佳 hybrid query oracle 为 `0.524089`，允许在 single 与 hybrid 之间事后选择时的 action-set oracle 为 `0.536715`；有 `22.5333%` 查询的 hybrid 严格优于最佳 single。纯 best-hybrid minus best-single oracle 差为 `+0.019043`；冻结汇总中的 `+0.031668` 则是把 hybrid 加入动作集合后，`single-or-hybrid oracle − best-single oracle` 的增量，不能标成纯 hybrid oracle 增益。

在 full100 协议下，Selected-7 的最佳 hybrid query oracle 为 `0.533562`，single-or-hybrid action-set oracle 为 `0.543295`；hybrid 严格获益查询占 `27.2666%`。纯 best-hybrid minus best-single 差为 `+0.028516`；冻结汇总中的 `+0.038249` 是 single-or-hybrid action-set oracle 相对 best-single oracle 的增量。

Equal50 cost-aware oracle action fractions 为 BM25 `0.166001`、Dense `0.306240`、hybrid `0.225333`、single tie `0.302426`；full100 对应 `0.146098/0.289913/0.272666/0.291324`。这些是 qrels-aware oracle allocations，不是可运行 cost-aware policy。

HotpotQA 自身在 equal50 下：best-single oracle `0.745992`、best-hybrid oracle `0.763532`、single-or-hybrid action-set oracle `0.773151`、hybrid 严格获益比例 `16.3268%`；纯 best-hybrid minus best-single 为 `+0.017540`，single-or-hybrid action-set 增量为 `+0.027158`。在 full100 下：best-hybrid oracle `0.771485`、single-or-hybrid action-set oracle `0.779289`、hybrid 严格获益比例 `19.4868%`；纯 best-hybrid minus best-single 为 `+0.025493`，action-set 增量为 `+0.033297`。

### 4.6 结论与对后续实验的影响

前置检索分析支持“不同查询需要不同检索动作”这一研究动机，也显示两路候选池有互补相关文档。但最佳 alpha 和 query oracle 都使用了 test qrels；weighted fusion 还需要两路检索。它们是描述性上界，不是已训练策略。因此后续 HotpotQA 主线必须转向端到端答案效用，并严格区分：候选池互补、答案级 oracle headroom、可预测 Router gain 和可部署成本是四个不同问题。

## 5. Phase 0：数据、分组、动作与评估协议冻结

### 5.1 研究问题

Phase 0 不尝试证明 Router 有效，而是冻结后续所有阶段共同依赖的实验契约：数据来自哪里、如何分组、哪些分区可以何时打开、动作如何运行、答案如何生成、什么是最终指标、什么信息允许作为部署输入。

### 5.2 数据划分

- Dataset：HotpotQA；eligible source split 为原始 train。
- Raw/eligible queries：85,000/85,000；excluded 0。
- Group key：`normalized_answer_plus_supporting_page_titles`。
- Information-need groups：83,167；最大 group size 7；split seed `20260820`。

| Frozen partition | Queries | Groups | 角色和状态 |
|---|---:|---:|---|
| Pilot | 100 | 98 | Generator、judge 和 context sanity check；已消费 |
| Train | 67,920 | 66,494 | 训练、内部 grouped OOF 与 acquisition |
| Internal development partition | 8,490 | 8,282 | Phase 2/2.5 消费 300-query historical-development subset；后续计划的 fresh-development subset 为其中 4,000 条且从未打开；其余 query 仍须按 group exposure 审计 |
| Internal final holdout | 8,490 | 8,293 | 从 source train 内部切出的 10% 留出；始终封存，outcome rows read 0；不是官方 HotpotQA split |

Official HotpotQA development split 不参与训练；official test 只保留为历史 retrieval-only evidence。报告中的 8,490-query development 与 8,490-query final holdout 都来自 HotpotQA source train 的内部 group-aware split，不是官方 development/test。`historical development 300` 指早期 Phase 2 已消费的 300-query subset；计划中的 `fresh development` 是同一 8,490-query internal development partition 内预定的 4,000-query subset，后期 Router gate 从未打开它，不能把 fresh-development 规模写成 8,490。

分组而不是单条 query 随机切分的原因是：具有相同答案和 supporting pages 的问题可能只是表述变化。若它们跨 fold 或跨 train/dev，会使模型看见同一信息需求的近重复形式。

### 5.3 冻结动作与运行配置

早期动作包括 no-context、BM25 和 Dense；正式 Router 后来聚焦 BM25 与 Dense 二动作。BM25 使用 `sqlite_bm25_v1`；Dense 使用 BGE-small-en-v1.5 和 `streaming_flat_ip`；retrieval candidate depth 为 50，generation context depth 为 5，不使用 reranker。生成配置与第 3.5 节一致。

关键设计决策是先按 information-need group 分割，再抽 query；Pilot、train、internal development、internal final 之间 group 隔离。Internal final holdout 在方法选择阶段禁止产生 outcomes；当前 query 的 retrieval result 只能用于 label 或单独标记的诊断，不能进入严格 pre-retrieval model。

### 5.4 冻结指标与阶段 gate

最终主要指标原定为 Answer Correctness；大规模 Phase 3 screening metric 为 normalized token F1。由于 Answer Correctness 调用昂贵，只有 F1 screen 达到 `+0.01` practical gain 并满足稳定性要求后，才允许打开计划的 4,000-query fresh-development subset 和扩展 Answer Correctness。Phase 0 同时规定 fresh development 不能用于模型筛选，internal final holdout 只能在全部前置 gate 通过后打开。

### 5.5 结论

Phase 0 的成果是冻结协议而非性能数字。它保证后续任何“继续训练”“打开新数据”“调用 Answer Correctness”或“进入 final”都有明确逻辑前提，也为后续多次停止决策提供可审计边界。

## 6. Phase 1：生成链路、上下文价值与 Answer Correctness 评审校准

### 6.1 研究问题

Phase 1 回答：生成模型是否真正利用检索上下文，gold page context 是否显著优于 no-context，BM25/Dense baseline 是否处于合理区间，以及 Answer Correctness 自动评审能否与人工检查保持足够一致。如果这一阶段失败，后续 Router 标签就没有可信基础。

### 6.2 实验设计

- 样本：100 pilot queries。
- 动作：no-context、gold-page-context、BM25、Dense。
- Primary metric：Answer Correctness。
- 每动作 100 次主生成，共 400；每动作前 10 queries 再生成一次，共 40 generation repeat pairs。
- Answer Correctness 做 20 judge repeat pairs；另抽取 50 条做 blind manual calibration。
- 人工校准：抽查 50 条评审结果。
- 自动 gate：gold context 相对 no-context 的 F1 与 Answer Correctness gain 均至少 `+0.05`；generation、judging 和 pending failure 均为 0；最后还需人工接受 calibration。

### 6.3 具体结果

| Action | Answer Correctness | Normalized token F1 | Exact match | Evidence-page recall | Retrieval hit | Mean retrieval latency |
|---|---:|---:|---:|---:|---:|---:|
| No context | 0.470000 | 0.424520 | 0.310000 | 0 | 0 | 0 ms |
| Gold page context | 0.940000 | 0.814556 | 0.670000 | 1.000000 | 1.000000 | 0 ms |
| BM25 | 0.645000 | 0.544056 | 0.440000 | 0.575000 | 0.860000 | 1,669.836 ms |
| Dense | 0.685000 | 0.602762 | 0.480000 | 0.730000 | 0.910000 | 331.481 ms |

Gold page context 相对 no-context 的 Answer Correctness 增益为 `+0.470000`，95% confidence interval 为 `[0.373737,0.564373]`；normalized token F1 增益为 `+0.390036`，区间 `[0.292557,0.485963]`。

重复稳定性：40 generation pairs 的 normalized answer exact agreement `0.925000`，F1、exact match 和 Answer Correctness score agreement 均为 `1.000000`；20 judge repeat pairs 的 exact label agreement 和 linear weighted kappa 均为 `1.000000`。

50 条人工 calibration 的完整混淆矩阵为：

| Human label / Judge label | Incorrect | Partially correct | Correct |
|---|---:|---:|---:|
| Incorrect | 5 | 1 | 0 |
| Partially correct | 2 | 1 | 3 |
| Correct | 0 | 1 | 37 |

自动评审与人工评分 exact agreement 为 `0.86`，linear weighted kappa 为 `0.754558`，没有 two-step disagreement；judge mean minus human mean score 为 `+0.01`。按 human class 的 exact agreement 为 incorrect `0.833333`、partially correct `0.166667`、correct `0.973684`。Partially-correct 只有 6 条且一致性弱，是该 judge 校准必须保留的局限。

Generation、Answer Correctness 和 pending failures 均为 0；generation cost `0.069606` 美元、judge cost `0.091229` 美元、总估算 `0.160834` 美元。

### 6.4 决策与结论

冻结决策为 `GO`。这证明：上下文质量会显著影响最终答案；检索动作之间存在可测差异；Answer Correctness judge 的整体校准足以支持受控后续实验。它并不表示 judge 在 partially-correct 类别完美可靠。Phase 1 没有证明任何 Router 能预测动作，只证明评估链路可用。

## 7. Phase 2：固定动作、逐查询 oracle 与重复生成稳定性

### 7.1 研究问题

Phase 2 回答四个问题：

1. Dense 是否是总体最佳固定动作？
2. BM25 是否仍在一部分查询上更好？
3. 逐查询 oracle 相对最佳固定动作有多大答案级 headroom？
4. 每个动作生成三次后，winner 与答案指标是否足够稳定？

### 7.2 初始 BM25/Dense headroom run：因重复稳定性不足而修订

初始 run 使用无放回抽取的 300 train queries 和 300 historical development queries。每个 query/action 先生成一次主答案，并在每个 partition 各抽 20 queries/action 做重复敏感性，共 80 generation repeat pairs。Primary screening 为 normalized token F1，Answer Correctness 为 confirmatory endpoint；以 group 为单位 bootstrap 2,000 次。F1 practical headroom 及 interval lower 均要求至少 `0.01`；high-margin 为 `|gap|>=0.5`，两动作各需至少 10 个 high-margin winners；repeat F1、Answer Correctness score 和 judge label agreement 都预期至少 `0.95`。

| Partition | Action | Normalized token F1 | Exact match | Answer Correctness |
|---|---|---:|---:|---:|
| Train, 300 | BM25 | 0.643020 | 0.561667 | 0.698333 |
| Train, 300 | Dense | 0.687756 | 0.601667 | 0.741667 |
| Historical development, 300 | BM25 | 0.627204 | 0.541667 | 0.683750 |
| Historical development, 300 | Dense | 0.693640 | 0.591667 | 0.760000 |

Historical development 上的固定 Dense F1 为 `0.693640`，query-wise F1 oracle 为 `0.747770`，headroom 为 `+0.054130`，95% confidence interval 为 `[0.032729, 0.078278]`。如果使用 F1 oracle 的动作选择，Answer Correctness 增益为 `+0.046667`、exact match 增益为 `+0.060000`；独立 Answer Correctness oracle headroom 为 `+0.048333`。在 `0.5` high-margin threshold 下，BM25 high-margin winner 16、Dense high-margin winner 37、tie 或低 margin 247。正 headroom 仅来自 25 条 query，其中单条最大贡献占总正收益 `6.158%`，前五条占 `30.790%`，说明收益集中。

稳定性抽查包括 80 个 generation repeat pairs 和 40 个 judge repeat pairs：normalized answer agreement `0.8625`、F1 agreement `0.9000`、exact match agreement `0.9500`、Answer Correctness score agreement `0.9625`、judge label agreement `0.9750`。Generation、Answer Correctness 和 pending failure 都为 0；估算 generation 成本 `0.301606` 美元、Answer Correctness 成本 `0.263911` 美元、总成本 `0.565516` 美元。

F1 headroom、Answer Correctness confirmation、两动作 high-margin support 和完整性均通过，但 `repeat_stability_passed=false`，所以该 run 的正式决策是 `REVISE`，不能把它写成最终放行。

### 7.3 600-query、三重复 BM25/Dense revision：稳定性放行

Revision 将 300 train 与 300 historical development 的 BM25/Dense 两动作结果按三重复重新汇总。每个 query/action 先对三次生成取均值，再计算固定动作和 oracle。

| Partition | Action | F1 | Exact match | Answer Correctness |
|---|---|---:|---:|---:|
| Train, 300 | BM25 | 0.637989 | 0.557778 | 0.692778 |
| Train, 300 | Dense | 0.687148 | 0.597778 | 0.743333 |
| Historical development, 300 | BM25 | 0.632704 | 0.545556 | 0.685556 |
| Historical development, 300 | Dense | 0.695528 | 0.594444 | 0.758333 |

Historical development 的 F1 oracle 为 `0.746050`，相对固定 Dense `0.695528` 的 headroom 为 `+0.050521`，95% confidence interval 为 `[0.031298, 0.073820]`。使用 F1-oracle 动作评估 Answer Correctness 时，相对固定 Dense 增益为 `+0.042778`；独立 Answer Correctness oracle headroom 为 `+0.046667`。

重复稳定性指标为：normalized answer pair agreement `0.90444`、F1 pair agreement `0.95444`、exact match pair agreement `0.97556`、Answer Correctness pair agreement `0.96722`；leave-one-repeat-out 的三个旋转均通过。

| Omitted repeat | Fixed Dense F1 | Oracle F1 | Headroom | 95% confidence interval | AC gain using F1 oracle | Passed |
|---:|---:|---:|---:|---|---:|---|
| 0 | 0.695639 | 0.745731 | 0.050091 | [0.030632,0.073293] | 0.042500 | True |
| 1 | 0.696350 | 0.751321 | 0.054971 | [0.033701,0.077984] | 0.047500 | True |
| 2 | 0.694596 | 0.743411 | 0.048815 | [0.029436,0.070458] | 0.042500 | True |

600-query pooled 结果中，F1 winner 为 BM25 60、Dense 106、tie 434；Answer Correctness winner 为 BM25 46、Dense 86、tie 468。固定 BM25 pooled mean F1 为 `0.635347`，固定 Dense 为 `0.691338`，query oracle 为 `0.742469`。

该 revision 共有 1,200 个完整 query/action cells 和 3,600 个 generation repeat pairs，无失败；估算 generation 成本 `0.849642` 美元、Answer Correctness 成本 `0.728005` 美元、合计 `1.577647` 美元。全部 gate 通过，决策为 `GO`。

### 7.4 Answer Correctness 为主的三动作 revision

在 BM25/Dense 重复协议稳定后，Phase 2 又用相同的 300 train 加 300 historical development queries 汇总 `no_context`、`bm25`、`dense` 三动作，每个 query/action 三次生成，以 Answer Correctness 作为主要 endpoint。Historical development 结果为：

| Partition | Action | Answer Correctness | Normalized token F1 | Exact match |
|---|---|---:|---:|---:|
| Train | No context | 0.499444 | 0.472211 | 0.394444 |
| Train | BM25 | 0.692778 | 0.637989 | 0.557778 |
| Train | Dense | 0.743333 | 0.687148 | 0.597778 |
| Historical development | No context | 0.499444 | 0.470028 | 0.376667 |
| Historical development | BM25 | 0.685556 | 0.632704 | 0.545556 |
| Historical development | Dense | 0.758333 | 0.695528 | 0.594444 |

Dense 仍是最佳固定动作。按 Answer Correctness 事后选择的三动作 oracle 为 `0.833333`，相对 fixed Dense `0.758333` 的 headroom 为 `+0.075000`，95% confidence interval `[0.048875,0.105000]`。在 AC-oracle actions 下 F1 为 `0.756987`，相对 fixed Dense F1 增加 `+0.061459`，区间 `[0.036380,0.089485]`；独立 F1 oracle headroom `+0.087538`。AC-oracle exact-match delta `+0.063333`；独立 exact-match oracle headroom `+0.102222`。

Historical development unique winners 为 no-context 10、BM25 15、Dense 32；tied-best patterns 为 BM25+Dense 68、no-context+BM25 5、三者全 tie 156、no-context+Dense 14。Pairwise AC preferences：no-context vs BM25 为 27/184 ties/89，no-context vs Dense 为 15/181/104，BM25 vs Dense 为 20/233/47。Train unique winners 为 16/17/23，BM25-vs-Dense 为 26/235/39。

Leave-one-repeat-out AC headroom 为 `0.074167/0.077500/0.073333`，全部通过。Repeat diagnostics：normalized answer `0.900185`、F1 `0.959630`、exact match `0.978148`、Answer Correctness `0.966296`、judge label `0.975000`。30 条 positive-gain queries 中 top one/top five 占总 headroom `4.444%/22.222%`。

总 outcomes 为 600×3×3=`5,400`；generation cost `0.906210` 美元、Answer Correctness cost `1.087417` 美元、合计 `1.993627` 美元。失败数为 0，决策为 `GO`。

### 7.5 决策与结论

Phase 2 的完整决策链是：初始 BM25/Dense run 因重复稳定性不足而 `REVISE`；三重复 BM25/Dense revision 通过后为 `GO`；Answer Correctness 为主的三动作 revision 也为 `GO`。答案级逐查询 headroom 明确存在，但集中在少量非 tie query 上。该阶段只证明“如果能够知道正确动作，Router 有价值”，并不证明查询文本能够预测正确动作。

## 8. Phase 2.5：Answer Correctness 跨重复稳定性与 normalized token F1 辅助一致性审计

### 8.1 研究问题

大规模 Answer Correctness judging 成本较高。Phase 2.5 使用已有 600-query、BM25/Dense、三重复数据先回答 primary question：Answer Correctness preference 是否能够跨 generation repeats 稳定，在两个 repeats 上选择动作后，能否在未参与选择的第三个 repeat 上仍提高 Answer Correctness。Secondary question 才是 normalized token F1 winner 是否与 Answer Correctness winner 大体一致，以及 F1 selector 是否也能改善 Answer Correctness。该阶段没有把 F1 晋升为 primary Router training label。

### 8.2 实验设计

该阶段不进行外部调用，也不重新生成答案。Primary gate partition 是 300-query train；300-query historical development 只作 corroboration；600-query pooled 结果只作描述。Train 与 historical development 分开计算 winner、非 tie sign agreement、按 Answer Correctness gap 加权的一致性、F1 selector 的 Answer Correctness gain、coverage 和 precision，并分别对 F1 selector 与 Answer Correctness selector 做三次 cross-repeat rotation：每次用两个 repeats 形成动作 preference，再在第三个 repeat 上评价 Answer Correctness，最后在 query 内对三次旋转求平均。Tie-break 为 Dense；bootstrap unit 为 information-need group；resamples 为 10,000。

### 8.3 具体结果

| 指标 | Train | Historical development |
|---|---:|---:|
| F1 winner：BM25 / tie / Dense | 29 / 222 / 49 | 31 / 212 / 57 |
| Answer Correctness winner：BM25 / tie / Dense | 26 / 235 / 39 | 20 / 233 / 47 |
| Answer Correctness non-tie 上 F1/AC sign agreement | 0.90769 | 0.94030 |
| Answer Correctness gap weighted sign agreement | 0.94576 | 0.97324 |
| F1 selector 的 Answer Correctness gain over Dense | +0.05056 | +0.04278 |
| 95% confidence interval | [0.02944, 0.07389] | [0.02278, 0.06556] |
| F1 selector switch coverage | 9.67% | 10.33% |
| F1 selector switch precision | 75.86% | 61.29% |

Cross-repeat F1 selector 的 mean Answer Correctness gain：Train 为 `+0.046667`，95% confidence interval `[0.025556, 0.069444]`；historical development 为 `+0.033889`，区间 `[0.013333, 0.057222]`。

正式 gate 使用的是 cross-repeat Answer Correctness selector，而不是上述 F1 selector。完整结果为：

| Partition | Held-out repeat 0：AC gain / switches / coverage | Held-out repeat 1 | Held-out repeat 2 | 三旋转 mean AC gain | 95% confidence interval | Mean coverage |
|---|---|---|---|---:|---|---:|
| Train primary gate | +0.056667 / 24 / 0.080000 | +0.038333 / 25 / 0.083333 | +0.051667 / 21 / 0.070000 | +0.048889 | [0.026667,0.072778] | 0.077778 |
| Historical development corroboration | +0.040000 / 17 / 0.056667 | +0.033333 / 19 / 0.063333 | +0.043333 / 18 / 0.060000 | +0.038889 | [0.018889,0.061667] | 0.060000 |

F1 preference 与 Answer Correctness preference 的完整 confusion 为：

| Partition | AC preference | F1 says BM25 | F1 tie | F1 says Dense |
|---|---|---:|---:|---:|
| Train | BM25 | 22 | 4 | 0 |
| Train | Tie | 7 | 216 | 12 |
| Train | Dense | 0 | 2 | 37 |
| Historical development | BM25 | 19 | 0 | 1 |
| Historical development | Tie | 10 | 211 | 12 |
| Historical development | Dense | 2 | 1 | 44 |
| Pooled | BM25 | 41 | 4 | 1 |
| Pooled | Tie | 17 | 427 | 24 |
| Pooled | Dense | 2 | 3 | 81 |

Train cross-repeat F1 selector 在 held-out repeats 0/1/2 的 Answer Correctness gain 为 `0.051667/0.035000/0.053333`，switches `29/27/27`，coverage `0.096667/0.090000/0.090000`；leave-one-repeat stable preference 为 BM25 19、Dense 34、unstable/tie 247。Historical development 的 cross-repeat F1 selector 对应 gain `0.036667/0.023333/0.041667`，switches `30/29/25`，coverage `0.100000/0.096667/0.083333`；stable preference 为 BM25 14、Dense 38、unstable/tie 248。

Train F1 selector 在 AC 上的 switched-query conditional gain 为 `0.522989`；independent AC oracle gain `+0.056667`，区间 `[0.035000,0.080556]`。Historical development F1 selector conditional gain 为 `0.413978`。这些数据支持 F1 具有低成本方向性诊断价值，但冻结角色仍是 auxiliary or diagnostic；对少量 disagreement queries，它不等同于语义评审，也没有在本阶段被授权为 primary screen。

### 8.4 决策与结论

决策为 `GO_AC_LABEL_EXPANSION`。其 primary 依据是 train cross-repeat Answer Correctness selector 的平均增益 `+0.048889`，95% confidence interval `[0.026667,0.072778]`，说明 Answer Correctness preference 对 generation repeat 具有足够稳定性，可以扩大 Answer Correctness labels。F1 的冻结状态是 `f1_retained_as_auxiliary=true`、`f1_role=auxiliary_or_diagnostic`：F1 与 Answer Correctness 的一致性支持它作辅助分析，但本阶段没有授权把 F1 当作 primary training/screening label。后续 Phase 3 另行选择 normalized token F1 作为廉价 screen，属于后续协议决策，不能倒写成 Phase 2.5 的 gate 结论。

### 8.5 历史三动作 Phase 3 learning curve：为什么收缩到二动作 Phase 2.6

历史运行目录把这组实验称为 Phase 3；在本研究链中它位于 Phase 2.5 与 Phase 2.6 之间。研究问题是：严格 query-only/corpus-static 的 `no_context`、BM25、Dense 三动作 pairwise Router 是否会随训练 queries 从 1,200 增加到 9,600 而兑现 oracle headroom。

三个 pair 为 no-context vs BM25、no-context vs Dense、BM25 vs Dense。Pair label 是左动作 mean F1 是否高于右动作，exact tie 跳过，sample weight 为两动作 gap 绝对值。三个 pairwise probabilities 用 Borda-style sum 聚合为 action，聚合 tie 默认 Dense。Full base query feature vector 为 414 维，Tier-A base vector 为 17 维；实际 pairwise design 还拼接 3 维 pair one-hot 与三组 pair-specific feature interactions。因此 full model matrix 为 `414 + 3 + 3×414 = 1,659` 维，Tier-A matrix 为 `17 + 3 + 3×17 = 71` 维。候选为 1,659D full XGBoost、同一 full pair design 的 regularized logistic-regression capacity control、71D Tier-A XGBoost；每个规模 5 folds、5 model seeds。不能把 414/17 的 base dimensions 写成最终 pairwise model dimensions。

| Queries | Fixed no-context | Fixed BM25 | Fixed Dense | Three-action oracle | Oracle headroom | Full XGBoost median gain | Seed range | Positive seeds |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1,200 | 0.468873 | 0.611435 | 0.655932 | 0.749612 | 0.093681 | -0.006965 | [-0.011782,-0.002007] | 0/5 |
| 2,400 | 0.469031 | 0.615735 | 0.648342 | 0.745488 | 0.097146 | -0.003120 | [-0.007453,-0.000535] | 0/5 |
| 4,800 | 0.450434 | 0.618936 | 0.647134 | 0.744213 | 0.097079 | -0.002814 | [-0.003932,+0.001833] | 1/5 |

| Queries | Candidate | Median gain | Minimum | Maximum | Positive seeds |
|---:|---|---:|---:|---:|---:|
| 1,200 | Full logistic | -0.025112 | -0.031740 | -0.020959 | 0/5 |
| 1,200 | Tier-A XGBoost | -0.007998 | -0.011532 | -0.004511 | 0/5 |
| 2,400 | Full logistic | -0.027564 | -0.029197 | -0.023290 | 0/5 |
| 2,400 | Tier-A XGBoost | -0.005897 | -0.008191 | -0.003496 | 0/5 |
| 4,800 | Full logistic | -0.016234 | -0.019662 | -0.014035 | 0/5 |
| 4,800 | Tier-A XGBoost | -0.001546 | -0.002867 | +0.002301 | 2/5 |

4,800 时 non-tie pair rows 5,678、tie pair rows 8,722。Full XGBoost pairwise weighted accuracy 约 `0.758–0.767`，但 end utility 仍不超过 Dense；这说明 pair classification accuracy 不能替代 policy F1。

规划的 9,600 run 在此旧三动作协议下取消；partial state 有 65,500 successes、20,900 pending，禁止用于正式训练或指标。Model 未冻结，计划的 4,000-query fresh-development subset、Answer Correctness evaluation 和 internal final holdout 均未进入。1,200→4,800 只把负 gain 推近零，没有接近 `+0.01`，因此后续 Phase 2.6 将问题收缩为“以 Dense 为安全默认，只选择性切换到 BM25”的二动作 gate。

## 9. Phase 2.6：4,800-query 严格预检索 Router

### 9.1 研究问题

Phase 2.6 是首次正式检验 Dense-default、BM25/Dense 二动作 selective gate：只用 query 与 corpus-static 特征，是否能预测 BM25 相对 Dense 的三重复平均答案 F1 差，并以实际策略超过固定 Dense 至少 `+0.01 F1`。它不是整个项目首次运行严格 query-only Router；上一节旧三动作 Phase 3 已经做过三动作 pairwise learning curve。

### 9.2 数据与特征

- Queries：4,800。
- Actions：BM25、Dense。
- Repeats：每个 query/action 3 次。
- Successful outcome rows：28,800。
- BM25 winners：596。
- Dense winners：797。
- Exact ties：3,407，占 `70.98%`。
- Tier-A 特征：17 维 lexical compatibility。
- Full 特征：17 维 lexical compatibility、13 维 corpus-static prototype summaries、384 维 BGE query embedding，共 414 维。

13 维名为 `dense` 的特征块是静态 corpus prototype summaries，不是当前 query 的 Dense retrieval 结果。当前 retrieval outcomes、qrels、gold evidence、生成答案和答案指标全部禁止作为输入。

17 个 lexical features 的完整名称是：`corpus_vocabulary_coverage`、`corpus_oov_ratio`、`idf_mean`、`idf_std`、`idf_min`、`idf_max`、`idf_p90`、`idf_max_share`、`log_document_frequency_mean`、`log_document_frequency_std`、`df_at_most_10_ratio`、`df_at_most_100_ratio`、`df_at_most_1000_ratio`、`numeric_anchor_max_idf`、`year_anchor_max_idf`、`capitalized_anchor_max_idf`、`quoted_anchor_max_idf`。

13 个 prototype summaries 的完整名称是：`prototype_similarity_top_1` 至 `prototype_similarity_top_8`、`prototype_similarity_margin`、`prototype_similarity_mean`、`prototype_similarity_std`、`prototype_similarity_entropy`、`prototype_local_density`。它们来自训练前冻结的 64 个 corpus prototypes。

### 9.3 模型与验证设计

- Nested grouped cross-validation：5 outer folds、4 inner folds。
- Fold seed：20260901。
- Model seeds：11、23、37、53、71。
- Bootstrap：10,000 次。
- Default action：Dense。
- Practical gain：`+0.01`。
- Pairwise 模型预测 BM25/Dense winner；continuous-gain 模型直接预测 (g(q))。
- Inner folds 只用于选择切换阈值；outer fold 只用于无泄漏评估。
- 五个 model seed 的策略以 majority vote 聚合，平票保留 Dense。

三个候选为：full pairwise XGBoost classifier 在 non-ties 上训练，label 为 `g(q)>0`、weight 为 `|g(q)|`；full gain XGBoost regressor 在全部 queries 上直接预测连续 gap；Tier-A pairwise XGBoost 只用 17 lexical features。XGBoost 固定参数为 300 estimators、maximum depth 3、learning rate 0.05、minimum child weight 5、subsample 0.8、column sample by tree 0.8、L2 lambda 5、histogram tree method、8 jobs；classifier objective 为 binary logistic，regressor 为 squared error。

### 9.4 固定动作、oracle 与候选结果

| 项目 | Mean F1 |
|---|---:|
| Fixed BM25 | 0.618936 |
| Fixed Dense | 0.647134 |
| BM25/Dense query oracle | 0.717841 |
| Oracle headroom over Dense | +0.070707 |

Retrieval 与答案的描述数据为：BM25 F1/exact match/evidence-page recall/hit=`0.618936/0.518611/0.563958/0.842917`；Dense=`0.647134/0.546111/0.703438/0.918125`。

| Candidate | Router F1 | Gain over Dense | 95% confidence interval | Switch coverage | Switch precision | Positive seeds |
|---|---:|---:|---|---:|---:|---:|
| Full pairwise XGBoost | 0.646419 | -0.000715 | [-0.003756, 0.002293] | 9.81% | 11.68% | 2/5 |
| Full continuous-gain XGBoost | 0.649283 | +0.002149 | [-0.000585, 0.004876] | 6.92% | 15.36% | 4/5 |
| Tier-A pairwise XGBoost | 0.645317 | -0.001817 | [-0.006099, 0.002583] | 17.98% | 13.33% | 0/5 |

最强 full continuous-gain XGBoost 的 seed median gain 为 `+0.000511`，seed range 为 `[-0.000605,+0.002768]`。它的 conditional F1 gain 为 `+0.031074`，但只覆盖 6.92% 查询且 switch precision 仅 15.36%；少量正确切换的收益被错误切换抵消。

| Candidate | Seed 11 | Seed 23 | Seed 37 | Seed 53 | Seed 71 | Median |
|---|---:|---:|---:|---:|---:|---:|
| Full pairwise | -0.004084 | -0.000431 | +0.000228 | -0.001994 | +0.000976 | -0.000431 |
| Full continuous gain | +0.002503 | +0.000511 | +0.000377 | -0.000605 | +0.002768 | +0.000511 |
| Tier-A pairwise | -0.002710 | -0.002500 | -0.004220 | -0.002013 | -0.001921 | -0.002500 |

对应 seed switch counts：full pairwise `555/341/538/724/795`；full gain `309/560/369/495/430`；Tier-A `1004/734/1122/837/1142`。最强 full-gain 的 five-seed precision 为 `18.45%/11.96%/11.92%/13.74%/15.35%`，表现并不稳定。

### 9.5 决策与结论

没有候选达到 `+0.01`，没有候选的 confidence interval 下界大于 0，也没有候选在全部 seeds 上为正。正式决策为：

`STOP_BEFORE_AC_LABELS`

这不是“BM25 没有价值”。Oracle headroom 为 `+0.070707`，说明 BM25 对部分查询很有价值；失败点是 414 维预检索输入无法高精度识别这部分查询。由于 F1 screen 未过 gate，9,600 learning curve、计划的 4,000-query fresh-development subset、扩大 Answer Correctness 和 internal final holdout 均不应在此协议下继续。

## 10. 辅助实验一：Contriever 是否可以解决检索器配对问题

### 10.1 研究问题

Phase 2.6 的失败可能来自 BGE 与 BM25 的配对结构，而不是 Router 架构本身。Contriever 修订因此检验：将 Dense 动作换成 Contriever 后，是否会形成更容易从 query-only 特征预测的 BM25/Contriever 决策边界。

### 10.2 实验设计

- Data：冻结 train 前 1,200 queries。
- Actions：BM25、Contriever，并保留 BGE 作为外部固定参考。
- Repeats：每个 query/action 3 次。
- Successful rows：10,800。
- Contriever query embedding：768 维。
- Lexical features：17 维。
- Corpus-static summaries：13 维。
- Nested grouped cross-validation、model seeds 和主要 gate 与 Phase 2.6 一致。
- 计划的 4,000-query fresh-development subset、internal final holdout 和 Answer Correctness 均未读取或调用。

Contriever contract：`facebook/contriever` revision `2bd46a25019aeea091fd42d1f0fd4801675cf699`；attention-masked mean pooling；document input 为 `title + space + text`；raw vectors 不归一化，使用 raw inner product；maximum sequence length 512、batch size 16。Corpus 为 5,233,329 rows，index 为 exact `streaming_flat_ip`，query batch size 100。Feature 总维数为 17 lexical + 13 prototype + 768 query embedding=`798`；64 prototypes 由 100,000 corpus rows 构造。

### 10.3 具体结果

| Fixed action | Mean F1 | Exact match | Evidence-page recall | Retrieval hit |
|---|---:|---:|---:|---:|
| BM25 | 0.611435 | 0.504444 | 0.577500 | 0.850000 |
| BGE | 0.655932 | 0.549444 | 0.703333 | 0.913333 |
| Contriever | 0.577866 | 0.460833 | 0.491250 | 0.783333 |

三动作 oracle F1 为 `0.738682`，unique/tie winners 为 BM25 87、BGE 120、Contriever 67、tie 926。BM25/Contriever pair winner 为 BM25 221、Contriever 155、tie 824；pair oracle 为 `0.681258`，比最佳 pair-fixed BM25 高 `+0.069823`。候选结果：

| Candidate | Gain over pair-fixed BM25 | 95% confidence interval | Gain over fixed BGE | Switch precision |
|---|---:|---|---:|---:|
| Full pairwise XGBoost | -0.000153 | [-0.003889, 0.003638] | -0.044649 | 14.29% |
| Full continuous-gain XGBoost | +0.001208 | [-0.002219, 0.004823] | -0.043288 | 16.00% |
| Tier-A pairwise XGBoost | -0.003561 | [-0.007688, -0.000070] | -0.048058 | 10.81% |

Full pairwise 的 five-seed pair gains 为 `+0.001419/+0.001189/-0.000384/-0.000576/+0.001476`；full continuous gain 为 `-0.001019/+0.002233/-0.003293/+0.002871/-0.000278`；Tier-A 为 `-0.004394/-0.003651/-0.002756/-0.004394/-0.004456`。Full gain 虽在 25 个 switches 上有 `+0.058000` conditional gain，却无法形成总体显著收益。

### 10.4 决策与结论

决策为 `ARCHITECTURE_DIAGNOSIS_REQUIRED`。换成 Contriever 后仍然出现约 `0.07` oracle headroom 与接近零 Router gain 的组合；而 Contriever 的全局 F1 还显著低于固定 BGE。因此瓶颈不是简单选择了错误的 Dense retriever pair，而是严格预检索输入对动作效用差缺少足够可观测性。

## 11. 辅助实验二：Natural Questions 上的 Dense Passage Retrieval 表征诊断

### 11.1 研究问题与证据边界

该实验检验：另一种 dense query encoder 的向量空间中是否已经形成 BM25-only 与 BGE-only 查询的可解码结构。它是使用 qrels 标签的离线 representation diagnostic；不运行 retrieval、reranking 或 generation，也不训练 HotpotQA 部署 Router。

### 11.2 数据与设计

- Natural Questions queries：3,452。
- BM25-only queries：260。
- BGE-only queries：1,049。
- Exclusive binary diagnostic set：1,309。
- Encoder：`facebook/dpr-question_encoder-multiset-base`。
- Embedding dimensions：768。
- 比较 DPR-only linear probe、surface-feature probe、surface plus DPR probe。
- 同时计算 k-nearest-neighbor same-label purity 与 cosine silhouette，检查局部几何是否形成自然决策簇。

标签定义为 equal-50 protocol 下 BM25 nDCG@10 大于 0 且 BGE 等于 0，或反向 exclusive success；两者同成功/同失败排除。Natural Questions 在 18 Selected-7 units 中有最大的 exclusive set 1,309；其次为 HotpotQA 974、CQADupStack/tex 661、CQADupStack/english 432、gaming 407、ArguAna 351、unix 296、physics 267、programmers 238、gis 226、mathematica 198、FiQA 194、android 166、stats 125、webmasters 114、wordpress 108、NFCorpus 43、Touché 12。

DPR revision 为 `5325e4ee906435291d63046f535476cb3fc60d43`，pooling=`pooler_output`，maximum sequence length 256。Encoder 原始输出本身不归一化，但分析阶段建立一份 L2-normalized embedding copy；所有 cosine geometry，以及 DPR-only、surface-plus-DPR 和所谓 margin-sensitivity 的全部含 DPR 线性 probes，都使用这份 L2-normalized copy。CUDA encode time 为 2.2245 seconds。

Surface baseline 共 20 维：character count、raw token count、unique token ratio、BM25-analyzed token count、stopword fraction、digit-token fraction、punctuation fraction、question-mark indicator、mean/max/min inverse document frequency、inverse-document-frequency out-of-vocabulary fraction，以及 what/who/when/where/why/how/which/whom indicators。

Duplicate normalized questions 共享 group；outer 5-fold StratifiedGroupKFold、inner 4-fold；fold-local StandardScaler；balanced-class logistic regression，liblinear，maximum iterations 4,000；C candidates=`0.001/0.01/0.1/1/10`，inner objective 为 receiver operating characteristic area under curve；group bootstrap 2,000、permutation 1,000。

### 11.3 具体结果

| Input | Dimensions | OOF ROC-AUC | 95% CI | Balanced accuracy | 95% CI | Macro F1 | 95% CI |
|---|---:|---:|---|---:|---|---:|---|
| Surface | 20 | 0.601221 | [0.565009,0.638250] | 0.579708 | [0.545142,0.612951] | 0.520932 | [0.495602,0.546421] |
| DPR | 768 | 0.677403 | [0.642912,0.711954] | 0.628738 | [0.596514,0.660437] | 0.550511 | [0.524404,0.576020] |
| Surface plus DPR | 788 | 0.663082 | [0.625356,0.701865] | 0.616296 | [0.582204,0.648874] | 0.538771 | [0.511845,0.564685] |
| DPR margin sensitivity | 768 | 0.668659 | [0.631965,0.705518] | 0.610527 | [0.575894,0.641692] | 0.535445 | [0.509682,0.558960] |

DPR minus surface AUC 为 `+0.076182`，区间 `[0.033772,0.119092]`；combined minus surface 为 `+0.061861`，区间 `[0.024219,0.099981]`。四个 probes 的 permutation p-value 均为 `0.000999`，null mean 约 `0.499–0.501`。

`DPR margin sensitivity` 这一 artifact 名称不能解释为更严格的 high-margin 子集：冻结摘要中 margin threshold 为 `0.1`，但 exclusive queries 与 margin rows 都是 `1,309`，没有过滤掉任何 query。该 run 对同一批 1,309 条 L2-normalized embeddings 只更换了 grouped cross-validation seed。因此其 `0.668659` AUC 是另一 fold assignment 下的重复结果，不是“更高 margin 样本上仍稳健”的独立证据。

20 次每类各抽 260 的 balanced geometry：centroid cosine distance median/min/max=`0.005341/0.004585/0.006162`；same-label purity at 10=`0.532981`、at 20=`0.528029`、at 50=`0.517596`；cosine silhouette=`0.003675`。PCA 前两维 explained variance ratios 为 `0.042431/0.029615`。t-SNE 已执行；UMAP 因未安装 `umap-learn` 未执行。

### 11.4 结论

DPR AUC 至少 `0.60`、interval lower 高于 0.5，以及 artifact 中名为 margin sensitivity 的 alternate-seed AUC 至少 `0.55` 这三项数值检查均通过；但第三项没有形成 margin-filtered subset，不能作为高 margin robustness 证据。Purity at 10/20/50 没有全部达到 `0.55`，因此未达到预注册 practical geometry threshold。DPR embedding 中存在方向性可解码信号，但局部 purity 接近随机边界，silhouette 几乎为 0，没有形成稳定聚类或自然 decision boundary。Combined probe 还低于 DPR-only，简单拼接 surface features 没有自动改善 representation。该结果不支持“已测试 DPR encoder 已经自然形成可直接路由的边界”；标签是 NQ retrieval-only qrels success，不是 HotpotQA 答案 F1 或 Answer Correctness，也不排除其他 query encoders。

## 12. Phase 2.7：9,600-query query-only 模型、目标、表示与规模审计

- **实验时间范围：** 2026-08-26 至 2026-08-27。
- **最终 query-only 决策：** `STOP_QUERY_ONLY_V1`。
- **Stage 12 诊断决策：** `GOLD_EVIDENCE_SIGNAL_SUFFICIENT`。
- **可部署 Router：** 无。

### 12.1 研究问题

Phase 2.6 的最佳模型只有 `+0.002149 F1`，但该结果尚不能区分信息不足与建模不当。Phase 2.7 因而系统回答：

1. 30 维 structured query/corpus-static 特征是否含有可泛化信号？
2. 384 维 BGE query embedding 是否含有可解码的 Retriever 偏好？
3. 将 30D structured 与 384D raw embedding 直接拼接后交给 XGBoost，是否是错误的 model–feature inductive bias？
4. Fold-internal principal component analysis、线性模型或 late fusion 是否更适合 embedding 几何？
5. Continuous utility gap、robust regression、weighted sign classification 或 dual utility heads 中，哪种监督目标最合理？
6. 从 1,200、2,400、4,800 扩大到 9,600 query，是否仍能带来足够收益？
7. 如果所有 query-only 候选失败，信息是否主要存在于实际 retrieval output 与 gold evidence sufficiency？

正式问题仍是：只使用部署时可获得的 query 特征和 corpus-static summaries，能否预测每个查询的 BM25-minus-Dense Answer-F1 gap，并稳定、显著且安全地优于固定 Dense？

### 12.2 数据、生成、成本与基线

| 项目 | 实际设置 |
|---|---|
| Dataset / partition | HotpotQA / train-only |
| Queries | 9,600 |
| Groups | 使用冻结 `group_id`；相关 query 不跨 fold |
| Actions | BM25、Dense/BGE |
| Repeats per action | 3 |
| Complete outcome rows | 57,600 |
| Retriever candidate K | 50 |
| Generation context K | 5 |
| Context limit | 1,800 tokens |
| Prompt | `hotpot_short_answer_v1` |
| Generator | `gpt-4.1-mini-2025-04-14` |
| Temperature | 0 |
| Maximum output | 64 tokens |
| Utility | 三次 normalized token F1 的动作内均值 |
| Default action | Dense |

本阶段保持 Phase 2.6 的旧 prompt、Top-5 context、generator 和三重复，以保证 4,800 与 9,600 监督标签语义一致；新版 `hotpot_multihop_short_answer_v2` 没有混入。

补齐到 9,600 query 时，新尝试并完成 13,934 个缺失 generation cells，失败 0；输入 `7,940,094` tokens，输出 `80,045` tokens，估算新增费用 `$3.3041096`，低于 `$4.5` hard budget，workers=4。补齐后 BM25/Dense 各有 28,800 successful cells，无 pending/retryable。Snapshot 冻结后，query-only 模型审计与 Stage 12 都是 zero external call。

| Strategy | Mean Answer F1 |
|---|---:|
| Fixed BM25 | 0.620175 |
| Fixed Dense | 0.648209 |
| Per-query BM25/Dense oracle | 0.717565 |
| Oracle headroom over Dense | +0.069356 |

Winner strata：BM25 winner 1,196、Dense winner 1,564、exact tie 6,840；tie fraction 为 `71.25%`。Tie 只表示两个动作的三重复平均 F1 完全相同，不表示检索结果相同。

Stage 2–6 quick screen 只使用冻结前 4,800 queries：28,800 outcomes，BM25/Dense/tie=`596/797/3,407`，fixed BM25/Dense/oracle=`0.618936/0.647134/0.717841`。9,600 snapshot 的前 4,800 outcomes、summary 和 features 与原冻结池相同；候选筛选没有读取新增 4,800 labels。Stage 7.2 与 Stage 8 才是 9,600-query formal confirmation。

### 12.3 特征与模型候选空间

严格 query-only 输入共 414 维：

- 17D lexical compatibility：corpus vocabulary coverage、out-of-vocabulary ratio、inverse document frequency 的 mean、standard deviation、minimum、maximum、90th percentile、maximum share；log document frequency mean 与 standard deviation；document frequency 不高于 10、100、1000 的比例；numeric、year、capitalized 与 quoted anchor 的最大 inverse document frequency。
- 13D corpus prototype summaries：Top-1 至 Top-8 prototype similarity、margin、mean、standard deviation、entropy 与 local density。
- 384D BGE query embedding：`BAAI/bge-small-en-v1.5`。

17D lexical 与 13D prototype 合称 30D structured block。Embedding 可保持 raw 384D，也可在每个 outer-train fold 内中心化并降至 16、32 或 64 个 principal components；不 whiten。Raw concatenation 为 414D。

17D 的精确字段名为 `corpus_vocabulary_coverage`、`corpus_oov_ratio`、`idf_mean`、`idf_std`、`idf_min`、`idf_max`、`idf_p90`、`idf_max_share`、`log_document_frequency_mean`、`log_document_frequency_std`、`df_at_most_10_ratio`、`df_at_most_100_ratio`、`df_at_most_1000_ratio`、`numeric_anchor_max_idf`、`year_anchor_max_idf`、`capitalized_anchor_max_idf`、`quoted_anchor_max_idf`。13D 精确字段名为 `prototype_similarity_top_1` 至 `prototype_similarity_top_8`、`prototype_similarity_margin`、`prototype_similarity_mean`、`prototype_similarity_std`、`prototype_similarity_entropy`、`prototype_local_density`；embedding fields 为 `query_embedding_0` 至 `query_embedding_383`。

候选 family：

- M0：30D structured 输入 XGBoost，比较 squared-error 与 pseudo-Huber。
- M1a：raw 384D embedding 输入 Ridge。
- M1b：raw 384D embedding 输入 Elastic Net。
- M2：raw 414D concatenation 输入 XGBoost；同时承担 Phase 2.6 recipe control。
- M3：PCA-16/32/64 embedding 与 structured 拼接，比较 Ridge、Huber 与 robust XGBoost。
- M4：structured 父模型与 embedding 父模型的 inner-out-of-fold late fusion；meta-model 为 Ridge。
- M5：分别预测 BM25 utility 与 Dense utility，再取差；使用 dual Huber heads 与 gap consistency loss。

### 12.4 交叉验证、预处理与搜索空间

- Repeated StratifiedGroupKFold split seeds：20260901、20260917、20261003。
- 每个 split 5 个 outer folds。
- 每个 outer-train 内 4 个 inner folds。
- Strata：BM25 winner、exact tie、Dense winner。
- Group field：`group_id`。
- XGBoost model seeds：11、23、37、53、71。
- StandardScaler、embedding centering、principal component analysis、XGBoost early-stopping split、late-fusion meta model 和 affine calibration 全部只在 outer-train 内拟合。
- 对单个 candidate 的每个 fold，outer validation 只接收一次最终预测，不反向调节该 fold 的 preprocessing、模型参数、early stopping 或 calibration；但 Stage 2 会汇总 4,800-query screen split 的五个 outer-fold gains 来选择进入 Stage 7 的 candidate families，所以这 4,800 条本身属于已消费的 candidate-selection data，而不是独立确认数据。

冻结 YAML 的 search-space namespace 记录了 Ridge alpha `0.1/1/10/100`、Elastic Net alpha `0.0001/0.001/0.01` 与 l1 ratio `0.1/0.5`、Huber epsilon `1.2/1.35/1.5` 与 alpha `0.0001/0.001/0.01`，并把 screen mode 标为 `fixed_reference_or_small_grid`。但是，实际执行器没有对这些线性超参数运行完整 grid search：它固定使用 Ridge alpha `10.0`；Elastic Net alpha `0.001`、l1 ratio `0.1`、maximum iterations `20,000`；Huber epsilon `1.35`、alpha `0.001`、maximum iterations `2,000`；DualUtilityHuber gap lambda `0.5`。因此这些 YAML 数组是协议允许范围，不是实际已逐项评估的超参数网格。

XGBoost 同样按 candidate specification 固定 preset，而不是在每个 fold 内比较三个 preset：M0 squared-error 使用 balanced，M0 robust 使用 regularized，M2 squared-error 使用 legacy，M2 robust 使用 balanced，M3 robust 和 late-fusion structured branch 使用 regularized。三个 preset 的 `max_depth/learning_rate/min_child_weight/subsample/colsample_bytree/reg_lambda` 分别为：legacy `3/0.05/5/0.8/0.8/5`；regularized `2/0.03/15/0.8/0.5/20`；balanced `2/0.03/5/0.8/0.8/10`。每个 XGBoost candidate 最多 2,000 trees，early stopping patience 为 75；执行器在 outer-train 内再划 group-aware stopping subset 决定 tree count，然后用该 count 在完整 outer-train 上重训。Tree method 为 histogram，线程数为 8。

Screen 的 candidate selection 也必须与超参数选择区分：Stage 2 只用 split seed `20260901`、model seed `11`。每个 candidate 先产生五个 outer-fold gains，再计算 `mean(fold gains)-standard error(fold gains)` 作为 conservative screen score；这不是 inner-out-of-fold hyperparameter grid score。冻结选择规则为保留 M2 squared-error reproduction baseline，再从不同 family 中按该 screen score 选前三名。选定列表在新增 4,800 labels 可见前冻结，随后才以三个 split seeds 做 formal confirmation。Inner out-of-fold predictions 的实际职责是 late-fusion 父预测和 affine calibration，不应把它们误写成所有 candidate 的完整超参数搜索。

每个 formal outer fold 有 7,680 train、1,920 validation；validation tie 恒为 1,368，BM25 winners 为 239 或 240，Dense winners 为 312 或 313；train/validation group overlap 恒为 0。Calibration 是 inner-OOF 上的 nonnegative affine regression `a+b*predicted_gap, b>=0`。Bootstrap unit 为 `group_id`，10,000 resamples，seed `20260902`。

连续 predicted gaps 先在 model seeds 间取平均，再使用 inner-out-of-fold prediction 拟合 nonnegative affine calibrator。Primary policy 固定为 calibrated gap strictly greater than 0 切到 BM25，否则保留 Dense。自由阈值搜索和 isotonic calibration 不进入 primary gate。

### 12.5 Stage 0：冻结协议与输入预检

Stage 0 的研究目的不是比较模型，而是证明所有后续模型看到完全一致、完整且未泄漏的输入：

- 验证 57,600 个 outcome rows 与 9,600 个 feature rows。
- 验证每个 query 的 BM25/Dense 两动作与三个 repeat 完整。
- 验证所有 F1 有限。
- 验证 outcomes、summary、features、schema 与 snapshot manifest 的 SHA-256。
- 验证 query ordering、query identifier 与 group identifier 对齐。
- 验证前 4,800 query 与早期冻结池逐元素一致。
- 拒绝 v2 prompt 或其他协议的 outcome。
- 验证数据库 integrity 为 `ok`。
- 计划的 4,000-query fresh-development subset 与 internal final holdout 读取均为 0。

Stage 0 结果通过，输出 preflight 和 snapshot manifest。

### 12.6 Stage 1：严格训练/验证切分

Stage 1 为三套 split seeds 记录每个 fold 的 query 数、group 数与 winner strata 分布，并确认 train/validation 的 group 交集为 0。Stage 1 本身只构造和验证 splits，不训练模型或产出预测；这些 splits 保证后续每个模型对每条 query 恰有一次真正 out-of-fold prediction。冻结执行契约要求所有实际预处理、fold-local XGBoost early stopping、late-fusion 元模型和 calibration 只在 outer-train 内拟合，后续运行产物再验证该契约；实际 runner 未执行的完整 search-space grid 不称为已经完成的选参。

### 12.7 Stage 2：17 个封闭候选的特征块与模型结构消融

Stage 2 在 4,800-query quick screen 上运行一个冻结 split。它只用于选择正式候选，不构成可发布确认结果。17 个候选完整结果如下：

| Candidate | Family | Gain | 95% confidence interval | Non-tie AUC | Gap Spearman | Coverage | Harmful/beneficial |
|---|---|---:|---|---:|---:|---:|---:|
| M4 out-of-fold late fusion | M4 | +0.003622 | [0.000124, 0.007148] | 0.572689 | 0.079317 | 12.79% | 0.6815 |
| M3 PCA32 structured Ridge | M3 | +0.003579 | [0.000148, 0.007073] | 0.552653 | 0.061570 | 13.50% | 0.6769 |
| M3 PCA16 structured Ridge | M3 | +0.002609 | [-0.000979, 0.006214] | 0.548656 | 0.058630 | 14.88% | 0.7741 |
| M0 structured XGBoost squared error | M0 | +0.002411 | [0.000035, 0.004853] | 0.589457 | 0.095322 | 4.94% | 0.5503 |
| M3 PCA64 structured robust XGBoost | M3 | +0.002411 | [-0.000974, 0.005846] | 0.581032 | 0.085273 | 11.48% | 0.7552 |
| M3 PCA32 structured robust XGBoost | M3 | +0.002170 | [-0.000726, 0.005105] | 0.588280 | 0.091333 | 8.65% | 0.7067 |
| M3 PCA16 structured robust XGBoost | M3 | +0.001567 | [-0.001560, 0.004670] | 0.572913 | 0.079656 | 10.42% | 0.8045 |
| M3 PCA64 structured Ridge | M3 | +0.001516 | [-0.001933, 0.004930] | 0.550188 | 0.057286 | 14.06% | 0.8488 |
| M0 structured XGBoost pseudo-Huber | M0 | +0.001354 | [-0.001657, 0.004344] | 0.584329 | 0.090781 | 8.38% | 0.8075 |
| M3 PCA64 structured Huber | M3 | +0.000779 | [-0.001108, 0.002680] | 0.522218 | 0.024481 | 3.02% | 0.7494 |
| M3 PCA16 structured Huber | M3 | -0.000155 | [-0.001656, 0.001344] | 0.520486 | 0.022761 | 1.85% | 1.0955 |
| M3 PCA32 structured Huber | M3 | -0.000206 | [-0.001707, 0.001293] | 0.523368 | 0.025174 | 1.75% | 1.1306 |
| M5 dual utility Huber | M5 | -0.000284 | [-0.002997, 0.002346] | 0.542064 | 0.052444 | 8.94% | 1.0513 |
| M1b embedding Elastic Net | M1 | -0.000289 | [-0.001807, 0.001244] | 0.532991 | 0.036164 | 1.96% | 1.1745 |
| M1a embedding Ridge | M1 | -0.001194 | [-0.003508, 0.001138] | 0.531557 | 0.034356 | 5.02% | 1.3279 |
| M2 raw full XGBoost squared error | M2 | -0.001651 | [-0.003598, 0.000237] | 0.545468 | 0.049199 | 3.35% | 1.8217 |
| M2 raw full XGBoost pseudo-Huber | M2R | -0.002337 | [-0.005231, 0.000514] | 0.561641 | 0.065436 | 9.29% | 1.4424 |

Stage 2 的具体判断：structured block 有弱信号；raw embedding 单独线性建模没有实用收益；raw 414D 加 XGBoost 很差；fold-local PCA32 加 Ridge 和 late fusion 在 screen 中最好；分别预测两种动作的 absolute utility 没有优于直接预测 gap。

### 12.8 Stage 3：训练目标、损失与重复噪声诊断

| Candidate / target | Gain | 95% confidence interval | Coverage | Harmful/beneficial | Non-tie AUC | Spearman | 参与模型选择 |
|---|---:|---|---:|---:|---:|---:|---|
| M0 squared-error continuous gap | +0.002411 | [0.000035, 0.004853] | 4.94% | 0.5503 | 0.589457 | 0.095322 | 是 |
| M0 pseudo-Huber continuous gap | +0.001354 | [-0.001657, 0.004344] | 8.38% | 0.8075 | 0.584329 | 0.090781 | 是 |
| M2 squared-error continuous gap | -0.001651 | [-0.003598, 0.000237] | 3.35% | 1.8217 | 0.545468 | 0.049199 | 是 |
| M2 pseudo-Huber continuous gap | -0.002337 | [-0.005231, 0.000514] | 9.29% | 1.4424 | 0.561641 | 0.065436 | 是 |
| M3 PCA16 Huber | -0.000155 | [-0.001656, 0.001344] | 1.85% | 1.0955 | 0.520486 | 0.022761 | 是 |
| M3 PCA32 Huber | -0.000206 | [-0.001707, 0.001293] | 1.75% | 1.1306 | 0.523368 | 0.025174 | 是 |
| M3 PCA64 Huber | +0.000779 | [-0.001108, 0.002680] | 3.02% | 0.7494 | 0.522218 | 0.024481 | 是 |
| Weighted-sign raw-full XGBoost | +0.000700 | [-0.002268, 0.003695] | 7.85% | 0.8999 | 0.558721 | 0.057454 | 否，仅诊断 |

PCA plus Huber、weighted sign 与 M5 dual utility heads 均未稳定改善，因此 continuous BM25-minus-Dense gap 与 squared error 继续作为正式主目标。Exact ties 保留在 regression；weighted sign classification 跳过 exact ties。

重复噪声结果：BM25/Dense repeat variance mean=`0.007613/0.005930`；零方差 queries=`4,479/4,524`；4,271/4,800 queries 的两个动作都为零方差。Gap standard error mean/median/90th percentile/maximum=`0.018482/0/0.031596/0.471405`；3×3 cross-repeat soft preference 对 non-tie winner 方向准确率 `0.934003`；4/9–5/9 ambiguous queries 11；paired-repeat winner 与三重复均值 winner 平均一致率 `0.859397`，三个 paired repeats 全一致 999。标签存在噪声，但不足以解释 Router 接近零的收益；每个动作只有三次重复，因此不使用不稳定 inverse-variance weighting。

一个不影响正式结论的 tolerance-level 口径差异必须如实记录：正式 winner strata 用 `|gap|<=1e-12` 判 tie，因此 4,800 formal non-ties 为 596+797=`1,393`；repeat diagnostic 使用 `np.sign(gap)!=0`，artifact 记录 `non_tie_queries=1,394`。差异仅涉及一个极小 gap query 的噪声诊断计数，不影响正式 strata、out-of-fold predictions、policy 或 gate。

### 12.9 Stage 4：随机正交旋转诊断

三个固定 rotation seeds 为 20261011、20261023、20261037。正交旋转保持 inner product、cosine similarity 和 Euclidean distance，只改变 embedding 坐标轴。

| Rotation | Candidate | Gain | Non-tie AUC | Spearman | Coverage | Harmful/beneficial |
|---|---|---:|---:|---:|---:|---:|
| Unrotated | Raw embedding XGBoost | -0.000420 | 0.547814 | 0.047655 | 10.69% | 1.0492 |
| 20261011 | Raw embedding XGBoost | -0.000409 | 0.508955 | 0.011609 | 1.06% | 1.3802 |
| 20261023 | Raw embedding XGBoost | -0.001787 | 0.518529 | 0.021393 | 3.65% | 1.7268 |
| 20261037 | Raw embedding XGBoost | -0.001490 | 0.518244 | 0.015794 | 3.44% | 1.7265 |
| Unrotated / all rotations | Embedding Ridge | -0.001194 | 0.531557 | 0.034356 | 5.02% | 1.3279 |
| Unrotated | PCA32 plus structured Ridge | +0.003579 | 0.552653 | 0.061570 | 13.50% | 0.6769 |
| 20261011 | PCA32 plus structured Ridge | +0.002580 | 0.553245 | 0.062241 | 13.85% | 0.7593 |
| 20261023 | PCA32 plus structured Ridge | +0.002770 | 0.552750 | 0.061474 | 13.65% | 0.7550 |
| 20261037 | PCA32 plus structured Ridge | +0.003872 | 0.552761 | 0.061852 | 13.46% | 0.6577 |

Raw embedding tree 的 AUC 因坐标旋转明显下降；Ridge 完全旋转不变；PCA32 plus Ridge 相对稳定。因此 raw tree 依赖任意 axis-aligned coordinates，model–embedding fit 不合适。Stage 4 只作机制诊断，不参与 champion selection。

### 12.10 Stage 5：连续集成、校准与阈值修正

旧流程会先将每个 seed 二值化再 majority vote，丢失置信度。Stage 5 改为先平均连续 predicted gaps，再用 inner-out-of-fold prediction 拟合 nonnegative affine calibrator，最后固定零阈值。五-seed M2 在 4,800 screen 上经修正后 Router F1 `0.647952`、gain `+0.000817`、confidence interval `[-0.002640,+0.004415]`；554 switches、coverage `11.54%`、beneficial/neutral/harmful=`73/412/69`、missed beneficial 523、beneficial/harmful mass=`0.009402/0.008585`、ratio `0.913064`、AUC `0.567744`、Spearman `0.069340`、calibration slope `1.068473`、top-decile gap `+0.015933`。因此校准流程得到修正，但信号不足仍未解决。

这里必须区分两个 4,800-query 数字口径：Stage 2 screen 表中的 M2 gain `-0.001651`、M0 gain `+0.002411` 只使用单一 XGBoost model seed `11`；Stage 5 和 Stage 8 的 split `20260901` 结果使用五个 model seeds 的连续分数集成，因此 M2/M0 gain 分别变为 `+0.000817/+0.000626`。它们是不同 ensemble 口径，不是同一实验数字互相矛盾。M3 Ridge 与 M4 的相应首 split 数字不依赖多 XGBoost seed ensemble 或在该设置下保持等价，所以没有相同幅度的口径变化。

### 12.11 Stage 6：失败机制评价

Stage 6 不只报告平均 gain，还同时报告 beneficial、neutral 与 harmful switch 数；beneficial 与 harmful mass；missed BM25-beneficial queries；policy regret；oracle headroom recovery；non-tie AUC；Spearman；coverage；calibration intercept、slope 与 deciles；outer-fold 和 split-seed variance。Retrieval Recall/Hit 始终与 Answer F1 utility 分开。

4,800 screen 的代表性分解：M4 Router F1/gain=`0.650756/+0.003622`，614 switches，B/N/H=`87/461/66`，missed=509，beneficial/harmful mass=`0.011371/0.007749`，ratio `0.681487`，oracle recovery `5.12%`；M3 为 `0.650713/+0.003579`，648 switches，`89/493/66`，missed 507，ratio `0.676925`，recovery `5.06%`；M0 gain `+0.002411`、237 switches、`36/178/23`、ratio `0.550287`；M2 gain `-0.001651`、161 switches、`17/115/29`、ratio `1.821681`。多数 switches 落在 tie，missed beneficial 很多，错误切换的质量损失接近或超过正确收益。

### 12.12 Stage 7：候选冻结与 9,600-query 正式确认

Stage 7.1 在新 9,600 标签可见前冻结四个正式候选：

1. `M2_raw_full_xgb_mse`：raw 414D plus XGBoost 复现控制。
2. `M3_pca32_structured_ridge`：PCA32 embedding plus structured plus Ridge。
3. `M4_oof_late_fusion`：structured 与 embedding 父模型的 inner-out-of-fold late fusion。
4. `M0_structured_xgb_mse`：structured-only XGBoost。

Stage 2–6 在研究逻辑上是不同问题，但计算上共享同一 quick-screen out-of-fold prediction：模型预测是 target、calibration、rotation 和 failure decomposition 的共同前提。共享计算不等于逻辑阶段已完成；必须分别生成并接受 Stage 2–6 reports 后，才能冻结候选并进入 Stage 7.2。历史执行曾在各独立报告接受前过早启动 formal：只有 M2 split `20260901` 的完整 task 可复用，`20260917` 中断 task 被排除；修正依赖顺序后 12 个 formal tasks 全部完成，任何 partial output 都未进入最终指标。

Stage 7.2 执行 4 candidates × 3 split seeds，共 12 个 candidate/split tasks；每个 task 有 5 outer folds。

| Candidate | Split 20260901 gain | Split 20260917 gain | Split 20261003 gain | Mean split gain | Consensus gain | 95% confidence interval |
|---|---:|---:|---:|---:|---:|---|
| M2 raw 414D XGBoost | +0.000828 | -0.000732 | +0.000931 | +0.000342 | -0.000690 | [-0.002778,+0.001387] |
| M3 PCA32 plus Ridge | +0.002792 | +0.002193 | +0.002248 | +0.002411 | +0.002455 | [-0.000083,+0.005069] |
| M4 late fusion | +0.002244 | +0.001126 | +0.000582 | +0.001317 | +0.001602 | [-0.001212,+0.004468] |
| M0 structured XGBoost | +0.000104 | -0.000250 | -0.000875 | -0.000340 | -0.000349 | [-0.002580,+0.001857] |

Mean split gain 是三个独立 split policy 的 gain 均值；consensus gain 是合并三套连续 out-of-fold predictions 后形成的单一 consensus policy。

最强 M3 consensus：Router F1 `0.650664`；switch queries 1,558/9,600；coverage `16.23%`；beneficial/neutral/harmful 为 186/1,210/162；missed BM25-beneficial queries 1,010；beneficial gain mass `0.011645`；harmful loss mass `0.009190`；harmful/beneficial `0.789`；non-tie AUC `0.580`；gap Spearman `0.086`；oracle headroom recovery `3.54%`。

四个 consensus policy 的完整 failure profile：

| Candidate | Router F1 | Switches | B/N/H | Harm/benefit | Oracle recovery | AUC | Spearman | Slope | Top decile gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 raw 414D XGBoost | 0.647520 | 953 | 99/724/130 | 1.109277 | -0.994% | 0.562242 | 0.067769 | 1.114992 | -0.006298 |
| M3 PCA32 Ridge | 0.650664 | 1,558 | 186/1,210/162 | 0.789190 | 3.539% | 0.580132 | 0.086367 | 1.156244 | +0.013594 |
| M4 late fusion | 0.649811 | 1,683 | 210/1,275/198 | 0.876056 | 2.309% | 0.570655 | 0.077523 | 1.052261 | -0.000916 |
| M0 structured XGBoost | 0.647860 | 1,230 | 121/973/136 | 1.046913 | -0.503% | 0.573586 | 0.078540 | 1.181236 | -0.008263 |

M3 的 conditional gain over all switches `0.015126`、conditional beneficial gain `0.601023`、conditional harmful loss `-0.544591`、high-margin BM25 recall `0.159624`、policy regret `0.066901`、gap mean absolute/root-mean-square error=`0.187360/0.362531`。这些指标说明它不是随机，但离安全策略仍很远。

### 12.13 Stage 8：训练规模学习曲线

| Candidate | 1,200 | 2,400 | 4,800 | 9,600 | 9,600−4,800 |
|---|---:|---:|---:|---:|---:|
| M2 raw XGBoost | -0.001211 | -0.000229 | +0.000360 | +0.000342 | -0.000018 |
| M3 PCA32 Ridge | -0.000432 | +0.000207 | +0.001091 | +0.002411 | +0.001320 |
| M4 late fusion | 0.000000 | +0.000275 | +0.001827 | +0.001317 | -0.000510 |
| M0 structured XGBoost | -0.001181 | +0.000384 | +0.000341 | -0.000340 | -0.000681 |

上述均值背后的完整三 split gains 为：

| N | Candidate | 20260901 | 20260917 | 20261003 | Mean |
|---:|---|---:|---:|---:|---:|
| 1,200 | M2 | -0.003632 | 0 | 0 | -0.001211 |
| 1,200 | M3 | -0.000667 | 0 | -0.000630 | -0.000432 |
| 1,200 | M4 | 0 | 0 | 0 | 0 |
| 1,200 | M0 | +0.000401 | -0.000954 | -0.002991 | -0.001181 |
| 2,400 | M2 | 0 | -0.000417 | -0.000271 | -0.000229 |
| 2,400 | M3 | +0.000445 | +0.001430 | -0.001253 | +0.000207 |
| 2,400 | M4 | -0.001634 | +0.001306 | +0.001153 | +0.000275 |
| 2,400 | M0 | -0.000659 | -0.000028 | +0.001838 | +0.000384 |
| 4,800 | M2 | +0.000817 | -0.000782 | +0.001046 | +0.000360 |
| 4,800 | M3 | +0.003579 | -0.001682 | +0.001377 | +0.001091 |
| 4,800 | M4 | +0.003622 | -0.000136 | +0.001996 | +0.001827 |
| 4,800 | M0 | +0.000626 | +0.001854 | -0.001458 | +0.000341 |
| 9,600 | M2 | +0.000828 | -0.000732 | +0.000931 | +0.000342 |
| 9,600 | M3 | +0.002792 | +0.002193 | +0.002248 | +0.002411 |
| 9,600 | M4 | +0.002244 | +0.001126 | +0.000582 | +0.001317 |
| 9,600 | M0 | +0.000104 | -0.000250 | -0.000875 | -0.000340 |

只有 M3 从 4,800 到 9,600 继续改善，但 9,600 mean split gain 仍只有 `+0.002411`，约为 practical threshold 的四分之一。M2 平台化，M4 与 M0 下降。该曲线否定“继续增加同类型 query-only 数据与同一候选即可解决”的解释，但不否定未测试的新信息边界或全新架构。

### 12.14 Stage 9：正式 advancement gate

M3 虽然三个 split gain 均为正，coverage 合格，top-decile 方向也有信号，但失败于：gain 不到 `+0.01`、confidence interval 下界不大于 0、harmful/beneficial `0.789` 高于 `0.5`。其他候选失败项更多。正式决策为：

| Candidate | Gain>=0.01 | CI lower>0 | 3 splits positive | Harm ratio<=0.5 | Coverage | Slope | Top decile | All pass |
|---|---|---|---|---|---|---|---|---|
| M2 | Fail | Fail | Fail | Fail | Pass | Pass | Fail | No |
| M3 | Fail | Fail | Pass | Fail | Pass | Pass | Pass | No |
| M4 | Fail | Fail | Pass | Fail | Pass | Pass | Fail | No |
| M0 | Fail | Fail | Fail | Fail | Pass | Pass | Fail | No |

`STOP_QUERY_ONLY_V1`

没有 candidate 被冻结为可部署 Router。

### 12.15 Stage 10 与 Stage 11：因前置 gate 失败而不适用

Stage 10 原计划在 formal gate 通过和用户单独授权后冻结完整模型、prompt、schema、transforms、ensemble、calibrator、threshold 与 evaluator，再一次性打开 fresh development，禁止打开后重选或调参。

Stage 11 原计划只有在 fresh-development F1 也通过后，才购买 Answer Correctness labels，验证 frozen F1 policy 是否转移到语义正确性。

由于 Stage 9 失败：没有创建 `model_freeze.json`；fresh-development rows read 为 0；final-holdout rows read 为 0；Answer Correctness labels 未购买；未进行 fresh-development 调参或选择。这不是“工作尚未做完”，而是冻结逻辑判定这些阶段不适用。

### 12.16 Stage 12：qrels-free post-retrieval probe 与 privileged gold diagnostic

Stage 12 只在 query-only 全部失败后进入。它没有进行 Teacher-to-Student distillation；P1、G1 与 M3 是用各自信息块独立训练并比较的诊断模型。

Qrels-free probe 共 50 个特征，包括：BM25 与 Dense Top-1/2/3/5 scores；score mean、standard deviation、range、Top-1/2 和 Top-1/5 margin；relative margin；coefficient of variation；rank slope；normalized entropy；context token count；truncation；query-context Jaccard；两列表 Top-1 equality；Top-1/3/5 overlap；Jaccard；reciprocal-rank overlap；以及两 Retriever 对应统计差。它们需要先运行两个 Retriever，因此是 post-retrieval，不是 pre-retrieval。

Gold block 共 34 个特征，包括两 Retriever 的 evidence coverage at 1/2/3/5、complete coverage、hit at 5、first gold rank、mean reciprocal rank 以及动作间差值。Gold 特征依赖 qrels/supporting documents，永远不可作为部署输入。

候选是 P0 probe Ridge、P1 probe XGBoost、G0 gold Ridge、G1 gold XGBoost、G2 probe-plus-gold XGBoost。Ridge alpha 10 并标准化；Stage 12 XGBoost 使用 250 estimators、maximum depth 2、learning rate 0.03、minimum child weight 10、subsample/column sample 0.8、L2 10、L1 0.1、squared-error、histogram tree。仍使用相同 3 split seeds×5 outer×4 inner grouped out-of-fold protocol。

| Candidate / rule | 信息块 | Gain | 95% confidence interval | Coverage | Harmful/beneficial | Non-tie AUC |
|---|---|---:|---|---:|---:|---:|
| P0 probe Ridge | Qrels-free probe | +0.013439 | [+0.009464,+0.017491] | 35.1% | 0.580 | 0.656 |
| P1 probe XGBoost | Qrels-free probe | +0.014834 | [+0.010703,+0.018926] | 34.6% | 0.572 | 0.667 |
| G0 gold Ridge | Gold | +0.038390 | [+0.033399,+0.043573] | 65.6% | 0.372 | 0.805 |
| G1 gold XGBoost | Gold | +0.039265 | [+0.034225,+0.044398] | 62.6% | 0.358 | 0.801 |
| G2 probe plus gold XGBoost | Probe plus gold | +0.038447 | [+0.033538,+0.043442] | 58.2% | 0.347 | 0.809 |
| Gold coverage@1 rule | Gold | +0.010335 | [+0.007829,+0.012911] | 7.7% | 0.291 | 不适用 |
| Gold coverage@3 rule | Gold | +0.022458 | [+0.019396,+0.025637] | 9.4% | 0.095 | 不适用 |
| Gold coverage@5 rule | Gold | +0.026574 | [+0.023418,+0.029841] | 8.9% | 0.053 | 不适用 |

P1 通过 practical gain、confidence interval、三 split 稳定性、coverage、calibration 与 ranking，只因 harmful/beneficial `0.572` 高于 `0.5` 未过完整 safety gate；oracle recovery 为 `21.4%`。G1 oracle recovery 为 `56.6%`。

详细 policy profile：P0 split gains=`0.014016/0.012913/0.013074`，Router F1 `0.661648`，3,370 switches，B/N/H=`495/2,511/364`，missed 701，Spearman/AUC/slope/top-decile=`0.166864/0.655670/1.021083/0.069434`；P1 split gains=`0.014113/0.013083/0.015216`，Router F1 `0.663044`，3,319 switches，`541/2,419/359`，missed 655，high-margin recall `46.76%`，Spearman/AUC/slope/top-decile=`0.183312/0.666580/1.067150/0.081027`。

G0 split gains=`0.038582/0.038638/0.038126`，Router F1 `0.686599`，6,299 switches，`998/4,784/517`，AUC/Spearman/slope=`0.804589/0.345970/1.005677`；G1 split gains=`0.038761/0.039249/0.039109`，Router F1 `0.687475`，6,010 switches，`988/4,525/497`，missed 208，high-margin recall `83.57%`，AUC/Spearman/slope/top-decile=`0.800706/0.343101/1.008274/0.270681`；G2 split gains=`0.038836/0.038802/0.037592`，Router F1 `0.686657`，5,589 switches，`940/4,188/461`，AUC/Spearman/slope=`0.808907/0.351771/1.011731`。Gold models 只因 coverage 超过 50% 未过统一 policy gate；这不影响其作为 privileged information diagnostic 的作用。

Paired increments：P1 probe XGBoost minus query-only M3 为 `+0.012380`，95% confidence interval `[+0.008471,+0.016380]`；G1 gold XGBoost minus P1 probe XGBoost 为 `+0.024431`，区间 `[+0.020462,+0.028516]`。

Stage 12 决策 `GOLD_EVIDENCE_SIGNAL_SUFFICIENT` 的准确含义是：实际 retrieval behavior 与 evidence sufficiency 足以解释大量动作效用差；它不表示 gold teacher 可部署，也不表示已经完成蒸馏。Qrels-free probe 显著优于 query-only，说明下一步值得研究 shallow probe/cascade，但 P1 本身仍未通过 harmful-switch safety gate。

三条预注册 rule 的 B/N/H 与 oracle recovery：coverage@1=`195/453/87`、`14.90%`；coverage@3=`314/532/58`、`32.38%`；coverage@5=`353/459/41`、`38.31%`。Coverage@5 通过 Stage 12 diagnostic-rule criterion，但其 calibration slope 约 `0.436`，且输入本身需要 gold evidence，所以不能称为部署 gate 通过。

### 12.17 Phase 2.7 总结、局限与验证

Phase 2.7 得出以下正式判断：

1. 当前严格 pre-retrieval query-only Router 不可部署。
2. 最强 M3 只恢复约 3.5% oracle headroom，错误切换损失未受控。
3. Raw embedding plus XGBoost、直接 414D concatenation、单纯增加同类 query-only 数据都没有得到支持。
4. Continuous utility gap 与 leakage-safe out-of-fold evaluation 足以揭示模型不稳定性。
5. 相对当前 query-only Version 1，主要可增量预测信号出现在实际排名、scores、list overlap、context 与 evidence coverage；query wording 与 corpus-static 特征仍有弱的非零信号，但不足以通过策略 gate。P1、G1 与 M3 是不同信息块的独立 out-of-fold 诊断模型，因此这里是信息层比较，不是把同一模型逐项干预后的严格因果消融。
6. Gold diagnostic 只定位机制上界；qrels-free probe 是新的候选信息边界，但需要新的成本与安全协议。

适用范围限于 HotpotQA、旧 prompt、Top-5 context、GPT-4.1-mini 和 normalized token F1。没有测试完整 v2 prompt labels、multilayer perceptron、partial least squares、encoder fine-tuning 或新 embedding；没有 Answer Correctness 或 fresh-development 结果。Stage 12 需要两路 Retriever，不能直接声称节省完整检索成本。

Formal artifact/integrity validation 与 Stage 12 artifact/integrity validation 均为 `passed`，表示产物结构、预测数组、任务计数与数据完整性通过；这不表示 Phase 2.7 advancement policy gate 通过，该 gate 的正式结果仍是失败并触发 `STOP_QUERY_ONLY_V1`。Database integrity 为 `ok`；formal tasks 12；formal prediction arrays 24；learning-curve rows 48；Stage 12 candidate/split tasks 15；fresh-development/final rows 0/0；专项测试 12 passed、1 个不影响结果的 numexpr 版本 warning。

冻结 snapshot：outcomes 4,532,391 bytes、SHA-256 `846e2a43de8aabd9962e1e7fe16d8f6425823cff5954771ed583934a2dc831ad`；query summary 689,419 bytes、`31dbebe281c8d87039c723c7a78deddca00c56fe17d43274742c16750fc804b4`；features 14,634,125 bytes、`a0435bb5ab94b80099713e537c5b621363543cac606b02f4307ac0416c3afa2d`；feature schema `0007551aeb82e903d90672109ae88ba59c527742ca028a8059ff8084c4fd1ffa`；manifest `e1f984e58b063896f252ef98b6dbdac4a50a8ef724d69d48a1a4ebc508c467b5`。运行环境：Python 3.11.5、NumPy 1.26.4、SciPy 1.15.3、scikit-learn 1.9.0、XGBoost 3.0.5。

## 13. Phase 2.8：winner query 扩充、T0/T1/T2 训练视图与候选重训

实验执行时间为 2026-08-27 至 2026-08-29。

### 13.1 研究问题

Phase 2.7 表明 query-only M3 在自然 train-only 9,600 queries 上只有约 `+0.0024` F1，而 BM25/Dense 非 tie winner 稀疏。Phase 2.8 不再同时搜索大量结构，而是回答两个限定问题：

1. 能否用 retrieval evidence 只作为定向采样工具，在不把它输入 Router、也不使用 sample weighting 的前提下，扩充 BM25 winner 和 Dense winner？
2. 用扩充数据重训已冻结的 M3、M4、M4-PCA 后，答案效用的严格 out-of-fold 信号是否随 winner 数增加而稳定改善？

监督目标仍为每个 query 两动作各三次生成的 normalized token F1 均值差：

\[
y(q)=\overline{F1}_{BM25}(q)-\overline{F1}_{Dense}(q).
\]

正值表示 BM25 的答案效用更高，负值表示 Dense 更高，绝对值不超过 `1e-12` 的值按 exact tie 处理。Retrieval recall 差只决定先标注哪些 query，最终 winner 只由答案 F1 决定。

### 13.2 冻结边界和 confirmation 预留

- 新 acquisition candidate pool 与预留 confirmation pool 的数据源为 HotpotQA train，并排除所有 Phase 2.7 已使用 information-need groups；该排除只约束“新池”，不是丢弃 Phase 2.7 base cohort。
- 在任何新 retrieval 或 generation 之前，用冻结随机种子从新 groups 中预留 2,000-query confirmation，每个 group 最多一条。
- 预留集不进入 acquisition、特征提取、训练或 Phase 2.8 正式 out-of-fold 指标。
- Pool seeds：`2026082701`、`2026082702`、`2026082703`；retrieval block size 1,000；Dense query batch size 100。
- Acquisition strata：`bm25_plus`、`dense_plus`、`equal_zero_overlap`、`equal_low_overlap`、`equal_other`。
- 每个 primary stratum 初始 pilot 500；期望 winner rate 0.25；单批最多 2,000；定向 acquisition 最多 15,000。
- Generation 硬预算 20 美元；实际估算 11.3226844 美元。
- High-margin winner 定义为 `|F1 gap|>=0.10`。
- 总配额：BM25 winner 至少 3,000、Dense winner 至少 3,000；BM25 和 Dense high-margin winner 各至少 2,500。
- Phase 2.7 base cohort 仍作为 T0/T1/T2 的旧数据来源：BM25 winner 1,196、Dense winner 1,564、tie 6,840；其中 high-margin winner 为 1,065 和 1,433。新池与该 base cohort 保持 group-disjoint，随后两者按冻结规则合并构造训练视图。

允许的 Router 输入仍只有 query text transformations、冻结 BGE query embedding、静态 corpus lexical statistics 和冻结 corpus prototype summaries。当前 query 的 retrieval documents、scores、ranks、qrels、gold supporting facts、acquisition stratum、生成答案和答案分数均禁止进入模型。

### 13.3 定向扩充的实际执行数据

| 项目 | 实际结果 |
|---|---:|
| 完成 BM25/Dense paired retrieval 的 queries | 48,866 |
| 未检索备用池 | 6,069 |
| 进入答案生成的 queries | 7,885 |
| 成功 generation outcomes | 47,310 |
| Provider failure | 0 |
| 新 BM25 winners | 2,020 |
| 新 Dense winners | 1,502 |
| 新 ties | 4,363 |
| 合并后 BM25 winners | 3,216 |
| 合并后 Dense winners | 3,066 |
| 合并后 ties | 11,203 |
| 合并后 BM25 high-margin winners | 2,986 |
| 合并后 Dense high-margin winners | 2,840 |

两条主要 acquisition lane 的效率为：

| Lane | Queries | BM25 winner | Dense winner | Tie |
|---|---:|---:|---:|---:|
| `bm25_plus` | 4,185 | 1,806 | 219 | 2,160 |
| `dense_plus` | 3,700 | 214 | 1,283 | 2,203 |

因此 evidence-recall 差有效提高了获取目标 winner 的效率，但 lane 仍包含大量 tie 和反向 winner；它只是 sampling heuristic，不是答案效用 label。

### 13.4 三个嵌套训练视图

Tie 均从旧 Phase 2.7 数据中用冻结随机顺序无放回选择 6,000 条；winner 按冻结 acquisition 顺序逐步加入：

| 训练视图 | BM25 winners | Dense winners | Ties | 总 queries | 新增 winner features |
|---|---:|---:|---:|---:|---:|
| T0 tie-capped | 1,196 | 1,564 | 6,000 | 8,760 | 0 |
| T1 winner-2,000 | 2,000 | 2,000 | 6,000 | 10,000 | T0 以外配额所需部分 |
| T2 winner-3,000 | 3,000 | 3,000 | 6,000 | 12,000 | 累计 3,240 |

未进入 quota 的新增 ties 和超出 3,000/3,000 的 winners 不进入这些冻结训练视图。三种视图既改变样本量，也改变 action prior 和 tie 比例，因此它们不是纯样本量 learning curve。

### 13.5 候选模型与 leakage-safe 训练

三个候选在扩充 outcome 生成前已经冻结：

1. `M3_pca32_structured_ridge`：30 维 structured features 经标准化，与只在 outer-train 内拟合的 BGE query embedding 主成分分析前 32 维拼接，Ridge 回归预测连续 F1 gap。
2. `M4_oof_late_fusion`：structured 分支使用 regularized XGBoost preset，384 维 raw BGE query embedding 分支使用 Ridge；只用 inner out-of-fold branch predictions 拟合 Ridge late-fusion 元模型和校准。
3. `M4_pca32_oof_late_fusion`：structured 分支不变，embedding-related branch 改用 fold-internal PCA32 plus structured Ridge；仍以 inner out-of-fold predictions 训练 late fusion，检验紧凑表示能否替代 raw 384 dimensions。

训练不使用 sample weights。每个视图执行 3 个 split seeds × 5 outer folds × 4 inner folds。Scaler、PCA、branch model、late fusion 和 calibration 都只在对应 fold 的训练部分拟合。三个 split 的连续 out-of-fold predictions 取均值形成 consensus；95% confidence interval 用 information-need group 为单位 bootstrap 10,000 次。

冻结 gate 同时要求：gain 至少 `+0.01`、confidence interval 下界大于 0、三个 split gain 全正、harmful/beneficial mass ratio 不超过 `0.5`、switch coverage 在 `1%–50%`、calibration slope 在 `0.5–1.5`、最高预测 decile 的 realized gap 大于 0。

### 13.6 九个正式 out-of-fold 结果

| View | Candidate | Consensus Router F1 | Gain over fixed Dense | 95% confidence interval | Coverage | Beneficial/Harmful switches | Harmful/beneficial mass | Non-tie AUC | Gap Spearman | Gate failures |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| T0 | M3 | 0.645349 | +0.002623 | [-0.000189,0.005427] | 17.17% | 205/179 | 0.803 | 0.585 | 0.096 | gain, interval, harm |
| T0 | M4 | 0.645289 | +0.002562 | [-0.000339,0.005481] | 15.26% | 185/176 | 0.809 | 0.567 | 0.078 | gain, interval, harm |
| T0 | M4-PCA | 0.645728 | +0.003002 | [-0.000008,0.005987] | 17.55% | 202/179 | 0.786 | 0.584 | 0.094 | gain, interval, harm |
| T1 | M3 | 0.633614 | +0.019587 | [0.013978,0.024935] | 44.80% | 939/660 | 0.664 | 0.594 | 0.108 | harm |
| T1 | M4 | 0.635530 | +0.021503 | [0.015541,0.027401] | 49.60% | 1,067/787 | 0.681 | 0.599 | 0.116 | harm |
| T1 | M4-PCA | 0.636926 | +0.022898 | [0.016955,0.028597] | 47.95% | 1,033/718 | 0.647 | 0.598 | 0.113 | harm |
| T2 | M3 | 0.630801 | +0.040241 | [0.033561,0.046722] | 52.50% | 1,827/1,204 | 0.608 | 0.638 | 0.176 | harm, coverage |
| T2 | M4 | 0.632839 | +0.042278 | [0.035421,0.048999] | 55.87% | 1,954/1,345 | 0.618 | 0.639 | 0.185 | harm, coverage |
| T2 | M4-PCA | 0.633298 | +0.042738 | [0.035896,0.049432] | 55.62% | 1,935/1,307 | 0.610 | 0.637 | 0.178 | harm, coverage |

完整 split、neutral switch、calibration 和 top-decile 补充如下：

| View/model | Split gains 20260901 / 20260917 / 20261003 | Neutral switches | Calibration slope | Top-decile realized gap |
|---|---|---:|---:|---:|
| T0 M3 | 0.003578 / 0.003730 / 0.001757 | 1,120 | 1.1553 | 0.02090 |
| T0 M4 | 0.003961 / 0.001108 / 0.000328 | 976 | 1.1377 | 0.02142 |
| T0 M4-PCA | 0.002935 / 0.003276 / 0.000404 | 1,156 | 1.1254 | 0.01124 |
| T1 M3 | 0.018917 / 0.018390 / 0.019071 | 2,881 | 1.0825 | 0.05258 |
| T1 M4 | 0.020955 / 0.019737 / 0.019034 | 3,106 | 1.0405 | 0.06211 |
| T1 M4-PCA | 0.023231 / 0.020356 / 0.018940 | 3,044 | 1.0567 | 0.05765 |
| T2 M3 | 0.041225 / 0.040802 / 0.038746 | 3,269 | 1.0544 | 0.11127 |
| T2 M4 | 0.041932 / 0.041379 / 0.042761 | 3,405 | 1.0232 | 0.14769 |
| T2 M4-PCA | 0.041027 / 0.042691 / 0.040957 | 3,432 | 1.0279 | 0.11673 |

各视图固定动作与 oracle 也随构成改变：T0 fixed BM25/Dense/oracle 为 `0.612004/0.642726/0.718733`；T1 为 `0.608489/0.614028/0.731767`；T2 为 `0.593758/0.590560/0.750635`。T2 中 BM25 已略高于 Dense，且 oracle headroom 相对 Dense 达 `0.160074`。因此 T2 的 `+0.043` 不能外推为自然分布增益。

### 13.7 观察、决策与局限

T0 复现 Phase 2.7 的弱小正方向，但没有实用或显著增益。T1/T2 在其人为 winner-rich 分布上出现强 out-of-fold 信号，支持“winner 稀缺是限制因素之一”这一解释；然而 T0/T1/T2 同时改变样本量、winner/tie 比例、动作先验、best fixed 和 oracle headroom，因此不能把增益因果归因于 winner 数量这一项。Coverage 与错误切换质量仍失败。模型结构差异远小于训练视图差异：M4-PCA 在三个 consensus 点估计上略高，但 T2 的 split-gain 简单均值又是 M4 略高，不能宣称 PCA32 显著优于 raw embedding。

九个组合没有一个通过全部七项 gate，正式决策为 `DO_NOT_OPEN_CONFIRMATION_POOL`。Phase 2.8 原预留 confirmation rows read 0；fresh-development rows 0；final-holdout rows 0。后来执行的 Phase 2.8b 是另行冻结的协议修订，不能倒写成 Phase 2.8 已按原 gate 晋级。

扩充阶段的一次旧 Phase 2.7 screen 被误触发；只完成一个无关 M0 partial baseline 后中断，未进入候选选择、表格或结论。Phase 2.8 聚焦测试 5 passed；三个视图的 preflight 和正式任务均完成。

## 14. Phase 2.8b：natural-2,000 独立候选选择

### 14.1 研究问题与数据角色

Phase 2.8b 于 2026-08-29 执行，回答：Phase 2.8 的 9 个固定“训练视图 × 模型”候选能否泛化到在 acquisition 之前预留的自然分布 2,000 queries？

该集合的正式角色是 candidate-selection holdout，不是 internal final holdout，更不是官方 HotpotQA test。它打开后即被消费，后续只能作为 post-selection diagnostic 或预先规定的自然复现集合，不能重新调参后再称未见验证。

### 14.2 独立性、冻结和运行设计

- Queries/groups：2,000/2,000；与 acquisition groups 和最大 T2 training groups 的交集均为 0。
- Sample SHA-256：`91d3c71b5b82eab336e3abc2c6b2862497e5e045cb6bf72f61a30d85efbbe58b`。
- Protocol config SHA-256：`2394ca0408426d6b3d8cfd5cbeef998171d1c7252e237a3f76c398ad1870bfcc`。
- 9×2,000 predictions 在 generation labels 出现前冻结；prediction NPZ SHA-256：`1601abf8e94af95a8710cb4a27e30d2e7672347c3433bafdfbdf629a819f914e`。
- 每个候选以对应 Phase 2.8 训练视图 full fit；replica seeds 为 `20260901`、`20260917`、`20261003`。
- M4/M4-PCA 的 base branches、meta fusion 和 calibration 只使用训练集内部四折 out-of-fold predictions。
- Full-fit model seed 为 11；M3 base Ridge alpha 10，M4/M4-PCA meta Ridge alpha 1；三个 replica 的差异来自固定 replica split seeds 与各自训练内 calibration。
- Router threshold 固定为 0；三个 replica continuous predictions 取均值。
- Retrieval：每条 query 各运行 BM25/BGE Top-50；generation 使用各自 Top-5、1,800-token context。
- Generator：`gpt-4.1-mini-2025-04-14`，temperature 0，最大 64 tokens，每 query/action 3 repeats。
- Generation：12,000 success，pending/failure 0；4 workers；实际估算 2.842166 美元，低于 5 美元硬预算。

Phase 2.8b 不是原样复制 Phase 2.8 gate。它在 natural-2,000 上以该集合的 best fixed action 为参照，deployable gate 要求 gain 至少 `+0.01`、group-bootstrap interval 下界大于 0、harmful/beneficial 不超过 `0.5`、coverage 位于 `0.01–0.50`、calibration slope 位于 `0.5–1.5`，且 top predicted-gap decile 的 realized gap 为正；Phase 2.8b config 没有 Phase 2.8 的“每个 split/replica gain 都为正”条款。若没有部署候选，放宽 research-shortlist gate 仍要求 gain 至少 `+0.01`、interval 下界大于 0、harmful/beneficial 不超过 `0.75`、coverage 不超过 `0.60`、non-tie AUC 至少 `0.55`。

### 14.3 Natural-2,000 数据分布

- BM25 winners 267，占 13.35%；Dense winners 337，占 16.85%；exact ties 1,396，占 69.80%。
- High-margin winners：BM25 238、Dense 311。
- Fixed BM25 F1 `0.604848`；fixed Dense `0.636340`；Dense 优势 `0.031492`。
- Query-wise oracle F1 `0.707624`；相对 fixed Dense headroom `0.071284`。

### 14.4 九个候选的完整结果

| Robust rank | View/model | Router F1 | Gain over Dense | 95% confidence interval | Coverage | Beneficial/neutral/harmful | Harmful/beneficial mass | AUC | Deploy/research gate |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 1 | T0 M4 | 0.639927 | +0.003587 | [-0.001053,0.008401] | 11.70% | 37/176/21 | 0.621 | 0.568 | fail/fail |
| 2 | T0 M4-PCA | 0.640082 | +0.003742 | [-0.001443,0.009284] | 16.25% | 38/255/32 | 0.675 | 0.581 | fail/fail |
| 3 | T0 M3 | 0.640460 | +0.004120 | [-0.002031,0.010758] | 17.80% | 48/270/38 | 0.728 | 0.580 | fail/fail |
| 4 | T2 M3 | 0.634777 | -0.001563 | [-0.012615,0.009853] | 53.05% | 144/772/145 | 1.039 | 0.575 | fail/fail |
| 5 | T1 M3 | 0.633650 | -0.002690 | [-0.012883,0.007574] | 46.20% | 121/682/121 | 1.081 | 0.570 | fail/fail |
| 6 | T1 M4-PCA | 0.633236 | -0.003104 | [-0.013753,0.007649] | 48.50% | 134/703/133 | 1.086 | 0.578 | fail/fail |
| 7 | T2 M4 | 0.633411 | -0.002928 | [-0.014108,0.008384] | 53.60% | 150/770/152 | 1.073 | 0.562 | fail/fail |
| 8 | T2 M4-PCA | 0.632529 | -0.003811 | [-0.015363,0.007803] | 55.10% | 153/794/155 | 1.090 | 0.578 | fail/fail |
| 9 | T1 M4 | 0.631149 | -0.005191 | [-0.015885,0.005622] | 48.45% | 131/697/141 | 1.146 | 0.561 | fail/fail |

| View/model | Gap Spearman | Calibration slope | Top-decile realized gap | Oracle recovery |
|---|---:|---:|---:|---:|
| T0 M4 | 0.077104 | 1.310100 | 0.027745 | 0.050321 |
| T0 M4-PCA | 0.087646 | 1.124371 | 0.029632 | 0.052491 |
| T0 M3 | 0.086720 | 1.059514 | 0.026733 | 0.057799 |
| T2 M3 | 0.082373 | 0.404948 | 0.014424 | -0.021931 |
| T1 M3 | 0.078432 | 0.704706 | 0.027795 | -0.037732 |
| T1 M4-PCA | 0.083934 | 0.698833 | 0.019345 | -0.043537 |
| T2 M4 | 0.070074 | 0.345575 | 0.012543 | -0.041082 |
| T2 M4-PCA | 0.084888 | 0.382926 | 0.005408 | -0.053461 |
| T1 M4 | 0.070593 | 0.644303 | 0.017006 | -0.072822 |

排序是预注册 robust diagnostic ranking，优先考虑 confidence interval 下界和安全性，不是按 Router F1 点估计排序。T0 M3 点估计最高，但 interval 下界和 harm ratio 更差，所以不进入 top two。

### 14.5 观察、正式决策和数据消费边界

T0 的 Phase 2.8 out-of-fold gain `+0.00256–+0.00300` 在 natural-2,000 上复现为 `+0.00359–+0.00412`，方向重复但不显著、也低于实用门槛。T1/T2 的内部 `+0.0196–+0.0427` 反转为负，coverage 约 46%–55%、harm ratio 大于 1，表明 winner-balanced 训练改变 action prior，使模型在自然分布中过度切向 BM25。M3、M4、M4-PCA 的同视图差异小于视图间差异。

正式决策为 `NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY`：

- 部署候选：无；在已评估的 natural-2,000 样本中，fixed Dense 是 best fixed baseline，因此研究决策保留它作为非 Router 对照；这不表示 internal final holdout 或生产切换已经验证。
- 主要诊断候选：T0 M4。
- 次要诊断候选：T0 M4-PCA。
- 简单对照：T0 M3。
- T1/T2 全部停止推进。
- Internal final holdout rows read 0。

该 natural-2,000 已标记 `consumed_candidate_selection=true`，后续复用必须明确写作已消费数据，不能在其上继续筛特征、组合或阈值后报告独立泛化。聚焦测试 17 passed，另有一个不影响结果的 pandas/numexpr warning。

## 15. Phase 2.9：207 个原始 query 与 query–collection 单特质审计

### 15.1 研究问题与严格限制

Phase 2.9 于 2026-08-30 执行。研究问题是：在不做 query rewriting、不训练组合模型、不使用当前 ranked results、Top-k documents、scores、ranks、qrels、gold evidence 或生成答案作为输入的条件下，是否存在一个原始 query 或 query-conditioned frozen-corpus trait，能稳定关联 BM25-versus-Dense 答案 F1 gap？

该阶段只做单变量统计审计；任何组合、interaction、threshold policy 或 Router training 都被禁止。通过只意味着该 trait 可以进入另行冻结的最小增量诊断，不意味着它本身产生 policy gain。

### 15.2 Discovery、old/new 和 natural 数据

- Discovery：T2 的 6,000 non-ties，BM25/Dense 各 3,000；5,991 groups，其中 9 groups 各含两条 query。
- Old cohort：来自 Phase 2.7 的 BM25 1,196、Dense 1,564。
- New cohort：Phase 2.8 新加入的 BM25 1,804、Dense 1,436。
- Natural confirmation：已消费的 2,000 queries；winner endpoint 只使用 604 non-ties，continuous gap endpoint 使用全 2,000。
- Natural role：`previously_consumed_natural_confirmation_not_final_holdout`；internal final rows 0。

Discovery 是人为 50/50 winner-balanced，因此必须同时通过 old/new direction stability 和 natural confirmation，不能只依赖 discovery 效应。

### 15.3 207 个冻结特质的构成

矩阵为 6,000×207 和 2,000×207，finite fraction 均为 1。207 个特质中 180 个 continuous、27 个 binary；按 18 个 family 完整计数如下：

| Feature family | Count | 信息来源 |
|---|---:|---|
| Historical legacy lexical | 17 | 旧 17D query/corpus lexical block |
| Historical dense prototype | 13 | 冻结 corpus prototype similarities |
| BM25-aligned unique lexical | 17 | 去重 term 后与实际 analyzer 对齐的 lexical block |
| Stage-C collection | 16 | collection frequency 与 inverse document frequency |
| Stage-C scope | 11 | 冻结 corpus sample 中 term coverage/scope |
| Stage-C coherence | 13 | term-pair pointwise mutual information 与 normalized PMI |
| Stage-C term impact | 13 | term impact 与 concentration |
| Stage-C length | 7 | sample document length statistics |
| Stage-C title/body | 10 | title/body occurrence shares |
| Stage-C morphology competition | 5 | term form competition |
| Morphology and tokenizer | 15 | query length、tokenization、uniqueness |
| Relation lexicalization | 8 | relation markers and prepositions |
| Entity and anchor | 13 | capitalization、numeric/year/quoted anchors |
| Lexical economy | 8 | repetition and concentration |
| Answer type and constraints | 17 | interrogative and answer-type surface constraints |
| Ambiguity | 8 | ambiguity markers |
| Hotpot multihop | 8 | bridge/comparison style markers |
| General style controls | 8 | punctuation and general style controls |

### 15.4 Stage-C corpus cache v2

Stage-C 不运行主 BM25/BGE Retriever，也不产生 ranked Top-k list、retrieval scores 或 ranks；但它并非只看 query 字符串。它会针对当前 query，在冻结的 500,000-document uniform sample 上执行 query-conditioned、unranked postings aggregation，统计 term coverage、共同出现和 scope。因此它属于“检索排序前的 query–collection lookup”，而不是零语料访问的纯 query-only 特征。具体缓存为：

- Corpus documents：5,233,329；sample seed `2026083001`；无放回 500,000，占 9.55%。
- Term universe：96,060；已知/在词表 terms 40,118。
- Full exact posting rows aggregated：105,792,593。
- Sample postings：10,103,415；sample title postings：905,005。
- Cache version 2 SHA-256：`ee9077eee60184fa4d760f0f457fe693c47913162e3e3eda4e486dbd66bdfebd`。
- Cache 构建 100.764 seconds；discovery feature extraction 117.799 seconds；natural extraction 50.619 seconds；外部调用 0。

Version 1 cache 被排除，因为 title/body 统计实现有 bug、缺少 total token count，并使用了不合适的 pointwise mutual information smoothing；所有正式结论只使用 version 2。Version 2 的 cache vocabulary 由 discovery 与 natural query 文本预先供应，尚不是覆盖任意生产新 query term 的完整在线索引。若部署契约只禁止主 Retriever 的 ranked Top-k 与 scores，Stage-C scope 可以视为 pre-ranking static lookup；若契约禁止动作选择前任何 matched-document 或 postings 操作，它就不满足严格 query-only 边界。当前尚未实现或测量其在线索引补全、延迟和资源成本。

### 15.5 冻结统计门槛

连续特质在 discovery 上需要：directional area under the receiver operating characteristic curve 至少 `0.56`；directional Spearman 至少 `0.10`；directional fourth-quartile-minus-first-quartile gap 至少 `0.020` 且 bootstrap interval 支持；winner 和 gap 的 permutation tests 分别在全部 207 hypotheses 上做 Holm correction 后不超过 `0.05`。

Binary 特质要求 directional odds ratio 至少 `1.30`，directional gap difference 至少 `0.020` 并满足 confidence interval 与两个 Holm-adjusted tests。所有 permutation/bootstrap 以 group 为单位，各 10,000 次。

进入 natural 前还要求 old/new 方向一致，且两 cohort 的 directional effects 至少保留 full effect 的 50%。Natural continuous gate 要求 AUC 至少 `0.55`、Spearman 至少 `0.08`、gap effect 至少 `0.015`，同时满足 interval 和 Holm correction；missing 必须为 0。

### 15.6 审计漏斗与 18 个 natural shortlist

207 个冻结特质中：139 个通过基础质量，25 个通过 discovery point-effect gate，18 个通过 old/new/full stability 并在查看 natural outcome 前冻结为 shortlist，最终仅 1 个通过 natural confirmation。最终状态计数为：114 个 effect too small、68 个 insufficient variation、7 个 old/new unstable、17 个 natural not replicated、1 个 supported single feature。

| Frozen shortlist feature | Family | Expected BM25 direction | Natural result |
|---|---|---:|---|
| `legacy_lexical__idf_max_share` | Historical lexical | lower | Not replicated |
| `legacy_dense__prototype_similarity_top_1` | Dense prototype | higher | Not replicated |
| `unique_lexical__idf_max_share` | BM25-aligned lexical | lower | Not replicated |
| `c_exact_idf_sum` | Collection | higher | Not replicated |
| `c_scope_log_at_least_2_docs` | Scope | higher | Supported |
| `c_scope_log_half_docs` | Scope | lower | Not replicated |
| `c_scope_half_over_any` | Scope | lower | Not replicated |
| `c_pair_pmi_min` | Coherence | lower | Not replicated |
| `c_pair_npmi_min` | Coherence | lower | Not replicated |
| `c_global_impact_max_share` | Term impact | lower | Not replicated |
| `q_raw_word_count` | Morphology/tokenizer | higher | Not replicated |
| `q_analyzed_token_count` | Morphology/tokenizer | higher | Not replicated |
| `q_unique_analyzed_term_count` | Morphology/tokenizer | higher | Not replicated |
| `q_character_count` | Morphology/tokenizer | higher | Not replicated |
| `q_capitalized_token_ratio` | Entity/anchor | lower | Not replicated |
| `q_max_token_frequency_share` | Lexical economy | lower | Not replicated |
| `q_bridge_style_marker` | Hotpot multihop | higher | Not replicated |
| `q_comparison_style_marker` | Hotpot multihop | lower | Not replicated |

除 supported feature 外，17 个自然结果的失败原因由 effect size、confidence interval 和 Holm-adjusted tests 的组合构成；它们不能被二次筛选或组合后再以同一 2,000 集合作为独立确认。

17 个未复现 traits 的具体 natural point estimates 为：

| Feature | Directional AUC or odds ratio | Directional Spearman | Gap effect | Holm winner / gap |
|---|---:|---:|---:|---:|
| `legacy_lexical__idf_max_share` | AUC 0.549184 | 0.055688 | 0.060093 | 1 / 1 |
| `legacy_dense__prototype_similarity_top_1` | AUC 0.531224 | 0.027176 | 0.022305 | 1 / 1 |
| `unique_lexical__idf_max_share` | AUC 0.546161 | 0.053270 | 0.055788 | 1 / 1 |
| `c_exact_idf_sum` | AUC 0.539693 | 0.054914 | 0.068066 | 1 / 1 |
| `c_scope_log_half_docs` | AUC 0.492382 | 0.007762 | 0.027035 | 1 / 1 |
| `c_scope_half_over_any` | AUC 0.502184 | 0.016881 | 0.030321 | 1 / 1 |
| `c_pair_pmi_min` | AUC 0.536831 | 0.042125 | 0.077997 | 1 / 1 |
| `c_pair_npmi_min` | AUC 0.537275 | 0.042782 | 0.066262 | 1 / 1 |
| `c_global_impact_max_share` | AUC 0.534214 | 0.041684 | 0.046062 | 1 / 1 |
| `q_raw_word_count` | AUC 0.565549 | 0.070070 | 0.060762 | 0.710429 / 0.263874 |
| `q_analyzed_token_count` | AUC 0.569255 | 0.076478 | 0.079599 | 0.350165 / 0.123588 |
| `q_unique_analyzed_term_count` | AUC 0.566671 | 0.075049 | 0.076466 | 0.368963 / 0.123588 |
| `q_character_count` | AUC 0.562804 | 0.069257 | 0.080400 | 0.944606 / 0.244776 |
| `q_capitalized_token_ratio` | AUC 0.561142 | 0.050419 | 0.057704 | 0.908909 / 1 |
| `q_max_token_frequency_share` | AUC 0.546716 | 0.055367 | 0.046754 | 1 / 1 |
| `q_bridge_style_marker` | odds ratio 1.587301 | Binary | 0.045378 | 0.673133 / 0.747325 |
| `q_comparison_style_marker` | odds ratio 1.228519 | Binary | 0.021547 | 1 / 1 |

其中长度类 features 的 AUC 点估计仍超过 0.56，但 Spearman 低于 0.08 或 Holm-adjusted tests 不通过；binary bridge marker 的 odds ratio 点估计较大，但 interval/multiple-testing evidence 不足。冻结 gate 要求全部条件同时通过，不能按单个漂亮数字保留。

### 15.7 唯一通过特质的定义与全部关键统计

`c_scope_log_at_least_2_docs` 的公式为：

\[
x(q)=\log\left(1+C_{\ge 2}(q)\right),
\]

其中 `C_{>=2}(q)` 是冻结 500,000-document corpus sample 中，同时包含 query 的至少两个不同、有效、在 BM25 analyzer 词表内 terms 的文档数量。它不读取当前 query 的 ranked result list、Top-k scores 或主 Retriever 输出，但需要执行 query-conditioned unranked postings aggregation；其是否满足“预检索”取决于系统契约是否允许动作选择前访问这种静态匹配统计。

| Cohort | Directional AUC | Directional Spearman | Fourth-minus-first quartile gap |
|---|---:|---:|---:|
| Discovery full 6,000 | 0.566747 | 0.117577 | 0.219399 |
| Old 2,760 | 0.536414 | 0.060857 | 0.136058 |
| New 3,240 | 0.590266 | 0.153769 | 0.286974 |
| Natural 2,000/604 non-tie endpoint | 0.586270 | 0.082574 | 0.072009 |

Discovery intervals：AUC `[0.552267,0.581528]`；Spearman `[0.092359,0.142638]`；gap effect `[0.166945,0.270360]`；winner 与 gap raw permutation p-value 均为 `0.00009999`，Holm-adjusted 均为 `0.020698`。与 retrieval evidence gap 的 Spearman 为 `0.174046`，但该值只用于机制描述。

Natural intervals：AUC `[0.541370,0.631046]`；Spearman `[0.040211,0.124149]`；gap effect `[0.026235,0.117541]`；winner 与 gap raw p-value 均为 `0.00019998`，Holm-adjusted 均为 `0.041396`。

### 15.8 决策、允许的下一步与局限

正式决策为 `SINGLE_FEATURE_GATE_PASS`。准确含义是：一个冻结的单一原始 query–collection trait 同时通过 discovery、old/new stability 和已消费 natural replication，允许单独冻结一个最小组合/增量模型诊断。它不证明 policy gain，也不允许在同一 natural-2,000 上继续搜索多特征组合。

本阶段没有 query rewrite、特征组合、interaction、Router training 或 internal final read；外部调用为 0。Sensitivity 使用 100,000 resamples 时关键 Holm 结论仍为约 `0.02898/0.02070`，不改变决定。

主要局限是：discovery 人为平衡；scope 来自 9.55% corpus sample 而非全语料；它与 query length、inverse document frequency 和 term co-occurrence 相关，单变量关联不能证明独立因果；sample count 依赖当前 corpus/analyzer；HotpotQA distractor construction 可能放大 scope pattern；96,060-term universe 仅 40,118 terms 在词表；natural 数据已消费；尚无 policy、安全或 latency 结果。

## 16. Phase 2.10：scope 对冻结 M3 的最小条件增量诊断

### 16.1 研究问题

Phase 2.10 于 2026-08-30 执行。它不重新开放大规模搜索，只回答：Phase 2.9 唯一通过的 `c_scope_log_at_least_2_docs`，在控制 Phase 2.8 冻结 M3 的 30D structured plus PCA32 embedding 表示后，是否提供稳定、实用且安全的条件增量？

主要比较是 augmented M3S minus frozen M3，而不是 M3S 相对 fixed retriever 的绝对 gain。若 paired increment 的 confidence interval 下界不大于 0，即使 M3S 的绝对 Router F1 很高，也必须停止。

### 16.2 数据、特征复核与唯一候选集合

- Training view：T2，12,000 queries、11,964 groups；BM25 winner 3,000、Dense winner 3,000、tie 6,000。
- Outcome rows：72,000，即 12,000×2 actions×3 repeats。
- Fixed BM25 F1 `0.593758`；fixed Dense `0.590560`；oracle `0.750635`。
- Scope 特征用 Phase 29 cache v2 独立为全部 12,000 rows 重提取；artifact SHA-256 `c672197bacec91300fae7de38af2d7c9061803276eae4027cd37d9313cefff33`。
- 与 Phase 2.9 discovery 6,000 non-ties 及 natural 2,000 的抽取逐值 exact parity，maximum absolute difference 0。
- Scope extraction 322.725 seconds；训练与评估 162.769 seconds；external calls 0。
- 主实验只核验 natural artifact hashes，不解析 outcome values；natural outcome rows/value read 均为 0；internal final rows 0。

没有搜索其他特征、组合、threshold 或模型。唯一三个候选为：

1. `S0_scope_only_ridge`：单个 scope 特征，fold-training StandardScaler 后 Ridge alpha 10。
2. `M3_pca32_structured_ridge`：执行器按 Phase 2.8 的 30D structured plus fold-internal PCA32 Ridge recipe、相同 folds 和 seeds 重新训练 M3，再把新产生的 calibrated out-of-fold predictions 与历史冻结 artifact 逐值比较；fold identifiers 完全一致，三个 seeds 的 maximum absolute prediction difference 均为 0。它不是直接复用历史 prediction array。
3. `M3S_pca32_structured_scope_ridge`：在 structured block 追加 scope 成 31D，再与 fold-internal PCA32 拼接，用相同 Ridge alpha 10。

Split seeds 为 `20260901`、`20260917`、`20261003`；每个 5 outer folds、4 inner folds；threshold 固定 0；group bootstrap 10,000，seed `20260930`。所有数据 split、fold IDs、preprocessing、calibration 和 evaluator 与冻结 M3 对齐。

### 16.3 T2 内部绝对策略结果

| Candidate | Consensus Router F1 | Gain over Dense | Gain over best fixed BM25 | 95% confidence interval vs best fixed | Coverage | Harmful/beneficial | Non-tie AUC | Gap Spearman |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Frozen M3 | 0.630801 | +0.040241 | +0.037042 | [0.030731,0.043297] | 52.50% | 0.607584 | 0.637612 | 0.176016 |
| Scope-only S0 | 0.608318 | +0.017758 | +0.014560 | [0.009139,0.020115] | 62.75% | 0.839058 | 0.566054 | 0.082167 |
| Augmented M3S | 0.631048 | +0.040488 | +0.037290 | [0.030987,0.043571] | 52.575% | 0.606170 | 0.637157 | 0.175465 |

完整 consensus switch profile：M3 switches 6,300，B/N/H=`1,827/3,269/1,204`，missed BM25-beneficial 1,173，calibration slope/top decile=`1.054449/0.111271`；S0 switches 7,530，`1,971/3,846/1,713`，missed 1,029，slope/top decile=`0.998205/0.064175`；M3S switches 6,309，`1,831/3,275/1,203`，missed 1,169，slope/top decile=`1.053808/0.110465`。

S0 在 winner-balanced T2 上有绝对增益，但 harm ratio 和 coverage 很差；其性能不能被解释成自然分布 policy。M3S 的绝对结果几乎与 M3 相同。

### 16.4 主要 paired increment

| Split | M3S minus M3 F1 | 95% confidence interval | Action changes |
|---|---:|---|---:|
| 20260901 | +0.000141 | [-0.000537,0.000849] | 未单列于主表 |
| 20260917 | +0.000182 | [-0.000397,0.000777] | 未单列于主表 |
| 20261003 | +0.000056 | [-0.000615,0.000711] | 未单列于主表 |
| Consensus | +0.000247 | [-0.000141,0.000662] | 29/12,000 |

Consensus 的 29 个 action changes 中，19 个为 Dense-to-BM25，10 个为 BM25-to-Dense。虽然三个 seed 与 consensus 点估计都为正，但 interval 跨 0，增量只有 `0.000247`，远低于 `0.01` 实用量级。AUC 从 `0.637612` 降至 `0.637157`，Spearman 从 `0.176016` 降至 `0.175465`。

### 16.5 系数诊断

| Model中的 scope coefficient | Folds | Mean | Sample standard deviation | Min | Max | Positive/negative folds |
|---|---:|---:|---:|---:|---:|---:|
| Scope-only raw Ridge | 15 | +0.044067 | 0.002725 | +0.039765 | +0.051074 | 15/0 |
| Scope-only after affine calibration | 15 | +0.043329 | 0.002583 | +0.039409 | +0.050228 | 15/0 |
| M3S raw Ridge | 15 | -0.001524 | 0.009055 | -0.019119 | +0.014292 | 7/8 |
| M3S after affine calibration | 15 | -0.001301 | 0.007931 | -0.016937 | +0.012737 | 7/8 |

Scope 单独看有稳定正方向；进入 M3 后系数接近 0、符号不稳定，说明其 discovery association 大部分已被现有 structured/PCA representation 吸收，或不足以形成独立决策信号。

### 16.6 冻结 gate 和正式停止决定

通过项：threshold 为 0；三个 seed paired gain 都为正；consensus paired point estimate 为正；M3S 相对 best fixed gain 和 interval 通过；三个 seed 的绝对 gain 为正；calibration slope `1.053808` 合格；top-decile realized gap `0.110465` 为正。

失败项：paired consensus interval 下界 `-0.000141` 不大于 0；harmful/beneficial `0.606170` 大于 0.5；coverage `52.575%` 高于 50%。因此正式决策为：

`STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL`

主协议据此禁止打开 natural outcomes。没有搜索 scope interaction 或其他 shortlist，没生成 natural predictions，internal final rows 0。该结论是“没有发现这个单特质在 M3 之上有稳定条件增量”，而不是否定 scope 的边际关联。

Validation：preflight `passed_and_frozen`；7 个聚焦 tests passed；post-run reporting audit complete；M3 reproduction exact。

## 17. Phase 2.10 追加 natural-2,000 诊断

### 17.1 证据角色和冻结顺序

主 Phase 2.10 已按 gate 停止。之后根据单独请求运行 natural appendix，角色明确为 `consumed_post_selection_diagnostic_not_holdout_not_final_test`；它不撤销或改写主实验 `STOP_NO_INCREMENTAL_SCOPE_POLICY_SIGNAL`。

三个候选用完整 T2 训练视图、相同三个 replica seeds full fit；模型、features、threshold 和 prediction array 在解析 natural outcomes 前冻结。Natural prediction freeze SHA-256 为 `7284e25f7b3289b97b301857a3d770d89a78fa8caba2d8cd9e2f8034fa9ce051`；predictions NPZ SHA-256 为 `162d95f17be47dbea9d8e383eb23a05997e58091a91c7b2258b5f4faf4c77cbf`。External calls 0；internal final rows 0。

### 17.2 完整 natural 结果

Natural 数据仍是 2,000 queries、267 BM25 winners、337 Dense winners、1,396 ties；fixed BM25/Dense/oracle 为 `0.604848/0.636340/0.707624`。

| Candidate | Router F1 | Gain over fixed Dense | 95% confidence interval | Coverage | Harmful/beneficial | AUC | Spearman | Calibration slope | Top decile realized gap |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Scope-only S0 | 0.628320 | -0.008020 | [-0.021227,0.005020] | 62.35% | 1.159277 | 0.586270 | 0.082574 | 0.596725 | -0.023308 |
| Frozen M3 | 0.634777 | -0.001563 | [-0.012720,0.009740] | 53.05% | 1.038577 | 0.575434 | 0.082373 | 0.404948 | +0.014424 |
| Augmented M3S | 0.634777 | -0.001563 | [-0.012720,0.009740] | 53.00% | 1.038577 | 0.575579 | 0.082498 | 0.406370 | +0.014424 |

Natural switches 与 B/N/H：S0 1,247，`182/881/184`，missed 85；M3 1,061，`144/772/145`，missed 123；M3S 1,060，`144/771/145`，missed 123。S0 top decile 为负，M3/M3S 虽 top decile 略正，但 score calibration slope 低于 0.5，且总体错误切换质量仍超过正确收益。

### 17.3 M3S 与 M3 的 natural paired changes

| Replica/consensus | M3S minus M3 | 95% confidence interval | Action changes |
|---|---:|---|---:|
| 20260901 | +0.0000714 | [0,0.0002143] | 4 |
| 20260917 | 0 | [0,0] | 3 |
| 20261003 | 0 | [0,0] | 3 |
| Consensus | 0 | [0,0] | 1 |

Consensus 仅一条 query 改变动作，该 query 的实际 BM25-minus-Dense gap 为 0，因此 end-to-end F1 完全不变。

### 17.4 决策与解释

Continuation gate 九项中仅 top-decile realized gap positive 和 non-tie AUC at least 0.55 通过；absolute gain、confidence interval、paired increment、paired interval、harm ratio、coverage 和 calibration slope 全部失败。决策为：

`STOP_NO_POST_SELECTION_NATURAL_STABILITY`

Phase 2.9 的单变量自然关联确实可以复现为 S0 AUC `0.586270`，但 threshold 0 会在自然分布切换 62.35%，最终 F1 比 fixed Dense 低 `0.008020`。这清楚区分了“trait 与 winner/gap 相关”和“trait 能形成安全 routing policy”两个命题。M3S 在 natural 上几乎逐动作等同 M3，进一步证实没有条件增量。

## 18. 跨阶段研究链、已取得成果与被排除的解释

### 18.1 从问题到证据的完整链条

本研究不是一次模型竞赛，而是逐层收缩解释空间：

1. Selected-7 retrieval-only 分析首先证明 BM25 与 Dense 的候选池和成功查询不完全重叠，因此“逐查询选择”具有研究价值。
2. Phase 0 冻结 group-aware 数据、动作、生成、指标和保留集；Phase 1 证明 context 对答案有强因果影响且自动评审链可用。
3. Phase 2 证明答案层面的 BM25/Dense 逐查询 oracle headroom 在三重复后仍约 `0.05`，不是单次生成随机性假象。
4. Phase 2.5 的 primary gate 证明 Answer Correctness preference 在 cross-repeat evaluation 中稳定，因而允许扩大 Answer Correctness labels；同阶段还观察到 normalized token F1 preference 与 Answer Correctness preference 高度一致，但 F1 的冻结角色仅为 auxiliary or diagnostic。后续使用 F1 作低成本 screen 是 Phase 3 的另行协议选择，不能归因于 Phase 2.5 已把 F1 晋升为主标签。
5. 原三动作 learning curve 和 Phase 2.6 证明严格 pre-retrieval query-only Router 虽然面对约 `0.07–0.10` oracle headroom，却只能取得接近零的实际 gain；失败来自低 switch precision。
6. Contriever 和 Natural Questions Dense Passage Retrieval 诊断显著削弱“只需换成已测试的 Contriever 或使用已测试的 Natural Questions Dense Passage Retrieval 高维 query embedding，就会自然解决路由”的简单解释；它们没有排除其他 encoder、其他检索器训练目标或其他数据集上的表示方案。
7. Phase 2.7 系统比较目标、表示、模型、旋转稳定性、校准和数据规模，正式停止 Query-only V1；随后 Stage 12 将缺失信息定位到实际 retrieval behavior 与 evidence sufficiency。
8. Phase 2.8 显示 winner enrichment 与构造分布上的内部可学习性提高同时出现，支持 winner 稀缺是限制因素之一；由于训练视图还同时改变样本量、tie 比例、action prior、best fixed 和 oracle headroom，该实验不能单独识别 winner scarcity 的因果贡献。它同时揭示 winner-balanced sampling 会造成严重分布错位风险。
9. Phase 2.8b 用 natural-2,000 证明 T1/T2 的大内部 gain 不迁移，错误切换在自然分布反而超过收益。
10. Phase 2.9 从 207 个原始 query/query–collection traits 中找到一个可复现的 scope proxy；Phase 2.10 的 post-selection 配对诊断未发现该 proxy 在既有 M3 条件下具有稳定增量，natural policy utility 精确不改善。

这条链把最初的宽泛问题“能否用 query 选择 Retriever”收敛为更具体的机制判断：可利用的动作异质性存在，但严格 query-only 可观察信号弱；人为平衡 winner 可以提高同分布可学习性，却会破坏自然动作先验；目前强信号主要在 retrieval 后的 score/rank/list/context/evidence 状态中。

### 18.2 已经取得的科学与工程成果

#### 成果一：建立了答案效用层而非只看检索指标的 Router 评价框架

研究统一用 BM25 与 Dense 各三次生成的平均答案 F1 计算逐 query utility gap，并把 fixed action、oracle、Router、beneficial/harmful mass、coverage、calibration 与 group bootstrap 放在同一协议中。它避免了用 nDCG、Recall 或 evidence hit 直接替代最终 RAG 效果。

#### 成果二：定量确认了真实且稳定的动作互补性

- 600-query repeated BM25/Dense revision 中，300-query historical-development subset 的 headroom：`+0.050521`；另外 300 条属于 train subset，不能把该数值写成 600 条 historical development 的结果。
- 4,800-query BM25/Dense headroom：`+0.070707`。
- 9,600-query BM25/Dense headroom：`+0.069356`。
- Natural-2,000 headroom：`+0.071284`。

不同样本上均出现约 `0.05–0.07` 的答案 F1 上界，说明研究问题不是由偶然样本制造的。

#### 成果三：识别并修正了 generation nondeterminism

Phase 2 初版虽有 headroom，但 F1 repeat agreement 只有 `0.90`，正式做出 `REVISE` 而非错误放行。统一三重复、组内均值、group-plus-repeat uncertainty 和 leave-one-repeat-out 后稳定性通过。这使后续监督 label 的噪声边界可审计。

#### 成果四：量化了 F1 与 Answer Correctness 的一致性及辅助价值

F1 与 Answer Correctness 的 non-tie sign agreement 为约 `0.91–0.94`，F1 selector 在 held-out repeats 上仍产生正 Answer Correctness gain；但 Phase 2.5 的正式放行依据是 cross-repeat Answer Correctness selector，F1 的冻结角色仅为 auxiliary or diagnostic。后续 Phase 3 独立选择 F1 作低成本 screen；报告同时保留“没有后期 policy 的 Answer Correctness generalization 证据”这一边界。

#### 成果五：给出了严格 query-only V1 的高可信负结果

Phase 2.7 不是单一模型失败，而是覆盖 17 个 screening candidates、四个正式 families、三套 grouped splits、nested calibration、1,200–9,600 learning curve、目标与旋转诊断的系统审计。最强 M3 只有 `+0.002455` consensus gain，interval 跨零、harm ratio `0.789`，只恢复 `3.54%` oracle headroom。这个负结果足以停止继续堆相同数据和相同 Query-only V1 表示。

#### 成果六：区分了“表示中有弱信号”与“能形成安全策略”

DPR probe AUC `0.6774`、Phase 2.7 M3 AUC `0.5801`、Phase 2.9 scope natural AUC `0.5863` 都说明信号不为零；但低局部纯度、低 policy gain、过高 harmful mass 和 natural reversal 表明排序信号不足以自动转化为动作效用。这一分离避免了把 AUC 或相关性误写成 Router 成功。

#### 成果七：定位了表示与模型的匹配问题

Raw 414D embedding plus XGBoost 在屏幕与正式结果中表现差，且正交旋转后 coverage/AUC 明显变化；Ridge 对旋转稳定，PCA32 plus structured Ridge 是最强 query-only family。结论不是 embedding 无信息，而是 raw axis-aligned tree 对当前几何不稳健。

#### 成果八：验证了 winner acquisition 的作用与风险

定向 retrieval lane 以 7,885 条新 labels 补足两侧 winner quota，使 T2 内部 gain 达 `+0.040–+0.043`。这一共同变化支持 winner scarcity 是可学习性限制之一，但不能从同时改变多个分布因素的 T0/T1/T2 对比中识别其独立因果效应。Natural-2,000 进一步显示同一构造过程伴随 action prior shift 与 over-switch，不能把 balanced out-of-fold 结果当作 natural generalization。

#### 成果九：完成了受多重检验和来源稳定性约束的单特质搜索

207 traits 经 207-wide Holm correction、old/new stability 和 natural replication，最终只保留 `c_scope_log_at_least_2_docs`。这一严格漏斗比事后挑选若干直觉特征更可信，也展示了大多数表面语言模式不能稳定复现。

#### 成果十：用最小配对增量诊断检验了 scope 的条件价值

Scope 单独系数在 15 folds 全正，但加入 M3 后 7 正/8 负；T2 paired increment `+0.000247`、interval 跨零；natural consensus increment 为 0。该实验是受控的 post-selection 配对增量诊断：scope 已经在完整 T2 与 consumed natural 数据上参与选择，grouped cross-validation 不能消除全局特征选择偏差，因此它不是因果实验，也不是无偏外部确认。可支持的结论是“没有发现 scope 在 M3 之上提供稳定的条件策略增量”，并据此停止继续围绕该 proxy 扩展模型。

#### 成果十一：建立了严格的数据消费和停止纪律

Phase 0 划出的 8,490 条 internal development partition 并非整体未见：Phase 2/2.5 已按早期协议消费其中 300 条 historical-development queries；只读 group join 还发现其中 11 个已消费 information-need groups 包含另外 14 条 development queries，这 14 个 query identifiers 虽未直接读取，其 group 已经暴露，也不能再算独立未见。后续原计划的 4,000 条 fresh-development subset 从未打开，internal final holdout 也始终未打开。Natural-2,000 打开后被明确标记为 consumed candidate selection；Phase 2.7 Stage 10/11 因 gate 失败标记不适用；Answer Correctness expansion 未在 F1 gate 失败后继续。因此，未来只能从整个 information-need group 都从未被读取、训练或用于选择的 groups 中重新冻结数据，不能按未读 query identifier 单独判断，也不能把整个 8,490 条 partition 声称为 untouched。

### 18.3 被当前证据排除或显著削弱的解释

| 曾经可能的解释 | 关键反证 | 当前判断 |
|---|---|---|
| 没有逐 query action headroom | 多个样本 oracle headroom 约 0.05–0.07 | 排除 |
| 单次生成随机性制造了 headroom | 三重复、leave-one-repeat-out 后仍稳定 | 排除 |
| 只需增加相同 query-only 样本 | 1,200–9,600 曲线仍远低于 +0.01 | 显著削弱 |
| 只需把 F1 换成 Answer Correctness | F1/AC preference 高一致；问题出在预测而非 surrogate 完全错位 | 显著削弱，但最终 AC 仍待验证 |
| 只需换 Dense retriever 为已测试的 Contriever | Contriever pair 同样 oracle 高、Router gain 近零，且 fixed performance 更差 | 显著削弱该具体方案；不排除其他 encoder |
| Raw 384D embedding 加树模型会更强 | Raw 414D XGBoost 弱且旋转敏感 | 排除当前实现 |
| Robust loss、weighted sign 或双头 utility 会解决 | Phase 2.7 目标消融均未改善 | 显著削弱 |
| Winner-balanced 内部 gain 可直接外推 | T1/T2 natural gain 反转为负 | 排除 |
| M4/M4-PCA 结构差异是主要因素 | 同视图模型差远小于视图差 | 排除为主因 |
| 一个显著 query trait 就能改善 Router | Scope 对 M3 的 paired increment 不显著，natural 为零，但属于 post-selection diagnostic | 当前证据不支持该 trait 的稳定条件增量；仍需全新外部协议才能作无偏确认 |
| Gold teacher 已经蒸馏到 Student | Stage 12 没有 Teacher-to-Student transfer | 未实施，不能声称 |
| 当前运行时已经能 adaptive route/hybrid | 配置只接受单个 `dense` 或 `bm25` method，pipeline 只实例化一个 retriever | 未实现 |

## 19. 当前瓶颈、运行时能力边界与下一项研究问题

### 19.1 第一瓶颈：严格预检索可观测性不足

答案效用差取决于“Retriever 实际找到了什么、排名怎样、上下文是否覆盖关键证据”，但 query-only 输入只能看问题表述和静态语料摘要。Phase 2.7 的成对增量最直接地量化了这一点：qrels-free post-retrieval P1 相对 query-only M3 增加 `+0.012380`，gold G1 再相对 P1 增加 `+0.024431`。因此缺失信息主要不是更复杂分类器，而是当前信息层不可见的实际 retrieval/evidence state。

### 19.2 第二瓶颈：自然动作先验与 winner-balanced 训练分布错位

Natural-2,000 中 69.8% 为 ties、Dense winners 多于 BM25 winners，fixed Dense 比 BM25 高 `0.031492`；T2 刻意构造 3,000/3,000 winners 与 6,000 ties，且 fixed BM25 反而略高于 Dense。固定零阈值在 T2 是合理对称规则，在 natural 上却让模型把约 53%–55% 查询切向 BM25，造成 harm mass 大于 benefit mass。当前学习到的 ranking 与部署 prior/calibration 没有分离。

### 19.3 第三瓶颈：错误切换代价没有被安全约束控制

多个候选的 AUC、Spearman 或内部 F1 为正，但 harmful/beneficial ratio 持续超过 `0.5`。模型会正确切换一部分 BM25 winners，却也在 Dense winners 上承担大幅损失；大量 tie switches 不产生收益，却增加不必要动作变化和潜在系统成本。当前需要的是 selective、harm-aware decision rule，而不是仅提高平均分类分数。

### 19.4 第四瓶颈：有效 winner 稀疏、监督信号被 ties 与重复噪声稀释

自然分布中约 70% queries 两动作三重复平均 F1 完全相同。真正有决策价值的 non-tie query 少，且其中一部分 gap 很小；少数大 gap query 对总收益贡献很高。三重复已降低噪声，但 sample efficiency、grouped uncertainty 和 tail harm 仍比普通平衡分类困难。

### 19.5 第五瓶颈：新的自然验证数据已经成为必要条件

Natural-2,000 已先后用于 Phase 2.8b candidate selection、Phase 2.9 single-feature replication 和 Phase 2.10 appendix，不能再作为未见数据调 threshold 或模型。任何新 policy 必须重新冻结自然分布 development/confirmation 协议；internal final holdout 虽仍未打开，但当前没有候选具备直接消费它的资格。

### 19.6 第六瓶颈：qrels-free probe 的收益与计算成本尚未共同验证

P1 使用两路 Retriever 的 scores、rank shape、overlap 和 context statistics，必须先运行 BM25 与 Dense。它能提高答案 F1 诊断，但若两路都完整运行，便失去“预检索选择一个 Retriever”原本的成本优势。尚未测量浅层 probe、early-exit、dual-retrieval cascade 的真实 latency、throughput、GPU/CPU cost 和 end-to-end quality trade-off。

### 19.7 第七瓶颈：当前 runtime 没有自适应路由能力

当前配置解析只允许 `retrieval.method` 为 `dense` 或 `bm25`；`src/pipeline.py` 根据这个单值加载 Dense embedder/index 或 BM25 path，然后只创建一个 `self.retriever`。没有 Router model loader、online 207-feature extractor、calibrator/threshold service、dual-retrieval probe、fallback、adaptive action logging 或 hybrid policy。因此所有 Router 结果仍是离线研究策略，不是可运行生产功能。

### 19.8 第八瓶颈：最终语义正确性和跨域泛化仍未确认

当前正式后期结果以 normalized token F1 为 endpoint。Phase 2.5 证明的是 Answer Correctness preference 的跨重复稳定性，并把 F1 保留为 auxiliary or diagnostic；后续 Phase 3/2.6–2.10 另行采用 F1 作廉价 endpoint。Phase 2.7 因该 F1 policy gate 失败没有购买新的 Answer Correctness labels；也没有在其他 QA corpus、不同 prompt/generator/context budget 上做答案级 replication。结论目前只适用于 HotpotQA、旧 short-answer prompt、Top-5 context、三重复和当前 retriever artifacts。

### 19.9 最值得提出的下一项研究问题

当前最高价值的问题不再是“能否再加 query-only feature”，而是：

> 在一个全新冻结、未消费的自然 HotpotQA protocol 上，能否用不依赖 qrels 的低成本 shallow retrieval probe 或 cascade，只观察两路浅层 scores、rank shape、overlap 与 context sufficiency proxy，在显式计算预算下把相对最佳固定动作的答案 F1 提升至少 `+0.01`，使 95% confidence interval 下界大于 0、harmful/beneficial mass ratio不超过 `0.5`、coverage 不超过 `50%`，并在冻结策略上复现 Answer Correctness 增益？

这个问题直接使用 Phase 2.7 的最强新信息，同时把 Phase 2.8b 揭示的自然 prior、错误切换和计算成本纳入同一个 estimand。

### 19.10 建议的新协议关键设计

1. 在任何 retrieval/outcome 可见前冻结新的自然-distribution development 和 confirmation groups；不复用 natural-2,000 做选择。
2. 把“ranking model”和“deployment action calibration”分开：winner-rich 数据只学习相对排序，新自然 development 只冻结 threshold/abstention，不再改 representation。
3. 只允许一个很小的 qrels-free probe family；不得加入 gold evidence、generation outputs 或事后 utility。
4. 同时报告 quality 和成本：单路 fixed Dense、单路 fixed BM25、两路 full probe、shallow probe/cascade 的 F1、Answer Correctness、latency、queries per second 和 resource cost。
5. 以 Dense 为自然默认，并允许 abstain；目标是少量高精度 BM25 switches，而不是覆盖大多数 queries。
6. 预先冻结 catastrophic-harm tail 指标和 subgroup checks，不能只用 mean gain。
7. 内部 gate 未通过则不得打开 internal final holdout；通过后只能对唯一冻结策略评估一次。
8. 若要研究真正 Teacher-to-Student distillation，必须明确 Teacher 产生何种 soft targets、Student 只接收哪些 pre-retrieval inputs、蒸馏 loss 如何与答案 utility 分离，并在新的未见自然数据上验证 Student；不能把现有 P1/G1 对照称为蒸馏。

## 20. 局限、外推范围、数据消费状态与复现产物索引

### 20.1 结论适用范围

本报告结论适用于当前本地 HotpotQA corpus、冻结 BM25 analyzer/index、`BAAI/bge-small-en-v1.5` Dense artifacts、Top-50 retrieval、Top-5 generation context、1,800-token 上限、`hotpot_short_answer_v1`、`gpt-4.1-mini-2025-04-14`、temperature 0、每动作三重复及 normalized token F1。改变 corpus、chunking、prompt、generator、context budget、retriever revision 或 answer metric 后，需要重新验证。

Selected-7 与 Natural Questions 结果是跨数据集 retrieval/representation diagnostics，不是 HotpotQA 答案 Router generalization。Stage 12 是同一 9,600 train-only cohort 的 nested out-of-fold 信息诊断，不是 fresh validation。Phase 2.8 T0/T1/T2 是构造分布，不能作为自然部署分布。Phase 2.9/2.10 是 post-selection diagnostics。

### 20.2 数据消费与可复用状态

| 数据/分区 | 规模 | 已用于 | 当前角色 | 可否作为未来未见确认 |
|---|---:|---|---|---|
| Pilot | 100 | Phase 1 | Consumed sanity/calibration | 否 |
| Frozen train partition | 67,920 total | 是下列 Phase 2.7 base cohort、Phase 2.8 新 acquisition sampling frame 及其他早期 train samples 的总来源，不是 67,920 条全部具有相同暴露状态 | Mixed-exposure source partition，必须按 query identifier 与 group identifier 追踪，而不能整体称为 consumed 或 untouched | 不能把整个 partition 直接当确认集；只有完成逐 query/group exposure audit 后，才可从真正未读且未参与选择的 groups 中按新协议重新冻结 |
| Frozen development partition | 8,490 total | Phase 2/2.5 消费其中 300 条 historical-development queries；11 个已消费 groups 另含 14 条未直接读取的 group-mate queries；原计划的 4,000 条 fresh-development subset 未打开 | Partially consumed partition；query 未读不等于 information-need group 未暴露 | 只能从整个 group 都从未读取、训练或参与选择的 groups 中按新协议重新冻结 |
| Phase 2.6 prefix | 4,800 | Phase 2.6 and Phase 2.7 screen | Consumed screening | 否 |
| Phase 2.7 cohort | 9,600 | Query-only formal and Stage 12 | Consumed train-only OOF | 否 |
| Phase 2.8 paired-retrieval acquisition pool | 48,866 | 两路 retrieval、evidence-based acquisition strata 与 query selection；其中 7,885 条继续生成答案 | Retrieval exposure 与候选选择已消费；其中 40,981 条没有 generation outcomes，但也不是独立未见确认数据 | 否；若未来只研究答案标签，仍须新协议明确处理既有 retrieval/selection exposure |
| Phase 2.8 acquisition answer labels | 7,885，属于上一行 48,866 的子集 | T1/T2 winner quota construction；产生 47,310 条两动作三重复 outcomes | Consumed directed acquisition labels | 否 |
| Phase 2.8 unretrieved backup pool | 6,069 | 未执行 paired retrieval 或 generation，但已位于 acquisition sampling frame | 没有 outcome exposure；也没有被预注册为独立 holdout | 只有核验 group overlap、sampling-frame exposure 和 selection history 后，才能按全新协议考虑；不能直接称为确认集 |
| Phase 2.8 reserved natural sample | 2,000 | Phase 2.8b, 2.9, 2.10 appendix | Consumed candidate selection/post-selection diagnostic | 否 |
| Internal final holdout | 8,490 | 无 | 从 HotpotQA source train 内部切出；sealed，outcomes read 0；不是官方 HotpotQA split | 是；只有唯一冻结且前置 gate 通过的策略可打开 |

Phase 2.9 和 Phase 2.10 尚未进入 `registry.yaml`；它们有冻结配置、decision、metrics、hash 和 archive conclusion，但当前应写作“已完成的 hash-frozen diagnostics”，而不是“registry-managed complete phases”。Phase 2.8b 在 registry 中明确标记 `consumed: true`、`reuse_for_retuning: forbidden`、`next_gate: NEW_NATURAL_DISTRIBUTION_PROTOCOL_REQUIRED`。

### 20.3 尚未实现或尚未验证的内容

- 没有可部署 Router checkpoint 或唯一正式模型冻结。
- 没有 online adaptive Router runtime、feature service、calibrator、threshold、fallback 或 telemetry。
- 没有 qrels-free shallow probe/cascade 的成本约束自然验证。
- 没有真正 Teacher-to-Student distillation。
- 没有后期候选的 Answer Correctness generalization。
- 没有 fresh-development 或 official-final policy result。
- 没有跨 prompt、generator、context budget 或 dataset 的答案级 replication。
- 没有测试 multilayer perceptron、partial least squares、encoder fine-tuning 或新 embedding；这意味着当前负结论针对 Query-only V1 候选空间，不是所有可能模型的不可能性定理。

### 20.4 主要实验产物索引

| 阶段 | 配置/协议 | 主要结果和结论 |
|---|---|---|
| 全局状态 | [`registry.yaml`](registry.yaml) | 当前本地路径、状态、数据角色权威 |
| Phase 0–2.6 | [`phases/phase00_26/config.yaml`](phases/phase00_26/config.yaml) | [`phases/phase00_26/conclusion.md`](phases/phase00_26/conclusion.md)、[`phases/phase00_26/results/`](phases/phase00_26/results/) |
| Selected-7 | [`auxiliary/retrieval_context/metadata.json`](auxiliary/retrieval_context/metadata.json) | [`auxiliary/retrieval_context/`](auxiliary/retrieval_context/) |
| Contriever | Phase 0–2.6 source config | [`auxiliary/contriever/data/summary.json`](auxiliary/contriever/data/summary.json) |
| Natural Questions DPR | Diagnostic manifest | [`auxiliary/nq_dpr_probe/summary.json`](auxiliary/nq_dpr_probe/summary.json) |
| Phase 2.7 | [`phases/phase27/config.yaml`](phases/phase27/config.yaml) | [`phases/phase27/conclusion.md`](phases/phase27/conclusion.md)、[`phases/phase27/results/`](phases/phase27/results/) |
| Phase 2.8 | [`phases/phase28/config.yaml`](phases/phase28/config.yaml) | [`phases/phase28/conclusion.md`](phases/phase28/conclusion.md)、[`phases/phase28/results/training_scale_comparison.csv`](phases/phase28/results/training_scale_comparison.csv) |
| Phase 2.8b | [`phases/phase28b/config.yaml`](phases/phase28b/config.yaml) | [`phases/phase28b/conclusion.md`](phases/phase28b/conclusion.md)、[`phases/phase28b/results/`](phases/phase28b/results/) |
| Phase 2.9 | [`phase29_query_trait_audit_config.yaml`](phase29_query_trait_audit_config.yaml) | [`../../outputs/router/hotpotqa_bd_router_v1/runs/phase29_single_trait_audit_v1/`](../../outputs/router/hotpotqa_bd_router_v1/runs/phase29_single_trait_audit_v1/) |
| Phase 2.10 | [`phase30_scope_incremental_config.yaml`](phase30_scope_incremental_config.yaml) | [`../../outputs/router/hotpotqa_bd_router_v1/runs/phase30_scope_incremental_v1/`](../../outputs/router/hotpotqa_bd_router_v1/runs/phase30_scope_incremental_v1/) |
| Phase 2.10 appendix | [`phase30_scope_natural_appendix_config.yaml`](phase30_scope_natural_appendix_config.yaml) | [`../../outputs/router/hotpotqa_bd_router_v1/runs/phase30_scope_natural_appendix_v1/`](../../outputs/router/hotpotqa_bd_router_v1/runs/phase30_scope_natural_appendix_v1/) |

Phase 2.9 与 Phase 2.10 的基准 Git HEAD 为 `522f0cd4fc289488c4b0d3c31f6f2090600a41ef`，但相关新文件在该 HEAD 中尚未跟踪；它们的复现身份依赖配置、runner、feature extractor 和 artifact SHA-256，而不能只靠该 commit。Phase 2.7 记录的 evidence code head 为 `a422e39398f456316f8166e13dcbd7d496be9ca5`，执行 worktree 当时为 dirty，同样不能声称由单一 clean commit 完全重建。

### 20.5 本报告的最终结论

我们已经证明 BM25 与 Dense 在 HotpotQA 答案层存在稳定且足够大的逐 query 互补性，也建立了一套能防止泄漏、错误放行和留出集过度消费的完整实验框架。我们同样以多阶段证据证明：当前严格 query-only/corpus-static Version 1 特征和候选不能把这种上界转化为自然分布上显著、安全、实用的 Router；winner-balanced 扩充产生的内部收益主要不具备自然迁移性；对唯一复现的 scope trait，受控 post-selection 诊断也未发现其在 M3 之上的稳定条件策略增量。

因此当前成果是一个机制清楚、边界严格的研究结论，而不是已部署系统。在已评估的 natural-2,000 样本中，fixed Dense 是 best fixed baseline；这一结论尚未在 internal final holdout 上确认，也不构成已经完成的生产策略切换。下一突破点应从“继续堆 query-only 特征”转向新的、成本受控、qrels-free shallow retrieval probe/cascade 协议，并用全新自然数据同时验证答案收益、错误切换和系统成本。
