先不考虑我刚才说的方案，我现在真正头疼的问题是，仅从query本身，是很难找到一些决定性因素用于决策sparse还是dense检索更好；仅从retrieval本身，bm25偏向词法，dense偏向语义，但是query有可能会同时出现需要词法和语义同时作用的情况。如果我们将query和retrieval一起看，寻找query特性和retrieval特性的主要联系，retrieval的一些特性是很难表述的，只能借助类似向量空间来表示，而且只能在离线检索来确认retrieval的特性；即使是很好的将query和retrieval联系到一起，也不能保证检索结果可以有效的提升rag下游结果。我个人比较赞同深度学习训练的方法，因为它能以下游rag质量为训练目标，同时将query和retrieval的某种特质联系起来。但是我总感觉这种方法不够精准，很难发挥bm25偏向词法，dense偏向语义的真正优势。

我认为你现在抓到的，才是这个方向最核心、也最难的矛盾。你的不适感是合理的：

> **黑箱深度学习 Router 可能学会“哪个 retriever 经常赢”，却未必真正学会“BM25 为什么赢、Dense 为什么赢”。**

问题不在于深度学习本身不精准，而在于当前很多方法的**问题建模和归纳偏置不够精准**。

# 1. 最优 retriever 本来就不是 query 的固有属性

真正要预测的是：

[
R^*(q)=\arg\max_R U(q,R\mid C,G)
]

其中：

* (q)：query；
* (R)：BM25 或 Dense；
* (C)：具体 corpus；
* (G)：具体 generator；
* (U)：最终 RAG utility。

所以，“这个 query 是 lexical query 还是 semantic query”本身就是一个过度简化的问题。

同一个 query：

* 换一个 corpus，BM25/Dense 胜负可能变化；
* 换一个 dense retriever，胜负也可能变化；
* 换 generator，最终 RAG utility 还可能再次变化。

因此，严格的 query-only 分类一定存在上限。它只能学习训练环境中的统计规律，而不能仅凭 query 确定 corpus 中是否真的存在精确匹配词、dense retriever 是否学过这种语义关系、检索到的内容是否能被 generator 使用。

---

# 2. “BM25 词法、Dense 语义”还不够准确

BM25 获胜，并不是因为 query 在抽象意义上“偏词法”，而是因为：

> query 中存在能够在当前 corpus 中有效区分目标文档的词法锚点。

例如：

* 稀有实体；
* 专有名词；
* 数字、日期、型号；
* 准确短语；
* 高 IDF 术语。

Dense 获胜，也不是因为 query 单纯“偏语义”，而是因为：

> query 与目标文档之间存在词汇错配，并且当前 dense retriever 恰好能够跨越这种语义差距。

例如：

[
bad\ guy \leftrightarrow villain
]

但 dense retriever 是否能跨越某种语义差距，还取决于它的训练数据和表示空间，并不是所有 dense model 都具有同样能力。Environment-aware IR 的实验就发现，不同 retriever 偏好的 query formulation 明显不同；更重要的是，retriever 之间的差异主要表现为结构和表达方式差异，而不只是信息需求本身不同。

因此，更准确的三个变量应该是：

[
L(q,C)=\text{Lexical Discriminativeness}
]

即 query 中的词法线索在当前 corpus 中有多强的区分能力；

[
S(q,C,R_D)=\text{Semantic Bridgeability}
]

即 query–document 的词汇差距是否需要语义桥接，并且当前 dense retriever 能否完成这种桥接；

[
K(q,C,R_B,R_D)=\text{Complementarity}
]

即 BM25 和 Dense 是否分别捕获了不同但有用的证据。

第三个变量尤其重要。一个 query 完全可能同时包含：

* 必须准确匹配的实体；
* 需要语义理解的关系；
* 多个不同的信息约束。

这种情况下，它不应该被硬塞进 BM25 或 Dense，而应被识别成 **complementary query**。

---

# 3. 为什么普通端到端深度学习方法会让你感觉“不精准”

假设直接训练：

[
f_\theta(q,R_i)\rightarrow U(q,R_i)
]

然后用下游 answer correctness 做监督。

这个设计方向没有错，但有四个风险。

## 第一，它可能学习数据集捷径，而不是 retriever 机制

例如模型可能学到：

* 医学 query 通常 BM25 好；
* 某些问句形式通常 Dense 好；
* 某些答案类型通常属于某个数据集。

它能够提高测试集准确率，但不代表它真正理解了：

* exact lexical anchors；
* semantic mismatch；
* sparse–dense complementarity。

换 corpus 后，这些 topic shortcut 很可能失效。

## 第二，一个通用 retriever embedding 会混合太多能力

R³AG 已经比普通分类器前进了一步：它把 retriever capability 分成 retrieval quality 和 generation utility，并用下游回答正确性监督 generation utility。

但它学到的依然是两个抽象 capability embeddings。它没有明确要求模型分别识别：

* BM25 的 exact-match 能力；
* Dense 的 semantic-bridging 能力；
* 两者的互补程度。

所以 R³AG 更接近：

> 哪个 retriever 与 query 的下游需求更匹配？

而不是：

> 当前 query 为什么需要 sparse、为什么需要 dense，以及是否同时需要两者？

R³AG 自己也将“组合多个 retriever”列为尚未覆盖的复杂场景。

## 第三，硬分类标签会丢失关键结构

下面三种情况完全不同：

[
U_B=0.90,\quad U_D=0.20
]

[
U_B=0.90,\quad U_D=0.88
]

[
U_B=0.10,\quad U_D=0.05
]

但硬分类都会标成 BM25。

它们实际分别表示：

1. BM25 明显占优；
2. 两者都好，几乎不需要路由；
3. 两者都差，不应该高置信选择 BM25。

## 第四，下游 utility 监督虽然正确，但太稀疏且有噪声

一次 answer correctness 同时混合了：

* retrieval 是否找对；
* document 是否完整；
* context 是否含噪声；
* generator 是否正确利用；
* generator 自身随机性。

所以只给最终的 0/1 标签，模型很难知道自己究竟应该学习 sparse/dense 的哪个差异。

---

# 4. 我认为更有希望的方向：机制约束的端到端 Router

我建议的不是放弃深度学习，而是把它改成：

> **Mechanism-aware、utility-aligned 的结构化学习。**

核心是同时满足两件事：

1. 用下游 RAG quality 保证最终目标正确；
2. 用明确的辅助结构，让模型不能完全绕过 sparse/dense 的真实机制。

可以设计三个分支。

## 4.1 Lexical branch：明确建模 BM25 的优势来源

输入不是只有 raw query，而是 query 与 corpus 的低成本词法关系：

[
h_L(q,C)
]

例如：

* query token 的平均、最大和方差 IDF；
* rare term 数量；
* named entity、number、date、quoted phrase；
* query token 在 corpus 中的覆盖率；
* 是否包含高区分度 n-gram；
* query 是 keyword style 还是完整自然语言。

这个分支预测：

[
\hat L(q,C)
]

即：

> 当前 query 是否具有可被 BM25 有效利用的词法锚点。

这不是简单地说“query 有多少实体”，而是看这些实体在**当前 corpus**中是否有区分能力。

---

## 4.2 Semantic branch：建模 Dense 是否能跨越词汇差距

构造：

[
h_S(q,C,R_D)
]

输入可以包括：

* query embedding；
* query 到 corpus prototypes / centroids 的距离；
* query 在 dense corpus space 中的局部密度；
* dense retriever 对这类 query 的训练分布熟悉度；
* query 的抽象程度、改写程度和词汇缺失程度。

它预测：

[
\hat S(q,C,R_D)
]

即：

> 当前 query 是否需要语义桥接，以及这个具体 Dense retriever 是否有能力完成它。

这里必须强调“具体 Dense retriever”。Environment-aware IR 发现，针对一个 retriever 学到的 query formulation 策略往往不能可靠转移到另一个 retriever，这说明 Dense 不能被当作一个统一能力类别。

---

## 4.3 Complementarity branch：不要强迫所有 query 二选一

再增加：

[
\hat K(q)
]

预测：

> BM25 和 Dense 是否可能提供互补证据。

离线可以用以下信号监督它：

* 两个结果集的 overlap；
* 两个 retriever 各自独有的 relevant documents；
* union 是否优于任意单独 retriever；
* hybrid RAG answer 是否超过两个 single retriever；
* 两个 retriever 的 utility 是否都高但覆盖不同 evidence。

最终 Router 输出不应该只有：

[
{\mathrm{BM25},\mathrm{Dense}}
]

而应该至少有：

[
{\mathrm{BM25\ dominant},
\mathrm{Dense\ dominant},
\mathrm{Complementary/uncertain}}
]

第一阶段甚至不需要学习复杂的 hybrid weight。第三类 query 统一采用一个固定 hybrid baseline 即可。之后再研究动态权重。

这能避免让本来就需要两种信号的 query，被强制分类成其中一种。

---

# 5. 离线检索特性可以作为“训练时特权信息”

你提到一个关键困难：

> retrieval 的很多性质只能实际检索后才能确认，但在线路由又不希望先运行全部 retrievers。

这个矛盾可以用 **Teacher–Student** 解决。

## Offline Teacher

训练阶段运行 BM25、Dense 和 generator。

Teacher 可以看到：

[
(q,D_B,D_D,U_B^{ret},U_D^{ret},U_B^{rag},U_D^{rag})
]

因此它可以较准确地学习：

* 哪种 lexical evidence 被 BM25 找到；
* 哪种 semantic evidence 被 Dense 找到；
* 两个结果是否互补；
* 哪个结果真正帮助了 generator。

## Online Student

Student 在推理时只看到：

[
q+\text{低成本 corpus/retriever descriptors}
]

然后学习模仿 Teacher 的：

* utility predictions；
* sparse/dense preference；
* complementarity probability；
* latent capability representation。

也就是：

[
T(q,\text{retrieval outcomes})
\longrightarrow
S(q,\text{pre-retrieval signals})
]

这样：

* retrieval results 只用于离线监督；
* 推理时仍然是 pre-retrieval；
* Student 不只学习单一 winner label，而是学习 Teacher 提供的丰富行为信息。

这比直接：

[
q\rightarrow\mathrm{BM25/Dense}
]

精确很多。

---

# 6. 最值得加入的训练方式：反事实 query 对

这是我认为最能解决你“不够发挥 sparse/dense 真正优势”担忧的部分。

对同一个 information need，构造多个保持答案不变、但改变 retrieval mechanism 的 query：

### 原始表达

> Who wrote God's Plan?

### 词法增强版本

> God's Plan songwriter

### 语义改写版本

> Which musician received the writing credit for Drake's 2018 hit single?

它们：

* topic 相同；
* gold evidence 相同；
* answer 相同；
* 但 lexical overlap 和语义改写程度不同。

然后实际运行 BM25 和 Dense，观察 preference 是否变化。

这种数据可以迫使模型学习：

> 哪些表达变化导致 BM25 受益，哪些表达变化导致 Dense 受益。

而不是简单学习：

> 音乐问题属于 Dense，人物问题属于 BM25。

Environment-aware IR 的结果正说明了这种结构差异非常重要：BM25 学出的有效 query 更偏 keyword-style，而其他 retriever 可能偏好描述式或 question-like 表达。

这类 counterfactual supervision 比单纯扩大训练集数量更重要。

---

# 7. 一个较完整但仍然清晰的训练目标

模型最终预测：

[
\hat U_B(q),\quad
\hat U_D(q),\quad
\hat P_{\mathrm{comp}}(q)
]

损失可以由四部分组成：

[
\mathcal L
==========

\mathcal L_{\mathrm{RAG}}
+
\lambda_1\mathcal L_{\mathrm{retrieval}}
+
\lambda_2\mathcal L_{\mathrm{comp}}
+
\lambda_3\mathcal L_{\mathrm{distill}}
]

其中：

* (\mathcal L_{\mathrm{RAG}})：预测两个 pipeline 的 answer correctness，保证最终目标；
* (\mathcal L_{\mathrm{retrieval}})：预测 Recall/nDCG 或 retrieval quality，帮助模型理解检索机制；
* (\mathcal L_{\mathrm{comp}})：预测 sparse–dense 是否互补；
* (\mathcal L_{\mathrm{distill}})：模仿看过实际 retrieval outcomes 的 Teacher。

这样 retrieval metrics 不再是最终优化目标，而是**帮助模型理解为什么下游 utility 会变化的辅助监督**。

这也比 R³AG 单纯将 retrieval quality 和 generation utility 编码成抽象空间，更明确地保留了 BM25/Dense 的机制差异。R³AG 已证明同时考虑 retrieval quality 与 generation utility 是必要的，但它并没有显式约束 lexical、semantic 和 complementarity 三类因素。

---

# 8. 我对你当前困境的最终判断

你真正面临的不是：

> 还没有找到足够好的 query features。

而是：

> **“BM25 还是 Dense”不应该被建模成 query 的单一类别。**

更准确的建模对象应当是：

[
\boxed{
\text{Lexical discriminativeness}
+
\text{Semantic bridgeability}
+
\text{Sparse–dense complementarity}
+
\text{Generation utility}
}
]

深度学习仍然是合理方向，但不能只是：

[
\text{raw query}
\rightarrow
\text{black-box encoder}
\rightarrow
\text{retriever label}
]

而应该是：

[
(q,C,R)
\rightarrow
\begin{cases}
\text{lexical mechanism}\
\text{semantic mechanism}\
\text{complementarity}
\end{cases}
\rightarrow
\text{retrieval utility}
\rightarrow
\text{RAG utility}
]

我建议将当前研究问题收敛成：

> **能否通过机制约束的表示学习，分别建模 query–corpus 的词法可区分性、query–dense retriever 的语义可桥接性以及 sparse–dense 的互补性，并利用离线 retrieval outcomes 和下游 RAG utility 进行监督，实现比黑箱 query-only Router 更准确、更可泛化的检索策略选择？**

这个问题既保留了你认可的端到端训练，也正面解决了你担心的“模型只学胜负、不学 sparse/dense 真正优势”的问题。







如果我们按照你的思路进行尝试，你会怎样进行设计安排





我会把它设计成一个**分阶段、可验证、能逐步加复杂度**的研究，而不是一开始就训练一个很大的端到端 Router。

核心目标是：

> 用下游 RAG utility 作为最终监督，同时显式约束模型分别学习
> **词法可利用性、语义可桥接性、Sparse–Dense 互补性**。

暂时可以把方法称为：

[
\textbf{Mechanism-Aware Utility Router}
]

---

# 一、先固定研究范围

第一版不要同时变化太多组件，否则无法判断 Router 学到了什么。

固定：

* 一个 corpus (C)；
* 一个 BM25；
* 一个 Dense retriever；
* 一个固定 hybrid 方法，例如 RRF；
* 一个 generator (G)；
* 相同的 top-(k)、prompt 和 context budget。

候选策略设为：

[
\mathcal R=
{R_B,R_D,R_H}
]

分别表示：

* (R_B)：BM25；
* (R_D)：Dense；
* (R_H)：固定的 BM25–Dense hybrid。

我建议一开始就保留 Hybrid，而不是只做 BM25/Dense 二分类。因为你已经意识到，有些 query 同时需要精确实体匹配和语义匹配；如果输出空间只有两个类别，模型无论学得多好，都只能把这种 query 错误压入其中一类。

BM25 和 Dense 的互补性本身有明确机制基础：BM25利用 term matching，Dense 则可通过学习到的表示匹配不同词面的同义或改写关系。 QuDAR 也表明 sparse–dense 的最优组合会随 query 和 corpus 变化，固定权重无法充分覆盖这种差异。

完整框架可以概括为：

```text
                         Offline Training
                                │
                ┌───────────────┴────────────────┐
                │                                │
        Run BM25 / Dense / Hybrid         Fixed Generator
                │                                │
                ▼                                ▼
       Retrieval Outcomes                 RAG Answers
                │                                │
                ▼                                ▼
       Mechanism Supervision             Generation Utility
       lexical / semantic /
         complementarity
                │                                │
                └───────────────┬────────────────┘
                                ▼
                    Mechanism-Aware Router
                                │
                             Online
                                │
          Query + low-cost corpus/retriever descriptors
                                │
                                ▼
                 BM25 / Dense / Hybrid
```

---

# 二、第一阶段：建立完整的离线 Oracle 表

对每一个训练 query (q_i)，都运行三条完整 pipeline：

[
q_i\rightarrow R_B\rightarrow D_i^B\rightarrow G\rightarrow y_i^B
]

[
q_i\rightarrow R_D\rightarrow D_i^D\rightarrow G\rightarrow y_i^D
]

[
q_i\rightarrow R_H\rightarrow D_i^H\rightarrow G\rightarrow y_i^H
]

然后为每一个 query 保存完整信息：

[
\mathcal O_i=
\left(
q_i,
D_i^B,D_i^D,D_i^H,
u_{B}^{ret},u_D^{ret},u_H^{ret},
u_B^{gen},u_D^{gen},u_H^{gen}
\right)
]

其中：

## Retrieval utility

可以包括：

[
u_r^{ret}(q)
============

\mathrm{Recall@k},\ \mathrm{nDCG@k},\ \mathrm{MRR}
]

## Generation utility

可以包括：

[
u_r^{gen}(q)
============

\mathrm{EM},\ \mathrm{F1},\ \mathrm{AnswerCorrectness}
]

第一版最好选择有标准答案的 QA 数据集，优先用 EM/F1，减少 LLM judge 噪声。

最终 Router 的主要目标是：

[
r^*(q)
======

\arg\max_{r\in\mathcal R}
u_r^{gen}(q)
]

Retrieval utility 不作为最终目标，而作为辅助监督，帮助模型理解为什么某条 pipeline 最后更好。R³AG 的核心发现正是：retrieval quality 和 generation utility 是不同能力维度，仅判断文档相关性不足以保证最终回答正确。

---

# 三、先进行一次必要的可行性诊断

在训练复杂模型前，我会先回答三个问题。

## 1. 是否存在足够的 Oracle headroom？

计算：

[
U_{\text{fixed}}
================

\max_r\frac1N\sum_i u_r^{gen}(q_i)
]

即全体 query 使用同一个最优固定策略。

再计算：

[
U_{\text{oracle}}
=================

\frac1N
\sum_i
\max_r u_r^{gen}(q_i)
]

二者差值：

[
H=
U_{\text{oracle}}-U_{\text{fixed}}
]

代表 routing 的最大理论收益空间。

如果这个差值本身很小，那么无论 Router 多复杂，研究价值都有限。MoR 的工作先验证不同 retriever 在 query level 确实存在 comparative advantage，也是出于同样原因。

---

## 2. Hybrid 是否真的形成第三种有价值的策略？

定义：

[
g_H(q)
======

## u_H^{gen}(q)

\max
\left(
u_B^{gen}(q),
u_D^{gen}(q)
\right)
]

观察有多少 query 满足：

[
g_H(q)>\epsilon
]

如果 Hybrid 在相当一部分 query 上明显超过两个单独 retriever，那么 Complementarity 分支值得做。

如果 Hybrid 几乎从不超过最佳单一 retriever，那么第一版可以退回 BM25/Dense 二选一，把 Hybrid 放到后续。

---

## 3. Query preference 是否存在可学习结构？

先用 frozen query encoder 对所有 query 编码，检查：

* 相邻 query 的最优 retriever 是否一致；
* 邻域内 utility vector 是否相似；
* query embedding 距离与 utility 差异是否相关；
* lexical features 是否能预测 BM25 胜出；
* dense corpus familiarity 是否能预测 Dense 胜出。

这一步用于判断：

> 普通 query embedding 是否已经包含一定路由信息，还是必须进行机制约束训练。

---

# 四、构造三类“机制监督”

这是整个设计最核心的部分。

这些监督只在离线阶段计算，在线时不需要实际运行所有 retriever。

---

## 1. Lexical Discriminativeness：词法可区分性

这个分支回答：

> 当前 query 是否包含能够被 BM25 有效利用的词法锚点？

针对训练 query，可以从 gold evidence 和 BM25 retrieval outcomes 中构造监督：

[
t_L(q)
======

[
\text{BM25 rank of gold},
\text{BM25 score margin},
\text{IDF-weighted overlap},
\text{entity match},
\text{number/date match},
\text{phrase match}
]
]

重点不是简单统计 query 中有没有实体，而是看：

> query 的这些词，在当前 corpus 中是否真的具有区分目标证据的能力。

例如：

* 高 IDF 专有名词；
* 准确论文名、产品名、型号；
* 数字、年份；
* 专业术语；
* 引号短语。

这些才是 BM25 真正能够发挥优势的条件。

---

## 2. Semantic Bridgeability：语义可桥接性

这个分支回答：

> query 与相关文档之间是否存在词汇错配，以及当前 Dense retriever 能否跨越这种错配？

离线监督可以包括：

[
t_S(q)
======

[
\text{Dense rank of gold},
\text{Dense score margin},
\text{query–gold dense similarity},
\text{lexical mismatch},
\text{Dense success under low overlap}
]
]

特别关注这类样本：

[
\text{LexicalOverlap}(q,d^*)\ \text{低}
]

但：

[
\mathrm{DenseRank}(d^*)\ \text{高}
]

它们是真正代表 Dense semantic bridging 能力的样本。

这里不能把“Dense”视为一个统一能力类别。Environment-aware IR 的实验表明，不同 Dense retriever 甚至可能偏好不同的 query style，而且针对一个 retriever 学到的策略不一定能转移到另一个 retriever。

因此，最终模型学习的是：

[
S(q,C,R_D)
]

而不是抽象的：

[
S(q)
]

---

## 3. Sparse–Dense Complementarity：互补性

这个分支回答：

> 两个 retriever 是否分别找到了不同但有用的证据？

可以从两类信号构造监督。

### Retrieval-level complementarity

例如：

* 两个 top-(k) 结果集的 Jaccard overlap；
* 各自独有 relevant documents 数量；
* union Recall 是否超过任何单独结果；
* relevant evidence coverage 是否互补。

定义一个简单信号：

[
c^{ret}(q)
==========

## u_H^{ret}(q)

\max
\left(
u_B^{ret}(q),
u_D^{ret}(q)
\right)
]

### Generation-level complementarity

更重要的是：

[
c^{gen}(q)
==========

## u_H^{gen}(q)

\max
\left(
u_B^{gen}(q),
u_D^{gen}(q)
\right)
]

只有当 Hybrid 最终真的提高答案质量时，才说明这种互补对 RAG 有价值。

---

# 五、在线 Router 输入什么

在线不能使用真实 retrieval outcomes，否则就失去 pre-retrieval routing 的意义。

所以 Student Router 只输入：

[
q+\text{low-cost corpus/retriever descriptors}
]

具体分为三部分。

## Query representation

[
h_q=E_\theta(q)
]

使用一个预训练语言模型编码 query。

---

## BM25-oriented lexical descriptors

[
x_L(q,C)
]

包括：

* query 长度；
* 平均、最大 IDF；
* IDF 方差；
* rare term 数量；
* corpus vocabulary coverage；
* entity、number、date、quoted phrase；
* unigram/bigram 的 corpus frequency；
* keyword-style 程度。

这些数据只依赖 query 和预先计算好的 corpus term statistics，不需要真正运行 BM25。

---

## Dense-oriented corpus descriptors

[
x_S(q,C,R_D)
]

包括：

* query embedding；
* query 到 corpus centroids 的距离；
* 最近 corpus prototype 的相似度；
* query 在 dense corpus space 中的局部密度；
* query 到 Dense retriever 训练 query prototypes 的距离；
* query style：keyword、question-like、descriptive。

MoR 的 pre-retrieval 思路也使用 query 与 retriever corpus clusters 的关系估计 retriever familiarity，因此使用 corpus prototypes 是有依据的。

这些 descriptors 本质上是：

> 用少量预计算的 corpus/retriever 表示，近似描述当前 query 与 retrieval environment 的关系。

---

# 六、模型架构

我会使用一个共享 Query Encoder，加三个机制分支。

[
h_q=E_\theta(q)
]

## Lexical branch

[
h_L=
\operatorname{MLP}_L
\left(
[h_q;x_L]
\right)
]

它主要服务 BM25 utility prediction。

## Semantic branch

[
h_S=
\operatorname{MLP}_S
\left(
[h_q;x_S]
\right)
]

它主要服务 Dense utility prediction。

## Complementarity branch

[
h_C=
\operatorname{MHA}
\left(
h_q,
[h_L,h_S],
[h_L,h_S]
\right)
]

它根据当前 query 动态判断：

* lexical 信息是否占主导；
* semantic 信息是否占主导；
* 两者是否同时有价值。

最终分别预测：

[
\hat u_B^{ret},\quad \hat u_D^{ret},\quad \hat u_H^{ret}
]

以及：

[
\hat u_B^{gen},\quad \hat u_D^{gen},\quad \hat u_H^{gen}
]

结构可以表示为：

```text
                         Query q
                            │
                      Query Encoder
                            │
                  ┌─────────┴─────────┐
                  │                   │
        Lexical descriptors     Dense descriptors
                  │                   │
                  ▼                   ▼
          Lexical Branch        Semantic Branch
              hL                     hS
                  └─────────┬─────────┘
                            │
                  Complementarity Fusion
                            hC
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       BM25 Head         Dense Head        Hybrid Head
          │                 │                 │
    Retrieval/RAG      Retrieval/RAG      Retrieval/RAG
       Utility            Utility            Utility
```

它和 R³AG 的共同点是：

* 都显式拆分 capability；
* 都利用 retrieval 和 generation 两种监督；
* 都进行 query-conditioned capability fusion。

区别在于 R³AG 拆成：

[
\text{Retrieval Quality}
+
\text{Generation Utility}
]

而这里进一步把 retrieval mechanism 明确拆成：

[
\text{Lexical}
+
\text{Semantic}
+
\text{Complementarity}
]

---

# 七、训练目标

我不会使用单一的三分类交叉熵，而会采用多任务训练。

## 1. Retrieval utility regression

[
\mathcal L_{ret}
================

\sum_{r\in{B,D,H}}
\operatorname{Huber}
\left(
\hat u_r^{ret},
u_r^{ret}
\right)
]

---

## 2. Generation utility regression

[
\mathcal L_{gen}
================

\sum_{r\in{B,D,H}}
\operatorname{Huber}
\left(
\hat u_r^{gen},
u_r^{gen}
\right)
]

这是主要目标。

---

## 3. Pairwise routing loss

对于任意两个策略 (r_a,r_b)，令：

[
\Delta_{ab}
===========

## u_{r_a}^{gen}

u_{r_b}^{gen}
]

训练模型保持同样的排序：

[
\mathcal L_{rank}
=================

\sum_{a\neq b}
|\Delta_{ab}|
\log
\left(
1+
\exp
\left[
-\operatorname{sign}(\Delta_{ab})
(\hat u_{r_a}^{gen}-\hat u_{r_b}^{gen})
\right]
\right)
]

乘上 (|\Delta_{ab}|) 很重要：

* 0.90 vs 0.10 的错误应受到严重惩罚；
* 0.51 vs 0.50 的错误不应受到同等惩罚。

---

## 4. Mechanism auxiliary losses

[
\mathcal L_{mech}
=================

\mathcal L_{lex}
+
\mathcal L_{sem}
+
\mathcal L_{comp}
]

分别要求三个分支预测：

* lexical discriminativeness；
* semantic bridgeability；
* hybrid complementarity。

最终：

[
\mathcal L
==========

\mathcal L_{gen}
+
\lambda_1\mathcal L_{ret}
+
\lambda_2\mathcal L_{rank}
+
\lambda_3\mathcal L_{mech}
]

这样最终目标仍然是 RAG answer quality，但模型不能只通过 topic shortcut 完成训练，因为它还必须解释 retrieval mechanism。

---

# 八、Teacher–Student 作为第二阶段，而不是一开始就做

第一版先直接用离线标签训练上述 Router。

如果结果表明：

* post-retrieval outcomes 很有预测力；
* 但 pre-retrieval Student 学不到；

再加入 Teacher。

## Teacher 输入

Teacher 可以看到：

[
q,D_B,D_D,D_H
]

以及：

* top-(k) scores；
* score margins；
* result overlap；
* retrieved document relevance；
* document diversity；
* actual generation outcomes。

Teacher 输出：

[
p_T(r\mid q)
]

以及机制表示：

[
h_L^T,h_S^T,h_C^T
]

## Student 输入

Student 仍然只看到：

[
q,x_L,x_S
]

训练 Student 模仿 Teacher：

[
\mathcal L_{distill}
====================

\operatorname{KL}
\left(
p_T\parallel p_S
\right)
+
\sum_{m\in{L,S,C}}
|h_m^T-h_m^S|^2
]

完整损失变成：

[
\mathcal L_{\text{total}}
=========================

\mathcal L
+
\lambda_4\mathcal L_{distill}
]

这相当于：

> 离线阶段利用真实 retrieval outcomes 学习 retriever 行为，在线阶段把这些知识压缩进一个 pre-retrieval Router。

---

# 九、推理阶段怎样决策

Router 输出：

[
\hat u_B^{gen},
\quad
\hat u_D^{gen},
\quad
\hat u_H^{gen}
]

最简单的选择是：

[
\hat r(q)
=========

\arg\max_r
\hat u_r^{gen}(q)
]

如果考虑成本：

[
\hat r(q)
=========

\arg\max_r
\left[
\hat u_r^{gen}(q)
-----------------

\lambda\operatorname{Cost}(r)
\right]
]

还应增加置信度机制。

如果：

[
\max_r\hat u_r^{gen}<\tau_a
]

说明所有 retrieval strategy 可能都失败，进入：

* query rewrite；
* deeper retrieval；
* fallback；
* no-answer。

如果：

[
\hat u_H^{gen}
--------------

\max(\hat u_B^{gen},\hat u_D^{gen})

> \tau_c
> ]

才真正选择 Hybrid。

这样不会因为模型不确定，就默认把所有 query 都送入昂贵的 Hybrid。

---

# 十、实验安排

我会分成五个阶段。

## 阶段 A：Oracle 与数据诊断

先回答：

1. BM25、Dense、Hybrid 的 query-level 胜率；
2. Oracle 相比 best fixed strategy 有多少空间；
3. Hybrid 是否真的有独立价值；
4. retrieval winner 与 generation winner 一致率；
5. query-only features 的可预测上限。

如果这一步显示几乎不存在 oracle headroom，就不继续复杂建模。

---

## 阶段 B：必要基线

至少包括：

1. Fixed BM25；
2. Fixed Dense；
3. Fixed Hybrid；
4. Oracle；
5. Query-only black-box classifier；
6. 手工 lexical features + MLP；
7. query embedding + classifier；
8. query + corpus descriptors；
9. 完整 mechanism-aware Router。

这样才能判断提升到底来自：

* 深度模型；
* corpus awareness；
* 机制辅助任务；
* 还是 Hybrid action 本身。

---

## 阶段 C：关键消融

分别移除：

* lexical branch；
* semantic branch；
* complementarity branch；
* retrieval supervision；
* generation supervision；
* pairwise ranking loss；
* corpus descriptors；
* Teacher distillation。

最重要的比较是：

[
\text{Black-box query encoder}
]

对比：

[
\text{Mechanism-aware multi-task model}
]

如果后者只在同分布测试集有轻微提高，却没有改善跨数据集泛化和机制子集表现，那么它的价值就有限。

---

## 阶段 D：Counterfactual 测试

对同一 information need 构造不同 query formulation：

```text
Natural:
Who wrote God's Plan?

Keyword:
God's Plan songwriter

Descriptive:
Which musician received writing credit for Drake's
2018 hit single?
```

然后真实运行三个 retriever。

检查：

1. 实际 retriever utility 是否随 query formulation 改变；
2. Router 的预测是否随之合理改变；
3. Topic 不变时，模型是否仍能识别 lexical/semantic 差异。

这一步是判断模型有没有真正学到 BM25/Dense 机制，而不是 topic shortcut 的关键测试。

---

## 阶段 E：泛化测试

至少进行两类泛化：

### 换 corpus / dataset

训练于一个数据集，测试另一个数据集。

### 换 Dense retriever

例如训练时使用 Dense (R_D^1)，测试时换成 (R_D^2)。

第二项尤其重要，因为已有研究显示不同 dense retriever 的最优 query formulation 并不相同。

若更换 retriever 后模型彻底失效，说明它学到的是特定模型行为；这不一定使研究无效，但应明确将方法定义为 retriever-conditioned。

---

# 十一、主要评价指标

不要只报告 routing accuracy。

最重要的是：

## Average downstream utility

[
U_{\text{router}}
=================

\frac1N
\sum_i
u_{\hat r(q_i)}^{gen}(q_i)
]

## Regret

[
\operatorname{Regret}
=====================

\frac1N
\sum_i
\left[
\max_r u_r^{gen}(q_i)
---------------------

u_{\hat r(q_i)}^{gen}(q_i)
\right]
]

## Oracle recovery

[
\operatorname{Recovery}
=======================

\frac{
U_{\text{router}}-U_{\text{best fixed}}
}{
U_{\text{oracle}}-U_{\text{best fixed}}
}
]

还应报告：

* EM/F1/Answer Correctness；
* high-margin query accuracy；
* Hybrid query identification；
* calibration；
* latency和成本；
* lexical-heavy subset；
* low-overlap semantic subset；
* complementarity subset。

---

# 十二、最值得先做的最小版本

考虑实现成本，我不会立即做 Teacher–Student 或 synthetic counterfactual training。

第一轮只做：

### 数据

选择一个带：

* queries；
* gold answers；
* relevant passages；

的数据集。

使用大约：

* 5,000–10,000 个训练 query；
* 1,000 个 validation；
* 1,000 个 test。

### 策略

* BM25；
* 一个固定 Dense；
* RRF Hybrid。

### 模型对比

训练三个模型：

1. **Query-only classifier**
2. **Query + corpus descriptors utility regressor**
3. **Lexical/Semantic/Complementarity 三分支 Router**

### 监督

* 所有样本计算 retrieval utility；
* generation utility 先在全部样本上使用 deterministic generation；
* 如果成本过高，则优先计算：

  * BM25/Dense retrieval disagreement；
  * Hybrid potential；
  * high-margin query；
  * 再加一个随机控制集。

### 第一次实验需要回答的核心问题

[
\boxed{
\text{显式的 retrieval mechanism supervision，是否比普通 query-only 深度模型更好？}
}
]

具体看三个结果：

1. 是否降低 downstream regret；
2. 是否提高 oracle recovery；
3. 是否在 counterfactual query formulation 上更符合真实 retriever 行为。

只有这三点中至少两点成立，我才会继续加入：

* Teacher–Student distillation；
* synthetic counterfactual training；
* 动态 hybrid weight；
* no-retrieval/query rewrite fallback。

---

## 最终建议的研究推进顺序

```text
1. 固定 BM25 / Dense / Hybrid / Generator
                     │
2. 构建离线 retrieval + RAG utility 表
                     │
3. 验证 Oracle headroom 和 Hybrid 价值
                     │
4. 训练 query-only baseline
                     │
5. 加入 query–corpus descriptors
                     │
6. 加入 lexical / semantic / complementarity 分支
                     │
7. 进行消融和 counterfactual 测试
                     │
8. 若有效，再加入 Teacher–Student
                     │
9. 最后研究动态 Hybrid 权重与跨环境泛化
```

这个安排的优点是，每一步都有清楚的研究问题：

* **第 3 步**验证 routing 是否值得做；
* **第 5 步**验证 corpus awareness 是否必要；
* **第 6 步**验证机制约束是否优于黑箱训练；
* **第 7 步**验证模型是否真正学习 sparse/dense 差异；
* **第 8 步**验证离线 retrieval outcomes 能否有效蒸馏到在线 Router。

这样即使最终完整方法没有显著超过强基线，你仍然能够清楚定位问题到底发生在：缺少 oracle 空间、query 信息不足、机制特征不可预测、generation utility 噪声，还是 retriever preference 不具备跨环境稳定性。
