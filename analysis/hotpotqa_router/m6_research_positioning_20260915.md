# M6 有界文献定位

日期：2026-09-15。范围固定为三篇；这是问题定位与目标解释，不构成新颖性判断。未读取 BEIR 答案或改变采集、模型与诊断实现。

## 1. 检索前路由与答案效用

**Tong Zhao 等，2026，R³AG: Retriever Routing for Retrieval-Augmented Generation。** 已实际打开 arXiv v1 全文，核读 §3、§4.1–4.2、§5.1 与 Appendix B。[作者论文全文](https://arxiv.org/html/2604.22849v1)

其决策输入为 query 和可训练的检索器标识 token；Appendix B 明确排除检索轨迹。因此按论文给定的计算图，它属于检索前路由。候选包括 BM25、七种 Dense，以及不检索动作。监督包含答案 EM、F1、检索器全局正确率，并另用文档质量评分；随后训练对比目标和融合分类目标。[§3–4 与 Appendix B](https://arxiv.org/html/2604.22849v1#S3)

与 M6 共同点是：query 驱动的检索器选择，以最终答案表现为目标。差异是多动作、双能力编码器和注意力融合；它不是 M6 的二动作绝对效用差加权分类。它的生成效用标签包含 EM/F1，不应直接称为独立语义 Answer Correctness。本文只据其方法定位，不移植其收益结论。[§4.2](https://arxiv.org/html/2604.22849v1#S4.SS2)

## 2. 绝对效用差加权分类

**Baqun Zhang、Anastasios A. Tsiatis、Marie Davidian、Min Zhang、Eric Laber，2012，Estimating Optimal Treatment Regimes from a Classification Perspective。** Stat 1:103–114。[作者稿入口及 §3](https://pmc.ncbi.nlm.nih.gov/articles/PMC3640350/)

**核读限制：**出版社摘要页成功打开；搜索工具返回了作者稿 §2–3 的正文与公式，但直接打开 PMC 遇验证码，大学库 PDF 返回 403（另一次无凭证下载也失败）。所以本条属于已定位并核读索引正文，尚未完成所要求的独立全文打开核对。[出版社页面](https://onlinelibrary.wiley.com/doi/10.1002/sta.411)

索引正文 §3 将条件均值差记作 C(X)，以 sign(C) 为类别、|C| 为错误代价，把最大期望效用转换成加权分类。输入是动作实施前的协变量；应用是治疗选择，监督是临床结局，并非答案质量。与 M6 的相同点是决策理论形式；差异是该文需要估计未观测的动作结局，而 M6 使用离线成对生成效用。[作者稿 §2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC3640350/)

### 本轮独立代数推导，不依赖上述网页公式的转码

设 X 为检索前表示，Δ=U_B−U_D，a(X)∈{0,1} 表示是否选择 BM25。定义 Δ₊=max(Δ,0)。则逐样本有

\[
|\Delta|\mathbf{1}\{a(X)\ne\mathbf{1}[\Delta>0]\}
=\Delta_+-a(X)\Delta.
\]

因此最小化其期望，等价于最大化相对 Dense 的期望收益。对未经截断的 w=|Δ|、标签 y=1[Δ>0]，条件加权二元交叉熵为

\[
L(p\mid X=x)=-A(x)\log p-B(x)\log(1-p),
\quad A=\mathbb E[\Delta_+\mid x],\ B=\mathbb E[(-\Delta)_+\mid x].
\]

当 A+B>0 时，最优 p*=A/(A+B)，故

\[
p^*(x)>1/2\iff\mathbb E[\Delta\mid X=x]>0.
\]

这是期望效用的决策边界。p* 通常不是 BM25 获胜概率，logit 也不是 F1 差；A=B=0 时任意 p 的风险相同。例如 Δ 以 90% 概率为 +0.1、10% 为 −1：BM25 获胜概率为 0.9，但均值差为 −0.01，p*=0.09/0.19<0.5。

以上是无限函数类、总体期望和精确权重下的目标性质，不保证有限样本、正则化线性头或分布转移后的实际收益。若实现对近零差设容差，其严格目标是经该规则处理后的加权分布；实际损失和归一化仍以本地代码核读为准。

## 3. mean pooling 的相关证据及层选择边界

**Nils Reimers、Iryna Gurevych，2019，Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks。** 已实际打开官方 PDF，核读 §3、§6/Table 6。[论文 PDF](https://aclanthology.org/D19-1410.pdf)

SBERT 比较 CLS、MEAN、MAX，默认采用 MEAN；§6 显示池化差异依赖训练目标：NLI 分类设置影响较小，STS 回归设置中 MAX 明显较差。推理时可独立编码单句，因此这种表示可用于检索前输入；但原文监督是 NLI/语义相似度，不是 BM25/Dense 的答案效用。[§3、§6](https://aclanthology.org/D19-1410.pdf#page=7)

这支持把池化视作需要验证的表示选择，不能推出第六层更适合路由，也不能证明 M6 机制成立。本轮三篇限额内没有核读到直接证明“冻结 BGE 第六层 mean pooling 能更好预测答案效用差”的文献。

## 与当前实现对齐的事实范围

本轮仅查本地源码和协议元数据：`work/router_research/m6_beir_validation_v1/protocol.json:56–65` 固定 raw-query-only 输入、logit>0 选 BM25 和 normalized-token answer F1；`run_layer_pooling.py:27–33` 定义层/池化比较，`:290–293` 定义绝对 gap 加权 BCE 诊断。以上引用不表示重新训练或打开当前验证结果。

可使用的定位是：**固定双动作、检索前表示、答案效用监督下的成本敏感决策学习**。三项文献证据分别解释问题族、决策目标和池化选择；M6 是否有效仍由原完整验证及独立复核决定。
