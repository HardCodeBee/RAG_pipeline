# NQ BM25/BGE exclusive-success DPR 空间诊断

## 冻结范围

- 标签来自现有 qrels-based first-stage nDCG@10；未重跑检索、reranker 或生成。
- BM25-only：BM25 nDCG@10 > 0 且 BGE nDCG@10 = 0。
- BGE-only：BGE nDCG@10 > 0 且 BM25 nDCG@10 = 0。
- 样本：BM25-only 260，BGE-only 1049。
- 表示：facebook/dpr-question_encoder-multiset-base 的 pooler_output；余弦分析使用 L2 副本。

## 主要结果

| 指标 | 结果 |
|---|---:|
| DPR OOF ROC-AUC | 0.6774 |
| DPR ROC-AUC 95% CI | [0.6429, 0.7120] |
| Surface OOF ROC-AUC | 0.6012 |
| Surface + DPR OOF ROC-AUC | 0.6631 |
| Margin sensitivity DPR ROC-AUC | 0.6687 |
| k=10 邻域同标签比例（20 次中位数） | 0.5330 |
| k=20 邻域同标签比例（20 次中位数） | 0.5280 |
| k=50 邻域同标签比例（20 次中位数） | 0.5176 |
| cosine silhouette（20 次中位数） | 0.0037 |

## 预注册判定

**未达到预注册的实用结构阈值**：至少一项线性可解码性、邻域纯度或 margin sensitivity 检查未通过；降维图不能改变该判定。

这个结论只表示：在 NQ 的冻结检索协议下，DPR query representation 是否包含与 BM25/BGE exclusive-success 标签相关的可解码信号。它不证明语言空间天然知道最佳 retriever，也不证明该信号可以带来端到端 router 增益。

NQ 与 DPR 训练范式可能存在 encoder-dataset alignment；若为阳性，下一步应使用相同 DPR encoder 在 HotpotQA 上复现。
