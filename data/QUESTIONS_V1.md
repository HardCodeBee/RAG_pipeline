# Questions v1

`questions_v1.jsonl` 是正式 baseline 的冻结题集，共 24 题：22 题可回答、2 题不可回答。证据页码按仓库 corpus 中 PDF 的物理页码人工核对；新增的 Lewis RAG 题位于 PDF 第 3、4 页，DPR 题位于 PDF 第 3–4 页。

该题集当前既用于 baseline 评价，也用于第一轮检索诊断，因此不能把其结果称为未见测试集性能。后续参数选择不得再把它作为独立测试集。

`questions_heldout_v1.jsonl` 含 5 个未用于本轮参数比较的问题；在下一轮方案冻结前不要运行它，以保留其 held-out 作用。
