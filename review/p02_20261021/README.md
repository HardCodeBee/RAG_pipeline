# P02 审核入口

这份材料回答四个问题：方法想解决什么问题，冻结实现实际上做什么，已有证据支持哪些结论，以及接下来准备如何评价。

## 建议阅读顺序

1. [ARCHITECTURE.md](ARCHITECTURE.md)：从真实实现解释多粒度分解、历史问答辅助改写、双路检索、goal 与最终短答。
2. [RESULTS.md](RESULTS.md)：先看三个数据集各100题的历史开发成绩，再看相关变体和正式500题的未完成状态。
3. [NEXT_EVALUATION.md](NEXT_EVALUATION.md)：主评不强制等预算；采用有来源的论文配置；等预算另作扩展。
4. [源码与当前计划](../../experiments/query_decomposition/unified_rag_eval_v1_20261021/PLAN.txt)、[决策日志](../../experiments/query_decomposition/unified_rag_eval_v1_20261021/DECISIONS.jsonl)、[配置来源映射](../../experiments/query_decomposition/unified_rag_eval_v1_20261021/reviews/20261021_official_config_selection/PAPER_CONFIG_MAP.json)。

## 材料如何组织

- `src/p02_multigrain_timing_cachefix_v1/`：冻结独立包；算法、prompt、模型身份与来源文件保留原字节。
- `experiments/query_decomposition/`：当前统一层、既有v2基础层、P02直接演进实验和必要本地源码依赖。历史执行器仅用于代码审核，不代表新主评已就绪。
- `evidence/`：按原相对路径保存报告、逐题明细、审计和来源。旧报告的部分正式分数、旧费用口径、调用次数等字段原样留作历史证据，不构成新主表的指标。
- `raw_cases/`：完整RAG演进及分解器组件实验的逐题记录压缩归档，包含失败、重试和未采用的变体。按实验和子组分片；子组可为数据集或分解版本，具体实验类型以其来源报告为准。组件记录不计为完整RAG结果。每个成员的原路径、字节数及SHA256列在清单中。
- `formal_runtime/`：正式1500题的已生成分解输出；有分解不等于有完整RAG结果。
- `component_records/`：分解器训练与确认阶段的逐条输出无损压缩；对应汇总报告和评审说明仍可直接阅读。
- `results/`：便于直接查看的历史开发300题输出和当前完成状态；未重新运行评分器。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：入选来源、哈希、归档成员以及明确排除项。

仅保留与P02、其直接演进、所用分解器和完整流程比较有关的材料。未推送模型权重、整库语料/索引、凭据、全局计费数据库或无关Router研究；较大的分解器训练正文以原始路径、哈希、配方和结果说明其身份。由原始SQLite支撑的全部历史账务并未在本分支复制，离线可回放范围与本地原始实验可重启范围并不相同。

## 离线核查

以下命令只校验已有字节或回放记录，不建立API客户端，不加载GPU模型或语料索引：

```powershell
python -B review/p02_20261021/verify_bundle.py
python -B review/p02_20261021/hydrate_replay.py
python -B src/p02_multigrain_timing_cachefix_v1/run.py --verify
python -B src/p02_multigrain_timing_cachefix_v1/run.py --replay --output replay-review.json
```

`expected300.jsonl` 和 `replay.sqlite3` 无损压缩保存；`hydrate_replay.py` 校验SHA256后在独立包原路径还原，已有文件若不匹配会拒绝覆盖。该步骤仅写入这两个已存在记录的副本。回放核对记录的prompt、上下文、检索包和输出一致性，**不是新推理，也不证明独立泛化**。

`SOURCE_MANIFEST.json` 覆盖选入的原始来源。新编写的审核说明与工具另由 `DELIVERY_FILES.json` 覆盖。路径中的Windows用户名、远端部署路径和工作区服务地址是历史环境信息，不能将它们当成可直接部署的新配置。

所有评估继续暂停。请勿把这个审核分支的提交、克隆或离线回放视为恢复评估授权。

本次交付的来源、归档、凭据模式和Git字节核验结果见 [FINAL_VALIDATION.json](FINAL_VALIDATION.json)。
