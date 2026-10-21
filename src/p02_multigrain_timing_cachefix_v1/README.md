# 冻结多粒度候选: `p02_multigrain_timing_cachefix_v1`

本目录包含该方法的独立代码、固定提示词和示例、R02/v12身份、配置，以及300题完整回放材料。方法代码不导入父仓库或另一候选。

```powershell
python src/p02_multigrain_timing_cachefix_v1/run.py --verify
python src/p02_multigrain_timing_cachefix_v1/run.py --replay --output outputs/p02_multigrain_timing_cachefix_v1_replay300.json
python src/p02_multigrain_timing_cachefix_v1/run.py --dry-run
```

在本目录下也可以直接运行 `python run.py --verify` / `python run.py --replay`。回放不需要API密钥、远端或原实验数据库。

读取顺序：`core/pipeline.py: Pipeline.execute()` → `core/child_prompts.py` / `rewrite_prompts.py` / `final_prompts.py` → `config.json`。配置和示例每轮加载一次。`FLOW.txt`说明固定流程。`FROZEN.json`列出包文件；`--verify`检查清单文件存在及方法身份，`--replay`逐项核对记录的提示、检索材料、节点和输出，不再做文件摘要校验。

新问题仍由远端R02分解，本地只读取query-only JSONL：`id`、`qid`、`dataset`、`question`、`supporting_questions`。数组不带前题引用；程序追加原query为goal。`prompts/decomposition_v12.txt`与`models/DECOMPOSER_IDENTITY.json`固定远端分解约定；权重未复制且本次未连接远端。

实际生成的调用方式：

```powershell
python src/p02_multigrain_timing_cachefix_v1/run.py --run --inputs PATH_TO_RUNTIME.jsonl --output NEW_OUTPUT_DIR --max-cost-cny 1
```

只有显式`--run`会提交API。密钥从`DASHSCOPE_API_KEY`读取；模型固定为`qwen-plus-2025-12-01`。每轮费用上限必须为有限正数，并受原累计300元账本约束。成功相同输入直接复用；未知请求不自动重发。配置或预算在提交前拒绝时，不创建发送中记录。费用按冻结token单价估算，实际账单未核对。

大型语料、BGE索引/模型及累计费用账本通过`config.json`中的明确路径引用，不复制大型资源。移动目录后需要配置这些外部资源。运行配置仅保留本方法使用的资源；变更有效配置时使用新的输出目录。检索使用目录内私有副本的原BGE编码、50000行块FAISS内积和offset读取组件，保持单query编码及search_many(query_batch_size=1)。HotpotQA 从构建目录的 `embeddings` 子目录读取向量，其余两个数据集直接读取配置的向量目录。

检查点表结构及节点、结果字段保持原样，旧记录中的占位字段仍保留。原完整请求缓存可继续复用；新的检索缓存键直接保存规范化内容，旧检索键未命中时按原流程读取已存材料或执行检索。历史记录和模型来源材料保持原样。

300题为已使用的开发集，每数据集100题。回放仅验证已保存轨迹等价，不能称为新推理或独立测试。`SOURCE_PROVENANCE.json`保存历史来源；新结果和原实验隔离。
