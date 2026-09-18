# BEIR test7405：确认资源的补充身份与用途审计

> 后续更新（2026-09-15）：[完整 group 身份审计与独立复核](m6_test_group_identity_results_20260915.md)已补齐本快照中的 group 缺口：test 对内部共享 236 查询 / 234 组；全 7405 行保留，确认资格仍待选择使用史审计。以下保留原先有界库存快照。

2026-09-15。有界只读审计；未拟合、预测、生成或下载，未打开新问题正文、参考答案或答案效果。机器记录见[本次身份库存](m6_confirmation_inventory_20260915.json)。

**新增发现：7405条还全部用于另一项 BGE/DPR/Contriever 检索基线矩阵，既有使用不限于原四条件实验。当前扫描没有找到该集合的答案效用或 Router 动作缓存，但完整 group 排除及历史选择依赖仍未解决，因此不能据此认证独立确认资格。**

## 1. 当前身份库存中的实际命中

本次遍历仓库 `outputs`、`output`、`artifacts`、`analysis/hotpotqa_router` 及相邻 `work/router_research`。1904份NPZ只检查目录成员，657份含身份字段；仅加载 `query_ids` / `question_ids`。另投影672份metadata、manifest、protocol、completion、budget、state或progress JSON的身份及用途字段。两类扫描均无读取失败。

BEIR test ID由 `data/beir/hotpotqa/qrels/test.tsv` 首列取得：7405个不同ID；ID集合摘要为 `c4e82cbc7873610cba32b04c7aab821e14e2037517da94511e1ae4d76a8337e8`。未解析qrels其余列。

| 命中的运行 | 命中NPZ | 每份与test的ID交集 | 保存成员表明的用途 |
|---|---:|---:|---|
| `beir_hotpotqa_four_condition_full_v2` | 4 | 7405 | BM25/BGE Dense候选，以及两路BGE重排分数 |
| `beir_selected7_dense_baselines_v2` | 2 | 7405 | DPR和Contriever完整候选列表 |
| 其余当前身份NPZ | 651 | 0 | 未发现test ID的Router动作或答案效用缓存 |

上述6份命中缓存只含ID与检索scores/vector_ids/indptr/timing等成员。没有读取这些分数。新定位的[矩阵元信息](../../outputs/beir_selected7_dense_baselines_v2/metadata.json)为completed、test、全量运行；HotpotQA的[BGE](../../outputs/beir_selected7_dense_baselines_v2/units/hotpotqa/bge/metadata.json)、[DPR](../../outputs/beir_selected7_dense_baselines_v2/units/hotpotqa/dpr/metadata.json)、[Contriever](../../outputs/beir_selected7_dense_baselines_v2/units/hotpotqa/contriever/metadata.json)三个条目均completed且num_questions=7405。BGE没有另一份候选NPZ，不把它误报为本轮找到的第7份身份缓存。

[矩阵执行器](../../scripts/run_beir_dense_matrix.py)的 `_score`（316行起）只计算各cutoff的BEIR检索指标；原[四条件执行器](../../scripts/run_beir_suite.py)调用[检索行构造器](../../src/evaluators/beir_suite.py)，同样是retrieval-only路径。这些命中不是答案生成或答案F1评价的证据。

## 2. 与已消费来源的交叉

| 对照来源 | query规模 | 与test的query ID交集 | group层面本次能确认的内容 |
|---|---:|---:|---|
| 内部主体 / BEIR train | 85000 | 0 | 至少8个group共享；完整交叉数未知 |
| 旧开发池 | 9600 | 0 | test逐ID group映射缺失，不能仅按ID宣称隔离 |
| 完整扩充池 | 7885 | 0 | 同上 |
| natural-2000 | 2000 | 0 | 同上 |
| fresh4000 | 4000 | 0 | 同上 |
| 已验证Pool | 12566 | 0 | 同上 |
| 本次已消费BEIR dev子集 | 5209 | 0 | 与test共享group为0，来自保留的原身份排除标记 |

对应身份文件分别可用 `layer_pooling_v1/features.npz`、`expanded_features/expanded_7885_features.npz`、`phase28_holdout_selection_v1/holdout_features.npz`、`validation_fresh4000_e03b_e1_v1/actions.npz`、`validation_pool12566_NI_v1/actions.npz`、`m6_beir_validation_v1/identities.npz`复算；完整路径和每份ID集合摘要保存在本次JSON。

新的group边界来自原[dev身份归档](../../../work/router_research/beir_dev_group_identities.npz)：5447条dev中21条、21个不同group同时出现于test；其中8条、8个不同group又命中内部85000。因此**至少8个test group已经与内部来源共享**。这是通过dev连接得到的下界，不是test与内部来源的完整交集。原5209合格子集已经排除全部test共享group，所以其group交集为0。

原[audit_beir_dev_eligibility.py](../../../work/router_research/audit_beir_dev_eligibility.py)在内存中建立过test group键，但只落盘dev的逐ID键和交叉标记。当前未找到7405的已保存逐ID group哈希；本轮遵守不再读取gold的限制，没有从参考元数据重建它。也未据不完整交叉计数挑选新的验证子集。

## 3. 资格判断与尚缺证据

1. **答案用途：** 本次672份用途元信息和657份身份NPZ中，没有匹配到test7405的答案生成、答案效用分析或Router策略预测记录；命中的是上述两个检索运行。这个结论只覆盖已列库存，不等于证明所有历史答案使用不存在。未读取JSONL结果/答案payload，也未覆盖远程、工作区外、未记录人工查看或其他命名格式的身份存储。
2. **group隔离：** 需要先取得或在另行明确的访问边界下重建test逐ID group哈希，再与全部已消费group统一比较。现有8个桥接group下界已经说明“test与train ID不交”不足以保证独立性；现在不能给出合格剩余N。
3. **选择使用史：** DPR/Contriever/BGE矩阵增加了已知检索比较记录。还须核实这些检索比较是否参与了检索器、encoder、重排、top-k或当前评价条件的选择；completed元信息无法回答决策依赖。未发现答案F1记录不能自动消除这种选择历史。
4. **来源映射：** 仍沿用已有官方文档与本地归档摘要支持；本轮没有新的官方原始ID逐条对应证据，也没有复现原始拆分算法/seed。

因此当前只可列为“**已有多轮检索使用记录，未在限定库存找到答案效用使用，group及选择史待补齐的候选确认资源**”。本次不将7405整体或任何剩余子集批准为新的独立验证集，不启动采集，也不改变现有Router结论。

## 4. 复算约定

- test集合：TSV首列去重。所有query交叉：`len(set(npz[id_member]) & test_ids)`，不加载其他NPZ成员。
- 已保存group事实：仅用dev NPZ的 `group_key_sha256`、`overlaps_old_group`、`overlaps_BEIR_test_group`、`no_either_group_overlap`。21与8均对对应哈希去重计数；5209部分取后两标记交集得到0。
- 元信息使用白名单字段投影；JSON不输出指标、问题、参考答案或生成答案。完整路径、文件摘要、成员名及扫描缺口保存在同目录JSON。文件名/源码搜索仅用于定位运行用途，没有读取新的逐query检索结果。

新增网络、API和模型拟合次数均为0。以上是有限库存审计，不是新的效果实验。
