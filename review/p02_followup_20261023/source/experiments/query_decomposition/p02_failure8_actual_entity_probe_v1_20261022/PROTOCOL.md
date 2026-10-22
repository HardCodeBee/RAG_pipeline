# I8固定节点的实际实体候选接口

只取已完成I8的同8个ordinary节点，沿原顺序，6个局部QA失效与2个合理缺证弃答边界分列。不换题、不看gold选择输入、不改变真实R02/v12分解。此为暴露题目的局部开发探针，不是完整RAG、正式质量成绩或最终F1验证。

- B0：完整原始payload，严格复用I8本轮fresh I0的8个完整stop、匹配requested/returned model及三项整数usage回执；不会提交B0，不导入I1输出或它的核对句。历史正式回执也不作B0替代。
- B1：原Current subquestion保持原样。在同一Actual task中紧接其后追加已审核R15 candidate block，候选原样重复原current。
- B2：相同位置、相同block，仅候选文本改为同source节点已有rewritten_query。原Current subquestion始终权威；候选不是金标，不保证正确实体化。

block逐字沿用R15：

```text
Candidate interpretation (not a replacement for the Current subquestion):
{candidate}
Treat this interpretation as a candidate only. Resolve factual or role conflicts using the Original question, the Current subquestion and retrieved passages about the matching entities.
```

B0→B1测包装/重复问题的影响；B1→B2比较已有候选解释文本。原current、原题、history、reader及顺序、4个fewshot和其他主体指令、raw分解、模型/设置与128上限固定；不启用formatter，不重检索、不重跑goal/final。只复制已有rewrite，不从诊断标签或gold填实体。

24逻辑条件中，B0的8条为零新增费用的精确复用；B1/B2的16条有5对文本不同、3对相同，对应13个不同新payload。3对相同payload必须在本实验同DB共享实际回执（含算法terminal），不能冒充独立重复模型抽样。完整共21种wire payload，但只有13种计划新增。B1/B2顺序沿R15的固定seed/case哈希轮换。

B0源费用单独列为已在I8记账的复用归属成本，不进入新增支出。候选新增实际费用由既有NoCapBudget累计账本记录，所有unknown/retired预留保留。共享provider gate一次1请求、开始间隔3秒；SDK/HTTP重试0，最多5次实际技术尝试含initial并跨restart持久化。算法length/empty/refusal/content_filter仅在匹配model、三项非负整数usage且total=sum时记terminal，不择优重抽。缺usage、错误model等保留技术held。

独立CandidateAsk继承I8的回执/账本crash-window guard，并额外将单独非空或空字符串的refusal/finish_reason返回字段（只要非null）视为已有返回迹象，禁止盲重发。I8和共享runtime原文件不改。B1/B2只查本实验相同完整payload缓存，seed=None；不从R15、I1或历史正式结果导入候选回答。

费用异常继续按对应I8 I0单阶段3倍、最近20个实际不同返回payload的费用均值1.5倍记录提醒；这是阶段开销，非完整query均值。相同payload逻辑复用不重复计入费用趋势。无固定预算上限。

先冻结8输入、B0全部回执、公开source/raw证明、24纯渲染payload与可机检单因素合同。prepare/dry-run只读源SQLite，不构造Cache或provider客户端、不访问网络。fake验证只用临时SQLite/假客户端/假账本。run另需root审核并创建精确protocol/control hash授权文件，本准备任务不写RUN_AUTHORIZATION，也不执行run。主评保持暂停。

```powershell
python -B -m experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.prepare prepare
python -B -m experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.prepare dry-run
python -B -m experiments.query_decomposition.p02_failure8_actual_entity_probe_v1_20261022.validate_offline
```
