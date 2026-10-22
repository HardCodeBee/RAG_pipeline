# 事实句goal × 角色核对句：仅离线准备

此补充分支检验：在原G0事实句协议下，追加角色核对句能否获得类似于长综合段落的收益。完整RAG同题F1提升才是最终目标，局部句子标注或被选挑战集的改善不代表目标达成。

全部沿用原诊断挑战27题（2Wiki17、HotpotQA10），不按新分数筛选。保持已有实际分解D、普通QA、goal改写及双路reader、final原query材料不变；不用人工60，不改训练、不重新分解/检索。

- T0：共同协议下原G0事实句prompt逐字相同，在本新目录重新生成同期控制。
- T1：仅在T0主指令末、唯一Original question输入标题之前追加原S2的exact ROLE单句；原goal指令、输入标题及所有材料逐字不变；这27个formal G0本就没有fewshots或Actual task区块，不新增它们。

```text
Before answering, check that each named entity keeps its stated role in every relation and that the final conclusion preserves the original subject, object, relationship direction, generation and requested attribute, including the distinction between a parent and a grandparent.
```

两臂共同goal512、final64，无formatter，final使用原中性候选信息包装。原G0的“一句事实句、不写推理”要求继续保留。本轮是512共同协议适配，不能称原P02原生128-cap复现或把旧128/旧512控制当新同期控制。

共54条件、最多108个逻辑阶段payload（每条件goal及final）；相同完整final payload可在本实验内精确复用，真实技术恢复另按最多5次实际尝试计入持久账务。原G0与已完成G2的S0/S2仅作已曝光的历史参照，不导入缓存；旧G2不在本次同期随机交错运行。因此这只是分阶段补充的探索性设计，不能单凭两次差分证明格式×ROLE的确认性交互或跨时间因果机制。

固定seed20261022，按冻结case序在T0/T1间轮换先后，首位各14/13题。所有结果、损害、terminal和技术held均保留，不择优、不重试算法失败。共享原StrictAsk、provider gate、NoCapBudget：SDK/HTTP0，技术失败最多5次含initial及restart；unknown原回执、retired和负债保留。成功及算法terminal都要求请求/返回模型身份一致、usage非负整数；只有完整归属的length/empty/refusal/contentfilter记条件算法结果，在固定计划分母计F1=EM=0，goal终止则不发final。缺usage、模型错配、technical unknown单列held，不静默归0。

结果分别报告每数据集及整体固定计划分母、正常完整输出敏感性、配对F1/EM、输入/输出tokens与费用/未知预留。每数据集scorer只构造一次；报告只读SQLite，references仅本地join，不发送模型。没有新增LLM调用数指标。

输入与已核定公开来源证明按原27题字节复制，publicproof不声称本轮重新联网查证。准备只写模板、输入、渲染合同与协议，不发API，不生成启动授权。root收到W90及阶段审查后再决定是否执行；本目录存在不代表执行获准，主评保持暂停。

```powershell
python -B -m experiments.query_decomposition.p02_goal_fact_role_probe_v1_20261022.run --prepare
python -B -m experiments.query_decomposition.p02_goal_fact_role_probe_v1_20261022.run --dry-run
```
