# Goal目标范围与实体角色约束：离线冻结，等待root审核启动

全部沿用上一轮挑战27题（2Wiki17、HotpotQA10），不按final分数再次筛选。每题三个条件，共81条件：S0为原G2逐字相同的新鲜同步控制；S1仅在G2指令末尾追加目标/条件保持句；S2仅追加实体角色/关系链核对句。S1与S2独立，不叠加，不改final，不挑最佳输出。准确英文句子与解释见CONSTRAINTS.frozen.json。

S1检验是否出现“将原问题没有的限制加入求解目标”：人物可用作品识别，但不意味着其奖项必须因该作品获得；问共享属性不等于要求全部属性集合完全相同。S2检验实体在关系中的主客体、方向、代际和所求属性能否保持，例如父亲与祖父。它们都可能影响整体输出协议，不宣称只改变一个不可观测心理机制。

固定原普通QA、goal改写与双路reader、final原query材料、模型和中性候选包装。goal512、final64，不运行formatter、分解器、检索或外部评审模型。原生references单独保存在本地，只报告时join；历史错误标签与gold不送入模型。

每题按固定seed20261022生成的循环顺序执行三个arm，各arm在9题首先执行。只允许本新实验内严格相同完整payload的stop+usage缓存用于续跑/去重；不读旧G2缓存，因此S0为新鲜同步控制。历史G2已经曝光，只能辅助观察时间/随机变化，不能作为独立确认。原27题也由已观察错误选择；全部结果必须报告，不能估计一般化收益或晋升主方案。

共享StrictAsk：SDK/HTTP重试0，每个技术失败逻辑payload最多5次真实尝试（含initial与restart）；unknown回执和费用预留保留，仅沿既有明确授权的无任何返回迹象分支有限恢复。成功完整输出须stop、非空、无refusal、usage完整。length/empty/refusal是不可重试的条件结果，保存完整回执；goal阶段发生则不再发final，在固定计划分母计F1=EM=0，另报正常完整输出敏感性。基础设施unknown、缺usage、模型错配等暂留工作项，不静默当0。report只读SQLite，每个数据集scorer只构造一次。

provider endpoint、共享跨进程gate、ledger与无固定预算上限政策不变，保留每个query输入/输出tokens、合理费用及未知负债。不额外加入LLM调用数指标。主评保持暂停，原有STOP不变。

原公开来源证明与输入/参考完全按挑战集字节复制；源hash和public Git blob路径见PUBLIC_SOURCE_PROOF.jsonl。root已独立复验公开性。本次离线准备不发送API；实际执行需root写RUN_AUTHORIZATION.json，并记录共享DECISIONS，核对冻结hash后启动。

```powershell
python -B -m experiments.query_decomposition.p02_goal_scope_role_probe_v1_20261022.run --prepare
python -B -m experiments.query_decomposition.p02_goal_scope_role_probe_v1_20261022.run --dry-run
```
