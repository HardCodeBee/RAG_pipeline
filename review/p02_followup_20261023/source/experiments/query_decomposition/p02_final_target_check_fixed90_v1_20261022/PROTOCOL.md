# W0/W1：固定90题完整RAG最终输出对照（仅离线准备）

目标是检验一次final指令修改能否改善原P02完整RAG最终F1。固定原90题（每集30），保留实际R02/v12分解、普通子答、goal、检索、辅助回答和最终证据的所有字节；仅重新生成final。这里不是重新跑上游流程，也不是独立正式确认集：这90题已多次开发暴露。

W0使用原始完整final请求，在本实验新数据库中重新请求；不用历史答案缓存作W0。W1仅在原指令prefix末尾、`Worked examples:`之前增加以下已确认单句，原示例、材料、问题、生成参数及64-token上限保持：

> Before answering, check that your answer matches the entity, attribute or relation asked in the Original question and applies its requested comparison or combination, without adding conditions that the question does not state.

180条件均保留。每个case的W0/W1按固定seed散列轮换先后，题序不变。原P02 F1=0.7668887868810469、EM=0.6666666666666666只作历史参照；主要比较新W1与同期新W0。原source完整payload、stop与usage逐条验证，当前renderer逐字复现，且90个sourcehash与此前已验证公开manifest一致。离线准备不宣称重新联网核验公开仓库。

新checkpoint与`StrictAsk(seed=None)`保证原响应不迁入新控制。恢复只使用该checkpoint；每条技术失败最多5次，跨重启计数并保留旧reservation。共享提交gate、累计费用ledger和现有异常费用提醒沿用。length、拒答、空答、过滤在模型身份与usage有效时作为终止算法失败计0，不能为了分数重试。usage缺失、身份不符、未知结果或技术重试耗尽单列held；主表完整F1/EM必须每arm90条全部质量终止后才能报告。部分运行只报告固定分母上下界、覆盖和held，不能当完整90结果。

只在report阶段读取本地参考答案并各数据集初始化一次原生scorer；模型输入不读入gold。报告F1/EM、配对改善/持平/退化，完整流程逻辑输入/输出tokens及估算费用（冻结上游加新final），另列本轮final实付及未决预留；不报告LLM调用数，不据此次缓存上游运行声称完整流程耗时。当前主评暂停不变。

`prepare`、`dry-run`、`report`不提交请求。`run`另需root审核后写入绑定协议hash的RUN_AUTHORIZATION，且decision存在于主计划决策日志。当前未提供该授权文件。验收目标是完整最终F1对照，不把措辞变化或组件正确率当改进。
