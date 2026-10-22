# 普通QA已知失效/弃答边界8节点：离线准备

固定采用`ORDINARY_FAILURE_MECHANISMS.json`既有8案例，按原顺序全纳：6条明确局部陈述、关系、角色或提取问题，2条合理弃答边界。不按本轮新输出或后续结果重新选择。共16条件，是反复暴露的开发机制挑战，不是总体准确率、完整RAG收益或独立确认。

I0是原child完整payload的同期新鲜控制；I1仅在原主指令末尾、`Worked examples:`之前增加已确认的一句：

> Before answering, verify which entity and relation the Current subquestion asks about; preserve the relation's direction, requested answer type and explicit conditions, and correct a premise when the passages contradict it rather than restating that premise as fact.

实际Current subquestion、原题、history、reader、fewshot示例、顺序、原R02/v12支持问题数组、模型及128-token上限均保持。检索改写不是实际QA题，不把显式改写问题替换进实际问题；不插入人工修正或gold实体。所有历史formatter_request_id为空；此挑战不调用formatter、下游goal/final、分解器或检索，不做F1/EM提升声明。

2Wiki371的原分解已相对原题多走一代；即便当前问题关系陈述被修好，也不等于原题目标修复。Hotpot122局部QA把“尝试废黜”说成成功，但旧下游已恢复tried，原在位端点与企图日期范围问题保留，不能把局部修复直接算新增最终正确。HP27的开发者/引入者分离，HP31可排除Maury却未覆盖Bob投手正证，HP7候选正确电影在reader合并时丢失，HP50只有猎犬而无群猎关系；保留这些证据边界。两条弃答边界单独报告，不能混成reader充分的6条。

8/8原child全文可用当前renderer重现，完整wire请求匹配原正式round checkpoint的stop/usage回执。8/8原sourceSHA匹配此前验证公开manifest；8/8支持问题数组与原formal1500的实际R02/v12 remote_raw.parsed逐项相同。这里只复用已验证公开manifest身份，不新联网核验。新输入只复制原推理材料，机制备注/旧标签与模型消息分离。

本目录只有离线prepare/dry-run入口，无可调用的API runner，没有RUN_AUTHORIZATION，不构建客户端或Cache。执行由root待W90后决定。若执行，应复用既有串行gate、累计ledger、五次技术恢复、匹配模型及完整整数usage校验；I0不得以旧答案缓存替代同期新生成，I1与I0均保留终止失败与技术held，不选最佳输出。后续只先盲审当前子答的完整事实、响应性、必要信息和角色/条件；9组相同payload的既往R15不作为本8的独立证据。

```powershell
python -B -m experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.prepare prepare
python -B -m experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.prepare dry-run
```
