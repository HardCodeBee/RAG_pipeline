# Goal组合挑战集27题：仅准备

从既有723条完整诊断记录中，按“普通子回答全部正确、历史goal回答错误”选择全部27题：2Wiki17、HotpotQA10。没有按历史final分数再筛；原final成功的5题（2Wiki2、HotpotQA3）与失败题共同保留。

这是根据已观察错误选择的机制挑战集，不能用来估计一般化收益，不能称为独立确认或主评成绩。历史标签只说明选择原因，不输入新生成prompt。

“普通子回答正确”沿用原诊断定义，允许正确但不完整的事实与合理弃答，因此不保证所有必要桥接事实都已取得。这27题也不能直接等同于27个纯组合错误；后续需区分缺信息、关系/属性绑定、组合推导与最终信息使用。

复用goal pilot15冻结的G0/G1/G2模板与同一中性final包装，goal512、final64，关闭formatter。固定普通QA、原goal改写、双路合并reader材料及final原query材料；不重新检索、不跑分解、不使用GPU、不调用评审模型。G1实际可能也生成推理，因此比较的是整体输出协议。

模型仅收到inputs27.frozen.jsonl中的白名单字段。原native references从诊断STAGE_PACKETS的reference字段逐题核验身份后单列到本地references文件，只在报告阶段join；不构造或推测gold。原goal错误与历史final端点存在独立selection/history文件中。

每条source的SHA256必须匹配已公开SOURCE_MANIFEST中的formal1500 tar成员。原tar的Git blob通过公开GitHub树API核验，本地tar按Git blob算法重算一致，再逐条解出成员核对source原始bytes。证明见PUBLIC_BLOB_METADATA.json及PUBLIC_SOURCE_PROOF.jsonl。

本目录首次仅离线prepare/dry-run。后续执行必须由root补RUN_AUTHORIZATION.json并核对共享DECISIONS记录；共享StrictAsk负责最多5次实际技术恢复、完整输出与usage校验、跨进程gate和无旧固定预算上限账务。unknown回执和费用预留完整保留，主评保持暂停。

三个条件共81条goal/final流程，最多162个不同阶段payload，实际严格同payload成功缓存可减少新增请求。费用、tokens与未知预留分别保留，不增加LLM调用数指标。

```powershell
python -B -m experiments.query_decomposition.p02_goal_composition_challenge_v1_20261022.run --prepare
python -B -m experiments.query_decomposition.p02_goal_composition_challenge_v1_20261022.run --dry-run
```
