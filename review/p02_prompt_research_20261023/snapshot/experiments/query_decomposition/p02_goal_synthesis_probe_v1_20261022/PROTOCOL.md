# Goal 输出协议：冻结前缀15题探索

主评估继续暂停。本目录是用户授权的独立小实验，不覆盖原P02或其他quick90结果。
只对原quick90每集5题重新生成goal与final，不运行分解、检索、GPU或评审模型。

先仅按冻结分解非空筛出eligible题，再于打开轨迹和参考答案前选择：seed20261022，已有题型每型至少一题，剩余名额按比例缺额分配，SHA256排序。eligible分母记入COHORT.frozen.json。开发题已被多次查看，不能称为独立确认。

三个goal条件均固定原普通QA、goal改写及双路合并reader材料、最终原query检索材料、同一Qwen配置。统一goal上限512，关闭formatter。G0是原事实句prompt；G1是80–120词事实摘要；G2是80–120词关键事实、必要组合、候选结论或缺口段落。长度为生成目标，不硬裁剪，实际长度另记。

三组final均采用同一中性“Candidate information for the Original question”包装，保留原普通QA和原query材料，原final prompt/examples/readout及64输出上限不变；所有新输出须stop且有usage。G0因上限、formatter与包装变化，不等于历史P02。历史端点单列。

G1/G2可能生成不同事实集合，因此这是goal整体输出协议比较，不能把分数差全部解释成推理文本的纯因果贡献。观察信息覆盖、属性/实体绑定、组合一致性、支持与反驳证据是否保留，并核对改进与损害案例。

缓存仅复用同完整payload、stop、有usage的成功响应。128上限旧goal不会当作512上限命中。继续运行复用成功；SDK/HTTP重试0，技术失败最多5次实际尝试，跨重启累计，获准重发的unknown旧回执与预留完整保留。禁止为提高F1重跑成功或算法失败题。无旧300元或单次硬预算上限，保留全局账务和本实验回执。

已预留两个独立后续条件，默认不执行：`--final-goal-none`只移除final中的goal候选段，不生成新goal；`--rewrite-off`将goal reader材料切为已保存的原query路线，其余前缀和final root固定。后者首轮只衡量此冻结前缀下goal补充检索的局部作用。

模型输入为`inputs15.frozen.jsonl`，不含参考答案、支持事实标注、历史final预测或F1。参考答案仅在报告阶段从本地reference文件join，原生F1/EM计分。不将本次网络等待或缓存耗时当作新的时间评测。

离线准备及检查：

```powershell
python -B -m experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022.run --prepare
python -B -m experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022.run --dry-run
```

只有root在检查授权、共享provider gate和运行条件后启动`--run`，并先写RUN_AUTHORIZATION.json使其decision_id对应共享决策记录。运行器支持`--arms G0 G1 G2`，一次只由root启动一种新实验。共享provider gate与无旧预算上限账务借用binding试验的shared_runtime。所有输出位于`outputs/query_decomposition/p02_goal_synthesis_probe_v1_20261022/`。
