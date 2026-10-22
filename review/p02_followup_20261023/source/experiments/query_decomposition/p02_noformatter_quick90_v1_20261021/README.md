# P02 去formatter：开发快速90题

用户授权：从开发300题按每数据集30题建立快速评测集，并测试删除额外formatter的P02。

- `COHORT.frozen.json`：冻结ID、seed20261021、按原题型/跳数比例的最大余数分层抽样；选择只用dataset/type/id。
- `runtime90.frozen.jsonl`：原始query与冻结R02分解；可复用为后续快速测试的输入。
- `quick90.questions.frozen.jsonl`：仅题目及身份元数据。
- `references90.local.frozen.jsonl`：离线原生F1/EM参考，禁止进入方法执行。
- `pipeline.py`：独立去formatter变体；其余算法、prompt、goal、检索合并和输出上限均沿用原版。
- `PIPELINE_CHANGE.diff`：相对冻结原版的代码差异。

因历史开发300题没有触发formatter，本次使用严格完整请求匹配、同题检索及回执绑定的完整流程回放。缓存缺失即失败，不提交付费请求。结果明确标注为回放等价验证；不将其称为新推理、独立泛化、完整500题主评或新统一层结果。

运行固定版本：

```powershell
python -B -m experiments.query_decomposition.p02_noformatter_quick90_v1_20261021.run_r002
```

已有结果不会覆盖。输出位于 `outputs/query_decomposition/hotpot_formal_eval_v1/p02_noformatter_quick90_v1_20261021/`。

合成短答案检查只验证原版会触发formatter、变体不会触发；不计入质量成绩。原生F1/EM仍使用既有HotpotQA、2Wiki aliases/demonyms及MuSiQue aliases口径。

冻结P02原包、正式500题和既有计时50题名单保持原样；主评与其他任务的暂停标记不清除。

执行修正001：保留首轮run.py及停止记录；r002按冻结原包的回放规则核对历史节点已存字段，同时严格比较两版新执行的全部节点字段。仅纠正历史导出省略审计字段造成的误报，题目与算法不变。
