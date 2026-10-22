# 实际P02子问的候选实体解释接口：仅离线准备

用户目标是冻结R02/v12实际分解下，完整RAG的同题F1优于原P02；本实验只诊断子问接口，不替代完整RAG验证。不用人工60、不改训练或分解、不替换goal。

全部取既有15个有历史普通子问（每集5个、原顺序），不按分数筛。其中9个current与已有rewrite逐字相同，为直接点名负对照；6个rewrite有差异，为描述绑定候选。6个不等于6个成功实体化：FortHill只是描述重排；其余rewrite也只是未验证候选，不能作为正确实体金标。

三个arm保持原Current subquestion、reader、历史QA、fewshots、其他指令、模型及128输出上限；无formatter：

- R0：原prompt逐字相同，只复用同期fewshot实验E0对应15条完整回执，绝不为本组新发R0。
- R1：在Actual task内原Current subquestion之后追加下列块，candidate重复原current逐字。
- R2：同位置同一块，只把candidate改成该源node已经保存的rewritten_query逐字。

```text
Candidate interpretation (not a replacement for the Current subquestion):
{candidate}
Treat this interpretation as a candidate only. Resolve factual or role conflicts using the Original question, the Current subquestion and retrieved passages about the matching entities.
```

R0→R1估计包装/重复及附加谨慎说明的整体影响；R1→R2仅隔离已有rewrite解释文本。9个直接点名节点的R1/R2 payload相同，将严格复用同一回执，其相同输出不是模型独立稳定性证据。45逻辑条件，新增R1/R2共30条件、21个不同payload；不重复付费抽取相同请求。

准备阶段不读新E0答案，只冻结15输入、历史public source和模板。root执行freeze-e0时只读来源SQLite，在一致事务中核验每条实际payload、请求ID、匹配模型、stop+usage完整成功或完整的terminal算法回执。只要E0缺失/held，就拒绝整个freeze/launch，不改题、不重跑或选更好控制。源结果与回执单独冻结在E0_SOURCE15.frozen.json；root授权必须附其准确hash。

R0费用已由同期E0实验记录，本组只单列可归属的复用tokens/费用，不在新增实际费用中重复累加。R1/R2使用原StrictAsk、共享provider gate和NoCapBudget账本；SDK/HTTP0，技术失败最多5次含initial和restart，unknown与旧预留保留。算法length/empty/refusal/contentfilter仅在完整模型身份与非负整数usage成立时登记terminal，不择优/重抽。缺usage/model mismatch继续held。原主评STOP不变。

报告所有45条件的完成/算法terminal/技术held、文本变化与tokens/费用，另行遮蔽语义审查。没有final F1、没有外部评审API，局部改善不算达成完整RAG目标。实际启动由root审核后单独授权。

```powershell
python -B -m experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022.run prepare
python -B -m experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022.run dry-run
# root later, after15 E0 controls resolved:
python -B -m experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022.run freeze-e0
```
