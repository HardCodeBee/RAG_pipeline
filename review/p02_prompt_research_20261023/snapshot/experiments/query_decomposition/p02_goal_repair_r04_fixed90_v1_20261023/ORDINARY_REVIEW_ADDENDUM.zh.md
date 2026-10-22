# 对 r03 普通子回答的追加复核

这是2026-10-23新增的离线复核，不修改 r03 结果，不向本轮 goal 历史注入新回答。由答案变化做定向筛选后查看实际 reader 的四例；不是随机样本，不计算总体错误率。此前预选18例的“主要答案相同”仍成立，但不能用它概括全部176节点。

## 1. 分解本身可能改了所求量，而不只是格式

`hotpotqa:5a70f9335542994082a3e46a/ordinary/2`

原问题比较两本杂志一年出版次数；冻结子问题写成“How many copies of Alt for Damerne are published in a year?”。reader 明确是 weekly。旧回答将它解释为期数，回答每年52期；r03 强调 copies 为总印量，保留 weekly，但拒绝给年度份数。

这不是 `#N` 能否解析的问题：**频率/期数与印刷份数是不同的量**。从子问题字面看，r03 对 copy count 的保守处理有依据；从原问题意图看，旧回答提供了更可用的比较信息。因此不能把这种分数损失一概归为子回答事实错误，也不能无条件要求回答器只服从分解字面。未来若单独修普通 QA，需要同时检验“借原题消歧”与“越界直接回答原题”的边界。

## 2. 额外证明要求会丢掉可能有用的桥接实体

`2wiki:923b15ba0bb011ebab90acde48001122/ordinary/1`

问 Araripe Júnior 的父亲。旧回答给 Tristão de Alencar Araripe；r03 弃答。reader 只有本人全名、日期及无关人物，没有父子关系句。因此当前 reader 确实缺这条关系，不能称 r03 漏读。旧研究曾核对并保留该父名，而本次只说明**桥接实体被替换成缺口表达**；未重新联网核验旧父名。

合理弃答可以满足已有“子回答正确”的标签，却可能降低下一轮检索的信息量。这里应同时记录事实正确性、材料支持程度和桥接信息是否仍可用。

## 3. 原问题的后续目标反过来改变当前实体选择

`musique:2hop__289517_139339/ordinary/1`

当前只问《After Hours》的演员。reader 第一篇是1985年电影并明确 Griffin Dunne；第五篇是 Bruce Hunter，提到他因同名电视剧获得加拿大喜剧奖提名。旧回答 Griffin Dunne；r03 的 Support 因原问题还问“提名”，选择了在当前材料中同时出现提名的 Bruce Hunter，并提前把奖写入 Answer。

材料中存在同名作品，而问题没有在字面上给年份，不能只因不是数据集参考就声称 Bruce 的全部陈述为假。可确认的是：**候选实体选择被当前可见的下游属性牵引，且普通答案提前完成了后续任务**。在固定旧普通历史的 goal 试验里可以修复 Griffin 的提名关系，不代表换上 r03 普通回答后仍会沿同一条路径。

## 4. 无依据添加“现代国家”限制

`musique:2hop__104095_40501/ordinary/1`

问 Clara Peeters 的公民所属国家。reader 说明她来自 Antwerp、可能主要在 Dutch Republic 活动，且生平存疑。旧回答 Dutch Republic；r03 改为“citizenship in a modern nation-state cannot be determined”。原题与当前子题都没有 modern/present-day 限制。

材料确实没有明确公民身份，不能把旧回答当作已由 reader 证明；但新版也确实自行增加了现代国家条件。应把“原事实缺支持”和“新添条件”分开诊断。

## 对当前修正的影响

这四例补强了暂不推广 r03 普通 QA 的理由，但不证明旧普通 QA 全面正确。当前 r04 保持冻结的 role_headers 父版本普通历史，只测试 goal，避免把上述实体、数量和证据政策变化混入本轮比较。生产实现从未被 r03 覆盖，本次不存在生产回滚。

证据路径：`outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/CHANGED_ORDINARY_NODES.local.jsonl` 中对应节点的原题、当前题、旧新输出、完整 reader。该文件及旧标注保持原样。
