# P02 主270条信息流离线复查完成说明

主cohort的材料复查现已全覆盖：**230条非满分＋40条固定seed满分对照＝270条**。其中258条完成各自声明的必要信息/路径检查，12条保留事实或口径待核。剩余非满分为0。8条补充包只保留原有案例发现，本轮复查尚未启动，不计入270完成数。

这是对既有723条诊断轨迹中目标cohort的离线复查，不是新的正式RAG评估，也不代表所有附加事实都已重新做独立金标。原F1/EM、truth标签与主评保持不动。

## 三批交付与验证

V7追加007、V8追加008、V9追加009，每批40条，与V6冻结任务包逐ID和顺序相同。新120条校验120个原始sourcehash、281个actual child_prompt当前题、270条reader/候选/goal摘录位置。摘录匹配只核位置，语义结论来自逐条亲读，不能以关键词命中代替充分证据。

原V6的150条记录在V9中逐字段完全一致；原64诊断与先前follow-up文件保留原字节。008/009的字符串待核备注只在新聚合版封装为结构化对象，原文及来源batchhash保留，原batch文件不改。

## 已核对的具体边界

- HP7、Wiki721：必要事实在候选中但被既有Top10合并选择丢弃；这不证明一律增加文档会提升F1。
- Wiki506、Wiki248：reader已有正确父亲或日期关系，QA却取错角色或弃答，必要信息未保留到helper/final。
- Wiki362、Wiki376、Wiki390、Wiki392、Wiki528、Wiki566：双方属性或日期已知，goal仍改变交集条件、反向比较或多追一代。
- HP187、HP222、Wiki550、Wiki541、Wiki403：目标属性或关系已在goal/helper，final仍取桥梁组织、错误国家、发布商或相反结论；部分含冲突goal，不能仅用增加可见文本解释。
- Wiki517、Wiki527、Wiki533、Wiki719：人物奖项被增加只限特定作品的条件，或正确奖项已可见仍被final弃答；533还含reader到goal传递丢失。
- HP141、HP150、HP152、HP189、HP126：goal或final也会纠正上游错误/歧义；这些恢复是后续改动必须检查的回归面。
- HP114、HP123、HP145、HP181、HP189、HP237、Wiki433、Wiki456、Wiki463：行政单位、有效替代答案、错误桥梁参考、HTML破损、布尔范围、物种证据、来源冲突和同名身份分列；不重写原指标。

这些案例支持将修复方向拆到证据选择、QA关系保留、goal综合、final目标字段选择等冻结对照中验证；复查本身不能推出某环节因果效应最大，也不能声称已有修复带来F1提升。

## 待核及真实QA输入

待核12条，保留已有truth/原生分数：Wiki317、254、319、386、518、519、520、531、570、619、656、674。涵盖既有出生/死亡地点与附加职业事实，以及历史国籍、国家归一化、创作credit、雇主时间和同名身份等范围问题，详见 `INFORMATION_FLOW_PENDING_WORK_ITEMS_V9.json`。

HP107以独立澄清为准：`child_prompt` Current subquestion仍是原描述问题，检索`rewritten_query`不是实际QA题。Licking县治Newark是真命题，不等于正确响应真实球队限定题；旧ordinary3=False保持。

## 分解来源与未来目标

007的40条已逐ID核对正式remote_raw：历史RAG的supporting_questions与R02_expanded/v12原始输出数组完全一致，没有语义修复。详见 `DECOMPOSITION_SOURCE_BINDING_BATCH007.json`。历史same60人工query-only修正及revision46另存，不能称当前270正式轨迹已使用修正分解。未来“修正后的分解”具体入口由root确认；本复查没有训练、推理、调用API或恢复主评。

## 当前入口

- [标志语义澄清与固定前5条实证](FINAL_FLAG_SEMANTICS_AUDIT_FIRST5.md)：非满分中120条“final充分且用对”只指声明范围内的必要信息保留/使用，不能解释为完整回答正确；147条充分也不是可修复数量，26条未用对不是总体因果占比。这里包含粒度、别名、有效替代答案与参考问题，原F1/EM保持。
- `CURRENT_REVIEW_VERSION.json` 指向V9，保留HP107澄清。
- `MANUAL_INFORMATION_FLOW_REVIEWS_V9.json`：270条逐条范围、证据和信息流观察。
- `REVIEW_PROGRESS_V9.json` / `REVIEW_INDEX_V9.jsonl`：状态；`REMAINING_NONFULLSCORE_QUEUE_V9.jsonl`为空。
- `PRIMARY270_REVIEW_COMPLETION.json`：完成范围、互斥批次、校验数字、待核与案例索引。
- `REVIEWED_SUCCESS40_NONFULLSCORE230_DESCRIPTIVE.json`：描述性分组；不输出总体原因占比或因果排名。
