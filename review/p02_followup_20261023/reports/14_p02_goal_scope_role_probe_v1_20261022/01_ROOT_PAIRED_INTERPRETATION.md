# S组27题最终F1变化的逐例解释

本次只读解释全部配对变化：S0→S1有5条、S0→S2有3条，涉及5个独立问题。81段goal的语义复核均已先锁定，再与arm及native/final合并；已有标签与分数不改。单一知情审阅者并非独立人工金标。27题来自已暴露、诊断挑选的开发挑战（2Wiki17、HotpotQA10），不能视为相对原P02整体F1的证据。

| arm | F1 | EM | 相对同期S0 F1 |
|---|---:|---:|---:|
| S0 | 40.3479% | 33.3333% | — |
| S1 | 43.2099% | 40.7407% | +2.8620个百分点 |
| S2 | 45.9035% | 40.7407% | +5.5556个百分点 |

已有配对bootstrap描述区间S1为[-10.1010,+15.9933]百分点，S2为[-3.7037,+18.5185]百分点，均跨0；它不消除挑选及反复开发暴露。

| 问题 | 对照 | 原final → 新final | F1变化 | 解释 |
|---|---|---|---:|---|
| Peter Grubb Jr.的祖父 | S0→S1 | Peter Grubb → Cannot be determined | .5→0 | 避免把父亲当祖父；reader内合理弃答，真实John及同名关系仍待核 |
| Peter Grubb Jr.的祖父 | S0→S2 | Peter Grubb → Cannot be determined | .5→0 | final避免代际错，goal开头仍称Peter为祖父，整段未修复 |
| Made in U.S.A.导演的工作地 | S0→S1 | Swinburne与Tongji → Swinburne | .7273→1 | 参考粒度/删去额外机构；导演与同名学者身份未证，不算真关系修复 |
| Woman in the Fifth与Evensong同国 | S0→S1 | No → Yes | 0→1 | 共有UK运算恢复；goal开头仍No，不能说全文全对 |
| Woman in the Fifth与Evensong同国 | S0→S2 | No → Yes | 0→1 | 共有UK运算恢复；仍加primary/sole等未问条件 |
| Nutcase与Mother Fish导演同国 | S0→S1 | Yes → No | 1→0 | from/出生/国籍混用，缺证当排除；恰为旧重复唯一final漂移题 |
| Ballad Prince主演剧的state-run频道 | S0→S1 | Cannot be determined → SBS | 0→1 | 频道正确可见，但S1明说原题未问state-run，实际删掉显式条件 |
| 同上 | S0→S2 | Cannot be determined → SBS | 0→1 | 保留频道并承认条件冲突；网络所有制不在材料，不能确证原前提真假 |

祖父题中ordinary2已经提出John Grubb，reader给出的年轻Jehu/John不能反证所有同名John；因此合理弃答不等于证明真实答案未知。工作地题在身份成立的前提下短列表可符合既有“不要求穷尽”规则，但取首位为主雇主没有来源支持。SBS题的来源只给频道及日期，KBS/MBC公有性质与SBS私营性质不能用模型断言替代来源。共享UK是最清楚的final运算修复，仍需保留goal内部矛盾及独立/唯一来源等附加断言的旧标注。

## 非同期同请求重复的边界

旧G2→本S0：27/27的goal完整wire请求相同，16段goal文本变化，final只1题变化（Nutcase/Mother Fish，旧No→新Yes），均值差+3.7037百分点。final输入随goal变化，并非27个同一final请求重复。该观察包含时间、服务与生成变化，不是纯随机误差/方差估计，不据此机械扣除本轮收益。同一题本轮S1又回No，使其单次归因尤其薄弱。

这些结果支持继续检验共享属性运算和目标字段保持，但也表明“提分”等于“推理更正确”不成立：F1收益可来自参考范围匹配或条件取舍，F1损失也可来自避免明显错误。不能由27题排序宣布S2全体更好；以冻结原90的同期最终F1对照继续验收。

完整原题、逐臂goal、实际final、原native数值、锁定盲标和reader摘录见[机器可读逐例报告](ROOT_PAIRED_INTERPRETATION.json)。输入sourcehash和48+30+3条复核锁已校验；本操作无API、新检索或GPU，不读取E未锁输出。
