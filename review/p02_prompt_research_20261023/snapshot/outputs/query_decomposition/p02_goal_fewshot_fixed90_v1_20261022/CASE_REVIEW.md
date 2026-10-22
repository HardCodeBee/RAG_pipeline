# 固定90题 goal 提示词实验：案例复核

以下案例用于解释实测差异，不构成对所有输出逐句事实准确率的估计。完整实际goal文本、最终回答、原生分数和冻结检索材料见CASE_REVIEW.local.jsonl。

## 保留前序关系事实：few-shot 的正例

`2wiki:654dbe680bb011ebab90acde48001122`

原题：Who is Hermann Otto I Of Limburg-Styrum's paternal grandfather?

G2和I均弃答；F使用前序QA中“父亲的父亲＝Hermann Georg”得到正确答案。当前材料未重述这条关系，不应自动视为反证。F同时有冗长、循环和弱证据归因，因此最终正确不等于整段解释均可靠。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Cannot be determined | 0.0000 / 0 |
| I | Cannot be determined | 0.0000 / 0 |
| F | Hermann Georg of Limburg | 1.0000 / 1 |

## 答案修复，同时出现错误的引文归因

`2wiki:079947de0baf11ebab90acde48001122`

原题：Who is Princess Elisabeth, Duchess Of Hohenberg's father-in-law?

G2因当前检索未再次确认而弃答；I/F正确使用前序QA中的公公身份。然而两者又把Maximilian自身父母的传记误写成支持他是Franz父亲的证据。正确性收益不能直接解释为完整证据链得到验证。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Cannot be determined | 0.0000 / 0 |
| I | Maximilian, Duke of Hohenberg | 1.0000 / 1 |
| F | Maximilian, Duke of Hohenberg | 1.0000 / 1 |

## 原题条件被擅自收紧：相近示例仍未防止回退

`musique:2hop__289517_139339`

原题：What was the cast member in After Hours nominated for?

问题通过After Hours指认演员，并未要求提名来自这部电影。I正确回答Griffin Dunne的Academy Award for Best Live Action Short Film提名；F看到同一条事实仍附加“因出演After Hours获提名”的条件并弃答。实际F请求含全部四个示例，包括直接针对不同电影提名的Example 2；返回stop，非缺模板或截断。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Cannot be determined | 0.0000 / 0 |
| I | Academy Award for Best Live Action Short Film | 1.0000 / 1 |
| F | Cannot be determined | 0.0000 / 0 |

## 过度要求完整字段而弃答

`hotpotqa:5ab64cfa55429953192ad2b7`

原题：Who lived longer Ellen Glasgow or N. Scott Momaday ?

F声称寿命比较必需双方出生、死亡日期，因Momaday死亡日期缺失而弃答。G2/I给出数据集正确答案。比较原则上可以通过下界判定，不必总知道死亡日期；本例应依据冻结材料的时间和生存描述判断。G2/I部分解释也粗糙，不能以答案满分证明推理均正确。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | N. Scott Momaday | 1.0000 / 1 |
| I | N. Scott Momaday | 1.0000 / 1 |
| F | Cannot be determined | 0.0000 / 0 |

## 数值表达造成原生分数下降

`hotpotqa:5ab8c3c65542991b5579effd`

原题：How many albums did the company who released Tift Merritt's debut album release for Tif Merritt?

G2回答Two而I/F回答2。三者表达同一数量，原生Hotpot计分却从F1/EM=1降为0。保留原生评分，同时把这项变化与事实性退步区分。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Two | 1.0000 / 1 |
| I | 2 | 0.0000 / 0 |
| F | 2 | 0.0000 / 0 |

## 额外日期造成答案匹配损失

`hotpotqa:5a71474b5542994082a3e74f`

原题：What was triggered by the resignation of Conservative MP Stephen Phillips on 4 November 2016 and was when Caroline Johnson became a Member of Parliament?

三组指向同一场补选；I/F额外给出日期，原生F1由1下降到0.7143/0.6667，EM下降。它说明需要控制最终短答案粒度，不足以证明组合事实错误。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Sleaford and North Hykeham by-election | 1.0000 / 1 |
| I | The Sleaford and North Hykeham by-election on 8 December 2016 | 0.7143 / 0 |
| F | Sleaford and North Hykeham by-election held on 8 December 2016 | 0.6667 / 0 |

## 512-token硬上限截断

`musique:4hop3__275416_24325_156850_10557`

原题：Despite being located in East Belgium, the Carnival of the birth place of Guido Maus harks purely to an area. What was the language having the same name as this area of the era with Fastrada's spouse's name later known as?

F的goal输出达到512 tokens、finish_reason=length，未进入final；按预先协议保留为算法失败，F1/EM计0。G2回答Latin（F1=0.6667），I回答Walloon（0）。不通过增加长度或重试改写这一轮结果。

| 版本 | 实际最终回答 | F1 / EM |
|---|---|---:|
| G2 | Latin | 0.6667 / 0 |
| I | Walloon | 0.0000 / 0 |
| F | [算法失败，无final输出] | 0.0000 / 0 |

