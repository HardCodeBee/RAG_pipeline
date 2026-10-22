# 固定90题：子问题与goal阶段核查

两组各90/90完成，所有请求成功。A仅删除普通子问题prompt首句的`only`；B保留完整新版的普通子问题，固定上游答案与goal检索材料，仅删除goal的`Resolve each #N...`句。完整goal改写尚未执行，候选未启用，主评保持暂停。

旧P02指第一次修改普通子问题prompt之前的冻结版本；完整新版指本轮两组实验共同的对照。以下是全量逐项差异，不能当作错误数量。

| 相对旧P02 | 普通子问题阅读材料变化 /176 | 普通子答案文字变化 /176 | goal阅读材料变化 /90 | goal答案文字变化 /90 |
|---|---:|---:|---:|---:|
| 完整新版 | 4 | 53 | 9 | 27 |
| A：删only | 6 | 55 | 10 | 28 |
| B：删Resolve句 | 4 | 53 | 9 | 44 |

| 相对完整新版 | 普通子问题改写query变化 | 普通子问题阅读材料变化 | 普通子答案变化 | goal改写query变化 | goal阅读材料变化 | goal答案变化 |
|---|---:|---:|---:|---:|---:|---:|
| A | 3/176 | 3/176 | 47/176 | 9/90 | 7/90 | 24/90 |
| B | 0/176 | 0/176 | 0/176 | 0/90 | 0/90 | 35/90 |

各版本原始子问题、原始goal query及它们的首路检索文档均相同。差异来自历史答案影响query改写，以及回答prompt变化。阅读材料比较采用实际送入reader的有序文档及文本；原始与改写两路检索包另行保留。B冻结上游，因此不能用B评估普通子问题prompt是否改善。

## 子问题：已核实的具体问题

1. **Clemson冠军数，完整新版新增读表错误，A修回，B保留。** `musique:4hop3__129456_29339_508306_70744`，第7步。相同材料中表头依次为`Varsity Sports / NCAA Championships / Pre-ACC Championships`，Clemson对应`19 / 5 / 0`。旧P02答5，完整新版答0，A答5，B答0。完整新版把另一列的0读成所问冠军数。该错误没有改变最终比较的胜者，所以最终F1看不出来。
2. **Clara Peeters，A恢复答案但缺少当前材料支持。** `musique:2hop__104095_40501`，第1步。材料说她来自Antwerp、可能主要在Dutch Republic发展事业，并称其生平仍很不清楚，没有明确给出国籍。旧P02/A答Dutch Republic，完整新版/B答不能确定。不能把A恢复有答案直接视为证据忠实度提升。
3. **Jerusalem，A混入goal要求并自相矛盾。** `musique:3hop1__106042_64399_53006`，第1步只问城市。A先断言Jerusalem，随后因为材料没有确定“建在旧神庙遗址上的大清真寺”而改称城市不可确定；大清真寺是原问题后续目标。检索材料相同，A同时违背仅回答当前子问题、一句事实句和保留一致结论的要求。
4. **Emma Gramatica，A恢复了旧版的过强否定。** `hotpotqa:5a90bb4e55429916514e7536`，第1步的分解问句本身错配人物与城镇。材料说Emma是演员、出生地Borgo San Donnino现称Fidenza。完整新版/B只说材料未识别出叫Emma Gramatica的城镇；旧P02/A扩大成现实中没有城镇被命名为Emma Gramatica。局部材料没有支持这个全称否定。这里也有上游分解问题，不能只归因于回答prompt。
5. **Araripe Júnior，缺少实体答案后改写无法完成绑定。** `2wiki:923b15ba0bb011ebab90acde48001122`。完整新版、A、B均在父亲子问题上不能确定，后续查询保留“Araripe Júnior的父亲”，没有旧P02中的具体姓名。导致后续材料变化的起点在子答案；删only没有恢复此例。首步材料并未直接支持旧P02给出的父亲姓名。

## goal query和检索

1. **新增的关系替换错误确实存在。** `2wiki:6c0c4c1a0bb011ebab90acde48001122`。原问Henry Beresford（三世）的祖父，历史答案已给出George Beresford（一世）。完整新版/B改写成`Who is George Beresford, 1st Marquess of Waterford's paternal grandfather?`，等于又追问了两代。旧P02/A问George是谁。此次合并阅读材料没有变化、答案仍给出George，因此这是query语义错误，未证明造成此次检索损失。
2. **把实体替换成结果列表。** `hotpotqa:5a84d2ea5542992a431d1aa8`。完整新版/B把Ed Wood、Fernand Rivers替换成两串职业，查询变成“这些职业与那些职业共同有什么职业”，丢失人物锚点。A恢复两个人名，阅读材料也变化。原始query一路仍保留，两路合并减轻了影响；最终答案未因此变化。
3. **把排除对象扩成全部成员。** `hotpotqa:5a7a59d15542996c55b2dd67`。原问与Brittany Nicole Carpentero一起出现的其他女rapper，完整新版/B改写成与Diamond、Eve、Trina一起出现的其他rapper。A恢复只以Brittany为锚点的查询。普通子答案的列举形式影响了未修改的rewriter。
4. **明确的合并证据丢失，各版本共有。** `musique:4hop3__373866_5189_38229_86687`。goal改写为`When did the Soviet Union agree to a unified Germany inside of NATO?`，改写检索第9条`German reunification`明确包含`In May 1990`及NATO成员资格语境。但原始query优先交错合并、去重、保留10条后，这个文档不在实际reader包中。该包主要是华约、1939年条约和战后分区等文本。不能把这个问题归为query没有命中，也不能归为goal看到了证据却没有利用。

## goal回答：证据相同仍可能出现错误

1. **原问题可答，却被中间缺答拖住，A新增。** 杂志题`hotpotqa:5a70f9335542994082a3e46a`。所有版本goal材料都直接给出Alt for Damerne每周、Essence每月。A受子问`copies`及回答中`circulation count`影响，以不知道全年发行份数为由拒绝比较；原题及参考答案的比较对象是出版频率。完整新版、旧P02、B都能答Alt for Damerne。这里无需追加检索，也不应等所有中间事实解决。
2. **材料明确给答案，仍输出never/zero，各版本已有。** 南京题`musique:3hop2__326964_265410_7713`的实际goal材料含`had been the capital city of Yangzhou for about 400 years from late Han to early Tang`。旧P02、完整新版、A都答南京从未是扬州首都、因此0年；B去掉字面zero但仍断言从未任首都。改写的现代城市/历史区域范围需要核对；至少不能无视这句直接材料而下全称否定。
3. **B继承错误历史答案，未完成证据纠正。** 语言题`musique:4hop3__275416_24325_156850_10557`，goal材料保留Carnival的`walloon and Latin`及《Middle Ages》对拉丁语到`Medieval Latin`的明确叙述。完整新版/A结论改为Medieval Latin；B返回历史答案Old French。完整新版/A句中仍沿用Walloon绑定，也未把整个关系链修正干净。不能把这次变化仅描述为措辞差别。
4. **other的排除范围未执行。** rapper题材料明确Brittany就是Diamond。完整新版/B把Diamond也列作other；A的goal说Diamond（Brittany）、Eve、Trina都参与，识别了别名却没有直接给出排除自身后的Eve、Trina。A最终答案是后续final阶段才给出Eve、Trina。
5. **证据缺失时补出更精确日期。** 德国统一题的实际goal reader未包含上文May 1990文档。旧P02/完整新版给1990并附两加四条约日期，A/B进一步把同意加入NATO的时间写成12 September 1990。参考答案是May 1990，允许别名1990；不能把新的日期差异仅称作无害的日期粒度变化。这里同时存在合并损失和回答超出材料的问题。

## 本轮结论的范围

两个最小修改会影响输出，不能说残留Resolve句“没有影响”。但当前记录不支持把问题主要归因于`#N`指令冲突：有子问题读表和范围错误，有历史答案引起的query语义漂移，有合并丢证据，也有goal已经看到直接答案却回答错误。它们需要分别处理。

上表来自全部90题的机械对齐；上列错误判断针对已核对原文的具体案例，不是全量人工正确率，也没有新增付费模型评审。

最终Answer-F1/EM仅作为完整流程附录：完整新版74.1003%/63.3333%；A74.5884%/64.4444%；B73.2432%/62.2222%。旧P02历史参考76.6889%/66.6667%。本次两组增量返回用量估算费用¥0.6172244，未知费用预留0；旧实验未知费用仍按原记录保留。

机器可读逐项对照见`STAGE_COMPARISON.json`；答案分数及轨迹变化见`CHANGE_REVIEW.json`；具体证据包见`STAGE_AUDIT_EVIDENCE.json`。
