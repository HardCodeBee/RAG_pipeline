# P02既有完整RAG轨迹诊断报告

本次已完成723条完整流程的分解与节点标注，并做第二遍复核。主评估保持暂停。分析对象是接入统一请求层之前的历史正式P02结果；这不是统一层P02的新成绩，也不是500题正式总表的替代版本。

最终正确固定为已有官方F1=1或EM=1。分解后没有普通子query的路线不纳入。子回答以真实提示词中的当前问题判断；事实正确、合理证据不足弃答、真实但不完整列表均可判正确。完整句子有错误关系或额外事实则判错；真实但未回答所问的背景、计数替代对象、提名替代获奖单列答非所问。

来源固定为原SCORES选中的轨迹；不挑最佳重试，不重跑RAG，不调用外部评审模型。当前Codex是单一审阅者，标签不应称为独立人工金标。最终分数在首轮阶段标注和第二遍复核之后才合并。

## 样本和已有端点

| 数据集 | 完整非空流程 | 普通子回答 | goal预答 | 实际改写记录 | 既有F1均值 | 既有EM均值 | F1或EM满分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki | 485 | 980 | 485 | 980 | 82.25% | 73.61% | 358 |
| hotpotqa | 238 | 381 | 238 | 381 | 70.89% | 56.30% | 135 |

这是完整流程可用样本的事后诊断，不要求凑满500。未完成的37条非空路线仅有11条HotpotQA轨迹保存了13次可分析改写；它们仅进入Q2补充表，不虚构最终答案。另排除58条已完成空分解路线和3条未完成空分解路线。HotpotQA尚未完成的部分可能带来可用样本偏差。

## 四组条件概率

第1、3、4组按原始query计算；“子回答正确”表示该query全部普通子回答正确，“改写正确”表示全部已记录的实际生效改写同时满足语义保真与事实目标有效性。第2组以一次后续改写为单位，比较其已提供的全部前序普通回答是否都正确；不把当前回答与发生在它之前的同一步改写配对。goal与普通节点分列见补充表。

区间按原query整簇重采样2000次，seed=20261021，报告95%百分位区间。多个节点不当作独立query。下表第2行的结果变量为“改写正确”，其余为“最终回答正确”。

| 数据集 | 条件关系 | 条件正确时 | 条件错误时 | 正确组−错误组 |
|---|---|---|---|---|
| 2wiki | ①分解 → 最终回答 | 74.8%（356/476）；95%CI 70.9%–78.7% | 22.2%（2/9）；95%CI 0.0%–50.0% | +52.6个百分点 |
| 2wiki | ②前序普通回答 → 后续改写 | 95.5%（924/968）；95%CI 94.0%–96.7% | 41.7%（5/12）；95%CI 9.1%–83.3% | +53.8个百分点 |
| 2wiki | ③普通子回答 → 最终回答 | 74.8%（356/476）；95%CI 70.9%–78.7% | 22.2%（2/9）；95%CI 0.0%–50.0% | +52.6个百分点 |
| 2wiki | ④实际改写 → 最终回答 | 73.4%（320/436）；95%CI 69.2%–77.6% | 77.6%（38/49）；95%CI 65.9%–88.7% | -4.2个百分点 |
| hotpotqa | ①分解 → 最终回答 | 59.0%（124/210）；95%CI 52.6%–65.8% | 39.3%（11/28）；95%CI 22.7%–58.3% | +19.8个百分点 |
| hotpotqa | ②前序普通回答 → 后续改写 | 93.9%（323/344）；95%CI 91.1%–96.3% | 56.8%（21/37）；95%CI 35.5%–75.7% | +37.1个百分点 |
| hotpotqa | ③普通子回答 → 最终回答 | 59.9%（127/212）；95%CI 53.1%–66.5% | 30.8%（8/26）；95%CI 14.3%–50.0% | +29.1个百分点 |
| hotpotqa | ④实际改写 → 最终回答 | 58.8%（120/204）；95%CI 52.2%–65.7% | 44.1%（15/34）；95%CI 27.3%–61.1% | +14.7个百分点 |

这些是条件关联，不能解释为修复该组件后准确率会提高相同幅度。2Wiki的分解错误仅9题，其中6题只是U维度冗余；余下实质语义错误仅3题，最终正确1/3。HotpotQA分解错误28题，其中19题只有U冗余，实质语义错误9题、最终正确4/9。不能把“有冗余”与“分解事实或结构错误”合成同一种因果机制。

2Wiki“分解错误”和“普通回答有错”组碰巧各9题，成功数也都为2，成员并不相同；逐query标签保留在QUERY_LABELS_AND_OUTCOMES.jsonl。

## 改写错在哪里，以及最终是否受影响

完整流程共88条改写记录有保真或事实目标问题，分属83个原query。2Wiki为51条／49题，HotpotQA为37条／34题。普通子问题改写错误仅3条和5条；goal改写错误分别48条和32条，合计80/88发生在goal。这里比较的是实际生效问题；2条格式拒绝后回退原问题的记录单独保留，不把被拒绝的原提议冒充已传播错误。

2Wiki有改写错误时最终正确38/49=77.6%，全部改写正确时为320/436=73.4%；两者差值的95%区间跨0。HotpotQA对应15/34=44.1%与120/204=58.8%，差值区间同样跨0。现有数据不足以认定所有改写错误都会损害最终答案。

主要原因可由现有架构解释：改写仅影响补充检索；子回答仍收到原当前问题。合并后的子证据取原查询优先交错Top10；最终可见输入是所有普通QA、goal预答，以及原query检索Top10。已逐条验证723条保存的最终上下文确实进入保存的最终提示词。改写证据不会作为完整文段直接加入最终输入。因此前序QA、原检索或最终模型可能抑制改写错误；改写也可能检索命中却在合并中被丢掉。

| 数据集 | 错误位置 | 改写记录数 | 涉及query数 | 其中最终正确query |
|---|---|---:|---:|---:|
| 2wiki | 丢失特定实体身份 | 4 | 4 | 4 |
| 2wiki | 亲属关系或代际错误 | 3 | 3 | 1 |
| 2wiki | 候选范围或限定条件改变 | 1 | 1 | 1 |
| 2wiki | 关系丢失：改成已知实体是谁 | 36 | 36 | 29 |
| 2wiki | 实体与属性的类型混淆 | 2 | 2 | 1 |
| 2wiki | 带入错误的附加事实 | 1 | 1 | 0 |
| 2wiki | 比较命题被换成分别核实 | 3 | 3 | 3 |
| 2wiki | 绑定错误的前序对象 | 1 | 1 | 0 |
| hotpotqa | 丢失特定实体身份 | 1 | 1 | 0 |
| hotpotqa | 候选范围或限定条件改变 | 2 | 2 | 1 |
| hotpotqa | 关系或角色被改变 | 7 | 7 | 3 |
| hotpotqa | 实体与属性的类型混淆 | 1 | 1 | 1 |
| hotpotqa | 带入错误的附加事实 | 2 | 2 | 2 |
| hotpotqa | 未知答案被填入问题 | 16 | 16 | 8 |
| hotpotqa | 绑定错误的前序对象 | 8 | 6 | 0 |

同一query可能包含不止一种错误位置，本表各行不能相加当作独立query总数。36条2Wiki关系丢失是把亲属问题改成Who is已知人物；按保真口径判错，但通用人物检索仍可能提供有用资料。完整逐条改写、错误位置、最终输出、参考及F1/EM见 [REWRITE_ERROR_CATALOG.md](REWRITE_ERROR_CATALOG.md)。

## 子回答与goal分别报告

| 数据集 | 类型 | 正确 | 错误 | 待核事实，无真值标签 | 正确中包含合理弃答 |
|---|---|---:|---:|---:|---:|
| 2wiki | ordinary | 970 | 10 | 0 | 15 |
| 2wiki | goal | 457 | 23 | 5 | 23 |
| hotpotqa | ordinary | 352 | 29 | 0 | 32 |
| hotpotqa | goal | 214 | 22 | 2 | 18 |

普通回答的高事实正确率不能代表已取得完整桥接信息。比如Hasbro敌人列表真实但缺Wendigo；Jewel Staite案例合理弃答，却仍无法取得年份。需要把正确弃答、真实不完整列表与得到所需事实区分。下面的敏感性表仅排除含合理普通弃答的query；它仍不把真实不完整列表重新判错。

| 数据集 | 条件关系 | 条件正确时 | 条件错误时 | 正确组−错误组 |
|---|---|---|---|---|
| 2wiki | 普通回答 → 最终，排除合理普通弃答 | 75.6%（350/463）；95%CI 71.6%–79.4% | 22.2%（2/9）；95%CI 0.0%–54.5% | +53.4个百分点 |
| 2wiki | goal预答 → 最终 | 77.7%（355/457）；95%CI 73.8%–81.5% | 8.7%（2/23）；95%CI 0.0%–22.2% | +69.0个百分点 |
| hotpotqa | 普通回答 → 最终，排除合理普通弃答 | 63.4%（116/183）；95%CI 56.7%–70.1% | 32.0%（8/25）；95%CI 14.8%–52.0% | +31.4个百分点 |
| hotpotqa | goal预答 → 最终 | 59.3%（127/214）；95%CI 52.8%–65.9% | 27.3%（6/22）；95%CI 9.5%–47.4% | +32.1个百分点 |

2Wiki goal错误23题中21题最终不正确，2题仍最终正确；HotpotQA为22题中16题最终不正确，6题仍正确。此关联提示应优先查看goal是否与正确中间事实矛盾；它不证明删掉goal或改提示词必然提高成绩。

## 已核对的具体失败与恢复轨迹

下表是有明确材料支撑的选例，不是对全部230个既有非满分query进行互斥、穷尽的因果分摊。每条均保留原始文件路径与哈希于ROOT_CAUSE_CASES.json。

| 数据集/编号 | 观察 | 最终输出 / 既有参考 | 已有F1 |
|---|---|---|---:|
| hotpotqa/7 | 第2步改写检索的Doomsday Prophecy排第10；原查询优先交错后保留的10条读入证据没有它。子回答和goal均合理弃答，最终也弃答。此例已经找到证据进入候选集后被裁掉的具体位置。 | Cannot be determined / 2011 | 0.000 |
| hotpotqa/46 | 子回答列出的Hasbro作品与敌人均真实，但敌人列表只保留Decepticons／Inhumanoids，没有目标Wendigo。按用户口径这些子回答正确；信息覆盖不足仍会使下游无法完成桥接。 | Cannot be determined / Wendigo | 0.000 |
| hotpotqa/27 | Speedpass的developer Verifone被答成introducer Mobil；改写绑定Mobil，goal和最终输出都落到Mobil的总部。事实角色的混淆比实体名称相似更早发生。 | Fairfax County, Virginia / San Jose, California | 0.000 |
| hotpotqa/107 | 前序把应查的Youngstown队带成Newark Little Giants，随后又把New Jersey的队映射到Ohio的Newark/Licking。最终选Licking County。Newark确为Licking县治不等于它回答了当前关系问题。 | Licking County / Mahoning County | 0.500 |
| hotpotqa/133 | 把John Arledge与John Garfield混用，选错Flowing Gold，又给错其导演Ralph Murphy；最终沿用错误导演。AFI记录用于核对冲突。 | Ralph Murphy / John Ford | 0.000 |
| hotpotqa/21 | 四个普通子回答的生卒日期正确，goal却把67岁说成比78岁长寿；最终选择Lucy Maud Montgomery。最终输入同时包含正确日期和错误goal结论。 | Lucy Maud Montgomery / Owen Wister | 0.000 |
| 2wiki/362 | 普通答案正确给出Turkish-French与French，goal和最终却答No。共同国籍应看交集，不能要求两组国籍完全一样。 | No / yes | 0.000 |
| 2wiki/376 | French与French-American被goal判断为无共同国籍，最终答No。与362属同一可重复检查的比较错误。 | No / yes | 0.000 |
| 2wiki/566 | 普通出生日期1965与1966正确，goal却把1965年出生者称作更年轻，最终选择错误电影。 | Hell To Pay (2014 Film) / Darwin'S Nightmare | 0.000 |
| 2wiki/403 | 普通子回答已给祖母Mary Clifton，goal改成母亲Catherine Walmesley，最终也选母亲。只可说轨迹与错误结论继续传播一致，不能据此证明删除goal必定修好。 | Catherine Walmesley / Mary Clifton | 0.000 |
| hotpotqa/236 | goal明确同时给4000总容量、3677坐席，最终问题问seat，输出却选4000。正确坐席数在实际最终输入中可见，是明确的末端属性选择错误。 | 4,000 / 3,677 seated | 0.000 |
| 2wiki/296 | goal说明一地在Bolivia、另一地跨Bolivia/Chile且共享Bolivia；最终仍取No。存在交集的信息已在最终输入中，不应仅追责检索缺证据。 | No / yes | 0.000 |
| 2wiki/533 | goal只报告真实Grammy提名，并额外要求奖必须对应所提歌曲，最终弃答。该回答未提供所问的已获奖项，按最新用户规则判答非所问。 | Cannot be determined / Rock and Roll Hall of Fame | 0.000 |
| 2wiki/593 | goal从五次提名扩成没有Grammy Awards，最终也输出No Grammy Awards；官方Grammy记录有2001获奖。此处不是合理证据不足弃答。 | No Grammy Awards / Grammy | 0.500 |
| 2wiki/350 | goal错把父亲Peter Grubb当祖父；最终仍选普通子回答中正确的John Grubb。此例说明goal出错并不必然决定最终输出。 | John Grubb / John Grubb | 1.000 |
| 2wiki/534 | 普通回答把孙辈Michael列作子辈，后续仍找到Francis之妻Mary Lawson，最终正确；改写同时存在错误代际候选和generic Who is问题。 | Mary Lawson / Mary Lawson | 1.000 |
| hotpotqa/56 | 普通第一回答只给38所学院与6个厅的数量，属答非所问；第二步和goal恢复Magdalen College，最终正确。 | Magdalen College / Magdalen College | 1.000 |
| 2wiki/687 | 分解把两个片名当导演名，普通回答纠正为真实导演并给出生日期，最终正确。这是现有轨迹中的修复实例，不代表所有错误分解都可修复。 | Sujon Sokhi / Sujon Sokhi | 1.000 |
| hotpotqa/171 | goal错把Pamelyn Wanda Ferdin整个姓名称作middle name，最终抽取Wanda正确。需保留完整句子正确性与短答案既有评分两个口径。 | Wanda / Wanda | 1.000 |
| hotpotqa/97 | 源材料明确Stoltenberg为former managing editor；完整句子重复现在时错误，但最终所问aging issues正确。没有用2026资料追溯改判。 | aging issues / aging issues | 1.000 |
| hotpotqa/106 | goal重复SBS为state-run错误预设，最终短答案SBS仍与参考一致。官方SBS历史资料用于核对所有权。 | SBS / SBS | 1.000 |
| hotpotqa/15 | Rainn Wilson符合Smurfs与The Office两个条件，已有单一参考为Ellie Kemper，最终F1/EM均0。该既有失败不等于轨迹事实错误。 | Rainn Wilson / Ellie Kemper | 0.000 |
| hotpotqa/123 | Cigarette Rock符合当前Lewis and Clark County summit问题，单一参考为Scapegoat Mountain。保留已有0分，并单记替代答案。 | Cigarette Rock / Scapegoat Mountain | 0.000 |
| hotpotqa/193 | The Climb实际采用2002电影，参考采用2007纪录片；相对2004 Voices of Iraq的先后关系随版本变化。前序事实正确仍可能得到既有0分。 | The Climb / Voices of Iraq | 0.000 |
| hotpotqa/150 | 最终Forbes已修正Cowra，但参考为带州与区域描述的长串，已有F1仅0.1818。不能把这部分损失都归因于改写错误。 | Forbes / Forbes in the Central West region of New South Wales, Australia | 0.182 |
| hotpotqa/152 | 最终Mike Fiers与参考Michael Bruce Fiers指向同一球员，已有F1为0.4；普通第三回答另有答非所问错误，不能混成一个原因。 | Mike Fiers / Michael Bruce Fiers | 0.400 |
| 2wiki/246 | 最终Konstfack department of graphic design对比参考Konstfack，已有F1=0.5；此例展示严格token匹配对附加粒度的惩罚。 | Konstfack department of graphic design / Konstfack | 0.500 |
| 2wiki/258 | 最终包含Besancon及历史区划，参考只有Besancon，已有F1=0.25。保留评分，另记输出范围差异。 | Besançon, Prince-Archbishopric of Besançon, Holy Roman Empire / Besançon | 0.250 |
| hotpotqa/189 | goal读入第3条FindingKraftland纪录片却仍说无法判断；最终No可由真实材料支持，但既有参考是Shikashika is a shaved ice drink这一描述句。两种问题同时存在。 | No / Shikashika is a shaved ice drink | 0.000 |
| hotpotqa/201 | 题目问country，最终India，既有参考却为Jamnagar；另有前序把两个职位拆到两个人的问题。不能用参考不匹配掩盖前序缺陷，也不能把全部0分归因于前序。 | India / Jamnagar | 0.000 |
| hotpotqa/225 | 原题问两条运河的位置，最终分别给Louisiana和Pennsylvania，参考只给Pennsylvania；已有F1=1/3。实际改写另有把答案预填进问题的错误，需分别记录。 | New Orleans, Louisiana and Pennsylvania / Pennsylvania | 0.333 |
| hotpotqa/237 | 检索中的Osteomeles anthyllidifolia明确生长在多个Pacific islands，Medinilla也到westernPacific，最终Both有材料支持；单一参考只收Medinilla。 | Both / Medinilla | 0.000 |
| hotpotqa/238 | 最终Oklahoma/Texas/northernMexico由实际AfroSeminoleCreole片段支持；参考收Coahuila,Mexico。该0.2857分需与完整事实正确性分开。 | Oklahoma, Texas, and northern Mexico / Coahuila, Mexico | 0.286 |
| 2wiki/393 | DestinationDewsbury导演Jack Spring与1933年出生的棒球手Jack Russell Spring混淆。最终Parkinsons disease虽拿满分，完整事实错误；导演2019亲述资料排除2015已去世。 | Parkinson's disease / Parkinson's disease | 1.000 |
| 2wiki/674 | MadeInUSA电影导演与Fluxus设计学者Ken Friedman混淆，goal和最终给后者Swinburne/Tongji任职。NYUTisch官方导演履历用于核对。 | Swinburne University of Technology and Tongji University / Swinburne | 0.727 |
| 2wiki/668 | 本地人物资料区分Hon David与其父5thViscount，原gold把两代合并。子回答另有完整句子亲属修饰错误；不得以gold实体重叠当作事实正确的充分证据。 | Lady Ellen MacCarthy Reagh / Ellen MacCarthy | 0.667 |

## 结果偏低的含义和下一步审阅重点

已确认的实际流程问题包括：检索命中后的证据合并裁剪；真实部分列表没有覆盖所需桥接对象；子回答错认事实角色或同名实体；goal在日期比较、国籍集合、亲属代际上覆盖正确中间事实；最终抽取错属性。建议先围绕这些已核对问题设计后续对照，所有代码／提示词修改和重跑仍须另获用户批准。本报告没有启动任何修复评估。

同时存在明显评分或参考问题：有效替代答案、别名、答案附加地理／机构描述、题目和参考所问对象不同，以及gold同名串接。它们会影响已有F1/EM，也会使“最终正确”这个条件概率端点与事实正确性不完全一致。所有主表仍保留既有官方口径；本次没有把有争议答案重判成满分。

| 数据集 | 既有非满分query | 其中goal事实正确 | 其中0<F1<1 | 其中F1=0 | D、普通A、R全正确但仍非满分 |
|---|---:|---:|---:|---:|---:|
| 2wiki | 127 | 102 | 70 | 57 | 108 |
| hotpotqa | 103 | 87 | 56 | 47 | 69 |

“goal事实正确而端点非满分”也包含正确弃答、答案粒度不同和真实替代答案，不能直接当作最终模型错误数量；同理，F1部分得分也不自动意味着仅是格式问题。

## 待核项和查证范围

仍有7条goal完整陈述的附加事实未解决，未强行赋正确／错误标签。它们只从goal相关分母排除：2Wiki5条、HotpotQA2条。全部普通子回答、分解、改写与原Y保留，不删除整个query。

| 数据集/编号/步骤 | 待核事实 |
|---|---|
| 2wiki/309/2 | VSC founder JonathanGregg bornNYC |
| 2wiki/319/2 | FerencEsterhazy1533 bornGalanta |
| 2wiki/519/2 | ClaraGonzaga diedinMontpensier town |
| 2wiki/520/2 | IrvingKristol fatherJoseph wasarabbi |
| hotpotqa/160/2 | Boeing707-436 maximum219passengers |
| hotpotqa/234/2 | Jesuit traditions three core subjects weregrammar rhetoric logic |
| 2wiki/386/2 | Rivette diagnosis exactlythree years beforedeath |

检索材料或冻结数据能判断时以本地材料为准；仅缺证或冲突项做定向网页核对。没有以2026年的职位／球队变化追溯否定旧材料。WEB_FACT_CHECKS.json记录20项已作判断的查证，PENDING_FACTS.json保留未定事实与冲突。关键来源包括 [AFI的Flowing Gold](https://catalog.afi.com/Film/5028-FLOWING-GOLD)、[AFI的Street Kings](https://catalog.afi.com/Catalog/moviedetails/64772)、[Grammy的Joe Jackson记录](https://www.grammy.com/artists/joe-jackson/10992/)、[NYU Tisch的Kenneth A. Friedman履历](https://tisch.nyu.edu/about/directory/grad-film/101907044.html)、[Jack Spring在2019年的导演自述](https://www.comedy.co.uk/film/destination_dewsbury/special/how_to_make_a_movie_aged_19/)及[SBS官方历史](https://about.sbs.co.kr/history)。

## 补充表和复核记录

| 数据集 | 条件关系 | 条件正确时 | 条件错误时 | 正确组−错误组 |
|---|---|---|---|---|
| 2wiki | 前序普通A → 普通改写 | 99.8%（491/492）；95%CI 99.4%–100.0% | 33.3%（1/3）；95%CI 0.0%–100.0% | +66.5个百分点 |
| 2wiki | 前序普通A → goal改写 | 91.0%（433/476）；95%CI 88.2%–93.5% | 44.4%（4/9）；95%CI 12.5%–80.0% | +46.5个百分点 |
| 2wiki | 实际显式绑定的前序A → 改写 | 92.2%（522/566）；95%CI 90.1%–94.3% | 0.0%（0/6）；95%CI 0.0%–0.0% | +92.2个百分点 |
| 2wiki | 普通子节点 → 最终 | 83.6%（811/970）；95%CI 80.2%–86.7% | 20.0%（2/10）；95%CI 0.0%–50.0% | +63.6个百分点 |
| 2wiki | 普通改写记录 → 最终 | 92.3%（454/492）；95%CI 88.9%–95.3% | 33.3%（1/3）；95%CI 0.0%–100.0% | +58.9个百分点 |
| 2wiki | goal改写记录 → 最终 | 73.2%（320/437）；95%CI 69.0%–77.4% | 79.2%（38/48）；95%CI 67.4%–90.4% | -5.9个百分点 |
| 2wiki | 确有变化的实际改写 → 最终 | 79.6%（510/641）；95%CI 75.6%–83.2% | 76.5%（39/51）；95%CI 63.6%–88.5% | +3.1个百分点 |
| 2wiki | 前序普通A → 改写，含未完成前缀 | 95.5%（924/968）；95%CI 94.0%–96.7% | 41.7%（5/12）；95%CI 9.1%–83.3% | +53.8个百分点 |
| hotpotqa | 前序普通A → 普通改写 | 97.7%（129/132）；95%CI 94.8%–100.0% | 81.8%（9/11）；95%CI 44.4%–100.0% | +15.9个百分点 |
| hotpotqa | 前序普通A → goal改写 | 91.5%（194/212）；95%CI 87.6%–95.1% | 46.2%（12/26）；95%CI 26.9%–66.7% | +45.4个百分点 |
| hotpotqa | 实际显式绑定的前序A → 改写 | 89.4%（160/179）；95%CI 84.4%–93.9% | 17.6%（3/17）；95%CI 0.0%–40.9% | +71.7个百分点 |
| hotpotqa | 普通子节点 → 最终 | 58.5%（206/352）；95%CI 51.9%–65.4% | 31.0%（9/29）；95%CI 13.2%–52.4% | +27.5个百分点 |
| hotpotqa | 普通改写记录 → 最终 | 57.2%（79/138）；95%CI 47.9%–66.2% | 20.0%（1/5）；95%CI 0.0%–100.0% | +37.2个百分点 |
| hotpotqa | goal改写记录 → 最终 | 58.7%（121/206）；95%CI 52.2%–65.7% | 43.8%（14/32）；95%CI 26.9%–61.4% | +15.0个百分点 |
| hotpotqa | 确有变化的实际改写 → 最终 | 54.3%（102/188）；95%CI 45.5%–62.6% | 40.5%（15/37）；95%CI 23.7%–58.5% | +13.7个百分点 |
| hotpotqa | 前序普通A → 改写，含未完成前缀 | 93.6%（334/357）；95%CI 90.9%–95.8% | 56.8%（21/37）；95%CI 35.0%–76.5% | +36.8个百分点 |

冻结前的第二遍复核覆盖150个初判有误节点、固定随机抽取的193个其余节点；分解另复核40个初判不通过项及69个抽样通过项。保留首轮标签和修订，不覆盖旧记录。没有不同审阅者之间的一致性检验。小样本错误组的bootstrap区间可能很宽，单例或全成功组可退化；JSON保留非空重采样次数，不能把窄／退化区间视为足够证据。

可复查文件：

- [CONDITIONAL_PROBABILITIES.json](CONDITIONAL_PROBABILITIES.json)：全部条件概率、分母与原query簇bootstrap区间。
- [QUERY_LABELS_AND_OUTCOMES.jsonl](QUERY_LABELS_AND_OUTCOMES.jsonl)：逐query标签与既有端点。
- [STAGE_LABELS_LOCKED.json](STAGE_LABELS_LOCKED.json)：逐节点依据、实际改写与原文件哈希。
- [ROOT_CAUSE_CASES.json](ROOT_CAUSE_CASES.json)：选例观察，不作全量因果归因。
- [FINAL_VISIBILITY_VERIFIED.json](FINAL_VISIBILITY_VERIFIED.json)：723条实际最终提示词上下文校验。
- [PENDING_FACTS.json](PENDING_FACTS.json)：7个事实工作项，无真值标签。
- [CURRENT_VERIFICATION.json](CURRENT_VERIFICATION.json)：冻结源、主评暂停标记及费用来源核对。

本次诊断没有新增RAG推理、GPU任务或外部评审模型请求。共享账本另有用户在其他任务中明确批准的quick90提示词实验／最小对照费用，单独列在核对文件中；不得把共享账本的变化归给本诊断，也不得宣称全局账本未变。
