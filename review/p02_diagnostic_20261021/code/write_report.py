"""Render the fixed diagnostic tables and explicit, non-causal case findings."""
from collections import Counter
from pathlib import Path
import json
from prepare_inventory import ROOT,OUT,read,rows,write,sha

run=OUT/'offline_review_v2'
stats=read(run/'CONDITIONAL_PROBABILITIES.json')
qs=rows(run/'QUERY_LABELS_AND_OUTCOMES.jsonl');byi={r['review_index']:r for r in qs}
ys={r['id']:r for r in rows(run/'OUTCOMES_SEPARATE.jsonl')}
st=read(run/'STAGE_LABELS_LOCKED.json')['labels']
cats=rows(run/'REWRITE_ERROR_CATALOG.jsonl')
tt={(t['dataset'],t['table']):t for t in stats['tables']}
def pct(x):return '—' if x is None else f'{x*100:.1f}%'
def cell(g,ci=True):
 s=f"{pct(g['probability'])}（{g['successes']}/{g['n']}）"
 if ci and g['cluster_bootstrap_95_ci']:s+=f"；95%CI {pct(g['cluster_bootstrap_95_ci'][0])}–{pct(g['cluster_bootstrap_95_ci'][1])}"
 return s
def conditions(names):
 out=['| 数据集 | 条件关系 | 条件正确时 | 条件错误时 | 正确组−错误组 |','|---|---|---|---|---|']
 for ds in ['2wiki','hotpotqa']:
  for key,title in names:
   t=tt[ds,key];d=t['correct_minus_incorrect']
   out.append(f"| {ds} | {title} | {cell(t['groups']['correct'])} | {cell(t['groups']['incorrect'])} | {d['probability_difference']*100:+.1f}个百分点 |")
 return out
summary={}
for ds in ['2wiki','hotpotqa']:
 r=[q for q in qs if q['dataset']==ds];fails=[q for q in r if not q['final_correct']]
 summary[ds]={
 'final_failures':len(fails),
 'failures_with_all_D_ordinaryA_R_correct':sum(q['decomposition_correct'] and q['ordinary_answers_all_correct'] and q['rewrites_all_correct'] for q in fails),
 'failures_with_goal_answer_correct':sum(q['goal_answer_correct'] is True for q in fails),
 'failures_with_goal_pending':sum(q['goal_answer_correct'] is None for q in fails),
 'failures_with_ordinary_reasonable_abstention':sum(q['reasonable_ordinary_abstention'] for q in fails),
 'failures_with_partial_F1':sum(0<q['f1']<1 for q in fails),
 'failures_with_zero_F1':sum(q['f1']==0 for q in fails),
 'final_CannotBeDetermined':sum(ys[q['id']]['prediction'].strip().lower()=='cannot be determined' for q in r),
 'goal_wrong_final_wrong':sum(q['goal_answer_correct'] is False and not q['final_correct'] for q in r),
 'goal_wrong_final_correct':sum(q['goal_answer_correct'] is False and q['final_correct'] for q in r),
 'F1_1_EM_0_correct_queries':[q['review_index'] for q in r if q['f1']==1 and q['em']!=1],
 }
write(run/'DIAGNOSTIC_DESCRIPTIVE_COUNTS.json',summary)

cases=[
(7,['改写检索已命中但合并丢失','最终缺少有效证据'],'第2步改写检索的Doomsday Prophecy排第10；原查询优先交错后保留的10条读入证据没有它。子回答和goal均合理弃答，最终也弃答。此例已经找到证据进入候选集后被裁掉的具体位置。'),
(46,['部分正确列表缺关键对象'],'子回答列出的Hasbro作品与敌人均真实，但敌人列表只保留Decepticons／Inhumanoids，没有目标Wendigo。按用户口径这些子回答正确；信息覆盖不足仍会使下游无法完成桥接。'),
(27,['创建者与引入者混淆','错误对象传播'],'Speedpass的developer Verifone被答成introducer Mobil；改写绑定Mobil，goal和最终输出都落到Mobil的总部。事实角色的混淆比实体名称相似更早发生。'),
(107,['球队身份错误','同名城市错误绑定'],'前序把应查的Youngstown队带成Newark Little Giants，随后又把New Jersey的队映射到Ohio的Newark/Licking。最终选Licking County。Newark确为Licking县治不等于它回答了当前关系问题。'),
(133,['演员同名混淆','导演事实错误'],'把John Arledge与John Garfield混用，选错Flowing Gold，又给错其导演Ralph Murphy；最终沿用错误导演。AFI记录用于核对冲突。'),
(21,['goal比较错误','最终沿用错误结论'],'四个普通子回答的生卒日期正确，goal却把67岁说成比78岁长寿；最终选择Lucy Maud Montgomery。最终输入同时包含正确日期和错误goal结论。'),
(362,['共同国籍集合判断错误'],'普通答案正确给出Turkish-French与French，goal和最终却答No。共同国籍应看交集，不能要求两组国籍完全一样。'),
(376,['共同国籍集合判断错误'],'French与French-American被goal判断为无共同国籍，最终答No。与362属同一可重复检查的比较错误。'),
(566,['goal年龄方向错误'],'普通出生日期1965与1966正确，goal却把1965年出生者称作更年轻，最终选择错误电影。'),
(403,['goal覆盖正确中间关系'],'普通子回答已给祖母Mary Clifton，goal改成母亲Catherine Walmesley，最终也选母亲。只可说轨迹与错误结论继续传播一致，不能据此证明删除goal必定修好。'),
(236,['最终抽取属性错误'],'goal明确同时给4000总容量、3677坐席，最终问题问seat，输出却选4000。正确坐席数在实际最终输入中可见，是明确的末端属性选择错误。'),
(296,['最终选择错误的比较极性'],'goal说明一地在Bolivia、另一地跨Bolivia/Chile且共享Bolivia；最终仍取No。存在交集的信息已在最终输入中，不应仅追责检索缺证据。'),
(533,['提名代替获奖','过度增加按歌曲获奖的限制'],'goal只报告真实Grammy提名，并额外要求奖必须对应所提歌曲，最终弃答。该回答未提供所问的已获奖项，按最新用户规则判答非所问。'),
(593,['把未见获奖误推为从未获奖'],'goal从五次提名扩成没有Grammy Awards，最终也输出No Grammy Awards；官方Grammy记录有2001获奖。此处不是合理证据不足弃答。'),
(350,['错误被后续抑制'],'goal错把父亲Peter Grubb当祖父；最终仍选普通子回答中正确的John Grubb。此例说明goal出错并不必然决定最终输出。'),
(534,['部分错误事实被修正'],'普通回答把孙辈Michael列作子辈，后续仍找到Francis之妻Mary Lawson，最终正确；改写同时存在错误代际候选和generic Who is问题。'),
(56,['上游答非所问但最终恢复'],'普通第一回答只给38所学院与6个厅的数量，属答非所问；第二步和goal恢复Magdalen College，最终正确。'),
(687,['分解语义错误被后续修复'],'分解把两个片名当导演名，普通回答纠正为真实导演并给出生日期，最终正确。这是现有轨迹中的修复实例，不代表所有错误分解都可修复。'),
(171,['完整goal句子错误但短答案正确'],'goal错把Pamelyn Wanda Ferdin整个姓名称作middle name，最终抽取Wanda正确。需保留完整句子正确性与短答案既有评分两个口径。'),
(97,['原问题带过时现在时','最终属性答案正确'],'源材料明确Stoltenberg为former managing editor；完整句子重复现在时错误，但最终所问aging issues正确。没有用2026资料追溯改判。'),
(106,['原问题带错误预设','最终属性答案正确'],'goal重复SBS为state-run错误预设，最终短答案SBS仍与参考一致。官方SBS历史资料用于核对所有权。'),
(15,['有效替代答案未被单一参考覆盖'],'Rainn Wilson符合Smurfs与The Office两个条件，已有单一参考为Ellie Kemper，最终F1/EM均0。该既有失败不等于轨迹事实错误。'),
(123,['有效替代答案未被单一参考覆盖'],'Cigarette Rock符合当前Lewis and Clark County summit问题，单一参考为Scapegoat Mountain。保留已有0分，并单记替代答案。'),
(193,['题目未限定作品版本'],'The Climb实际采用2002电影，参考采用2007纪录片；相对2004 Voices of Iraq的先后关系随版本变化。前序事实正确仍可能得到既有0分。'),
(150,['最终已纠正实体','参考答案含额外描述'],'最终Forbes已修正Cowra，但参考为带州与区域描述的长串，已有F1仅0.1818。不能把这部分损失都归因于改写错误。'),
(152,['姓名别名匹配敏感'],'最终Mike Fiers与参考Michael Bruce Fiers指向同一球员，已有F1为0.4；普通第三回答另有答非所问错误，不能混成一个原因。'),
(246,['答案含真实的更具体机构描述'],'最终Konstfack department of graphic design对比参考Konstfack，已有F1=0.5；此例展示严格token匹配对附加粒度的惩罚。'),
(258,['地点附加历史区划'],'最终包含Besancon及历史区划，参考只有Besancon，已有F1=0.25。保留评分，另记输出范围差异。'),
(189,['goal忽略现有证据','原题与参考回答类型不匹配'],'goal读入第3条FindingKraftland纪录片却仍说无法判断；最终No可由真实材料支持，但既有参考是Shikashika is a shaved ice drink这一描述句。两种问题同时存在。'),
(201,['原题与参考目标不匹配'],'题目问country，最终India，既有参考却为Jamnagar；另有前序把两个职位拆到两个人的问题。不能用参考不匹配掩盖前序缺陷，也不能把全部0分归因于前序。'),
(225,['参考只覆盖原题两个对象之一'],'原题问两条运河的位置，最终分别给Louisiana和Pennsylvania，参考只给Pennsylvania；已有F1=1/3。实际改写另有把答案预填进问题的错误，需分别记录。'),
(237,['参考仅覆盖一个有效选项'],'检索中的Osteomeles anthyllidifolia明确生长在多个Pacific islands，Medinilla也到westernPacific，最终Both有材料支持；单一参考只收Medinilla。'),
(238,['答案地域范围与参考不同'],'最终Oklahoma/Texas/northernMexico由实际AfroSeminoleCreole片段支持；参考收Coahuila,Mexico。该0.2857分需与完整事实正确性分开。'),
(393,['参考同名实体串接错误','既有评分正确但事实错误'],'DestinationDewsbury导演Jack Spring与1933年出生的棒球手Jack Russell Spring混淆。最终Parkinsons disease虽拿满分，完整事实错误；导演2019亲述资料排除2015已去世。'),
(674,['参考同名职业串接错误'],'MadeInUSA电影导演与Fluxus设计学者Ken Friedman混淆，goal和最终给后者Swinburne/Tongji任职。NYUTisch官方导演履历用于核对。'),
(668,['原参考代际连接错误'],'本地人物资料区分Hon David与其父5thViscount，原gold把两代合并。子回答另有完整句子亲属修饰错误；不得以gold实体重叠当作事实正确的充分证据。'),
]
case_records=[]
for i,tags,finding in cases:
 q=byi[i];y=ys[q['id']]
 case_records.append({'review_index':i,'id':q['id'],'dataset':q['dataset'],'question':q['question'],'tags':tags,'finding':finding,'prediction':y['prediction'],'reference':y['reference'],'native_F1':y['f1'],'native_EM':y['em'],'existing_Y':q['final_correct'],'source':q['source'],'source_sha256':q['source_sha256'],'final_visibility_verified':True})
write(run/'ROOT_CAUSE_CASES.json',{'scope':'selected explicit evidence-based case findings, nonexclusive; not exhaustive attribution of all final failures','case_count':len(case_records),'cases':case_records})
lines=['# P02既有完整RAG轨迹诊断报告','',
'本次已完成723条完整流程的分解与节点标注，并做第二遍复核。主评估保持暂停。分析对象是接入统一请求层之前的历史正式P02结果；这不是统一层P02的新成绩，也不是500题正式总表的替代版本。',
'', '最终正确固定为已有官方F1=1或EM=1。分解后没有普通子query的路线不纳入。子回答以真实提示词中的当前问题判断；事实正确、合理证据不足弃答、真实但不完整列表均可判正确。完整句子有错误关系或额外事实则判错；真实但未回答所问的背景、计数替代对象、提名替代获奖单列答非所问。',
'', '来源固定为原SCORES选中的轨迹；不挑最佳重试，不重跑RAG，不调用外部评审模型。当前Codex是单一审阅者，标签不应称为独立人工金标。最终分数在首轮阶段标注和第二遍复核之后才合并。',
'', '## 样本和已有端点','', '| 数据集 | 完整非空流程 | 普通子回答 | goal预答 | 实际改写记录 | 既有F1均值 | 既有EM均值 | F1或EM满分 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for ds,s in stats['summary'].items():lines.append(f"| {ds} | {s['complete_eligible_queries']} | {s['ordinary_nodes']} | {s['goal_nodes']} | {s['rewrite_events']} | {s['mean_native_F1']*100:.2f}% | {s['mean_native_EM']*100:.2f}% | {s['final_correct']} |")
lines+=['', '这是完整流程可用样本的事后诊断，不要求凑满500。未完成的37条非空路线仅有11条HotpotQA轨迹保存了13次可分析改写；它们仅进入Q2补充表，不虚构最终答案。另排除58条已完成空分解路线和3条未完成空分解路线。HotpotQA尚未完成的部分可能带来可用样本偏差。',
'', '## 四组条件概率','',
'第1、3、4组按原始query计算；“子回答正确”表示该query全部普通子回答正确，“改写正确”表示全部已记录的实际生效改写同时满足语义保真与事实目标有效性。第2组以一次后续改写为单位，比较其已提供的全部前序普通回答是否都正确；不把当前回答与发生在它之前的同一步改写配对。goal与普通节点分列见补充表。',
'', '区间按原query整簇重采样2000次，seed=20261021，报告95%百分位区间。多个节点不当作独立query。下表第2行的结果变量为“改写正确”，其余为“最终回答正确”。', '']
lines+=conditions([('D_to_Y','①分解 → 最终回答'),('A_history_to_R','②前序普通回答 → 后续改写'),('A_to_Y_query','③普通子回答 → 最终回答'),('R_to_Y_query','④实际改写 → 最终回答')])
lines+=['', '这些是条件关联，不能解释为修复该组件后准确率会提高相同幅度。2Wiki的分解错误仅9题，其中6题只是U维度冗余；余下实质语义错误仅3题，最终正确1/3。HotpotQA分解错误28题，其中19题只有U冗余，实质语义错误9题、最终正确4/9。不能把“有冗余”与“分解事实或结构错误”合成同一种因果机制。',
'', '2Wiki“分解错误”和“普通回答有错”组碰巧各9题，成功数也都为2，成员并不相同；逐query标签保留在QUERY_LABELS_AND_OUTCOMES.jsonl。',
'', '## 改写错在哪里，以及最终是否受影响','',
'完整流程共88条改写记录有保真或事实目标问题，分属83个原query。2Wiki为51条／49题，HotpotQA为37条／34题。普通子问题改写错误仅3条和5条；goal改写错误分别48条和32条，合计80/88发生在goal。这里比较的是实际生效问题；2条格式拒绝后回退原问题的记录单独保留，不把被拒绝的原提议冒充已传播错误。',
'', '2Wiki有改写错误时最终正确38/49=77.6%，全部改写正确时为320/436=73.4%；两者差值的95%区间跨0。HotpotQA对应15/34=44.1%与120/204=58.8%，差值区间同样跨0。现有数据不足以认定所有改写错误都会损害最终答案。',
'', '主要原因可由现有架构解释：改写仅影响补充检索；子回答仍收到原当前问题。合并后的子证据取原查询优先交错Top10；最终可见输入是所有普通QA、goal预答，以及原query检索Top10。已逐条验证723条保存的最终上下文确实进入保存的最终提示词。改写证据不会作为完整文段直接加入最终输入。因此前序QA、原检索或最终模型可能抑制改写错误；改写也可能检索命中却在合并中被丢掉。',
'', '| 数据集 | 错误位置 | 改写记录数 | 涉及query数 | 其中最终正确query |','|---|---|---:|---:|---:|']
for ds in ['2wiki','hotpotqa']:
 for cat in sorted({v['error_location'] for v in cats if v['dataset']==ds}):
  cr=[v for v in cats if v['dataset']==ds and v['error_location']==cat];ids={v['id'] for v in cr}
  lines.append(f"| {ds} | {cat} | {len(cr)} | {len(ids)} | {sum(bool(ys[i]['final_correct']) for i in ids)} |")
lines+=['', '同一query可能包含不止一种错误位置，本表各行不能相加当作独立query总数。36条2Wiki关系丢失是把亲属问题改成Who is已知人物；按保真口径判错，但通用人物检索仍可能提供有用资料。完整逐条改写、错误位置、最终输出、参考及F1/EM见 [REWRITE_ERROR_CATALOG.md](REWRITE_ERROR_CATALOG.md)。',
'', '## 子回答与goal分别报告','', '| 数据集 | 类型 | 正确 | 错误 | 待核事实，无真值标签 | 正确中包含合理弃答 |','|---|---|---:|---:|---:|---:|']
for ds,s in stats['summary'].items():
 for role,r in s['by_role'].items():lines.append(f"| {ds} | {role} | {r['answer_correct']} | {r['answer_incorrect']} | {r['answer_pending']} | {r['correct_reasonable_abstentions']} |")
lines+=['', '普通回答的高事实正确率不能代表已取得完整桥接信息。比如Hasbro敌人列表真实但缺Wendigo；Jewel Staite案例合理弃答，却仍无法取得年份。需要把正确弃答、真实不完整列表与得到所需事实区分。下面的敏感性表仅排除含合理普通弃答的query；它仍不把真实不完整列表重新判错。','']
lines+=conditions([('A_to_Y_query_excluding_reasonable_abstentions','普通回答 → 最终，排除合理普通弃答'),('GoalA_to_Y','goal预答 → 最终')])
lines+=['', '2Wiki goal错误23题中21题最终不正确，2题仍最终正确；HotpotQA为22题中16题最终不正确，6题仍正确。此关联提示应优先查看goal是否与正确中间事实矛盾；它不证明删掉goal或改提示词必然提高成绩。',
'', '## 已核对的具体失败与恢复轨迹','', '下表是有明确材料支撑的选例，不是对全部230个既有非满分query进行互斥、穷尽的因果分摊。每条均保留原始文件路径与哈希于ROOT_CAUSE_CASES.json。','', '| 数据集/编号 | 观察 | 最终输出 / 既有参考 | 已有F1 |','|---|---|---|---:|']
def esc(x):return str(x).replace('|','\\|').replace('\n',' ')
for v in case_records:lines.append(f"| {v['dataset']}/{v['review_index']} | {v['finding']} | {esc(v['prediction'])} / {esc(v['reference'])} | {v['native_F1']:.3f} |")
lines+=['', '## 结果偏低的含义和下一步审阅重点','',
'已确认的实际流程问题包括：检索命中后的证据合并裁剪；真实部分列表没有覆盖所需桥接对象；子回答错认事实角色或同名实体；goal在日期比较、国籍集合、亲属代际上覆盖正确中间事实；最终抽取错属性。建议先围绕这些已核对问题设计后续对照，所有代码／提示词修改和重跑仍须另获用户批准。本报告没有启动任何修复评估。',
'', '同时存在明显评分或参考问题：有效替代答案、别名、答案附加地理／机构描述、题目和参考所问对象不同，以及gold同名串接。它们会影响已有F1/EM，也会使“最终正确”这个条件概率端点与事实正确性不完全一致。所有主表仍保留既有官方口径；本次没有把有争议答案重判成满分。',
'', '| 数据集 | 既有非满分query | 其中goal事实正确 | 其中0<F1<1 | 其中F1=0 | D、普通A、R全正确但仍非满分 |','|---|---:|---:|---:|---:|---:|']
for ds,s in summary.items():lines.append(f"| {ds} | {s['final_failures']} | {s['failures_with_goal_answer_correct']} | {s['failures_with_partial_F1']} | {s['failures_with_zero_F1']} | {s['failures_with_all_D_ordinaryA_R_correct']} |")
lines+=['', '“goal事实正确而端点非满分”也包含正确弃答、答案粒度不同和真实替代答案，不能直接当作最终模型错误数量；同理，F1部分得分也不自动意味着仅是格式问题。',
'', '## 待核项和查证范围','', '仍有7条goal完整陈述的附加事实未解决，未强行赋正确／错误标签。它们只从goal相关分母排除：2Wiki5条、HotpotQA2条。全部普通子回答、分解、改写与原Y保留，不删除整个query。','', '| 数据集/编号/步骤 | 待核事实 |','|---|---|']
for p in read(run/'PENDING_FACTS.json')['items']:lines.append(f"| {byi[p['review_index']]['dataset']}/{p['review_index']}/{p['step_id']} | {p['claim']} |")
lines+=['', '检索材料或冻结数据能判断时以本地材料为准；仅缺证或冲突项做定向网页核对。没有以2026年的职位／球队变化追溯否定旧材料。WEB_FACT_CHECKS.json记录20项已作判断的查证，PENDING_FACTS.json保留未定事实与冲突。关键来源包括 [AFI的Flowing Gold](https://catalog.afi.com/Film/5028-FLOWING-GOLD)、[AFI的Street Kings](https://catalog.afi.com/Catalog/moviedetails/64772)、[Grammy的Joe Jackson记录](https://www.grammy.com/artists/joe-jackson/10992/)、[NYU Tisch的Kenneth A. Friedman履历](https://tisch.nyu.edu/about/directory/grad-film/101907044.html)、[Jack Spring在2019年的导演自述](https://www.comedy.co.uk/film/destination_dewsbury/special/how_to_make_a_movie_aged_19/)及[SBS官方历史](https://about.sbs.co.kr/history)。',
'', '## 补充表和复核记录','']
lines+=conditions([('A_history_to_R_ordinary','前序普通A → 普通改写'),('A_history_to_R_goal','前序普通A → goal改写'),('A_bound_to_R','实际显式绑定的前序A → 改写'),('A_to_Y_node','普通子节点 → 最终'),('R_to_Y_event_ordinary','普通改写记录 → 最终'),('R_to_Y_event_goal','goal改写记录 → 最终'),('R_to_Y_changed_event','确有变化的实际改写 → 最终'),('A_history_to_R_with_prefix','前序普通A → 改写，含未完成前缀')])
lines+=['', '冻结前的第二遍复核覆盖150个初判有误节点、固定随机抽取的193个其余节点；分解另复核40个初判不通过项及69个抽样通过项。保留首轮标签和修订，不覆盖旧记录。没有不同审阅者之间的一致性检验。小样本错误组的bootstrap区间可能很宽，单例或全成功组可退化；JSON保留非空重采样次数，不能把窄／退化区间视为足够证据。',
'', '可复查文件：', '', '- [CONDITIONAL_PROBABILITIES.json](CONDITIONAL_PROBABILITIES.json)：全部条件概率、分母与原query簇bootstrap区间。','- [QUERY_LABELS_AND_OUTCOMES.jsonl](QUERY_LABELS_AND_OUTCOMES.jsonl)：逐query标签与既有端点。','- [STAGE_LABELS_LOCKED.json](STAGE_LABELS_LOCKED.json)：逐节点依据、实际改写与原文件哈希。','- [ROOT_CAUSE_CASES.json](ROOT_CAUSE_CASES.json)：选例观察，不作全量因果归因。','- [FINAL_VISIBILITY_VERIFIED.json](FINAL_VISIBILITY_VERIFIED.json)：723条实际最终提示词上下文校验。','- [PENDING_FACTS.json](PENDING_FACTS.json)：7个事实工作项，无真值标签。','- [CURRENT_VERIFICATION.json](CURRENT_VERIFICATION.json)：冻结源、主评暂停标记及费用来源核对。',
'', '本次诊断没有新增RAG推理、GPU任务或外部评审模型请求。共享账本另有用户在其他任务中明确批准的quick90提示词实验／最小对照费用，单独列在核对文件中；不得把共享账本的变化归给本诊断，也不得宣称全局账本未变。']
(run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report':str(run/'REPORT.md'),'root_case_count':len(cases),'descriptive':summary},ensure_ascii=False))
