"""Posthoc human-readable notes for completed developer cases only.

This script is not imported by generation, sampling or strategy code. Notes are
bound to literal parent/candidate answers, not used to alter any official score.
"""
import json
from .report_round2 import OUT, read


NOTES = {
('g02_evidence_goal', '2wiki:6d2f94300bb011ebab90acde48001122'): (
    '同一实体的称谓扩展，词面F1下降',
    'Timothy Laurence变为Sir Timothy Laurence，被识别的人没有改变。此次F1下降来自多出的称谓；没有从返回片段确认Sir本身，不能据此宣称新增称谓有证据或身份事实恶化。', []),
('g02_evidence_goal', 'musique:2hop__789565_416401'): (
    '同一实体的别名表达变化，词面F1上升',
    '两版均指王继鹏。用Emperor Kangzong of Min替代also known as Wang Chang改变冗余词数量，未修复新的亲属事实。',
    ['goal检索《Wang Jipeng》：used the name Wang Chang ... formally Emperor Kangzong of Min；其父为Wang Yanjun。']),
('g02_evidence_goal', 'hotpotqa:5ae0a48155429945ae9593df'): (
    '定义表达补全，F1上升但没有新的事实判断',
    '父版fee or commission和候选均表达vigorish的手续费含义。候选补bookmaker/accepting a bet，贴近标准答案用词。不是错误对象被修复。',
    ['原Top2《Vigorish》：the amount charged by a bookmaker ... for taking a bet from a gambler。']),
('g02_evidence_goal', 'hotpotqa:5a84ad3a5542992a431d1a8b'): (
    '从标准答案改为缺答；当前缺少完整关系证据',
    '父版Back to December匹配标准答案，候选改为None。原Top1只列出Lemoine执导作品中包含该歌，未在实际返回材料中找到该作品表达致歉/祈求原谅的关系。更保守拒答损失F1，但不能把它直接称为丢弃已有完整支持的答案。',
    ['原Top1《Yoann Lemoine》：notable works include ... Taylor Swift’s single Back to December；该片段不解释致歉含义。']),
('g02_evidence_goal', 'musique:3hop1__109422_720914_27537'): (
    '缺失桥接前提后拒答，损失可用局部答案',
    '前置子回答未证实Titian死亡地点；原Top1已直接说明Roncalli离开Venice是赴罗马秘密会议。g02因完整链有缺口拒答，最终也放弃已有局部原因。不能称整条证据链已完整，也不能把该损失全归于检索不含最终原因。',
    ['原Top1《Pope John XXIII》：Roncalli left Venice for the conclave in Rome well aware that he was papabile。',
     '原Top2《The Punishment of Tythus》只说明作者Titian；前置回答仍未知其死亡地点。']),
('g02_evidence_goal', 'musique:4hop1__94201_642284_131926_13165'): (
    '错误确定答案转为缺答，F1下降不等于事实恶化',
    '父版Adams–Onís Treaty并非标准Treaty of Paris，却因共同Treaty词得到0.4。前置死亡城市和河流均未知。候选承认无法确定，是更保守的结果；不能把0.4到0解释为把正确条约答错。',
    ['原Top1《Adams–Onís Treaty》谈Pacific Ocean；原Top7《Military history of the United States》谈Treaty of Paris和Mississippi River。',
     '冻结前置回答：Ralph Rapson死亡城市、该城市附近水域均不能确定。']),
('g02_evidence_goal', 'hotpotqa:5abec66b5542997ec76fd360'): (
    '同义改写，词面F1上升',
    'and recipients of改为who received，仍为同一组共同事实。两版共同的一战飞行王牌和勋章陈述均有原文支持，没有新的组合事实修复。',
    ['原Top1《Josef Veltjens》及Top2《Hermann Göring》均描述World War I fighter ace及Pour le Mérite。']),
('g02_evidence_goal', '2wiki:cdb8b7540bd911eba7f7acde48001122'): (
    '正确实体附加受支持地点，词面F1下降',
    '墓地名称相同，候选附加in Los Angeles。该地点有原文支持，不能算事实错误；额外地点降低F1。',
    ['goal Top2《Sidney Lanfield》：He is interred in the Westwood Village Memorial Park Cemetery in Los Angeles。']),
('g02_evidence_goal', 'hotpotqa:5a72a9ac5542991f9a20c55f'): (
    '同一节目名称增加受支持限定，词面F1下降',
    '两版均指Today Show第四小时。加NBC未改变节目，原文支持；不是组合或检索事实恶化。',
    ['原《Hoda Kotb》和《Brian Balthazar》均将NBC、Today Show第四小时及Hoda Kotb连接。']),
('g02_evidence_goal', '2wiki:604119f2088011ebbd6cac1f6bf848b6'): (
    'goal中事实错误得到修正，最终答案和F1未变',
    '父goal把Kenneth MacKenna出生年误写1989，却仍选择Sleepers East。候选改为原文1899，修正了比较前提；两版最终都为Sleepers East，因此F1不反映这项事实修复。',
    ['实际reader_packet《Kenneth MacKenna》：August 19, 1899 – January 15, 1962。']),
('g02_evidence_goal', 'musique:3hop2__368986_89048_66294'): (
    'goal将未找到证据扩大为不存在，最终F1未变',
    '父goal只说没有检索片段说明某人扮演Joe Jackson。候选却断言No movie titled Anthology exists。返回的同名专辑材料不能证明世界上不存在同名电影，属于无依据的全称否定；最终仍为Cannot be determined、F1为0，掩盖该中间结果风险。',
    ['实际reader_packet《Anthology (Michael Jackson album)》仅定义该专辑，没有给出不存在同名电影的穷尽性证据。']),
('g02_evidence_goal', 'musique:3hop2__326964_265410_7713'): (
    '与原文冲突的goal事实仍保留，最终F1持平0',
    '父版Zero和候选长句均声称南京从未为Yangzhou首府。实际已读原文写about400years。新版规则没有修复古今地名关系误读，只是改变了错误结论的表达。',
    ['实际reader_packet《Nanjing》：had been the capital city of Yangzhou for about 400 years from late Han to early Tang。']),
('g02_evidence_goal', 'musique:4hop1__443274_17130_70784_61381'): (
    '最终对象和条件改变，F1仍为0',
    'Iran/1979改成Lebanon/22 November1943；这不是词面变化。候选只沿Israel北邻取Lebanon，未满足原query的Persian Gulf关联。原题本身存在复杂地理指代，当前变化不能认定为正确修复。',
    ['实际《Israel》写Lebanon to the north；该局部关系不足以满足原query全部条件。']),
('p02_role_headers', '2wiki:16a91a500bb011ebab90acde48001122'): (
    '同一名单去除连接词，词面F1上升',
    '五个人名完全不变，只去掉最后and，未改变亲属关系判断。标准答案仅Stefan Adam，返回完整名单仍使F1较低；不能仅凭标准答案判其他四人都是事实错误。', []),
('p02_role_headers', 'musique:2hop__789565_416401'): (
    '同一实体去除多余别名，短答案匹配改善',
    '仍回答Wang Jipeng，只删除括号中的Wang Chang别名，F1从0.4444到1。不属于新事实或新增证据。',
    ['冻结goal明确给出Wang Jipeng别名Wang Chang；实际《Wang Jipeng》支持该别名。']),
('p02_role_headers', 'hotpotqa:5ae0a48155429945ae9593df'): (
    '定义表达补全，F1上升但没有新的事实判断',
    'fee or commission扩展成charged by a bookmaker on a bet，贴近标准答案的关系词；goal和证据未变。',
    ['原Top2《Vigorish》明确给出bookmaker向gambler收取赌注费用的定义。']),
('p02_role_headers', 'musique:2hop__289517_139339'): (
    '同一错误否定的措辞扩展，F1下降',
    'No award改No award nomination，仍沿用goal的未获提名否定，和标准Academy Award for Best Live Action Short Film不一致。goal把识别演员的After Hours额外变成for his role in After Hours的提名范围，原query并未明确该限定。词面部分匹配减少，范围误读未修复。', []),
('p02_role_headers', 'musique:3hop1__857_846_7770'): (
    '相同地点关系扩为整句，词面F1下降',
    '均为长江经过南京西侧后经北侧；候选把主语及flows补回，增加原题未要求重复的词。关系事实未变，短答案压缩退步。', []),
('p02_role_headers', 'hotpotqa:5a72a9ac5542991f9a20c55f'): (
    '同一节目名称增加受支持限定，词面F1下降',
    '增加NBC称谓，仍指Today Show第四小时；不属于事实错误。',
    ['原《Hoda Kotb》和《Brian Balthazar》均支持NBC与该节目第四小时的关系。']),
('p02_role_headers', 'musique:3hop2__326964_265410_7713'): (
    '错误确定答案转为缺答，F1持平0',
    '父Zero和冻结goal的never结论与已读about400years冲突。候选未采用Zero而改为缺答，但没有恢复正确时长。可以观察到减轻依赖错误goal，不能声称正确解决了问题。',
    ['实际reader_packet《Nanjing》明确about400years。']),
('p02_role_headers', 'musique:4hop1__443274_17130_70784_61381'): (
    '错误确定答案转为缺答，F1持平0',
    'goal仍为Iran/1979，候选最终改为Cannot be determined。说明角色标题条件下最终答案未强制复制goal，但没有恢复标准1932，不能算事实答对。', []),
('p02_role_headers', '2wiki:99cf9bb2085a11ebbd5cac1f6bf848b6'): (
    '依据相同事实修正整体比较，真实组合修复',
    '冻结goal错误声称Feltrinelli活得更久，同句却列其45岁8月、Gogarten80岁9月。父final复制错误赢家，角色标题版改答Gogarten。实际原文的两组生卒日也支持80>45；不是新增检索或别名匹配。',
    ['实际《Friedrich Gogarten》：January13,1887–October16,1967。',
     '实际《Giangiacomo Feltrinelli》：19June1926–14March1972。']),
('p02_role_headers', '2wiki:f37818400bd911eba7f7acde48001122'): (
    '同一地点去除别名，短答案匹配改善',
    'Mitau(now Jelgava)缩短为Mitau，没有改变死亡地点；未取得新关系事实。', []),
('p02_role_headers', 'musique:3hop1__458916_651917_47686'): (
    '同一地点去除街道地址，短答案匹配改善',
    '两版均为Toronto Coach Terminal，只去除at610BayStreet。原文支持地址，删除提升词面F1，不属于旧答案地点错误。',
    ['实际《Toronto Coach Terminal》：It is located at610BayStreet ... leased ... Coach Canada and Greyhound Canada。']),
('p02_role_headers', 'musique:3hop1__671542_651917_47686'): (
    '同一地点去除街道地址，短答案匹配改善',
    '两版均为Toronto Coach Terminal，去掉额外街道地址后匹配标准答案。', []),
('p02_role_headers', 'musique:3hop2__326964_734214_7713'): (
    '错误确定零值转为缺答，F1下降不等于事实恶化',
    '父Zero years并不是标准about400years，只因共享years得0.4。候选改为缺答，未恢复正确历史时长，但也未产生新的错误零值。不能把该下降说成丢失正确答案；与另一Anhua Gao题相比，本题实际packet未检到400years原文。', []),
('p02_role_headers', 'musique:4hop3__373866_5189_38229_86687'): (
    '时间粒度缩短命中计分别名，不证明事件时间修复',
    '12September1990缩为1990，在原生Mu计分中达到1；标准主答案是May1990。goal仍把问题与9月12日条约混接，因此不能声称修复了月份/事件对应错误。', []),
('p02_role_headers', 'hotpotqa:5a84d2ea5542992a431d1aa8'): (
    '相同职业列表去连接词，词面F1下降',
    'actor、film producer、director均未变，只去掉and。两版同样未包含标准writer；本次F1下降并不说明新增了错误职业事实。', []),
('p02_role_headers', 'musique:2hop__22431_16335'): (
    '压缩删除原题所求关系细节，答案信息损失',
    '候选删除父回答末尾的maternal lineages及Roman-era intermarriage or conversion，尤其删除了标准要求的通婚和皈依两个因素。留下较宽泛的ancestry/admixture不能完整替代这些具体因素。属于内容压缩损失，不能只算同义词面差异。',
    ['实际《Ashkenazi Jews》：genetic-similarity ... may be due to inter-marriage and conversions in the time of the Roman Empire。']),
('p02_role_headers', 'hotpotqa:5ac298805542990b17b15429'): (
    '最终回答对象/类型修复',
    '冻结goal已经给出Austral Líneas Aéreas。父final回答Yes，候选改为该实体名并匹配标准。没有新增证据，但恢复数据集所求对象；原query表面近似是非句。另须注意原文说Austral是Aerolíneas Argentinas的姊妹公司，而非LAPA，goal仍误绑该关系；不能将实体答对解释为整条链事实正确。',
    ['实际《Austral Líneas Aéreas》：the sister company of Aerolíneas Argentinas。',
     '实际《Líneas Aéreas Privadas Argentinas》：LAPA打破Aerolíneas Argentinas及其姊妹Austral控制的市场。']),
('p02_role_headers', 'musique:2hop__20256_71302'): (
    '从缺证据的绝对否定转为缺答，F1持平0',
    'No actor produced the Freddie Mercury film改为Cannot be determined，不再复制goal的绝对否定；未恢复标准Vito Corleone。仅凭没有取得该身份联系，不能断言不存在这样的演员制片人。', []),
('p02_role_headers', 'musique:2hop__562154_69048'): (
    '同一未匹配目标人物的姓名扩展，F1持平0',
    'Rosaline Bozimo扩为Rosaline Patricia Irorefe Bozimo，仍是同一人物，标准为Honorable Justice Abiodun Smith。本次既未修复人物目标，也不能把姓名变长称为新的事实错误。', []),
('s03_g01_plus_f01', 'hotpotqa:5a77cb5e55429967ab1052a7'): (
    '复制错误goal扩大角色集合，事实性回退',
    '父final Kodos正确，候选跟随错误goal改为Kang and Kodos。实际原文明确Kang由Harry Shearer配音，Kodos由Dan Castellaneta配音；不是两个都正确而仅词面变长。新增最终核对提示未阻止采用错误辅助结论。',
    ['实际《Kang and Kodos》：Kang is voiced by Harry Shearer and Kodos by Dan Castellaneta。']),
('s03_g01_plus_f01', 'musique:3hop1__68802_89752_49923'): (
    '同一数值改为阿拉伯数字，纯词面F1下降',
    'Sixteen改成16，仍为16个县，事实与冻结goal没有改变。原生计分不等价处理该拼写，使F1从1到0；保留原始分数，不事后改标准答案或挑选格式。', []),
('s03_g01_plus_f01', 'hotpotqa:5adf9ee75542992d7e9f93c0'): (
    '省略问题已限定的计数单位，词面F1下降',
    '1.5million households缩为1.5million，数值相同且问题已问households；不是新增错误数量。词面计分因少households从1到0.8。',
    ['实际《Nickelodeon (Africa)》：reaches1.5million households in49countries。']),
('s03_g01_plus_f01', 'hotpotqa:5ab7f0b9554299366779405d'): (
    '相同错误否定改为明确整句，F1持平0',
    'None改为No vegetation type in common，仍沿goal仅比较woody/herbaceous等生长形态后否定共同类型，未恢复标准的共同上位类别plants。没有证据表明此次解释更具体便代表事实更准确。', []),
}

# These observed answer pairs are byte-identical across the two completed pilots;
# inherit only the offline reading note, never generation, scores or traces.
NOTES[('s03_g01_plus_f01', 'musique:2hop__789565_416401')] = NOTES[('p02_role_headers', 'musique:2hop__789565_416401')]
NOTES[('s03_g01_plus_f01', 'hotpotqa:5a72a9ac5542991f9a20c55f')] = NOTES[('p02_role_headers', 'hotpotqa:5a72a9ac5542991f9a20c55f')]
for duplicate_id in ['2wiki:99cf9bb2085a11ebbd5cac1f6bf848b6', 'musique:3hop1__458916_651917_47686',
                     'hotpotqa:5a84d2ea5542992a431d1aa8', 'hotpotqa:5ac298805542990b17b15429',
                     'musique:2hop__562154_69048']:
    NOTES[('s03_g01_plus_f01', duplicate_id)] = NOTES[('p02_role_headers', duplicate_id)]


def main():
    existing = read(OUT / 'CASE_REVIEWS.progress.json', optional=True) or []
    saved = {(x['arm'], x['id']): x for x in existing}
    for row in read(OUT / 'CHANGED_CASES.progress.json'):
        key = row['arm'], row['id']
        if key not in NOTES: continue
        classification, note, evidence = NOTES[key]
        saved[key] = {'arm': row['arm'], 'id': row['id'], 'parent_final': row['parent_final'],
                      'candidate_final': row['candidate_final'], 'classification': classification,
                      'note': note, 'evidence': evidence, 'scope': 'posthoc recorded-case reading; not a blinded hallucination judge'}
    (OUT / 'CASE_REVIEWS.progress.json').write_text(json.dumps(list(saved.values()), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'reviewed_cases': len(saved), 'new_API_calls': 0}))


if __name__ == '__main__': main()
