"""Group already adjudicated errors and associate existing final outcomes."""
from collections import Counter
import json
from prepare_inventory import OUT,read,rows,write
run=OUT/'offline_review_v2'
events=[e for e in rows(run/'REWRITE_EVENTS.jsonl') if not e['rewrite_correct']]
qs={r['id']:r for r in rows(run/'QUERY_LABELS_AND_OUTCOMES.jsonl')}
ys={r['id']:r for r in rows(run/'OUTCOMES_SEPARATE.jsonl')}
groups={}
def assign(keys,category,description):
 for key in keys:
  assert key not in groups,key
  groups[key]=(category,description)
generic=[240,247,256,265,339,340,350,363,371,379,382,399,402,403,415,418,436,451,457,498,506,530,534,569,603,605,627,642,667,668,676,677,683,694,700,715]
assign([(i,3) for i in generic],'关系丢失：改成已知实体是谁','将原来要查的亲属关系改成通用人物介绍。实体已经在前序回答中给出；实际改写丢掉了原关系目标。作为检索词仍可能召回人物资料，因此不能仅凭保真错误断定最终效果下降。')
assign([(29,2),(268,3),(344,3),(377,3),(664,3)],'丢失特定实体身份','用不唯一的履历或年份描述替换姓名／片名，使检索问题不再固定于原来的对象。')
assign([(27,2),(45,2),(45,3),(84,3),(93,2),(107,2),(107,3),(133,2),(500,2)],'绑定错误的前序对象','改写沿用了前序回答中的错误对象或词义；可能忠实复述前序内容，但实际目标已偏离当前问题。')
assign([(30,3),(166,2),(668,2)],'带入错误的附加事实','把前序回答中错误的子团体、作品类型或亲属描述写入查询。')
assign([(349,3),(534,2),(560,3)],'亲属关系或代际错误','把女婿当配偶、孙辈当子辈，或在父系祖母之外又增加了一代亲属关系。')
assign([(305,5),(394,5),(479,3)],'比较命题被换成分别核实','原问题问是否同国，改写变为分别来自两个不同国家是否属实；两者可以有相反的真假答案。')
assign([(146,3),(599,3),(708,3)],'实体与属性的类型混淆','把国家／大学系统这一属性值当成原来的电影／大学，保留旧比较句式后产生类型错误。')
assign([(31,3),(56,3),(43,3),(63,2),(71,2),(77,2),(92,3),(107,4),(108,2),(144,3),(150,2),(187,2),(190,3),(198,2),(225,3),(229,3)],'未知答案被填入问题','把待求的姓名、地点或属性直接写成既定对象／结论，改成确认句或重复询问已知值。')
assign([(58,3),(87,3),(119,3),(140,3),(149,2),(201,2),(221,2)],'关系或角色被改变','替换了原问题要求的关系、施事／受事或共同任职约束；逐条原问题与改写见下表。')
assign([(128,3),(158,3),(507,5)],'候选范围或限定条件改变','扩成所有电影版本、删除人口条件，或把年龄比较换成某个生日条件，改变了需要回答的命题。')
assert set(groups)=={(e['review_index'],e['step_id']) for e in events},(set(groups)-{(e['review_index'],e['step_id']) for e in events},{(e['review_index'],e['step_id']) for e in events}-set(groups))
out=[]
for e in events:
 cat,note=groups[e['review_index'],e['step_id']];y=ys[e['id']]
 out.append({**e,'error_location':cat,'error_description_zh':note,'original_query':qs[e['id']]['question'],'final_prediction':y['prediction'],'existing_reference':y['reference'],'native_F1':y['f1'],'native_EM':y['em'],'Y_existing_F1_or_EM_1':bool(y['final_correct'])})
(run/'REWRITE_ERROR_CATALOG.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in out),encoding='utf-8')
lines=['# 改写错误逐项记录','', '仅收录已完成流程中按既定保真／目标口径判错的实际生效改写。最终正确性沿用已有F1=1或EM=1，不重新评分。前序回答、原查询和改写检索结果可能抑制错误传播；这里的同现不代表因果。','', '| 数据集 | 审阅编号/步骤 | 错误位置 | 最终答案 | 既有参考答案 | F1 / EM | 最终是否正确 |','|---|---|---|---|---|---|---|']
def esc(s):return str(s).replace('|','\\|').replace('\n',' ')
for v in out:lines.append(f"| {v['dataset']} | {v['review_index']}/{v['step_id']} | {v['error_location']} | {esc(v['final_prediction'])} | {esc(v['existing_reference'])} | {v['native_F1']:.3f} / {v['native_EM']:.0f} | {'是' if v['final_correct'] else '否'} |")
for v in out:
 lines+=['',f"## {v['dataset']} · {v['review_index']}/{v['step_id']}",'',f"原始query：{v['original_query']}",'',f"当前问题：{v['current_request']}",'',f"实际改写：{v['effective_rewrite']}",'',f"错误位置：{v['error_location']}。{v['error_description_zh']}",'',f"逐条依据：{v['reason']}",'',f"最终输出：{v['final_prediction']}；既有参考：{v['existing_reference']}；已有F1={v['native_F1']:.6f}，EM={v['native_EM']:.0f}。"]
(run/'REWRITE_ERROR_CATALOG.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write(run/'REWRITE_ERROR_COUNTS.json',{'events':len(out),'queries':len({v['id'] for v in out}),'by_dataset':{ds:{'events':sum(v['dataset']==ds for v in out),'queries':len({v['id'] for v in out if v['dataset']==ds}),'locations':dict(Counter(v['error_location'] for v in out if v['dataset']==ds))} for ds in ['2wiki','hotpotqa']}})
print({'rewrite_error_events':len(out),'queries':len({v['id'] for v in out})})
