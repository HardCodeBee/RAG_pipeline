"""Create the third actual design snapshot; offline, without production imports."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
from design_tools import build_review_document

HERE=Path(__file__).resolve().parent


def main():
    dest=HERE/'revisions'/'r03'
    assert not dest.exists(), 'Preserve existing snapshot'
    shutil.copytree(HERE/'revisions'/'r02', dest)
    changes={
      'rewrite':[
        ('Keep the language of the Current question. Return the question only.',
         'Keep the language of the Current question. Return one question on one line,\nwith no explanation. Treat literal symbols in titles as text, not step references.')],
      'ordinary':[
        ('answer when a passage concerns the same entity, relation and scope. Distinguish a',
         'answer when a passage concerns the same entity, relation and scope. Prior answers\nremain usable when passages do not repeat them; do not invent corroboration. If\nmaterial claims conflict at the same scope, do not resolve them by repetition or\nrecency alone. Distinguish a'),
        ('Return two fields:',
         'Return exactly the two fields below, in order, each label starting a new line.\nUse one paragraph per field, without a preamble, extra headings or Markdown fences.')],
      'goal':[
        ('scope, state the correction. Do not prefer a prior answer merely because it came first.',
         'scope, state the correction. An earlier answer can remain useful without being\nrepeated in a new passage. Resolve apparent conflicts by entity, time and scope;\nif a material same-scope conflict remains, do not invent a resolution.'),
        ('fields are a candidate synthesis, not independently verified evidence:',
         'fields are a candidate synthesis, not independently verified evidence. Use\nexactly these labels in this order, each starting a new line, with one paragraph\nper field and no preamble, extra headings or Markdown fences:')],
      'final':[
        ('When passages contradict an intermediate claim about the same entity and scope,\nuse the passage. Do not substitute a different person\'s role or a related attribute.',
         'Prefer a directly stated, matching passage to an unsupported intermediate\nclaim. Resolve apparent conflicts by entity, relation, time and scope; if a material\nconflict remains unresolved, do not invent a resolution. Do not substitute a\ndifferent person\'s role or a related attribute.')]
    }
    for stage, pairs in changes.items():
        path=dest/(stage+'.txt'); text=path.read_text(encoding='utf8')
        for old,new in pairs:
            assert old in text, (stage,old)
            text=text.replace(old,new,1)
        path.write_text(text,encoding='utf8')
    path=dest/'examples.json'
    bank=json.loads(path.read_text(encoding='utf8'))
    stages=bank['stages']
    corrected='Follow the spouse relation to Tomas Reed, then the father relation to Orin Reed; the requested birthplace is Orin Reed\'s.'
    for x in stages['goal']:
        if x['id']=='family_role':x['output']['Link']=corrected
    for x in stages['final']:
        if x['id']=='family_role':x['synthesis']['Link']=corrected
    stages['ordinary'].append({
      'id':'truthful_partial_list',
      'original_question':'Which novels did Seren Pike write?',
      'current_question':'Which novels did Seren Pike write?',
      'retrieval_question':'Which novels did Seren Pike write?',
      'history':[],
      'passages':['Seren Pike wrote Quiet Harbor. This excerpt discusses only one of her novels.'],
      'output':{
        'Support':'The passage supplies one responsive novel title but does not claim a complete bibliography.',
        'Answer':'Seren Pike wrote Quiet Harbor; the available passage does not establish her complete list of novels.'}})
    # Adjacent counterexamples differ in whether the lower bound is decisive.
    goal_order=['family_role','seats','incomplete_total','sufficient_lower_bound',
                'alternative_route','correct_prior','multiple_posts','decisive_false']
    stages['goal'].sort(key=lambda x:goal_order.index(x['id']))
    stages['final'].append({
      'id':'map_back_to_requested_object',
      'original_question':'Which film has the older director, Willow Dawn or Stone Dawn?',
      'history':[],
      'passages':['Rena Moss directed Willow Dawn and was born in 1940. Dena Lake directed Stone Dawn and was born in 1950.'],
      'synthesis':{
        'Facts':'Rena Moss, born in 1940, directed Willow Dawn. Dena Lake, born in 1950, directed Stone Dawn.',
        'Link':'Rena Moss was born earlier and is the older director.',
        'Candidate':'Rena Moss'},
      'output':'Willow Dawn',
      'perturbation':'Final-only demonstration: the intermediate person is correct but Candidate has the wrong requested answer type.'})
    path.write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (dest/'INTERFACE.md').write_text('''# r03 输入与输出约定

这是独立设计快照，不是生产实现、模型运行结果或已批准的新主评估协议。

1. 分解器、描述式子问题、原始步骤顺序、阶段独立上下文、双路检索与原query的root Top10不变。无普通子步骤时允许直接goal；这只是有效路径覆盖，不更改旧诊断样本。
2. rewrite消费原始当前问题与此前Answer事实句。无历史不新增改写调用。只输出一条检索问题；原始当前问题仍是ordinary的任务。
3. ordinary输入原题、原始当前子问题、全部此前Answer、当前合并reader及实际检索问题。最后一个字段是新增且可能错误的上下文，必须作为独立变化披露。
4. ordinary原始Support与Answer全文保存；仅Answer进入后续history、改写及final普通QA区。Answer必须带上本地结果的实体、关系和必要条件。Support不会累积为新上下文或检索证据。
5. goal输入原题、普通Answer历史、goal合并reader和实际goal检索问题。输出Facts、Link、Candidate全文进入final；Facts和Link不是已验证证据，也不能证明模型内部推理忠实。
6. final输入原题、普通Answer历史、goal三字段和原query的root Top10。不会额外偷加goal原始reader。goal来源的新事实可以通过综合段落进入final；未出现在root不等于被root反驳。
7. 候选ordinary不再调用旧短句LLM formatter，避免把Support/Answer误当一句话处理。旧P02对照保留原行为。这是方法包差异，不能归因于提示文字或CoT单一因素。
8. 只有完成原因为stop的响应进入接受解析。缺失、重复、空字段、标签顺序错误、前导解释或代码围栏记为格式失败；保留原文，不调用额外模型修复，不从Support猜Answer，不复用旧答案补洞。解析不会核验事实、完整性或逻辑。
9. 输出按完整内容保存；50/140词都是软目标，不由客户端按词数裁剪。具体API上限未冻结，后续验证必须同时记录上限、finish_reason及对照条件。length响应不作为成功字段结果，也不得通过截取正确前半段改善分数。
10. few-shot只用人工虚构题材。示例元数据、预期行为、禁止推断、标准答案等审核字段不进入运行时真实题目。final的错误Candidate示例只是在final输入端展示纠正任务，不作为goal正确输出。

离线design_tools.py仅是接口草图。它不连接模型、不访问账务缓存，也不意味着生产runner已经接入。最终短答原样保留；解析不自动删除前缀、解释或补充限定，不利用参考答案挑选输出。
''',encoding='utf8')
    stats=build_review_document('r03')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:
        h.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='r03_saved',
          changes=['removed unsupported cross-stage family fact','clarified usable prior facts and material conflicts',
          'made field protocol explicit','added truthful partial-list and answer-type examples'],sizes=stats),ensure_ascii=False)+'\n')
    print(json.dumps(stats,ensure_ascii=False))


if __name__=='__main__':main()
