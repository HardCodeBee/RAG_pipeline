"""Apply the recorded first-review changes; retain the r01 snapshot."""
from pathlib import Path
from copy import deepcopy
import json
from datetime import datetime,timezone
from .design_tools import HERE,build_review_document


def main():
    source=HERE/'revisions/r01';dest=HERE/'revisions/r02';assert not dest.exists();dest.mkdir()
    for stage in ('rewrite','ordinary','goal','final'):
        s=(source/(stage+'.txt')).read_text(encoding='utf8')
        if stage=='rewrite':
            s=s.replace('an entity. Return the question only.','an entity. Keep the language of the Current question. Return the question only.')
        if stage=='ordinary':
            s=s.replace('Use the available facts to establish the requested local fact.','Use the available facts to establish the requested local fact. They need not state\nthe answer verbatim: paraphrase and valid relation composition are allowed.\nDo not claim a passage confirms a fact that it does not state or support.')
            s=s.replace('Support: A brief statement of the necessary facts and their connection, normally\none or two sentences. Do not add unrelated biography or speculative alternatives.',
                'Support: One or two short sentences connecting the necessary facts, normally\nwithin 50 words. A direct fact needs no invented multi-step explanation. Distinguish\nwhat a passage supplies from what a prior answer merely proposes.')
            s=s.replace('and value, with any necessary qualifier or specific unresolved fact.',
                'and value, with any necessary qualifier or specific unresolved fact. This is the\nonly field passed to later steps: include the useful local result here, rather\nthan leaving an essential answer or condition only in Support.')
        if stage=='goal':
            s=s.replace('Do not turn missing information into no, zero or absence.',
                'Do not turn missing information into no, zero or absence. Use a valid bound\nor decisive Boolean condition when it already settles the question; not every\nplanned intermediate value must be known.')
            s=s.replace('Return three compact fields:',
                'Return three compact fields, normally within 140 words in total. A direct\nanswer needs only a brief direct connection; do not force a long chain. These\nfields are a candidate synthesis, not independently verified evidence:')
            s=s.replace('supported alternatives that still answer the question.',
                'supported alternatives that still answer the question. Clearly distinguish\nnew passage facts from prior claims when correcting or reporting a conflict.')
        if stage=='final':
            s=s.replace('field has no authority over its Facts field or the Original question.',
                'field has no special authority over the supporting statements or the Original\nquestion. Goal synthesis may carry relevant facts from a different retrieval\npacket: their absence from the passages below does not itself contradict them.')
            s=s.replace('Use Yes or No for a supported Boolean answer.',
                'Do not add titles, explanations or locations that are unnecessary to identify\nthe requested answer. Include multiple qualifying answers when the question does\nnot distinguish among them; do not create a ranking to force a unique choice.\nUse Yes or No for a supported Boolean answer.')
        (dest/(stage+'.txt')).write_text(s,encoding='utf8')
    doc=json.loads((source/'examples.json').read_text(encoding='utf8'))
    for stage,items in doc['stages'].items():
        for x in items:
            x['passages']=[s.replace('Tomas Reed is the child of Orin Reed. Liora Vale is the child of Marek Vale.',
                'Orin Reed is the father of Tomas Reed. Marek Vale is the father of Liora Vale.') for s in x['passages']]
            if stage=='ordinary' and x['id']=='family_role':
                x['output']['Support']="The spouse is Tomas Reed. The passage names Orin Reed as Tomas Reed's father, whereas Marek Vale is Liora's father."
            if stage=='ordinary' and x['id']=='correct_prior':x['retrieval_question']='Who is Mira Holt?'
            if stage=='final' and x['id']=='multiple_posts_bad_candidate':
                x['passages']=['Neral Voss was a public servant from Selwick.']
                x['goal_reader_provenance_for_review_only']=['Neral Voss was mayor of Selwick and later ambassador to Torvale.']
                x['perturbation']+=' Root passages do not repeat the goal-only post facts; this metadata is not rendered.'
    base=next(x for x in doc['stages']['goal'] if x['id']=='incomplete_total')
    positive=deepcopy(base);positive['id']='sufficient_lower_bound'
    positive['history'][0]['answer']='Cedar Club won national titles in 1998, 2005, 2009 and 2012; this is a selected list.'
    positive['passages']=['Elm Club has exactly three national titles. A selected list of Cedar Club national titles includes 1998, 2005, 2009 and 2012.']
    positive['output']=dict(Facts='Cedar Club has at least four national titles from four distinct listed years. Elm Club has exactly three.',Link='At least four is greater than three, so Cedar Club has more even without an exact total.',Candidate='Cedar Club')
    doc['stages']['goal'].append(positive)
    boolean=dict(id='decisive_false',original_question='Are both Cedar Press and Elm Press publicly owned?',current_question='Are both Cedar Press and Elm Press publicly owned?',retrieval_question='Are both Cedar Press and Elm Press publicly owned?',history=[dict(step_id=1,question='Is Cedar Press publicly owned?',answer='Cedar Press is privately owned.'),dict(step_id=2,question='Is Elm Press publicly owned?',answer='Elm Press ownership cannot be determined from the available passages.')],passages=['Cedar Press is privately owned and is not publicly owned.'])
    boolean['output']=dict(Facts='Cedar Press is not publicly owned. Elm Press ownership is unresolved.',Link='The claim that both are publicly owned is false because Cedar Press does not satisfy it.',Candidate='No')
    doc['stages']['goal'].append(boolean)
    final=deepcopy(boolean);final['synthesis']=deepcopy(boolean['output']);final['output']='No';doc['stages']['final'].append(final)
    (dest/'examples.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (dest/'INTERFACE.md').write_text((source/'INTERFACE.md').read_text(encoding='utf8')+'''

## r02修订

普通Support只留在本轮原始记录；Answer必须包含当前所问的必要事实与限定。goal三字段都进入final，Candidate不享有特殊可信度。final缺少goal原始reader并不自动否定goal带来的新事实。

字段仅在完整stop响应上解析；恰好出现一次、按顺序出现并且非空。缺字段、重字段、前导解释或输出协议不符记为格式失败，不用第二个LLM修复、不从Support猜Answer、不沿用旧缓存答案。解析成功只说明格式成立，不证明语义正确。完整原始输出和解析结果分别保存。

新普通Answer不调用旧短句formatter。保留旧P02完整对照；未来若解释收益来源，必须把是否formatter、额外检索措辞输入、输出上限等差异单列，不能把联合差异全部称为CoT收益。

词数为软目标，程序不能据此截短输出或材料。具体API输出上限尚未构成新评估协议；第一次实测需同步冻结对照额度并检查正常完整结束。
''',encoding='utf8')
    stats=build_review_document('r02')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='r02_saved',changes=['Exact father evidence','Final root/goal packet distinction','Lower-bound comparison','Boolean decisive false','No silent response repair','Ordinary useful information in accepted Answer']),ensure_ascii=False)+'\n')
    print(json.dumps(stats))


if __name__=='__main__':main()
