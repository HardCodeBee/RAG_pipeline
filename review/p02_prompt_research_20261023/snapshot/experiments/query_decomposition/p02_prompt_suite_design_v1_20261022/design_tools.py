"""Offline rendering/parse sketch for review. No production imports or execution."""
from pathlib import Path
import json
import re

HERE=Path(__file__).resolve().parent
STAGES=('rewrite','ordinary','goal','final')
INPUTS={
    'rewrite':('history','current_question'),
    'ordinary':('original_question','history','current_question','retrieval_question','passages'),
    'goal':('original_question','history','retrieval_question','passages'),
    'final':('original_question','history','synthesis','passages'),
}
FIELDS={'ordinary':('Support','Answer'),'goal':('Facts','Link','Candidate')}


def text(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Expected nonempty text')
    return value


def history(rows):
    if not isinstance(rows,list):raise ValueError('History must be a list')
    ids=[];parts=[]
    for row in rows:
        if set(row)!= {'step_id','question','answer'}:raise ValueError('Unexpected history fields')
        sid=row['step_id']
        if type(sid) is not int or sid<1:raise ValueError('Invalid step identifier')
        ids.append(sid)
        parts.append(f"Step {sid}\nQuestion: {text(row['question'])}\nAnswer: {text(row['answer'])}")
    if ids!=sorted(set(ids)):raise ValueError('History must have increasing unique identifiers')
    return '\n\n'.join(parts) if parts else '(No previous answers.)'


def passages(rows):
    if not isinstance(rows,list):raise ValueError('Passages must be a list')
    return '\n\n'.join(f'Passage {i+1}: {text(v)}' for i,v in enumerate(rows)) if rows else '(No passages supplied.)'


def response(stage,value):
    if stage not in STAGES:raise ValueError('Unknown stage')
    if stage not in FIELDS:return text(value)
    fields=FIELDS[stage]
    if not isinstance(value,dict) or tuple(value)!=fields:raise ValueError('Wrong output fields/order')
    return '\n'.join(f'{k}: {text(value[k])}' for k in fields)


def parse(stage,raw):
    """Parse reviewed structural contract only, never establish factual truth."""
    raw=text(raw).strip()
    if stage=='rewrite':
        if '\n' in raw or '\r' in raw or not raw.endswith(('?','？')):raise ValueError('Expected one retrieval question')
        return raw
    if stage=='final':
        # A valid book/work name can itself contain a colon or a field-like word.
        # Do not silently extract, relabel or reject an answer by that heuristic.
        return raw
    if stage not in FIELDS:raise ValueError('Unknown stage')
    names=FIELDS[stage]
    matches=list(re.finditer(r'(?m)^(Support|Answer|Facts|Link|Candidate):[ \t]*',raw))
    if tuple(m.group(1) for m in matches)!=names or matches[0].start()!=0:
        raise ValueError('Expected exactly the declared fields, without preamble')
    result={}
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(raw)
        result[m.group(1)]=text(raw[m.end():end].strip())
    return result


def completed_response(stage,raw,finish_reason):
    """A structural acceptance guard, not a provider client or retry mechanism."""
    if finish_reason!='stop':raise ValueError('Only a completed stop response is accepted')
    return {'raw':raw,'parsed':parse(stage,raw)}


def values(stage,data):
    if set(data)!=set(INPUTS[stage]):raise ValueError('Input shape differs from reviewed contract')
    out={}
    for k,v in data.items():
        out[k]=history(v) if k=='history' else passages(v) if k=='passages' else response('goal',v) if k=='synthesis' else text(v)
    return out


def render(revision,stage,data):
    folder=HERE/'revisions'/revision
    if revision not in ('r01','r02','r03'):raise ValueError('Unknown design revision')
    template=(folder/(stage+'.txt')).read_text(encoding='utf8')
    fixtures=json.loads((folder/'examples.json').read_text(encoding='utf8'))['stages'][stage]
    prefix,actual=template.split('\nActual task:\n',1)
    examples=[]
    for n,item in enumerate(fixtures,1):
        inp=values(stage,{k:item[k] for k in INPUTS[stage]})
        examples.append('Example '+str(n)+' (fictional):\n'+actual.format(**inp)+response(stage,item['output']))
    return prefix.format(examples='\n\n'.join(examples))+'\nActual task:\n'+actual.format(**values(stage,data))


def accepted_history_row(step,current,raw):
    answer=parse('ordinary',raw)['Answer']
    return dict(step_id=step,question=text(current),answer=answer)


def goal_input_after_ordinary(original,current,retrieval,ordinary_raw,previous,goal_passages):
    """Synthetic handoff helper; no mutable pipeline state is accessed."""
    copied=[dict(x) for x in previous]
    sid=max([x['step_id'] for x in copied]+[0])+1
    copied.append(accepted_history_row(sid,current,ordinary_raw))
    return dict(original_question=original,history=copied,retrieval_question=retrieval,passages=list(goal_passages))


def build_review_document(revision):
    folder=HERE/'revisions'/revision
    fixtures=json.loads((folder/'examples.json').read_text(encoding='utf8'))
    lines=['# '+revision+'：完整英文指令、全部示例与运行时字段','',
           '人工编写的虚构示例；不是模型生成或评测结果。示例所用数据与真实任务严格分开。','']
    stats={}
    for stage in STAGES:
        data={k:[] if k in ('history','passages') else {'Facts':'{goal_facts}','Link':'{goal_link}','Candidate':'{goal_candidate}'} if k=='synthesis' else '{'+k+'}' for k in INPUTS[stage]}
        prompt=render(revision,stage,data)
        # Runtime placeholders replace the empty illustrative rendering in this review only.
        tail=prompt.rsplit('\nActual task:\n',1)
        tail[-1]=tail[-1].replace('(No previous answers.)','{history}').replace('(No passages supplied.)','{passages}')
        prompt='\nActual task:\n'.join(tail)
        lines+=['## '+stage,'','```text',prompt,'```','']
        stats[stage]=dict(examples=len(fixtures['stages'][stage]),characters=len(prompt),whitespace_words=len(prompt.split()),
                          actual_fields=list(INPUTS[stage]),not_model_token_count=True)
    (folder/'FULL_PROMPTS.en.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (folder/'PROMPT_SIZE.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return stats
