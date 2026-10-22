"""Improve challenge readability while preserving IDs, numbers and earlier draft."""
from datetime import datetime,timezone
import json
import re
from design_tools import HERE


def polish(value):
    if isinstance(value,list):return [polish(x) for x in value]
    if isinstance(value,dict):return {k:v if k in ('id','stage') else polish(v) for k,v in value.items()}
    if not isinstance(value,str):return value
    value=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',value)
    value=re.sub(r',(?=\d)',', ',value)
    return value.replace('orchoosea convenient source','or choose a convenient source')


def main():
    path=HERE/'CHALLENGES.local.json'
    original=path.read_text(encoding='utf8')
    snapshot=HERE/'reviews'/'CHALLENGES.first_draft.json'
    assert not snapshot.exists(), 'Preserve prior snapshot'
    snapshot.write_text(original,encoding='utf8')
    bank=json.loads(original)
    updated=polish(bank)
    assert [x['id'] for x in bank['cases']]==[x['id'] for x in updated['cases']]
    for a,b in zip(bank['cases'],updated['cases']):
        # This is textual spacing only, never a change to numerical evidence.
        assert re.findall(r'\d+',json.dumps(a))==re.findall(r'\d+',json.dumps(b))
    # Later semantic review: two dated posts alone do not prove the first ever.
    # Preserve the original weak example in the first-draft snapshot.
    case=next(x for x in updated['cases'] if x['id']=='C37_new_post_not_required')
    case['input']['passages']=["Pera's first public post was mayor of Quen in 1980. Pera later became ambassador to Ruen in 1990."]
    path.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 合成边界检查清单','',
      '这些输入和预期由本次设计人工编写，用于逐条审查提示词与接口。不是模型通过率、独立测试集或最终F1。预期与禁止推断字段不得进入真实任务prompt。初稿保留于reviews/CHALLENGES.first_draft.json。除数字空格整理外，C37已补足第一份公职的明确证据，详见第五轮复核。','',
      '| ID | 阶段 | 应有行为 | 必须避免 |','|---|---|---|---|']
    for x in updated['cases']:
        lines.append('| '+' | '.join(x[k] for k in ('id','stage','expected_behavior','forbidden_inference'))+' |')
    (HERE/'CHALLENGE_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:
        h.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='challenge_readability_review',
          cases=40,numbers_and_ids_preserved=True,model_executions=0))+'\n')
    print('Readability updated;40 IDs and numerical evidence preserved.')


if __name__=='__main__':main()
