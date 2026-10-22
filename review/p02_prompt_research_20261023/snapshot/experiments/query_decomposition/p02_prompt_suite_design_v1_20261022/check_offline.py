"""Meaningful interface checks only. No model calls; no semantic pass-rate claim."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
from design_tools import (HERE, INPUTS, STAGES, accepted_history_row, completed_response,
                          goal_input_after_ordinary, history, parse, render, response)

checks=[]


def check(name,fn):
    try:
        fn()
        checks.append({'name':name,'passed':True})
    except Exception as exc:
        checks.append({'name':name,'passed':False,'detail':str(exc)})


def equal(actual,expected):
    assert actual==expected, repr((actual,expected))


def rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected explicit structural rejection')


def contains(large,small):
    assert small in large, repr(small)


def excludes(large,small):
    assert small not in large, repr(small)


def main():
    revision='r03'
    bank=json.loads((HERE/'revisions'/revision/'examples.json').read_text(encoding='utf8'))
    for stage in STAGES:
        for item in bank['stages'][stage]:
            raw=response(stage,item['output'])
            check('authored_output_roundtrip/'+stage+'/'+item['id'],
                  lambda stage=stage,raw=raw,item=item:equal(parse(stage,raw),item['output']))
    challenge=json.loads((HERE/'CHALLENGES.local.json').read_text(encoding='utf8'))['cases']
    check('challenge_ids_unique',lambda:equal(len({x['id'] for x in challenge}),len(challenge)))
    for case in challenge:
        before=deepcopy(case)
        prompt=render(revision,case['stage'],case['input'])
        tail=prompt.rsplit('\nActual task:\n',1)[1]
        key='current_question' if case['stage']=='rewrite' else 'original_question'
        check('runtime_question_preserved/'+case['id'],lambda tail=tail,case=case,key=key:contains(tail,case['input'][key]))
        check('challenge_not_mutated/'+case['id'],lambda case=case,before=before:equal(case,before))
        augmented=deepcopy(case['input']);augmented['expected_behavior']='FORBIDDEN_GOLD_SENTINEL'
        check('runtime_gold_field_rejected/'+case['id'],lambda case=case,augmented=augmented:rejected(lambda:render(revision,case['stage'],augmented)))
    malformed=[
      ('ordinary','Answer: a'),('ordinary','Support: a\nAnswer:'),
      ('ordinary','Support: a\nAnswer: b\nAnswer: c'),
      ('ordinary','Preamble\nSupport: a\nAnswer: b'),
      ('ordinary','Answer: b\nSupport: a'),('ordinary','Facts: a\nAnswer: b'),
      ('ordinary','```text\nSupport: a\nAnswer: b\n```'),
      ('goal','Facts: a\nCandidate: c'),('goal','Facts: a\nLink: b\nCandidate:'),
      ('goal','Facts: a\nLink: b\nCandidate: c\nLink: d'),
      ('rewrite','Who is A?\nWho is B?'),('rewrite','The answer is A.'),('final','')]
    for i,(stage,raw) in enumerate(malformed):
        check('reject_malformed/'+str(i),lambda stage=stage,raw=raw:rejected(lambda:parse(stage,raw)))
    for reason in ('length',None,'content_filter','tool_calls'):
        check('reject_nonstop/'+str(reason),lambda reason=reason:rejected(lambda:completed_response('ordinary','Support: a\nAnswer: b',reason)))
    raw='  Support: Supplied premise.\r\nAnswer: Kept result.  '
    check('raw_response_bytes_preserved',lambda:equal(completed_response('ordinary',raw,'stop')['raw'],raw))
    check('crlf_fields',lambda:equal(parse('ordinary',raw),{'Support':'Supplied premise.','Answer':'Kept result.'}))
    check('final_literal_title_preserved',lambda:equal(parse('final','Answer: A History'),'Answer: A History'))
    check('final_explanatory_text_not_silently_cleaned',lambda:equal(parse('final','Fenwick because Orin was born there.'),'Fenwick because Orin was born there.'))
    previous=[dict(step_id=1,question='Who is A?',answer='A is B.')]
    previous_before=deepcopy(previous)
    handed=goal_input_after_ordinary('Root question?','Current question?','Search question?',
        'Support: PRIVATE_SUPPORT_SENTINEL\nAnswer: PUBLIC_FACT_SENTINEL',previous,['GOAL_PACKET_SENTINEL'])
    check('only_answer_propagates',lambda:excludes(json.dumps(handed),'PRIVATE_SUPPORT_SENTINEL'))
    check('answer_survives_handoff',lambda:contains(json.dumps(handed),'PUBLIC_FACT_SENTINEL'))
    check('history_original_not_mutated',lambda:equal(previous,previous_before))
    check('history_step_order',lambda:equal([x['step_id'] for x in handed['history']],[1,2]))
    check('raw_current_survives_history',lambda:equal(handed['history'][-1]['question'],'Current question?'))
    check('retrieval_distinct_from_current',lambda:equal(handed['retrieval_question'],'Search question?'))
    for invalid in ([dict(step_id=1,question='q',answer='a'),dict(step_id=1,question='q2',answer='a2')],
                    [dict(step_id=2,question='q',answer='a'),dict(step_id=1,question='q2',answer='a2')],
                    [dict(step_id=True,question='q',answer='a')],
                    [dict(step_id=1,question='q',answer='a',gold='leak')]):
        check('reject_invalid_history/'+str(len(checks)),lambda invalid=invalid:rejected(lambda:history(invalid)))
    final_input=dict(original_question='Original ROOT?',history=handed['history'],
        synthesis={'Facts':'FACT_SENTINEL','Link':'LINK_SENTINEL','Candidate':'CANDIDATE_SENTINEL'},
        passages=['ROOT_PACKET_SENTINEL'])
    final_tail=render(revision,'final',final_input).rsplit('\nActual task:\n',1)[1]
    for marker in ('FACT_SENTINEL','LINK_SENTINEL','CANDIDATE_SENTINEL','ROOT_PACKET_SENTINEL','PUBLIC_FACT_SENTINEL'):
        check('final_input_retains/'+marker,lambda marker=marker:contains(final_tail,marker))
    for marker in ('GOAL_PACKET_SENTINEL','PRIVATE_SUPPORT_SENTINEL'):
        check('final_input_excludes/'+marker,lambda marker=marker:excludes(final_tail,marker))
    # Metadata inside demonstration fixtures must not become model-visible input.
    example=next(x for x in bank['stages']['final'] if x['id']=='multiple_posts_bad_candidate')
    rendered=render(revision,'final',final_input)
    check('review_only_goal_reader_not_rendered',lambda:excludes(rendered,example['goal_reader_provenance_for_review_only'][0]))
    check('perturbation_metadata_not_rendered',lambda:excludes(rendered,example['perturbation']))
    # Independent primary sources, controls and holds are inspected, never modified.
    session=json.loads((HERE/'SESSION.json').read_text(encoding='utf8'))
    root=HERE.parents[2]
    protected=[]
    for rel,wanted in session['source_core_sha256'].items():
        actual=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        protected.append({'path':rel,'unchanged':actual==wanted})
        check('protected_file_unchanged/'+rel,lambda actual=actual,wanted=wanted:equal(actual,wanted))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),revision=revision,
        model_executions=0,meaning='Structural and interface checks, NOT semantic/model success rates',
        authored_examples=sum(len(v) for v in bank['stages'].values()),authored_challenges=len(challenge),
        passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks),
        checks=checks,protected_files=protected)
    (HERE/'OFFLINE_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:
        h.write(json.dumps({k:v for k,v in result.items() if k not in ('checks','protected_files')},ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','protected_files')},ensure_ascii=False))
    if result['failed']:
        print(json.dumps([x for x in checks if not x['passed']],ensure_ascii=False))
        sys.exit(1)


if __name__=='__main__':main()
