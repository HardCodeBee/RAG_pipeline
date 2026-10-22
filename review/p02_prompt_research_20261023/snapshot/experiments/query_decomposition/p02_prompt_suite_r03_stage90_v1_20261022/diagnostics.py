"""Local-only structural, historical-reference and case-coverage summaries."""
from collections import Counter
from contextlib import closing
from statistics import mean
from . import run
import json


def describe(values):
    return dict(n=len(values),mean=mean(values) if values else None,
                min=min(values) if values else None,max=max(values) if values else None)


def main():
    details=run.rows(run.OUT/'DETAILS.local.jsonl')
    goal=[r for r in details if r['suite']=='goal' and r['status']=='success']
    ordinary=[r for r in details if r['suite']=='ordinary' and r['status']=='success']
    groups={}
    for arm in run.GOAL_ARMS:
        values=[r for r in goal if r['arm']==arm]
        groups[arm]=dict(n=len(values),goal_words=describe([len(r['goal_text'].split()) for r in values]),
          final_words=describe([len(r['prediction'].split()) for r in values]),
          final_exact_abstentions=sum(r['prediction'].strip().lower()=='cannot be determined' for r in values),
          over140_soft_goal_words=sum(len(r['goal_text'].split())>140 for r in values))
        if arm!='C':
            for r in values:
                assert list(run.design.parse('goal',r['goal_text']))==['Facts','Link','Candidate']
            groups[arm]['fields']='Facts/Link/Candidate parsed'
    gi={(r['id'],r['arm']):r for r in goal}
    pairs=[(gi[(r['id'],'G')],gi[(r['id'],'GF')]) for r in run.rows(run.HERE/'INPUTS90.frozen.jsonl') if all((r['id'],a) in gi for a in ('G','GF'))]
    same_final=sum(a['prediction']==b['prediction'] for a,b in pairs)
    ordinary_stats={}
    for arm in run.ORDINARY_ARMS:
        values=[r for r in ordinary if r['arm']==arm]
        ordinary_stats[arm]=dict(n=len(values),answer_words=describe([len(r['answer'].split()) for r in values]),
          formatters=sum(any(s['stage'].endswith('/formatter') for s in r['stages']) for r in values))
        if arm=='O1':ordinary_stats[arm]['support_over50_soft_words']=sum(len(r['support'].split())>50 for r in values)
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        receipts=[(s,json.loads(r) if r else {}) for s,r in db.execute('SELECT status,result FROM requests')]
        retired=db.execute('SELECT count(*) FROM retired_requests').fetchone()[0]
    success=[r for s,r in receipts if s=='success']
    unique_usage=[r['usage'] for r in success if r.get('usage')]
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers={ds:metric(ds) for ds in run.DATASETS}
    refs={r['id']:r for r in run.rows(run.HERE/'REFERENCES90.local.frozen.jsonl')}
    historical=[]
    for r in run.rows(run.HERE/'HISTORY90.local.frozen.jsonl'):
        historical.append(dict(**r,metrics=scorers[r['dataset']](r['prediction'],refs[r['id']])))
    hs={}
    for ds in ('all',)+run.DATASETS:
        values=[r for r in historical if ds=='all' or r['dataset']==ds]
        hs[ds]=dict(n=len(values),F1=mean(r['metrics']['f1'] for r in values),EM=mean(r['metrics']['em'] for r in values))
    cases=run.rows(run.OUT/'CHANGED_FINAL_CASES.local.jsonl')
    reviews=run.read(run.OUT/'CASE_REVIEW.local.json')['cases']
    missing=[r['id'] for r in cases if r['id'] not in reviews]
    categories={}
    for r in cases:
        if r['id'] not in reviews:continue
        category=reviews[r['id']]['category'];x=categories.setdefault(category,dict(n=0,G_minus_C_F1_sum=0.0,GF_minus_C_F1_sum=0.0,ids=[]))
        x['n']+=1;x['ids'].append(r['id'])
        for arm in ('G','GF'):x[arm+'_minus_C_F1_sum']+=r['arms'][arm]['metrics']['f1']-r['arms']['C']['metrics']['f1']
    result=dict(utc=run.utc(),terminal=len(details),complete=len(details)==622,goal=groups,ordinary=ordinary_stats,
      G_GF=dict(pairs=len(pairs),identical_final_text=same_final,different_final_text=len(pairs)-same_final),
      requests=dict(statuses=dict(Counter(s for s,r in receipts)),finish_reasons=dict(Counter(r.get('finish_reason') for r in success)),
        retired=retired,unique_prompt_tokens=sum(u['prompt_tokens'] for u in unique_usage),
        unique_completion_tokens=sum(u['completion_tokens'] for u in unique_usage),unique_receipt_cost_CNY=sum(run.cost(u) for u in unique_usage)),
      historical_role_headers_reference=hs,historical_note='Historical source outputs only; different original caps; not original P02 full-flow baseline or a primary comparator.',
      changed_case_review=dict(changed=len(cases),reviewed=len(cases)-len(missing),missing=missing,categories=categories,
        note='Posthoc descriptive classification. F1 sums retain native scores; category attribution does not establish a single causal mechanism.'),
      main_evaluation_paused=True)
    run.atomic_json(run.OUT/'DIAGNOSTICS.json',result)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
