"""Read-only checkpoint audit and native-score report; no new generation."""
from collections import Counter
from contextlib import closing
import json
import numpy as np
from . import run as r


def analyze():
    r.check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers={d:metric(d) for d in r.base.DATASETS}
    refs={x['id']:x for x in r.rows(r.HERE/'REFERENCES90.local.frozen.jsonl')}
    inputs=r.rows(r.HERE/'INPUTS90.frozen.jsonl');byid={x['id']:x for x in inputs}
    cfg=r.read(r.HERE/'CONFIG.frozen.json');ex=r.read(r.HERE/'FINAL_EXAMPLES.frozen.json')
    with closing(r.ro(r.OUT/'checkpoint.sqlite3')) as db:
        actual=[json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        requests={rid:dict(status=status,payload=json.loads(p),result=json.loads(v) if v else {})
                  for rid,p,status,v in db.execute('SELECT id,payload,status,result FROM requests')}
    audit_count=0
    for x in actual:
        q=byid[x['id']]
        for stage in x['stages']:
            req=requests[stage['request_id']];name=stage['stage'].rsplit('/',1)[1]
            if name=='goal':prompt=r.templates.goal_prompt(q,x['arm']);cap=512
            elif name=='formatter':prompt=r.base.formatter(q['question'],x['goal_raw'],q['goal_reader_packet']);cap=128
            elif name=='final':prompt=r.base.final_prompt(q,x['goal_text'],x['arm'],cfg,ex);cap=128
            else:raise AssertionError(name)
            assert req['payload']==r.base.build_payload(cfg['generation'],prompt,cap),(x['key'],name)
            assert req['result'].get('usage')==stage['usage']
            if x['status']=='success':
                assert r.complete_receipt(req['result']) and req['status']=='success'
                assert req['result']['returned_model']==cfg['generation']['model']
            audit_count+=1
    parent=r.rows(r.HERE/'PARENT_OUTCOMES.frozen.jsonl')
    parents={(x['arm'],x['id']):x for x in parent}
    for x in actual:
        if x['arm']=='C':
            for k in ('goal_raw','goal_text','prediction'):
                assert x[k]==parents[('C',x['id'])][k]
    # G is an explicitly historical r03 reference, not an r04 arm or new receipt.
    details=actual+[dict(x,arm='G_r03',historical_reference=True) for x in parent if x['arm']=='G']
    for x in details:
        x['metrics']=scorers[x['dataset']](x['prediction'],refs[x['id']]) if x['status'] in ('success','algorithm_failure') else {'f1':None,'em':None}
        us=[s['usage'] for s in x['stages'] if s.get('usage')]
        x['input_tokens']=sum(u['prompt_tokens'] for u in us);x['output_tokens']=sum(u['completion_tokens'] for u in us)
        x['logical_CNY']=sum(r.cost(u) for u in us)
    index={(x['arm'],x['id']):x for x in details};arms=('C','G_r03','S','T');summary={};paired={}
    for ds in ('all',)+r.base.DATASETS:
        ids=[x['id'] for x in inputs if ds=='all' or x['dataset']==ds];summary[ds]={};paired[ds]={}
        for arm in arms:
            vals=[index[(arm,i)] for i in ids if (arm,i) in index]
            complete=len(vals)==len(ids) and all(x['metrics']['f1'] is not None for x in vals)
            summary[ds][arm]=dict(planned=len(ids),attempted=len(vals),statuses=dict(Counter(x['status'] for x in vals)),
                F1=sum(x['metrics']['f1'] for x in vals)/len(ids) if complete else None,
                EM=sum(x['metrics']['em'] for x in vals)/len(ids) if complete else None,
                mean_input_tokens=sum(x['input_tokens'] for x in vals)/len(vals) if vals else None,
                mean_output_tokens=sum(x['output_tokens'] for x in vals)/len(vals) if vals else None,
                mean_logical_CNY=sum(x['logical_CNY'] for x in vals)/len(vals) if vals else None,
                mean_goal_words=sum(len(x.get('goal_text','').split()) for x in vals)/len(vals) if vals else None)
        for a,b in [('C','S'),('C','T'),('S','T'),('G_r03','S'),('G_r03','T')]:
            if any((arm,i) not in index or index[(arm,i)]['metrics']['f1'] is None for i in ids for arm in (a,b)):
                continue
            diff=[index[(b,i)]['metrics']['f1']-index[(a,i)]['metrics']['f1'] for i in ids]
            rng=np.random.default_rng(20261023);accum=np.zeros(10000)
            for dataset in (r.base.DATASETS if ds=='all' else (ds,)):
                values=np.array([index[(b,i)]['metrics']['f1']-index[(a,i)]['metrics']['f1'] for i in ids if byid[i]['dataset']==dataset])
                accum+=values[rng.integers(0,len(values),size=(10000,len(values)))].sum(axis=1)
            paired[ds][a+'->'+b]=dict(n=len(ids),F1_delta=sum(diff)/len(ids),improved=sum(v>1e-12 for v in diff),
                harmed=sum(v< -1e-12 for v in diff),tied=sum(abs(v)<=1e-12 for v in diff),
                descriptive95_CI=np.quantile(accum/len(ids),[.025,.975]).tolist())
    namespace='p02_prompt_research_v1_20261022/'+str(r.OUT.resolve())
    with closing(r.ro(cfg['budget']['ledger'])) as db:
        paid,unknown=db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(namespace,)).fetchone()
        billed={rid:actual_cost for rid,actual_cost in db.execute('SELECT id,actual FROM charges WHERE namespace=?',(namespace,))}
    assert all(rid in requests for rid in billed)
    # No retired requests are expected; if present, account explicitly before completing.
    with closing(r.ro(r.OUT/'checkpoint.sqlite3')) as db:
        retired=db.execute('SELECT count(*) FROM retired_requests').fetchone()[0]
    assert retired==0,'Review retired technical attempts and their charges explicitly'
    receipt_paid=sum(r.cost(requests[rid]['result']['usage']) for rid,v in billed.items() if v is not None and requests[rid]['result'].get('usage'))
    assert abs(paid-receipt_paid)<1e-9
    result=dict(utc=r.utc(),status='all270_attempted' if len(actual)==270 else 'partial',planned=270,attempted=len(actual),
        statuses=dict(Counter(x['status'] for x in actual)),by_dataset_arm=summary,paired=paired,actual_new_CNY=paid,unknown_reserved_CNY=unknown,
        receipt_audit=dict(exact_payloads_checked=audit_count,cached_C_outputs_match=True,actual_cost_reconciled=True,retired_requests=retired),
        scope='Fixed-reader exposed90; C strict previous replay; G_r03 previous reference; no fresh whole-pipeline or generalization claim',
        main_evaluation_resumed=False,production_changed=False)
    r.jsonl(r.OUT/'DETAILS.local.jsonl',details);r.save(r.OUT/'ANALYSIS.json',result)
    if len(actual)!=270:return result
    old_review=r.read(r.base.OUT/'CASE_REVIEW.local.json')['cases']
    changed=[];review=[]
    for q in inputs:
        allarms={a:index[(a,q['id'])] for a in arms}
        obj=dict(id=q['id'],question=q['question'],dataset=q['dataset'],reference=refs[q['id']],ordinary_nodes=q['ordinary_nodes'],
            goal_reader_packet=q['goal_reader_packet'],root_packet=q['final_root_packet'],
            previous_observation=old_review.get(q['id']),
            arms={a:{k:x.get(k) for k in ('status','goal_text','prediction','metrics')} for a,x in allarms.items()})
        if len({allarms[a].get('prediction') for a in ('C','S','T')})>1:changed.append(obj)
        if obj['previous_observation'] or len({allarms[a].get('prediction') for a in ('C','S','T')})>1:review.append(obj)
    r.jsonl(r.OUT/'CHANGED_FINAL_CASES.local.jsonl',changed)
    r.jsonl(r.OUT/'REVIEW_CASES.local.jsonl',review)
    lines=['# r04全部变化及旧错误复查','', '这是暴露开发题上的事后诊断；原生分数不修改。','']
    for obj in review:
        lines+=['## '+obj['id'],'',obj['question'],'']
        if obj['previous_observation']:lines+=['旧观察：'+obj['previous_observation']['category'],'']
        for a,x in obj['arms'].items():
            lines+=['### '+a,'',x.get('goal_text') or '(无完整goal)','',
                'Final: '+str(x.get('prediction')),'',str(x.get('metrics')),'']
    (r.OUT/'REVIEW_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    result['review_cases']=len(review);result['new_final_changed_cases']=len(changed)
    r.save(r.OUT/'ANALYSIS.json',result);r.save(r.REVIEW/'RESULTS.json',result)
    return result


if __name__=='__main__':
    result=analyze()
    print(json.dumps({k:result[k] for k in ('status','attempted','statuses','actual_new_CNY','unknown_reserved_CNY','receipt_audit')},ensure_ascii=False))
