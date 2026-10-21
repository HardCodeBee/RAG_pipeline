"""Offline reporting fix: an interrupted planned call need not have been sent."""
from collections import Counter,defaultdict
import json
import sqlite3
from pathlib import Path
from .common import DATA,OUTPUT,read,rows,save,sha
from .scoring import metric

def evaluate(dataset,directory,references):
    refs=rows(references);preds={};evidence=defaultdict(set)
    dbpath=Path(directory)/'checkpoint.sqlite3'
    with sqlite3.connect('file:'+str(dbpath.resolve())+'?mode=ro',uri=True) as db:
        for key,val in db.execute("SELECT key,value FROM data WHERE kind='prediction'"):preds[key]=json.loads(val)
        for (val,) in db.execute("SELECT value FROM data WHERE kind='evidence'"):
            r=json.loads(val);evidence[r['qid']].update(r['doc_ids'])
        requests={r[0]:{'status':r[1],'result':json.loads(r[2]) if r[2] else None} for r in db.execute('SELECT id,status,result FROM requests')}
        namespace=str(Path(directory).resolve().relative_to(OUTPUT.resolve()))
    score=metric(dataset);details=[]
    for ref in refs:
        p=preds.get(ref['qid'],{'prediction':None,'status':'unstarted'})
        valid=isinstance(p.get('prediction'),str) and p['status'] in {'success','algorithm_failure'}
        s=score(p['prediction'],ref) if valid else {'em':None,'f1':None}
        gold=set(ref['positive_doc_ids']);seen=evidence[ref['qid']]
        details.append({'qid':ref['qid'],'type':ref['type'],'group_id':ref['group_id'],'answer':ref['answer'],
          'prediction':p.get('prediction'),'status':p['status'],**s,'gold_documents':len(gold),
          'evidence_doc_recall':len(gold&seen)/len(gold) if gold else None,
          'full_gold_document_coverage':gold.issubset(seen) if gold else None,
          'final_doc_ids':p.get('final_doc_ids',[]),'fallback':p.get('fallback')})
    def summarize(data):
        valid=[r for r in data if r['f1'] is not None];n=len(data)
        total=sum(r['f1'] for r in valid);missing=n-len(valid)
        return {'planned':n,'terminal_records':sum(r['status']!='unstarted' for r in data),'available':len(valid),'missing':missing,
            'f1_available':total/len(valid) if valid else None,'em_available':sum(r['em'] for r in valid)/len(valid) if valid else None,
            'f1_all_planned_bounds':[total/n,(total+missing)/n],
            'statuses':dict(Counter(r['status'] for r in data))}
    calls={}
    trace=Path(directory)/'trace.jsonl'
    if trace.exists():
        for r in rows(trace):
            if r['event']=='logical_api_call':calls.setdefault((r.get('qid'),r['stage'],r['request_id']),r)
    logical_cost=0;unknown_calls=0;unsubmitted_calls=0;prompt_tokens=completion_tokens=0
    for key,r in calls.items():
        entry=requests.get(r['request_id'])
        if entry is None:
            unsubmitted_calls+=1
            continue
        response=entry['result']
        if response is None or response.get('status')!='success':unknown_calls+=1;continue
        u=response.get('usage',{});p=u.get('prompt_tokens',0);c=u.get('completion_tokens',0)
        logical_cost+=p*.8e-6+c*2e-6;prompt_tokens+=p;completion_tokens+=c
    with sqlite3.connect('file:'+str(OUTPUT/'budget.sqlite3')+'?mode=ro',uri=True) as db:
        actual,reserved=db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(namespace,)).fetchone()
    report={'dataset':dataset,'reference_sha256':sha(references),'overall':summarize(details),
       'by_type':{kind:summarize([r for r in details if r['type']==kind]) for kind in sorted({r['type'] for r in details})},
       'returned_usage_cny':actual,'reserved_unreconciled_cny':reserved,'logical_calls':len(calls)-unsubmitted_calls,'unsubmitted_trace_calls':unsubmitted_calls,
       'logical_cold_cost_cny':logical_cost,'logical_cost_missing_calls':unknown_calls,
       'logical_tokens':{'prompt':prompt_tokens,'completion':completion_tokens},
       'metric':'native official Answer EM/F1; 2Wiki v1.1 entity aliases/demonyms; MuSiQue Ans answer_aliases',
       'support_metric':'gold document coverage of passages actually shown; not official sentence-support or joint F1',
       'hallucination_rate_evaluated':False}
    save(Path(directory)/'scores.json',report)
    (Path(directory)/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf8')
    return report


