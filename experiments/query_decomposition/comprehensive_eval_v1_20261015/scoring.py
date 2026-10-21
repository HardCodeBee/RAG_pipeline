"""Offline-only native answer metrics. Never imported by method controllers."""
import ast
from collections import Counter,defaultdict
import json
import sqlite3
from pathlib import Path
from .common import DATA, OUTPUT, read, rows, save, sha

def official_functions(path,names):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    selected=[n for n in tree.body if isinstance(n,ast.Import) and all(a.name in {'re','string','collections','itertools','sys','os','pickle','ujson'} for a in n.names)
              or isinstance(n,ast.ImportFrom) and n.module=='collections'
              or isinstance(n,ast.FunctionDef) and n.name in names]
    found={n.name for n in selected if isinstance(n,ast.FunctionDef)}
    if not set(names).issubset(found):raise ValueError('Official metric function missing')
    env={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
    return env

def metric(dataset):
    if dataset=='hotpotqa':
        from experiments.query_decomposition.hotpot_formal_eval_v1.common.scoring import f1_score,exact_match_score
        return lambda pred,ref:{'em':float(exact_match_score(pred,ref['answer'])),'f1':f1_score(pred,ref['answer'])[0]}
    if dataset=='2wiki':
        path=DATA/dataset/'official/2wikimultihop_evaluate_v1.1.py'
        env=official_functions(path,['normalize_answer','f1_score','exact_match_score','eval_answer','update_answer'])
        aliases={r['Q_id']:r['aliases']+r['demonyms'] for r in rows(DATA/dataset/'native/id_aliases.json')}
        def score(pred,ref):
            answers=[ref['answer']]+aliases.get(ref['answer_id'],[])
            m={'em':0,'f1':0,'prec':0,'recall':0};env['update_answer'](m,pred,answers)
            return {'em':m['em'],'f1':m['f1']}
        return score
    path=DATA/'musique/official/metrics/answer.py'
    env=official_functions(path,['normalize_answer','get_tokens','compute_exact','compute_f1','metric_max_over_ground_truths'])
    def score(pred,ref):
        answers=[ref['answer']]+ref['aliases']
        return {'em':max(env['compute_exact'](a,pred) for a in answers),'f1':max(env['compute_f1'](a,pred) for a in answers)}
    return score

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
    logical_cost=0;unknown_calls=0;prompt_tokens=completion_tokens=0
    for key,r in calls.items():
        response=requests[r['request_id']]['result']
        if response is None or response.get('status')!='success':unknown_calls+=1;continue
        u=response.get('usage',{});p=u.get('prompt_tokens',0);c=u.get('completion_tokens',0)
        logical_cost+=p*.8e-6+c*2e-6;prompt_tokens+=p;completion_tokens+=c
    with sqlite3.connect('file:'+str(OUTPUT/'budget.sqlite3')+'?mode=ro',uri=True) as db:
        actual,reserved=db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(namespace,)).fetchone()
    report={'dataset':dataset,'reference_sha256':sha(references),'overall':summarize(details),
       'by_type':{kind:summarize([r for r in details if r['type']==kind]) for kind in sorted({r['type'] for r in details})},
       'returned_usage_cny':actual,'reserved_unreconciled_cny':reserved,'logical_calls':len(calls),
       'logical_cold_cost_cny':logical_cost,'logical_cost_missing_calls':unknown_calls,
       'logical_tokens':{'prompt':prompt_tokens,'completion':completion_tokens},
       'metric':'native official Answer EM/F1; 2Wiki v1.1 entity aliases/demonyms; MuSiQue Ans answer_aliases',
       'support_metric':'gold document coverage of passages actually shown; not official sentence-support or joint F1',
       'hallucination_rate_evaluated':False}
    save(Path(directory)/'scores.json',report)
    (Path(directory)/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf8')
    return report
