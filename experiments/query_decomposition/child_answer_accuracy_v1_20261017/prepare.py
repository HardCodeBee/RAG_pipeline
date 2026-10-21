"""Freeze unseen parent IDs and already-retrieved node inputs, never answers."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT/HERE.name
CAMPAIGN = OUTPUT/'comprehensive_v1_20261015/campaign_api_v1'
INPUTS = ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def excluded():
    ids=set()
    for d in (ROOT/'experiments/query_decomposition').glob('multigrain*'):
        for f in d.glob('cases*.jsonl'):
            ids.update(r.get('qid',r.get('id')) for r in rows(f))
    for f in HERE.glob('round*/cases.inputs.jsonl'):
        ids.update(r['qid'] for r in rows(f) if r['group']=='fresh')
    for f in (HERE/'child_examples.frozen.json',):
        ids.update(r.get('source_qid') for r in load(f))
    return ids

def cached_pool(ds):
    questions={r['qid']:r['question'] for r in rows(INPUTS/ds/'formal.questions.jsonl')}
    if ds in ('hotpotqa','2wiki'):
        path=CAMPAIGN/ds/'formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3'
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            # Only text of current requests, not historical answers, enters sampling.
            decomp={k:load_json['qdmr'] for k,raw in db.execute("SELECT key,value FROM data WHERE kind='decomposition'") if (load_json:=json.loads(raw))}
            for key,raw in db.execute("SELECT key,value FROM data WHERE kind='contextual_node_input'"):
                v=json.loads(raw);qid=v['qid'];sid=v['step_id']
                if v.get('role')!='ordinary' or qid not in questions:
                    continue
                request=next(n['request'] for n in decomp[qid]['steps'] if n['id']==sid)
                if re.search(r'#\d+',request) or not request.rstrip().endswith('?'):
                    continue
                yield {'dataset':ds,'qid':qid,'step_id':sid,'original':questions[qid],
                    'question':request,'history':v['dependencies'],'packet':v['packet'],
                    'source':str(path.relative_to(ROOT)),'source_key':key}
    else:
        path=CAMPAIGN/ds/'formal/b02_selfask_dense/k10_s3/checkpoint.sqlite3'
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            retrieval={}
            for key,raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
                p=json.loads(raw);retrieval[(p['qid'],p['query'])]=p
            for key,raw in db.execute("SELECT key,value FROM data WHERE kind='continuation'"):
                if not key.endswith('/1'):
                    continue
                qid=key.rsplit('/',1)[0];v=json.loads(raw);action=v['next_action']
                if qid not in questions or action[0]!='follow':
                    continue
                request=action[1];packet=retrieval.get((qid,request))
                if not packet or not request.rstrip().endswith('?') or re.search(r'#\d+',request):
                    continue
                yield {'dataset':ds,'qid':qid,'step_id':1,'original':questions[qid],
                    'question':request,'history':[],'packet':packet,
                    'source':str(path.relative_to(ROOT)),'source_key':key}

def known_cases():
    specs=[
        ('multigrain_partial_bind_v5_20261017','2wiki','support_new','bc7073a80baf11ebab90acde48001122',1),
        ('multigrain_partial_bind_v5_20261017','2wiki','support_new','bc7073a80baf11ebab90acde48001122',2),
        ('multigrain_dual_retrieval_pilot_v1_20261017','hotpotqa','dual_original_rewritten','5a83431d5542993344745ff5',1),
        ('multigrain_evidence_v4r1_20261017','hotpotqa','partial_dual','5ab67ce2554299710c8d1ead',1),
        ('multigrain_partial_bind_v5_20261017','2wiki','support_new','32f0334c0bb011ebab90acde48001122',2),
        ('multigrain_partial_bind_v5_20261017','musique','support_new','2hop__805610_392181',2),
    ]
    for run,ds,condition,qid,sid in specs:
        path=OUTPUT/run/'cases'/ds/condition/(qid+'.json');r=load(path)
        n=next(n for n in r['nodes'] if n['step_id']==sid)
        yield {'dataset':ds,'qid':qid,'step_id':sid,'original':r['question'],
            'question':n['request'],'history':n['prior_complete_answers'],
            'packet':n['reader_packet'],'source':str(path.relative_to(ROOT)),
            'source_key':'nodes/'+str(sid),'group':'known_regression'}

def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--round',default='round1');a.add_argument('--per-dataset',type=int,default=6);a.add_argument('--include-prior-errors',action='store_true')
    args=a.parse_args();directory=HERE/args.round
    if (directory/'cases.inputs.jsonl').exists():
        raise RuntimeError('Inputs already frozen; use a new round name')
    blocked=excluded();selected=[]
    for ds in ('hotpotqa','2wiki','musique'):
        byid={}
        for r in cached_pool(ds):
            if r['qid'] not in blocked and r['packet']['docs']:
                # Earliest self-contained ordinary node, deterministic and score-blind.
                if r['qid'] not in byid or r['step_id']<byid[r['qid']]['step_id']:
                    byid[r['qid']]=r
        order=sorted(byid,key=lambda q:hashlib.sha256(('child-accuracy-'+args.round+'|'+ds+'|'+q).encode()).hexdigest())
        chosen=[dict(byid[q],group='fresh') for q in order[:args.per_dataset]]
        assert len(chosen)==args.per_dataset
        selected+=chosen
    selected+=list(known_cases())
    seen={(c['dataset'],c['qid'],c['step_id']) for c in selected}
    # Explicit user requirement: new rounds retain every preceding regression.
    # Such cases are never counted as fresh or used for score-based output choice.
    for previous in sorted(OUT.glob('round*/RESULTS.json')):
        old_inputs=HERE/previous.parent.name/'cases.inputs.jsonl'
        if not old_inputs.exists():continue
        source={c['case_id']:c for c in rows(old_inputs)}
        result=load(previous)
        lost={cid for value in result['overall']['vs_baseline'].values()
              for cid in value['losses']+value.get('planned_loss_ids',[])}
        if args.include_prior_errors:
            lost.update(r['case_id'] for r in result['records'] if r['condition']=='baseline' and r['correct'] is False)
        for cid in sorted(lost):
            c=source[cid];key=(c['dataset'],c['qid'],c['step_id'])
            if key not in seen:
                selected.append(dict(c,group='known_regression',regression_origin=previous.parent.name));seen.add(key)
    for i,r in enumerate(selected):r['case_id']=f'{i+1:02d}_{r["dataset"]}_{r["qid"]}_{r["step_id"]}'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'cases.inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    write(directory/'SELECTION.json',{'round':args.round,'fresh_per_dataset':args.per_dataset,'fresh_parent_ids_disjoint_from_prior_multigrain':True,
        'excluded_ids_count':len(blocked),'seed':'child-accuracy-'+args.round,'known_regression_nodes':len(selected)-3*args.per_dataset,
        'source_difference':'Hotpot/2Wiki existing S02 ordinary nodes; Mu existing first SelfAsk follow-up and its Top10. These are fixed-reader diagnostics, not a unified retriever comparison.',
        'history':'actual frozen history; no correction using gold; candidate results never feed another node',
        'inputs_sha256':sha(directory/'cases.inputs.jsonl'),'reference_annotations':'not yet written; must freeze before generation',
        'selection':'hash order, no historical or new answer/score used; neither independent evaluation of all benchmark queries nor automatic-decomposition accuracy'})
    preview=[]
    for r in selected:
        preview+=['',r['case_id']+' / '+r['group'],'Original: '+r['original'],'Current: '+r['question'],'History: '+json.dumps(r['history'],ensure_ascii=False)]
        seen=set()
        for d in r['packet']['docs']:
            text=' '.join(d['text'].split())
            if text in seen:continue
            seen.add(text);preview.append('['+d['source']+'] '+text)
    (directory/'ANNOTATION_INPUTS.txt').write_text('\n'.join(preview)+'\n',encoding='utf-8')
    print(json.dumps({'round':args.round,'nodes':len(selected),'fresh':3*args.per_dataset,'known':len(selected)-3*args.per_dataset}))

if __name__=='__main__':main()
