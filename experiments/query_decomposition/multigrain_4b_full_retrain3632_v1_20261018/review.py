"""Same F/S/C/U rubric; inherit exact arrays from locked T02 reviews."""
import argparse,collections,json
from prepare import HERE,OUT,PRIOR,ROOT,rows,sha,freeze
OLDOUT=ROOT/'outputs/query_decomposition'/PRIOR.name
PARENT='p00_parent_t02'

def raw(phase,checkpoint=None):
    folder=OUT/'raw_dev'/checkpoint if phase=='dev60' else OUT/'raw_confirmation'
    return {r['index']:r for p in sorted(folder.glob('[0-9][0-9][0-9].json')) for r in [json.loads(p.read_text(encoding='utf-8'))]}

def audit(phase,checkpoint=None):
    name=checkpoint if phase=='dev60' else 'selected'
    pred=raw(phase,checkpoint);roster=rows(HERE/(phase+'.questions.frozen.jsonl'))
    locked=rows(OLDOUT/'reviews'/phase/(PARENT+'.jsonl'));base={r['index']:r for r in locked}
    assert len(locked)==len(roster)
    rr={}
    for path in sorted((HERE/'reviews'/phase/name).glob('batch_*.json')):
        b=json.loads(path.read_text(encoding='utf-8'));lo,hi=b['reviewed_indices']
        for i in range(lo,hi+1):
            assert i not in rr and i in pred
            r=pred[i];ref=roster[i-1]
            assert (r['id'],r['query'],r['dataset'])==(ref['id'],ref['query'],ref['dataset'])
            exception=b.get('exceptions',{}).get(str(i));parsed=json.loads(r['raw_output']) if r['format_valid'] else None
            bp=json.loads(base[i]['raw_output']) if base[i].get('raw_output') else None
            if parsed is not None and parsed==bp:
                assert exception is None;v,d,reason=base[i]['verdict'],base[i]['failed_dimensions'],'Exact array: locked T02 verdict. '+base[i]['reason']
            elif exception:v,d,reason=exception['verdict'],exception['dimensions'],exception['reason']
            else:v,d,reason='pass',[],'Every question inspected against original under frozen F/S/C/U. Same Codex, not independent gold.'
            if v=='pass':assert r['format_valid'] and not r['truncated'] and not r.get('generation_error') and not r['explicit_references']
            rr[i]={'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':r['raw_output'],
              'verdict':v,'failed_dimensions':d,'reason':reason,'review_batch_sha256':sha(path)}
    result=sorted(rr.values(),key=lambda r:r['index']);dest=OUT/'reviews'/phase/(name+'.jsonl');dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    return result

def summarize(rr):
    return {'n':len(rr),'passed':sum(r['verdict']=='pass' for r in rr),
      'by_dataset':{d:{'n':sum(r['dataset']==d for r in rr),'passes':sum(r['dataset']==d and r['verdict']=='pass' for r in rr)} for d in ['hotpotqa','2wiki','musique']},
      'mu_FS':[r['index'] for r in rr if r['dataset']=='musique' and set(r['failed_dimensions'])&{'F','S'}],
      'verdicts':dict(collections.Counter(r['verdict'] for r in rr))}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',default='dev60');ap.add_argument('--checkpoint');ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=60);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:print(json.dumps(summarize(audit(a.phase,a.checkpoint)),indent=2));raise SystemExit
    pp=raw(a.phase,a.checkpoint);bb={r['index']:r for r in rows(OLDOUT/'reviews'/a.phase/(PARENT+'.jsonl'))}
    for i in range(a.start,a.end+1):
        if i not in pp:continue
        r=pp[i];same=r['format_valid'] and json.loads(r['raw_output'])==json.loads(bb[i]['raw_output'])
        if same:print(i,'SAME',bb[i]['verdict'])
        else:print(f'{i} [{r["dataset"]}] {r["query"]} => {r["raw_output"]}')
