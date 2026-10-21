"""Manual query/output audits; exact identical arrays inherit the locked parent."""
import argparse,collections,json,statistics
from prepare import HERE,OUT,rows,sha,write

PARENT='p00_parent_t02'
def raw(phase,name):
    folder={'dev60':'raw_dev','confirm120':'raw_confirmation'}[phase]
    return {r['index']:r for p in sorted((OUT/folder/name).glob('[0-9][0-9][0-9].json')) for r in [json.loads(p.read_text(encoding='utf-8'))]}

def audit(phase,name):
    pp=raw(phase,name);roster=rows(HERE/(phase+'.questions.frozen.jsonl'));done={};bp=OUT/'reviews'/phase/(PARENT+'.jsonl')
    locked={r['index']:r for r in rows(bp)} if name!=PARENT and bp.exists() else {};base=raw(phase,PARENT) if name!=PARENT else {}
    for path in sorted((HERE/'reviews'/phase/name).glob('batch_*.json')):
        batch=json.loads(path.read_text(encoding='utf-8'));lo,hi=batch['reviewed_indices']
        for i in range(lo,hi+1):
            assert i not in done and i in pp;r=pp[i];ref=roster[i-1]
            assert (r['id'],r['query'],r['dataset'])==(ref['id'],ref['query'],ref['dataset'])
            f=batch.get('exceptions',{}).get(str(i))
            if name!=PARENT:assert i in locked
            if name!=PARENT and r['parsed']==base[i]['parsed']:
                assert f is None;v,d,reason=locked[i]['verdict'],locked[i]['failed_dimensions'],'Identical array: inherit locked parent. '+locked[i]['reason']
            else:v,d,reason=(f['verdict'],f['dimensions'],f['reason']) if f else ('pass',[],'Every question inspected against original under frozen F/S/C/U; no definite defect. No answers/passages/native target consulted.')
            if v=='pass':assert r['format_valid'] and not r['generation_error'] and not r['truncated'] and not r['explicit_references']
            done[i]={'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':r['raw_output'],'verdict':v,'failed_dimensions':d,'reason':reason,'review_batch_sha256':sha(path),'reviewer':'Current Codex non-independent query/output review. Uncertainty retained as nonpass.'}
    rr=sorted(done.values(),key=lambda r:r['index']);p=OUT/'reviews'/phase/(name+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf-8');return rr

def summary(phase):
    result={}
    for name in [PARENT,'r01_old_data','r02_expanded','r03_expanded20','original165']:
        rr=audit(phase,name)
        if not rr:continue
        by={d:{'n':sum(r['dataset']==d for r in rr),'passes':sum(r['dataset']==d and r['verdict']=='pass' for r in rr)} for d in ['hotpotqa','2wiki','musique']}
        result[name]={'n':len(rr),'passed':sum(r['verdict']=='pass' for r in rr),'by_dataset':by,'verdicts':dict(collections.Counter(r['verdict'] for r in rr)),
            'mu_FS':[r['index'] for r in rr if r['dataset']=='musique' and set(r['failed_dimensions'])&{'F','S'}]}
        if name!=PARENT:
            bb={r['index']:r for r in rows(OUT/'reviews'/phase/(PARENT+'.jsonl'))}
            result[name]['fixes']=[r['index'] for r in rr if r['verdict']=='pass' and bb[r['index']]['verdict']!='pass']
            result[name]['regressions']=[r['index'] for r in rr if r['verdict']!='pass' and bb[r['index']]['verdict']=='pass']
    write(OUT/'reviews'/phase/'SUMMARY.progress.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',default='dev60');ap.add_argument('--variant',default=PARENT);ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=60);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:print(json.dumps(summary(a.phase),ensure_ascii=False,indent=2));return
    pp=raw(a.phase,a.variant);base=raw(a.phase,PARENT);bp=OUT/'reviews'/a.phase/(PARENT+'.jsonl');locked={r['index']:r for r in rows(bp)} if bp.exists() else {}
    for i in range(a.start,a.end+1):
        if i not in pp:continue
        r=pp[i]
        if a.variant!=PARENT:
            assert i in locked
            if r['parsed']==base[i]['parsed']:print(i,'SAME',locked[i]['verdict']);continue
        print(f'{i} [{r["dataset"]}] {r["query"]} => {r["raw_output"]}')

if __name__=='__main__':main()
