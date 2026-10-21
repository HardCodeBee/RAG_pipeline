"""Lock baseline judgments before reading candidates; retain all denominators."""
import argparse,collections,json,statistics
from prepare import HERE,OUT,rows,write,sha,freeze

def raw(phase,name):
    folder={'screen90':'raw_screen','confirm120':'raw_confirmation','rare16':'raw_rare'}[phase]
    return {r['index']:r for p in sorted((OUT/folder/name).glob('[0-9][0-9][0-9].json')) for r in [json.loads(p.read_text(encoding='utf-8'))]}

def audit(phase,name):
    predictions=raw(phase,name);judged={};base_path=OUT/'reviews'/phase/'b00_v12.jsonl'
    roster=rows(HERE/('rare_queries.frozen.jsonl' if phase=='rare16' else phase+'.questions.frozen.jsonl'))
    baseline={r['index']:r for r in rows(base_path)} if name!='b00_v12' and base_path.exists() else {}
    baseline_raw=raw(phase,'b00_v12') if name!='b00_v12' else {}
    for p in sorted((HERE/'reviews'/phase/name).glob('batch_*.json')):
        batch=json.loads(p.read_text(encoding='utf-8'));lo,hi=batch['reviewed_indices']
        for i in range(lo,hi+1):
            assert i not in judged and i in predictions
            r=predictions[i];f=batch.get('exceptions',{}).get(str(i))
            expected=roster[i-1];assert r['id']==expected['id'] and r['query']==expected['query'] and r['dataset']==expected['dataset']
            if name!='b00_v12':assert i in baseline
            if name!='b00_v12' and r['parsed']==baseline_raw[i]['parsed']:
                assert f is None
                v,d,reason=baseline[i]['verdict'],baseline[i]['failed_dimensions'],'Identical question array: inherit locked baseline. '+baseline[i]['reason']
            else:v,d,reason=(f['verdict'],f['dimensions'],f['reason']) if f else ('pass',[],'Inspected every original/query pair under unchanged F/S/C/U; no definite defect. No factual answers or native decomposition labels consulted.')
            if v=='pass':assert r['format_valid'] and not r['generation_error'] and not r['truncated'] and not r['explicit_references']
            judged[i]={'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':r['raw_output'],'verdict':v,'failed_dimensions':d,'reason':reason,'review_batch_sha256':sha(p),'reviewer':'Current Codex, non-independent query/output review. Uncertainty retained as nonpass.'}
    result=sorted(judged.values(),key=lambda r:r['index']);write_path=OUT/'reviews'/phase/(name+'.jsonl');write_path.parent.mkdir(parents=True,exist_ok=True)
    write_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    return result

def summarize(phase):
    names=['b00_v12','m01_static_mu','m02_adaptive_mu','m03_scope_contrast','t01_original_mix','t02_mu50'];summary={}
    for name in names:
        rr=audit(phase,name)
        if not rr:continue
        by={ds:{'reviewed':sum(r['dataset']==ds for r in rr),'passed':sum(r['dataset']==ds and r['verdict']=='pass' for r in rr)} for ds in ['hotpotqa','2wiki','musique']}
        summary[name]={'reviewed':len(rr),'passed':sum(r['verdict']=='pass' for r in rr),'verdicts':dict(collections.Counter(r['verdict'] for r in rr)),'by_dataset':by,
                       'mu_F_or_S_indices':[r['index'] for r in rr if r['dataset']=='musique' and set(r['failed_dimensions'])&{'F','S'}]}
        if all(v['reviewed'] for v in by.values()):summary[name]['macro_pass_rate']=statistics.mean(v['passed']/v['reviewed'] for v in by.values())
        if name!='b00_v12' and 'b00_v12' in summary:
            bb={r['index']:r for r in rows(OUT/'reviews'/phase/'b00_v12.jsonl')}
            summary[name]['paired']={'fixes':[r['index'] for r in rr if r['verdict']=='pass' and bb[r['index']]['verdict']!='pass'],'regressions':[r['index'] for r in rr if r['verdict']!='pass' and bb[r['index']]['verdict']=='pass']}
    write(OUT/'reviews'/phase/'SUMMARY.progress.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',default='screen90');ap.add_argument('--variant',default='b00_v12');ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=90);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:summarize(a.phase);return
    preds=raw(a.phase,a.variant);base=raw(a.phase,'b00_v12')
    locked={r['index']:r for r in rows(OUT/'reviews'/a.phase/'b00_v12.jsonl')} if (OUT/'reviews'/a.phase/'b00_v12.jsonl').exists() else {}
    for i in range(a.start,a.end+1):
        if i not in preds:continue
        r=preds[i]
        if a.variant!='b00_v12':assert i in locked,'Lock baseline before candidate inspection'
        if a.variant!='b00_v12' and r['parsed']==base[i]['parsed']:print(f'{i} SAME {locked[i]["verdict"]}')
        else:print(f'{i} [{r["dataset"]}] {r["query"]} => {r["raw_output"]}')

if __name__=='__main__':main()
