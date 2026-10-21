"""Query-only F/S/C/U audit; never changes prompts, labels or selection sets."""
import argparse,collections,json
from pathlib import Path
from reaudit import HERE,ROOT,OUT,rows,write,write_rows,sha
BASE=ROOT/'outputs/query_decomposition/multigrain_4b_frozen_eval300_v1_20261017/returned_20261017T113858Z'
META=HERE.parent/'multigrain_4b_frozen_eval300_v1_20261017/questions300.frozen.jsonl'

def audit():
    predictions=rows(BASE/'predictions.jsonl');meta={r['id']:r for r in rows(META)};decisions={}
    for p in sorted((HERE/'baseline_review').glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf-8'));lo,hi=d['reviewed_indices']
        for index in range(lo,hi+1):
            assert index not in decisions
            pred=predictions[index-1];assert pred['index']==index
            finding=d.get('exceptions',{}).get(str(index))
            r=meta[pred['id']]
            decisions[index]={'index':index,'id':r['id'],'dataset':r['dataset'],'type':r['type'],'query':r['query'],'raw_output':pred['raw_output'],'format_valid':pred['format_valid'],'verdict':finding['verdict'] if finding else 'pass','failed_dimensions':finding.get('dimensions',[]) if finding else [],'reason':finding['reason'] if finding else 'Literal output satisfies F/S/C/U under the frozen v12 protocol; multiple useful granularities allowed.','batch':p.name,'batch_sha256':sha(p),'reviewer':d['reviewer']}
    write_rows(OUT/'baseline300/REVIEW.progress.jsonl',sorted(decisions.values(),key=lambda r:r['index']))
    result={'reviewed':len(decisions),'planned':300,'complete':len(decisions)==300,'verdicts':dict(collections.Counter(r['verdict'] for r in decisions.values())),'passed_over_full300':sum(r['verdict']=='pass' for r in decisions.values())/300,'by_dataset':{},'failed_dimensions_overlapping':dict(collections.Counter(k for r in decisions.values() for k in r['failed_dimensions'])),'provenance':'Original4B/no adapter/v12/greedy; current Codex query-only nonindependent review; not downstream Answer F1.','source_prediction_sha256':sha(BASE/'predictions.jsonl'),'api_calls':0}
    for ds in ['hotpotqa','2wiki','musique']:
        items=[r for r in decisions.values() if r['dataset']==ds];result['by_dataset'][ds]={'reviewed':len(items),'denominator':100,'verdicts':dict(collections.Counter(r['verdict'] for r in items))}
    write(OUT/'baseline300/SUMMARY.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:audit()
    else:
        meta={r['id']:r for r in rows(META)}
        for r in rows(BASE/'predictions.jsonl'):
            if a.start<=r['index']<=a.end:print(str(r['index'])+' ['+meta[r['id']]['dataset']+'] '+r['query']+' => '+r['raw_output'])
