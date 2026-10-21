"""Full-denominator semantic before/after comparison with paired uncertainty."""
import argparse,collections,json,random
from pathlib import Path
from reaudit import HERE,OUT,rows,write,write_rows,sha
from prepare_data import components,EVAL

def percentile_ci(values):
    s=sorted(values);return [s[int(.025*len(s))],s[min(len(s)-1,int(.975*len(s)))]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trained-predictions',type=Path,required=True);ap.add_argument('--trained-review',type=Path,required=True);a=ap.parse_args()
    selected=json.loads((OUT/'SELECTED_CHECKPOINT.json').read_text());assert not selected['selection_uses_final_eval300']
    meta={r['id']:r for r in rows(EVAL/'questions300.frozen.jsonl')}
    baseline={r['id']:r for r in rows(OUT/'baseline300/REVIEW.progress.jsonl')}
    pred={r['id']:r for r in rows(a.trained_predictions)};review={r['id']:r for r in rows(a.trained_review)}
    assert set(meta)==set(baseline)==set(pred)==set(review) and len(meta)==300
    comparisons=[]
    for qid,r in meta.items():
        p=pred[qid];j=review[qid];assert p['query']==r['query'] and j['raw_output']==p['raw_output']
        assert j['verdict'] in ['pass','fail','uncertain'] and j['reason']
        before=baseline[qid]['verdict']=='pass';after=j['verdict']=='pass'
        if after:assert p['format_valid'] and not p['truncated'] and not p.get('generation_error') and not p['explicit_references']
        comparisons.append({'id':qid,'dataset':r['dataset'],'type':r['type'],'query':r['query'],'before_pass':before,'after_pass':after,'delta':int(after)-int(before),'before_verdict':baseline[qid]['verdict'],'after_verdict':j['verdict'],'before_reason':baseline[qid]['reason'],'after_reason':j['reason']})
    def describe(items):
        n=len(items);return {'n':n,'before_passed':sum(r['before_pass'] for r in items),'after_passed':sum(r['after_pass'] for r in items),'before_rate':sum(r['before_pass'] for r in items)/n,'after_rate':sum(r['after_pass'] for r in items)/n,'delta_percentage_points':100*sum(r['delta'] for r in items)/n,'improved':sum(r['delta']==1 for r in items),'regressed':sum(r['delta']==-1 for r in items),'unchanged':sum(r['delta']==0 for r in items),'after_uncertain':sum(r['after_verdict']=='uncertain' for r in items)}
    by_ds={ds:[r for r in comparisons if r['dataset']==ds] for ds in ['hotpotqa','2wiki','musique']}
    rng=random.Random(20261017);boot=[]
    for _ in range(10000):boot.append(100*sum(rng.choice(items)['delta'] for items in by_ds.values() for _ in range(100))/300)
    # Sensitivity: resample connected components of shared MuSiQue primitive IDs.
    mu=by_ds['musique'];parents=list(range(len(mu)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    owner={}
    for i,r in enumerate(mu):
        for part in components(meta[r['id']]):
            if part in owner:parents[find(i)]=find(owner[part])
            else:owner[part]=i
    clusters=collections.defaultdict(list)
    for i,r in enumerate(mu):clusters[find(i)].append(r['delta'])
    groups=list(clusters.values());sensitivity=[]
    for _ in range(10000):
        means=[sum(rng.choice(by_ds[ds])['delta'] for _ in range(100))/100 for ds in ['hotpotqa','2wiki']]
        sampled=[rng.choice(groups) for _ in groups];means.append(sum(sum(g) for g in sampled)/sum(len(g) for g in sampled))
        sensitivity.append(100*sum(means)/3)
    report={'metric':'Frozen v12 complete decomposition F/S/C/U pass, not RAG Answer F1','overall':describe(comparisons),'by_dataset':{ds:describe(items) for ds,items in by_ds.items()},'paired_stratified_bootstrap_95ci_delta_pp':percentile_ci(boot),'musique_component_cluster_sensitivity_95ci_delta_pp':percentile_ci(sensitivity),'musique_components':len(groups),'target_observed_at_least90_percent':sum(r['after_pass'] for r in comparisons)>=270,'selection':selected['selected'],'trained_review_sha256':sha(a.trained_review),'trained_predictions_sha256':sha(a.trained_predictions),'limitations':'Current Codex review is not independent human gold. Components/text/ID checks do not establish full paraphrase or pretraining decontamination. Test300 did not select checkpoint.','api_calls':0}
    write_rows(OUT/'FINAL300_PAIRED_CASES.jsonl',comparisons);write(OUT/'FINAL300_COMPARISON.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
