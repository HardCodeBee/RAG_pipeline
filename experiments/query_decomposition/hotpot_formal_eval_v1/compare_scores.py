"""Offline paired Answer-F1 statistics; never part of method execution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import write_json,utc


def read(path):
    return {r['qid']:r for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip() for r in [json.loads(line)]}


def paired(treatment,control,replicates=10000,seed=20261014):
    if set(treatment)!=set(control):raise ValueError('Planned query sets differ')
    known=[];lower=upper=0.0;missing=defaultdict(int)
    for qid in sorted(treatment):
        a,b=treatment[qid],control[qid];x,y=a['f1'],b['f1']
        if a.get('type')!=b.get('type') or a.get('group_id')!=b.get('group_id'):
            raise ValueError('Frozen strata/groups differ')
        if x is not None and y is not None:
            d=x-y;lower+=d;upper+=d
            known.append((a.get('type') or 'unspecified',a.get('group_id') or qid,d))
        elif x is not None:lower+=x-1;upper+=x;missing['control_only']+=1
        elif y is not None:lower-=y;upper+=1-y;missing['treatment_only']+=1
        else:lower-=1;upper+=1;missing['both']+=1
    n=len(treatment)
    report={'planned':n,'common_scored':len(known),'missing_pair_counts':dict(missing),
            'planned_mean_difference_interval':[lower/n,upper/n] if n else None,
            'difference_units':'F1 fraction; multiply by100 for percentage points',
            'ci_scope':'conditional on common scored cohort; all planned missingness bounds retained',
            'query_group_bootstrap_stratified_by_type':True,'seed':seed,'replicates':replicates}
    if not known:return report
    delta=np.array([r[2] for r in known]);mean=float(delta.mean())
    report.update(mean_difference_common=mean,gain=int((delta>1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()),loss=int((delta < -1e-12).sum()))
    strata=defaultdict(lambda:defaultdict(list));group_types=defaultdict(set)
    for kind,group,d in known:strata[kind][group].append(d);group_types[group].add(kind)
    if any(len(kinds)>1 for kinds in group_types.values()):raise ValueError('Mixed-type cluster needs a separately declared resampling policy')
    rng=np.random.default_rng(seed);bootstrap=np.zeros(replicates)
    totals=[]
    for kind,groups in sorted(strata.items()):
        values=list(groups.values());sums=np.array([sum(v) for v in values]);sizes=np.array([len(v) for v in values]);weight=float(sizes.sum())/len(known)
        for start in range(0,replicates,250):
            count=min(250,replicates-start);indices=rng.integers(0,len(values),size=(count,len(values)))
            bootstrap[start:start+count]+=weight*sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
        totals.extend(sums.tolist())
    report['paired_95_ci_common']=np.quantile(bootstrap,[.025,.975]).tolist()
    totals=np.asarray(totals);exceed=0
    for start in range(0,replicates,250):
        count=min(250,replicates-start);signs=rng.choice([-1.,1.],size=(count,len(totals)))
        null=np.abs((signs*totals).sum(axis=1)/len(known));exceed+=int((null>=abs(mean)-1e-12).sum())
    report['two_sided_group_sign_flip_p']=(exceed+1)/(replicates+1)
    return report


def holm(pvalues):
    ordered=sorted(pvalues,key=lambda k:pvalues[k]);adjusted={};previous=0
    for rank,key in enumerate(ordered):
        previous=max(previous,min(1,pvalues[key]*(len(ordered)-rank)));adjusted[key]=previous
    return adjusted


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--treatment',required=True,type=Path);parser.add_argument('--control',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    report={'utc':utc(),'treatment_file':str(args.treatment),'control_file':str(args.control),**paired(read(args.treatment),read(args.control))}
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
