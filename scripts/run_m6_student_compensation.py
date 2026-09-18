"""Single analytically specified Student regularization compensation contrast."""
import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import m6_probe_math as hard
import m6_student_math as soft
import run_m6_objective_readout as old

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1'
LOCAL=ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_localization_v1'
OUT=ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_compensation_v1'
PLAN=ROOT/'analysis/hotpotqa_router/m6_student_compensation_plan_20260915.md'
ARMS=['DirectC','PreC','ProbeC']
OLD_ARMS=['Direct','Pre','Probe']
POLICIES=OLD_ARMS+ARMS+['Dense','BM25']
PRIMARY=['net_transfer_interaction','ProbeC_minus_PreC','ProbeC_minus_DirectC',
         'ProbeC_minus_Direct','ProbeC_minus_Dense','ProbeC_minus_BM25']
CONFIG=dict(lambda_student=.0005,lambda_teacher=.001,alpha=.5,formal_solves=15,pilot_solves=3,
    new_teacher_fits=0,new_targets=0,new_thresholds=0,bootstrap_draws=20000,
    bootstrap_seed=2026091512,quantiles=[.05/12,1-.05/12],minimum_increment=.002,
    minimum_fixed_gain=.01,new_encoder_forwards=0,new_api_calls=0)
sha,read,write,save=old.sha,old.read,old.write,old.save


def versions():
    return {name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','threadpoolctl')}


def sources():
    return [Path(__file__).resolve(),PLAN,ROOT/'scripts/check_m6_student_compensation.py',
        ROOT/'scripts/check_m6_student_transfer.py',Path(hard.__file__),Path(soft.__file__),
        Path(old.__file__),ROOT/'scripts/m6_objective_math.py',
        old.BASE/'weighted_linear_probe.py',old.BASE/'weighted_linear_probe_refined.py']


def inputs():
    return old.input_paths()+[PRIOR/name for name in ('protocol.json','results.json','separate_checks.json',
        'predictions.npz','pilot_targets.npz','pilot.json')]+[PRIOR/f'fold{f}_targets.npz' for f in range(5)]+[
        PRIOR/f'fold{f}_S_{name}.npz' for f in range(5) for name in OLD_ARMS]+[
        LOCAL/'protocol.json',LOCAL/'results.json',LOCAL/'separate_checks.json']


def validate_prior():
    for folder in (PRIOR,LOCAL):
        c=read(folder/'separate_checks.json')
        assert c['results_sha256']==sha(folder/'results.json')
        assert c['protocol_sha256']==sha(folder/'protocol.json')
    assert read(PRIOR/'separate_checks.json')['status']=='passed_independent_student_transfer_checks'
    assert read(LOCAL/'separate_checks.json')['status']=='passed_independent_student_localization_checks'
    for p,digest in read(PRIOR/'separate_checks.json')['artifact_sha256'].items():
        assert sha(p)==digest
    for folder in (PRIOR,LOCAL):
        for kind in ('source_sha256','input_sha256'):
            assert all(sha(path)==digest for path,digest in read(folder/'protocol.json')[kind].items())


def bound(binding):
    assert binding and sha(OUT/'protocol.json')==binding
    p=read(OUT/'protocol.json')
    assert p['config']==CONFIG and p['primary']==PRIMARY
    assert p['versions']==versions()
    for kind in ('source_sha256','input_sha256'):
        assert all(sha(path)==digest for path,digest in p[kind].items())
    return p,old.data()


def freeze():
    assert not OUT.exists()
    validate_prior(); d=old.data()
    with np.load(PRIOR/'predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(d['query_ids'],z['query_ids']) and np.array_equal(d['L_scores'],z['arm_scores'][0])
    OUT.mkdir(parents=True)
    p=dict(status='frozen_before_compensation_pilot_and_fits',created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=CONFIG,primary=PRIMARY,policies=POLICIES,versions=versions(),
        source_sha256={str(p):sha(p) for p in sources()},input_sha256={str(p):sha(p) for p in inputs()},
        scope='Consumed old9600 single regularization mechanism contrast; no independent source validation')
    write(OUT/'protocol.json',p)
    print(json.dumps(dict(status=p['status'],protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def solve(x,q,w,arm,stem,metadata,journal=None):
    model=hard.refined.fit(x,q,w,regularization=.0005) if arm=='DirectC' else soft.fit_soft(x,q,w,.0005)
    record=dict(metadata,arm=arm,**{k:v for k,v in model.items() if k!='coef'})
    save(OUT/(stem+'.npz'),coef=model['coef'],intercept=np.array(model['intercept']))
    write(OUT/(stem+'.json'),record)
    if journal is not None:
        journal.write(json.dumps(dict(stem=stem,**record),allow_nan=False)+'\n');journal.flush()
    assert model['accepted'],'Preserve rejected solution; do not increase budget'
    return model,[OUT/(stem+'.npz'),OUT/(stem+'.json')]


def weights(gap):
    return np.where(abs(gap)>1e-12,abs(gap),0.)


def bce(s,gap):
    w=weights(gap)
    return float(w@np.logaddexp(0.,(1-2*(gap>0))*s.astype(float))/w.sum())


def pilot(binding):
    _,d=bound(binding)
    assert not (OUT/'pilot.json').exists() and not (OUT/'fit_started.json').exists()
    torch=old.gpu();controls=old.old_replay(d,torch)
    with np.load(PRIOR/'pilot_targets.npz',allow_pickle=False) as z:
        ids=z['target_indices'].copy();targets=z['targets'].copy()
    assert len(ids)==64 and targets.shape==(3,64)
    assert np.array_equal(targets[0],(d['gap'][ids]>0).astype(float))
    artifacts=[]
    for arm,q in zip(ARMS,targets):
        model,paths=solve(d['features'][ids],q,weights(d['gap'][ids]),arm,'pilot_'+arm,dict(role='pilot',target_rows=64))
        artifacts+=paths
        scores=old.native(d['features'][ids],model['coef'],model['intercept'],torch)
        assert np.isfinite(scores).all()
    write(OUT/'pilot.json',dict(status='passed_three_compensation_pilot_fits_and_480_native_controls',protocol_sha256=binding,
        original_controls=controls,pilot_solves=3,artifact_sha256={str(p):sha(p) for p in artifacts},new_policy_effects=False))
    print(json.dumps(dict(status='pilot_passed',pilot_solves=3)),flush=True)


def run(binding):
    p,d=bound(binding);checked=read(OUT/'pilot.json')
    assert checked['protocol_sha256']==binding and checked['status']=='passed_three_compensation_pilot_fits_and_480_native_controls'
    assert all(sha(path)==digest for path,digest in checked['artifact_sha256'].items())
    assert not (OUT/'fit_started.json').exists()
    write(OUT/'fit_started.json',dict(protocol_sha256=binding,started_at_unix=time.time()))
    started=time.perf_counter();torch=old.gpu();x,gap=d['features'],d['gap'];heads=[];artifacts=[];records=[]
    with threadpool_limits(limits=2),(OUT/'solutions.jsonl').open('x',encoding='utf-8') as journal:
        for f,(fit,cal,test) in enumerate(d['folds']):
            with np.load(PRIOR/f'fold{f}_targets.npz',allow_pickle=False) as z:
                assert np.array_equal(fit,z['fit_indices'])
                q=z['targets'].copy();w=z['weights'].copy()
            assert np.array_equal(w,weights(gap[fit])) and np.array_equal(q[0],(gap[fit]>0).astype(float))
            models=[];fs=[];cs=[];geometry={}
            with np.load(PRIOR/f'fold{f}_S_Direct.npz',allow_pickle=False) as z:
                direct_beta=z['coef'].copy()
            for arm,target,old_arm in zip(ARMS,q,OLD_ARMS):
                m,paths=solve(x[fit],target,w,arm,f'fold{f}_{arm}',dict(role='student',fold=f,targets_sha256=sha(PRIOR/f'fold{f}_targets.npz')),journal)
                artifacts+=paths;models.append(m)
                fs.append(old.native(x[fit],m['coef'],m['intercept'],torch));cs.append(old.native(x[cal],m['coef'],m['intercept'],torch))
                with np.load(PRIOR/f'fold{f}_S_{old_arm}.npz',allow_pickle=False) as z:
                    beta=z['coef'].copy()
                geometry[arm]=dict(beta_norm=float(np.linalg.norm(m['coef'])),bias=float(m['intercept']),
                    beta_distance_to_old_corresponding=float(np.linalg.norm(m['coef']-beta)),
                    beta_distance_to_old_Direct=float(np.linalg.norm(m['coef']-direct_beta)))
            path=OUT/f'fold{f}_fit_cal.npz';save(path,fit_indices=fit,cal_indices=cal,fit_scores=np.stack(fs),cal_scores=np.stack(cs));artifacts.append(path)
            records.append(dict(fold=f,fit_BCE=[bce(s,gap[fit]) for s in fs],cal_BCE=[bce(s,gap[cal]) for s in cs],heads=geometry));heads.append(models)
            print(json.dumps(dict(status='fold_complete',fold=f,formal_solves=(f+1)*3)),flush=True)
    artifacts.append(OUT/'solutions.jsonl')
    write(OUT/'fit_completion.json',dict(status='all_15_heads_before_new_test_predictions',protocol_sha256=binding,formal_solves=15,
        artifact_sha256={str(path):sha(path) for path in artifacts},completed_at_utc=datetime.now(timezone.utc).isoformat()))
    scores=np.full((3,9600),np.nan,np.float32);fold_id=np.full(9600,-1,int)
    for f,(_,_,test) in enumerate(d['folds']):
        for j,m in enumerate(heads[f]):scores[j,test]=old.native(x[test],m['coef'],m['intercept'],torch)
        fold_id[test]=f
    assert np.isfinite(scores).all() and np.all(fold_id>=0)
    save(OUT/'predictions.npz',query_ids=d['query_ids'],group_ids=d['group_ids'],utility=d['utility'],arm_scores=scores,fold_id=fold_id)
    write(OUT/'predictions_frozen.json',dict(status='all_compensated_predictions_before_effects',protocol_sha256=binding,
        predictions_sha256=sha(OUT/'predictions.npz'),fit_completion_sha256=sha(OUT/'fit_completion.json')))
    with np.load(PRIOR/'predictions.npz',allow_pickle=False) as z:previous_scores=z['arm_scores'].copy()
    actions={arm:previous_scores[j]>0 for j,arm in enumerate(OLD_ARMS)}
    actions.update({arm:scores[j]>0 for j,arm in enumerate(ARMS)})
    actions.update(Dense=np.zeros(9600,bool),BM25=np.ones(9600,bool))
    u={name:np.where(a,d['utility'][:,0],d['utility'][:,1]) for name,a in actions.items()}
    values=np.column_stack([(u['ProbeC']-u['DirectC'])-(u['Probe']-u['Direct'])]+[u['ProbeC']-u[name] for name in ('PreC','DirectC','Direct','Dense','BM25')])
    _,inverse=np.unique(d['group_ids'],return_inverse=True);sizes=np.bincount(inverse)
    sums=np.column_stack([np.bincount(inverse,weights=values[:,j]) for j in range(6)])
    rng=np.random.default_rng(2026091512);draws=np.empty((20000,6))
    with threadpool_limits(limits=2):
        for start in range(0,20000,100):
            ix=rng.integers(len(sizes),size=(100,len(sizes)));draws[start:start+100]=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)[:,None]
    save(OUT/'bootstrap.npz',draws=draws)
    intervals=np.quantile(draws,CONFIG['quantiles'],axis=0).T
    primary={name:dict(mean=float(values[:,j].mean()),interval=intervals[j].tolist()) for j,name in enumerate(PRIMARY)}
    transfer=all(primary[k]['mean']>=.002 and primary[k]['interval'][0]>0 for k in PRIMARY[1:4])
    candidate=transfer and all(primary[k]['mean']>=.01 and primary[k]['interval'][0]>0 for k in PRIMARY[4:])
    decision='PREPARE_COMPENSATED_STUDENT_INDEPENDENT_CONFIRMATION' if candidate else ('COMPENSATED_INCREMENT_ONLY_NO_QUALIFIED_CANDIDATE' if transfer else 'END_SINGLE_REGULARIZATION_COMPENSATION_NO_CONFIRMED_TRANSFER')
    for f,(_,_,test) in enumerate(d['folds']):
        records[f]['test_BCE']=[bce(s[test],gap[test]) for s in scores]
        records[f]['primary_means']={name:float(values[test,j].mean()) for j,name in enumerate(PRIMARY)}
    result=dict(status='complete_compensation_pending_separate_check',protocol_sha256=binding,primary=primary,
        policy={name:old.policy_summary(actions[name],d['utility'],gap) for name in POLICIES},folds=records,
        descriptive_compensation_gains={a:float(np.mean(u[b]-u[a])) for a,b in zip(OLD_ARMS,ARMS)},
        transfer_gate=transfer,candidate_preparation_gate=candidate,decision=decision,formal_solves=15,pilot_solves=3,
        new_teacher_fits=0,new_targets=0,new_thresholds=0,new_encoder_forwards=0,new_api_calls=0,runtime_available=False,core_goal_achieved=False,
        predictions_frozen_sha256=sha(OUT/'predictions_frozen.json'),bootstrap_sha256=sha(OUT/'bootstrap.npz'),elapsed_seconds=time.perf_counter()-started,scope=p['scope'])
    bound(binding);write(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],primary=primary,decision=decision)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','pilot','run']);parser.add_argument('--protocol-sha');args=parser.parse_args()
    try:
        if args.mode=='freeze':freeze()
        elif args.mode=='pilot':pilot(args.protocol_sha)
        else:run(args.protocol_sha)
    except Exception:
        if OUT.exists() and not (OUT/(args.mode+'_failure.json')).exists():write(OUT/(args.mode+'_failure.json'),dict(traceback=traceback.format_exc()))
        raise
