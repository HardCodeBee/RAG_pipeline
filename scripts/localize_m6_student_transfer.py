"""Describe fixed Student target moments, head responses and action changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
PRIOR = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_localization_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_student_localization_plan_20260915.md'
ARMS = ['Direct', 'Pre', 'Probe']
PAIRS = [(0, 1), (1, 2), (0, 2)]
NAMES = ['Pre_minus_Direct', 'Probe_minus_Pre', 'Probe_minus_Direct']
LAMBDA = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator else None


def loss_gradient(u, q, w, theta):
    s = u@theta
    beta = theta[:-1]
    loss = w@(q*np.logaddexp(0., -s)+(1.-q)*np.logaddexp(0., s))+.5*LAMBDA*(beta@beta)
    gradient = u.T@(w*(expit(s)-q)) + LAMBDA*np.r_[beta, 0.]
    return float(loss), gradient


def moments(x, w, qa, qb, ta, tb):
    u = np.column_stack((x.astype(float), np.ones(len(x))))
    dq, delta = qb-qa, tb-ta
    m = u.T@(w*dq)
    xbar = w@x
    centered = m[:-1]-xbar*m[-1]
    dq_rms = float(np.sqrt(w@(dq*dq)))
    dq_center_rms = float(np.sqrt(w@((dq-m[-1])**2)))
    xrms = float(np.sqrt(w@np.sum((x-xbar)**2, axis=1)))
    denominator = dq_center_rms*xrms
    la, ga = loss_gradient(u, qa, w, ta)
    lb, gb = loss_gradient(u, qb, w, tb)
    error_a = loss_gradient(u, qb, w, ta)[0]-la+ta@m
    error_b = lb-loss_gradient(u, qa, w, tb)[0]+tb@m
    sa, sb = u@ta, u@tb
    data_vector = u.T@(w*(expit(sb)-expit(sa)))
    response = data_vector+LAMBDA*np.r_[delta[:-1], 0.]
    residual = response-m
    data_energy = float(w@((sb-sa)*(expit(sb)-expit(sa))))
    regularization_energy = float(LAMBDA*(delta[:-1]@delta[:-1]))
    total = data_energy+regularization_energy
    out = dict(target_difference_mean=float(m[-1]), target_difference_rms=dq_rms,
        target_difference_centered_rms=dq_center_rms, moment_norm=float(np.linalg.norm(m)),
        centered_feature_moment_norm=float(np.linalg.norm(centered)),
        centered_first_moment_association=float(np.linalg.norm(centered)/denominator) if denominator else None,
        delta_beta_norm=float(np.linalg.norm(delta[:-1])), delta_bias=float(delta[-1]),
        source_gradient_inf=float(np.max(np.abs(ga))), target_gradient_inf=float(np.max(np.abs(gb))),
        objective_identity_error=max(abs(error_a), abs(error_b)),
        response_stationarity_residual_inf=float(np.max(np.abs(residual))),
        response_identity_error=float(np.max(np.abs(residual-(gb-ga)))),
        data_response_energy=data_energy, regularization_response_energy=regularization_energy,
        regularization_energy_fraction=regularization_energy/total if total else None,
        moment_response_energy=float(delta@m), energy_stationarity_residual=float(total-delta@m),
        energy_identity_error=float(abs(total-delta@m-delta@(gb-ga))),
        fit_fp64_score_delta_weighted_rms=float(np.sqrt(w@((sb-sa)**2))))
    return out, dict(moment=m, centered_moment=centered, delta=delta, gradient_a=ga, gradient_b=gb)


def shrinkage(x, w, y, teacher_probability, theta_direct, direct_native):
    u=np.column_stack((x.astype(float),np.ones(len(x))))
    p_direct=expit(u@theta_direct)
    native_probability=expit(direct_native.astype(float))
    _, gradient=loss_gradient(u,y,w,theta_direct)
    components=dict(regularization=-.5*LAMBDA*np.r_[theta_direct[:-1],0.],
        optimizer_residual=.5*gradient,
        native_rounding=.5*u.T@(w*(native_probability-p_direct)),
        oof_vs_full_native=.5*u.T@(w*(teacher_probability-native_probability)))
    m=.5*u.T@(w*(teacher_probability-y))
    residual=m-sum(components.values())
    current=gradient-m
    compensated=current+components['regularization']
    record=dict(decomposition_error=float(np.max(abs(residual))),
        component_norms={key:float(np.linalg.norm(v)) for key,v in components.items()},
        component_cosine_to_moment={key:cosine(v,m) for key,v in components.items()},
        Direct_point_current_soft_gradient_norm=float(np.linalg.norm(current)),
        Direct_point_compensated_soft_gradient_norm=float(np.linalg.norm(compensated)),
        compensated_to_current_gradient_ratio=float(np.linalg.norm(compensated)/np.linalg.norm(current)) if np.linalg.norm(current) else None)
    return record,dict(**components,current_soft_gradient_at_Direct=current,compensated_soft_gradient_at_Direct=compensated)


def action_changes(sa, sb, gap):
    a, b = sa>0, sb>0
    difference = sb.astype(float)-sa.astype(float)
    quality = (b.astype(int)-a.astype(int))*gap
    denom = float(np.std(sa)*np.std(sb))
    out = dict(rows=len(gap), score_delta_mean=float(np.mean(difference)),
        score_delta_rms=float(np.sqrt(np.mean(difference**2))),
        score_correlation=float(np.cov(sa.astype(float), sb.astype(float), ddof=0)[0,1]/denom) if denom else None,
        changed_actions=int(np.sum(a!=b)), unchanged_action_score_delta_rms=float(np.sqrt(np.mean(difference[a==b]**2))) if np.any(a==b) else None,
        gain=float(np.mean(quality)), cells={})
    for av, bv, label in [(False,False,'D_to_D'), (False,True,'D_to_B'), (True,False,'B_to_D'), (True,True,'B_to_B')]:
        mask = (a==av)&(b==bv)
        v = quality[mask]
        out['cells'][label] = dict(rows=int(mask.sum()), improvements=int(np.sum(v>1e-12)),
            harms=int(np.sum(v < -1e-12)), ties=int(np.sum(abs(v)<=1e-12)),
            quality_sum=float(v.sum()), contribution=float(v.sum()/len(gap)))
    assert sum(v['rows'] for v in out['cells'].values()) == len(gap)
    assert abs(sum(v['contribution'] for v in out['cells'].values())-out['gain'])<1e-12
    return out


def self_test():
    rng = np.random.default_rng(2026091511)
    x = rng.normal(size=(50,4)); w = rng.uniform(.1, 1, 50); w/=w.sum()
    qa, qb = rng.random((2,50)); ta, tb = rng.normal(size=(2,5))
    record, vectors = moments(x,w,qa,qb,ta,tb)
    for key in ('objective_identity_error','response_identity_error','energy_identity_error'):
        assert record[key]<1e-12
    zero, _ = moments(x,w,qa,qa,ta,ta)
    assert zero['moment_norm']==0 and zero['delta_beta_norm']==0 and zero['regularization_energy_fraction'] is None
    u=np.column_stack((x,np.ones(50))); dq=qb-qa
    assert np.max(abs(vectors['centered_moment']-((x-w@x).T@(w*dq))))<1e-14
    changes=action_changes(np.array([-1.,-1.,1.,1.]), np.array([-1.,1.,-1.,1.]), np.array([.4,.3,.2,-.5]))
    assert all(c['rows']==1 for c in changes['cells'].values()) and abs(changes['gain']-.025)<1e-14
    sh,_=shrinkage(x,w,qa,qb,ta,np.column_stack((x,np.ones(50)))@ta)
    assert sh['decomposition_error']<1e-12
    return dict(status='passed_synthetic_moment_response_and_action_identities',
        objective_error=record['objective_identity_error'], response_error=record['response_identity_error'],
        energy_error=record['energy_identity_error'], new_fits=0, real_data_reads=0)


def input_paths():
    paths=[BASE/'layer_pooling_v1/features.npz', PRIOR/'protocol.json', PRIOR/'results.json', PRIOR/'separate_checks.json', PRIOR/'predictions.npz']
    for f in range(5):
        paths += [PRIOR/f'fold{f}_targets.npz', PRIOR/f'fold{f}_fit_cal.npz']
        paths += [PRIOR/f'fold{f}_S_{arm}.npz' for arm in ARMS]
    return paths


def freeze():
    assert not OUT.exists()
    checks=read(PRIOR/'separate_checks.json')
    assert checks['status']=='passed_independent_student_transfer_checks'
    assert checks['results_sha256']==sha(PRIOR/'results.json')
    assert checks['protocol_sha256']==sha(PRIOR/'protocol.json')
    for kind in ('source_sha256','input_sha256'):
        assert all(sha(p)==s for p,s in read(PRIOR/'protocol.json')[kind].items())
    for p in input_paths():
        if str(p) in checks['artifact_sha256']:
            assert sha(p)==checks['artifact_sha256'][str(p)]
    OUT.mkdir(parents=True)
    p=dict(status='frozen_before_new_descriptive_localization', created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path):sha(path) for path in [Path(__file__).resolve(), PLAN]},
        input_sha256={str(path):sha(path) for path in input_paths()}, synthetic_checks=self_test(),
        comparisons=NAMES, lambda_value=LAMBDA, new_fits=0, new_policies=0, new_api_calls=0,
        scope='Consumed development descriptive localization; former aggregate effects already known; no new candidate test')
    write(OUT/'protocol.json',p)
    print(json.dumps(dict(status=p['status'],protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def run(binding):
    assert binding and sha(OUT/'protocol.json')==binding
    p=read(OUT/'protocol.json')
    for kind in ('source_sha256','input_sha256'):
        assert all(sha(path)==s for path,s in p[kind].items())
    assert not (OUT/'results.json').exists()
    started=time.perf_counter()
    with np.load(BASE/'layer_pooling_v1/features.npz',allow_pickle=False) as z:
        x, qids, groups=[z[k].copy() for k in ('M6','query_ids','group_ids')]
    with np.load(PRIOR/'predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(qids,z['query_ids']) and np.array_equal(groups,z['group_ids'])
        scores=z['arm_scores'].astype(float); fold=z['fold_id'].copy(); utility=z['utility'].copy()
    gap=utility[:,0]-utility[:,1]
    records, arrays=[], {}
    with threadpool_limits(limits=2):
        for f in range(5):
            with np.load(PRIOR/f'fold{f}_targets.npz',allow_pickle=False) as z:
                fit=z['fit_indices'].copy(); targets=z['targets'].copy(); weight=z['weights'].copy(); teacher_probability=z['teacher_probability'].copy()
            with np.load(PRIOR/f'fold{f}_fit_cal.npz',allow_pickle=False) as z:
                assert np.array_equal(fit,z['fit_indices'])
                direct_native=z['fit_scores'][0].copy()
            assert np.array_equal(weight,np.where(abs(gap[fit])>1e-12,abs(gap[fit]),0.))
            active=weight>0; w=weight[active]/weight[active].max(); w/=w.sum()
            heads=[]
            for arm in ARMS:
                with np.load(PRIOR/f'fold{f}_S_{arm}.npz',allow_pickle=False) as z:
                    heads.append(np.r_[z['coef'],float(z['intercept'])])
            head_summaries={arm:dict(beta_norm=float(np.linalg.norm(t[:-1])),bias=float(t[-1]),
                beta_norm_ratio_to_Direct=float(np.linalg.norm(t[:-1])/np.linalg.norm(heads[0][:-1])),
                beta_cosine_to_Direct=cosine(t[:-1],heads[0][:-1])) for arm,t in zip(ARMS,heads)}
            comparisons={}
            u=np.column_stack((x[fit][active].astype(float),np.ones(int(active.sum()))))
            direct_fp64=u@heads[0]
            for name,(a,b) in zip(NAMES,PAIRS):
                summary,vectors=moments(x[fit][active].astype(float),w,targets[a,active],targets[b,active],heads[a],heads[b])
                dq=targets[b,active]-targets[a,active]
                summary['target_difference_covariance_with_Direct_fp64_score']=float(w@((dq-w@dq)*(direct_fp64-w@direct_fp64)))
                assert max(summary[k] for k in ('objective_identity_error','response_identity_error','energy_identity_error'))<1e-11
                assert max(summary['source_gradient_inf'],summary['target_gradient_inf'])<=1e-8
                summary['test_action_changes']=action_changes(scores[a,fold==f],scores[b,fold==f],gap[fold==f])
                assert abs(summary['test_action_changes']['gain']-read(PRIOR/'results.json')['folds'][f]['primary_means'][name])<1e-12
                comparisons[name]=summary
                arrays.update({f'fold{f}_{name}_{key}':value for key,value in vectors.items()})
            shrink_records={}
            for arm_index in (1,2):
                sh,sv=shrinkage(x[fit][active].astype(float),w,targets[0,active],teacher_probability[arm_index-1,active],heads[0],direct_native[active])
                assert sh['decomposition_error']<1e-11
                shrink_records[ARMS[arm_index]]=sh
                arrays.update({f'fold{f}_{ARMS[arm_index]}_shrinkage_{key}':value for key,value in sv.items()})
            records.append(dict(fold=f,fit_positive_weight_rows=int(active.sum()),heads=head_summaries,comparisons=comparisons,shrinkage=shrink_records))
        aggregate={name:action_changes(scores[a],scores[b],gap) for name,(a,b) in zip(NAMES,PAIRS)}
        for name in NAMES:
            assert abs(aggregate[name]['gain']-read(PRIOR/'results.json')['primary'][name]['mean'])<1e-12
        cosines={name:[[cosine(arrays[f'fold{a}_{name}_centered_moment'],arrays[f'fold{b}_{name}_centered_moment']) for b in range(5)] for a in range(5)] for name in NAMES}
    with (OUT/'vectors.npz').open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    result=dict(status='complete_descriptive_localization_pending_separate_check',protocol_sha256=binding,
        folds=records,aggregate=aggregate,centered_moment_cosines=cosines,vectors_sha256=sha(OUT/'vectors.npz'),
        elapsed_seconds=time.perf_counter()-started,scope=p['scope'],new_fits=0,new_policies=0,new_api_calls=0,
        candidate_preparation_gate=False,core_goal_achieved=False)
    for kind in ('source_sha256','input_sha256'):
        assert all(sha(path)==s for path,s in p[kind].items())
    write(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],aggregate=aggregate,elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['self-test','freeze','run']);parser.add_argument('--protocol-sha');args=parser.parse_args()
    if args.mode=='self-test':print(json.dumps(self_test()))
    elif args.mode=='freeze':freeze()
    else:run(args.protocol_sha)
