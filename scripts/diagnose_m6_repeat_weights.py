"""Fixed-M6 own-fit repeat-weight diagnostic; no fitting or policy evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_repeat_weight_diagnostic_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_repeat_weight_diagnostic_plan_20260917.md'
CHECKER = ROOT / 'scripts/check_m6_repeat_weight_diagnostic.py'
EPS, LAMBDA = 1e-12, .001
CONFIG = dict(eps=EPS, regularization=LAMBDA, algebra_atol=1e-11,
              algebra_rtol=1e-10, permutation_atol=1e-12,
              original_stationarity_limit=1e-7, solve_residual_limit=1e-10,
              folds=5, fit_rows=6144, cpu_threads=2)
VECTOR_NAMES = ('extra', 'scale', 'base_gradient', 'penalty', 'directed',
                'uncompensated_extra', 'k_tie_flip', 'k_nontie_flip',
                'k_remaining', 'delta_trunc', 'delta_num')
METRIC_NAMES = ('extra', 'scale', 'base_gradient', 'penalty')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def input_paths():
    paths = [BASE / p for p in ('e08_results/old9600_diagnostic_inputs.npz',
        'e08_independent_checks.json', 'e08_protocol.json',
        'layer_pooling_v1/features.npz', 'layer_pooling_v1/predictions.npz',
        'layer_pooling_v1/completion_record.json', 'e02_results/fold_indices.npz')]
    paths += [BASE / f'layer_pooling_v1/fold{f}_M6{suffix}'
              for f in range(5) for suffix in ('_head.pt', '_fit.npz', '.json')]
    return paths


def load_inputs():
    import torch
    torch.set_num_threads(2)
    check = read(BASE / 'e08_independent_checks.json')
    assert check['status'] == 'passed'
    assert check['protocol_sha256'] == sha(BASE / 'e08_protocol.json')
    assert check['populations']['old9600']['diagnostic_inputs_sha256'] == sha(BASE / 'e08_results/old9600_diagnostic_inputs.npz')
    old = read(BASE / 'layer_pooling_v1/completion_record.json')
    for path in input_paths():
        if path.parent == BASE / 'layer_pooling_v1' and path.name != 'completion_record.json':
            assert str(path) in old['artifact_sha256'], path
        if str(path) in old['artifact_sha256']:
            assert sha(path) == old['artifact_sha256'][str(path)], path
    assert sha(BASE / 'e02_results/fold_indices.npz') == 'ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6'
    with np.load(BASE / 'layer_pooling_v1/features.npz', allow_pickle=False) as z:
        x, qids, groups = (z[k].copy() for k in ('M6', 'query_ids', 'group_ids'))
    assert x.shape == (9600, 384) and x.dtype == np.float32 and np.isfinite(x).all()
    assert len(set(qids)) == 9600 and len(set(groups)) == 9559
    with np.load(BASE / 'e08_results/old9600_diagnostic_inputs.npz', allow_pickle=False) as z:
        assert np.array_equal(qids, z['query_ids']) and np.array_equal(groups, z['group_ids'])
        repeats = z['repeats'].copy()
    assert repeats.shape == (9600, 2, 3) and repeats.dtype == np.float64
    assert np.isfinite(repeats).all() and ((repeats >= 0) & (repeats <= 1)).all()
    with np.load(BASE / 'layer_pooling_v1/predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(qids, z['query_ids']) and np.array_equal(groups, z['group_ids'])
        utility = z['utility'].copy()  # No policy score/action field is accessed.
    assert np.array_equal(repeats.mean(2), utility)
    heads, folds = [], []
    with np.load(BASE / 'e02_results/fold_indices.npz', allow_pickle=False) as z:
        for f in range(5):
            fit, cal, test = (z[f'fold{f}_{name}'].copy() for name in ('fit', 'calibration', 'test'))
            assert [len(v) for v in (fit, cal, test)] == [6144, 1536, 1920]
            assert np.array_equal(np.sort(np.r_[fit, cal, test]), np.arange(9600))
            assert not (set(groups[fit]) & set(groups[cal]) or set(groups[fit]) & set(groups[test]) or set(groups[cal]) & set(groups[test]))
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as fit_file:
                assert np.array_equal(fit, fit_file['fit_indices'])
            record = read(BASE / f'layer_pooling_v1/fold{f}_M6.json')
            assert record['selected_lambda'] == LAMBDA and record['fold'] == f and record['arm'] == 'M6'
            state = torch.load(BASE / f'layer_pooling_v1/fold{f}_M6_head.pt', map_location='cpu', weights_only=True)
            assert state['weight'].shape == (1, 384) and state['bias'].shape == (1,)
            assert state['weight'].dtype == state['bias'].dtype == torch.float32
            heads.append(np.r_[state['weight'].numpy().ravel(), state['bias'].numpy()].astype(np.float64))
            folds.append(fit.astype(np.int64))
    return x, repeats, qids, groups, folds, heads


def quantities(repeats, eps=EPS):
    d = (repeats[:, 0, :, None] - repeats[:, 1, None, :]).reshape(-1, 9)
    g = repeats[:, 0].mean(1) - repeats[:, 1].mean(1)
    td = np.where(abs(d) > eps, d, 0.)
    a = np.where(abs(g) > eps, g, 0.)
    b, c = td.mean(1), abs(td).mean(1)
    delta = b - a
    delta_num = d.mean(1) - g
    delta_trunc = (td - d).mean(1) - (a - g)
    flip = np.any(td > 0, axis=1) & np.any(td < 0, axis=1)
    category = np.where((a == 0) & flip, 0, np.where((a != 0) & flip, 1, 2)).astype(np.int64)
    return dict(d=d, g=g, td=td, a=a, b=b, c=c, k=c-abs(a), delta=delta,
                delta_num=delta_num, delta_trunc=delta_trunc, category=category)


def cosine(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den else None


def ratio(a, b):
    return float(a / b) if b else None


def diagnose(x, repeats, theta, regularization=LAMBDA):
    q = quantities(repeats)
    u = np.column_stack((x.astype(np.float64), np.ones(len(x))))
    s = u @ theta
    p = expit(s)
    h = np.logaddexp(s/2, -s/2)
    a, c, k, delta = (q[name] for name in ('a', 'c', 'k', 'delta'))
    wa = abs(a)
    sa, sr = float(wa.sum()), float(c.sum())
    assert sa > 0 and sr > 0
    rho = sr / sa
    r = theta.copy(); r[-1] = 0.
    reg = regularization * r
    penalty = regularization * float(r @ r) / 2
    base_data = float(wa @ np.logaddexp(0., np.where(q['g'] > 0, -s, s)) / sa)
    pair_loss = abs(q['td']) * np.logaddexp(0., np.where(q['d'] > 0, -s[:, None], s[:, None]))
    rep_data = float(pair_loss.mean(1).sum() / sr)
    base_gradient = u.T @ (wa * (p - (q['g'] > 0)) / sa) + reg
    rep_data_gradient = u.T @ ((abs(q['td']) * (p[:, None] - (q['d'] > 0))).mean(1) / sr)
    extra_row = (k * (p-.5) - delta/2) / sa
    extra = u.T @ extra_row
    scale = (rho-1) * reg
    vectors = dict(extra=extra, scale=scale, base_gradient=base_gradient,
        penalty=reg, directed=u.T @ a / (2*sa), uncompensated_extra=extra+scale,
        delta_trunc=-u.T @ q['delta_trunc']/(2*sa),
        delta_num=-u.T @ q['delta_num']/(2*sa))
    for index, name in enumerate(('k_tie_flip', 'k_nontie_flip', 'k_remaining')):
        vectors[name] = u.T @ np.where(q['category'] == index, k*(p-.5)/sa, 0.)
    diag = np.diag(np.r_[np.full(x.shape[1], regularization), 0.])
    ha = (u.T * (wa/sa*p*(1-p))) @ u + diag
    hr = (u.T * (c/sr*p*(1-p))) @ u
    hk = (u.T * (k/sa*p*(1-p))) @ u
    errors = dict(
        cross9_mean=float(np.max(abs(q['d'].mean(1)-q['g']))),
        delta_decomposition=float(np.max(abs(delta-q['delta_num']-q['delta_trunc']))),
        compensated_loss=abs(rho*(rep_data+penalty/rho)-(base_data+penalty)-float(np.mean(k*h-delta*s/2)*len(x)/sa)),
        uncompensated_loss=abs(rho*(rep_data+penalty)-(base_data+penalty)-float(np.sum(k*h-delta*s/2)/sa)-(rho-1)*penalty),
        compensated_gradient=float(np.max(abs(rho*(rep_data_gradient+reg/rho)-base_gradient-extra))),
        uncompensated_gradient=float(np.max(abs(rho*(rep_data_gradient+reg)-base_gradient-extra-scale))),
        gradient_components=float(np.max(abs(extra-sum(vectors[n] for n in ('k_tie_flip','k_nontie_flip','k_remaining','delta_trunc','delta_num'))))),
        compensated_hessian=float(np.max(abs(rho*(hr+diag/rho)-ha-hk))))
    assert max(errors.values()) <= CONFIG['algebra_atol']
    eig = np.linalg.eigvalsh(ha)
    assert eig[0] > 0
    rhs = np.column_stack([vectors[name] for name in METRIC_NAMES])
    chol = np.linalg.cholesky(ha)
    response = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    solve_error = float(np.max(abs(ha @ response-rhs)))
    assert solve_error <= CONFIG['solve_residual_limit']
    gram = rhs.T @ response
    assert np.max(abs(gram-gram.T)) <= CONFIG['algebra_atol']
    norms = {name: float(np.linalg.norm(vectors[name])) for name in VECTOR_NAMES}
    row_norm_sum = float(abs(extra_row) @ np.linalg.norm(u, axis=1))
    penalty_norm = norms['penalty']
    orth = extra - (extra @ reg)/(reg @ reg)*reg if reg @ reg else extra.copy()
    stats = dict(rows=len(x), SA=sa, SR=sr, rho=rho, compensated_lambda=regularization/rho,
        category_counts=[int(np.sum(q['category'] == j)) for j in range(3)],
        nonzero_weight_rows=int(np.count_nonzero(a)), repeat_weight_rows=int(np.count_nonzero(c)),
        k_sum=float(k.sum()), k_min=float(k.min()), k_max=float(k.max()),
        k_positive_rows=int(np.sum(k>EPS)), k_negative_rows=int(np.sum(k < -EPS)),
        delta_max_abs=float(np.max(abs(delta))), delta_num_max_abs=float(np.max(abs(q['delta_num']))),
        delta_trunc_max_abs=float(np.max(abs(q['delta_trunc']))),
        loss_A=base_data+penalty, loss_R_compensated=rep_data+penalty/rho,
        loss_R_uncompensated=rep_data+penalty,
        gradient_norms=norms, original_gradient_inf=float(np.max(abs(base_gradient))),
        extra_to_penalty_norm=ratio(norms['extra'], penalty_norm),
        scale_to_penalty_norm=ratio(norms['scale'], penalty_norm),
        extra_to_directed_norm=ratio(norms['extra'], norms['directed']),
        extra_scale_cosine=cosine(extra, scale), extra_penalty_cosine=cosine(extra,reg),
        extra_orthogonal_to_penalty_norm=float(np.linalg.norm(orth)),
        extra_orthogonal_fraction=ratio(np.linalg.norm(orth),norms['extra']),
        per_query_extra_norm_sum=row_norm_sum,
        aggregate_to_sum_query_norm=ratio(norms['extra'],row_norm_sum),
        hessian_min_eigenvalue=float(eig[0]), hessian_max_eigenvalue=float(eig[-1]),
        hessian_condition=float(eig[-1]/eig[0]), solve_residual_inf=solve_error,
        metric_names=list(METRIC_NAMES), inverse_hessian_gram=gram.tolist(),
        identity_errors=errors)
    arrays = {name:q[name] for name in ('g','a','b','c','k','delta','delta_num','delta_trunc','category')}
    arrays.update(scores=s, theta=theta, vectors=np.stack([vectors[n] for n in VECTOR_NAMES]),
                  metric_solutions=response, inverse_hessian_gram=gram)
    return stats, arrays


def permutations_check(repeats):
    original = quantities(repeats)
    maximum = 0.
    for b_order in itertools.permutations(range(3)):
        for d_order in itertools.permutations(range(3)):
            permuted = repeats.copy()
            permuted[:,0] = repeats[:,0,b_order]
            permuted[:,1] = repeats[:,1,d_order]
            q = quantities(permuted)
            for name in ('g','a','b','c','k','delta'):
                maximum = max(maximum, float(np.max(abs(q[name]-original[name]))))
            assert np.array_equal(q['category'], original['category'])
    assert maximum <= CONFIG['permutation_atol']
    return dict(permutations=36, maximum_scalar_array_error=maximum, categories_exact=True)


def self_test():
    repeats = np.array([[[.8,.9,1.],[.1,.2,.3]], [[0.,1.,.5],[.5,.5,.5]],
        [[.1,.8,.7],[.4,.3,.8]], [[2*EPS,EPS,EPS],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]])
    q = quantities(repeats)
    assert abs(q['k'][0]) < 1e-14 and q['k'][1] > 0 and q['a'][1] == 0
    assert q['k'][3] < 0 and abs(q['c'][3]/abs(q['a'][3])-.5) < 1e-14
    x = np.array([[1.,0.],[0.,1.],[.5,.5],[-1.,0.],[0.,-1.]])
    with threadpool_limits(limits=2):
        row, arrays = diagnose(x,repeats,np.array([.3,-.7,.2]))
        _, extreme = diagnose(x,repeats,np.array([100.,-100.,30.]))
    assert arrays['vectors'][VECTOR_NAMES.index('penalty'),-1] == 0
    assert np.isfinite(extreme['scores']).all()
    step = 1e-5
    fd = []
    for j in range(3):
        plus=np.array([.3,-.7,.2]); minus=plus.copy(); plus[j]+=step; minus[j]-=step
        a,_=diagnose(x,repeats,plus); b,_=diagnose(x,repeats,minus)
        fd.append((a['loss_A']-b['loss_A'])/(2*step))
    error=float(np.max(abs(np.asarray(fd)-arrays['vectors'][VECTOR_NAMES.index('base_gradient')])))
    assert error<1e-9
    permutations=permutations_check(repeats)
    return dict(status='passed_repeat_weight_synthetic_checks',identity_max_error=max(row['identity_errors'].values()),
        finite_difference_gradient_error=error,negative_k_and_rho_below_one=True,
        permutation_checks=permutations,real_data_reads=0,new_model_fits=0)


def freeze():
    assert not OUT.exists(), 'Keep the original diagnostic run immutable'
    synthetic=self_test()
    x,repeats,qids,groups,folds,heads=load_inputs()
    paths=input_paths()
    sources=[Path(__file__).resolve(),CHECKER,PLAN]
    assert all(p.is_file() for p in sources+paths)
    OUT.mkdir(parents=True)
    p=dict(status='frozen_before_real_repeat_weight_gradient_diagnostic',created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=CONFIG,source_sha256={str(p):sha(p) for p in sources},input_sha256={str(p):sha(p) for p in paths},
        vector_names=list(VECTOR_NAMES),metric_names=list(METRIC_NAMES),synthetic_checks=synthetic,
        identity=dict(queries=len(qids),groups=len(set(groups)),repeat_means_exact=True,query_group_order_exact=True),
        scope='Consumed old9600 fixed original M6 own-fit descriptive diagnostic; ideal FP64 objective only',
        new_model_fits=0,new_policy_evaluations=0,new_encoder_forwards=0,new_api_calls=0)
    write(OUT/'protocol.json',p)
    print(json.dumps(dict(status=p['status'],protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def bound(binding):
    assert sha(OUT/'protocol.json')==binding
    p=read(OUT/'protocol.json'); assert p['config']==CONFIG
    for key in ('source_sha256','input_sha256'):
        for path,expected in p[key].items(): assert sha(path)==expected,path
    return p


def run(binding):
    protocol=bound(binding)
    assert not (OUT/'results.json').exists() and not (OUT/'diagnostics.npz').exists()
    started=time.perf_counter()
    x,repeats,qids,groups,folds,heads=load_inputs()
    records,arrays=[],{}
    with threadpool_limits(limits=2):
        for f,fit in enumerate(folds):
            row,values=diagnose(x[fit],repeats[fit],heads[f])
            assert row['original_gradient_inf']<=CONFIG['original_stationarity_limit']
            row.update(fold=f,fit_groups=len(set(groups[fit])),permutation_checks=permutations_check(repeats[fit]))
            records.append(row)
            values.update(fit_indices=fit)
            arrays.update({f'fold{f}_{name}':value for name,value in values.items()})
    bound(binding)
    with (OUT/'diagnostics.npz').open('xb') as stream: np.savez_compressed(stream,**arrays)
    result=dict(status='complete_fixed_M6_repeat_weight_diagnostic_pending_independent_check',
        protocol_sha256=binding,arrays_sha256=sha(OUT/'diagnostics.npz'),folds=records,
        vector_names=list(VECTOR_NAMES),metric_names=list(METRIC_NAMES),
        diagnostic_factorizations=5,linear_right_hand_sides=20,
        new_model_fits=0,new_policy_evaluations=0,new_encoder_forwards=0,new_api_calls=0,
        runtime_deployed=False,core_goal_achieved=False,elapsed_seconds=time.perf_counter()-started,
        scope=protocol['scope'])
    write(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],
        folds=[{k:r[k] for k in ('fold','rho','original_gradient_inf','extra_to_penalty_norm','extra_scale_cosine','aggregate_to_sum_query_norm')} for r in records])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('self-test','freeze','run'))
    parser.add_argument('--protocol-sha256'); args=parser.parse_args()
    if args.mode=='self-test': print(json.dumps(self_test()))
    elif args.mode=='freeze': freeze()
    else: run(args.protocol_sha256)
