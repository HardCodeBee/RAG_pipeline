"""Independent CPU checks for the fixed-M6 repeat-weight diagnostic.

No runner is imported. Nine hard-BCE terms supply an independent reference;
local curvature is checked without constructing or evaluating a new head.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_repeat_weight_diagnostic_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_repeat_weight_diagnostic_plan_20260917.md'
RUNNER = ROOT / 'scripts/diagnose_m6_repeat_weights.py'
EPS = 1e-12
LAMBDA = .001
ATOL, RTOL = 1e-11, 1e-10
FOLD_SHA = 'ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6'
VECTOR_NAMES = ('extra', 'scale', 'base_gradient', 'penalty', 'directed',
    'uncompensated_extra', 'k_tie_flip', 'k_nontie_flip', 'k_remaining',
    'delta_trunc', 'delta_num')
METRIC_NAMES = ('extra', 'scale', 'base_gradient', 'penalty')
CONFIG = dict(eps=EPS, regularization=LAMBDA, algebra_atol=ATOL,
    algebra_rtol=RTOL, permutation_atol=1e-12, original_stationarity_limit=1e-7,
    solve_residual_limit=1e-10, folds=5, fit_rows=6144, cpu_threads=2)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bindings(mapping):
    assert isinstance(mapping, dict) and mapping
    canonical = {Path(p).resolve(): v for p, v in mapping.items()}
    assert len(canonical) == len(mapping)
    for path, expected in canonical.items():
        assert isinstance(expected, str) and len(expected) == 64
        assert sha(path) == expected, str(path)
    return canonical


def close(actual, expected, errors, name, atol=ATOL, rtol=RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, name + ' shape'
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name + ' finite'
    delta = np.abs(actual.astype(float) - expected.astype(float))
    assert np.all(delta <= atol + rtol * np.abs(expected)), name + ' numerical mismatch'
    errors[name] = float(np.max(delta, initial=0.))


def sigmoid(scores):
    values = np.empty_like(scores, dtype=np.float64)
    positive = scores >= 0
    values[positive] = 1 / (1 + np.exp(-scores[positive]))
    ex = np.exp(scores[~positive])
    values[~positive] = ex / (1 + ex)
    return values


def row_reference(repeats, eps=EPS):
    """Use explicit scalar summation across the nine pairs."""
    repeats = np.asarray(repeats, dtype=np.float64)
    assert repeats.ndim == 3 and repeats.shape[1:] == (2, 3)
    assert np.isfinite(repeats).all()
    assert np.all((repeats >= 0) & (repeats <= 1))
    # Keep the original mean-of-each-action contract; independent pair sums
    # explicitly expose the reduction difference instead of assuming zero.
    g = repeats[:, 0].mean(1) - repeats[:, 1].mean(1)
    pairs = np.column_stack([repeats[:, 0, i] - repeats[:, 1, j]
                             for i in range(3) for j in range(3)])
    truncated = np.where(abs(pairs) > eps, pairs, 0.)
    a = np.where(abs(g) > eps, g, 0.)
    average = lambda table: np.asarray([math.fsum(map(float, row)) / 9 for row in table])
    b, c = average(truncated), average(abs(truncated))
    numerical = average(pairs) - g
    truncation = average(truncated - pairs) - (a - g)
    both = (truncated.max(1) > 0) & (truncated.min(1) < 0)
    tie = a == 0
    groups = np.column_stack([tie & both, ~tie & both, ~both])
    assert np.all(groups.sum(1) == 1)
    return dict(g=g, pairs=pairs, truncated_pairs=truncated, a=a, b=b, c=c,
        k=c - abs(a), delta=b - a, delta_num=numerical,
        delta_trunc=truncation, groups=groups)


def hard_reference(U, theta, repeats, lam=LAMBDA, eps=EPS):
    U, theta = np.asarray(U, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    assert U.ndim == 2 and theta.shape == (U.shape[1],) and len(U) == len(repeats)
    assert np.isfinite(U).all() and np.isfinite(theta).all() and np.all(U[:, -1] == 1)
    rows = row_reference(repeats, eps)
    scores = U @ theta
    p = sigmoid(scores)
    w = abs(rows['a'])
    SA = math.fsum(map(float, w)); SR = math.fsum(map(float, rows['c']))
    assert SA > 0 and SR > 0, 'Positive own-fit normalizers required'
    rho = SR / SA
    signed = np.where(rows['g'] > 0, -scores, scores)
    loss_A_data = math.fsum(map(float, w * np.logaddexp(0., signed))) / SA
    base_row_gradient = w * (p - (rows['g'] > 0)) / SA
    # Compute loss and derivative from each hard-label pair separately. No
    # soft-target or k/delta expression enters this direct reference.
    pair_losses, pair_gradients, pair_hessians = [], [], []
    for j in range(9):
        pair = rows['truncated_pairs'][:, j]
        weight = abs(pair)
        target = rows['pairs'][:, j] > 0
        pair_losses.append(weight * np.logaddexp(0., np.where(target, -scores, scores)))
        pair_gradients.append(weight * (p - target))
        pair_hessians.append(weight * p * (1 - p))
    aggregate = lambda items: np.asarray([math.fsum(map(float, row)) / 9
                                         for row in np.stack(items, axis=1)])
    direct_R_rows = aggregate(pair_gradients) / SA
    direct_R_hessian_rows = aggregate(pair_hessians) / SA
    loss_R_scaled_data = math.fsum(map(float, aggregate(pair_losses))) / SA
    reg = lam * theta.copy(); reg[-1] = 0
    penalty = .5 * lam * math.fsum(float(x) ** 2 for x in theta[:-1])
    direct_A_gradient = U.T @ base_row_gradient + reg
    direct_R_gradient = U.T @ direct_R_rows + reg
    even = rows['k'] * (p - .5) / SA
    trunc = -rows['delta_trunc'] / (2 * SA)
    numerical = -rows['delta_num'] / (2 * SA)
    full = (rows['k'] * (p - .5) - rows['delta'] / 2) / SA
    Gextra = U.T @ full
    Gscale = (rho - 1) * reg
    Glab = U.T @ (rows['a'] / (2 * SA))
    group_vectors = np.stack([U.T @ np.where(rows['groups'][:, j], even, 0.)
                              for j in range(3)])
    D = np.eye(len(theta)) * lam; D[-1, -1] = 0.
    HA = (U.T * (w * p * (1 - p) / SA)) @ U + D
    HR_scaled = (U.T * direct_R_hessian_rows) @ U + D
    Hextra = (U.T * (rows['k'] * p * (1 - p) / SA)) @ U
    h = np.logaddexp(scores / 2, -scores / 2)
    expanded_difference = math.fsum(map(float, rows['k'] * h - rows['delta'] * scores / 2)) / SA
    vectors = dict(Gextra=Gextra, Gscale=Gscale, Gsum=Gextra + Gscale,
        original_gradient=direct_A_gradient, original_regularizer=reg,
        label_moment=Glab, group_even=group_vectors,
        trunc_gradient=U.T @ trunc, numerical_gradient=U.T @ numerical)
    return dict(rows=rows, U=U, theta=theta, scores=scores, probabilities=p,
        SA=SA, SR=SR, rho=rho, lambda_compensated=lam/rho,
        loss_A=loss_A_data + penalty, loss_R_scaled=loss_R_scaled_data + penalty,
        loss_difference=expanded_difference, direct_A_gradient=direct_A_gradient,
        direct_R_scaled_gradient=direct_R_gradient, HA=HA, HR_scaled=HR_scaled,
        Hextra=Hextra, vectors=vectors, extra_row_coefficients=full,
        even_row_coefficients=even, trunc_row_coefficients=trunc,
        numerical_row_coefficients=numerical)


def check_algebra(reference, errors, prefix):
    r = reference
    close(r['rows']['delta'], r['rows']['delta_num'] + r['rows']['delta_trunc'], errors, prefix+'/delta_split')
    close(r['loss_R_scaled'] - r['loss_A'], r['loss_difference'], errors, prefix+'/loss')
    close(r['direct_R_scaled_gradient'] - r['direct_A_gradient'], r['vectors']['Gextra'], errors, prefix+'/gradient')
    close(r['HR_scaled'] - r['HA'], r['Hextra'], errors, prefix+'/Hessian')
    v = r['vectors']
    close(v['group_even'].sum(0) + v['trunc_gradient'] + v['numerical_gradient'], v['Gextra'], errors, prefix+'/group_decomposition')
    close(r['HR_scaled'], r['HR_scaled'].T, errors, prefix+'/repeat_Hessian_symmetry')


def curvature(reference):
    """Eigensystem reference, distinct from the runner's Cholesky solve."""
    H = reference['HA']
    values, basis = np.linalg.eigh(H)
    assert np.isfinite(values).all() and values[0] > 0
    vectors = reference['vectors']
    rhs = np.column_stack([vectors[n] for n in
        ('Gextra', 'Gscale', 'original_gradient', 'original_regularizer')])
    solved = basis @ ((basis.T @ rhs) / values[:, None])
    residual = float(np.max(abs(H @ solved - rhs)))
    assert residual <= 1e-10
    gram = rhs.T @ solved
    return dict(eigenvalues=values, gram=gram, rhs=rhs, metric_solutions=solved,
        minimum_eigenvalue=float(values[0]), maximum_eigenvalue=float(values[-1]),
        condition_number=float(values[-1] / values[0]), solve_residual=residual)


def permutation_check(repeats, errors, prefix):
    original = row_reference(repeats)
    maximum = 0.
    for bp in itertools.permutations(range(3)):
        for dp in itertools.permutations(range(3)):
            changed = np.stack([repeats[:, 0, list(bp)], repeats[:, 1, list(dp)]], axis=1)
            candidate = row_reference(changed)
            for name in ('g', 'a', 'b', 'c', 'k', 'delta', 'delta_num', 'delta_trunc'):
                close(candidate[name], original[name], errors, prefix+'/'+str(bp)+str(dp)+'/'+name,
                    atol=1e-12, rtol=0.)
                maximum = max(maximum, float(np.max(abs(candidate[name] - original[name]), initial=0.)))
            assert np.array_equal(candidate['groups'], original['groups'])
    return maximum


def self_test():
    repeated = np.array([
        [[.7,.8,.9],[.1,.2,.3]], [[.1,.2,.3],[.7,.8,.9]],
        [[0.,.5,1.],[0.,.5,1.]], [[.2,.8,1.],[.1,.3,.9]],
        [[2*EPS,EPS,EPS],[0.,0.,0.]], [[.49,.5,.51],[.5,.5,.5]],
        [[0.,0.,0.],[0.,0.,0.]]], dtype=np.float64)
    rng = np.random.default_rng(2026091723)
    U = np.column_stack([rng.normal(size=(len(repeated),4)),np.ones(len(repeated))])
    theta = rng.normal(size=5)
    errors = {}
    r = hard_reference(U,theta,repeated)
    check_algebra(r, errors, 'synthetic')
    spectral = curvature(r)
    assert np.all(spectral['gram'].diagonal() >= 0)
    assert r['vectors']['original_regularizer'][-1] == 0.
    assert r['rows']['groups'][2,0] and r['rows']['groups'][3,1]
    negative = hard_reference(np.array([[1.,1.]]),np.array([.4,.7]),repeated[4:5])
    assert negative['rho'] < 1 and negative['rows']['k'][0] < 0
    check_algebra(negative, errors, 'negative_k')
    permutation_error = permutation_check(repeated,errors,'permutation')
    finite_difference = 0.
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = 1e-5
        plus = hard_reference(U,theta+delta,repeated)
        minus = hard_reference(U,theta-delta,repeated)
        observed = (plus['loss_A']-minus['loss_A'])/(2e-5)
        finite_difference = max(finite_difference, abs(observed-r['direct_A_gradient'][j]))
        close((plus['direct_A_gradient']-minus['direct_A_gradient'])/(2e-5),r['HA'][:,j],errors,'finite_Hessian_'+str(j),atol=2e-10)
    assert finite_difference < 2e-10
    for offset in (-1000.,1000.):
        extreme = theta.copy(); extreme[-1] = offset
        x = hard_reference(U,extreme,repeated)
        check_algebra(x,errors,'extreme_'+str(offset))
    rejected = False
    try:
        hard_reference(U[:1],theta,np.zeros((1,2,3)))
    except AssertionError:
        rejected = True
    assert rejected
    return dict(status='passed_independent_repeat_weight_synthetic_checks',
        hard_pair_algebra_max_error=max(errors.values()),finite_difference_max_error=finite_difference,
        all_36_permutations_max_error=permutation_error,negative_k_and_rho_below_one=True,
        zero_weight_normalizer_rejected=True,unpenalized_bias=True,
        real_data_reads=0,GPU_forwards=0,new_fits=0,files_written=0)


def expected_inputs():
    base = ROOT.parent / 'work/router_research'
    paths = [base / name for name in ('e08_results/old9600_diagnostic_inputs.npz',
        'e08_independent_checks.json', 'e08_protocol.json',
        'layer_pooling_v1/features.npz', 'layer_pooling_v1/predictions.npz',
        'layer_pooling_v1/completion_record.json', 'e02_results/fold_indices.npz')]
    paths += [base / f'layer_pooling_v1/fold{f}_M6{suffix}' for f in range(5)
              for suffix in ('_head.pt', '_fit.npz', '.json')]
    return paths


def load_inputs(protocol):
    """Only identities/utility and own-fit data enter new calculations."""
    import torch
    torch.set_num_threads(2)
    mapping = bindings(protocol['input_sha256'])
    assert set(mapping) == {p.resolve() for p in expected_inputs()}
    assert len(mapping) == 22
    by_name = {p.name: p for p in mapping}
    assert len(by_name) == len(mapping)
    e08 = read(by_name['e08_independent_checks.json'])
    assert e08['status'] == 'passed'
    assert e08['protocol_sha256'] == sha(by_name['e08_protocol.json'])
    assert e08['populations']['old9600']['diagnostic_inputs_sha256'] == sha(by_name['old9600_diagnostic_inputs.npz'])
    historical = {Path(p).resolve(): digest for p,digest in
                  read(by_name['completion_record.json'])['artifact_sha256'].items()}
    for path in mapping:
        if path.parent.name == 'layer_pooling_v1' and path.name != 'completion_record.json':
            assert path in historical and sha(path) == historical[path], str(path)
    assert sha(by_name['fold_indices.npz']) == FOLD_SHA
    with np.load(by_name['features.npz'],allow_pickle=False) as archive:
        features, qids, groups = (archive[n].copy() for n in ('M6','query_ids','group_ids'))
    assert features.shape == (9600,384) and features.dtype == np.float32
    assert np.isfinite(features).all()
    assert qids.shape == groups.shape == (9600,)
    assert qids.dtype.kind in 'US' and groups.dtype.kind in 'US'
    assert len(set(qids)) == 9600 and len(set(groups)) == 9559
    assert np.max(abs(np.linalg.norm(features.astype(float),axis=1)-1)) < 2e-6
    with np.load(by_name['old9600_diagnostic_inputs.npz'],allow_pickle=False) as archive:
        assert np.array_equal(qids,archive['query_ids']) and np.array_equal(groups,archive['group_ids'])
        repeats = archive['repeats'].copy()
    assert repeats.shape == (9600,2,3) and repeats.dtype == np.float64
    assert np.isfinite(repeats).all() and np.all((repeats>=0)&(repeats<=1))
    with np.load(by_name['predictions.npz'],allow_pickle=False) as archive:
        assert np.array_equal(qids,archive['query_ids']) and np.array_equal(groups,archive['group_ids'])
        utility = archive['utility'].copy()
    assert utility.shape == (9600,2) and utility.dtype == np.float64
    assert np.array_equal(repeats.mean(2),utility)
    fits,heads,visit = [],[],np.zeros(9600,dtype=np.int64)
    with np.load(by_name['fold_indices.npz'],allow_pickle=False) as archive:
        for f in range(5):
            parts = [archive[f'fold{f}_{n}'].copy() for n in ('fit','calibration','test')]
            assert [len(v) for v in parts] == [6144,1536,1920]
            assert all(v.ndim==1 and v.dtype.kind in 'iu' and len(set(v))==len(v) for v in parts)
            assert np.array_equal(np.sort(np.concatenate(parts)),np.arange(9600))
            group_sets = [set(groups[v]) for v in parts]
            assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3))
            fit,_,test = parts
            visit[test] += 1
            with np.load(by_name[f'fold{f}_M6_fit.npz'],allow_pickle=False) as fitted:
                assert np.array_equal(fit,fitted['fit_indices'])
            record = read(by_name[f'fold{f}_M6.json'])
            assert (record['fold'],record['arm'],record['selected_lambda']) == (f,'M6',LAMBDA)
            head = torch.load(by_name[f'fold{f}_M6_head.pt'],map_location='cpu',weights_only=True)
            assert set(head) == {'weight','bias'}
            assert head['weight'].shape == (1,384) and head['bias'].shape == (1,)
            assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in head.values())
            theta = np.concatenate([head['weight'].numpy().ravel(),head['bias'].numpy()]).astype(np.float64)
            fits.append(fit.astype(np.int64)); heads.append(theta)
    assert np.all(visit==1)
    return dict(features=features,repeats=repeats,query_ids=qids,group_ids=groups,fits=fits,heads=heads)


def scalar_reference(r,curved):
    v,q = r['vectors'],r['rows']
    vec = dict(extra=v['Gextra'],scale=v['Gscale'],base_gradient=v['original_gradient'],
        penalty=v['original_regularizer'],directed=v['label_moment'],
        uncompensated_extra=v['Gsum'],k_tie_flip=v['group_even'][0],
        k_nontie_flip=v['group_even'][1],k_remaining=v['group_even'][2],
        delta_trunc=v['trunc_gradient'],delta_num=v['numerical_gradient'])
    norms = {name:float(np.linalg.norm(vec[name])) for name in VECTOR_NAMES}
    ratio = lambda a,b: float(a/b) if b else None
    cosine = lambda a,b: ratio(float(a@b),float(np.linalg.norm(a)*np.linalg.norm(b)))
    reg,extra = vec['penalty'],vec['extra']
    projection = float(extra@reg)/float(reg@reg)*reg if reg@reg else np.zeros_like(extra)
    orth = extra-projection
    row_sum = math.fsum(abs(float(t))*math.sqrt(math.fsum(float(x)**2 for x in row))
                       for t,row in zip(r['extra_row_coefficients'],r['U']))
    penalty = .5*LAMBDA*math.fsum(float(x)**2 for x in r['theta'][:-1])
    stats = dict(rows=len(r['U']),SA=r['SA'],SR=r['SR'],rho=r['rho'],compensated_lambda=LAMBDA/r['rho'],
        category_counts=q['groups'].sum(0).astype(int).tolist(),nonzero_weight_rows=int(np.count_nonzero(q['a'])),
        repeat_weight_rows=int(np.count_nonzero(q['c'])),k_sum=math.fsum(map(float,q['k'])),
        k_min=float(q['k'].min()),k_max=float(q['k'].max()),
        k_positive_rows=int(np.sum(q['k']>EPS)),k_negative_rows=int(np.sum(q['k']< -EPS)),
        delta_max_abs=float(np.max(abs(q['delta']))),delta_num_max_abs=float(np.max(abs(q['delta_num']))),
        delta_trunc_max_abs=float(np.max(abs(q['delta_trunc']))),loss_A=r['loss_A'],
        loss_R_compensated=r['loss_R_scaled']/r['rho'],
        loss_R_uncompensated=(r['loss_R_scaled']-penalty)/r['rho']+penalty,
        gradient_norms=norms,original_gradient_inf=float(np.max(abs(vec['base_gradient']))),
        extra_to_penalty_norm=ratio(norms['extra'],norms['penalty']),scale_to_penalty_norm=ratio(norms['scale'],norms['penalty']),
        extra_to_directed_norm=ratio(norms['extra'],norms['directed']),
        extra_scale_cosine=cosine(extra,vec['scale']),extra_penalty_cosine=cosine(extra,reg),
        extra_orthogonal_to_penalty_norm=float(np.linalg.norm(orth)),
        extra_orthogonal_fraction=ratio(float(np.linalg.norm(orth)),norms['extra']),
        per_query_extra_norm_sum=row_sum,aggregate_to_sum_query_norm=ratio(norms['extra'],row_sum),
        hessian_min_eigenvalue=curved['minimum_eigenvalue'],hessian_max_eigenvalue=curved['maximum_eigenvalue'],
        hessian_condition=curved['condition_number'],metric_names=list(METRIC_NAMES),
        inverse_hessian_gram=curved['gram'].tolist())
    arrays = {name:q[name] for name in ('g','a','b','c','k','delta','delta_num','delta_trunc')}
    arrays.update(category=np.argmax(q['groups'],axis=1).astype(np.int64),scores=r['scores'],theta=r['theta'],
        vectors=np.stack([vec[name] for name in VECTOR_NAMES]),metric_solutions=curved['metric_solutions'],
        inverse_hessian_gram=curved['gram'])
    return stats,arrays


def compare_tree(actual,expected,errors,name):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),name+' keys'
        for key in expected:
            compare_tree(actual[key],expected[key],errors,name+'/'+key)
    elif expected is None or isinstance(expected,(str,bool)):
        assert actual==expected and type(actual) is type(expected),name
    elif isinstance(expected,int):
        assert isinstance(actual,int) and not isinstance(actual,bool) and actual==expected,name
    elif isinstance(expected,list) and all(isinstance(v,str) for v in expected):
        assert actual==expected,name
    else:
        close(actual,expected,errors,name)


def check(binding):
    started = time.perf_counter()
    destination = OUT/'separate_checks.json'
    assert not destination.exists(),'Preserve the existing independent receipt'
    assert binding and sha(OUT/'protocol.json')==binding
    protocol = read(OUT/'protocol.json')
    assert protocol['status']=='frozen_before_real_repeat_weight_gradient_diagnostic'
    assert protocol['config']==CONFIG
    assert protocol['vector_names']==list(VECTOR_NAMES) and protocol['metric_names']==list(METRIC_NAMES)
    source = bindings(protocol['source_sha256'])
    assert set(source)=={Path(__file__).resolve(),RUNNER.resolve(),PLAN.resolve()}
    checker_hash = sha(__file__)
    assert protocol['identity']==dict(queries=9600,groups=9559,repeat_means_exact=True,query_group_order_exact=True)
    for name in ('new_model_fits','new_policy_evaluations','new_encoder_forwards','new_api_calls'):
        assert protocol[name]==0
    d = load_inputs(protocol)
    result = read(OUT/'results.json')
    result_hash,arrays_hash = sha(OUT/'results.json'),sha(OUT/'diagnostics.npz')
    assert result['status']=='complete_fixed_M6_repeat_weight_diagnostic_pending_independent_check'
    assert result['protocol_sha256']==binding and result['arrays_sha256']==arrays_hash
    assert result['scope']==protocol['scope']
    assert result['vector_names']==list(VECTOR_NAMES) and result['metric_names']==list(METRIC_NAMES)
    assert result['diagnostic_factorizations']==5 and result['linear_right_hand_sides']==20
    assert len(result['folds'])==5
    for name in ('new_model_fits','new_policy_evaluations','new_encoder_forwards','new_api_calls'):
        assert result[name]==0
    assert result['runtime_deployed'] is result['core_goal_achieved'] is False
    assert math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds']>0
    saved_names = ('g','a','b','c','k','delta','delta_num','delta_trunc','category','scores',
                   'theta','vectors','metric_solutions','inverse_hessian_gram','fit_indices')
    errors,reports = {},[]
    with np.load(OUT/'diagnostics.npz',allow_pickle=False) as saved:
        assert set(saved.files)=={f'fold{f}_{n}' for f in range(5) for n in saved_names}
        for f,fit in enumerate(d['fits']):
            U = np.column_stack([d['features'][fit].astype(np.float64),np.ones(len(fit))])
            r = hard_reference(U,d['heads'][f],d['repeats'][fit])
            check_algebra(r,errors,f'fold{f}/independent_hard_pair')
            curved = curvature(r)
            stats,arrays = scalar_reference(r,curved)
            stats.update(fold=f,fit_groups=len(set(d['group_ids'][fit])))
            actual = result['folds'][f]
            assert set(actual)==set(stats)|{'solve_residual_inf','identity_errors','permutation_checks'}
            compare_tree({k:actual[k] for k in stats},stats,errors,f'fold{f}/stats')
            assert actual['original_gradient_inf']<=1e-7 and stats['original_gradient_inf']<=1e-7
            arrays['fit_indices']=fit
            for name,expected in arrays.items():
                observed=saved[f'fold{f}_{name}']
                assert observed.dtype==expected.dtype and observed.shape==expected.shape,name
                if expected.dtype.kind in 'iu':
                    assert np.array_equal(observed,expected),name
                else:
                    close(observed,expected,errors,f'fold{f}/array/{name}')
            saved_solutions=saved[f'fold{f}_metric_solutions']
            residual=float(np.max(abs(r['HA']@saved_solutions-curved['rhs'])))
            assert math.isfinite(actual['solve_residual_inf']) and 0<=actual['solve_residual_inf']<=1e-10
            assert residual<=1e-10
            close(actual['solve_residual_inf'],residual,errors,f'fold{f}/saved_solve_residual')
            close(saved[f'fold{f}_inverse_hessian_gram'],curved['rhs'].T@saved_solutions,errors,f'fold{f}/saved_Gram')
            identity_names={'cross9_mean','delta_decomposition','compensated_loss','uncompensated_loss',
                'compensated_gradient','uncompensated_gradient','gradient_components','compensated_hessian'}
            assert set(actual['identity_errors'])==identity_names
            assert all(math.isfinite(v) and 0<=v<=ATOL for v in actual['identity_errors'].values())
            # The independent hard-pair evaluation above checks the identities;
            # their machine-roundoff maxima need not equal another summation order.
            rho=r['rho']; penalty=r['vectors']['original_regularizer']
            close(rho*((r['direct_R_scaled_gradient']-penalty)/rho+penalty)-r['direct_A_gradient'],
                r['vectors']['Gsum'],errors,f'fold{f}/uncompensated_gradient')
            penval=.5*LAMBDA*float(r['theta'][:-1]@r['theta'][:-1])
            close(rho*stats['loss_R_uncompensated']-stats['loss_A'],
                r['loss_difference']+(rho-1)*penval,errors,f'fold{f}/uncompensated_loss')
            permutation_error=permutation_check(d['repeats'][fit],errors,f'fold{f}/permutation')
            item=actual['permutation_checks']
            assert set(item)=={'permutations','maximum_scalar_array_error','categories_exact'}
            assert item['permutations']==36 and item['categories_exact'] is True
            assert math.isfinite(item['maximum_scalar_array_error']) and 0<=item['maximum_scalar_array_error']<=1e-12
            reports.append(dict(fold=f,own_fit_rows=len(fit),fit_groups=stats['fit_groups'],
                saved_arrays_checked=15,saved_vectors_checked=11,hard_pair_loss_gradient_Hessian_checked=True,
                original_gradient_inf=stats['original_gradient_inf'],spectral_solve_residual=curved['solve_residual'],
                saved_solution_residual=residual,permutations_checked=36,permutation_max_error=permutation_error))
            print(json.dumps(dict(status='independent_repeat_weight_fold_checked',fold=f,own_fit_rows=len(fit))),flush=True)
    bindings(protocol['source_sha256']); bindings(protocol['input_sha256'])
    assert sha(OUT/'protocol.json')==binding and sha(OUT/'results.json')==result_hash
    assert sha(OUT/'diagnostics.npz')==arrays_hash and sha(__file__)==checker_hash
    artifacts={str(OUT/n):sha(OUT/n) for n in ('protocol.json','results.json','diagnostics.npz')}
    receipt=dict(status='passed_independent_fixed_M6_repeat_weight_diagnostic_checks',
        protocol_sha256=binding,results_sha256=result_hash,arrays_sha256=arrays_hash,checker_sha256=checker_hash,
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],artifact_sha256=artifacts,
        folds=reports,own_fit_rows_checked=30720,arrays_checked=75,gradient_vectors_checked=55,
        inverse_Hessian_right_hand_sides_checked=20,repeat_permutations_checked=180,
        maximum_scalar_or_array_error=max(errors.values()),comparison_count=len(errors),
        scalar_and_array_errors=errors,independent_method='Nine hard-label BCE terms; explicit scalar pair sums; spectral inverse-Hessian metric; no runner import',
        new_model_fits=0,new_policy_evaluations=0,new_encoder_forwards=0,new_api_calls=0,
        new_heads_constructed=0,policy_actions_computed=0,cal_test_effects_computed=0,
        runtime_deployed=False,core_goal_achieved=False,
        coverage_boundary='All five own-fit scalar/vector archives and local metrics; consumed labels, ideal FP64 head objective. No optimization trajectory replay, new policy, or independent-source effect.',
        elapsed_seconds=time.perf_counter()-started,completed_at_utc=datetime.now(timezone.utc).isoformat())
    with destination.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(receipt,handle,indent=2,allow_nan=False);handle.write('\n')
    return {k:receipt[k] for k in ('status','own_fit_rows_checked','arrays_checked','maximum_scalar_or_array_error','elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--protocol-sha256')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        receipt = self_test() if args.self_test else check(args.protocol_sha256)
    print(json.dumps(receipt,allow_nan=False))
