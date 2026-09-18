"""Independently check descriptive Student localization; no fitting or GPU.

This module imports neither the localization runner nor a training module.
It verifies only the saved Student diagnostic and its already-audited inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
PRIOR = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_localization_v1'
ARMS = ['Direct', 'Pre', 'Probe']
PAIRS = [('Pre_minus_Direct', 0, 1), ('Probe_minus_Pre', 1, 2), ('Probe_minus_Direct', 0, 2)]
LAMBDA = .001
ALPHA = .5
ATOL = 1e-11


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    tail = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + tail), tail / (1 + tail))


def dot(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(np.ravel(a), np.ravel(b)))


def norm(a):
    return math.sqrt(dot(a, a))


def columns(u, v):
    # Sum rows explicitly instead of the runner's transposed BLAS product.
    return np.sum(u * v[:, None], axis=0, dtype=np.float64)


def scores(u, theta):
    return np.sum(u * theta, axis=1, dtype=np.float64)


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else None


def loss_gradient(u, q, w, theta, regularization=LAMBDA):
    s = scores(u, theta)
    rows = q * np.logaddexp(0., -s) + (1 - q) * np.logaddexp(0., s)
    penalty_vector = np.r_[theta[:-1], 0.]
    return (dot(w, rows) + .5 * regularization * dot(theta[:-1], theta[:-1]),
            columns(u, w * (sigmoid(s) - q)) + regularization * penalty_vector)


def secant_curvature(sa, sb):
    """Exact sigmoid integral curvature, with stable coincident endpoints."""
    hi, lo = np.maximum(sa, sb), np.minimum(sa, sb)
    span = hi - lo
    ratio = np.ones_like(span)
    active = span != 0
    ratio[active] = -np.expm1(-span[active]) / span[active]
    return sigmoid(hi) * sigmoid(-lo) * ratio


def pair_quantities(x, w, qa, qb, ta, tb):
    u = np.column_stack((x.astype(np.float64), np.ones(len(x))))
    dq, delta = qb - qa, tb - ta
    m = columns(u, w * dq)
    mean_x = columns(x, w)
    centered_x = x - mean_x
    centered = columns(centered_x, w * dq)
    mean_dq = dot(w, dq)
    rms_dq = math.sqrt(dot(w, dq * dq))
    rms_centered = math.sqrt(dot(w, (dq - mean_dq) ** 2))
    rms_x = math.sqrt(dot(w, np.sum(centered_x * centered_x, axis=1)))
    denominator = rms_centered * rms_x
    la, ga = loss_gradient(u, qa, w, ta)
    lb, gb = loss_gradient(u, qb, w, tb)
    laba, _ = loss_gradient(u, qb, w, ta)
    labb, _ = loss_gradient(u, qa, w, tb)
    sa, sb = scores(u, ta), scores(u, tb)
    curvature = secant_curvature(sa, sb)
    difference = sb - sa
    response = columns(u, w * curvature * difference) + LAMBDA * np.r_[delta[:-1], 0.]
    residual = response - m
    data_energy = dot(w, curvature * difference * difference)
    penalty_energy = LAMBDA * dot(delta[:-1], delta[:-1])
    total = data_energy + penalty_energy
    moment_energy = dot(delta, m)
    quantities = dict(target_difference_mean=mean_dq, target_difference_rms=rms_dq,
        target_difference_centered_rms=rms_centered, moment_norm=norm(m),
        centered_feature_moment_norm=norm(centered),
        centered_first_moment_association=norm(centered) / denominator if denominator else None,
        delta_beta_norm=norm(delta[:-1]), delta_bias=float(delta[-1]),
        source_gradient_inf=float(np.max(np.abs(ga))), target_gradient_inf=float(np.max(np.abs(gb))),
        objective_identity_error=max(abs(laba - la + dot(ta, m)), abs(lb - labb + dot(tb, m))),
        response_stationarity_residual_inf=float(np.max(np.abs(residual))),
        response_identity_error=float(np.max(np.abs(residual - (gb - ga)))),
        data_response_energy=data_energy, regularization_response_energy=penalty_energy,
        regularization_energy_fraction=penalty_energy / total if total else None,
        moment_response_energy=moment_energy, energy_stationarity_residual=total - moment_energy,
        energy_identity_error=abs(total - moment_energy - dot(delta, gb - ga)),
        fit_fp64_score_delta_weighted_rms=math.sqrt(dot(w, difference * difference)))
    assert np.max(np.abs(m[:-1] - mean_x * m[-1] - centered)) <= ATOL
    assert abs((laba - lb) + (labb - la) - moment_energy) <= ATOL
    for key in ('objective_identity_error', 'response_identity_error', 'energy_identity_error'):
        assert quantities[key] <= ATOL, key
    return quantities, dict(moment=m, centered_moment=centered, delta=delta, gradient_a=ga, gradient_b=gb)


def action_quantities(sa, sb, gap):
    sa, sb, gap = [np.asarray(v, dtype=np.float64) for v in (sa, sb, gap)]
    assert sa.shape == sb.shape == gap.shape and sa.ndim == 1 and len(sa)
    assert all(np.isfinite(v).all() for v in (sa, sb, gap))
    aa, bb, change = sa > 0, sb > 0, sb - sa
    effect = (bb.astype(int) - aa.astype(int)) * gap
    n = len(gap)
    mean_a, mean_b = math.fsum(sa) / n, math.fsum(sb) / n
    centered_a, centered_b = sa - mean_a, sb - mean_b
    denominator = norm(centered_a) * norm(centered_b)
    same = aa == bb
    result = dict(rows=n, score_delta_mean=math.fsum(change) / n,
        score_delta_rms=norm(change) / math.sqrt(n),
        score_correlation=dot(centered_a, centered_b) / denominator if denominator else None,
        changed_actions=int(np.count_nonzero(~same)),
        unchanged_action_score_delta_rms=norm(change[same]) / math.sqrt(int(same.sum())) if same.any() else None,
        gain=math.fsum(effect) / n, cells={})
    for av, bv, name in [(False, False, 'D_to_D'), (False, True, 'D_to_B'),
                         (True, False, 'B_to_D'), (True, True, 'B_to_B')]:
        mask = (aa == av) & (bb == bv)
        values = effect[mask]
        quality_sum = math.fsum(values)
        result['cells'][name] = dict(rows=int(mask.sum()), improvements=int(np.count_nonzero(values > 1e-12)),
            harms=int(np.count_nonzero(values < -1e-12)), ties=int(np.count_nonzero(np.abs(values) <= 1e-12)),
            quality_sum=quality_sum, contribution=quality_sum / n)
    assert sum(v['rows'] for v in result['cells'].values()) == n
    assert all(v['improvements'] + v['harms'] + v['ties'] == v['rows'] for v in result['cells'].values())
    assert abs(math.fsum(v['contribution'] for v in result['cells'].values()) - result['gain']) <= ATOL
    return result


def shrink_vectors(x, w, y, teacher_probability, td, native_direct):
    u = np.column_stack((x, np.ones(len(x))))
    direct64 = sigmoid(scores(u, td))
    direct_native = sigmoid(native_direct)
    _, gd = loss_gradient(u, y, w, td)
    shrink = -ALPHA * LAMBDA * np.r_[td[:-1], 0.]
    residual = ALPHA * gd
    rounding = ALPHA * columns(u, w * (direct_native - direct64))
    oof = ALPHA * columns(u, w * (teacher_probability - direct_native))
    target = (1 - ALPHA) * y + ALPHA * teacher_probability
    moment = columns(u, w * (target - y))
    _, current = loss_gradient(u, target, w, td)
    _, compensated = loss_gradient(u, target, w, td, (1 - ALPHA) * LAMBDA)
    assert np.max(np.abs(shrink + residual + rounding + oof - moment)) <= ATOL
    assert np.max(np.abs(compensated - ((1 - ALPHA) * gd - rounding - oof))) <= ATOL
    return dict(regularization=shrink, optimizer_residual=residual,
                native_rounding=rounding, oof_vs_full_native=oof,
                current_soft_gradient_at_Direct=current,
                compensated_soft_gradient_at_Direct=compensated), moment


class Checks:
    def __init__(self):
        self.maximum_errors = {}
        self.scalar_and_array_checks = 0

    def same(self, actual, expected, name, tolerance=ATOL):
        if isinstance(expected, dict):
            assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
            for key, value in expected.items():
                self.same(actual[key], value, name + '/' + key, tolerance)
            return
        if expected is None or isinstance(expected, (bool, str)):
            assert actual == expected, name
        elif isinstance(expected, (int, np.integer)):
            assert isinstance(actual, (int, np.integer)) and actual == expected, name
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
            error = float(np.max(np.abs(a - b))) if a.size else 0.
            assert error <= tolerance, (name, error, tolerance)
            self.maximum_errors[name] = error
        self.scalar_and_array_checks += 1


def self_test():
    rng = np.random.default_rng(2026091512)
    x = rng.normal(size=(23, 4)); u = np.column_stack((x, np.ones(len(x))))
    w = rng.uniform(.1, 2, len(x)); w /= w.sum()
    qa, qb = rng.uniform(0, 1, (2, len(x))); qa[:2] = [0., 1.]
    ta, tb = rng.normal(size=(2, 5))
    record, _ = pair_quantities(x, w, qa, qb, ta, tb)
    _, gradient = loss_gradient(u, qa, w, ta)
    finite_difference = np.empty(len(ta)); h = 1e-6
    for j in range(len(ta)):
        displacement = np.zeros(len(ta)); displacement[j] = h
        finite_difference[j] = (loss_gradient(u, qa, w, ta + displacement)[0] -
                                loss_gradient(u, qa, w, ta - displacement)[0]) / (2 * h)
    gradient_error = float(np.max(np.abs(gradient - finite_difference)))
    assert gradient_error < 2e-9
    zero, _ = pair_quantities(x, w, qa, qa, ta, ta)
    assert zero['moment_norm'] == 0 and zero['regularization_energy_fraction'] is None
    equal = np.array([-1000., -1., 0., 1., 1000.])
    np.testing.assert_allclose(secant_curvature(equal, equal), sigmoid(equal) * sigmoid(-equal), atol=0, rtol=0)
    native = scores(u, ta) + rng.normal(0, .01, len(x))
    shrink_vectors(x, w, (qa > .5).astype(float), qb, ta, native)
    changes = action_quantities(np.array([-1., -1., 1., 1., 0.]),
        np.array([-1., 1., -1., 1., 1.]), np.array([.4, .3, .2, -.5, 5e-13]))
    assert changes['changed_actions'] == 3
    assert changes['cells']['D_to_B']['ties'] == 1
    assert abs(changes['gain'] - (.1 + 5e-13) / 5) < 1e-15
    return dict(status='passed_synthetic_independent_localization_checks',
        finite_difference_gradient_error=gradient_error,
        response_identity_error=record['response_identity_error'],
        new_fits=0, real_data_reads=0)


def check(output_dir=OUT):
    started = time.perf_counter()
    out = Path(output_dir).resolve()
    destination = out / 'separate_checks.json'
    assert not destination.exists(), 'Preserve an existing check receipt'
    protocol_path, result_path = out / 'protocol.json', out / 'results.json'
    protocol, result = read(protocol_path), read(result_path)
    assert protocol['status'] == 'frozen_before_new_descriptive_localization'
    assert result['status'] == 'complete_descriptive_localization_pending_separate_check'
    assert result['protocol_sha256'] == sha(protocol_path)
    assert protocol['comparisons'] == [v[0] for v in PAIRS]
    assert protocol['lambda_value'] == LAMBDA
    assert all(protocol[k] == result[k] == 0 for k in ('new_fits', 'new_policies', 'new_api_calls'))
    assert result['candidate_preparation_gate'] is False and result['core_goal_achieved'] is False
    for kind in ('source_sha256', 'input_sha256'):
        assert all(sha(path) == digest for path, digest in protocol[kind].items()), kind
    upstream = read(PRIOR / 'separate_checks.json')
    assert upstream['status'] == 'passed_independent_student_transfer_checks'
    assert upstream['results_sha256'] == sha(PRIOR / 'results.json')
    assert upstream['protocol_sha256'] == sha(PRIOR / 'protocol.json')
    input_set = {Path(v).resolve() for v in protocol['input_sha256']}
    required = [BASE / 'layer_pooling_v1/features.npz', PRIOR / 'predictions.npz',
                PRIOR / 'protocol.json', PRIOR / 'results.json', PRIOR / 'separate_checks.json']
    for f in range(5):
        required += [PRIOR / f'fold{f}_targets.npz', PRIOR / f'fold{f}_fit_cal.npz']
        required += [PRIOR / f'fold{f}_S_{arm}.npz' for arm in ARMS]
    assert {v.resolve() for v in required} <= input_set
    assert result['vectors_sha256'] == sha(out / 'vectors.npz')
    prior_result = read(PRIOR / 'results.json')
    with np.load(BASE / 'layer_pooling_v1/features.npz', allow_pickle=False) as archive:
        x, qids, groups = [archive[k].copy() for k in ('M6', 'query_ids', 'group_ids')]
    assert x.shape == (9600, 384) and x.dtype == np.float32 and np.isfinite(x).all()
    with np.load(PRIOR / 'predictions.npz', allow_pickle=False) as archive:
        assert np.array_equal(qids, archive['query_ids']) and np.array_equal(groups, archive['group_ids'])
        native_scores, fold_ids, utility = [archive[k].copy() for k in ('arm_scores', 'fold_id', 'utility')]
    assert native_scores.shape == (3, 9600) and np.isfinite(native_scores).all()
    gap = utility[:, 0] - utility[:, 1]
    with np.load(out / 'vectors.npz', allow_pickle=False) as archive:
        saved_vectors = {key: archive[key].copy() for key in archive.files}
    verified_vectors, expected_folds = {}, []
    checks = Checks()
    maximum_gradient, target_rows = 0., 0
    synthetic = self_test()
    with threadpool_limits(limits=2):
        for f in range(5):
            with np.load(PRIOR / f'fold{f}_targets.npz', allow_pickle=False) as archive:
                fit, targets, weight, teacher = [archive[k].copy() for k in
                    ('fit_indices', 'targets', 'weights', 'teacher_probability')]
            assert targets.shape == (3, len(fit)) and teacher.shape == (2, len(fit))
            assert np.array_equal(weight, np.where(np.abs(gap[fit]) > 1e-12, np.abs(gap[fit]), 0.))
            assert np.array_equal(targets[0], (gap[fit] > 0).astype(float))
            assert np.max(np.abs(targets[1:] - ((1 - ALPHA) * targets[0] + ALPHA * teacher))) <= ATOL
            active = weight > 0
            w = weight[active].astype(float) / float(weight[active].max())
            w = w / math.fsum(w)
            xx = x[fit][active].astype(float)
            q = targets[:, active]
            u = np.column_stack((xx, np.ones(len(xx))))
            target_rows += len(fit)
            with np.load(PRIOR / f'fold{f}_fit_cal.npz', allow_pickle=False) as archive:
                assert np.array_equal(fit, archive['fit_indices'])
                native_direct = archive['fit_scores'][0, active].copy()
            heads = []
            for arm in ARMS:
                with np.load(PRIOR / f'fold{f}_S_{arm}.npz', allow_pickle=False) as archive:
                    theta = np.r_[archive['coef'], float(archive['intercept'])]
                assert theta.shape == (385,) and np.isfinite(theta).all()
                heads.append(theta)
            head_summaries = {arm: dict(beta_norm=norm(t[:-1]), bias=float(t[-1]),
                beta_norm_ratio_to_Direct=norm(t[:-1]) / norm(heads[0][:-1]),
                beta_cosine_to_Direct=cosine(t[:-1], heads[0][:-1])) for arm, t in zip(ARMS, heads)}
            direct_scores = scores(u, heads[0])
            comparisons = {}
            for name, a, b in PAIRS:
                quantities, vectors = pair_quantities(xx, w, q[a], q[b], heads[a], heads[b])
                dq = q[b] - q[a]
                quantities['target_difference_covariance_with_Direct_fp64_score'] = dot(
                    w, (dq - dot(w, dq)) * (direct_scores - dot(w, direct_scores)))
                maximum_gradient = max(maximum_gradient, quantities['source_gradient_inf'], quantities['target_gradient_inf'])
                assert quantities['source_gradient_inf'] <= 1e-8 and quantities['target_gradient_inf'] <= 1e-8
                mask = fold_ids == f
                assert int(mask.sum()) == 1920
                quantities['test_action_changes'] = action_quantities(native_scores[a, mask], native_scores[b, mask], gap[mask])
                checks.same(quantities['test_action_changes']['gain'], prior_result['folds'][f]['primary_means'][name],
                            f'fold{f}/{name}/previous_gain')
                comparisons[name] = quantities
                for key, value in vectors.items():
                    artifact_key = f'fold{f}_{name}_{key}'
                    checks.same(saved_vectors[artifact_key], value, 'vector/' + artifact_key)
                    verified_vectors[artifact_key] = value
            shrink_records = {}
            for index in (1, 2):
                arm = ARMS[index]
                vectors, moment = shrink_vectors(xx, w, q[0], teacher[index - 1, active], heads[0], native_direct)
                for key, value in vectors.items():
                    artifact_key = f'fold{f}_{arm}_shrinkage_{key}'
                    checks.same(saved_vectors[artifact_key], value, 'vector/' + artifact_key)
                    verified_vectors[artifact_key] = value
                components = {key: value for key, value in vectors.items() if key in
                    ('regularization', 'optimizer_residual', 'native_rounding', 'oof_vs_full_native')}
                current, compensated = vectors['current_soft_gradient_at_Direct'], vectors['compensated_soft_gradient_at_Direct']
                # Tiny optimizer-residual vectors have unstable directions under
                # alternate summation. Their saved entries are independently
                # checked above; check reported angles from those exact entries.
                saved_moment = saved_vectors[f'fold{f}_{arm}_minus_Direct_moment']
                angles = {key: cosine(saved_vectors[f'fold{f}_{arm}_shrinkage_{key}'], saved_moment)
                          for key in components}
                shrink_records[arm] = dict(decomposition_error=float(np.max(np.abs(moment - sum(components.values())))),
                    component_norms={key: norm(value) for key, value in components.items()},
                    component_cosine_to_moment=angles,
                    Direct_point_current_soft_gradient_norm=norm(current),
                    Direct_point_compensated_soft_gradient_norm=norm(compensated),
                    compensated_to_current_gradient_ratio=norm(compensated) / norm(current) if norm(current) else None)
            expected = dict(fold=f, fit_positive_weight_rows=int(active.sum()), heads=head_summaries,
                            comparisons=comparisons, shrinkage=shrink_records)
            checks.same(result['folds'][f], expected, f'fold{f}')
            expected_folds.append(expected)
        assert len(result['folds']) == 5
        aggregate = {name: action_quantities(native_scores[a], native_scores[b], gap) for name, a, b in PAIRS}
        checks.same(result['aggregate'], aggregate, 'aggregate')
        for name, _, _ in PAIRS:
            checks.same(aggregate[name]['gain'], prior_result['primary'][name]['mean'], 'previous/' + name)
            for cell in aggregate[name]['cells']:
                for key in ('rows', 'improvements', 'harms', 'ties', 'quality_sum'):
                    total = math.fsum(row['comparisons'][name]['test_action_changes']['cells'][cell][key]
                                      for row in expected_folds)
                    checks.same(total, float(aggregate[name]['cells'][cell][key]), f'fold_sum/{name}/{cell}/{key}')
        cosines = {name: [[cosine(verified_vectors[f'fold{a}_{name}_centered_moment'],
                                  verified_vectors[f'fold{b}_{name}_centered_moment'])
                          for b in range(5)] for a in range(5)] for name, _, _ in PAIRS}
        checks.same(result['centered_moment_cosines'], cosines, 'centered_moment_cosines')
    assert set(saved_vectors) == set(verified_vectors)
    for kind in ('source_sha256', 'input_sha256'):
        assert all(sha(path) == digest for path, digest in protocol[kind].items()), kind
    value = dict(status='passed_independent_student_localization_checks',
        created_at_utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=sha(protocol_path), results_sha256=sha(result_path),
        vectors_sha256=sha(out / 'vectors.npz'),
        original_student_protocol_sha256=sha(PRIOR / 'protocol.json'),
        original_student_check_sha256=sha(PRIOR / 'separate_checks.json'),
        source_bindings_checked=len(protocol['source_sha256']), input_bindings_checked=len(protocol['input_sha256']),
        synthetic_checks=synthetic, folds_checked=5, pair_moment_response_checks=15,
        shrinkage_decompositions_checked=10, saved_vectors_checked=len(verified_vectors),
        strict_fit_target_rows=target_rows, Student_heads_checked=15,
        fold_action_tables_checked=15, aggregate_action_tables_checked=3,
        all_previous_effects_replayed=True, maximum_independent_gradient_inf=maximum_gradient,
        numeric_scalar_and_array_checks=checks.scalar_and_array_checks,
        maximum_absolute_error=max(checks.maximum_errors.values(), default=0.),
        maximum_absolute_errors=checks.maximum_errors,
        inference_scope='Descriptive fixed-artifact identities; no new hypothesis tests, refits, policies, or GPU replay.',
        cosine_scope='Reported component angles use independently entry-checked saved vectors; tiny residual directions are numerically sensitive.',
        new_fits=0, new_policies=0, new_api_calls=0, independent_native_GPU_replay=False,
        elapsed_seconds=time.perf_counter() - started)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    result = self_test() if args.self_test else check(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != 'maximum_absolute_errors'},
                     ensure_ascii=False, allow_nan=False), flush=True)
