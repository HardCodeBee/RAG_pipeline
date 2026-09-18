"""Independent fixed regularization-compensation checks; no training or GPU.

Reuses the frozen independent Student checker for objective arithmetic,
historical inputs, BF16 bounds, and multiplicity-based group resampling.
Never imports the new runner or either training loss implementation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import check_m6_student_transfer as independent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1'
LOCALIZATION = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_localization_v1'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_compensation_v1'
ARMS = ['DirectC', 'PreC', 'ProbeC']
OLD_ARMS = ['Direct', 'Pre', 'Probe']
POLICIES = OLD_ARMS + ARMS + ['Dense', 'BM25']
PRIMARY = ['net_transfer_interaction', 'ProbeC_minus_PreC', 'ProbeC_minus_DirectC',
           'ProbeC_minus_Direct', 'ProbeC_minus_Dense', 'ProbeC_minus_BM25']
REGULARIZATION = .0005
QUANTILES = [.05 / 12, 1 - .05 / 12]
BOOTSTRAP_SEED = 2026091512
sha, read = independent.sha, independent.read


def compare_tree(actual, expected, name, errors):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
        for key, value in expected.items():
            compare_tree(actual[key], value, name + '/' + key, errors)
    elif expected is None or isinstance(expected, (bool, str)):
        assert actual == expected, name
    elif isinstance(expected, (int, np.integer)):
        assert isinstance(actual, (int, np.integer)) and actual == expected, name
    else:
        independent.compare(actual, expected, name, errors)


def contrast_values(gap, actions):
    """Per-query signed changes, including the prespecified interaction."""
    gap = np.asarray(gap, dtype=np.float64)
    a = {}
    for name in POLICIES:
        values = np.asarray(actions[name])
        assert values.shape == gap.shape and values.dtype == bool
        a[name] = values.astype(np.int64)
    return np.column_stack([
        ((a['ProbeC'] - a['DirectC']) - (a['Probe'] - a['Direct'])) * gap,
        (a['ProbeC'] - a['PreC']) * gap,
        (a['ProbeC'] - a['DirectC']) * gap,
        (a['ProbeC'] - a['Direct']) * gap,
        a['ProbeC'] * gap,
        (a['ProbeC'] - 1) * gap])


def decisions(point, intervals):
    point, intervals = np.asarray(point), np.asarray(intervals)
    assert point.shape == (6,) and intervals.shape == (6, 2)
    recipe = bool(all(point[j] >= .002 and intervals[j, 0] > 0 for j in (1, 2, 3)))
    candidate = recipe and all(point[j] >= .01 and intervals[j, 0] > 0 for j in (4, 5))
    return recipe, bool(candidate)


def model_check(directory, stem, features, gap, target, metadata, errors):
    record = read(directory / (stem + '.json'))
    assert all(record[key] == value for key, value in metadata.items()), stem
    assert record['accepted'] is True and record['optimizer_success'] is True
    assert record['status'] == 'accepted_stationary_solution' and record['acceptance_gradient_inf'] == 1e-8
    assert record['regularization'] == REGULARIZATION and record['cpu_threads'] == 2
    assert record['numerical_budget'] == independent.NUMERICAL_BUDGET
    assert 0 <= record['iterations'] <= 5000 and 0 < record['evaluations'] <= 50000
    refinement = record['numerical_refinement']
    assert refinement['config'] == independent.REFINEMENT and 0 <= refinement['iterations'] <= 8
    assert len(refinement['trace']) == refinement['iterations']
    assert all(0 <= row['backtracks'] < 25 for row in refinement['trace'])
    if metadata['arm'] == 'DirectC':
        assert np.array_equal(target, (gap > 0).astype(float))
    with np.load(directory / (stem + '.npz'), allow_pickle=False) as archive:
        assert set(archive.files) == {'coef', 'intercept'}
        coef = independent.real_array(archive['coef'], (384,), stem + ' coef', np.float64).copy()
        bias = float(independent.real_array(archive['intercept'], (), stem + ' bias', np.float64))
    quantities = independent.loss_quantities(features, target, independent.weights_for_gap(gap),
                                             coef, bias, regularization=REGULARIZATION)
    assert quantities['grad_inf'] <= 1e-8
    assert quantities['positive_weight_rows'] == record['positive_weight_rows']
    independent.compare(record['intercept'], bias, 'model_intercept', errors)
    for key in ('data_loss', 'penalty', 'loss', 'grad_inf'):
        independent.compare(record[key], quantities[key], key, errors)
    return dict(coef=coef, intercept=bias, record=record, gradient_inf=quantities['grad_inf'])


def self_test():
    rng = np.random.default_rng(2026091513)
    x = rng.normal(size=(17, 4)); gap = rng.uniform(-1, 1, 17)
    weight = np.abs(gap); weight[[1, 5]] = 0
    target = rng.random(17); target[:2] = [0., 1.]
    beta, bias = rng.normal(size=4), .27
    actual = independent.loss_quantities(x, target, weight, beta, bias, regularization=REGULARIZATION)
    theta = np.r_[beta, bias]; finite_difference = np.empty(5); step = 1e-6
    for j in range(5):
        delta = np.zeros(5); delta[j] = step
        losses = []
        for sign in (1, -1):
            candidate = theta + sign * delta
            losses.append(independent.loss_quantities(x, target, weight, candidate[:-1], candidate[-1],
                                                      regularization=REGULARIZATION)['loss'])
        finite_difference[j] = (losses[0] - losses[1]) / (2 * step)
    error = float(np.max(np.abs(finite_difference - actual['gradient'])))
    assert error < 2e-9
    old = independent.loss_quantities(x, target, weight, beta, bias, regularization=.001)
    assert abs(old['loss'] - actual['loss'] - .5 * (.001 - REGULARIZATION) * float(beta @ beta)) < 1e-14
    assert abs(old['gradient'][-1] - actual['gradient'][-1]) < 1e-14
    np.testing.assert_allclose(old['gradient'][:-1] - actual['gradient'][:-1],
                               (.001 - REGULARIZATION) * beta, atol=1e-14, rtol=0)
    actions = {name: rng.integers(0, 2, 17).astype(bool) for name in POLICIES}
    actions['Dense'] = np.zeros(17, bool); actions['BM25'] = np.ones(17, bool)
    values = contrast_values(gap, actions)
    for i in range(17):
        chosen = {name: float(actions[name][i]) * gap[i] for name in POLICIES}
        expected = [(chosen['ProbeC'] - chosen['DirectC']) - (chosen['Probe'] - chosen['Direct']),
            chosen['ProbeC'] - chosen['PreC'], chosen['ProbeC'] - chosen['DirectC'],
            chosen['ProbeC'] - chosen['Direct'], chosen['ProbeC'], chosen['ProbeC'] - gap[i]]
        np.testing.assert_allclose(values[i], expected, atol=1e-15, rtol=0)
    groups = np.array(['g' + str(i // 3) for i in range(17)])
    samples, _ = independent.frequency_bootstrap(values, groups, draws=31, seed=912)
    replay = np.random.default_rng(912); members = [np.flatnonzero(groups == name) for name in sorted(set(groups))]
    for j in range(31):
        chosen = replay.integers(len(members), size=len(members))
        ids = np.concatenate([members[k] for k in chosen])
        expected = [math.fsum(values[ids, k]) / len(ids) for k in range(6)]
        np.testing.assert_allclose(samples[j], expected, atol=1e-15, rtol=0)
    point = np.array([-1., .002, .002, .002, .01, .01])
    intervals = np.column_stack((np.full(6, 1e-9), np.ones(6)))
    assert decisions(point, intervals) == (True, True)
    intervals[0, 0] = -1.; assert decisions(point, intervals) == (True, True)
    intervals[1, 0] = 0.; assert decisions(point, intervals) == (False, False)
    return dict(status='passed_synthetic_compensation_objective_contrast_and_bootstrap_checks',
        finite_difference_gradient_error=error, bootstrap_draws=31, new_fits=0, real_data_reads=0)


def expected_config():
    return dict(lambda_student=.0005, lambda_teacher=.001, alpha=.5, formal_solves=15, pilot_solves=3,
        new_teacher_fits=0, new_targets=0, new_thresholds=0, bootstrap_draws=20000,
        bootstrap_seed=BOOTSTRAP_SEED, quantiles=QUANTILES, minimum_increment=.002,
        minimum_fixed_gain=.01, new_encoder_forwards=0, new_api_calls=0)


def check(output_dir=OUT, protocol_sha=None):
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    destination = output / 'separate_checks.json'
    assert not destination.exists(), 'Preserve an existing check receipt'
    protocol = read(output / 'protocol.json')
    binding = sha(output / 'protocol.json')
    assert protocol_sha is None or protocol_sha == binding
    assert protocol['status'] == 'frozen_before_compensation_pilot_and_fits'
    assert protocol['config'] == expected_config()
    assert protocol['primary'] == PRIMARY and protocol['policies'] == POLICIES
    assert protocol['versions'] == {name: importlib.metadata.version(name)
        for name in ('numpy', 'scipy', 'torch', 'threadpoolctl')}
    source_binding = independent.bindings(protocol['source_sha256'], 'compensation source')
    input_binding = independent.bindings(protocol['input_sha256'], 'compensation input')
    required_sources = [Path(__file__), Path(independent.__file__), ROOT / 'scripts/run_m6_student_compensation.py',
        ROOT / 'scripts/m6_student_math.py', ROOT / 'scripts/m6_probe_math.py',
        ROOT / 'analysis/hotpotqa_router/m6_student_compensation_plan_20260915.md',
        ROOT / 'scripts/run_m6_objective_readout.py', ROOT / 'scripts/m6_objective_math.py',
        independent.RESEARCH / 'weighted_linear_probe.py', independent.RESEARCH / 'weighted_linear_probe_refined.py']
    assert {p.resolve() for p in required_sources} <= set(source_binding)
    required_inputs = [PRIOR / name for name in ('protocol.json', 'results.json', 'separate_checks.json',
                        'predictions.npz', 'pilot_targets.npz', 'pilot.json')]
    required_inputs += [LOCALIZATION / name for name in ('protocol.json', 'results.json', 'separate_checks.json')]
    for f in range(5):
        required_inputs += [PRIOR / f'fold{f}_targets.npz']
        required_inputs += [PRIOR / f'fold{f}_S_{arm}.npz' for arm in OLD_ARMS]
    assert {p.resolve() for p in required_inputs} <= set(input_binding)
    prior_check = read(PRIOR / 'separate_checks.json')
    localization_check = read(LOCALIZATION / 'separate_checks.json')
    assert prior_check['status'] == 'passed_independent_student_transfer_checks'
    assert localization_check['status'] == 'passed_independent_student_localization_checks'
    for directory, receipt in ((PRIOR, prior_check), (LOCALIZATION, localization_check)):
        assert receipt['protocol_sha256'] == sha(directory / 'protocol.json')
        assert receipt['results_sha256'] == sha(directory / 'results.json')
        original_protocol = read(directory / 'protocol.json')
        for key in ('source_sha256', 'input_sha256'):
            independent.bindings(original_protocol[key], directory.name + '/' + key)
    # Prior targets/heads remain linked to the completed independent check.
    prior_artifacts = independent.bindings(prior_check['artifact_sha256'], 'previous audited artifact')
    for path in required_inputs:
        if path.parent == PRIOR and path.suffix == '.npz':
            assert path.resolve() in prior_artifacts, str(path)
    pilot = read(output / 'pilot.json')
    assert pilot['status'] == 'passed_three_compensation_pilot_fits_and_480_native_controls'
    assert pilot['protocol_sha256'] == binding and pilot['pilot_solves'] == 3 and pilot['new_policy_effects'] is False
    assert pilot['original_controls'] == {f'fold{f}_{part}': 0. for f in range(5) for part in ('fit', 'cal', 'test')}
    pilot_binding = independent.bindings(pilot['artifact_sha256'], 'compensation pilot')
    expected_pilot = {output / ('pilot_' + arm + suffix) for arm in ARMS for suffix in ('.npz', '.json')}
    assert set(pilot_binding) == expected_pilot
    completed = read(output / 'fit_completion.json')
    assert completed['status'] == 'all_15_heads_before_new_test_predictions'
    assert completed['protocol_sha256'] == binding and completed['formal_solves'] == 15
    formal_binding = independent.bindings(completed['artifact_sha256'], 'compensation fit')
    expected_formal = {output / f'fold{f}_{arm}{suffix}' for f in range(5) for arm in ARMS for suffix in ('.npz', '.json')}
    expected_formal |= {output / f'fold{f}_fit_cal.npz' for f in range(5)} | {output / 'solutions.jsonl'}
    assert set(formal_binding) == expected_formal
    fit_start = read(output / 'fit_started.json')
    assert fit_start['protocol_sha256'] == binding
    assert datetime.fromisoformat(protocol['created_at_utc']).timestamp() <= fit_start['started_at_unix']
    assert fit_start['started_at_unix'] <= datetime.fromisoformat(completed['completed_at_utc']).timestamp()
    frozen = read(output / 'predictions_frozen.json')
    assert frozen['status'] == 'all_compensated_predictions_before_effects' and frozen['protocol_sha256'] == binding
    assert frozen['predictions_sha256'] == sha(output / 'predictions.npz')
    assert frozen['fit_completion_sha256'] == sha(output / 'fit_completion.json')
    result = read(output / 'results.json')
    assert result['status'] == 'complete_compensation_pending_separate_check' and result['protocol_sha256'] == binding
    assert result['predictions_frozen_sha256'] == sha(output / 'predictions_frozen.json')
    assert result['bootstrap_sha256'] == sha(output / 'bootstrap.npz')
    assert result['formal_solves'] == 15 and result['pilot_solves'] == 3
    for key in ('new_teacher_fits', 'new_targets', 'new_thresholds', 'new_encoder_forwards', 'new_api_calls'):
        assert result[key] == 0
    assert result['runtime_available'] is False and result['core_goal_achieved'] is False
    assert result['scope'] == protocol['scope']
    data = independent.load_original_inputs()
    x, gap, utility, groups = [data[key] for key in ('features', 'gap', 'utility', 'group_ids')]
    with np.load(PRIOR / 'predictions.npz', allow_pickle=False) as archive:
        assert np.array_equal(archive['query_ids'], data['query_ids']) and np.array_equal(archive['group_ids'], groups)
        assert np.array_equal(archive['utility'], utility)
        old_scores = archive['arm_scores'].copy()
    with np.load(output / 'predictions.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {'query_ids', 'group_ids', 'utility', 'arm_scores', 'fold_id'}
        assert np.array_equal(archive['query_ids'], data['query_ids']) and np.array_equal(archive['group_ids'], groups)
        assert np.array_equal(archive['utility'], utility)
        new_scores = independent.real_array(archive['arm_scores'], (3, 9600), 'compensated test scores', np.float32).copy()
        fold_id = archive['fold_id'].copy()
    errors, native_counters, models, fold_records, journal_expected = {}, {}, [], [], []
    with np.load(PRIOR / 'pilot_targets.npz', allow_pickle=False) as archive:
        pilot_ids, pilot_targets = archive['target_indices'].copy(), archive['targets'].copy()
    assert pilot_ids.shape == (64,) and pilot_targets.shape == (3, 64)
    with threadpool_limits(limits=2):
        for j, arm in enumerate(ARMS):
            model = model_check(output, 'pilot_' + arm, x[pilot_ids], gap[pilot_ids], pilot_targets[j],
                dict(role='pilot', target_rows=64, arm=arm), errors)
            models.append(model)
        for f, parts in enumerate(data['folds']):
            fit, cal, test = [parts[key] for key in ('fit', 'calibration', 'test')]
            assert np.array_equal(np.flatnonzero(fold_id == f), np.sort(test))
            with np.load(PRIOR / f'fold{f}_targets.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['fit_indices'], fit)
                targets, weights = archive['targets'].copy(), archive['weights'].copy()
            assert targets.shape == (3, 6144) and np.array_equal(weights, independent.weights_for_gap(gap[fit]))
            assert np.array_equal(targets[0], (gap[fit] > 0).astype(float))
            with np.load(output / f'fold{f}_fit_cal.npz', allow_pickle=False) as archive:
                assert set(archive.files) == {'fit_indices', 'cal_indices', 'fit_scores', 'cal_scores'}
                assert np.array_equal(archive['fit_indices'], fit) and np.array_equal(archive['cal_indices'], cal)
                fit_scores = independent.real_array(archive['fit_scores'], (3, 6144), 'compensated fit scores', np.float32).copy()
                cal_scores = independent.real_array(archive['cal_scores'], (3, 1536), 'compensated cal scores', np.float32).copy()
            with np.load(PRIOR / f'fold{f}_S_Direct.npz', allow_pickle=False) as archive:
                direct_beta = archive['coef'].copy()
            head_record = {}
            for j, arm in enumerate(ARMS):
                stem = f'fold{f}_{arm}'
                model = model_check(output, stem, x[fit], gap[fit], targets[j], dict(role='student', fold=f,
                    arm=arm, targets_sha256=sha(PRIOR / f'fold{f}_targets.npz')), errors)
                models.append(model)
                journal_expected.append(dict(stem=stem, **model['record']))
                for label, ids, values in [('fit', fit, fit_scores[j]), ('cal', cal, cal_scores[j]), ('test', test, new_scores[j, test])]:
                    independent.native_score_bound(x[ids], model['coef'], model['intercept'], values,
                        native_counters, stem + '/' + label)
                with np.load(PRIOR / f'fold{f}_S_{OLD_ARMS[j]}.npz', allow_pickle=False) as archive:
                    old_beta = archive['coef'].copy()
                norm = lambda v: math.sqrt(math.fsum(float(k) ** 2 for k in v))
                head_record[arm] = dict(beta_norm=norm(model['coef']), bias=model['intercept'],
                    beta_distance_to_old_corresponding=norm(model['coef'] - old_beta),
                    beta_distance_to_old_Direct=norm(model['coef'] - direct_beta))
            fold_records.append(dict(fold=f, fit_BCE=[independent.weighted_bce(row, gap[fit]) for row in fit_scores],
                cal_BCE=[independent.weighted_bce(row, gap[cal]) for row in cal_scores], heads=head_record,
                test_BCE=[independent.weighted_bce(row[test], gap[test]) for row in new_scores]))
        journal = [json.loads(line) for line in (output / 'solutions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert journal == journal_expected and len(journal) == 15
        actions = {arm: old_scores[j] > 0 for j, arm in enumerate(OLD_ARMS)}
        actions.update({arm: new_scores[j] > 0 for j, arm in enumerate(ARMS)})
        actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
        values = contrast_values(gap, actions)
        draws, _ = independent.frequency_bootstrap(values, groups, draws=20000, seed=BOOTSTRAP_SEED)
        intervals = np.quantile(draws, QUANTILES, axis=0, method='linear').T
        point = independent.means(values)
        with np.load(output / 'bootstrap.npz', allow_pickle=False) as archive:
            assert set(archive.files) == {'draws'}
            independent.compare(archive['draws'], draws, 'all_bootstrap_draws', errors)
        primary = {name: dict(mean=float(point[j]), interval=intervals[j].tolist()) for j, name in enumerate(PRIMARY)}
        compare_tree(result['primary'], primary, 'primary', errors)
        policy = {name: independent.policy_summary(utility, actions[name]) for name in POLICIES}
        compare_tree(result['policy'], policy, 'policy', errors)
        description = {old: math.fsum((actions[new].astype(int) - actions[old].astype(int)) * gap) / len(gap)
                       for old, new in zip(OLD_ARMS, ARMS)}
        compare_tree(result['descriptive_compensation_gains'], description, 'descriptive_compensation_gains', errors)
        for f, parts in enumerate(data['folds']):
            means = independent.means(values[parts['test']])
            fold_records[f]['primary_means'] = {name: float(means[j]) for j, name in enumerate(PRIMARY)}
        assert len(result['folds']) == 5
        for f, expected in enumerate(fold_records):
            compare_tree(result['folds'][f], expected, f'fold{f}', errors)
    recipe, candidate = decisions(point, intervals)
    decision = ('PREPARE_COMPENSATED_STUDENT_INDEPENDENT_CONFIRMATION' if candidate else
        'COMPENSATED_INCREMENT_ONLY_NO_QUALIFIED_CANDIDATE' if recipe else
        'END_SINGLE_REGULARIZATION_COMPENSATION_NO_CONFIRMED_TRANSFER')
    assert result['transfer_gate'] == recipe and result['candidate_preparation_gate'] == candidate
    assert result['decision'] == decision
    independent.bindings(protocol['source_sha256'], 'final compensation source')
    independent.bindings(protocol['input_sha256'], 'final compensation input')
    synthetic = self_test()
    receipt = dict(status='passed_independent_student_compensation_checks',
        created_at_utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        reused_checker_sha256=sha(independent.__file__), protocol_sha256=binding,
        results_sha256=sha(output / 'results.json'), predictions_sha256=sha(output / 'predictions.npz'),
        original_student_check_sha256=sha(PRIOR / 'separate_checks.json'),
        localization_check_sha256=sha(LOCALIZATION / 'separate_checks.json'),
        source_bindings_checked=len(source_binding), input_bindings_checked=len(input_binding),
        pilot_solutions_checked=3, formal_solutions_checked=15, hard_solutions_checked=6, soft_solutions_checked=12,
        reused_formal_target_rows=30720, reused_pilot_target_rows=64, journal_solutions_checked=15,
        all_reused_targets_and_weights_preserved=True, Student_heads_checked=15,
        maximum_independent_gradient_inf=max(model['gradient_inf'] for model in models),
        maximum_absolute_errors=errors, native_score_consistency=native_counters,
        bootstrap_draws_checked=20000, bootstrap_quantiles=QUANTILES, primary=primary,
        transfer_gate=recipe, candidate_preparation_gate=candidate, decision=decision,
        synthetic_checks=synthetic, new_fits=0, new_teacher_fits=0, new_paid_calls=0,
        independent_native_GPU_replay=False,
        native_score_scope='Saved formal fit/cal/test outputs checked against BF16-operand/FP32-accumulation bounds; no independent GPU replay.',
        pilot_native_scope='Pilot heads/objectives checked; pilot native outputs were not saved. Historical 480 exact controls are source-bound records, not rerun here.',
        inference_scope='Consumed-development fixed-prediction conditional intervals; no independent-source or refitting uncertainty.',
        elapsed_seconds=time.perf_counter() - started)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    result = self_test() if args.self_test else check(args.output_dir, args.protocol_sha)
    print(json.dumps({key: value for key, value in result.items() if key != 'maximum_absolute_errors'},
                     ensure_ascii=False, allow_nan=False), flush=True)
