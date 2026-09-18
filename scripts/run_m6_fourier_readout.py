"""Frozen two-seed Fourier extension of the existing M6 weighted-BCE readout."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

import m6_fourier_math as fm
import run_m6_objective_readout as old


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work' / 'router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_fourier_readout_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_fourier_readout_plan_20260915.md'
SEEDS = [2026091502, 2026091503]
PRIMARY = ['N_minus_L', 'N_minus_Dense', 'N_minus_BM25']
POLICIES = ['N', 'L', 'S0', 'S1', 'Dense', 'BM25']
CONFIG = {
    'queries': 9600, 'groups': 9559, 'folds': 5, 'fit_queries': 6144,
    'cal_queries': 1536, 'test_queries': 1920, 'inner_folds': 3,
    'regularizations': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'seeds': SEEDS, 'geometry_seed': 2026091505, 'geometry_max_pairs': 512,
    'scales': [.5, 1., 2.], 'frequencies_per_scale': 64,
    'linear_dimensions': 384, 'fourier_dimensions': 384, 'dimensions': 768,
    'tie_atol': 1e-12, 'gradient_atol': 1e-8, 'batch_size': 8,
    'formal_solves': 160, 'pilot_solves': 2, 'new_thresholds': 0,
    'bootstrap_draws': 20000, 'bootstrap_seed': 2026091504,
    'interval_quantiles': [.05 / 6, 1 - .05 / 6],
    'minimum_recipe_increment': .002, 'minimum_gain_over_both_fixed': .01,
    'cpu_threads': 2, 'new_encoder_forwards': 0, 'new_external_calls': 0,
    'ensemble': 'FP64 mean of two FP32 native scores; one shared inner-selected lambda',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save(path, **values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def sources():
    return [Path(__file__), Path(fm.__file__), ROOT / 'scripts/check_m6_fourier_readout.py', PLAN,
            Path(old.__file__), ROOT / 'scripts/m6_objective_math.py',
            BASE / 'weighted_linear_probe.py', BASE / 'weighted_linear_probe_refined.py']


def versions():
    return {key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'torch', 'threadpoolctl')}


def bound(binding):
    assert binding and sha(OUT / 'protocol.json') == binding
    p = read(OUT / 'protocol.json')
    assert p['config'] == CONFIG and p['primary'] == PRIMARY and p['policies'] == POLICIES
    assert p['versions'] == versions()
    for kind in ('source_sha256', 'input_sha256'):
        assert all(sha(path) == digest for path, digest in p[kind].items()), kind
    assert sha(OUT / 'frequencies.npz') == p['frequencies_sha256']
    with np.load(OUT / 'frequencies.npz', allow_pickle=False) as z:
        frequencies = [z[f'seed{s}'].copy() for s in range(2)]
    assert all(np.array_equal(frequencies[s], fm.make_frequencies(SEEDS[s])) for s in range(2))
    return p, old.data(), frequencies


def weights(gap):
    return np.where(np.abs(gap) > 1e-12, np.abs(gap), 0.)


def bce(scores, gap):
    scores = np.asarray(scores, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    assert scores.shape == gap.shape and np.isfinite(scores).all()
    w = weights(gap)
    assert w.sum() > 0
    return float(np.sum(w * np.logaddexp(0., (1. - 2. * (gap > 0)) * scores)) / w.sum())


def original_head(fold, torch):
    head = torch.load(BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt', map_location='cpu', weights_only=True)
    return head['weight'].numpy().reshape(384), float(head['bias'].item())


def freeze():
    assert not OUT.exists(), 'Preserve any existing experiment directory'
    d = old.data()
    OUT.mkdir(parents=True)
    save(OUT / 'frequencies.npz', **{f'seed{s}': fm.make_frequencies(seed) for s, seed in enumerate(SEEDS)})
    p = {'status': 'frozen_before_real_pilot_and_formal_Fourier_fits',
         'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
         'primary': PRIMARY, 'policies': POLICIES, 'versions': versions(),
         'source_sha256': {str(path.resolve()): sha(path) for path in sources()},
         'input_sha256': {str(path.resolve()): sha(path) for path in old.input_paths()},
         'frequencies_sha256': sha(OUT / 'frequencies.npz'),
         'identity': {'queries': len(d['query_ids']), 'groups': len(set(d['group_ids']))},
         'scope': 'Consumed old9600 development; no independent-source confirmation or deployment'}
    write(OUT / 'protocol.json', p)
    print(json.dumps({'status': p['status'], 'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)


def pilot(binding):
    _p, d, frequencies = bound(binding)
    assert not (OUT / 'pilot.json').exists() and not (OUT / 'fit_started.json').exists()
    torch = old.gpu()
    fit = d['folds'][0][0][:128]
    x, gap = d['features'], d['gap']
    models, controls = [], {}
    with threadpool_limits(limits=2):
        geometry = fm.geometry(x[fit])
        for seed in range(2):
            model = fm.fit_augmented(x[fit], gap[fit], frequencies[seed], geometry, regularization=.001)
            save(OUT / f'pilot_seed{seed}.npz', coef=model['coef'], intercept=np.array(model['intercept']))
            record = {k: v for k, v in model.items() if k != 'coef'}
            write(OUT / f'pilot_seed{seed}.json', record)
            assert model['accepted'], 'Preserve rejected pilot; do not enlarge budget'
            scores = fm.native_scores(x[fit], frequencies[seed], geometry['scale'], model['coef'], model['intercept'], torch)
            assert scores.shape == (128,) and scores.dtype == np.float32 and np.isfinite(scores).all()
            models.append(record)
        for f, (fit_ids, cal, test) in enumerate(d['folds']):
            coef, intercept = original_head(f, torch)
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                expected_fit = z['final_native_fit_logits'][:32]
            for name, ids, expected in (('fit', fit_ids[:32], expected_fit),
                    ('cal', cal[:32], d['L_cal'][f][:32]), ('test', test[:32], d['L_scores'][test[:32]])):
                observed = fm.native_scores(x[ids], frequencies[0], geometry['scale'],
                                            np.r_[coef, np.zeros(384)], intercept, torch)
                assert np.array_equal(observed, expected), f'Zero-extension native replay differs: {f}/{name}'
                controls[f'fold{f}_{name}'] = 0.
    paths = [OUT / f'pilot_seed{s}.{suffix}' for s in range(2) for suffix in ('json', 'npz')]
    record = {'status': 'passed_two_real_pilot_solves_and_480_zero_extension_native_controls',
        'protocol_sha256': binding, 'pilot_rows': 128, 'pilot_solves': 2,
        'original_native_control_queries': 480, 'original_controls': controls,
        'pilot_gradients': [model['grad_inf'] for model in models],
        'artifact_sha256': {str(path): sha(path) for path in paths},
        'new_policy_effects': False, 'new_encoder_forwards': 0, 'new_external_calls': 0}
    bound(binding)
    write(OUT / 'pilot.json', record)
    print(json.dumps(record), flush=True)


def bootstrap(groups, values):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    totals = np.column_stack([np.bincount(inverse, weights=values[:, j]) for j in range(3)])
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    samples = np.empty((CONFIG['bootstrap_draws'], 3))
    for start in range(0, len(samples), 100):
        selected = rng.integers(len(sizes), size=(min(100, len(samples) - start), len(sizes)))
        samples[start:start + len(selected)] = totals[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None]
    return np.quantile(samples, CONFIG['interval_quantiles'], axis=0).T


def run(binding):
    p, d, frequencies = bound(binding)
    pilot_record = read(OUT / 'pilot.json')
    assert pilot_record['status'] == 'passed_two_real_pilot_solves_and_480_zero_extension_native_controls'
    assert pilot_record['protocol_sha256'] == binding
    assert all(sha(path) == digest for path, digest in pilot_record['artifact_sha256'].items())
    assert not (OUT / 'fit_started.json').exists() and not (OUT / 'results.json').exists()
    write(OUT / 'fit_started.json', {'protocol_sha256': binding, 'started_at_unix': time.time()})
    started = time.perf_counter()
    torch = old.gpu()
    x, gap = d['features'], d['gap']
    all_models, fold_records, artifacts = [], [], []
    with threadpool_limits(limits=2), (OUT / 'solutions.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
        for f, (fit, cal, _test) in enumerate(d['folds']):
            assignment = old.assignment(d['group_ids'][fit], f)
            contexts = []
            for context in range(4):
                train = fit[assignment != context] if context < 3 else fit
                geometry = fm.geometry(x[train])
                design = [fm.augmented(x[train], omega, geometry['scale']) for omega in frequencies]
                active = weights(gap[train]) > 0
                path = OUT / f'fold{f}_context{context}.npz'
                save(path, train_indices=train, active_indices=train[active],
                     pair_indices=geometry['pair_indices'], scale=np.array(geometry['scale']),
                     **{f'rff_seed{s}': design[s][active, 384:] for s in range(2)})
                artifacts.append(path)
                contexts.append({'train': train, 'geometry': geometry, 'design': design})
            cv = np.full((2, 5, len(fit)), np.nan, dtype=np.float32)
            params, intercepts, solves = [], [], []

            def solve(seed, lambda_index, context):
                ctx = contexts[context]
                model = fm.fit_design(ctx['design'][seed], gap[ctx['train']], CONFIG['regularizations'][lambda_index])
                record = {k: v for k, v in model.items() if k != 'coef'}
                record.update(seed_index=seed, seed=SEEDS[seed], lambda_index=lambda_index,
                    context=context, inner_split=context if context < 3 else None,
                    solution_index=len(solves), training_queries=len(ctx['train']),
                    role='inner' if context < 3 else 'refit')
                journal.write(json.dumps({'fold': f, 'coef': model['coef'].tolist(), **record}, allow_nan=False) + '\n')
                journal.flush()
                params.append(model['coef']); intercepts.append(model['intercept']); solves.append(record)
                assert model['accepted'], f'Fold {f} solve {len(solves)-1} rejected; journal preserved'
                return model

            for lambda_index in range(5):
                for context in range(3):
                    valid = assignment == context
                    for seed in range(2):
                        model = solve(seed, lambda_index, context)
                        cv[seed, lambda_index, valid] = fm.native_scores(x[fit[valid]], frequencies[seed],
                            contexts[context]['geometry']['scale'], model['coef'], model['intercept'], torch)
            assert np.isfinite(cv).all()
            cv_mean = cv.astype(np.float64).mean(axis=0)
            losses = np.array([bce(row, gap[fit]) for row in cv_mean])
            chosen = int(np.flatnonzero(losses <= losses.min() + 1e-12)[0])
            models = [solve(seed, chosen, 3) for seed in range(2)]
            scale = contexts[3]['geometry']['scale']
            fit_native = np.stack([fm.native_scores(x[fit], frequencies[s], scale, model['coef'], model['intercept'], torch)
                                   for s, model in enumerate(models)])
            cal_native = np.stack([fm.native_scores(x[cal], frequencies[s], scale, model['coef'], model['intercept'], torch)
                                   for s, model in enumerate(models)])
            linear_coef, linear_intercept = original_head(f, torch)
            linear_scores = x[fit].astype(float) @ linear_coef.astype(float) + linear_intercept
            feasible = bce(linear_scores, gap[fit]) + .5 * CONFIG['regularizations'][chosen] * float(linear_coef.astype(float) @ linear_coef.astype(float))
            assert all(model['loss'] <= feasible + 1e-8 for model in models), 'Optimized objective exceeds fixed linear feasible point'
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                old_fit = z['final_native_fit_logits']
            record = {'fold': f, 'selected_lambda_index': chosen, 'cv_weighted_BCE': losses.tolist(),
                'solves': solves, 'geometry_scales': [ctx['geometry']['scale'] for ctx in contexts],
                'fit_BCE_seed': [bce(row, gap[fit]) for row in fit_native],
                'fit_BCE_ensemble': bce(fit_native.astype(float).mean(axis=0), gap[fit]),
                'cal_BCE_seed': [bce(row, gap[cal]) for row in cal_native],
                'cal_BCE_ensemble': bce(cal_native.astype(float).mean(axis=0), gap[cal]),
                'L_fit_BCE': bce(old_fit, gap[fit]), 'L_cal_BCE': bce(d['L_cal'][f], gap[cal]),
                'linear_feasible_objective': feasible}
            npz_path, json_path = OUT / f'fold{f}.npz', OUT / f'fold{f}.json'
            save(npz_path, coef=np.stack(params), intercept=np.asarray(intercepts),
                 inner_assignment=assignment, cv_native=cv, fit_native=fit_native, cal_native=cal_native)
            write(json_path, record)
            artifacts.extend([npz_path, json_path])
            fold_records.append(record)
            all_models.append(models)
            print(json.dumps({'status': 'fold_fit_complete_no_new_test_effects', 'fold': f,
                'completed_formal_solves': 32 * (f + 1), 'selected_lambda': CONFIG['regularizations'][chosen]}), flush=True)
    artifacts.append(OUT / 'solutions.jsonl')
    write(OUT / 'fit_completion.json', {'status': 'all_160_solves_and_10_heads_fixed_before_new_test_predictions',
        'protocol_sha256': binding, 'formal_solves': 160, 'final_heads': 10, 'new_thresholds': 0,
        'artifact_sha256': {str(path): sha(path) for path in artifacts}})
    bound(binding)
    seed_scores = np.full((2, 9600), np.nan, dtype=np.float32)
    fold_id = np.full(9600, -1, dtype=np.int64)
    with threadpool_limits(limits=2):
        for f, (_fit, _cal, test) in enumerate(d['folds']):
            for seed, model in enumerate(all_models[f]):
                seed_scores[seed, test] = fm.native_scores(x[test], frequencies[seed],
                    fold_records[f]['geometry_scales'][3], model['coef'], model['intercept'], torch)
            fold_id[test] = f
    assert np.isfinite(seed_scores).all() and np.all(fold_id >= 0)
    scores = seed_scores.astype(float).mean(axis=0)
    actions = {'N': scores > 0, 'L': d['L_scores'] > 0, 'S0': seed_scores[0] > 0, 'S1': seed_scores[1] > 0}
    save(OUT / 'predictions.npz', query_ids=d['query_ids'], group_ids=d['group_ids'], utility=d['utility'],
         L_scores=d['L_scores'], N_seed_scores=seed_scores, N_scores=scores, fold_id=fold_id, **actions)
    write(OUT / 'predictions_frozen.json', {'status': 'complete_9600_OOF_actions_before_effects',
        'protocol_sha256': binding, 'predictions_sha256': sha(OUT / 'predictions.npz'),
        'fit_completion_sha256': sha(OUT / 'fit_completion.json')})
    actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
    quality = {name: np.where(action, d['utility'][:, 0], d['utility'][:, 1]) for name, action in actions.items()}
    values = np.column_stack([quality['N'] - quality[baseline] for baseline in ('L', 'Dense', 'BM25')])
    with threadpool_limits(limits=2):
        intervals = bootstrap(d['group_ids'], values)
    primary = {name: {'mean': float(values[:, j].mean()), 'interval': intervals[j].tolist()} for j, name in enumerate(PRIMARY)}
    followup = bool(primary['N_minus_L']['interval'][0] > 0 and primary['N_minus_L']['mean'] >= .002)
    candidate = followup and all(primary[f'N_minus_{fixed}']['interval'][0] > 0 and primary[f'N_minus_{fixed}']['mean'] >= .01 for fixed in ('Dense', 'BM25'))
    for f, (_, _, test) in enumerate(d['folds']):
        fold_records[f]['primary_means'] = {name: float(values[test, j].mean()) for j, name in enumerate(PRIMARY)}
        fold_records[f]['test_BCE_seed'] = [bce(row[test], gap[test]) for row in seed_scores]
        fold_records[f]['test_BCE_ensemble'] = bce(scores[test], gap[test])
        fold_records[f]['L_test_BCE'] = bce(d['L_scores'][test], gap[test])
    result = {'status': 'complete_fixed_Fourier_comparison_pending_separate_check',
        'protocol_sha256': binding, 'primary': primary,
        'policy': {name: old.policy_summary(actions[name], d['utility'], gap) for name in POLICIES},
        'recipe_followup_gate': followup, 'candidate_preparation_gate': bool(candidate),
        'folds': fold_records, 'formal_solves': 160, 'final_heads': 10, 'new_thresholds': 0,
        'new_encoder_forwards': 0, 'new_external_calls': 0,
        'elapsed_seconds': time.perf_counter() - started, 'core_goal_achieved': False, 'scope': p['scope']}
    bound(binding)
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'primary': primary, 'candidate_preparation_gate': bool(candidate)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'pilot', 'run'))
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    try:
        if args.mode == 'freeze':
            freeze()
        elif args.mode == 'pilot':
            pilot(args.protocol_sha)
        else:
            run(args.protocol_sha)
    except Exception:
        path = OUT / (args.mode + '_failure.json')
        if OUT.exists() and not path.exists():
            write(path, {'protocol_sha256': args.protocol_sha, 'traceback': traceback.format_exc()})
        raise
