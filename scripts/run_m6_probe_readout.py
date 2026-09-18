"""Frozen real-probe and two group-deranged controls on fixed M6 features."""
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

import m6_probe_math as pm
import run_m6_objective_readout as old

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work' / 'router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_probe_readout_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_probe_readout_plan_20260915.md'
PROBE_DIR = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/privileged_teacher'
PROBE_SHA = '0794f1664b6b5b01475741dad86dc6690579d160440926456d141c8cbb2f08da'
ARMS = ['P', 'S0', 'S1']
SEEDS = [2026091506, 2026091507]
PRIMARY = ['P_minus_L', 'P_minus_S0', 'P_minus_S1', 'P_minus_Dense', 'P_minus_BM25']
POLICIES = ['P', 'L', 'S0', 'S1', 'Dense', 'BM25']
CONFIG = {
    'queries': 9600, 'groups': 9559, 'folds': 5, 'fit_queries': 6144,
    'cal_queries': 1536, 'test_queries': 1920, 'inner_folds': 3,
    'regularizations': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'arms': ARMS, 'shuffle_seeds': SEEDS, 'linear_dimensions': 384,
    'probe_dimensions': 50, 'dimensions': 434, 'std_ddof': 0, 'std_atol': 1e-12,
    'probe_normalizer': 'sqrt(50)', 'tie_atol': 1e-12, 'gradient_atol': 1e-8,
    'batch_size': 8, 'formal_solves': 240, 'pilot_solves': 3, 'final_heads': 15,
    'new_thresholds': 0, 'bootstrap_draws': 20000, 'bootstrap_seed': 2026091508,
    'interval_quantiles': [.005, .995], 'minimum_diagnostic_increment': .002,
    'cpu_threads': 2, 'new_encoder_forwards': 0, 'new_external_calls': 0,
    'runtime_available': False,
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
    return [Path(__file__), Path(pm.__file__), ROOT / 'scripts/check_m6_probe_readout.py', PLAN,
            Path(old.__file__), ROOT / 'scripts/m6_objective_math.py',
            BASE / 'weighted_linear_probe.py', BASE / 'weighted_linear_probe_refined.py',
            ROOT / 'scripts/run_router_phase27_privileged_teacher.py']


def inputs():
    return old.input_paths() + [PROBE_DIR / 'teacher_features.npz',
                               PROBE_DIR / 'feature_schema.json', BASE / 'e04_protocol.json']


def versions():
    return {key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'torch', 'threadpoolctl')}


def data():
    d = old.data()
    assert sha(PROBE_DIR / 'teacher_features.npz') == PROBE_SHA
    assert read(BASE / 'e04_protocol.json')['teacher_features_sha256'] == PROBE_SHA
    schema = read(PROBE_DIR / 'feature_schema.json')['probe']
    assert schema['dimensions'] == 50 and len(schema['feature_names']) == 50
    assert not schema['uses_qrels'] and not schema['uses_generation_outcomes']
    # Do not deserialize the co-located gold block or former Teacher predictions.
    with np.load(PROBE_DIR / 'teacher_features.npz', allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], d['query_ids'])
        assert np.array_equal(z['group_ids'], d['group_ids'])
        d['probe'] = z['probe'].copy()
    assert d['probe'].shape == (9600, 50) and d['probe'].dtype == np.float64
    assert np.isfinite(d['probe']).all()
    return d


def bound(binding):
    assert binding and sha(OUT / 'protocol.json') == binding
    p = read(OUT / 'protocol.json')
    assert p['config'] == CONFIG and p['primary'] == PRIMARY and p['policies'] == POLICIES
    assert p['versions'] == versions()
    for kind in ('source_sha256', 'input_sha256'):
        assert all(sha(path) == digest for path, digest in p[kind].items()), kind
    return p, data()


def weights(gap):
    return np.where(np.abs(gap) > 1e-12, np.abs(gap), 0.)


def bce(scores, gap):
    scores, gap = np.asarray(scores, dtype=float), np.asarray(gap, dtype=float)
    assert scores.shape == gap.shape and np.isfinite(scores).all()
    w = weights(gap)
    assert w.sum() > 0
    return float(np.sum(w * np.logaddexp(0., (1. - 2. * (gap > 0)) * scores)) / w.sum())


def original_head(fold, torch):
    head = torch.load(BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt', map_location='cpu', weights_only=True)
    return head['weight'].numpy().reshape(384), float(head['bias'].item())


def donors(d, ids, context):
    if not len(ids):
        return np.empty((3, 0), dtype=np.int64)
    return np.stack([np.arange(len(ids), dtype=np.int64)] + [
        pm.group_donors(d['query_ids'][ids], d['group_ids'][ids], seed, context) for seed in SEEDS])


def context_record(d, train, valid, train_context, valid_context):
    geometry = pm.fit_geometry(d['probe'][train])
    return dict(train_indices=train, active_indices=train[weights(d['gap'][train]) > 0],
                mean=geometry['mean'], std=geometry['std'], active=geometry['active'],
                probe_train=pm.transform(d['probe'][train], geometry),
                donors_train=donors(d, train, train_context), valid_indices=valid,
                probe_valid=pm.transform(d['probe'][valid], geometry) if len(valid) else np.empty((0, 50), np.float32),
                donors_valid=donors(d, valid, valid_context))


def freeze():
    assert not OUT.exists(), 'Preserve existing experiment directory'
    d = data()
    counts = {}
    for f, (fit, cal, test) in enumerate(d['folds']):
        assignment = old.assignment(d['group_ids'][fit], f)
        partitions = [(f'fold{f}/fit', fit), (f'fold{f}/cal', cal), (f'fold{f}/test', test)]
        for k in range(3):
            partitions += [(f'fold{f}/inner{k}/train', fit[assignment != k]),
                           (f'fold{f}/inner{k}/valid', fit[assignment == k])]
        for label, ids in partitions:
            mapping = donors(d, ids, label)
            _, sizes = np.unique(d['group_ids'][ids], return_counts=True)
            counts[label] = {'queries': len(ids), 'groups': len(sizes),
                             'group_size_counts': {str(size): int(np.sum(sizes == size)) for size in np.unique(sizes)},
                             'controls_differing_donors': int(np.sum(mapping[1] != mapping[2]))}
    OUT.mkdir(parents=True)
    p = {'status': 'frozen_before_real_pilot_and_formal_probe_fits',
         'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
         'primary': PRIMARY, 'policies': POLICIES, 'versions': versions(),
         'source_sha256': {str(path.resolve()): sha(path) for path in sources()},
         'input_sha256': {str(path.resolve()): sha(path) for path in inputs()},
         'identity': {'queries': 9600, 'groups': 9559, 'probe_shape': [9600, 50], 'partitions': counts},
         'scope': 'Consumed old9600 matched post-retrieval diagnostic; not a pre-only candidate or independent-source confirmation'}
    write(OUT / 'protocol.json', p)
    print(json.dumps({'status': p['status'], 'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)


def pilot(binding):
    _p, d = bound(binding)
    assert not (OUT / 'pilot.json').exists() and not (OUT / 'fit_started.json').exists()
    torch = old.gpu()
    _groups, inverse, sizes = np.unique(d['group_ids'], return_inverse=True, return_counts=True)
    full_fit = d['folds'][0][0]
    fit = full_fit[sizes[inverse[full_fit]] == 1][:128]
    assert len(fit) == 128
    ctx = context_record(d, fit, np.empty(0, np.int64), 'pilot/train', 'pilot/unused')
    save(OUT / 'pilot_context.npz', **ctx)
    models, controls, paths = [], {}, [OUT / 'pilot_context.npz']
    with threadpool_limits(limits=2):
        for arm in range(3):
            probe = ctx['probe_train'][ctx['donors_train'][arm]]
            model = pm.fit_design(pm.augmented(d['features'][fit], probe), d['gap'][fit], .001)
            save(OUT / f'pilot_{ARMS[arm]}.npz', coef=model['coef'], intercept=np.array(model['intercept']))
            record = {k: v for k, v in model.items() if k != 'coef'}
            write(OUT / f'pilot_{ARMS[arm]}.json', record)
            assert model['accepted'], 'Preserve rejected pilot; do not enlarge budget'
            scores = pm.native_scores(d['features'][fit], probe, model['coef'], model['intercept'], torch)
            assert scores.shape == (128,) and scores.dtype == np.float32 and np.isfinite(scores).all()
            models.append(record)
            paths += [OUT / f'pilot_{ARMS[arm]}.{ext}' for ext in ('json', 'npz')]
        for f, (fit_ids, cal, test) in enumerate(d['folds']):
            coef, intercept = original_head(f, torch)
            geometry = pm.fit_geometry(d['probe'][fit_ids])
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                expected_fit = z['final_native_fit_logits'][:32]
            for name, ids, expected in (('fit', fit_ids[:32], expected_fit),
                    ('cal', cal[:32], d['L_cal'][f][:32]), ('test', test[:32], d['L_scores'][test[:32]])):
                observed = pm.native_scores(d['features'][ids], pm.transform(d['probe'][ids], geometry),
                                            np.r_[coef, np.zeros(50)], intercept, torch)
                assert np.array_equal(observed, expected), f'Zero-probe native replay differs: {f}/{name}'
                controls[f'fold{f}_{name}'] = 0.
    record = {'status': 'passed_three_real_pilot_solves_and_480_zero_probe_native_controls',
              'protocol_sha256': binding, 'pilot_rows': 128, 'pilot_solves': 3,
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
    totals = np.column_stack([np.bincount(inverse, weights=values[:, j]) for j in range(5)])
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    samples = np.empty((CONFIG['bootstrap_draws'], 5))
    for start in range(0, len(samples), 100):
        selected = rng.integers(len(sizes), size=(min(100, len(samples) - start), len(sizes)))
        samples[start:start + len(selected)] = totals[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None]
    return np.quantile(samples, CONFIG['interval_quantiles'], axis=0).T


def run(binding):
    p, d = bound(binding)
    checked = read(OUT / 'pilot.json')
    assert checked['status'] == 'passed_three_real_pilot_solves_and_480_zero_probe_native_controls'
    assert checked['protocol_sha256'] == binding
    assert all(sha(path) == digest for path, digest in checked['artifact_sha256'].items())
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
            for k in range(4):
                train = fit[assignment != k] if k < 3 else fit
                valid = fit[assignment == k] if k < 3 else np.empty(0, np.int64)
                ctx = context_record(d, train, valid,
                    f'fold{f}/inner{k}/train' if k < 3 else f'fold{f}/fit', f'fold{f}/inner{k}/valid')
                path = OUT / f'fold{f}_context{k}.npz'
                save(path, **ctx)
                artifacts.append(path)
                contexts.append(ctx)
            cv = np.full((3, 5, len(fit)), np.nan, np.float32)
            params, intercepts, solves = [], [], []

            def solve(arm, lambda_index, k):
                ctx = contexts[k]
                design = pm.augmented(x[ctx['train_indices']], ctx['probe_train'][ctx['donors_train'][arm]])
                model = pm.fit_design(design, gap[ctx['train_indices']], CONFIG['regularizations'][lambda_index])
                record = {key: value for key, value in model.items() if key != 'coef'}
                record.update(arm_index=arm, arm=ARMS[arm], lambda_index=lambda_index,
                    context=k, inner_split=k if k < 3 else None, solution_index=len(solves),
                    training_queries=len(ctx['train_indices']), role='inner' if k < 3 else 'refit')
                journal.write(json.dumps({'fold': f, 'coef': model['coef'].tolist(), **record}, allow_nan=False) + '\n')
                journal.flush()
                params.append(model['coef']); intercepts.append(model['intercept']); solves.append(record)
                assert model['accepted'], f'Fold {f} solve {len(solves)-1} rejected; journal preserved'
                return model

            for lambda_index in range(5):
                for k in range(3):
                    valid = assignment == k
                    ctx = contexts[k]
                    for arm in range(3):
                        model = solve(arm, lambda_index, k)
                        cv[arm, lambda_index, valid] = pm.native_scores(x[fit[valid]],
                            ctx['probe_valid'][ctx['donors_valid'][arm]], model['coef'], model['intercept'], torch)
            assert np.isfinite(cv).all()
            losses = np.asarray([[bce(row, gap[fit]) for row in arm_cv] for arm_cv in cv])
            chosen = [int(np.flatnonzero(row <= row.min() + 1e-12)[0]) for row in losses]
            models = [solve(arm, chosen[arm], 3) for arm in range(3)]
            ctx = contexts[3]
            geometry = {key: ctx[key] for key in ('mean', 'std', 'active')}
            geometry['training_rows'] = len(ctx['train_indices'])
            cal_probe, cal_donors = pm.transform(d['probe'][cal], geometry), donors(d, cal, f'fold{f}/cal')
            fit_native = np.stack([pm.native_scores(x[fit], ctx['probe_train'][ctx['donors_train'][a]],
                                                   model['coef'], model['intercept'], torch) for a, model in enumerate(models)])
            cal_native = np.stack([pm.native_scores(x[cal], cal_probe[cal_donors[a]],
                                                   model['coef'], model['intercept'], torch) for a, model in enumerate(models)])
            linear_coef, linear_intercept = original_head(f, torch)
            linear_scores = x[fit].astype(float) @ linear_coef.astype(float) + linear_intercept
            feasible = [bce(linear_scores, gap[fit]) + .5 * CONFIG['regularizations'][idx] *
                        float(linear_coef.astype(float) @ linear_coef.astype(float)) for idx in chosen]
            assert all(model['loss'] <= feasible[a] + 1e-8 for a, model in enumerate(models))
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                old_fit = z['final_native_fit_logits'].copy()
            record = {'fold': f, 'selected_lambda_index': chosen, 'cv_weighted_BCE': losses.tolist(),
                'solves': solves, 'fit_BCE': [bce(row, gap[fit]) for row in fit_native],
                'cal_BCE': [bce(row, gap[cal]) for row in cal_native], 'L_fit_BCE': bce(old_fit, gap[fit]),
                'L_cal_BCE': bce(d['L_cal'][f], gap[cal]), 'linear_feasible_objective': feasible}
            npz_path, json_path = OUT / f'fold{f}.npz', OUT / f'fold{f}.json'
            save(npz_path, coef=np.stack(params), intercept=np.asarray(intercepts), inner_assignment=assignment,
                 cv_native=cv, fit_native=fit_native, cal_native=cal_native, probe_cal=cal_probe, donors_cal=cal_donors)
            write(json_path, record)
            artifacts.extend([npz_path, json_path]); fold_records.append(record); all_models.append(models)
            print(json.dumps({'status': 'fold_fit_complete_no_new_test_effects', 'fold': f,
                'completed_formal_solves': 48 * (f + 1),
                'selected_lambdas': [CONFIG['regularizations'][i] for i in chosen]}), flush=True)
    artifacts.append(OUT / 'solutions.jsonl')
    write(OUT / 'fit_completion.json', {'status': 'all_240_solves_and_15_heads_fixed_before_new_test_predictions',
          'protocol_sha256': binding, 'formal_solves': 240, 'final_heads': 15, 'new_thresholds': 0,
          'artifact_sha256': {str(path): sha(path) for path in artifacts}})
    bound(binding)
    arm_scores = np.full((3, 9600), np.nan, np.float32)
    fold_id, test_artifacts = np.full(9600, -1, np.int64), []
    with threadpool_limits(limits=2):
        for f, (_fit, _cal, test) in enumerate(d['folds']):
            with np.load(OUT / f'fold{f}_context3.npz', allow_pickle=False) as z:
                geometry = {key: z[key].copy() for key in ('mean', 'std', 'active')}
                geometry['training_rows'] = len(z['train_indices'])
            test_probe, test_donors = pm.transform(d['probe'][test], geometry), donors(d, test, f'fold{f}/test')
            for arm, model in enumerate(all_models[f]):
                arm_scores[arm, test] = pm.native_scores(x[test], test_probe[test_donors[arm]],
                                                       model['coef'], model['intercept'], torch)
            fold_id[test] = f
            path = OUT / f'fold{f}_test.npz'
            save(path, test_indices=test, probe_test=test_probe, donors_test=test_donors,
                 native_scores=arm_scores[:, test])
            test_artifacts.append(path)
    assert np.isfinite(arm_scores).all() and np.all(fold_id >= 0)
    actions = {arm: arm_scores[a] > 0 for a, arm in enumerate(ARMS)}
    actions['L'] = d['L_scores'] > 0
    save(OUT / 'predictions.npz', query_ids=d['query_ids'], group_ids=d['group_ids'], utility=d['utility'],
         L_scores=d['L_scores'], arm_scores=arm_scores, fold_id=fold_id, **actions)
    write(OUT / 'predictions_frozen.json', {'status': 'complete_9600_OOF_actions_before_effects',
          'protocol_sha256': binding, 'predictions_sha256': sha(OUT / 'predictions.npz'),
          'fit_completion_sha256': sha(OUT / 'fit_completion.json'),
          'test_artifact_sha256': {str(path): sha(path) for path in test_artifacts}})
    actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
    quality = {name: np.where(action, d['utility'][:, 0], d['utility'][:, 1]) for name, action in actions.items()}
    values = np.column_stack([quality['P'] - quality[baseline] for baseline in ('L', 'S0', 'S1', 'Dense', 'BM25')])
    with threadpool_limits(limits=2):
        intervals = bootstrap(d['group_ids'], values)
    primary = {name: {'mean': float(values[:, j].mean()), 'interval': intervals[j].tolist()} for j, name in enumerate(PRIMARY)}
    transfer = all(primary[name]['interval'][0] > 0 and primary[name]['mean'] >= .002 for name in PRIMARY[:3])
    for f, (_, _, test) in enumerate(d['folds']):
        fold_records[f]['primary_means'] = {name: float(values[test, j].mean()) for j, name in enumerate(PRIMARY)}
        fold_records[f]['test_BCE'] = [bce(row[test], gap[test]) for row in arm_scores]
        fold_records[f]['L_test_BCE'] = bce(d['L_scores'][test], gap[test])
    result = {'status': 'complete_fixed_probe_comparison_pending_separate_check', 'protocol_sha256': binding,
        'primary': primary, 'policy': {name: old.policy_summary(actions[name], d['utility'], gap) for name in POLICIES},
        'transfer_investigation_gate': bool(transfer), 'candidate_preparation_gate': False,
        'folds': fold_records, 'formal_solves': 240, 'final_heads': 15, 'new_thresholds': 0,
        'new_encoder_forwards': 0, 'new_external_calls': 0, 'runtime_available': False,
        'elapsed_seconds': time.perf_counter() - started, 'core_goal_achieved': False, 'scope': p['scope']}
    bound(binding)
    write(OUT / 'results.json', result)
    print(json.dumps({'status': result['status'], 'primary': primary,
                      'transfer_investigation_gate': bool(transfer)}), flush=True)


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
