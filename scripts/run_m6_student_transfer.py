"""Fixed strict own-fit OOF Teachers and pre-only M6 soft-BCE Students."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

import m6_student_math as sm
import m6_probe_math as pm
import run_m6_probe_readout as previous
import run_m6_objective_readout as old

ROOT = Path(__file__).resolve().parents[1]
BASE = old.BASE
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_student_transfer_plan_20260915.md'
ARMS = ['Direct', 'Pre', 'Probe']
TEACHERS = ['pre', 'probe']
POLICIES = ARMS + ['Dense', 'BM25']
PRIMARY = ['Pre_minus_Direct', 'Probe_minus_Pre', 'Probe_minus_Direct',
           'Probe_minus_Dense', 'Probe_minus_BM25']
CONFIG = dict(queries=9600, groups=9559, folds=5, fit_queries=6144,
    cal_queries=1536, test_queries=1920, inner_folds=3, regularization=.001,
    alpha=.5, student_dimensions=384, probe_dimensions=50, tie_atol=1e-12,
    std_ddof=0, std_atol=1e-12, probe_normalizer='sqrt(50)', gradient_atol=1e-8,
    batch_size=8, formal_solves=45, teacher_heads=30, student_heads=15,
    pilot_solves=5, new_thresholds=0, bootstrap_draws=20000,
    bootstrap_seed=2026091509, interval_quantiles=[.005, .995],
    minimum_transfer_increment=.002, minimum_gain_over_both_fixed=.01,
    cpu_threads=2, new_encoder_forwards=0, new_external_calls=0, runtime_available=False)

sha, read, write, save = previous.sha, previous.read, previous.write, previous.save


def sources():
    return [Path(__file__), Path(sm.__file__), Path(pm.__file__), Path(previous.__file__),
            Path(old.__file__), ROOT / 'scripts/m6_objective_math.py', PLAN,
            ROOT / 'scripts/check_m6_student_transfer.py',
            BASE / 'weighted_linear_probe.py', BASE / 'weighted_linear_probe_refined.py',
            ROOT / 'tests/test_m6_student_math.py']


def inputs():
    return previous.inputs() + [BASE / 'layer_pooling_v1/results.json']


def bound(binding):
    assert binding and sha(OUT / 'protocol.json') == binding
    p = read(OUT / 'protocol.json')
    assert p['config'] == CONFIG and p['primary'] == PRIMARY and p['policies'] == POLICIES
    assert p['versions'] == previous.versions()
    for kind in ('source_sha256', 'input_sha256'):
        assert all(sha(path) == digest for path, digest in p[kind].items()), kind
    return p, previous.data()


def freeze():
    assert not OUT.exists(), 'Preserve existing experiment directory'
    d = previous.data()
    assert all(row['selected_lambdas']['M6'] == .001
               for row in read(BASE / 'layer_pooling_v1/results.json')['per_fold'])
    partitions = []
    for f, (fit, cal, test) in enumerate(d['folds']):
        a = old.assignment(d['group_ids'][fit], f)
        for k in range(3):
            train, valid = fit[a != k], fit[a == k]
            assert not set(d['group_ids'][train]) & set(d['group_ids'][np.r_[valid, cal, test]])
            partitions.append(dict(fold=f, inner=k, train_rows=len(train), target_rows=len(valid)))
    OUT.mkdir(parents=True)
    p = dict(status='frozen_before_real_pilot_and_formal_student_fits',
        created_at_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
        primary=PRIMARY, policies=POLICIES, versions=previous.versions(),
        source_sha256={str(path.resolve()): sha(path) for path in sources()},
        input_sha256={str(path.resolve()): sha(path) for path in inputs()},
        identity=dict(queries=9600, groups=9559, partitions=partitions),
        scope='Consumed old9600 strict OOF transfer development; independent source confirmation and deployment pending')
    write(OUT / 'protocol.json', p)
    print(json.dumps(dict(status=p['status'], protocol_sha256=sha(OUT / 'protocol.json'))), flush=True)


def hard_fit(x, gap):
    return pm.refined.fit(x, (gap > 0).astype(float), previous.weights(gap), regularization=.001)


def store_model(stem, model, metadata, journal=None):
    path = OUT / (stem + '.npz')
    save(path, coef=model['coef'], intercept=np.array(model['intercept']))
    record = dict(metadata, **{key: value for key, value in model.items() if key != 'coef'})
    write(OUT / (stem + '.json'), record)
    if journal is not None:
        journal.write(json.dumps(dict(stem=stem, **record), allow_nan=False) + '\n')
        journal.flush()
    assert model['accepted'], 'Preserve rejected fit and stop without increasing budget'
    return [path, OUT / (stem + '.json')]


def teacher_pair(d, train, valid, prefix, torch, journal=None):
    assert not set(d['group_ids'][train]) & set(d['group_ids'][valid])
    geometry = pm.fit_geometry(d['probe'][train])
    pt, pv = pm.transform(d['probe'][train], geometry), pm.transform(d['probe'][valid], geometry)
    path = OUT / (prefix + '_context.npz')
    save(path, train_indices=train, valid_indices=valid, mean=geometry['mean'],
         std=geometry['std'], active=geometry['active'], probe_train=pt, probe_valid=pv)
    paths, scores = [path], []
    for name in TEACHERS:
        design = d['features'][train] if name == 'pre' else pm.augmented(d['features'][train], pt)
        model = hard_fit(design, d['gap'][train])
        paths += store_model(prefix + '_T_' + name, model, dict(role='teacher', teacher=name,
            context=prefix + '_context.npz', train_rows=len(train), target_rows=len(valid)), journal)
        s = old.native(d['features'][valid], model['coef'], model['intercept'], torch) if name == 'pre' else \
            pm.native_scores(d['features'][valid], pv, model['coef'], model['intercept'], torch)
        scores.append(s)
    return np.stack(scores), paths


def student_fit(x, gap, target, arm):
    return hard_fit(x, gap) if arm == 'Direct' else sm.fit_soft(x, target, previous.weights(gap), .001)


def pilot(binding):
    _, d = bound(binding)
    assert not (OUT / 'pilot.json').exists() and not (OUT / 'fit_started.json').exists()
    torch = old.gpu()
    controls = old.old_replay(d, torch)
    _, inverse, sizes = np.unique(d['group_ids'], return_inverse=True, return_counts=True)
    fit = d['folds'][0][0]
    ids = fit[sizes[inverse[fit]] == 1][:192]
    assert len(ids) == 192
    train, valid = ids[:128], ids[128:]
    with threadpool_limits(limits=2):
        logits, artifacts = teacher_pair(d, train, valid, 'pilot', torch)
        probs = expit(logits.astype(np.float64))
        y = (d['gap'][valid] > 0).astype(float)
        targets = np.vstack([y, .5 * y + .5 * probs])
        save(OUT / 'pilot_targets.npz', train_indices=train, target_indices=valid,
             teacher_logits=logits, teacher_probability=probs, targets=targets)
        artifacts.append(OUT / 'pilot_targets.npz')
        for arm, target in zip(ARMS, targets):
            model = student_fit(d['features'][valid], d['gap'][valid], target, arm)
            artifacts += store_model('pilot_S_' + arm, model, dict(role='student', arm=arm))
            scores = old.native(d['features'][valid], model['coef'], model['intercept'], torch)
            assert scores.shape == (64,) and np.isfinite(scores).all()
    value = dict(status='passed_five_pilot_fits_and_480_native_controls', protocol_sha256=binding,
        pilot_solves=5, teacher_train_rows=128, target_and_student_rows=64,
        original_native_control_queries=480, original_controls=controls,
        artifact_sha256={str(path): sha(path) for path in artifacts}, new_policy_effects=False,
        new_encoder_forwards=0, new_external_calls=0)
    bound(binding)
    write(OUT / 'pilot.json', value)
    print(json.dumps(dict(status=value['status'], pilot_solves=5)), flush=True)


def bootstrap(groups, values):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=values[:, j]) for j in range(5)])
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    draws = np.empty((20000, 5))
    for start in range(0, len(draws), 100):
        ix = rng.integers(len(sizes), size=(min(100, len(draws) - start), len(sizes)))
        draws[start:start + len(ix)] = sums[ix].sum(axis=1) / sizes[ix].sum(axis=1)[:, None]
    save(OUT / 'bootstrap.npz', draws=draws)
    return np.quantile(draws, [.005, .995], axis=0).T


def run(binding):
    p, d = bound(binding)
    pilot_record = read(OUT / 'pilot.json')
    assert pilot_record['status'] == 'passed_five_pilot_fits_and_480_native_controls'
    assert pilot_record['protocol_sha256'] == binding
    assert all(sha(path) == digest for path, digest in pilot_record['artifact_sha256'].items())
    assert not (OUT / 'fit_started.json').exists()
    write(OUT / 'fit_started.json', dict(protocol_sha256=binding, started_at_unix=time.time()))
    started = time.perf_counter()
    torch = old.gpu()
    x, gap = d['features'], d['gap']
    artifacts, heads, records = [], [], []
    with threadpool_limits(limits=2), (OUT / 'solutions.jsonl').open('x', encoding='utf-8') as journal:
        for f, (fit, cal, test) in enumerate(d['folds']):
            assignment = old.assignment(d['group_ids'][fit], f)
            logits, visits = np.full((2, len(fit)), np.nan, np.float32), np.zeros(len(fit), int)
            for k in range(3):
                train, valid = fit[assignment != k], fit[assignment == k]
                assert not set(d['group_ids'][train]) & set(d['group_ids'][np.r_[valid, cal, test]])
                scores, paths = teacher_pair(d, train, valid, f'fold{f}_inner{k}', torch, journal)
                logits[:, assignment == k] = scores
                visits[assignment == k] += 1
                artifacts += paths
            assert np.all(visits == 1) and np.isfinite(logits).all()
            with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                assert np.array_equal(logits[0], z['inner_validation_native_logits'][1]), 'Pre Teacher must reproduce original own-fit OOF logits at lambda .001'
            probs = expit(logits.astype(np.float64))
            y = (gap[fit] > 0).astype(float)
            targets = np.vstack([y, .5 * y + .5 * probs])
            path = OUT / f'fold{f}_targets.npz'
            save(path, fit_indices=fit, inner_assignment=assignment, teacher_logits=logits,
                 teacher_probability=probs, targets=targets, weights=previous.weights(gap[fit]))
            artifacts.append(path)
            models, fit_scores, cal_scores = [], [], []
            for arm, target in zip(ARMS, targets):
                model = student_fit(x[fit], gap[fit], target, arm)
                artifacts += store_model(f'fold{f}_S_{arm}', model,
                    dict(role='student', fold=f, arm=arm, targets=f'fold{f}_targets.npz'), journal)
                fs = old.native(x[fit], model['coef'], model['intercept'], torch)
                cs = old.native(x[cal], model['coef'], model['intercept'], torch)
                if arm == 'Direct':
                    oc, ob = previous.original_head(f, torch)
                    assert np.array_equal(model['coef'].astype(np.float32), oc)
                    assert np.float32(model['intercept']) == np.float32(ob)
                    with np.load(BASE / f'layer_pooling_v1/fold{f}_M6_fit.npz', allow_pickle=False) as z:
                        assert np.array_equal(fs, z['final_native_fit_logits'])
                    assert np.array_equal(cs, d['L_cal'][f])
                models.append(model)
                fit_scores.append(fs)
                cal_scores.append(cs)
            path = OUT / f'fold{f}_fit_cal.npz'
            save(path, fit_indices=fit, cal_indices=cal,
                 fit_scores=np.stack(fit_scores), cal_scores=np.stack(cal_scores))
            artifacts.append(path)
            heads.append(models)
            records.append(dict(fold=f, teacher_OOF_BCE=[previous.bce(row, gap[fit]) for row in logits],
                fit_BCE=[previous.bce(row, gap[fit]) for row in fit_scores],
                cal_BCE=[previous.bce(row, gap[cal]) for row in cal_scores], direct_original_replay_exact=True))
            print(json.dumps(dict(status='fold_targets_and_three_students_complete', fold=f,
                                  formal_solves_completed=(f + 1) * 9)), flush=True)
    artifacts.append(OUT / 'solutions.jsonl')
    write(OUT / 'fit_completion.json', dict(status='all_30_teacher_and_15_student_heads_frozen_before_test_prediction',
        protocol_sha256=binding, formal_solves=45, teacher_heads=30, student_heads=15,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(path): sha(path) for path in artifacts}))
    all_scores, fold_ids = np.full((3, 9600), np.nan, np.float32), np.full(9600, -1, int)
    for f, (_, _, test) in enumerate(d['folds']):
        for j, model in enumerate(heads[f]):
            all_scores[j, test] = old.native(x[test], model['coef'], model['intercept'], torch)
        fold_ids[test] = f
    assert np.isfinite(all_scores).all() and np.all(fold_ids >= 0)
    assert np.array_equal(all_scores[0], d['L_scores']), 'Direct must reproduce every original test logit'
    actions = {name: all_scores[j] > 0 for j, name in enumerate(ARMS)}
    save(OUT / 'predictions.npz', query_ids=d['query_ids'], group_ids=d['group_ids'], utility=d['utility'],
         arm_scores=all_scores, fold_id=fold_ids, **actions)
    write(OUT / 'predictions_frozen.json', dict(status='all_9600_OOF_student_predictions_before_effects',
        protocol_sha256=binding, predictions_sha256=sha(OUT / 'predictions.npz'),
        fit_completion_sha256=sha(OUT / 'fit_completion.json')))
    actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
    quality = {name: np.where(action, d['utility'][:, 0], d['utility'][:, 1]) for name, action in actions.items()}
    pairs = [('Pre', 'Direct'), ('Probe', 'Pre'), ('Probe', 'Direct'), ('Probe', 'Dense'), ('Probe', 'BM25')]
    values = np.column_stack([quality[a] - quality[b] for a, b in pairs])
    with threadpool_limits(limits=2):
        intervals = bootstrap(d['group_ids'], values)
    primary = {name: dict(mean=float(values[:, j].mean()), interval=intervals[j].tolist()) for j, name in enumerate(PRIMARY)}
    transfer = all(primary[name]['interval'][0] > 0 and primary[name]['mean'] >= .002 for name in PRIMARY[1:3])
    candidate = transfer and all(primary[name]['interval'][0] > 0 and primary[name]['mean'] >= .01 for name in PRIMARY[3:])
    for f, (_, _, test) in enumerate(d['folds']):
        records[f]['primary_means'] = {name: float(values[test, j].mean()) for j, name in enumerate(PRIMARY)}
        records[f]['test_BCE'] = [previous.bce(row[test], gap[test]) for row in all_scores]
    decision = ('PREPARE_FROZEN_STUDENT_INDEPENDENT_CONFIRMATION' if candidate else
                'TRANSFER_INCREMENT_ONLY_NO_QUALIFIED_CANDIDATE' if transfer else
                'END_FIXED_SOFT_BCE_STUDENT_RECIPE_NO_CONFIRMED_TRANSFER')
    result = dict(status='complete_fixed_student_transfer_pending_separate_check', protocol_sha256=binding,
        primary=primary, policy={name: old.policy_summary(actions[name], d['utility'], gap) for name in POLICIES},
        transfer_gate=bool(transfer), candidate_preparation_gate=bool(candidate), decision=decision,
        folds=records, formal_solves=45, teacher_heads=30, student_heads=15, pilot_solves=5,
        all_direct_original_test_logits_exact=True, new_thresholds=0, new_encoder_forwards=0,
        new_external_calls=0, runtime_available=False, core_goal_achieved=False,
        predictions_frozen_sha256=sha(OUT / 'predictions_frozen.json'),
        bootstrap_sha256=sha(OUT / 'bootstrap.npz'), elapsed_seconds=time.perf_counter() - started, scope=p['scope'])
    bound(binding)
    write(OUT / 'results.json', result)
    print(json.dumps(dict(status=result['status'], primary=primary, decision=decision)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'pilot', 'run'])
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    try:
        {'freeze': lambda: freeze(), 'pilot': lambda: pilot(args.protocol_sha), 'run': lambda: run(args.protocol_sha)}[args.mode]()
    except Exception:
        path = OUT / (args.mode + '_failure.json')
        if OUT.exists() and not path.exists():
            write(path, dict(protocol_sha256=args.protocol_sha, traceback=traceback.format_exc()))
        raise
