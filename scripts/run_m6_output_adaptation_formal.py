"""Fixed four-epoch M6-output adaptation, selection freeze, and evaluation.

Training primitives and all frozen configuration come from the core module.
This entry point never selects a learning rate, scope, threshold, or test epoch.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

import run_m6_output_adaptation as core

ARMS = ('A', 'C')
POLICIES = ('A', 'C', 'B', 'Dense', 'BM25')
TRACE_COLUMNS = ('supervised_loss', 'head_penalty', 'total_loss',
                 'encoder_gradnorm_before_clip', 'head_gradnorm_before_clip',
                 'lr_multiplier', 'zero_weight_batch')


def choose_epoch(gains, tie_atol):
    """Choose earliest candidate within tolerance of the global maximum."""
    gains = np.asarray(gains, dtype=np.float64)
    assert gains.ndim == 1 and len(gains) > 0 and np.isfinite(gains).all()
    assert np.isfinite(tie_atol) and tie_atol >= 0
    maximum = float(gains.max())
    eligible = np.flatnonzero(maximum - gains <= tie_atol)
    selected = int(eligible[0])
    return dict(selected_epoch=selected, selected_cal_gain=float(gains[selected]),
                best_cal_gain=maximum, eligible_epochs=eligible.tolist(),
                candidate_count=len(gains), tie_atol=float(tie_atol))


def group_bootstrap(groups, values, draws, seed):
    """Shared group draws; denominator is the number of sampled queries."""
    groups, values = np.asarray(groups), np.asarray(values, dtype=np.float64)
    assert groups.ndim == 1 and values.ndim == 2 and len(groups) == len(values) > 0
    assert np.isfinite(values).all() and draws > 0
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=values[:, j])
                            for j in range(values.shape[1])])
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=np.float64)
    with threadpool_limits(limits=2):
        for start in range(0, draws, 100):
            count = min(100, draws - start)
            indices = rng.integers(len(sizes), size=(count, len(sizes)))
            samples[start:start + count] = (sums[indices].sum(axis=1)
                / sizes[indices].sum(axis=1)[:, None])
    return samples


def self_test():
    # Global-max tolerance is not a transitive, sequential tie comparison.
    selection = choose_epoch([0., .75e-12, 1.5e-12, -1., -2.], 1e-12)
    assert selection['selected_epoch'] == 1 and selection['eligible_epochs'] == [1, 2]
    assert choose_epoch([.3, .1, .2, .3, .0], 0.)['selected_epoch'] == 0
    assert choose_epoch([0., -1., -2., -3., -4.], 1e-12)['selected_epoch'] == 0
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    values = np.arange(24, dtype=np.float64).reshape(6, 4) / 100.
    samples = group_bootstrap(groups, values, 107, 9281)
    rng = np.random.default_rng(9281)
    explicit = []
    unique = np.unique(groups)
    for _ in range(107):
        drawn = rng.integers(len(unique), size=len(unique))
        rows = np.concatenate([np.flatnonzero(groups == unique[k]) for k in drawn])
        explicit.append(values[rows].mean(axis=0))
    error = float(np.max(np.abs(samples - np.asarray(explicit))))
    assert error < 1e-15
    return dict(status='passed_formal_selection_and_group_bootstrap_synthetic_checks',
                global_max_tie_check=True, epoch0_retained=True,
                unequal_group_denominator_max_error=error, real_data_reads=0,
                model_fits=0, GPU_forwards=0)


def _pilot_gate(binding):
    pilot_path = core.OUT / 'pilot.json'
    checked_path = core.OUT / 'pilot_separate_checks.json'
    pilot, checked = core.read(pilot_path), core.read(checked_path)
    assert pilot['protocol_sha256'] == checked['protocol_sha256'] == binding
    assert checked['status'] == 'passed_independent_M6_output_adaptation_pilot_checks'
    assert pilot['status'] == 'complete_M6_output_pilot_pending_independent_check'
    assert pilot['formal_training_started'] is False
    assert pilot['cal_quality_evaluations'] == pilot['test_quality_evaluations'] == 0
    assert checked['pilot_sha256'] == core.sha(pilot_path)
    for path, expected in pilot['artifact_sha256'].items():
        assert core.sha(path) == expected, path
    return core.sha(pilot_path), core.sha(checked_path)


def _candidate(encoder, head, d, fold, arm, epoch, binding, torch, artifacts):
    fit, cal, _ = d['folds'][fold]
    indices = np.r_[fit, cal]
    features, scores = core.infer(encoder, head, d['tokens'], indices, torch)
    fit_scores, cal_scores = scores[:len(fit)], scores[len(fit):]
    if epoch == 0:
        assert np.array_equal(features, d['features'][indices])
        with np.load(core.BASE / f'layer_pooling_v1/fold{fold}_M6_fit.npz',
                     allow_pickle=False) as old_fit:
            assert np.array_equal(fit, old_fit['fit_indices'])
            assert np.array_equal(fit_scores, old_fit['final_native_fit_logits'])
        assert np.array_equal(cal_scores, d['L_cal'][fold])
    if arm == 'C':
        assert np.array_equal(features, d['features'][indices])
    path = core.OUT / f'fold{fold}_{arm}_epoch{epoch}.pt'
    core.save_checkpoint(path, encoder, head, arm, fold, epoch, binding)
    artifacts[str(path)] = core.sha(path)
    gap_cal = d['gap'][cal]
    record = dict(epoch=epoch, checkpoint=str(path), checkpoint_sha256=artifacts[str(path)],
        parameters_sha256=core.digest_parameters(core.named(encoder, head)),
        fit_BCE=core.bce(fit_scores, d['gap'][fit]),
        cal_BCE=core.bce(cal_scores, gap_cal),
        fit_gain=float(np.where(fit_scores > 0, d['gap'][fit], 0.).mean()),
        cal_gain=float(np.where(cal_scores > 0, gap_cal, 0.).mean()),
        fit_bm25_count=int(np.count_nonzero(fit_scores > 0)),
        cal_bm25_count=int(np.count_nonzero(cal_scores > 0)),
        initial_M6_feature_fit_cal_score_exact=(epoch == 0))
    return record, fit_scores, cal_scores


def _train_arm(d, fold, arm, binding, torch, artifacts):
    fit, cal, _ = d['folds'][fold]
    started = time.perf_counter()
    encoder, head = core.model(torch, fold, arm)
    parameters = core.named(encoder, head)
    trainable = [(n, p) for n, p in parameters if p.requires_grad]
    frozen = [(n, p) for n, p in parameters if not p.requires_grad]
    before = {n: p.detach().cpu().clone() for n, p in parameters}
    frozen_before = core.digest_parameters(frozen)
    initial_hash = core.digest_parameters(parameters)
    optimizer = core.optimizer_for(encoder, head, arm, torch)
    assert not optimizer.state
    candidates, fit_logits, cal_logits, training = [], [], [], []
    for epoch in range(core.CONFIG['epochs'] + 1):
        if epoch:
            summary, first, table, order = core.train_epoch(
                encoder, head, arm, d['tokens'], fit, d['gap'][fit], optimizer,
                fold, epoch - 1, torch)
            assert summary['optimizer_steps'] == core.CONFIG['steps_per_epoch']
            assert table.shape == (core.CONFIG['steps_per_epoch'], len(TRACE_COLUMNS))
            assert np.array_equal(np.sort(order), np.sort(fit))
            assert frozen_before == core.digest_parameters(frozen)
            gradients = first.pop('module_gradient_norm')
            trace_path = core.OUT / f'fold{fold}_{arm}_epoch{epoch}_training.npz'
            core.save(trace_path, step_trace=table, training_order=order,
                      **{'first_' + k: v for k, v in first.items()})
            artifacts[str(trace_path)] = core.sha(trace_path)
            training.append(dict(summary, trace_path=str(trace_path),
                trace_sha256=artifacts[str(trace_path)],
                first_parameter_gradient_norm=gradients))
        row, fs, cs = _candidate(encoder, head, d, fold, arm, epoch,
                                 binding, torch, artifacts)
        candidates.append(row); fit_logits.append(fs); cal_logits.append(cs)
        print(json.dumps(dict(status='formal_candidate_complete', fold=fold, arm=arm,
                              epoch=epoch, cumulative_optimizer_steps=epoch * 768)), flush=True)
    assert len(training) == core.CONFIG['epochs']
    assert sum(row['optimizer_steps'] for row in training) == core.CONFIG['total_steps']
    selection = choose_epoch([row['cal_gain'] for row in candidates], core.CONFIG['cal_tie_atol'])
    chosen = candidates[selection['selected_epoch']]
    fit_cal_path = core.OUT / f'fold{fold}_{arm}_fit_cal.npz'
    core.save(fit_cal_path, fit_indices=fit.astype(np.int64), cal_indices=cal.astype(np.int64),
              epochs=np.arange(core.CONFIG['epochs'] + 1, dtype=np.int64),
              fit_logits=np.stack(fit_logits), cal_logits=np.stack(cal_logits))
    artifacts[str(fit_cal_path)] = core.sha(fit_cal_path)
    changes = {n: float((p.detach().cpu() - before[n]).abs().max()) for n, p in parameters}
    assert all(changes[n] == 0 for n, _ in frozen)
    assert any(changes[n] > 0 for n, _ in trainable if n.startswith('head.'))
    if arm == 'A':
        assert any(changes[n] > 0 for n, _ in trainable if n.startswith('encoder.'))
    row = dict(fold=fold, arm=arm, initial_parameters_sha256=initial_hash,
        initial_head_path=str(core.BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt'),
        initial_head_sha256=core.sha(core.BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt'),
        trained_parameter_count=sum(p.numel() for _, p in trainable),
        trained_names=[n for n, _ in trainable], frozen_names=[n for n, _ in frozen],
        frozen_before_sha256=frozen_before, frozen_after_sha256=core.digest_parameters(frozen),
        final_parameter_max_changes=changes, final_parameters_sha256=core.digest_parameters(parameters),
        epochs_completed=core.CONFIG['epochs'], optimizer_steps=core.CONFIG['total_steps'],
        candidates=candidates, selection=selection, epoch_training=training,
        selected_checkpoint=chosen['checkpoint'], selected_checkpoint_sha256=chosen['checkpoint_sha256'],
        fit_cal_path=str(fit_cal_path), fit_cal_sha256=artifacts[str(fit_cal_path)],
        trace_columns=list(TRACE_COLUMNS), BCE_denominator='sum of weights within the scored partition',
        training_seconds=time.perf_counter() - started, training_encoder_query_forwards=len(fit) * 4,
        candidate_encoder_query_forwards=(len(fit) + len(cal)) * 5,
        outer_test_quality_evaluations=0)
    record_path = core.OUT / f'fold{fold}_{arm}.json'
    core.write(record_path, row); artifacts[str(record_path)] = core.sha(record_path)
    del optimizer, encoder, head, parameters, trainable, frozen, before
    gc.collect(); torch.cuda.empty_cache()
    return row, dict(fold=fold, arm=arm, epoch=selection['selected_epoch'],
                    checkpoint=chosen['checkpoint'], checkpoint_sha256=chosen['checkpoint_sha256'],
                    record_path=str(record_path), record_sha256=artifacts[str(record_path)])


def run(binding):
    protocol, d = core.bound(binding)
    pilot_sha, pilot_check_sha = _pilot_gate(binding)
    start_path = core.OUT / 'formal_started.json'
    core.write(start_path, dict(protocol_sha256=binding, pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        pilot_sha256=pilot_sha, pilot_separate_checks_sha256=pilot_check_sha))
    started = time.perf_counter()
    torch = core.old.gpu()
    torch.cuda.reset_peak_memory_stats()
    records, selected, artifacts = [], [], {str(start_path): core.sha(start_path)}
    for fold in range(5):
        pair = {}
        for arm in ARMS:
            row, choice = _train_arm(d, fold, arm, binding, torch, artifacts)
            pair[arm] = row; selected.append(choice)
        assert pair['A']['initial_parameters_sha256'] == pair['C']['initial_parameters_sha256']
        assert ([row['batch_order_sha256'] for row in pair['A']['epoch_training']]
                == [row['batch_order_sha256'] for row in pair['C']['epoch_training']])
        records.append(dict(fold=fold, arms=pair))
    assert len(selected) == 10 and len(artifacts) == 111
    for path, expected in artifacts.items():
        assert core.sha(path) == expected, path
    core.bound(binding)
    frozen_path = core.OUT / 'all_selected_frozen.json'
    core.write(frozen_path, dict(status='all_ten_selected_endpoints_before_any_formal_test',
        protocol_sha256=binding, formal_training_trajectories=10, encoder_training_trajectories=5,
        epochs_completed=40, optimizer_steps=30720, candidate_checkpoints=50,
        selected=selected, artifact_sha256=artifacts,
        pilot_sha256=pilot_sha, pilot_separate_checks_sha256=pilot_check_sha,
        test_encoder_query_forwards=0, elapsed_seconds=time.perf_counter() - started))
    frozen_sha = core.sha(frozen_path)
    n = len(d['gap'])
    scores = {arm: np.empty(n, dtype=np.float32) for arm in ARMS}
    fold_id = np.full(n, -1, dtype=np.int64)
    visits = {arm: np.zeros(n, dtype=np.int64) for arm in ARMS}
    test_artifacts = {}
    for item in selected:
        fold, arm = item['fold'], item['arm']
        fit, _, test = d['folds'][fold]
        assert core.sha(frozen_path) == frozen_sha
        assert core.sha(item['checkpoint']) == item['checkpoint_sha256']
        encoder, head, checkpoint = core.restore(torch, item['checkpoint'], binding)
        assert (checkpoint['fold'], checkpoint['arm'], checkpoint['epoch']) == (fold, arm, item['epoch'])
        _, replay = core.infer(encoder, head, d['tokens'], fit[:8], torch)
        row = records[fold]['arms'][arm]
        with np.load(row['fit_cal_path'], allow_pickle=False) as cached:
            assert np.array_equal(replay, cached['fit_logits'][item['epoch'], :8])
        features, native = core.infer(encoder, head, d['tokens'], test, torch)
        if arm == 'C' or item['epoch'] == 0:
            assert np.array_equal(features, d['features'][test])
        if item['epoch'] == 0:
            assert np.array_equal(native, d['L_scores'][test])
        scores[arm][test] = native; visits[arm][test] += 1; fold_id[test] = fold
        path = core.OUT / f'fold{fold}_{arm}_test.npz'
        core.save(path, test_indices=test.astype(np.int64), features=features, scores=native)
        test_artifacts[str(path)] = core.sha(path)
        row['selected_fit8_restore_exact'] = True
        row['test_archive'] = str(path)
        del encoder, head, checkpoint, features, native
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(dict(status='selected_test_predictions_complete', fold=fold, arm=arm,
                              selected_epoch=item['epoch'])), flush=True)
    assert all(np.all(v == 1) for v in visits.values()) and np.all(fold_id >= 0)
    actions = {arm: values > 0 for arm, values in scores.items()}
    actions.update(B=d['L_scores'] > 0, Dense=np.zeros(n, dtype=bool), BM25=np.ones(n, dtype=bool))
    predictions_path = core.OUT / 'predictions.npz'
    core.save(predictions_path, query_ids=d['query_ids'], group_ids=d['group_ids'],
              utility=d['utility'], fold_id=fold_id, A_scores=scores['A'], C_scores=scores['C'],
              B_scores=d['L_scores'], **actions)
    prediction_freeze = core.OUT / 'predictions_frozen.json'
    core.write(prediction_freeze, dict(status='all_selected_predictions_before_effects',
        protocol_sha256=binding, all_selected_frozen_sha256=frozen_sha,
        predictions_sha256=core.sha(predictions_path), test_artifact_sha256=test_artifacts,
        selected_fit8_replays=80, selected_test_encoder_query_forwards=19200))
    utility = {name: np.where(actions[name], d['utility'][:, 0], d['utility'][:, 1]) for name in POLICIES}
    values = np.column_stack([utility['A'] - utility[name] for name in ('C', 'B', 'Dense', 'BM25')])
    draws = group_bootstrap(d['group_ids'], values, core.CONFIG['bootstrap_draws'], core.CONFIG['bootstrap_seed'])
    bootstrap_path = core.OUT / 'bootstrap.npz'
    core.save(bootstrap_path, draws=draws)
    intervals = np.quantile(draws, core.CONFIG['quantiles'], axis=0).T
    primary = {name: dict(mean=float(values[:, j].mean()), interval=intervals[j].tolist())
               for j, name in enumerate(core.PRIMARY)}
    recipe = all(primary[k]['mean'] >= core.CONFIG['minimum_increment']
                 and primary[k]['interval'][0] > 0 for k in core.PRIMARY[:2])
    candidate = recipe and all(primary[k]['mean'] >= core.CONFIG['minimum_fixed_gain']
                               and primary[k]['interval'][0] > 0 for k in core.PRIMARY[2:])
    for fold, (_, _, test) in enumerate(d['folds']):
        for arm in ARMS:
            records[fold]['arms'][arm]['selected_test_BCE'] = core.bce(scores[arm][test], d['gap'][test])
        records[fold]['test_BCE'] = {name: core.bce(value[test], d['gap'][test])
            for name, value in dict(A=scores['A'], C=scores['C'], B=d['L_scores']).items()}
        records[fold]['policy'] = {name: core.old.policy_summary(actions[name][test],
            d['utility'][test], d['gap'][test]) for name in POLICIES}
        records[fold]['primary_means'] = {name: float(values[test, j].mean())
                                          for j, name in enumerate(core.PRIMARY)}
    torch.cuda.synchronize()
    result = dict(status='complete_fixed_M6_output_adaptation_pending_independent_check',
        protocol_sha256=binding, primary=primary,
        policy={name: core.old.policy_summary(actions[name], d['utility'], d['gap']) for name in POLICIES},
        folds=records, recipe_gate=recipe, candidate_preparation_gate=candidate,
        decision=('ADVANCE_FIXED_M6_OUTPUT_ADAPTATION_CANDIDATE_PREPARATION' if candidate
                  else 'FIXED_M6_OUTPUT_ADAPTATION_INCREMENT_ONLY' if recipe
                  else 'END_FIXED_M6_OUTPUT_ADAPTATION_NO_CONFIRMED_INCREMENT'),
        formal_training_trajectories=10, new_encoder_fits=5, head_training_trajectories=10,
        epochs_completed=40, optimizer_steps=30720, candidate_checkpoints=50,
        selected_epochs={arm: [records[f]['arms'][arm]['selection']['selected_epoch'] for f in range(5)] for arm in ARMS},
        encoder_query_forwards=dict(training=245760, candidate_fit_cal=384000,
                                    selected_fit_replay=80, selected_test=19200, total=649040),
        pilot_optimizer_steps=16, pilot_encoder_query_forwards=core.read(core.OUT / 'pilot.json')['encoder_query_forwards'],
        new_api_calls=0, runtime_deployed=False, core_goal_achieved=False,
        all_selected_frozen_sha256=frozen_sha, predictions_frozen_sha256=core.sha(prediction_freeze),
        bootstrap_sha256=core.sha(bootstrap_path), elapsed_seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_rss_bytes=psutil.Process().memory_info().rss,
        BCE_denominator='sum of weights within the scored partition', scope=protocol['scope'])
    core.bound(binding)
    core.write(core.OUT / 'results.json', result)
    print(json.dumps(dict(status=result['status'], recipe_gate=recipe,
                          candidate_preparation_gate=candidate, primary=primary,
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--protocol-sha256')
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        try:
            run(arguments.protocol_sha256)
        except Exception:
            failure = core.OUT / 'formal_failure.json'
            if core.OUT.exists() and not failure.exists():
                core.write(failure, dict(status='formal_interrupted_no_automatic_retry',
                    protocol_sha256=arguments.protocol_sha256,
                    traceback=traceback.format_exc(), recorded_at_utc=datetime.now(timezone.utc).isoformat()))
            raise
