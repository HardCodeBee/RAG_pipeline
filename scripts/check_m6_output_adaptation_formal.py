"""Independent formal M6-output adaptation audit; never import its runners.

Checks every saved endpoint, trace, selection and OOF effect. A full twelve-
block reference hooks layer six on outcome-independent samples and replays
each epoch's first backward pass, without optimizer steps. The separate pilot
receipt covers the independently reproduced initial eight optimizer steps.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import check_m6_output_adaptation_pilot as ref
import check_m6_student_transfer as numeric

ROOT, BASE, OUT, MODEL = ref.ROOT, ref.BASE, ref.OUT, ref.MODEL
ARMS = ('A', 'C')
POLICIES = ('A', 'C', 'B', 'Dense', 'BM25')
PRIMARY = ('A_minus_C', 'A_minus_B', 'A_minus_Dense', 'A_minus_BM25')
TRACE_COLUMNS = ('supervised_loss', 'head_penalty', 'total_loss',
    'encoder_gradnorm_before_clip', 'head_gradnorm_before_clip',
    'lr_multiplier', 'zero_weight_batch')
FIRST_NAMES = ('indices', 'features', 'logits', 'targets', 'weights', 'normalizer',
    'head_weight', 'head_bias', 'head_weight_grad', 'head_bias_grad')
SAMPLE_RULE = 'm6_output_adaptation_formal_check_v1|fold={fold}|partition={part}|query={query}'
sha, read = ref.sha, ref.read


def array(value, shape, dtype, name):
    result = np.asarray(value)
    assert result.shape == shape and result.dtype == np.dtype(dtype), name
    assert np.isfinite(result).all(), name
    return result


def close(actual, expected, errors, name, atol=2e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(abs(a.astype(float) - b.astype(float)), initial=0.))
    assert error <= atol, f'{name}: {error:.17g} exceeds {atol:.17g}'
    errors[name] = error


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def canonical(mapping):
    return {Path(path).resolve(): digest for path, digest in mapping.items()}


def check_files(mapping, expected_paths=None):
    assert isinstance(mapping, dict) and mapping
    if expected_paths is not None:
        assert set(canonical(mapping)) == {Path(p).resolve() for p in expected_paths}
    ref.require_bindings(mapping)


def candidate_choice(gains):
    values = np.asarray(gains, dtype=np.float64)
    assert values.shape == (5,) and np.isfinite(values).all()
    maximum = max(map(float, values))
    eligible = [i for i, value in enumerate(values) if maximum - float(value) <= 1e-12]
    chosen = eligible[0]
    return dict(selected_epoch=chosen, selected_cal_gain=float(values[chosen]),
        best_cal_gain=maximum, eligible_epochs=eligible, candidate_count=5, tie_atol=1e-12)


def schedule(epoch):
    result = []
    for step in range((epoch - 1) * 768, epoch * 768):
        current = step + 1
        result.append(current / 307 if current < 307 else (3072 - current) / 2765)
    return np.asarray(result, dtype=np.float64)


def partition_bce(scores, gap):
    weights = [abs(float(g)) if abs(g) > 1e-12 else 0. for g in gap]
    denominator = math.fsum(weights)
    assert denominator > 0
    numerator = math.fsum(w * numeric.signed_softplus(-float(s) if g > 0 else float(s))
                         for s, g, w in zip(scores, gap, weights))
    return numerator / denominator


def gain(scores, gap):
    return math.fsum(float(g) for s, g in zip(scores, gap) if s > 0) / len(gap)


def sample_positions(query_ids, indices, fold, part):
    return np.asarray(sorted(range(len(indices)), key=lambda i: (
        hashlib.sha256(SAMPLE_RULE.format(fold=fold, part=part,
            query=str(query_ids[indices[i]])).encode()).digest(), int(indices[i])))[:32], dtype=np.int64)


def gates(primary):
    recipe = all(primary[k]['mean'] >= .002 and primary[k]['interval'][0] > 0 for k in PRIMARY[:2])
    candidate = recipe and all(primary[k]['mean'] >= .01 and primary[k]['interval'][0] > 0 for k in PRIMARY[2:])
    decision = ('ADVANCE_FIXED_M6_OUTPUT_ADAPTATION_CANDIDATE_PREPARATION' if candidate
                else 'FIXED_M6_OUTPUT_ADAPTATION_INCREMENT_ONLY' if recipe
                else 'END_FIXED_M6_OUTPUT_ADAPTATION_NO_CONFIRMED_INCREMENT')
    return bool(recipe), bool(candidate), decision


def protocol_inputs(binding):
    assert binding and sha(OUT / 'protocol.json') == binding
    protocol = read(OUT / 'protocol.json')
    assert protocol['status'] == 'frozen_before_M6_output_adaptation_pilot_and_formal_training'
    assert protocol['config'] == ref.expected_config() and protocol['primary'] == list(PRIMARY)
    assert protocol['versions'] == {name: importlib.metadata.version(name)
        for name in ('numpy', 'torch', 'transformers', 'scipy')}
    assert protocol['queries'] == 9600
    sources = [ROOT / 'scripts' / name for name in ('run_m6_output_adaptation.py',
        'run_m6_output_adaptation_formal.py', 'check_m6_output_adaptation_pilot.py',
        'run_m6_objective_readout.py', 'm6_objective_math.py')]
    sources.append(ROOT / 'analysis/hotpotqa_router/m6_output_adaptation_plan_20260915.md')
    inputs = [BASE / name for name in ('layer_pooling_v1/features.npz',
        'layer_pooling_v1/predictions.npz', 'layer_pooling_v1/tokens.npz',
        'e02_results/fold_indices.npz', 'm6_pooled_offset_v1/cal_logits.npz',
        'layer_pooling_v1/completion_record.json', 'm6_pooled_offset_v1/completion_record.json')]
    inputs += [BASE / f'layer_pooling_v1/fold{f}_M6_{suffix}' for f in range(5)
               for suffix in ('head.pt', 'fit.npz')]
    inputs += [MODEL / name for name in ('model.safetensors', 'config.json')]
    check_files(protocol['source_sha256'], sources)
    check_files(protocol['input_sha256'], inputs)
    historical = {}
    for directory in ('layer_pooling_v1', 'm6_pooled_offset_v1'):
        historical.update(canonical(read(BASE / directory / 'completion_record.json')['artifact_sha256']))
    for path in inputs:
        if path.name != 'completion_record.json' and path.parent != MODEL and path.name != 'fold_indices.npz':
            assert path.resolve() in historical and sha(path) == historical[path.resolve()]
    assert sha(BASE / 'e02_results/fold_indices.npz') == numeric.EXPECTED_FOLDS_SHA256
    return protocol


def load_data(torch):
    features = archive(BASE / 'layer_pooling_v1/features.npz')
    x = array(features['M6'], (9600, 384), np.float32, 'original features')
    qids, groups = features['query_ids'], features['group_ids']
    assert qids.shape == groups.shape == (9600,) and len(set(qids)) == 9600 and len(set(groups)) == 9559
    assert np.max(abs(np.sqrt(np.einsum('nd,nd->n', x.astype(float), x.astype(float))) - 1)) < 2e-6
    old = archive(BASE / 'layer_pooling_v1/predictions.npz')
    assert np.array_equal(qids, old['query_ids']) and np.array_equal(groups, old['group_ids'])
    utility = array(old['utility'], (9600, 2), np.float64, 'original utility')
    assert np.all((utility >= 0) & (utility <= 1))
    original_scores = array(old['M6'], (9600,), np.float64, 'historical M6 scores')
    assert np.array_equal(original_scores, ref.bf16(original_scores).astype(np.float64))
    tokens = archive(BASE / 'layer_pooling_v1/tokens.npz')
    assert set(tokens) == {'input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask'}
    for key, value in tokens.items():
        array(value, (9600, 128), np.int64, 'tokens ' + key)
    splits = archive(BASE / 'e02_results/fold_indices.npz')
    cals = archive(BASE / 'm6_pooled_offset_v1/cal_logits.npz')
    folds, visit, heads, fit_scores, cal_scores = [], np.zeros(9600, np.int64), [], [], []
    for fold in range(5):
        parts = tuple(splits[f'fold{fold}_{part}'] for part in ('fit', 'calibration', 'test'))
        assert [len(v) for v in parts] == [6144, 1536, 1920]
        assert all(v.ndim == 1 and v.dtype.kind in 'iu' and len(set(v)) == len(v) for v in parts)
        assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(9600))
        group_sets = [set(groups[v]) for v in parts]
        assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        fit, cal, test = parts
        visit[test] += 1
        fitted = archive(BASE / f'layer_pooling_v1/fold{fold}_M6_fit.npz')
        assert np.array_equal(fit, fitted['fit_indices'])
        assert np.array_equal(numeric.inner_groups(groups[fit], fold), fitted['inner_assignment'])
        fit_scores.append(array(fitted['final_native_fit_logits'], (6144,), np.float32, 'original fit scores'))
        assert np.array_equal(cal, cals[f'fold{fold}_cal_indices'])
        cal_scores.append(array(cals[f'fold{fold}_O'], (1536,), np.float32, 'original cal scores'))
        head = torch.load(BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt', map_location='cpu', weights_only=True)
        assert set(head) == {'weight', 'bias'}
        array(head['weight'].numpy(), (1, 384), np.float32, 'initial head weight')
        array(head['bias'].numpy(), (1,), np.float32, 'initial head bias')
        heads.append(head); folds.append(parts)
    assert np.all(visit == 1)
    return dict(features=x, query_ids=qids, group_ids=groups, utility=utility,
        gap=utility[:, 0] - utility[:, 1], B_scores=original_scores, tokens=tokens,
        folds=folds, heads=heads, fit_scores=fit_scores, cal_scores=cal_scores)


def pilot_bindings(binding, protocol):
    pilot = read(OUT / 'pilot.json')
    checked = read(OUT / 'pilot_separate_checks.json')
    started = read(OUT / 'formal_started.json')
    assert pilot['protocol_sha256'] == checked['protocol_sha256'] == started['protocol_sha256'] == binding
    assert pilot['status'] == 'complete_M6_output_pilot_pending_independent_check'
    assert checked['status'] == 'passed_independent_M6_output_adaptation_pilot_checks'
    assert checked['checker_sha256'] == sha(Path(ref.__file__))
    assert checked['pilot_sha256'] == started['pilot_sha256'] == sha(OUT / 'pilot.json')
    assert started['pilot_separate_checks_sha256'] == sha(OUT / 'pilot_separate_checks.json')
    assert pilot['formal_training_started'] is False
    assert pilot['cal_quality_evaluations'] == pilot['test_quality_evaluations'] == pilot['new_api_calls'] == 0
    assert pilot['pilot_optimizer_steps'] == checked['replay_optimizer_steps'] == 16
    assert pilot['encoder_query_forwards'] == 12672 and checked['independent_encoder_query_forwards'] == 384
    assert checked['formal_training_authorized_by_implementation_gate'] is True
    assert checked['source_sha256'] == protocol['source_sha256'] and checked['input_sha256'] == protocol['input_sha256']
    check_files(pilot['artifact_sha256'], [OUT / f'pilot_{a}{suffix}' for a in ARMS for suffix in ('.pt', '.npz')])
    check_files(checked['artifact_sha256'])
    assert datetime.fromisoformat(protocol['created_at_utc']) <= datetime.fromisoformat(checked['completed_at_utc']) <= datetime.fromisoformat(started['started_at_utc'])
    return pilot, checked, started


def binding_artifacts(binding, protocol):
    pilot, checked, started = pilot_bindings(binding, protocol)
    selected = read(OUT / 'all_selected_frozen.json')
    assert selected['status'] == 'all_ten_selected_endpoints_before_any_formal_test'
    assert selected['protocol_sha256'] == binding
    for name, expected in dict(formal_training_trajectories=10, encoder_training_trajectories=5,
        epochs_completed=40, optimizer_steps=30720, candidate_checkpoints=50,
        test_encoder_query_forwards=0).items():
        assert selected[name] == expected, name
    assert selected['pilot_sha256'] == sha(OUT / 'pilot.json')
    assert selected['pilot_separate_checks_sha256'] == sha(OUT / 'pilot_separate_checks.json')
    expected = [OUT / 'formal_started.json']
    for fold in range(5):
        for arm in ARMS:
            expected += [OUT / f'fold{fold}_{arm}_epoch{e}.pt' for e in range(5)]
            expected += [OUT / f'fold{fold}_{arm}_epoch{e}_training.npz' for e in range(1, 5)]
            expected += [OUT / f'fold{fold}_{arm}{suffix}' for suffix in ('.json', '_fit_cal.npz')]
    assert len(expected) == 111
    check_files(selected['artifact_sha256'], expected)
    frozen = read(OUT / 'predictions_frozen.json')
    assert frozen['status'] == 'all_selected_predictions_before_effects'
    assert frozen['protocol_sha256'] == binding
    assert frozen['all_selected_frozen_sha256'] == sha(OUT / 'all_selected_frozen.json')
    assert frozen['predictions_sha256'] == sha(OUT / 'predictions.npz')
    assert frozen['selected_fit8_replays'] == 80 and frozen['selected_test_encoder_query_forwards'] == 19200
    check_files(frozen['test_artifact_sha256'], [OUT / f'fold{f}_{a}_test.npz' for f in range(5) for a in ARMS])
    result = read(OUT / 'results.json')
    assert result['status'] == 'complete_fixed_M6_output_adaptation_pending_independent_check'
    assert result['protocol_sha256'] == binding and result['scope'] == protocol['scope']
    for name in ('all_selected_frozen', 'predictions_frozen', 'bootstrap'):
        suffix = '.npz' if name == 'bootstrap' else '.json'
        assert result[name + '_sha256'] == sha(OUT / (name + suffix))
    snapshots = dict(selected['artifact_sha256']); snapshots.update(frozen['test_artifact_sha256'])
    snapshots.update(checked['artifact_sha256'])
    snapshots.update({str(OUT / name): sha(OUT / name) for name in ('protocol.json', 'pilot.json',
        'pilot_separate_checks.json', 'all_selected_frozen.json', 'predictions_frozen.json',
        'predictions.npz', 'bootstrap.npz', 'results.json')})
    return selected, frozen, result, snapshots


def checkpoint(path, binding, fold, arm, epoch, initial, torch):
    cp = torch.load(path, map_location='cpu', weights_only=True)
    assert set(cp) == {'trained_parameters', 'arm', 'fold', 'epoch', 'protocol_sha256', 'core_sha256'}
    assert (cp['fold'], cp['arm'], cp['epoch']) == (fold, arm, epoch)
    assert cp['protocol_sha256'] == binding and cp['core_sha256'] == sha(ROOT / 'scripts/run_m6_output_adaptation.py')
    wanted = [name for name in initial if arm == 'A' or name.startswith('head.')]
    params = cp['trained_parameters']
    assert list(params) == wanted
    assert len(params) == (103 if arm == 'A' else 2)
    assert sum(p.numel() for p in params.values()) == (22565761 if arm == 'A' else 385)
    for name, value in params.items():
        assert value.dtype == torch.float32 and value.shape == initial[name].shape and torch.isfinite(value).all(), name
        if epoch == 0:
            assert torch.equal(value, initial[name]), 'Epoch-zero mismatch: ' + name
    complete = {name: params.get(name, value) for name, value in initial.items()}
    return cp, complete


def trace_check(saved, summary, fold, arm, epoch, fit, gap, previous_head, counters, errors):
    assert set(saved) == {'step_trace', 'training_order'} | {'first_' + name for name in FIRST_NAMES}
    table = array(saved['step_trace'], (768, 7), np.float64, 'training trace')
    order = array(saved['training_order'], (6144,), np.int64, 'training order')
    rng = np.random.default_rng(2026091517 + fold)
    for _ in range(epoch):
        local = rng.permutation(6144)
    assert np.array_equal(order, fit[local])
    assert summary['batch_order_sha256'] == hashlib.sha256(order.tobytes()).hexdigest()
    assert (summary['fold'], summary['arm'], summary['epoch']) == (fold, arm, epoch)
    assert summary['optimizer_steps'] == 768 and summary['all_losses_and_gradients_finite'] is True
    assert np.all(table[:, :6] >= 0) and np.all((table[:, 6] == 0) | (table[:, 6] == 1))
    assert np.array_equal(table[:, 5], schedule(epoch))
    weights = np.where(abs(gap) > 1e-12, abs(gap), 0.)
    normalizer = float(weights.mean())
    assert summary['fit_mean_weight'] == normalizer
    assert np.array_equal(table[:, 6], np.asarray([float(weights[local[i:i + 8]].sum() == 0) for i in range(0, 6144, 8)]))
    assert summary['zero_weight_batches'] == int(table[:, 6].sum())
    if arm == 'C':
        assert np.count_nonzero(table[:, 3]) == 0
    if np.any(table[:, 6] > 0):
        assert np.count_nonzero(table[table[:, 6] > 0, 0]) == 0
    assert np.array_equal(table[:, :3], table[:, :3].astype(np.float32).astype(np.float64))
    assert np.array_equal(table[:, 2], (table[:, 0].astype(np.float32) + table[:, 1].astype(np.float32)).astype(np.float64))
    for name, col in (('online_weighted_BCE', 0), ('online_head_penalty', 1), ('online_loss', 2)):
        close(summary[name], math.fsum(map(float, table[:, col])) / 768, errors, f'{fold}/{arm}/{epoch}/{name}')
    close(summary['mean_encoder_head_gradnorm_before_clip'], table[:, 3:5].mean(0), errors, f'{fold}/{arm}/{epoch}/mean_gradnorm')
    assert summary['first_lr_multiplier'] == table[0, 5] and summary['last_lr_multiplier'] == table[-1, 5]
    assert math.isfinite(summary['seconds']) and summary['seconds'] >= 0
    first = {name: saved['first_' + name] for name in FIRST_NAMES}
    # first.indices is copied from the source fold, while training_order is
    # explicitly exported as int64. Preserve and check that distinct contract.
    array(first['indices'], (8,), fit.dtype, 'first indices')
    assert np.array_equal(first['indices'], order[:8])
    assert np.array_equal(first['targets'], (gap[local[:8]] > 0).astype(np.float32))
    assert np.array_equal(first['weights'], weights[local[:8]].astype(np.float32))
    array(first['normalizer'], (), np.float64, 'first normalizer')
    assert float(first['normalizer']) == normalizer
    assert np.array_equal(first['head_weight'], previous_head['weight']) and np.array_equal(first['head_bias'], previous_head['bias'])
    assert np.max(abs(np.linalg.norm(first['features'].astype(float), axis=1) - 1)) < 2e-6
    head_math = ref.first_head_math(first, table[0])
    numeric.native_score_bound(first['features'], first['head_weight'].reshape(384), float(first['head_bias'][0]),
        first['logits'], counters, f'{fold}/{arm}/{epoch}/first')
    gradients = summary['first_parameter_gradient_norm']
    assert isinstance(gradients, dict) and all(math.isfinite(v) and v >= 0 for v in gradients.values())
    return head_math


def initial_parameters(torch):
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    assert len(encoder.encoder.layer) == 12 and encoder.config.hidden_size == 384
    encoder.pooler = None
    head = torch.nn.Linear(384, 1)
    params = {name: value.detach().cpu().clone() for name, value in ref.parameters(encoder, head)}
    assert len(params) == 103 and sum(p.numel() for p in params.values()) == 22565761
    del encoder, head
    return params


def audit_records(binding, d, selected, result, torch, counters, errors):
    base = initial_parameters(torch)
    records, heads, reports, traces, caches, tests, choices = {}, {}, [], {}, {}, {}, []
    assert len(result['folds']) == 5
    for fold, (fit, cal, test) in enumerate(d['folds']):
        initial = dict(base)
        initial.update({'head.' + name: p for name, p in d['heads'][fold].items()})
        assert result['folds'][fold]['fold'] == fold
        assert set(result['folds'][fold]['arms']) == set(ARMS)
        for arm in ARMS:
            key = (fold, arm)
            path = OUT / f'fold{fold}_{arm}.json'
            row = read(path); records[key] = row
            assert (row['fold'], row['arm']) == key
            assert row['epochs_completed'] == 4 and row['optimizer_steps'] == 3072
            assert row['training_encoder_query_forwards'] == 24576 and row['candidate_encoder_query_forwards'] == 38400
            assert row['outer_test_quality_evaluations'] == 0
            assert row['trace_columns'] == list(TRACE_COLUMNS)
            assert row['BCE_denominator'] == 'sum of weights within the scored partition'
            assert math.isfinite(row['training_seconds']) and row['training_seconds'] >= 0
            train_names = [name for name in initial if arm == 'A' or name.startswith('head.')]
            frozen_names = [name for name in initial if name not in train_names]
            assert row['trained_names'] == train_names and row['frozen_names'] == frozen_names
            assert row['trained_parameter_count'] == (22565761 if arm == 'A' else 385)
            assert row['initial_parameters_sha256'] == ref.tensor_digest(initial.items())
            frozen_digest = ref.tensor_digest([(name, initial[name]) for name in frozen_names])
            assert row['frozen_before_sha256'] == row['frozen_after_sha256'] == frozen_digest
            assert Path(row['initial_head_path']).resolve() == (BASE / f'layer_pooling_v1/fold{fold}_M6_head.pt').resolve()
            assert row['initial_head_sha256'] == sha(row['initial_head_path'])
            cache_path = OUT / f'fold{fold}_{arm}_fit_cal.npz'
            assert Path(row['fit_cal_path']).resolve() == cache_path.resolve() and row['fit_cal_sha256'] == sha(cache_path)
            cache = archive(cache_path); caches[key] = cache
            assert set(cache) == {'fit_indices', 'cal_indices', 'epochs', 'fit_logits', 'cal_logits'}
            for name, expected in (('fit_indices', fit), ('cal_indices', cal), ('epochs', np.arange(5))):
                array(cache[name], expected.shape, np.int64, name)
                assert np.array_equal(cache[name], expected)
            array(cache['fit_logits'], (5, 6144), np.float32, 'fit logits')
            array(cache['cal_logits'], (5, 1536), np.float32, 'cal logits')
            assert np.array_equal(cache['fit_logits'], ref.bf16(cache['fit_logits']))
            assert np.array_equal(cache['cal_logits'], ref.bf16(cache['cal_logits']))
            assert np.array_equal(cache['fit_logits'][0], d['fit_scores'][fold])
            assert np.array_equal(cache['cal_logits'][0], d['cal_scores'][fold])
            assert len(row['candidates']) == 5 and len(row['epoch_training']) == 4
            gains, math_reports = [], []
            for epoch in range(5):
                cp_path = OUT / f'fold{fold}_{arm}_epoch{epoch}.pt'
                candidate = row['candidates'][epoch]
                assert candidate['epoch'] == epoch and candidate['initial_M6_feature_fit_cal_score_exact'] is (epoch == 0)
                assert Path(candidate['checkpoint']).resolve() == cp_path.resolve() and candidate['checkpoint_sha256'] == sha(cp_path)
                cp, params = checkpoint(cp_path, binding, fold, arm, epoch, initial, torch)
                assert candidate['parameters_sha256'] == ref.tensor_digest(params.items())
                heads[(fold, arm, epoch)] = {name: params['head.' + name].numpy().copy() for name in ('weight', 'bias')}
                for part, ids in (('fit', fit), ('cal', cal)):
                    logits = cache[part + '_logits'][epoch]
                    close(candidate[part + '_BCE'], partition_bce(logits, d['gap'][ids]), errors, f'{fold}/{arm}/{epoch}/{part}_BCE')
                    close(candidate[part + '_gain'], gain(logits, d['gap'][ids]), errors, f'{fold}/{arm}/{epoch}/{part}_gain')
                    assert candidate[part + '_bm25_count'] == int(np.count_nonzero(logits > 0))
                    if arm == 'C' or epoch == 0:
                        h = heads[(fold, arm, epoch)]
                        numeric.native_score_bound(d['features'][ids], h['weight'].reshape(384), float(h['bias'][0]),
                            logits, counters, f'{fold}/{arm}/{epoch}/{part}')
                gains.append(gain(cache['cal_logits'][epoch], d['gap'][cal]))
                if epoch:
                    summary = row['epoch_training'][epoch - 1]
                    trace_path = OUT / f'fold{fold}_{arm}_epoch{epoch}_training.npz'
                    assert Path(summary['trace_path']).resolve() == trace_path.resolve() and summary['trace_sha256'] == sha(trace_path)
                    assert list(summary['first_parameter_gradient_norm']) == train_names
                    saved = archive(trace_path); traces[(fold, arm, epoch)] = saved
                    math_reports.append(trace_check(saved, summary, fold, arm, epoch, fit, d['gap'][fit],
                        heads[(fold, arm, epoch - 1)], counters, errors))
                    if arm == 'C' or epoch == 1:
                        assert np.array_equal(saved['first_features'], d['features'][saved['first_indices']])
                if epoch == 4:
                    assert row['final_parameters_sha256'] == ref.tensor_digest(params.items())
                    changes = {name: float((p - initial[name]).abs().max()) for name, p in params.items()}
                    assert changes == row['final_parameter_max_changes']
                    assert all(changes[name] == 0 for name in frozen_names)
                    assert any(changes[name] > 0 for name in train_names if name.startswith('head.'))
                    if arm == 'A':
                        assert any(changes[name] > 0 for name in train_names if name.startswith('encoder.'))
                del cp, params
            expected_choice = candidate_choice(gains)
            # Use archived global gains for exact 1e-12 eligibility; recomputed
            # scalar means above are independently verified within 2e-12.
            archived_choice = candidate_choice([v['cal_gain'] for v in row['candidates']])
            assert expected_choice['selected_epoch'] == archived_choice['selected_epoch']
            assert expected_choice['eligible_epochs'] == archived_choice['eligible_epochs']
            assert row['selection'] == archived_choice
            chosen = row['selection']['selected_epoch']
            cp_path = OUT / f'fold{fold}_{arm}_epoch{chosen}.pt'
            assert Path(row['selected_checkpoint']).resolve() == cp_path.resolve() and row['selected_checkpoint_sha256'] == sha(cp_path)
            choices.append(dict(fold=fold, arm=arm, epoch=chosen, checkpoint=str(cp_path),
                checkpoint_sha256=sha(cp_path), record_path=str(path), record_sha256=sha(path)))
            reported = result['folds'][fold]['arms'][arm]
            assert {name: reported[name] for name in row} == row
            assert set(reported) - set(row) == {'selected_fit8_restore_exact', 'test_archive', 'selected_test_BCE'}
            assert reported['selected_fit8_restore_exact'] is True
            test_path = OUT / f'fold{fold}_{arm}_test.npz'
            assert Path(reported['test_archive']).resolve() == test_path.resolve()
            saved_test = archive(test_path); tests[key] = saved_test
            assert set(saved_test) == {'test_indices', 'features', 'scores'}
            array(saved_test['test_indices'], (1920,), np.int64, 'test indices')
            assert np.array_equal(saved_test['test_indices'], test)
            array(saved_test['features'], (1920, 384), np.float32, 'test features')
            array(saved_test['scores'], (1920,), np.float32, 'test scores')
            assert np.max(abs(np.linalg.norm(saved_test['features'].astype(float), axis=1) - 1)) < 2e-6
            if arm == 'C' or chosen == 0:
                assert np.array_equal(saved_test['features'], d['features'][test])
            if chosen == 0:
                assert np.array_equal(saved_test['scores'], d['B_scores'][test])
            h = heads[(fold, arm, chosen)]
            numeric.native_score_bound(saved_test['features'], h['weight'].reshape(384), float(h['bias'][0]),
                saved_test['scores'], counters, f'{fold}/{arm}/selected_test')
            close(reported['selected_test_BCE'], partition_bce(saved_test['scores'], d['gap'][test]), errors, f'{fold}/{arm}/selected_test_BCE')
            reports.append(dict(fold=fold, arm=arm, endpoints=5, epoch_traces=4,
                selected_epoch=chosen, trained_tensors=len(train_names),
                trained_parameters=row['trained_parameter_count'], first_head_math=math_reports,
                original_initialization_exact=True, checkpoint_scope_and_finiteness=True,
                every_trace_order_schedule_and_partition_metrics_checked=True,
                global_max_cal_selection_checked=True, final_changes_and_frozen_parameters_checked=True))
            print(json.dumps(dict(status='independent_formal_CPU_arm_checked', fold=fold, arm=arm)), flush=True)
        assert records[(fold, 'A')]['initial_parameters_sha256'] == records[(fold, 'C')]['initial_parameters_sha256']
        assert [v['batch_order_sha256'] for v in records[(fold, 'A')]['epoch_training']] == [v['batch_order_sha256'] for v in records[(fold, 'C')]['epoch_training']]
    assert selected['selected'] == choices
    return dict(records=records, heads=heads, reports=reports, traces=traces, caches=caches, tests=tests, base=base)


def policy_checks(d, state, result, counters, errors):
    predictions = archive(OUT / 'predictions.npz')
    assert set(predictions) == {'query_ids', 'group_ids', 'utility', 'fold_id', 'A_scores', 'C_scores', 'B_scores'} | set(POLICIES)
    for name in ('query_ids', 'group_ids', 'utility', 'B_scores'):
        assert np.array_equal(predictions[name], d[name]), name
        assert predictions[name].dtype == d[name].dtype
    array(predictions['fold_id'], (9600,), np.int64, 'OOF fold ID')
    for arm in ARMS:
        array(predictions[arm + '_scores'], (9600,), np.float32, arm + ' OOF scores')
    actions = {}
    for name in POLICIES:
        actions[name] = array(predictions[name], (9600,), bool, name + ' action')
        expected = np.zeros(9600, bool) if name == 'Dense' else np.ones(9600, bool) if name == 'BM25' else predictions[name + '_scores'] > 0
        assert np.array_equal(actions[name], expected)
    values = np.column_stack([(actions['A'].astype(np.int64) - actions[name].astype(np.int64)) * d['gap'] for name in ('C', 'B', 'Dense', 'BM25')])
    assert set(result['policy']) == set(POLICIES) and set(result['primary']) == set(PRIMARY)
    for name in POLICIES:
        expected = numeric.policy_summary(d['utility'], actions[name])
        assert set(result['policy'][name]) == set(expected)
        for field, value in expected.items():
            close(result['policy'][name][field], value, errors, name + '/' + field)
    for fold, (_, _, test) in enumerate(d['folds']):
        assert np.all(predictions['fold_id'][test] == fold)
        row = result['folds'][fold]
        for arm in ARMS:
            assert np.array_equal(predictions[arm + '_scores'][test], state['tests'][(fold, arm)]['scores'])
        assert set(row['test_BCE']) == {'A', 'C', 'B'}
        for name in ('A', 'C', 'B'):
            close(row['test_BCE'][name], partition_bce(predictions[name + '_scores'][test], d['gap'][test]), errors, f'{fold}/{name}/test_BCE')
        assert set(row['policy']) == set(POLICIES) and set(row['primary_means']) == set(PRIMARY)
        for name in POLICIES:
            expected = numeric.policy_summary(d['utility'][test], actions[name][test])
            assert set(row['policy'][name]) == set(expected)
            for field, value in expected.items():
                close(row['policy'][name][field], value, errors, f'{fold}/{name}/{field}')
        for j, name in enumerate(PRIMARY):
            close(row['primary_means'][name], math.fsum(map(float, values[test, j])) / len(test), errors, f'{fold}/{name}')
    with threadpool_limits(limits=2):
        draws, _ = numeric.frequency_bootstrap(values, d['group_ids'], draws=20000, seed=2026091519)
    archived = archive(OUT / 'bootstrap.npz')
    assert set(archived) == {'draws'}
    array(archived['draws'], (20000, 4), np.float64, 'bootstrap draws')
    close(archived['draws'], draws, errors, 'all_bootstrap_draws')
    intervals = np.quantile(draws, [.00625, .99375], axis=0, method='linear').T
    primary = {}
    for j, name in enumerate(PRIMARY):
        point = math.fsum(map(float, values[:, j])) / 9600
        close(result['primary'][name]['mean'], point, errors, name + '/mean')
        close(result['primary'][name]['interval'], intervals[j], errors, name + '/interval')
        primary[name] = dict(mean=point, interval=intervals[j].tolist())
    recipe, candidate, decision = gates(primary)
    assert result['recipe_gate'] is recipe and result['candidate_preparation_gate'] is candidate and result['decision'] == decision
    assert result['selected_epochs'] == {a: [state['records'][(f, a)]['selection']['selected_epoch'] for f in range(5)] for a in ARMS}
    for name, expected in dict(formal_training_trajectories=10, new_encoder_fits=5,
        head_training_trajectories=10, epochs_completed=40, optimizer_steps=30720,
        candidate_checkpoints=50, pilot_optimizer_steps=16, pilot_encoder_query_forwards=12672,
        new_api_calls=0, runtime_deployed=False, core_goal_achieved=False).items():
        assert result[name] == expected, name
    assert result['encoder_query_forwards'] == dict(training=245760, candidate_fit_cal=384000,
        selected_fit_replay=80, selected_test=19200, total=649040)
    assert result['BCE_denominator'] == 'sum of weights within the scored partition'
    for name in ('elapsed_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes', 'process_rss_bytes'):
        assert math.isfinite(result[name]) and result[name] > 0
    return dict(primary=primary, recipe_gate=recipe, candidate_preparation_gate=candidate,
        decision=decision, bootstrap_draws_checked=20000,
        bootstrap_max_error=errors['all_bootstrap_draws'], policies_checked=5, outer_folds_checked=5)


def restore_reference(encoder, head, cp, base, initial_head, torch):
    reference = dict(ref.parameters(encoder, head))
    assert list(reference) == list(base)
    initial = dict(base); initial.update({'head.' + name: p for name, p in initial_head.items()})
    with torch.no_grad():
        for name, parameter in reference.items():
            parameter.copy_(cp['trained_parameters'].get(name, initial[name]).to(parameter.device))
            parameter.grad = None
    assert ref.tensor_digest(reference.items()) == ref.tensor_digest(
        [(name, cp['trained_parameters'].get(name, initial[name])) for name in reference])


def probe_reference(encoder, head, d, indices, expected_scores, expected_features, torch, counters, name):
    features, logits, pool_errors = [], [], []
    assert len(indices) > 0 and len(indices) % 8 == 0
    for start in range(0, len(indices), 8):
        batch, special = ref.make_batch(d['tokens'], indices[start:start + 8], torch)
        replay = ref.full_depth_hook_reference(encoder, head, batch, special, torch)
        features.append(replay['features']); logits.append(replay['logits']); pool_errors.append(replay['FP64_pool_error'])
    features, logits = np.concatenate(features), np.concatenate(logits)
    assert np.array_equal(logits, expected_scores), name + ' full12 native scores'
    if expected_features is not None:
        assert np.array_equal(features, expected_features), name + ' full12 features'
    assert max(pool_errors) <= 2e-6
    numeric.native_score_bound(features, head.weight.detach().cpu().numpy().reshape(384), float(head.bias.detach().cpu()[0]),
        expected_scores, counters, name)
    return max(pool_errors)


def replay_first_backward(encoder, head, arm, d, saved, record, torch):
    all_parameters = ref.parameters(encoder, head)
    trainable = [(n, p) for n, p in all_parameters if p.requires_grad]
    frozen = [(n, p) for n, p in all_parameters if not p.requires_grad]
    assert [n for n, _ in trainable] == list(record['first_parameter_gradient_norm'])
    for _, parameter in all_parameters:
        parameter.grad = None
    batch, special = ref.make_batch(d['tokens'], saved['first_indices'], torch)
    target = torch.from_numpy(saved['first_targets']).to('cuda')
    weight = torch.from_numpy(saved['first_weights']).to('cuda')
    norm = float(saved['first_normalizer'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = ref.differentiable_hook_feature(encoder, batch, special, torch)
        assert features.requires_grad is (arm == 'A')
        logits = head(features).flatten().float()
        supervised = (weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')).mean() / norm
    penalty = .0005 * head.weight.float().square().sum()
    loss = supervised + penalty
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
    assert all(p.grad is None for _, p in frozen)
    assert np.array_equal(features.detach().cpu().numpy(), saved['first_features'])
    assert np.array_equal(logits.detach().cpu().numpy(), saved['first_logits'])
    for name in ('weight', 'bias'):
        parameter = getattr(head, name)
        assert np.array_equal(parameter.detach().cpu().numpy(), saved['first_head_' + name])
        assert np.array_equal(parameter.grad.detach().cpu().numpy(), saved['first_head_' + name + '_grad'])
    gradients = {name: float(p.grad.detach().float().norm()) for name, p in trainable}
    assert gradients == record['first_parameter_gradient_norm']
    encps = [p for name, p in trainable if name.startswith('encoder.')]
    encnorm = float(torch.nn.utils.clip_grad_norm_(encps, 1.)) if encps else 0.
    headnorm = float(torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.))
    actual = np.array([float(supervised.detach()), float(penalty.detach()), float(loss.detach()), encnorm, headnorm])
    assert np.array_equal(actual, saved['step_trace'][0, :5])
    for _, p in all_parameters:
        p.grad = None
    return dict(head_gradients_exact=True, every_trainable_parameter_gradient_norm_exact=True,
        loss_and_combined_gradient_norm_exact=True, trainable_gradient_tensors=len(trainable), optimizer_steps=0)


def GPU_checks(binding, d, state, torch, counters):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    reports, count, backward_count = [], 0, 0
    for fold, (fit, cal, test) in enumerate(d['folds']):
        cal_positions = sample_positions(d['query_ids'], cal, fold, 'cal')
        test_positions = sample_positions(d['query_ids'], test, fold, 'test')
        for arm in ARMS:
            encoder, head = ref.reference_model(arm, torch)
            row = state['records'][(fold, arm)]
            cache = state['caches'][(fold, arm)]
            chosen = row['selection']['selected_epoch']
            checks, pool_errors = [], []
            for epoch in range(5):
                cp = torch.load(OUT / f'fold{fold}_{arm}_epoch{epoch}.pt', map_location='cpu', weights_only=True)
                assert cp['protocol_sha256'] == binding and (cp['fold'], cp['arm'], cp['epoch']) == (fold, arm, epoch)
                restore_reference(encoder, head, cp, state['base'], d['heads'][fold], torch)
                expected_x = d['features'][cal[cal_positions]] if arm == 'C' or epoch == 0 else None
                pool_errors.append(probe_reference(encoder, head, d, cal[cal_positions], cache['cal_logits'][epoch, cal_positions],
                    expected_x, torch, counters, f'{fold}/{arm}/{epoch}/cal32'))
                count += 32
                if epoch < 4:
                    saved = state['traces'][(fold, arm, epoch + 1)]
                    checked = replay_first_backward(encoder, head, arm, d, saved, row['epoch_training'][epoch], torch)
                    checks.append(dict(epoch=epoch + 1, **checked)); count += 8; backward_count += 1
                if epoch == chosen:
                    saved = state['tests'][(fold, arm)]
                    pool_errors.append(probe_reference(encoder, head, d, test[test_positions], saved['scores'][test_positions],
                        saved['features'][test_positions], torch, counters, f'{fold}/{arm}/test32'))
                    expected_x = d['features'][fit[:8]] if arm == 'C' or epoch == 0 else None
                    pool_errors.append(probe_reference(encoder, head, d, fit[:8], cache['fit_logits'][epoch, :8],
                        expected_x, torch, counters, f'{fold}/{arm}/selected_fit8'))
                    count += 40
                del cp
            reports.append(dict(fold=fold, arm=arm, candidate_cal_sample_count=160,
                first_batch_gradient_queries=32, selected_test_queries=32, selected_fit_queries=8,
                encoder_query_forwards=232, cal_positions=cal_positions.tolist(), test_positions=test_positions.tolist(),
                selected_checkpoint=chosen, native_scores_exact=True,
                saved_first_batch_and_selected_test_features_exact=True,
                FP64_pool_max_error=max(pool_errors), epoch_first_backward=checks))
            del encoder, head
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(dict(status='independent_formal_GPU_arm_checked', fold=fold, arm=arm,
                query_forwards=232, backward_passes=4, optimizer_steps=0)), flush=True)
    assert count == 2320 and backward_count == 40
    return dict(arms=reports, encoder_query_forwards=count, first_batch_backward_passes=backward_count,
        optimizer_steps=0, full12_default_forward_layer6_hook=True,
        FP64_pool_max_error=max(v['FP64_pool_max_error'] for v in reports), sample_rule=SAMPLE_RULE)


def check(binding):
    started = time.perf_counter()
    destination = OUT / 'separate_checks.json'
    assert not destination.exists(), 'Preserve an existing formal independent receipt'
    checker_sources = {str(Path(__file__).resolve()): sha(__file__),
        str(Path(ref.__file__).resolve()): sha(ref.__file__), str(Path(numeric.__file__).resolve()): sha(numeric.__file__)}
    protocol = protocol_inputs(binding)
    selected, frozen, result, snapshots = binding_artifacts(binding, protocol)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    data = load_data(torch)
    counters, errors = {}, {}
    with threadpool_limits(limits=2):
        state = audit_records(binding, data, selected, result, torch, counters, errors)
        effects = policy_checks(data, state, result, counters, errors)
    gpu = GPU_checks(binding, data, state, torch, counters)
    assert counters['score_vectors_checked'] == 180 and counters['score_rows_checked'] == 251920
    check_files(snapshots); check_files(protocol['source_sha256']); check_files(protocol['input_sha256']); check_files(checker_sources)
    receipt = dict(status='passed_independent_M6_output_adaptation_formal_checks',
        protocol_sha256=binding, results_sha256=sha(OUT / 'results.json'),
        predictions_sha256=sha(OUT / 'predictions.npz'), checker_sha256=sha(__file__),
        checker_source_sha256=checker_sources, source_sha256=protocol['source_sha256'],
        input_sha256=protocol['input_sha256'], artifact_sha256=snapshots,
        checkpoint_count=50, trace_archive_count=40, trace_step_count=30720,
        first_head_formula_checks=40, fit_cal_archive_count=10, selected_endpoints=10,
        selection_frozen_artifacts=111, CPU_arms=state['reports'], numerical_score_bounds=counters,
        maximum_scalar_check_error=max(errors.values()), scalar_errors=errors,
        effects=effects, independent_GPU=gpu, independent_encoder_query_forwards=2320,
        independent_backward_passes=40, independent_optimizer_steps=0,
        pilot_replayed_optimizer_steps_in_bound_receipt=16,
        full_formal_training_replayed=False, all_formal_encoder_forwards_replayed=False,
        coverage_boundary='All saved artifacts, orders, schedules, first-step formulas, partition metrics, choices and effects; full12 GPU checks on fixed cal/test samples and every epoch first backward only. No full-trajectory retraining or proof from logs alone.',
        new_scientific_fits=0, new_api_calls=0, runtime_deployed=False, core_goal_achieved=False,
        storage_dtype_changes=False, scientific_rule_changes=False, numerical_tolerance_changes=False,
        elapsed_seconds=time.perf_counter() - started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Consumed old9600 fixed-recipe development; this numerical audit is not independent-source policy confirmation')
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return receipt


def self_test():
    """CPU-only synthetic checks; no real result, model or token reads."""
    assert candidate_choice([0., .75e-12, 1.5e-12, -1., -2.])['selected_epoch'] == 1
    assert candidate_choice([.3, .1, .2, .3, 0.])['selected_epoch'] == 0
    assert candidate_choice([0., -1., -2., -3., -4.])['selected_epoch'] == 0
    learning = np.concatenate([schedule(e) for e in range(1, 5)])
    assert len(learning) == 3072 and learning[0] == 1 / 307 and learning[306] == 1. and learning[-1] == 0.
    assert np.all(np.diff(learning[:307]) > 0) and np.all(np.diff(learning[306:]) < 0)
    x = np.array([-10., .3, 20., -2., 0.], dtype=np.float32)
    gap = np.array([.2, -.1, 0., .4, 1e-13])
    weights = np.where(abs(gap) > 1e-12, abs(gap), 0.)
    expected = float(weights @ np.logaddexp(0., (1 - 2 * (gap > 0)) * x.astype(float)) / weights.sum())
    assert abs(partition_bce(x, gap) - expected) < 1e-14
    assert gain(x, gap) == -.02
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    values = np.arange(24, dtype=np.float64).reshape(6, 4) / 100.
    samples, _ = numeric.frequency_bootstrap(values, groups, draws=107, seed=9281)
    rng, explicit = np.random.default_rng(9281), []
    unique = np.unique(groups)
    for _ in range(107):
        chosen = rng.integers(len(unique), size=len(unique))
        rows = np.concatenate([np.flatnonzero(groups == unique[k]) for k in chosen])
        explicit.append(values[rows].mean(0))
    error = float(np.max(abs(samples - np.asarray(explicit))))
    assert error < 1e-15
    zero = {name: dict(mean=0., interval=[-1., 1.]) for name in PRIMARY}
    assert gates(zero) == (False, False, 'END_FIXED_M6_OUTPUT_ADAPTATION_NO_CONFIRMED_INCREMENT')
    successful = {name: dict(mean=.02, interval=[.001, .03]) for name in PRIMARY}
    assert gates(successful)[:2] == (True, True)
    successful['A_minus_Dense']['mean'] = .005
    assert gates(successful)[:2] == (True, False)
    ids, indices = np.array([f'q{i}' for i in range(100)]), np.arange(100)
    positions = sample_positions(ids, indices, 2, 'cal')
    assert len(set(positions)) == 32 and np.array_equal(positions, sample_positions(ids, indices, 2, 'cal'))
    rejects = 0
    for bad in (np.zeros(3, np.float64), np.array([0., np.nan, 1.], np.float32)):
        try:
            array(bad, (3,), np.float32, 'synthetic bad score')
        except AssertionError:
            rejects += 1
    assert rejects == 2
    return dict(status='passed_formal_checker_synthetic_CPU_checks',
        global_max_earliest_selection=True, full_schedule_checked=True,
        partition_weight_denominator_checked=True, frequency_bootstrap_max_error=error,
        dtype_and_nonfinite_rejections=rejects, outcome_independent_sample_rule=SAMPLE_RULE,
        real_data_reads=0, model_fits=0, GPU_forwards=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        receipt = check(args.protocol_sha256)
        print(json.dumps({key: receipt[key] for key in ('status', 'independent_encoder_query_forwards',
            'independent_backward_passes', 'independent_optimizer_steps', 'elapsed_seconds')}, allow_nan=False))
