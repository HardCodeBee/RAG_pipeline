"""One matched term-aggregation screen, with a fixed training-only budget rule."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0')
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from run_m6_capability_screen import require_execution_allowed, save_arrays, write_json
from router_term_aggregation import TermRouter, make_batch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_term_aggregation_v1'
PREVIOUS = OUT.parent / 'm6_complete_cohort_extension_v1'
sys.path.insert(0, str(BASE))
import weighted_linear_probe_refined as probe

SPEC = {
    'scope': 'two_consumed_development_cohorts_no_independent_confirmation',
    'training_roles': str(PREVIOUS / 'source_labels_and_roles.npz'),
    'fit_queries': 10309, 'development_queries': {'old': 1536, 'beir': 1044},
    'arms': ['U', 'F', 'N', 'S', 'T'], 'learned_arms': ['N', 'S', 'T'],
    'features': 'concat_existing_M6_and_L2_normalized_raw_L6_distinct_term_pool',
    'attention': 't=tanh(W*L2norm(h)+b); e=a*t+z*(v*t); softmax_over_valid_terms',
    'rank': 8, 'idf': 'current_SQLite_BM25_stored_IDF_unknown_terms_zero',
    'idf_transform': 'log1p_then_center_scale_by_all_fit_term_occurrences_only',
    'permutation': 'within_query_PCG64_SHA256(seed|query_id)_one_fixed_draw_no_redraw',
    'permutation_seed': 2026091831, 'model_seed': 2026091832,
    'shared_fallback': 'original_M6_pool_if_no_terms_or_original_total_IDF_zero',
    'objective': 'sum_abs_mean3_gap_weighted_BCE_over_sum_weights_plus_L2',
    'tie_atol': 1e-12, 'head_l2': .001, 'attention_l2': .00001,
    'bias_l2': 0, 'threshold': 0, 'warm_start': 'common_exact_uniform_term_plus_M6_head',
    'optimizer': 'Adam', 'betas': [.9, .999], 'adam_eps': 1e-8,
    'initial_steps': 2000, 'optional_extension_steps': 1000,
    'initial_lr': .003, 'middle_lr': .0003, 'final_lr': .00003,
    'schedule': 'cosine_in_each_stage', 'batch': 'all_positive_weight_fit_rows',
    'extension_rule': 'any_arm_mean_objective_steps1701_1800_minus1901_2000_gt_0.0001_then_all_three_extend',
    'unsettled_rule': 'at_cap_mean_steps2701_2800_minus2901_3000_gt_0.0001',
    'training_loss_drop_threshold': .0001, 'clip_norm': 1.0,
    'checkpoint_every': 100, 'score_batch': 256, 'dtype': 'float32',
    'final_head': 'one_exact_lambda0.001_convex_fit_after_each_learned_pool_is_frozen',
    'convex_fits_total': 5, 'nonconvex_fits_total': 3,
    'minimum_gain_over_best_fixed_each_cohort': .01,
    'minimum_gain_over_best_archived_M6_each_cohort': .002,
    'minimum_T_gain_over_each_U_F_N_S_each_cohort': .002,
    'simpler_candidate_order': ['U', 'F', 'N', 'T'],
    'new_retrieval_calls': 0, 'new_answer_calls': 0, 'encoder_training_steps': 0,
    'old_outer_test_evaluations': 0, 'external_evaluations': 0,
    'no_automatic_extension': 'no new seed rank lambda threshold split or alternate loss; only declared training-loss extension',
}


def describe():
    require_execution_allowed()
    OUT.mkdir(parents=True, exist_ok=True)
    binding = {'spec': SPEC,
               'base_config': 'outputs/router/hotpotqa_bd_router_v1/config.yaml',
               'execution_contract': '../work/router_research/m6_beir_validation_v1/execution_contract.json',
               'source_roles': str(PREVIOUS / 'source_labels_and_roles.npz')}
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['binding'] != binding:
            raise ValueError('Saved protocol differs from this fixed recipe')
    else:
        write_json(path, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'binding': binding})
    return binding


def load_cache():
    directory = OUT / 'cache'
    metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'complete'
    keys = ['query_ids', 'group_ids', 'source', 'role', 'M6', 'utility',
            'term_hidden', 'term_indptr', 'term_idf', 'fallback']
    cache = {key: np.load(directory / f'{key}.npy', mmap_mode='r', allow_pickle=False) for key in keys}
    n = len(cache['query_ids'])
    assert n == 12889 and len(set(cache['query_ids'])) == n
    fit, dev = (np.flatnonzero(cache['role'] == name) for name in ('fit', 'dev'))
    assert len(fit) == SPEC['fit_queries'] and len(dev) == 2580
    assert not set(cache['group_ids'][fit]) & set(cache['group_ids'][dev])
    assert cache['M6'].shape == (n, 384) and cache['utility'].shape == (n, 2)
    assert cache['term_hidden'].shape == (int(cache['term_indptr'][-1]), 384)
    assert len(cache['term_idf']) == len(cache['term_hidden'])
    assert np.all(np.diff(cache['term_indptr']) > 0)
    term_fit = np.repeat(cache['role'] == 'fit', np.diff(cache['term_indptr']))
    log_idf = np.log1p(np.asarray(cache['term_idf'], dtype=np.float64))
    center, scale = float(log_idf[term_fit].mean()), float(log_idf[term_fit].std())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('No usable training IDF variation')
    scaled = ((log_idf-center)/scale).astype(np.float32)
    permutation = np.arange(len(log_idf), dtype=np.int64)
    changed = 0
    for i, qid in enumerate(cache['query_ids']):
        start, stop = cache['term_indptr'][i:i+2]
        material = f'{SPEC["permutation_seed"]}|{qid}'.encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], 'big')
        order = np.random.default_rng(seed).permutation(stop-start) + start
        permutation[start:stop] = order
        changed += int(not np.array_equal(cache['term_idf'][start:stop], cache['term_idf'][order]))
    cache['permutation'] = permutation
    save_arrays(OUT / 'input_transform.npz', permutation=permutation, fit=fit, dev=dev,
                idf_log_mean=np.asarray(center), idf_log_std=np.asarray(scale))
    write_json(OUT / 'input_summary.json', {'queries': n, 'fit_queries': len(fit), 'dev_queries': len(dev),
               'terms': len(log_idf), 'fallback_queries': int(cache['fallback'].sum()),
               'permutation_changed_weight_queries': changed, 'idf_log_mean': center, 'idf_log_std': scale,
               'cache_completion': metadata})
    return cache, scaled, fit, dev


def new_model(arm, initial_head=None):
    torch.manual_seed(SPEC['model_seed'])
    model = TermRouter(arm, rank=SPEC['rank']).cuda()
    if initial_head is not None:
        set_head(model, initial_head)
    return model


def set_head(model, fitted):
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(fitted['coef'], dtype=torch.float32, device='cuda')[None, :])
        model.head.bias.fill_(fitted['intercept'])


def all_features(model, cache, scaled, indices):
    features = []
    with torch.inference_mode():
        for start in range(0, len(indices), SPEC['score_batch']):
            batch = make_batch(cache, indices[start:start+SPEC['score_batch']], scaled, shuffled=model.arm == 'S')
            features.append(model.features(batch).cpu().numpy())
    return np.concatenate(features)


def exact_head(arm, features, gap):
    path = OUT / f'head_{arm}.json'
    if path.exists():
        fitted = json.loads(path.read_text(encoding='utf-8'))
        if not fitted['accepted']:
            raise RuntimeError(f'Saved convex {arm} head was rejected; no automatic refit')
        return fitted
    weights = np.where(np.abs(gap) > SPEC['tie_atol'], np.abs(gap), 0.)
    fitted = probe.fit(features, (gap > 0).astype(float), weights, regularization=SPEC['head_l2'])
    fitted['coef'] = fitted['coef'].tolist()
    write_json(path, fitted)
    if not fitted['accepted']:
        raise RuntimeError(f'Convex {arm} head did not meet the existing numerical acceptance rule')
    return fitted


def loss_drop(trace):
    if len(trace) < 300:
        return None
    return float(np.mean([row['objective'] for row in trace[-300:-200]])
                 - np.mean([row['objective'] for row in trace[-100:]]))


def train_to(arm, cache, scaled, active, gap, initial, target_steps):
    model = new_model(arm, initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=SPEC['initial_lr'],
                                 betas=tuple(SPEC['betas']), eps=SPEC['adam_eps'], foreach=False)
    path = OUT / f'training_{arm}.pt'
    trace, next_step, elapsed = [], 0, 0.0
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        trace, next_step, elapsed = saved['trace'], saved['next_step'], saved['elapsed_seconds']
    batch = make_batch(cache, active, scaled, shuffled=arm == 'S')
    labels = torch.tensor((gap > 0).astype(np.float32), device='cuda')
    weights = torch.tensor(np.abs(gap), dtype=torch.float32, device='cuda')
    weights = weights / weights.sum()
    started = time.monotonic()
    for step in range(next_step, target_steps):
        if step < SPEC['initial_steps']:
            fraction = step / (SPEC['initial_steps']-1)
            high, low = SPEC['initial_lr'], SPEC['middle_lr']
        else:
            fraction = (step-SPEC['initial_steps']) / (SPEC['optional_extension_steps']-1)
            high, low = SPEC['middle_lr'], SPEC['final_lr']
        lr = low+(high-low)*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        data_loss = (weights*F.binary_cross_entropy_with_logits(logits, labels, reduction='none')).sum()
        loss = data_loss + model.penalty(SPEC['head_l2'], SPEC['attention_l2'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {arm} training objective')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), SPEC['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        trace.append({'step': step+1, 'objective': float(loss.detach()), 'data_loss': float(data_loss.detach()),
                      'grad_norm': float(grad), 'lr': lr})
        if (step+1) % SPEC['checkpoint_every'] == 0 or step+1 == target_steps:
            temporary = path.with_suffix('.tmp')
            torch.save({'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(), 'next_step': step+1, 'trace': trace,
                        'elapsed_seconds': elapsed+time.monotonic()-started}, temporary)
            temporary.replace(path)
            print(json.dumps({'arm': arm, **trace[-1]}), flush=True)
    elapsed += time.monotonic()-started
    summary = {'steps': target_steps, 'elapsed_seconds': elapsed, 'training_loss_drop_last_windows': loss_drop(trace),
               'initial_objective': trace[0]['objective'], 'final_preupdate_objective': trace[-1]['objective'],
               'final_preupdate_grad_norm': trace[-1]['grad_norm'], 'trainable_parameters': sum(p.numel() for p in model.parameters()),
               'inactive_interaction_parameters': SPEC['rank'] if arm == 'N' else 0}
    with torch.inference_mode():
        pooled, attention = model.pool(batch, return_attention=True)
        uniform = batch['mask'].float()/batch['mask'].sum(-1, keepdim=True)
        valid = ~batch['fallback']
        summary['attention_mean_L1_from_uniform_nonfallback'] = float((attention-uniform).abs().sum(-1)[valid].mean())
        summary['attention_entropy_nonfallback'] = float((-(attention*attention.clamp_min(1e-12).log()).sum(-1))[valid].mean())
    write_json(OUT / f'training_{arm}_summary.json', summary)
    return summary


def summarize(logits, utility):
    gap, action = utility[:, 0]-utility[:, 1], logits > 0
    return {'F1': float(np.where(action, utility[:, 0], utility[:, 1]).mean()),
            'bm25_count': int(action.sum()), 'beneficial_count': int((action & (gap > 1e-12)).sum()),
            'harmful_count': int((action & (gap < -1e-12)).sum())}


def execute():
    describe()
    if (OUT / 'results.json').exists():
        print((OUT / 'results.json').read_text(encoding='utf-8'))
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache, scaled, fit, dev = load_cache()
    fit_gap = cache['utility'][fit, 0]-cache['utility'][fit, 1]
    positive = np.abs(fit_gap) > SPEC['tie_atol']
    active, active_gap = fit[positive], fit_gap[positive]
    if len(active) != 2981:
        raise ValueError('Complete-cohort nonzero utility support changed')
    initial = None
    for arm in ('U', 'F'):
        model = new_model(arm)
        features = all_features(model, cache, scaled, fit)
        fitted = exact_head(arm, features, fit_gap)
        if arm == 'U':
            initial = fitted
        del features, model
    training = {}
    for arm in SPEC['learned_arms']:
        training[arm] = train_to(arm, cache, scaled, active, active_gap, initial, SPEC['initial_steps'])
    extension_path = OUT / 'training_budget_decision.json'
    if extension_path.exists():
        extension = json.loads(extension_path.read_text(encoding='utf-8'))['extend_all_arms']
    else:
        extension = any(s['training_loss_drop_last_windows'] > SPEC['training_loss_drop_threshold'] for s in training.values())
        write_json(extension_path, {'extend_all_arms': extension, 'development_read': False,
                                   'decision_only_uses_training_objective': training})
    if extension:
        for arm in SPEC['learned_arms']:
            training[arm] = train_to(arm, cache, scaled, active, active_gap, initial,
                                     SPEC['initial_steps']+SPEC['optional_extension_steps'])
    # Freeze the pooling networks, then solve each final linear head exactly.
    for arm in SPEC['learned_arms']:
        model = new_model(arm)
        model.load_state_dict(torch.load(OUT / f'training_{arm}.pt', map_location='cpu', weights_only=True)['model'])
        features = all_features(model, cache, scaled, fit)
        fitted = exact_head(arm, features, fit_gap)
        set_head(model, fitted)
        torch.save(model.state_dict(), OUT / f'model_{arm}.pt')
        del features, model
    # Development outcomes enter policy evaluation only after every arm is frozen.
    scores = {}
    for arm in SPEC['arms']:
        model = new_model(arm)
        if arm in SPEC['learned_arms']:
            model.load_state_dict(torch.load(OUT / f'model_{arm}.pt', map_location='cpu', weights_only=True))
        else:
            set_head(model, json.loads((OUT / f'head_{arm}.json').read_text(encoding='utf-8')))
            torch.save(model.state_dict(), OUT / f'model_{arm}.pt')
        with torch.inference_mode():
            features = all_features(model, cache, scaled, dev)
            scores[arm] = model.head(torch.tensor(features, device='cuda')).squeeze(-1).cpu().numpy()
        del features, model
    with np.load(PREVIOUS / 'development_scores.npz', allow_pickle=False) as saved:
        historical = {key: saved[key].copy() for key in saved.files}
    report, numerical_unsettled = {}, []
    for arm, record in training.items():
        if extension and record['training_loss_drop_last_windows'] > SPEC['training_loss_drop_threshold']:
            numerical_unsettled.append(arm)
    for source in ('old', 'beir'):
        selected = cache['source'][dev] == source
        qids = cache['query_ids'][dev[selected]]
        positions = {str(q): i for i, q in enumerate(historical[f'{source}_query_ids'])}
        order = np.array([positions[str(q)] for q in qids])
        utility = np.asarray(cache['utility'][dev[selected]])
        assert len(qids) == SPEC['development_queries'][source]
        assert np.array_equal(utility, historical[f'{source}_utility'][order])
        policies = {arm: summarize(values[selected], utility) for arm, values in scores.items()}
        policies.update({arm: summarize(historical[f'{source}_{arm}'][order], utility) for arm in ('O', 'P')})
        fixed = {'BM25': float(utility[:, 0].mean()), 'Dense': float(utility[:, 1].mean())}
        quality_gate = {arm: policies[arm]['F1']-max(fixed.values()) >= SPEC['minimum_gain_over_best_fixed_each_cohort']
                        and policies[arm]['F1']-max(policies[r]['F1'] for r in ('O', 'P')) >= SPEC['minimum_gain_over_best_archived_M6_each_cohort']
                        for arm in ('U', 'F', 'N', 'T')}
        increment = {arm: policies['T']['F1']-policies[arm]['F1'] for arm in ('U', 'F', 'N', 'S')}
        report[source] = {'queries': len(qids), 'policies': policies, 'fixed_F1': fixed,
                          'quality_gates': quality_gate, 'T_minus_controls': increment,
                          'IDF_fusion_gate': quality_gate['T'] and min(increment.values()) >= SPEC['minimum_T_gain_over_each_U_F_N_S_each_cohort']}
    score_candidates = [arm for arm in SPEC['simpler_candidate_order'] if all(r['quality_gates'][arm] for r in report.values())]
    candidates = score_candidates
    mechanism_score_gate = all(r['IDF_fusion_gate'] for r in report.values())
    mechanism = mechanism_score_gate
    decision = ('promising_consumed_development_candidate_independent_protocol_needed' if candidates else
                'close_fixed_recipe_no_seed_rank_threshold_or_loss_extension')
    if numerical_unsettled and not candidates:
        decision = 'fixed_budget_ended_training_objective_unsettled_no_global_mechanism_rejection'
    save_arrays(OUT / 'development_scores.npz', query_ids=cache['query_ids'][dev], group_ids=cache['group_ids'][dev],
                source=cache['source'][dev], utility=cache['utility'][dev], **scores)
    results = {'status': 'completed_consumed_development_screen', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'training': training, 'training_extension': extension, 'training_unsettled_arms': numerical_unsettled,
               'convex_fits': 5, 'nonconvex_fits': 3, 'fit_queries': len(fit), 'positive_weight_fit_queries': len(active),
               'cohorts': report, 'quality_score_candidates': score_candidates, 'quality_candidates': candidates,
               'IDF_fusion_score_gate': mechanism_score_gate, 'IDF_fusion_development_gate': mechanism,
               'optimization_settled_within_budget': {arm: arm not in numerical_unsettled for arm in SPEC['arms']},
               'IDF_fusion_mechanism_established': False,
               'independent_gain_established': False, 'decision': decision}
    write_json(OUT / 'results.json', results)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    elif args.prepare:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(SPEC, ensure_ascii=False, indent=2))
