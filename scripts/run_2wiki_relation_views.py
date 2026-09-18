"""Fixed entity-proxy masking views for one consumed 2Wiki development screen.

Only the Router input views change. All retrieval/generation artifacts and the
895 reserved questions remain untouched. No paid calls or parameter search.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
import numpy as np
import yaml

from router_relation_view import relation_view
from run_2wiki_local_residual import read, write_new

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
SOURCE = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1'
ROLES = SOURCE.parent / '2wiki_local_residual_v1/roles.npz'
OUT = SOURCE.parent / '2wiki_relation_views_v1'
MODEL = Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
ARMS = ('R_original', 'E_masked', 'D_dual', 'N_control', 'H_duplicate')
POLICIES = ('BM25', 'Dense', 'M6', 'F_IDF') + ARMS
REFERENCES = ('Dense', 'BM25', 'R_original', 'E_masked', 'N_control', 'H_duplicate')
SPEC = {
    'scope': 'consumed_2Wiki304_group_OOF_development_not_independent_confirmation',
    'queries': 304, 'groups': 256, 'folds': 4, 'roles': str(ROLES),
    'mask_source': 'router_relation_view_unicode_capitalized_and_quoted_proxy_spans_no_gold_NER_or_docs',
    'encoding': 'same_local_BGE_snapshot_tokenizer_max128_padding_max_length_truncation_before_masking',
    'entity_mask': 'replace_original_ordinary_WordPieces_overlapping_proxy_character_spans_by_native_MASK_id',
    'first_word': 'first_Unicode_re_word_span_in_original_question_protected_in_both_views',
    'position_eligibility': 'original_attention_and_not_original_special_mask_and_not_first_word_overlap',
    'control': 'nonzero_cyclic_shift_of_E_mask_on_eligible_positions; accept_different_masks_with_same_linear_run_length_multiset_in_full_token_coordinates',
    'control_choice': 'candidate_shifts_ascending; PCG64_first8_big_endian_SHA256(seed|qid)_choose_once',
    'control_seed': 2026091861, 'control_fallback': 'N_equals_E_if_no_valid_shift_including_empty_mask',
    'encoder': 'frozen_BGE_first6_layers_max128_batch8_CUDA_BF16_no_TF32',
    'pool': 'FP32_mean_and_L2_on_original_ordinary_positions_including_replaced_MASK_slots',
    'raw_view': 'reuse_saved_M6_FP32_features_no_original_encoder_replay',
    'features': {'R_original': 'R', 'E_masked': 'E', 'D_dual': 'concat_R_E_unscaled',
                 'N_control': 'concat_R_N_unscaled', 'H_duplicate': 'concat_R_R_unscaled'},
    'duplicate_control': 'same_768_input_dimension_nominal_parameters; original_function_space; equivalent_raw_lambda_half_in_real_arithmetic',
    'target': 'abs(mean3_F1_B_minus_D)_weighted_BCE; label_gap_positive; zero_gap_zero_weight',
    'objective': 'sum_w_BCE_div_sum_w_plus_lambda_times_coef_norm_squared_over2; unpenalized_intercept',
    'lambda': 0.001, 'solver': 'existing_weighted_linear_probe_refined_fixed_numerical_budget',
    'head_inference_dtype': 'float64', 'threshold': 0, 'tie_action': 'Dense',
    'fit_count': 20, 'degenerate_fold': 'stop_if_no_positive_weight_or_only_one_weighted_class_no_repair',
    'primary_candidate': 'D_dual', 'references': list(REFERENCES),
    'minimum_fixed_gain': 0.01, 'minimum_control_gain': 0.002,
    'bootstrap_only_after_all_point_gates': True, 'bootstrap_draws': 10000, 'bootstrap_seed': 2026091862,
    'CI_each': 1-0.05/6, 'conditional_family_nominal_coverage': 0.95,
    'inference_role': 'conditional_on_fitted_OOF_predictions_consumed_development_not_refit_or_selection_uncertainty',
    'stop': 'no_mask_rule_layer_lambda_seed_threshold_split_or_control_promotion_after_failure',
    'retained_access': False, 'new_retrieval_calls': 0, 'new_answer_calls': 0,
}


def execution_allowed():
    state = yaml.safe_load((ROOT/'analysis/hotpotqa_router/registry.yaml').read_text(encoding='utf-8'))['research_continuation']
    if state['experiment_execution'] == 'paused_by_user':
        raise RuntimeError('Research paused by user')


def checked_protocol():
    execution_allowed()
    protocol = read(OUT/'protocol.json')
    assert protocol['spec'] == SPEC
    assert protocol['span_module_sha256'] == hashlib.sha256((ROOT/'scripts/router_relation_view.py').read_bytes()).hexdigest()
    return protocol


def prepare():
    execution_allowed()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROLES, allow_pickle=False) as a:
        assert len(a['query_ids']) == 304 and len(set(a['group_ids'])) == 256
        role_counts = [int(np.sum(a['outer_fold'] == k)) for k in range(4)]
    write_new(OUT/'protocol.json', {'status': 'frozen_before_view_encoding_and_fits',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spec': SPEC,
              'outer_held_queries': role_counts,
              'span_module_sha256': hashlib.sha256((ROOT/'scripts/router_relation_view.py').read_bytes()).hexdigest(),
              'tokenization_provenance': 'same_snapshot_and_arguments_as_prepare_2wiki_measurement.py; historical_token_ID_arrays_not_saved; no_bitwise_replay_claim',
              'environment': str(SOURCE/'execution_contract.json'),
              'literature': ['https://aclanthology.org/2020.emnlp-main.298/', 'https://aclanthology.org/D19-1340/',
                             'https://aclanthology.org/2021.acl-long.345/']})
    print(json.dumps({'status': 'protocol_frozen', 'folds_reused': role_counts, 'fits': 0}))


def run_lengths(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return sorted((np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)).tolist())


def control_mask(entity, eligible, qid):
    positions = np.flatnonzero(eligible)
    pattern = entity[positions]
    lengths = run_lengths(entity)
    choices = []
    for shift in range(1, len(positions)):
        candidate = np.zeros_like(entity)
        candidate[positions] = np.roll(pattern, shift)
        if not np.array_equal(candidate, entity) and run_lengths(candidate) == lengths:
            choices.append(shift)
    if not choices:
        return entity.copy(), 0, 0
    seed = int.from_bytes(hashlib.sha256(f'{SPEC["control_seed"]}|{qid}'.encode()).digest()[:8], 'big')
    shift = choices[int(np.random.default_rng(seed).integers(len(choices)))]
    result = np.zeros_like(entity)
    result[positions] = np.roll(pattern, shift)
    return result, shift, len(choices)


def encode():
    checked_protocol()
    if (OUT/'features.npz').exists() or (OUT/'view_records.jsonl').exists():
        raise ValueError('View preparation already exists; inspect instead of re-encoding')
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = [json.loads(line) for line in (SOURCE/'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    qids = np.array([row['query_id'] for row in records])
    with np.load(SOURCE/'actions.npz', allow_pickle=False) as a:
        assert np.array_equal(qids, a['query_ids'])
        raw, gids = a['M6'], a['group_ids']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.mask_token_id is not None
    texts = [row['question'] for row in records]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=128,
                       return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors='np')
    ordinary = tokens['attention_mask'].astype(bool) & ~tokens['special_tokens_mask'].astype(bool)
    masks = np.zeros((len(qids), 2, 128), dtype=bool)
    views, jobs = [], []
    features = np.repeat(raw[:, None, :], 2, axis=1).copy()
    for i, (qid, question) in enumerate(zip(qids, texts)):
        preview = relation_view(question)
        offsets = tokens['offset_mapping'][i]
        first = re.search(r'\w+', question, flags=re.UNICODE)
        protected = ((offsets[:, 0] < first.end()) & (offsets[:, 1] > first.start())) if first else np.zeros(128, dtype=bool)
        eligible = ordinary[i] & ~protected
        entity = np.zeros(128, dtype=bool)
        for span in preview['proxy_spans']:
            entity |= (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start'])
        entity &= eligible
        control, shift, candidates = control_mask(entity, eligible, str(qid))
        assert entity.sum() == control.sum() and run_lengths(entity) == run_lengths(control)
        assert not np.any((entity | control) & ~eligible)
        masks[i] = np.stack((entity, control))
        changed = not np.array_equal(entity, control)
        if entity.any():
            jobs.append((i, 0))
            if changed:
                jobs.append((i, 1))
        view = {'query_id': str(qid), **preview, 'entity_wordpiece_count': int(entity.sum()),
                'ordinary_wordpiece_count': int(ordinary[i].sum()), 'entity_positions': np.flatnonzero(entity).tolist(),
                'control_positions': np.flatnonzero(control).tolist(), 'control_shift': shift,
                'valid_control_shifts': candidates, 'control_equals_entity': not changed,
                'overlap_positions': int(np.sum(entity & control)),
                'entity_model_preview': tokenizer.decode(np.where(entity, tokenizer.mask_token_id, tokens['input_ids'][i])[:int(tokens['attention_mask'][i].sum())]),
                'preview_note': 'entity_word_preview_is_not_model_input; encoder_uses_original_tokens_with_MASK_ID_replacements'}
        views.append(view)
    # The record contains only query-derived transforms, never reference answers.
    with (OUT/'view_records.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in views:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    model.encoder.layer = torch.nn.ModuleList(model.encoder.layer[:6])
    model.config.num_hidden_layers = 6
    model.pooler = None
    model.requires_grad_(False).to('cuda').eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, len(jobs), 8):
            actual = jobs[lo:lo+8]
            batch_jobs = actual + [actual[-1]]*(8-len(actual))
            rows = np.array([item[0] for item in batch_jobs])
            ids = tokens['input_ids'][rows].copy()
            for b, (row, kind) in enumerate(batch_jobs):
                ids[b, masks[row, kind]] = tokenizer.mask_token_id
            batch = {key: torch.tensor(tokens[key][rows], device='cuda') for key in ('attention_mask', 'token_type_ids')}
            batch['input_ids'] = torch.tensor(ids, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = model(**batch).last_hidden_state
                pooling = torch.tensor(ordinary[rows], device='cuda')
                count = pooling.sum(1)
                mean = (hidden.float()*pooling[:,:,None]).sum(1)/count.clamp_min(1).float()[:,None]
                mean = torch.where((count == 0)[:,None], hidden[:,0].float(), mean)
                encoded = torch.nn.functional.normalize(mean, dim=1).cpu().numpy()
            for b, (row, kind) in enumerate(actual):
                features[row, kind] = encoded[b]
    seconds = time.perf_counter()-start
    for i, row in enumerate(views):
        if row['entity_wordpiece_count'] and row['control_equals_entity']:
            features[i, 1] = features[i, 0]
    assert np.isfinite(features).all()
    with (OUT/'features.npz').open('xb') as stream:
        np.savez_compressed(stream, query_ids=qids, group_ids=gids, R=raw, E=features[:,0], N=features[:,1],
                            original_input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
                            original_special_mask=tokens['special_tokens_mask'], view_masks=masks)
    count = np.array([r['entity_wordpiece_count'] for r in views])
    ordinary_count = np.array([r['ordinary_wordpiece_count'] for r in views])
    summary = {'status': 'complete_query_only_views_no_fit', 'queries': len(qids),
               'nonzero_mask_queries': int(np.sum(count > 0)), 'zero_mask_queries': int(np.sum(count == 0)),
               'same_control_nonzero_queries': sum(r['entity_wordpiece_count'] > 0 and r['control_equals_entity'] for r in views),
               'masked_WordPieces_per_view': int(count.sum()),
               'mask_fraction_mean': float(np.mean(count/ordinary_count)),
               'mask_fraction_max': float(np.max(count/ordinary_count)),
               'control_overlap_WordPieces': sum(r['overlap_positions'] for r in views),
               'unique_encoder_sequences': len(jobs), 'encoder_batches': (len(jobs)+7)//8,
               'encoder_seconds': seconds, 'new_encoder_training_steps': 0, 'retained_queries_accessed': 0,
               'mean_raw_entity_cosine': float(np.mean(np.sum(raw*features[:,0], axis=1))),
               'mean_raw_control_cosine': float(np.mean(np.sum(raw*features[:,1], axis=1))),
               'rules_changed_after_observation': False, 'new_answer_calls': 0}
    write_new(OUT/'view_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


def fit_and_evaluate():
    checked_protocol()
    if (OUT/'results.json').exists() or (OUT/'predictions.npz').exists() or (OUT/'failure.json').exists():
        raise ValueError('Existing result or terminal failure; do not refit')
    sys.path.insert(0, str(BASE))
    import weighted_linear_probe_refined as probe
    with np.load(OUT/'features.npz', allow_pickle=False) as a:
        qids, gids = a['query_ids'], a['group_ids']
        raw, entity, control = (a[k].astype(np.float64) for k in ('R','E','N'))
    with np.load(ROLES, allow_pickle=False) as a:
        assert np.array_equal(qids, a['query_ids']) and np.array_equal(gids, a['group_ids'])
        folds = a['outer_fold']
    with np.load(SOURCE/'measurement_scores.npz', allow_pickle=False) as a:
        assert np.array_equal(qids, a['query_ids']) and np.array_equal(gids, a['group_ids'])
        average = a['outcomes'].mean(axis=2)
    with np.load(SOURCE/'actions.npz', allow_pickle=False) as a:
        assert np.array_equal(qids, a['query_ids']) and np.array_equal(gids, a['group_ids'])
        frozen = np.c_[a['M6_switch'], a['F_switch']]
    gap = average[:,0,0]-average[:,1,0]
    matrices = {'R_original': raw, 'E_masked': entity, 'D_dual': np.c_[raw,entity],
                'N_control': np.c_[raw,control], 'H_duplicate': np.c_[raw,raw]}
    predictions = np.empty((len(qids),len(ARMS)))
    heads, diagnostics = {}, []
    began = time.perf_counter()
    for fold in range(4):
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(gids[fit]) & set(gids[held])
        weights = np.abs(gap[fit])
        if not np.any(gap[fit] > 0) or not np.any(gap[fit] < 0):
            write_new(OUT/'failure.json', {'reason': 'degenerate_weighted_classes', 'fold': fold, 'completed_fits': len(diagnostics)})
            raise RuntimeError('No finite unpenalized-bias fit; stopped without recipe change')
        for j, arm in enumerate(ARMS):
            trained = probe.fit(matrices[arm][fit], (gap[fit]>0).astype(float), weights, regularization=SPEC['lambda'])
            row = {'fold': fold, 'arm': arm, 'fit_queries': len(fit), 'active_fit_queries': int(np.sum(weights>0)),
                   'accepted': trained['accepted'], 'loss': trained['loss'], 'grad_inf': trained['grad_inf'],
                   'elapsed_seconds': trained['elapsed_seconds'],
                   'refinement_steps': trained['numerical_refinement']['iterations']}
            diagnostics.append(row)
            if not trained['accepted']:
                write_new(OUT/'failure.json', {'reason': 'existing_numerical_budget_not_accepted', 'fits': diagnostics})
                raise RuntimeError('Numerical acceptance failed; no extension')
            coef, intercept = trained['coef'], trained['intercept']
            predictions[held,j] = matrices[arm][held] @ coef + intercept
            heads[f'coef_{fold}_{arm}'] = coef
            heads[f'bias_{fold}_{arm}'] = np.asarray(intercept)
            if arm == 'H_duplicate':
                row['coefficient_half_max_difference'] = float(np.max(np.abs(coef[:384]-coef[384:])))
    seconds = time.perf_counter()-began
    assert np.isfinite(predictions).all()
    switches = np.c_[np.ones(len(qids),bool), np.zeros(len(qids),bool), frozen, predictions>0]
    utilities = np.where(switches[:,:,None], average[:,None,0], average[:,None,1])
    means = utilities.mean(axis=0)
    target = POLICIES.index('D_dual')
    comparisons = {}
    for name in REFERENCES:
        difference = float(means[target,0]-means[POLICIES.index(name),0])
        threshold = .01 if name in ('Dense','BM25') else .002
        comparisons[name] = {'f1_gain': difference, 'minimum_point_gain': threshold,
                             'point_gate': difference >= threshold, 'CI': None, 'confidence': SPEC['CI_each']}
    point_pass = all(r['point_gate'] for r in comparisons.values())
    bootstrap = None
    if point_pass:
        _, groups = np.unique(gids, return_inverse=True)
        sizes = np.bincount(groups)
        totals = np.zeros((len(sizes),len(REFERENCES)))
        difference = np.column_stack([utilities[:,target,0]-utilities[:,POLICIES.index(r),0] for r in REFERENCES])
        np.add.at(totals, groups, difference)
        rng = np.random.default_rng(SPEC['bootstrap_seed'])
        bootstrap = np.empty((SPEC['bootstrap_draws'],len(REFERENCES)))
        for start in range(0,len(bootstrap),200):
            stop = min(start+200,len(bootstrap))
            index = rng.integers(0,len(sizes),size=(stop-start,len(sizes)))
            bootstrap[start:stop] = totals[index].sum(1)/sizes[index].sum(1)[:,None]
        for j, name in enumerate(REFERENCES):
            comparisons[name]['CI'] = np.quantile(bootstrap[:,j],[.05/12,1-.05/12]).tolist()
    gate = point_pass and all(r['CI'][0] > 0 for r in comparisons.values())
    policies = {}
    for j, name in enumerate(POLICIES):
        selected = switches[:,j]
        policies[name] = {'f1': float(means[j,0]), 'em': float(means[j,1]),
                          'gain_over_Dense': float(means[j,0]-means[1,0]), 'BM25_count': int(selected.sum()),
                          'beneficial': int(np.sum(selected & (gap>0))), 'harmful': int(np.sum(selected & (gap<0))),
                          'ties': int(np.sum(selected & (gap==0))),
                          'benefit_mass': float(np.mean(selected*np.maximum(gap,0))),
                          'harm_mass': float(np.mean(selected*np.maximum(-gap,0)))}
    result = {'status': 'complete_consumed_external_group_OOF_development',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': str(OUT/'protocol.json'),
              'queries': len(qids), 'groups': len(set(gids)), 'policy_means': policies, 'D_comparisons': comparisons,
              'all_point_gates': point_pass, 'development_gate': gate,
              'decision': 'development_candidate_prepare_independent_protocol' if gate else 'close_proxy_mask_dual_view_recipe_no_rule_or_hyperparameter_extension',
              'fits_completed': len(diagnostics), 'fit_diagnostics': diagnostics,
              'fitting_and_scoring_seconds': seconds, 'bootstrap_draws_executed': SPEC['bootstrap_draws'] if point_pass else 0,
              'D_minus_Dense_by_fold': [float((utilities[folds==f,target,0]-utilities[folds==f,1,0]).mean()) for f in range(4)],
              'retained_queries_accessed': 0, 'new_retrieval_calls': 0, 'new_answer_calls': 0,
              'independent_gain_established': False, 'novelty_established': False, 'objective_achieved': False,
              'interpretation': 'proxy_targeted_masking_recipe_vs_structure_matched_position_masking_not_pure_entity_semantics_or_true_NER',
              'CI_role': SPEC['inference_role']}
    with (OUT/'predictions.npz').open('xb') as stream:
        saved = dict(query_ids=qids, group_ids=gids, outer_fold=folds, policy_names=np.asarray(POLICIES),
                     scores=predictions, switches=switches, query_policy_utilities=utilities, **heads)
        if bootstrap is not None:
            saved['conditional_bootstrap_differences'] = bootstrap
        np.savez_compressed(stream, **saved)
    write_new(OUT/'results.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='fit_diagnostics'},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--encode', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else encode() if args.encode else fit_and_evaluate()
