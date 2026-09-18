"""Freeze a group-random external development sample and existing policies.

Local preparation only: no generator client, new training, or retained-group
inference. Current corpus/retrievers/generator/evaluator remain the contract.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
SOURCE = ROOT / 'outputs/router/external_sources/2wiki_april7_2021'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1'
TERM_RUN = OUT.parent / 'm6_term_aggregation_v1'
MODEL = Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from run_m6_capability_screen import require_execution_allowed, write_json, save_arrays

SPEC = {
    'scope': 'conditional_2Wiki_external_development_not_HotpotQA_confirmation',
    'eligible_rule': 'reuse_unflagged_and_all_sentences_from_completed_compatibility_rows',
    'population_queries': 1199, 'population_components': 1006,
    'sample_components': 256, 'retain_components': 750,
    'sample_seed': 2026091841,
    'sampling': 'PCG64_choice_without_replacement_of_sorted_eligible_component_IDs_then_include_all_eligible_queries_in_selected_components',
    'sample_role': 'one_external_development_measurement_no_training_or_tuning_before_complete_analysis',
    'retained_role': 'no_model_inference_retrieval_or_answers; future_role_requires_new_frozen_protocol',
    'frozen_policies': ['BM25', 'Dense', 'M6', 'F'],
    'M6_head': str(BASE / 'm6_candidate_v1/M6_head.pt'),
    'F_head': str(TERM_RUN / 'model_F.pt'),
    'M6_training': 'historical_complete_old9600_head',
    'F_training': 'old_fit6144_plus_BEIR_fit4165_fixed_IDF_term_pool_with_M6_concat',
    'policy_selection_disclosure': 'M6_previous_frozen_reference; F_best_fixed_component_in_consumed_development; neither_newly_confirmed',
    'router_threshold': 0, 'router_max_length': 128, 'router_batch': 8,
    'router_depth': 6, 'router_encoder_autocast': 'cuda_bfloat16',
    'M6_head_dtype': 'native_bfloat16_autocast', 'F_head_dtype': 'float32_same_as_development',
    'no_policy_retuning': True,
    'repeats_per_query_action': 3, 'primary': 'normalized_token_answer_F1', 'secondary': 'exact_match',
    'reference_answer': 'literal_original_2Wiki_answer_only; no_official_alias_expansion',
    'missingness': 'all_selected_queries_both_actions_each_three_successes; no_outcome_based_deletion_or_zero_imputation',
    'measurements': ['fixed_policy_F1_and_EM', 'M6_and_F_gains_against_both_fixed_actions',
                     'beneficial_and_harmful_switch_mass', 'three_way_leave_one_repeat_out_selector',
                     'empirical_mean3_oracle_descriptive_only'],
    'uncertainty': 'joint_cluster_bootstrap_sampled_components_preserve_query_members_and_all_repeats; no_FPC',
    'bootstrap_draws': 10000, 'bootstrap_seed': 2026091842,
    'power_claim': 'no_assured_one_percentage_point_detection; descriptive_external_development_measurement',
    'new_training_steps': 0, 'paid_collection_authorized': False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sample():
    require_execution_allowed()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'sampling_protocol.json'
    if protocol.exists():
        if read(protocol)['spec'] != SPEC:
            raise ValueError('Sampling protocol differs; never redraw for a preferred sample')
        if (OUT / 'sample_summary.json').exists():
            return read(OUT / 'sample_summary.json')
        raise RuntimeError('Partial sample preparation exists; inspect before recovery')
    write_json(protocol, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spec': SPEC})
    rows = [json.loads(line) for line in (SOURCE / 'current_corpus_compatibility_rows.jsonl').read_text(encoding='utf-8').splitlines()]
    eligible = [r for r in rows if r['unflagged_and_all_sentences']]
    components = np.array(sorted({r['component'] for r in eligible}), dtype=np.int64)
    assert len(eligible) == 1199 and len(components) == 1006
    chosen = set(np.random.default_rng(SPEC['sample_seed']).choice(components, 256, replace=False).tolist())
    selected = sorted([r for r in eligible if r['component'] in chosen], key=lambda r: r['query_id'])
    retained = [r for r in eligible if r['component'] not in chosen]
    assert len({r['component'] for r in retained}) == 750
    selected_ids = {r['query_id'] for r in selected}
    # Source gold answers are projected for later scoring, never for policy input.
    raw = {str(r['_id']): r for r in read(SOURCE / 'dev.json') if str(r['_id']) in selected_ids}
    assert set(raw) == selected_ids
    with (OUT / 'questions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in selected:
            q = raw[row['query_id']]
            record = {'query_id': row['query_id'], 'group_id': f'2wiki_april2021_c{row["component"]}',
                      'component': row['component'], 'question': q['question'],
                      'reference_answers': [q['answer']], 'type': row['type']}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    save_arrays(OUT / 'partition.npz', sample_query_ids=np.array([r['query_id'] for r in selected]),
                sample_components=np.array([r['component'] for r in selected]),
                retained_query_ids=np.array([r['query_id'] for r in retained]),
                retained_components=np.array([r['component'] for r in retained]))
    contract = read(BASE / 'm6_beir_validation_v1/execution_contract.json')
    contract.update(id='2Wiki_current_corpus_group256_development_v1',
                    status='scientific_configuration_fixed_local_preparation_only_paid_budget_not_authorized',
                    missingness=SPEC['missingness'],
                    resource_policy='All previous paid budgets closed. This new measurement has no authorized paid budget yet.',
                    external_calls_at_freeze=0, supersedes=None)
    contract['dataset'] = {'name': '2WikiMultihopQA', 'version': 'april7_2021_dev', 'scope': SPEC['scope'],
                           'selected_queries': len(selected), 'selected_groups': 256,
                           'reference_answer_policy': SPEC['reference_answer']}
    # Historical source hashes remain references to the inherited unchanged environment.
    write_json(OUT / 'execution_contract.json', contract)
    summary = {'status': 'sample_frozen_before_policy_predictions_retrieval_or_answers',
               'eligible_queries': 1199, 'eligible_components': 1006,
               'sample_queries': len(selected), 'sample_components': 256,
               'retained_queries': len(retained), 'retained_components': 750,
               'query_inclusion_probability': 256/1006,
               'sample_by_type': dict(Counter(r['type'] for r in selected)),
               'sample_group_size_counts': dict(Counter(Counter(r['component'] for r in selected).values())),
               'required_successful_answers': len(selected)*6, 'paid_calls': 0,
               'sampling_does_not_use': ['policy_scores', 'retrieved_documents', 'answer_outcomes']}
    write_json(OUT / 'sample_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def freeze_policies():
    summary = sample()
    completion = OUT / 'actions_freeze.json'
    if completion.exists():
        print(json.dumps(read(completion), ensure_ascii=False), flush=True)
        return
    import torch
    from transformers import AutoModel, AutoTokenizer
    import prepare_token_corpus_inputs as terms
    from router_term_aggregation import TermRouter
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = [json.loads(line) for line in (OUT / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    questions = [r['question'] for r in records]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    encoded = tokenizer(questions, padding='max_length', truncation=True, max_length=128,
                        return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors='np')
    mapping, _ = terms.sparse_assembly(questions, encoded['offset_mapping'], encoded['special_tokens_mask'], encoded['attention_mask'])
    index = ROOT / 'artifacts/_sparse_indexes/sqlite_bm25_04df51906ff9598b'
    descriptor = read(index / 'manifest.json')
    stats = terms.read_sqlite_bm25_term_stats(index / descriptor['artifacts']['database']['file'], set(mapping['terms'].tolist()))
    idf = np.array([stats[t][1] if t in stats else 0. for t in mapping['terms']], dtype=np.float32)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    model.encoder.layer = torch.nn.ModuleList(model.encoder.layer[:6])
    model.config.num_hidden_layers = 6
    model.pooler = None
    model.requires_grad_(False).to('cuda').eval()
    m6head = torch.load(SPEC['M6_head'], map_location='cuda', weights_only=True)
    fmodel = TermRouter('F').cuda().eval()
    fmodel.load_state_dict(torch.load(SPEC['F_head'], map_location='cuda', weights_only=True))
    n, began = len(records), time.perf_counter()
    x, pool = np.empty((n, 384), dtype=np.float32), np.empty((n, 384), dtype=np.float32)
    m6_logits = np.empty(n, dtype=np.float32)
    fallback = np.zeros(n, dtype=bool)
    with torch.inference_mode():
        for start in range(0, n, 8):
            stop = min(start+8, n)
            positions = list(range(start, stop)) + [stop-1]*(8-stop+start)
            batch = {key: torch.tensor(encoded[key][positions], device='cuda') for key in ('input_ids', 'attention_mask', 'token_type_ids')}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = model(**batch).last_hidden_state
                mask = batch['attention_mask'].bool() & ~torch.tensor(encoded['special_tokens_mask'][positions], device='cuda').bool()
                count = mask.sum(1)
                mean = (hidden.float()*mask[:, :, None]).sum(1)/count.clamp_min(1).float()[:, None]
                mean = torch.where((count == 0)[:, None], hidden[:, 0].float(), mean)
                m6 = torch.nn.functional.normalize(mean, dim=1)
                logits = torch.nn.functional.linear(m6, m6head['weight'], m6head['bias']).squeeze(1)
            x[start:stop], m6_logits[start:stop] = m6[:stop-start].cpu().numpy(), logits[:stop-start].float().cpu().numpy()
            raw = hidden.float().cpu().numpy()
            # Reuse the same FP64 term accumulation, then FP32 pooling as F training.
            for local, query in enumerate(range(start, stop)):
                a, b = mapping['query_term_indptr'][query:query+2]
                if a == b or float(idf[a:b].sum()) == 0:
                    pool[query], fallback[query] = x[query], True
                    continue
                h = np.empty((b-a, 384), dtype=np.float32)
                for j, term in enumerate(range(a, b)):
                    lo, hi = mapping['term_token_indptr'][term:term+2]
                    h[j] = mapping['term_token_weights'][lo:hi] @ raw[local, mapping['term_token_indices'][lo:hi]].astype(np.float64)
                h = torch.tensor(h, device='cuda')
                weights = torch.tensor(idf[a:b], device='cuda')
                pooled = (weights[:, None]/weights.sum()*h).sum(0)
                pool[query] = torch.nn.functional.normalize(pooled, dim=0, eps=1e-12).cpu().numpy()
        f_logits = fmodel.head(torch.tensor(np.c_[x, pool], device='cuda')).squeeze(1).cpu().numpy()
    assert all(np.isfinite(values).all() for values in (x, pool, m6_logits, f_logits))
    save_arrays(OUT / 'actions.npz', query_ids=np.array([r['query_id'] for r in records]),
                group_ids=np.array([r['group_id'] for r in records]), M6=x, F_pool=pool,
                M6_logits=m6_logits, F_logits=f_logits, M6_switch=m6_logits > 0, F_switch=f_logits > 0, fallback=fallback)
    # Small exact bindings are required by the existing future budget loader;
    # no corpus, embedding or index hash replay is performed.
    record = {'status': 'frozen_pre_retrieval_predictions_before_new_answers',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'queries': n, 'groups': 256,
              'source_spec': SPEC, 'actions': ['bm25', 'dense'], 'policies': SPEC['frozen_policies'],
              'M6_bm25_count': int((m6_logits > 0).sum()), 'F_bm25_count': int((f_logits > 0).sum()),
              'fallback_count': int(fallback.sum()), 'encoder_batches': (n+7)//8,
              'encoder_and_pool_seconds': time.perf_counter()-began,
              'new_training_steps': 0, 'retained_group_predictions': 0, 'retrieval_calls_at_freeze': 0, 'answer_calls_at_freeze': 0,
              'artifact_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  (OUT / 'actions.npz', OUT / 'questions.jsonl', Path(SPEC['M6_head']), Path(SPEC['F_head']))}}
    write_json(completion, record)
    print(json.dumps({key: value for key, value in record.items() if key not in ('source_spec', 'artifact_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--freeze-policies', action='store_true')
    args = parser.parse_args()
    if args.freeze_policies:
        freeze_policies()
    elif args.sample:
        sample()
    else:
        print(json.dumps(SPEC, ensure_ascii=False, indent=2))
