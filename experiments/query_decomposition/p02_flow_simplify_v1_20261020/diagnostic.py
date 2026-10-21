"""Actual changed-trajectory exposure/efficiency diagnostics, entirely offline.

Candidate retention measures documents really written into saved model inputs;
it is neither gold support recall nor proof of factual correctness. Complete
successful cases are compared with frozen p02 on the identical available IDs.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

from .score import ROOT, HERE, OUT, OLD, DATASETS, read, rows, sha, select_case

ARMS = ('union20', 'discount20', 'single_final')


def _distribution(values):
    values = list(values)
    return {'n': len(values), 'total': sum(values), 'mean': sum(values) / len(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'histogram': {str(key): count for key, count in sorted(Counter(values).items())}}


def _reader(original, rewritten, packet, prompt, *, kind, step_id=None):
    active = [original]
    if rewritten is not None and rewritten['query'] != original['query']:
        active.append(rewritten)
    ids = [doc['doc_id'] for doc in packet['docs']]
    candidate_ids = {doc['doc_id'] for route in active for doc in route['docs']}
    block_present = [doc['block'] in prompt for doc in packet['docs']]
    rewritten_tail = [] if len(active) == 1 else rewritten['docs'][5:10]
    return {'kind': kind, 'step_id': step_id, 'dual_route': len(active) == 2,
            'original_query': original['query'], 'rewritten_query': rewritten['query'] if rewritten else None,
            'candidate_count': len(candidate_ids), 'reader_count': len(ids),
            'unique_reader_count': len(set(ids)),
            'full_union_retained': set(ids) == candidate_ids and len(ids) == len(set(ids)),
            'every_selected_block_present': all(block_present),
            'blocks_present': sum(block_present), 'blocks_expected': len(block_present),
            'original_docs_dropped': len({doc['doc_id'] for doc in original['docs']} - set(ids)),
            'rewritten_docs_dropped': len({doc['doc_id'] for doc in rewritten['docs']} - set(ids)) if len(active) == 2 else 0,
            'rewritten_rank6_10_occurrences': len(rewritten_tail),
            'rewritten_rank6_10_retained': sum(doc['doc_id'] in set(ids) for doc in rewritten_tail),
            'reader_raw_text_chars': sum(len(doc['text']) for doc in packet['docs']),
            'complete_prompt_chars': len(prompt), 'doc_ids': ids}


def _case_detail(case, value, baseline):
    arm = value['arm']
    readers, old_readers = [], []
    for node in value['nodes']:
        readers.append(_reader(node['original_retrieval'], node.get('rewritten_retrieval'), node['reader_packet'],
                               node['child_prompt'], kind=node['role'], step_id=node['step_id']))
    # p02's complete case export references its original B0 root packet via
    # source_prefix instead of directly storing original_retrieval itself.
    old_root = baseline.get('original_retrieval')
    if old_root is None:
        old_root = read(ROOT / baseline['source_prefix'])['original_retrieval']
    for node in baseline['nodes']:
        old_readers.append(_reader(node['original_retrieval'], node.get('rewritten_retrieval'), node['reader_packet'],
                                   node['child_prompt'], kind=node['role'], step_id=node['step_id']))
    old_readers.append(_reader(old_root, None, old_root, baseline['final_prompt'], kind='final'))
    if arm == 'single_final':
        final = value['final_retrieval']
        readers.append(_reader(final['original_retrieval'], final.get('rewritten_retrieval'), value['final_reader_packet'],
                               value['final_prompt'], kind='final'))
        routing_ok = (all(node['role'] == 'ordinary' for node in value['nodes'])
                      and 'Candidate conclusion for the Original question' not in value['helper']
                      and final['prior_complete_answers'] == value['history_all']
                      and all(doc['block'] in value['final_context'] for doc in value['final_reader_packet']['docs']))
    else:
        readers.append(_reader(value['original_retrieval'], None, value['final_reader_packet'], value['final_prompt'], kind='final'))
        routing_ok = (value['nodes'][-1]['role'] == 'goal'
                      and value['final_reader_packet'] == value['original_retrieval']
                      and 'Candidate conclusion for the Original question' in value['helper'])
    old_nodes = {(node['step_id'], node['request'], node['role']): node for node in baseline['nodes']}
    same_step = []
    for node in value['nodes']:
        old = old_nodes.get((node['step_id'], node['request'], node['role']))
        if old:
            same_step.append({'step_id': node['step_id'],
                              'rewrite_query_changed': node['rewritten_query'] != old['rewritten_query'],
                              'candidate_set_changed': ({doc['doc_id'] for route in [node['original_retrieval'], node.get('rewritten_retrieval')] if route for doc in route['docs']}
                                                        != {doc['doc_id'] for route in [old['original_retrieval'], old.get('rewritten_retrieval')] if route for doc in route['docs']}),
                              'answer_changed': node['answer'] != old['answer']})
    calls = [call for call in value['logical_calls'] if call['status'] == 'success']
    usage = {key: sum((call.get('usage') or {}).get(key, 0) or 0 for call in calls)
             for key in ['prompt_tokens', 'completion_tokens']}
    return {'id': case['id'], 'dataset': case['dataset'], 'arm': arm,
            'readers': readers, 'same_ID_p02_readers': old_readers,
            'final_retrieval_routing_verified': routing_ok,
            'same_request_step_comparisons': same_step,
            'logical_success_calls': len(calls), 'exact_cache_reused_calls': sum(call.get('reused', False) for call in calls),
            'logical_usage': usage,
            'logical_cold_usage_estimate_cny': usage['prompt_tokens'] * .8e-6 + usage['completion_tokens'] * 2e-6,
            'logical_usage_is_not_actual_incremental_charge': True}


def _summary(details):
    readers = [reader for case in details for reader in case['readers']]
    old = [reader for case in details for reader in case['same_ID_p02_readers']]
    leaves = [reader for reader in readers if reader['kind'] != 'final']
    final = [reader for reader in readers if reader['kind'] == 'final']
    dual = [reader for reader in readers if reader['dual_route']]
    same_steps = [step for case in details for step in case['same_request_step_comparisons']]
    return {'complete_success_cases': len(details),
            'reader_instances': len(readers), 'child_reader_instances': len(leaves), 'final_reader_instances': len(final),
            'dual_reader_instances': len(dual),
            'child_doc_count': _distribution(reader['reader_count'] for reader in leaves),
            'dual_doc_count': _distribution(reader['reader_count'] for reader in dual),
            'final_doc_count': _distribution(reader['reader_count'] for reader in final),
            'all_actual_reader_docs': _distribution(reader['reader_count'] for reader in readers),
            'same_ID_p02_reader_docs': _distribution(reader['reader_count'] for reader in old),
            'actual_complete_prompt_chars': sum(reader['complete_prompt_chars'] for reader in readers),
            'same_ID_p02_complete_prompt_chars': sum(reader['complete_prompt_chars'] for reader in old),
            'candidate_union_not_fully_retained_instances': sum(not reader['full_union_retained'] for reader in readers),
            'missing_selected_blocks': sum(reader['blocks_expected'] - reader['blocks_present'] for reader in readers),
            'reader_instances_over20': sum(reader['reader_count'] > 20 for reader in readers),
            'rewritten_rank6_10_occurrences_retrieved': sum(reader['rewritten_rank6_10_occurrences'] for reader in readers),
            'rewritten_rank6_10_occurrences_read': sum(reader['rewritten_rank6_10_retained'] for reader in readers),
            'original_route_docs_dropped': sum(reader['original_docs_dropped'] for reader in readers),
            'rewritten_route_docs_dropped': sum(reader['rewritten_docs_dropped'] for reader in readers),
            'same_ID_p02_rewritten_route_docs_dropped': sum(reader['rewritten_docs_dropped'] for reader in old),
            'final_routing_errors': sum(not case['final_retrieval_routing_verified'] for case in details),
            'same_request_steps_compared': len(same_steps),
            'same_request_rewrite_changed': sum(step['rewrite_query_changed'] for step in same_steps),
            'same_request_candidate_set_changed': sum(step['candidate_set_changed'] for step in same_steps),
            'same_request_answer_changed': sum(step['answer_changed'] for step in same_steps),
            'logical_success_calls': sum(case['logical_success_calls'] for case in details),
            'exact_cache_reused_calls': sum(case['exact_cache_reused_calls'] for case in details),
            'logical_prompt_tokens': sum(case['logical_usage']['prompt_tokens'] for case in details),
            'logical_completion_tokens': sum(case['logical_usage']['completion_tokens'] for case in details),
            'logical_cold_usage_estimate_cny': sum(case['logical_cold_usage_estimate_cny'] for case in details)}


def diagnose(*, sample='pilot60', output_dir=None):
    samples_path = HERE / ('pilot60.runtime.frozen.jsonl' if sample == 'pilot60' else 'runtime300.frozen.jsonl')
    samples = rows(samples_path)
    output_dir = Path(output_dir or OUT / 'diagnostics' / sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    details, states, errors = [], {arm: Counter() for arm in ARMS}, []
    for case in samples:
        _, old = select_case(OLD, case['dataset'], 'p02_role_headers', case['qid'])
        assert old and old['status'].startswith('success')
        for arm in ARMS:
            path, value = select_case(OUT, case['dataset'], arm, case['qid'])
            states[arm][value['status'] if value else 'unstarted'] += 1
            if not value or not value['status'].startswith('success'):
                continue
            try:
                details.append(_case_detail(case, value, old))
            except Exception as exc:
                errors.append({'id': case['id'], 'arm': arm, 'source': str(path), 'error': type(exc).__name__ + ': ' + str(exc)})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'sample': sample,
              'sample_sha256': sha(samples_path), 'planned_per_arm': len(samples),
              'scope': 'Actual complete successful changed trajectories on the same available IDs; algorithm failures/infrastructure missingness remain visible in status counts but do not have an invented complete reader trajectory.',
              'not_fixed_packet_replay': True,
              'historical_19_support_loss_nodes_not_reused_as_new_rate': True,
              'support_recall_or_semantic_relevance_evaluated': False,
              'gold_source_mapping_loaded': False,
              'support_note': 'No gold relevance labels are used here. Full candidate retention does not establish support completeness or answer correctness.',
              'arms': {arm: {**_summary([case for case in details if case['arm'] == arm]),
                            'statuses': dict(states[arm]),
                            'by_dataset': {ds: _summary([case for case in details if case['arm'] == arm and case['dataset'] == ds]) for ds in DATASETS}}
                       for arm in ARMS},
              'errors': errors, 'new_API_calls': 0, 'new_index_searches': 0,
              'actual_payment_note': 'Logical cold usage includes cache reuse; use report.py/budget.sqlite3 for actual task incremental cost and unresolved reservations.',
              'code_sha256': sha(Path(__file__))}
    (output_dir / 'EXPOSURE_EFFICIENCY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output_dir / 'CASE_DETAILS.jsonl').write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in details), encoding='utf8')
    lines = ['实际新轨迹：候选曝光与效率诊断', f'样本：{sample}；每组计划{len(samples)}题',
             '只对真实完整成功轨迹计阅读量；状态另列，不代替正式F1全分母。',
             '候选保留不是支持文档召回率；未复用旧19节点发生率作为新结果。', '',
             'arm\t完整成功\t双路reader\t双路平均篇数\t最多篇数\t候选丢失reader\t缺失block\t改写rank6-10保留/召回\t最终平均篇数\t路由错误']
    for arm in ARMS:
        a = result['arms'][arm]
        mean = a['dual_doc_count']['mean']
        final_mean = a['final_doc_count']['mean']
        lines.append('\t'.join([arm, str(a['complete_success_cases']), str(a['dual_reader_instances']),
                                f'{mean:.3f}' if mean is not None else '—', str(a['all_actual_reader_docs']['max']),
                                str(a['candidate_union_not_fully_retained_instances']), str(a['missing_selected_blocks']),
                                f'{a["rewritten_rank6_10_occurrences_read"]}/{a["rewritten_rank6_10_occurrences_retrieved"]}',
                                f'{final_mean:.3f}' if final_mean is not None else '—', str(a['final_routing_errors'])]))
        lines.append('  状态：' + json.dumps(a['statuses'], ensure_ascii=False))
        lines.append(f'  同题p02的改写文档裁剪出现次数：{a["same_ID_p02_rewritten_route_docs_dropped"]}；当前：{a["rewritten_route_docs_dropped"]}')
        lines.append(f'  同问句节点改写变化/候选集变化/子答案变化：{a["same_request_rewrite_changed"]}/{a["same_request_candidate_set_changed"]}/{a["same_request_answer_changed"]}，比较节点{a["same_request_steps_compared"]}')
    lines += ['', 'single_final最终reader直接读取其原query双路并集；union20/discount20仍通过goal完整句传递改写信息，最终原文保留原query Top10。',
              '错误：' + json.dumps(errors, ensure_ascii=False),
              '本诊断没有API或索引调用；实际付费以预算账本为准。']
    (output_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return result, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['pilot60', '300'], default='pilot60')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    value, directory = diagnose(sample=args.sample, output_dir=args.output_dir)
    print(json.dumps({'sample': args.sample, 'arms': {arm: {key: value['arms'][arm][key] for key in [
        'complete_success_cases', 'dual_reader_instances', 'candidate_union_not_fully_retained_instances',
        'missing_selected_blocks', 'rewritten_rank6_10_occurrences_retrieved', 'rewritten_rank6_10_occurrences_read', 'final_routing_errors']}
        for arm in ARMS}, 'errors': value['errors'], 'output_directory': str(directory)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
