"""Read-only final confirmation audit; no scoring before execution terminates.

Default/--check prints execution readiness without reading predictions or gold.
--report independently scores the frozen150 after continuation has terminated;
it writes new analysis files, never the controller's SCORES files or frozen code.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from . import fresh_confirm, confirm_data

OUT = fresh_confirm.OUT
RESUME = OUT / 'continuation_v001'
ARMS = ['parent_g01', 'p02_role_headers']
DATASETS = ['hotpotqa', '2wiki', 'musique']
DONE = {'complete', 'partial_missing_retained'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def check():
    path = RESUME / 'STATUS.json'
    state = read(path).get('event') if path.exists() else None
    return {'utc': utc(), 'ready_for_final_report': state in DONE,
            'continuation_state': state, 'new_API_calls': 0,
            'predictions_or_references_read_by_check': False}


def paired(values, planned):
    matched = [v for v in values if v['delta'] is not None]
    n = len(matched)
    total = sum(v['delta'] for v in matched)
    rng = np.random.default_rng(20261020)
    bootstrap = np.zeros(5000)
    for ds in DATASETS:
        dd = np.array([v['delta'] for v in matched if v['dataset'] == ds])
        if len(dd):
            bootstrap += dd[rng.integers(len(dd), size=(5000, len(dd)))].sum(axis=1) / n
    return {'n_cohort': planned, 'n_matched': n, 'n_unmatched': planned - n,
            'sum_matched_deltas': total,
            'parent_F1': sum(v['parent']['f1'] for v in matched) / n if n else None,
            'candidate_F1': sum(v['candidate']['f1'] for v in matched) / n if n else None,
            'delta_F1': total / n if n else None,
            'paired95': np.quantile(bootstrap, [.025, .975]).tolist() if n else None,
            'up': sum(v['delta'] > 1e-12 for v in matched),
            'tie': sum(abs(v['delta']) <= 1e-12 for v in matched),
            'down': sum(v['delta'] < -1e-12 for v in matched),
            'full_denominator_delta_bounds': [(total - (planned - n)) / planned,
                                               (total + (planned - n)) / planned],
            'full_denominator_bounds_definition': '(sum matched deltas +/- unmatched count) / full planned count; missing answers are not assigned zero',
            'bootstrap': '5000 paired resamples within dataset, seed20261020; confirmation interval'}


def summary(values, planned):
    usable = [v for v in values if v['f1'] is not None]
    n = len(usable)
    total = sum(v['f1'] for v in usable)
    return {'planned': planned, 'scorable': n, 'missing': planned - n,
            'F1_available': total / n if n else None,
            'EM_available': sum(v['em'] for v in usable) / n if n else None,
            'F1_full': total / planned if n == planned else None,
            'F1_full_bounds': ([total / planned] * 2 if n == planned else
                               [total / planned, (total + planned - n) / planned]),
            'statuses': dict(Counter(v['status'] for v in values))}


def main_sources(plan, sidecar):
    fresh_confirm.verify(plan)
    assert plan['selected'] == 'p02_role_headers' and plan['arms_available'] == ARMS
    assert fresh_confirm.sha(OUT / 'PROTOCOL.frozen.json') == sidecar['original_protocol_sha256']
    assert sidecar['candidate'] == 'p02_role_headers' and sidecar['original_cohort'] == 150
    assert sidecar['frozen_source_pins'] == plan['source_pins']
    path = Path(__file__).with_name('confirm_resume.py')
    assert fresh_confirm.sha(path) == sidecar['source_sha256']
    for name, expected in sidecar['protected_original_cases'].items():
        assert fresh_confirm.sha(ROOT / name) == expected, 'Original case changed: ' + name
    assert len(sidecar['protected_original_cases']) == 150
    ref_receipt = read(OUT / 'REFERENCES.frozen.json')
    assert fresh_confirm.sha(confirm_data.REFERENCES) == ref_receipt['references_sha256']
    core = confirm_data.private_core()
    assert core.metric_pins() == ref_receipt['metric_sources']
    return {'frozen_method_source_pins_verified': len(plan['source_pins']),
            'original_parent_r000_cases_unchanged': 150,
            'continuation_source_sha256': sidecar['source_sha256'],
            'runtime_sha256': fresh_confirm.sha(fresh_confirm.RUNTIME),
            'reference_sha256': ref_receipt['references_sha256'],
            'native_metrics_and_aliases_hashes_match': True}


def request_record(call, cache):
    key = (call['source'], call['request_id'])
    if key not in cache:
        path = ROOT / call['source']
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            row = db.execute('SELECT payload,status,result FROM requests WHERE id=?', (call['request_id'],)).fetchone()
        assert row is not None
        request = json.loads(row[0])
        cache[key] = {'payload': request, 'status': row[1], 'result': json.loads(row[2]) if row[2] else None}
    value = cache[key]
    assert digest(value['payload']) == call['payload_sha256']
    return value


def execution_audit(plan, sidecar, cases, bound, records):
    cache, counts, issues, first_payloads = {}, Counter(), [], []
    for cid, expected in sidecar['first_payload_sha256'].items():
        q = next(v for v in cases if v['id'] == cid)
        old = read(OUT / 'rounds/r000/cases' / q['dataset'] / 'parent_g01' / (q['qid'] + '.json'))
        old_call = next(v for v in old['logical_calls'] if v['request_id'] == old['failed_request_id'])
        old_row = request_record(old_call, cache)
        assert digest(old_row['payload']) == expected
        new = read(OUT / 'rounds/r001/cases' / q['dataset'] / 'parent_g01' / (q['qid'] + '.json'))
        physical = [call for call in new['logical_calls'] if not call['reused']]
        assert physical and physical[0]['payload_sha256'] == expected
        assert request_record(physical[0], cache)['payload'] == old_row['payload']
        first_payloads.append({'id': cid, 'payload_sha256': expected,
                               'old_request_id': old_call['request_id'],
                               'new_request_id': physical[0]['request_id'],
                               'old_status': old_row['status'],
                               'new_status': request_record(physical[0], cache)['status'],
                               'old_error_code': (old_row['result'] or {}).get('error_code'),
                               'actual_payload_equal': True})
        assert new['nodes'][:len(old['nodes'])] == old['nodes'], 'Successful parent prefix changed'
        counts['parent_exact_first_failed_payload_retried'] += 1
        counts['parent_successful_node_prefix_unchanged'] += 1

    continuation_case_paths = list((OUT / 'rounds/r001/cases').glob('*/parent_g01/*.json'))
    continuation_case_paths += list((OUT / 'rounds').glob('r*/cases/*/p02_role_headers/*.json'))
    for path in continuation_case_paths:
        record = read(path)
        for call in record.get('logical_calls', []):
            actual = request_record(call, cache)
            counts['continuation_logical_calls_payload_verified'] += 1
            if call['reused']:
                assert actual['status'] == 'success', 'A non-success receipt was reused as a prefix'
                counts['successful_full_payload_receipts_reused'] += 1
            if record['arm'] == 'p02_role_headers' and not call['reused']:
                assert call['stage'] == 'legacy/final', 'Candidate paid for upstream repair'
                q = next(v for v in cases if v['id'] == record['id'])
                parent = records[q['id']]['parent']
                assert parent and parent['status'].startswith('success')
                prompt, _ = bound.strategy().render_final(q['question'], parent['nodes'], parent['original_retrieval'], plan['readout'], ARMS[1])
                assert actual['payload'] == payload(plan['model'], prompt, 64)
                counts['candidate_physical_requests_exact_final_only'] += 1

    for q in cases:
        parent, candidate = records[q['id']]['parent'], records[q['id']]['candidate']
        if candidate and candidate['status'].startswith('success'):
            assert parent and parent['status'].startswith('success')
            assert parent['nodes'] == candidate['nodes']
            prompt, context = bound.strategy().render_final(q['question'], parent['nodes'], parent['original_retrieval'], plan['readout'], ARMS[1])
            assert candidate['final_prompt'] == prompt and candidate['final_context'] == context
            final_call = next(v for v in candidate['logical_calls'] if v['stage'] == 'legacy/final')
            actual = request_record(final_call, cache)
            assert actual['status'] == 'success' and actual['result']['answer'].strip() == candidate['prediction']
            counts['candidate_nodes_and_actual_final_payload_verified'] += 1
    return {'checks': dict(counts), 'first_recovery_payloads': first_payloads, 'issues': issues,
            'failed_parent_never_relabelled_as_algorithm_failure_by_report': True,
            'success_cache_is_complete_payload_only': True}


def physical_and_cost():
    physical, attempts = Counter(), Counter()
    for path in sorted((OUT / 'rounds').glob('r*/methods/*/*/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            physical.update(dict(db.execute('SELECT status,count(*) FROM requests GROUP BY status')))
            attempts.update(dict(db.execute('SELECT status,count(*) FROM attempts GROUP BY status')))
    cost = defaultdict(lambda: {'known_cny': 0., 'reserved_cny': 0., 'submissions': 0, 'statuses': Counter()})
    prefix = str(OUT.relative_to(OUTPUT))
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        for namespace, reserved, actual, status in db.execute('SELECT namespace,reserved,actual,status FROM charges WHERE namespace LIKE ?', (prefix + '%',)):
            parts = Path(namespace).parts
            label = 'original_parent_r000' if 'r000' in parts and 'parent_g01' in parts else 'continuation'
            for group in [label, 'confirmation_all']:
                cost[group]['known_cny'] += actual or 0.
                cost[group]['reserved_cny'] += reserved if actual is None else 0.
                cost[group]['submissions'] += 1
                cost[group]['statuses'][status] += 1
        global_row = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    for group in cost.values():
        group['occupied_cny'] = group['known_cny'] + group['reserved_cny']
        group['statuses'] = dict(group['statuses'])
    return {'request_statuses': dict(physical), 'attempt_statuses': dict(attempts), 'cost': dict(cost),
            'cumulative': {'known_cny': global_row[0], 'reserved_cny': global_row[1], 'occupied_cny': sum(global_row), 'cap_cny': 300},
            'cost_scope': 'Returned-usage price estimates and retained unknown reservations, not verified billing; original parent and continuation separated.'}


def compare_scores(report):
    comparisons = {}
    for name in ['SCORES.progress.json', 'SCORES.frozen.json']:
        path = OUT / name
        if not path.exists():
            continue
        old = read(path)
        pairs = [('parent', report['parent'], old['parent']),
                 ('candidate', report['candidate_summary'], old['candidate_summary'])]
        for ds in DATASETS:
            pairs += [(ds + '/parent', report['by_dataset'][ds]['parent'], old['by_dataset'][ds]['parent']),
                      (ds + '/candidate', report['by_dataset'][ds]['candidate'], old['by_dataset'][ds]['candidate'])]
        checked = 0
        for label, ours, theirs in pairs:
            for field in ['planned', 'scorable', 'missing', 'F1_available', 'EM_available', 'F1_full']:
                a, b = ours[field], theirs[field]
                assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12), (label, field, a, b)
                checked += 1
        for label, ours, theirs in [('overall', report['paired'], old['paired'])] + [(ds, report['by_dataset'][ds]['paired'], old['by_dataset'][ds]['paired']) for ds in DATASETS]:
            for field in ['n_matched', 'parent_F1', 'candidate_F1', 'delta_F1', 'up', 'tie', 'down']:
                a, b = ours[field], theirs[field]
                assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12), (label, field, a, b)
                checked += 1
            a, b = ours['paired95'], theirs['paired95']
            assert (a is None and b is None) or (a is not None and b is not None and np.allclose(a, b, atol=1e-12, rtol=0)), label
        comparisons[name] = {'sha256': fresh_confirm.sha(path), 'numeric_fields_checked': checked, 'native_scores_and_bootstrap_exact': True}
    assert comparisons, 'No original scorer report exists after terminal continuation state'
    return comparisons


def report():
    readiness = check()
    if not readiness['ready_for_final_report']:
        print(json.dumps(readiness, ensure_ascii=False))
        return readiness
    plan, sidecar = read(OUT / 'PROTOCOL.frozen.json'), read(RESUME / 'PROTOCOL.frozen.json')
    sources = main_sources(plan, sidecar)
    bound = fresh_confirm.private_driver()
    cases = bound.runtime_cases()
    refs = {v['id']: v for v in rows(confirm_data.REFERENCES)}
    assert len(cases) == len(refs) == 150 and Counter(v['dataset'] for v in cases) == {ds: 50 for ds in DATASETS}
    metrics = {ds: metric(ds) for ds in DATASETS}
    details, records = [], {}
    for q in cases:
        ref = refs[q['id']]
        assert ref['question'] == q['question']
        value = {'id': q['id'], 'dataset': q['dataset']}
        records[q['id']] = {}
        for label, arm in [('parent', ARMS[0]), ('candidate', ARMS[1])]:
            rec = bound.selected(q, arm)
            records[q['id']][label] = rec
            status = rec['status'] if rec else 'unstarted'
            prediction = rec.get('prediction') if rec else None
            valid = isinstance(prediction, str) and (status.startswith('success') or status == 'algorithm_failure')
            score = metrics[q['dataset']](prediction, ref) if valid else {'em': None, 'f1': None}
            value[label] = {'arm': arm, 'status': status, **score}
        a, b = value['parent']['f1'], value['candidate']['f1']
        value['delta'] = b - a if a is not None and b is not None else None
        details.append(value)
    events = rows(RESUME / 'EVENTS.jsonl')
    provider_errors = [v for v in events if v['event'] == 'provider_error']
    rejection = [v for v in provider_errors if v['id'] == 'musique:3hop2__607269_467331_162182' and v.get('http_status') == 400 and 'data_inspection_failed' in json.dumps(v).lower()]
    result = {'utc': utc(), 'cohort': 'Frozen one-time confirmation150;50 per dataset; no tuning or reselection',
              'candidate_arm': ARMS[1], 'continuation_state': readiness['continuation_state'],
              'source_audit': sources, 'execution_audit': execution_audit(plan, sidecar, cases, bound, records),
              'parent': summary([v['parent'] for v in details], 150),
              'candidate_summary': summary([v['candidate'] for v in details], 150),
              'paired': paired(details, 150),
              'by_dataset': {ds: {'parent': summary([v['parent'] for v in details if v['dataset'] == ds], 50),
                                   'candidate': summary([v['candidate'] for v in details if v['dataset'] == ds], 50),
                                   'paired': paired([v for v in details if v['dataset'] == ds], 50)} for ds in DATASETS},
              'missing_ids': [v['id'] for v in details if v['delta'] is None],
              'execution_exception': sidecar['diagnostic_exception'],
              'provider_input_rejection_confirmed': bool(rejection),
              'provider_diagnostic_events': provider_errors,
              'physical_and_cost': physical_and_cost(),
              'all_case_native_scores': details, 'new_API_calls': 0, 'new_retrieval_calls': 0,
              'missing_answers_assigned_zero': False, 'candidate_reselected': False,
              'frozen_method_modified': False}
    result['original_scorer_comparison'] = compare_scores(result)
    review_path = RESUME / 'CHANGED_CASE_REVIEW.independent.json'
    if review_path.exists():
        review = read(review_path)
        changed = {v['id']: v for v in details if v['delta'] is not None and abs(v['delta']) > 1e-12}
        assert {v['id'] for v in review['changed_cases']} == set(changed)
        for value in review['changed_cases']:
            actual, scores = records[value['id']], changed[value['id']]
            assert actual['parent']['prediction'] == value['parent_prediction']
            assert actual['candidate']['prediction'] == value['candidate_prediction']
            assert abs(scores['parent']['f1'] - value['parent_f1']) < 1e-12
            assert abs(scores['candidate']['f1'] - value['candidate_f1']) < 1e-12
        result['changed_case_review'] = review
    locked_json(RESUME / 'AUDIT_CONFIRMATION150.json', result)
    (RESUME / 'REPORT_CONFIRMATION150.txt').write_text(render(result), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['utc', 'parent', 'candidate_summary', 'paired', 'by_dataset', 'missing_ids', 'provider_input_rejection_confirmed', 'original_scorer_comparison']}, ensure_ascii=False), flush=True)
    return result


def render(result):
    lines = ['冻结p02确认150：独立原生指标复核',
             '原始样本保持150题（每数据集50），仅缺失答案保留为空，不记作0或算法失败。',
             '终结状态：' + result['continuation_state'],
             '数据集 | 已配对/原定 | g01 F1% | p02 F1% | 差值百分点 | 配对95%区间百分点 | 升/平/降']
    for name, group in [('总体', result['paired'])] + [(ds, result['by_dataset'][ds]['paired']) for ds in DATASETS]:
        n = group['n_matched']
        if n:
            lo, hi = group['paired95']
            lines.append('%s | %d/%d | %.5f | %.5f | %+.5f | [%+.5f, %+.5f] | %d/%d/%d' %
                         (name, n, group['n_cohort'], group['parent_F1'] * 100, group['candidate_F1'] * 100,
                          group['delta_F1'] * 100, lo * 100, hi * 100, group['up'], group['tie'], group['down']))
    lines += ['', '全150分母：', 'g01：' + json.dumps(result['parent'], ensure_ascii=False),
              'p02：' + json.dumps(result['candidate_summary'], ensure_ascii=False),
              '总体差值保守范围：' + json.dumps(result['paired']['full_denominator_delta_bounds']) +
              '；(sum已配对差值 ± 未配对数)/150，未把缺答当成0。',
              '缺失：' + json.dumps(result['missing_ids'], ensure_ascii=False),
              '本次执行例外：' + result['execution_exception'],
              'Mu输入内容审核拒绝data_inspection_failed已由新诊断确认：' + str(result['provider_input_rejection_confirmed']),
              '输入/缓存审计：' + json.dumps(result['execution_audit'], ensure_ascii=False),
              '源与150原记录保护：' + json.dumps(result['source_audit'], ensure_ascii=False),
              '与原SCORES逐项比对：' + json.dumps(result['original_scorer_comparison'], ensure_ascii=False),
              '物理调用及费用：' + json.dumps(result['physical_and_cost'], ensure_ascii=False),
              '费用为按返回用量估算及未知请求预留，未核对账单；本报告零API/检索，不更改候选、模型、prompt、样本或算法。',
              '置信区间：各数据集内5000次配对bootstrap，seed20261020；缺失造成的全150不确定范围与149同题CI分别报告。',
              '这是一次固定候选的确认结果，不用其答案或分数继续选择/调整该候选。']
    if result.get('changed_case_review'):
        lines += ['', '唯一升分题事实审查：',
                  '149题平均F1微增，但唯一增益来自错误条约名与标准答案共享Treaty的词面得分；新增EM=0，未证明事实修复或稳定提升。',
                  json.dumps(result['changed_case_review'], ensure_ascii=False, indent=2)]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.report:
        report()
    else:
        print(json.dumps(check(), ensure_ascii=False))


if __name__ == '__main__':
    main()
