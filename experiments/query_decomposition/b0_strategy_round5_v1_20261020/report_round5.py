"""Read-only old300 scoring/input/cost report; never reads confirmation results.

Only analysis artifacts are written. No frozen code, payload, result, candidate
or selection is changed. Manual mechanism reviews are optional and must name
the exact parent/candidate strings, so a stale review cannot silently apply.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from . import experiment as execution
from .strategy import ACTUAL_MARKER, ARMS, render_final

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
ARM = ARMS[0]
ORDINARY = 'Intermediate candidate results\n\n'
GOAL = 'Candidate conclusion for the Original question\n\n'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def selected(dataset, qid):
    latest = None
    for path in sorted((OUT / 'rounds').glob(f'*/cases/{dataset}/{ARM}/{qid}.json')):
        value = read(path)
        latest = (path, value)
        if isinstance(value.get('prediction'), str) and (value['status'].startswith('success') or value['status'] == 'algorithm_failure'):
            return latest
    return latest


def physical_requests(expected):
    requests, issues = [], []
    attempt_statuses, repeated_attempt_ids = Counter(), []
    for path in sorted((OUT / 'rounds').glob(f'*/methods/*/{ARM}/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            attempt_statuses.update(dict(db.execute('SELECT status,count(*) FROM attempts GROUP BY status')))
            repeated_attempt_ids.extend({'source': str(path.relative_to(ROOT)), 'request_id': rid, 'attempts': count}
                                        for rid, count in db.execute('SELECT request_id,count(*) FROM attempts GROUP BY request_id HAVING count(*)>1'))
            for rid, stage, raw, status, result in db.execute('SELECT id,stage,payload,status,result FROM requests ORDER BY rowid'):
                actual = json.loads(raw)
                key = digest(actual)
                first = db.execute('SELECT min(started) FROM attempts WHERE request_id=?', (rid,)).fetchone()[0]
                value = json.loads(result) if result else {}
                if key not in expected or actual != expected.get(key, {}).get('payload'):
                    issues.append({'request_id': rid, 'issue': 'Unexpected complete payload', 'source': str(path.relative_to(ROOT))})
                requests.append({'request_id': rid, 'source': str(path.relative_to(ROOT)), 'stage': stage,
                                 'payload_sha256': key, 'id': expected.get(key, {}).get('id'),
                                 'status': status, 'first_started': first, 'finish_reason': value.get('finish_reason'),
                                 'http_status': value.get('http_status'), 'usage': value.get('usage', {})})
    groups = defaultdict(list)
    for request in requests:
        groups[request['payload_sha256']].append(request)
    repeated_success = []
    for key, values in groups.items():
        ordered = sorted(values, key=lambda item: item.get('first_started') or '')
        successful_seen = False
        for item in ordered:
            if successful_seen:
                repeated_success.append({'payload_sha256': key, 'request_id': item['request_id']})
            successful_seen = successful_seen or item['status'] == 'success'
    return {'submissions': len(requests), 'statuses': dict(Counter(v['status'] for v in requests)),
            'attempts': sum(attempt_statuses.values()), 'attempt_statuses': dict(attempt_statuses),
            'retry_attempts_same_request': sum(v['attempts'] - 1 for v in repeated_attempt_ids),
            'repeated_attempt_ids': repeated_attempt_ids,
            'unique_payloads': len(groups),
            'retry_submissions_same_payload': sum(len(v) - 1 for v in groups.values()),
            'successful_payload_submitted_again': repeated_success,
            'unexpected_payloads': issues, 'requests': requests}


def cost_summary():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                         (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
        global_row = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges').fetchone()
        states = dict(db.execute('SELECT status,count(*) FROM charges WHERE namespace LIKE ? GROUP BY status',
                                (str(OUT.relative_to(OUTPUT)) + '%',)))
    return {'round5_known_cny': own[0], 'round5_unknown_reserved_cny': own[1],
            'round5_occupied_cny': own[0] + own[1], 'round5_ledger_submissions': own[2],
            'round5_ledger_statuses': states, 'cumulative_known_cny': global_row[0],
            'cumulative_reserved_cny': global_row[1], 'cumulative_occupied_cny': global_row[0] + global_row[1],
            'global_cap_cny': 300, 'round5_subcap_cny': 2,
            'cost_scope': 'Incremental physical API charges/reserves in round5 namespace; existing parent work not charged again. Returned-usage price estimate, not verified bill.'}


def summary(data):
    matched = [v for v in data if v['delta'] is not None]
    n, m = len(data), len(matched)
    parent_total = sum(v['parent_f1'] for v in data)
    candidate_total = sum(v['candidate_f1'] for v in matched)
    boot = np.zeros(5000)
    rng = np.random.default_rng(20261020)
    for ds in ['hotpotqa', '2wiki', 'musique']:
        delta = np.array([v['delta'] for v in matched if v['dataset'] == ds])
        if len(delta):
            boot += delta[rng.integers(len(delta), size=(5000, len(delta)))].sum(1) / m
    return {'planned': n, 'scorable': m, 'missing': n - m,
            'parent_F1_all': parent_total / n,
            'candidate_F1_full': candidate_total / n if m == n else None,
            'candidate_F1_available': candidate_total / m if m else None,
            'same_parent_F1_available': sum(v['parent_f1'] for v in matched) / m if m else None,
            'delta_available': sum(v['delta'] for v in matched) / m if m else None,
            'candidate_full_F1_bounds': ([candidate_total / n] * 2 if m == n else
                                         [candidate_total / n, (candidate_total + n - m) / n]),
            'paired95_descriptive': np.quantile(boot, [.025, .975]).tolist() if m else None,
            'up': sum(v['delta'] > 1e-12 for v in matched),
            'tie': sum(abs(v['delta']) <= 1e-12 for v in matched),
            'down': sum(v['delta'] < -1e-12 for v in matched),
            'statuses': dict(Counter(v['status'] for v in data))}


def manual_reviews():
    path = OUT / 'MANUAL_REVIEWS.json'
    return read(path) if path.exists() else {}


def evidence_packets(base, parent):
    # Reader exposure is stage-specific: the goal reader saw these source
    # passages, while the final reader receives original Top10 plus generated
    # QA. A goal passage is final-visible only if its full text is also present.
    final_context = parent['final_context']
    root = [{**doc, 'full_passage_supplied_to_final_reader': doc['text'] in final_context}
            for doc in base['original_retrieval']['docs']]
    goal = [{**doc, 'full_passage_supplied_to_final_reader': doc['text'] in final_context}
            for doc in parent['nodes'][-1]['reader_packet']['docs']]
    return {'original_query_top10': root, 'goal_reader_top10': goal,
            'boundary': 'Original Top10 is supplied as raw text to the final reader. Goal Top10 is supplied upstream to the goal child reader; it is not automatically final-reader evidence. The per-document flag identifies exact full-text final visibility. Passage presence alone does not prove answer support.'}


def report():
    plan = read(HERE / 'PROTOCOL.frozen.json')
    execution.verify(plan)
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    sources = {v['id']: v for v in rows(HERE / 'SOURCES.frozen.jsonl')}
    refs = {v['id']: v for v in rows(execution.SIX_E / 'references300.local.frozen.jsonl')}
    scorers = {ds: metric(ds) for ds in ['hotpotqa', '2wiki', 'musique']}
    reviews = manual_reviews()
    expected, values, changed, input_checks = {}, [], [], Counter()
    issues = []
    for case in cases:
        source = sources[case['id']]
        assert execution.sha(ROOT / source['base']) == source['base_sha256']
        assert execution.sha(ROOT / source['control']) == source['control_sha256']
        base, parent = read(ROOT / source['base']), read(ROOT / source['control'])
        parent_prompt, parent_context = render_final(case['question'], parent['nodes'], base['original_retrieval'], plan['readout'], 'parent_p02')
        assert parent_prompt == parent['final_prompt'] and parent_context == parent['final_context']
        candidate_prompt, candidate_context = render_final(case['question'], parent['nodes'], base['original_retrieval'], plan['readout'], ARM)
        assert candidate_context == parent_context
        pprefix, pactual = parent_prompt.split(ACTUAL_MARKER, 1)
        cprefix, cactual = candidate_prompt.split(ACTUAL_MARKER, 1)
        assert pactual == cactual and cprefix.replace(ORDINARY, '').replace(GOAL, '') == pprefix
        assert Counter(candidate_prompt) == Counter(parent_prompt) + Counter(ORDINARY * 3 + GOAL * 4)
        req = payload(plan['model'], candidate_prompt, 64)
        expected[digest(req)] = {'id': case['id'], 'payload': req}
        input_checks['blueprint_actual_task_context_readout_preserved'] += 1
        pair = selected(case['dataset'], case['qid'])
        path, value = pair if pair else (None, None)
        status = value['status'] if value else 'unstarted'
        valid = value and isinstance(value.get('prediction'), str) and (status.startswith('success') or status == 'algorithm_failure')
        ref = refs[case['id']]
        pscore = scorers[case['dataset']](parent['prediction'], ref)
        cscore = scorers[case['dataset']](value['prediction'], ref) if valid else {'em': None, 'f1': None}
        item = {'id': case['id'], 'dataset': case['dataset'], 'query': case['question'],
                'status': status, 'parent_prediction': parent['prediction'],
                'candidate_prediction': value['prediction'] if valid else None,
                'parent_f1': pscore['f1'], 'candidate_f1': cscore['f1'],
                'delta': cscore['f1'] - pscore['f1'] if valid else None,
                'parent_em': pscore['em'], 'candidate_em': cscore['em']}
        values.append(item)
        if valid and status.startswith('success'):
            if value['nodes'] != parent['nodes'] or value['final_context'] != parent_context or value['final_prompt'] != candidate_prompt:
                issues.append({'id': case['id'], 'issue': 'Actual nodes/context/prompt differ from frozen single-factor blueprint'})
            else:
                input_checks['actual_success_case_nodes_context_prompt_exact'] += 1
            call = next(r for r in value['logical_calls'] if r['payload_sha256'] == digest(req))
            actual = receipt(call, req)
            assert actual['status'] == 'success' and actual['result']['answer'].strip() == value['prediction']
            input_checks['actual_success_receipt_payload_and_answer_exact'] += 1
        if valid and abs(item['delta']) > 1e-12:
            review = reviews.get(case['id'], {'classification': 'unreviewed', 'reason': 'Requires passage-based manual review; F1 change alone is not proof of factual change.'})
            if review.get('parent_prediction') not in [None, parent['prediction']] or review.get('candidate_prediction') not in [None, value['prediction']]:
                raise AssertionError('Stale manual review: ' + case['id'])
            changed.append({**item, 'reference_for_offline_review_only': ref,
                            'parent_source': source['control'], 'candidate_source': str(path.relative_to(ROOT)),
                            'goal_question': parent['nodes'][-1]['request'], 'goal_answer_unchanged': parent['nodes'][-1]['answer'],
                            'ordinary_QAs_unchanged': [{k: n[k] for k in ['step_id', 'request', 'answer']} for n in parent['nodes'][:-1]],
                            'actual_evidence': evidence_packets(base, parent), 'review': review})
    requests = physical_requests(expected)
    issues += requests['unexpected_payloads']
    if requests['successful_payload_submitted_again']:
        issues.append({'issue': 'An already successful identical payload was submitted again', 'details': requests['successful_payload_submitted_again']})
    overall = summary(values)
    complete_answers = overall['scorable'] == 300
    terminal = (OUT / 'TERMINAL.json').exists()
    best = (ARM if complete_answers and overall['candidate_F1_full'] > overall['parent_F1_all'] + 1e-12 else 'parent_p02') if terminal else 'pending'
    manual_counts = Counter(v['review']['classification'] for v in changed)
    result = {'utc': utc(), 'phase': 'complete' if terminal else 'progress', 'arm': ARM,
              'sample_role': 'Exposed diagnostic300 development comparison; no new confirmation claim.',
              'overall': overall, 'by_dataset': {ds: summary([v for v in values if v['dataset'] == ds]) for ds in scorers},
              'selection_by_frozen_rule': best,
              'input_audit': {'checks': dict(input_checks), 'issues': issues,
                              'all_observed_physical_payloads_match_blueprint': not requests['unexpected_payloads'],
                              'no_new_actual_task_or_factual_material': True},
              'physical_requests': requests, 'cost': cost_summary(), 'manual_review_counts': dict(manual_counts),
              'manual_review_complete': manual_counts.get('unreviewed', 0) == 0,
              'changed_queries': changed, 'all_case_scores': values,
              'method_changed_by_report': False, 'confirmation_answers_or_F1_read': False,
              'new_api_calls_for_report': 0, 'new_retrieval_calls_for_report': 0}
    locked_json(OUT / 'ANALYSIS.progress.json', result)
    text = text_report(result)
    (OUT / 'REPORT.progress.txt').write_text(text, encoding='utf-8')
    if terminal:
        locked_json(OUT / 'ANALYSIS.complete.json', result)
        (OUT / 'REPORT_FINAL.txt').write_text(text, encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['phase', 'overall', 'by_dataset', 'selection_by_frozen_rule', 'cost', 'manual_review_counts']}, ensure_ascii=False), flush=True)
    return result


def text_report(result):
    overall = result['overall']
    lines = ['Round5：示例role标题一致化', '状态：' + result['phase'],
             '仅old300开发比较，不读取确认150答案/F1，不改变候选或算法。',
             '样本：%d/300可计分；缺失%d。' % (overall['scorable'], overall['missing']),
             '同题结果（F1百分比；差值与区间为百分点）：',
             '数据集 | N | p02 | e05 | 差值 | 配对95%描述区间 | 升/平/降']
    for name, group in [('整体', overall), *result['by_dataset'].items()]:
        if group['scorable']:
            lo, hi = group['paired95_descriptive']
            lines.append('%s | %d/%d | %.5f | %.5f | %+.5f | [%+.5f, %+.5f] | %d/%d/%d' %
                         (name, group['scorable'], group['planned'], group['same_parent_F1_available'] * 100,
                          group['candidate_F1_available'] * 100, group['delta_available'] * 100,
                          lo * 100, hi * 100, group['up'], group['tie'], group['down']))
    lines += ['',
             '整体及同题区间：' + json.dumps(overall, ensure_ascii=False),
             '三数据集：' + json.dumps(result['by_dataset'], ensure_ascii=False),
             '冻结规则选优：' + result['selection_by_frozen_rule'],
             '输入审计：' + json.dumps(result['input_audit'], ensure_ascii=False),
             '本轮/累计费用：' + json.dumps(result['cost'], ensure_ascii=False),
             '物理调用：' + json.dumps({k: v for k, v in result['physical_requests'].items() if k != 'requests'}, ensure_ascii=False),
             '人工机制审阅：' + json.dumps(result['manual_review_counts'], ensure_ascii=False),
             '说明：5000次按数据集分层的同题bootstrap，seed20261020；区间仅描述该开发样本。',
             '只加示例中3个ordinary标题和4个goal标题；示例仍docs-first，actual task仍helper-first。',
             'F1增降不自动视为事实修复/退化；标准答案仅用于离线评分和审阅，不进入运行时。', '']
    if result['phase'] == 'complete':
        lines += ['冻结决策：' + ('保留p02；e05未达到完整300题F1严格高于p02的门槛，不启动该候选fresh90。'
                                  if result['selection_by_frozen_rule'] == 'parent_p02' else
                                  'e05通过old300开发选择门槛；这不是新样本的确认结果。'), '']
    if result['manual_review_complete']:
        reviews = [value['review'] for value in result['changed_queries']]
        lines += ['变化题人工审阅已全覆盖：%d/%d；其中词面/表达变化%d题，已证明事实修复%d题。'
                  % (len(reviews), len(result['changed_queries']),
                     sum(value['classification'].startswith('lexical') for value in reviews),
                     sum(value.get('factual_repair_demonstrated') is True for value in reviews)),
                  '“goal reader检到原文”与“最终reader直接读到原文”分开记录，不能把仅上游可见的材料当成最终直接证据。', '']
    for i, case in enumerate(result['changed_queries'], 1):
        lines += ['=' * 72, f'{i}. {case["id"]}', '原query：' + case['query'],
                  '参考答案（离线）：' + case['reference_for_offline_review_only']['answer'],
                  '父版：' + case['parent_prediction'], '候选：' + case['candidate_prediction'],
                  'F1：%s → %s；delta=%s' % (case['parent_f1'], case['candidate_f1'], case['delta']),
                  'goal问题：' + case['goal_question'], 'goal答案（未改）：' + case['goal_answer_unchanged'],
                  '普通前置QA（未改）：' + json.dumps(case['ordinary_QAs_unchanged'], ensure_ascii=False),
                  '审阅：' + json.dumps(case['review'], ensure_ascii=False),
                  '父记录：' + case['parent_source'], '候选记录：' + case['candidate_source']]
        # Full actual source material is retained here and in JSON, rather than
        # pretending a token-overlap heuristic proves answer support.
        for name in ['original_query_top10', 'goal_reader_top10']:
            lines.append('实际证据组：' + name + ('（原文进入最终reader）' if name == 'original_query_top10' else
                                                   '（上游goal reader；逐篇标注最终原文是否可见）'))
            for doc in case['actual_evidence'][name]:
                lines += [f'[{doc["rank"]}] {doc["source"]} | doc_id={doc["doc_id"]} | full_final_visibility={doc["full_passage_supplied_to_final_reader"]}', doc['text']]
        lines.append('')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    report()
