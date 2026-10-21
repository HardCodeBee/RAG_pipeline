"""Read-only analysis of old300 round3 cases; never submits model/retrieval calls.

Writes only new analysis/report artifacts, not frozen execution inputs or results.
The independent round2 dev90 is inspected only for file counts and ledger costs.
No dev90 answers/scores, or sealed150 files, are opened by this module.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = FAMILY / HERE.name
FRESH = FAMILY / 'b0_strategy_round2_v1_20261020/fresh_dev90'
REFERENCE = HERE.parent / 'b0_six_single_changes_v1_20261020/references300.local.frozen.jsonl'
DATASETS = ('hotpotqa', '2wiki', 'musique')
ARMS = ('q03_question_last', 'e03_goal_evidence')
ADDITIONAL_HEADER = '\n\nAdditional retrieved passage for the Original question\n\n'
REVIEW_PATH = OUT / 'CASE_REVIEWS.report_round3.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, value):
    """Report files are replaceable snapshots, never execution/frozen outputs."""
    assert path.parent == OUT and not path.name.endswith('.frozen.json')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def surface(text):
    return ' '.join(re.findall(r'\w+', text.casefold(), flags=re.UNICODE))


def contains_surface(text, phrase):
    phrase = surface(phrase)
    return bool(phrase) and (' ' + phrase + ' ') in (' ' + surface(text) + ' ')


def additional_document(parent, base):
    original = {doc['doc_id'] for doc in base['original_retrieval']['docs']}
    return next((doc for doc in parent['nodes'][-1]['reader_packet']['docs']
                 if doc['doc_id'] not in original), None)


def chosen_cases():
    """First successful/algorithm-terminal record; never select by score."""
    candidates = defaultdict(list)
    for path in sorted((OUT / 'rounds').glob('*/cases/*/*/*.json')):
        value = read(path)
        if value.get('arm') in ARMS:
            candidates[value['arm'], value['id']].append((path, value))
    selected = {}
    for key, values in candidates.items():
        selected[key] = next(((p, v) for p, v in values if isinstance(v.get('prediction'), str)
                              and (v['status'].startswith('success') or v['status'] == 'algorithm_failure')),
                             values[-1])
    return selected


def summarize(values, planned, all_states=None):
    count = len(values)
    total = sum(value['f1'] for value in values)
    return {'planned': planned, 'scorable': count, 'missing': planned - count,
            'F1': total / count if count else None,
            'same_parent_F1': sum(v['parent_f1'] for v in values) / count if count else None,
            'delta_parent': sum(v['delta_parent'] for v in values) / count if count else None,
            'F1_all_planned_bounds': [total / planned, (total + (planned - count)) / planned] if planned else None,
            'up': sum(v['delta_parent'] > 1e-12 for v in values),
            'tie': sum(abs(v['delta_parent']) <= 1e-12 for v in values),
            'down': sum(v['delta_parent'] < -1e-12 for v in values),
            'states': dict(all_states or {})}


def costs():
    """Use namespace boundaries; do not relabel dev90 costs as old300."""
    with sqlite3.connect((FAMILY / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        def get(prefix=None):
            query = ('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL '
                     'THEN reserved ELSE 0 END),0),count(*) FROM charges')
            args = ()
            if prefix is not None:
                query += ' WHERE namespace LIKE ?'
                args = (str(prefix.relative_to(FAMILY)) + '%',)
            known, reserved, count = db.execute(query, args).fetchone()
            return {'known_cny': known, 'unreconciled_reserved_cny': reserved,
                    'occupied_cny': known + reserved, 'provider_submissions': count}
        own, fresh, global_value = get(OUT), get(FRESH), get()
        by_namespace = [{'namespace': n, 'known_cny': a, 'reserved_cny': r, 'submissions': c}
                        for n, a, r, c in db.execute(
                            'SELECT namespace,coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL '
                            'THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ? '
                            'GROUP BY namespace', (str(OUT.relative_to(FAMILY)) + '%',))]
    by_arm = {arm: {
        'known_cny': sum(v['known_cny'] for v in by_namespace if arm in v['namespace']),
        'unreconciled_reserved_cny': sum(v['reserved_cny'] for v in by_namespace if arm in v['namespace']),
        'provider_submissions': sum(v['submissions'] for v in by_namespace if arm in v['namespace'])}
        for arm in ARMS}
    return {'old300_round3': own, 'old300_by_arm': by_arm, 'round2_independent_dev90': fresh,
            'global': global_value, 'global_cap_cny': 300, 'own_namespaces': by_namespace,
            'interpretation': 'Incremental provider use, not cold end-to-end method cost; '
                              'unknown liabilities retained; actual bill not reconciled.'}


def independent_status():
    # File names and SQL status aggregates only; no answer/result payloads opened.
    written = Counter()
    for path in (FRESH / 'rounds').glob('*/cases/*/*/*.json'):
        written[path.parent.parent.name + '/' + path.parent.name] += 1
    return {'written_case_files_by_dataset_method': dict(written),
            'written_files_are_not_necessarily_unique_successes': True,
            'scores_answers_not_read': True, 'sealed150_not_read': True,
            'independence': 'Round2 frozen p02 dev90 validation remains separate; '
                            'it is not used to select or edit round3 candidates.'}


def percent(value):
    return 'NA' if value is None else f'{100 * value:.5f}%'


def build():
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    sources = {row['id']: row for row in rows(HERE / 'SOURCES.frozen.jsonl')}
    refs = {row['id']: row for row in rows(REFERENCE)}
    assert len(runtime) == len(sources) == 300
    assert Counter(row['dataset'] for row in runtime) == {ds: 100 for ds in DATASETS}
    scorers = {ds: metric(ds) for ds in DATASETS}  # Cache expensive native alias loaders.
    parent, base, extras, parent_scores = {}, {}, {}, {}
    for case in runtime:
        source = sources[case['id']]
        parent[case['id']] = read(ROOT / source['control'])
        base[case['id']] = read(ROOT / source['base'])
        extras[case['id']] = additional_document(parent[case['id']], base[case['id']])
        parent_scores[case['id']] = scorers[case['dataset']](parent[case['id']]['prediction'], refs[case['id']])['f1']
    selected = chosen_cases()
    manual = {(v['arm'], v['id']): v for v in read(REVIEW_PATH)} if REVIEW_PATH.exists() else {}
    independent_audit = OUT / 'ANALYSIS.e03_information_cases.json'
    if independent_audit.exists():
        for case_review in read(independent_audit)['cases']:
            review = case_review['review']
            manual.setdefault(('e03_goal_evidence', case_review['id']), {
                'arm': 'e03_goal_evidence', 'id': case_review['id'],
                'classification': review['classification'],
                'note': '\n'.join(review.get(k, '') for k in
                                  ['parent_information', 'new_information', 'final_adoption', 'failure_note']),
                'added_document_new_information': {
                    'new_answer_information': review.get('answer_required_new_information_absent_from_parent'),
                    'new_source_support_for_known_QA': review.get('adds_raw_support_for_fact_already_in_QA'),
                    'certainty': review.get('certainty')},
                'scope': 'Independent non-blind selected tie/failure case audit; not a population rate.',
                'audit_source': str(independent_audit.relative_to(ROOT))})
    details, changed, added_cases, result = [], [], [], {}
    checks = Counter()
    for arm in ARMS:
        values, states = [], Counter()
        for case in runtime:
            cid = case['id']
            entry = selected.get((arm, cid))
            state = entry[1]['status'] if entry else 'unstarted'
            states[state] += 1
            if not entry or not isinstance(entry[1].get('prediction'), str):
                continue
            path, value = entry
            if not (state.startswith('success') or state == 'algorithm_failure'):
                continue
            score = scorers[case['dataset']](value['prediction'], refs[cid])
            addition = extras[cid]
            row = {'id': cid, 'dataset': case['dataset'], 'arm': arm, 'status': state,
                   'prediction': value['prediction'], 'parent_prediction': parent[cid]['prediction'],
                   'f1': score['f1'], 'em': score['em'], 'parent_f1': parent_scores[cid],
                   'delta_parent': score['f1'] - parent_scores[cid],
                   'has_additional_goal_document': addition is not None}
            values.append(row)
            details.append(row)
            if state.startswith('success'):
                checks[arm + '/nodes_identical_to_p02'] += value['nodes'] == parent[cid]['nodes']
                if arm == 'q03_question_last':
                    checks[arm + '/context_identical_no_new_evidence'] += value['final_context'] == parent[cid]['final_context']
                elif addition:
                    checks[arm + '/expected_extra_context'] += value['final_context'] == parent[cid]['final_context'] + ADDITIONAL_HEADER + addition['block']
                else:
                    checks[arm + '/no_extra_exact_parent_prompt'] += value['final_prompt'] == parent[cid]['final_prompt']
                    checks[arm + '/no_extra_exact_parent_answer'] += value['prediction'] == parent[cid]['prediction']
            record = {**row, 'question': case['question'], 'gold_offline_only': refs[cid]['answer'],
                      'case_source': str(path.relative_to(ROOT)), 'parent_source': sources[cid]['control'],
                      'base_source': sources[cid]['base'], 'unchanged_goal_answer': parent[cid]['nodes'][-1]['answer'],
                      'prior_QA': parent[cid]['nodes'][-1]['prior_complete_answers'],
                      'review': manual.get((arm, cid), {'classification': 'unreviewed',
                          'note': 'Output changes alone do not establish factual improvement or degradation.',
                          'added_document_new_information': 'unreviewed' if arm == 'e03_goal_evidence' and addition else 'none'})}
            if arm == 'e03_goal_evidence' and addition:
                record['additional_document'] = {k: addition[k] for k in ['doc_id', 'rank', 'source', 'text', 'block']}
                record['surface_checks_not_entailment'] = {
                    'candidate_answer_in_added_text': contains_surface(addition['text'], value['prediction']),
                    'gold_answer_in_added_text_offline_only': contains_surface(addition['text'], refs[cid]['answer']),
                    'candidate_answer_in_parent_context': contains_surface(parent[cid]['final_context'], value['prediction']),
                    'warning': 'Literal presence is neither factual support nor proof of new information; manual review required.'}
                # Preserve every added passage, including unchanged/wrong answers.
                # A missing manual review is never silently counted as success.
                added_cases.append(record)
            if value['prediction'] == parent[cid]['prediction'] and abs(row['delta_parent']) <= 1e-12:
                continue
            changed.append(record)
        result[arm] = summarize(values, 300, states)
        result[arm]['by_dataset'] = {ds: summarize([v for v in values if v['dataset'] == ds], 100)
                                           for ds in DATASETS}
        if arm == 'e03_goal_evidence':
            result[arm]['by_extra_document_availability'] = {
                name: summarize([v for v in values if v['has_additional_goal_document'] == has],
                                sum((extras[c['id']] is not None) == has for c in runtime))
                for name, has in [('additional_document', True), ('exact_reuse_no_new_document', False)]}
    own_cost = costs()
    snapshot = {'utc': datetime.now(timezone.utc).isoformat(), 'sample_role': 'Exposed old300 development; not fresh validation',
                'parent_p02_F1_full300': sum(parent_scores.values()) / 300, 'arms': result,
                'simple_context_checks': dict(checks), 'costs': own_cost, 'independent_validation_status_only': independent_status(),
                'new_API_or_retrieval_calls_from_report': 0,
                'interpretation': ['q03 moves the actual Original question block only, without adding evidence.',
                                   'e03 adds at most one previously retrieved source passage and retains all QA/rootTop10; its gains cannot be called formatting-only.',
                                   'Native F1 is unchanged. Numeric spelling, aliases, conjunctions and verbose answers can change score without changing facts.',
                                   'Manual case reviews are non-blind; literal answer presence is not entailment.'],
                'manual_reviews_present': len(manual), 'changed_records': len(changed),
                'unreviewed_changed_records': sum(v['review'].get('classification') == 'unreviewed' for v in changed),
                'all_added_document_records': len(added_cases),
                'added_document_manual_review_states': dict(Counter(
                    str(v['review'].get('added_document_new_information', 'unreviewed')) for v in added_cases)),
                'scored_changes': sum(abs(v['delta_parent']) > 1e-12 for v in changed)}
    write(OUT / 'REPORT_DATA.progress.json', snapshot)
    write(OUT / 'CHANGED_CASES.report_round3.json', changed)
    write(OUT / 'ADDED_DOCUMENT_CASES.report_round3.json', added_cases)
    (OUT / 'PAIRED_DETAILS.report_round3.jsonl').write_text(
        ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in details), encoding='utf-8')
    text = ['B0第三轮：冻结p02的两项独立最终阅读端改动', '',
            '仅汇总实际已写回结果，不调用API或检索；不修改冻结配置/标准答案/原生评分。',
            '旧300为开发选择样本。独立p02 fresh90答案/分数及sealed150均未读取，不能与本轮选型混用。',
            f'快照：{snapshot["utc"]}', f'完整300题p02父版F1：{percent(snapshot["parent_p02_F1_full300"])}', '',
            '版本\t可计分/计划\t同题F1\t同题p02\t差值百分点\t升/平/降\t缺失']
    for arm, stats in result.items():
        delta = 'NA' if stats['delta_parent'] is None else f'{100 * stats["delta_parent"]:+.5f}'
        text.append(f'{arm}\t{stats["scorable"]}/300\t{percent(stats["F1"])}\t{percent(stats["same_parent_F1"])}\t{delta}\t{stats["up"]}/{stats["tie"]}/{stats["down"]}\t{stats["missing"]}')
        text.append('状态：' + json.dumps(stats['states'], ensure_ascii=False))
        text.append('完整计划分母F1范围（缺失保留0到1）：' + json.dumps(stats['F1_all_planned_bounds']))
        for ds, ds_stats in stats['by_dataset'].items():
            text.append(f'  {ds}：{ds_stats["scorable"]}/100，同题F1={percent(ds_stats["F1"])}, 同题父版={percent(ds_stats["same_parent_F1"])}, 差值={percent(ds_stats["delta_parent"])}。')
        for group, group_stats in stats.get('by_extra_document_availability', {}).items():
            text.append(f'  {group}：{group_stats["scorable"]}/{group_stats["planned"]}，F1={percent(group_stats["F1"])}, 同题父版={percent(group_stats["same_parent_F1"])}, 差值={percent(group_stats["delta_parent"])}, 升平降={group_stats["up"]}/{group_stats["tie"]}/{group_stats["down"]}。')
    text += ['', '费用按namespace独立统计：']
    for label in ['old300_round3', 'round2_independent_dev90', 'global']:
        cost = own_cost[label]
        text.append(f'{label}：已知¥{cost["known_cny"]:.6f}，未知请求预留¥{cost["unreconciled_reserved_cny"]:.6f}，提交{cost["provider_submissions"]}次。')
    for arm, cost in own_cost['old300_by_arm'].items():
        text.append(f'  {arm}：已知¥{cost["known_cny"]:.6f}，预留¥{cost["unreconciled_reserved_cny"]:.6f}，提交{cost["provider_submissions"]}次。')
    text += ['这是新增调用估算，非完整方法冷启动成本；未核对实际账单，未知预留不释放。', '',
             '独立验证仅状态计数：' + json.dumps(snapshot['independent_validation_status_only'], ensure_ascii=False), '',
             f'新增原文逐例档案：ADDED_DOCUMENT_CASES.report_round3.json，当前{len(added_cases)}题，包含最终答案和F1不变的题；未人工核查项保持unreviewed。',
             '变化实例：记录F1升/降及分数持平的文本变化；未审阅项明确未审，不按词面重合自动判事实正确。']
    for i, value in enumerate(changed, 1):
        review = value['review']
        text += ['', f'{i}. {value["arm"]} | {value["id"]}', 'Query: ' + value['question'],
                 '标准答案（离线解释）：' + value['gold_offline_only'],
                 f'F1：{value["parent_f1"]:.6f} -> {value["f1"]:.6f}',
                 'p02：' + value['parent_prediction'], '候选：' + value['prediction'],
                 '未改动goal：' + value['unchanged_goal_answer'],
                 '审阅：' + review.get('classification', 'unreviewed'), '观察：' + review.get('note', ''),
                 '新增原文是否提供新信息：' + str(review.get('added_document_new_information', 'not applicable')),
                 '实际文件：' + value['case_source']]
        if 'additional_document' in value:
            doc = value['additional_document']
            text += [f'实际新增原文：goal rank={doc["rank"]}; source={doc["source"]}; doc_id={doc["doc_id"]}', doc['text'],
                     '词面检查（不等于支持）：' + json.dumps(value['surface_checks_not_entailment'], ensure_ascii=False)]
    changed_keys = {(v['arm'], v['id']) for v in changed}
    stable_reviewed = [v for v in added_cases if (v['arm'], v['id']) not in changed_keys
                       and v['review'].get('classification') != 'unreviewed']
    text += ['', '新增原文但最终输出不变的补充审阅：',
             '以下为目的性抽取的失败/部分分代表例，不估计214题总体的信息命中率。']
    for value in stable_reviewed:
        review, doc = value['review'], value['additional_document']
        text += ['', value['id'] + ' | ' + value['question'],
                 f'F1：{value["parent_f1"]:.6f} -> {value["f1"]:.6f}',
                 '最终：' + value['prediction'], '未改动goal：' + value['unchanged_goal_answer'],
                 '审阅：' + review['classification'], '观察：' + review.get('note', ''),
                 '新增信息判断：' + str(review.get('added_document_new_information')),
                 f'新增原文：rank={doc["rank"]}; source={doc["source"]}; doc_id={doc["doc_id"]}',
                 doc['text'], '实际文件：' + value['case_source']]
    (OUT / 'REPORT.progress.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'REPORT.progress.txt'),
                      'scorable': {a: v['scorable'] for a, v in result.items()},
                      'changed': len(changed), 'manual_reviews': len(manual), 'new_API_calls': 0}, ensure_ascii=False))
    return snapshot


if __name__ == '__main__':
    build()
