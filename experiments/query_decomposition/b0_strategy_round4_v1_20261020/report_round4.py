"""Offline old300 2x2 report: p02/e03/r04/s04. No inference or retrieval.

The independent round2 fresh/confirmation areas are inspected only through
directory-name counts and ledger namespace totals, never answers or scores.
Only replaceable report/analysis outputs are written by this module.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.b0_strategy_round3_v1_20261020.report_round3 import (
    additional_document, contains_surface, percent, read, rows, summarize)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = FAMILY / HERE.name
OLD3 = FAMILY / 'b0_strategy_round3_v1_20261020'
OLD2 = FAMILY / 'b0_strategy_round2_v1_20261020'
REFS = HERE.parent / 'b0_six_single_changes_v1_20261020/references300.local.frozen.jsonl'
DATASETS = ('hotpotqa', '2wiki', 'musique')
ARMS = ('r04_scope_contract', 's04_scope_plus_evidence')
CELLS = ('p02', 'e03', *ARMS)
MANUAL = OUT / 'CASE_REVIEWS.report_round4.json'
ATTEMPT_AUDITS = {}


def save(name, value):
    path = OUT / name
    assert not name.endswith('.frozen.json') and path.parent == OUT
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def selected(directory, allowed):
    found = defaultdict(list)
    attempts = []
    for path in sorted((directory / 'rounds').glob('*/cases/*/*/*.json')):
        value = read(path)
        if value.get('arm') in allowed:
            found[value['arm'], value['id']].append((path, value))
            attempts.append({k: value.get(k) for k in
                             ['arm', 'id', 'status', 'error_type', 'retryable_infrastructure',
                              'failed_request_id', 'request_status', 'http_status']})
    ATTEMPT_AUDITS[str(directory)] = attempts
    result = {}
    for key, values in found.items():
        result[key] = next(((p, v) for p, v in values if isinstance(v.get('prediction'), str)
                           and (v['status'].startswith('success') or v['status'] == 'algorithm_failure')),
                          values[-1])
    return result


def ready(value):
    return (value is not None and isinstance(value.get('prediction'), str)
            and (value['status'].startswith('success') or value['status'] == 'algorithm_failure'))


def finance():
    """All figures are current ledger estimates, not provider billing receipts."""
    with sqlite3.connect((FAMILY / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        def aggregate(condition='', args=()):
            sql = ('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL '
                   'THEN reserved ELSE 0 END),0),count(*) FROM charges')
            if condition:
                sql += ' WHERE ' + condition
            actual, reserved, count = db.execute(sql, args).fetchone()
            return {'known_cny': actual, 'unreconciled_reserved_cny': reserved,
                    'occupied_cny': actual + reserved, 'provider_submissions': count}
        p4, p3, p2 = (str(p.relative_to(FAMILY)) + '%' for p in (OUT, OLD3, OLD2))
        fresh = str((OLD2 / 'fresh_dev90').relative_to(FAMILY)) + '%'
        values = {'old_round3': aggregate('namespace LIKE ?', (p3,)),
                  'new_round4': aggregate('namespace LIKE ?', (p4,)),
                  'round2_fresh90': aggregate('namespace LIKE ?', (fresh,)),
                  'round2_confirmation': aggregate(
                      "namespace LIKE ? AND (lower(namespace) LIKE '%confirm%' OR lower(namespace) LIKE '%sealed%')",
                      (p2,)), 'global': aggregate(), 'cap_cny': 300}
        values['new_round4_by_arm'] = {arm: aggregate('namespace LIKE ? AND namespace LIKE ?',
                                                     (p4, '%' + arm + '%')) for arm in ARMS}
    values['interpretation'] = 'Incremental charges only; unknown liabilities retained; actual bill not reconciled.'
    return values


def independent_status_only():
    areas = [OLD2 / 'fresh_dev90']
    # Do not open any confirmation file. Names and counts are sufficient.
    areas += [path for path in OLD2.iterdir() if path.is_dir()
              and ('confirm' in path.name.casefold() or 'sealed' in path.name.casefold())]
    return {'case_file_counts_only': {p.name: dict(Counter(
        f.parent.parent.name + '/' + f.parent.name for f in (p / 'rounds').glob('*/cases/*/*/*.json')))
        for p in areas}, 'counts_are_written_files_not_unique_successes': True,
        'answers_scores_not_read': True, 'confirmation_contents_not_read': True,
        'selection_boundary': 'Independent frozen p02 validation is not used to select these candidates.'}


def mean(values):
    return sum(values) / len(values) if values else None


def build():
    required = ['PROTOCOL.frozen.json', 'runtime300.frozen.jsonl', 'SOURCES.frozen.jsonl']
    if not all((HERE / name).exists() for name in required):
        print(json.dumps({'waiting_for_frozen_inputs': True, 'new_API_calls': 0}))
        return
    plan = read(HERE / required[0])
    runtime = rows(HERE / required[1])
    manifest = {row['id']: row for row in rows(HERE / required[2])}
    refs = {row['id']: row for row in rows(REFS)}
    assert len(runtime) == 300 and Counter(c['dataset'] for c in runtime) == {ds: 100 for ds in DATASETS}
    score = {ds: metric(ds) for ds in DATASETS}
    e03 = selected(OLD3, {'e03_goal_evidence'})
    current = selected(OUT, set(ARMS))
    manual = {(v['arm'], v['id']): v for v in read(MANUAL)} if MANUAL.exists() else {}
    cells, paths, scores, bases, extras = {}, {}, {}, {}, {}
    for case in runtime:
        cid, ds = case['id'], case['dataset']
        source = manifest[cid]
        bases[cid] = read(ROOT / source['base'])
        parent = read(ROOT / source['control'])
        values = {'p02': parent, 'e03': e03['e03_goal_evidence', cid][1]}
        extras[cid] = additional_document(parent, bases[cid])
        paths['p02', cid] = source['control']
        paths['e03', cid] = str(e03['e03_goal_evidence', cid][0].relative_to(ROOT))
        for arm in ARMS:
            entry = current.get((arm, cid))
            values[arm] = entry[1] if entry else None
            if entry:
                paths[arm, cid] = str(entry[0].relative_to(ROOT))
        for arm, value in values.items():
            cells[arm, cid] = value
            if ready(value):
                scores[arm, cid] = score[ds](value['prediction'], refs[cid])
    metrics, all_details, changed, checks = {}, [], [], Counter()
    no_extra = [c['id'] for c in runtime if extras[c['id']] is None]
    for arm in CELLS:
        values, states = [], Counter()
        for case in runtime:
            cid, ds = case['id'], case['dataset']
            value = cells[arm, cid]
            states[value['status'] if value else 'unstarted'] += 1
            if (arm, cid) not in scores:
                continue
            s, parent_s = scores[arm, cid], scores['p02', cid]
            row = {'id': cid, 'dataset': ds, 'arm': arm, 'prediction': value['prediction'],
                   'f1': s['f1'], 'em': s['em'], 'parent_f1': parent_s['f1'],
                   'delta_parent': s['f1'] - parent_s['f1'],
                   'has_extra_document': extras[cid] is not None}
            values.append(row)
            all_details.append(row)
            if arm not in ARMS:
                continue
            parent, old_evidence = cells['p02', cid], cells['e03', cid]
            anchor = parent if arm == ARMS[0] else old_evidence
            if value['status'].startswith('success'):
                checks[arm + '/nodes_equal_p02'] += value['nodes'] == parent['nodes']
                checks[arm + '/context_equal_factor_control'] += value['final_context'] == anchor['final_context']
                old, new = (plan['scope_replacement'][k] for k in ('old', 'new'))
                checks[arm + '/only_scope_sentence_replaced'] += value['final_prompt'] == anchor['final_prompt'].replace(old, new, 1)
            other = cells['e03', cid] if arm == ARMS[1] else parent
            if (value['prediction'] == parent['prediction'] == other['prediction']
                    and abs(row['delta_parent']) <= 1e-12):
                continue
            record = {**row, 'question': case['question'], 'gold_offline_only': refs[cid]['answer'],
                      'p02_prediction': parent['prediction'], 'e03_prediction': old_evidence['prediction'],
                      'e03_f1': scores['e03', cid]['f1'],
                      'delta_vs_e03': s['f1'] - scores['e03', cid]['f1'],
                      'unchanged_goal_answer': parent['nodes'][-1]['answer'],
                      'prior_QA': parent['nodes'][-1]['prior_complete_answers'],
                      'case_source': paths[arm, cid], 'p02_source': paths['p02', cid],
                      'e03_source': paths['e03', cid], 'base_source': manifest[cid]['base'],
                      'review': manual.get((arm, cid), {'classification': 'unreviewed',
                          'note': 'F1 change is not an automatic factual correctness label.'})}
            if extras[cid] and arm == ARMS[1]:
                doc = extras[cid]
                record['additional_document'] = {k: doc[k] for k in ['rank', 'doc_id', 'source', 'text']}
                record['literal_checks_not_entailment'] = {
                    'answer_surface_in_added_text': contains_surface(doc['text'], value['prediction']),
                    'gold_surface_in_added_text_offline_only': contains_surface(doc['text'], refs[cid]['answer'])}
            changed.append(record)
        metrics[arm] = summarize(values, 300, states)
        metrics[arm]['by_dataset'] = {ds: summarize([v for v in values if v['dataset'] == ds], 100)
                                      for ds in DATASETS}
        metrics[arm]['by_extra_document'] = {
            label: summarize([v for v in values if v['has_extra_document'] == has],
                             sum((extras[c['id']] is not None) == has for c in runtime))
            for label, has in [('novel_document_214', True), ('no_novel_document_86', False)]}
    contrasts = {}
    for name, left, right in [('evidence_without_scope', 'e03', 'p02'),
                              ('scope_without_evidence', ARMS[0], 'p02'),
                              ('scope_with_evidence', ARMS[1], 'e03'),
                              ('evidence_with_scope', ARMS[1], ARMS[0])]:
        vv = [{'id': c['id'], 'dataset': c['dataset'], 'f1': scores[left, c['id']]['f1'],
               'parent_f1': scores[right, c['id']]['f1'],
               'delta_parent': scores[left, c['id']]['f1'] - scores[right, c['id']]['f1']}
              for c in runtime if (left, c['id']) in scores and (right, c['id']) in scores]
        contrasts[name] = {'left': left, 'right': right, **summarize(vv, 300),
                           'by_dataset': {ds: summarize([v for v in vv if v['dataset'] == ds], 100)
                                          for ds in DATASETS}}
    interaction_values = [{'id': c['id'], 'dataset': c['dataset'], 'interaction':
                           scores[ARMS[1], c['id']]['f1'] - scores[ARMS[0], c['id']]['f1']
                           - scores['e03', c['id']]['f1'] + scores['p02', c['id']]['f1']}
                          for c in runtime if all((a, c['id']) in scores for a in CELLS)]
    interactions = {'planned': 300, 'same_case_n': len(interaction_values),
                    'definition': '(s04-r04)-(e03-p02)',
                    'mean': mean([v['interaction'] for v in interaction_values]),
                    'positive': sum(v['interaction'] > 1e-12 for v in interaction_values),
                    'zero': sum(abs(v['interaction']) <= 1e-12 for v in interaction_values),
                    'negative': sum(v['interaction'] < -1e-12 for v in interaction_values),
                    'by_dataset': {ds: {'n': len(vv := [v for v in interaction_values if v['dataset'] == ds]),
                                        'mean': mean([v['interaction'] for v in vv])} for ds in DATASETS}}
    for cid in no_extra:
        r, s = cells[ARMS[0], cid], cells[ARMS[1], cid]
        if not (ready(r) and ready(s)):
            continue
        checks['no_extra_both_scorable'] += 1
        if r['status'].startswith('success') and s['status'].startswith('success'):
            checks['no_extra_s04_prompt_equals_r04'] += s['final_prompt'] == r['final_prompt']
            checks['no_extra_s04_prediction_equals_r04'] += s['prediction'] == r['prediction']
            rc, sc = r['logical_calls'][-1], s['logical_calls'][-1]
            checks['no_extra_s04_reused_receipt'] += bool(sc['reused'])
            checks['no_extra_same_physical_request'] += (rc['request_id'], rc['source']) == (sc['request_id'], sc['source'])
    money = finance()
    attempts = ATTEMPT_AUDITS.get(str(OUT), [])
    completed_both = all(metrics[arm]['scorable'] == 300 for arm in ARMS)
    representative_path = OUT / 'ANALYSIS.round4_scope_cases.json'
    representatives = read(representative_path)['cases'] if representative_path.exists() else []
    assert all(v['id'] in manifest for v in representatives)
    snapshot = {'utc': datetime.now(timezone.utc).isoformat(), 'sample_role': plan['sample_role'],
                'factorial_cells': {'scope_off/evidence_off': 'p02', 'scope_off/evidence_on': 'e03',
                                    'scope_on/evidence_off': ARMS[0], 'scope_on/evidence_on': ARMS[1]},
                'scope_replacement': plan['scope_replacement'], 'cells': metrics,
                'paired_contrasts': contrasts, 'paired_interaction': interactions,
                'simple_actual_case_checks': dict(checks), 'expected_no_extra_cases': len(no_extra),
                'expected_new_submissions_without_infrastructure_retries': 514,
                'observed_new_submissions': money['new_round4']['provider_submissions'],
                'extra_submissions_over_ideal_if_complete': (
                    money['new_round4']['provider_submissions'] - 514 if completed_both else None),
                'actual_attempt_status_counts': dict(Counter(v['status'] for v in attempts)),
                'non_success_attempts_preserved': [v for v in attempts if not str(v['status']).startswith('success')],
                'costs': money, 'independent_status_only': independent_status_only(),
                'changed_text_records': len(changed),
                'unreviewed_changed_records': sum(v['review']['classification'] == 'unreviewed' for v in changed),
                'independent_representative_cases': len(representatives),
                'representative_case_audit': str(representative_path.relative_to(ROOT)) if representatives else None,
                'new_API_or_retrieval_calls_from_report': 0,
                'round4_development_decision': (
                    'Retain frozen p02; neither r04 nor s04 exceeds its complete300 native mean F1.'
                    if completed_both and all(metrics[a]['F1'] <= metrics['p02']['F1'] for a in ARMS)
                    else 'Incomplete or candidate gain: refer to the frozen selection protocol; no independent claim here.'),
                'interpretation': ['S04 is the interaction of scope replacement and added source evidence, not one factor against p02.',
                                   'Only same-case paired means enter contrasts and the interaction; partial cohorts are not full300 rankings.',
                                   'Positive numeric interaction can reflect overlapping lexical losses or output compression; it does not by itself establish semantic synergy.',
                                   'All QA and upstream retrieval are retained. Scope correction, new facts and lexical F1 effects require separate non-blind review.',
                                   'No fresh/confirmation answer or score is read for selection. Exposed300 cannot establish independent generalization.']}
    save('REPORT_DATA.progress.json', snapshot)
    save('CHANGED_CASES.report_round4.json', changed)
    save('INTERACTION_CASES.report_round4.json', interaction_values)
    (OUT / 'PAIRED_DETAILS.report_round4.jsonl').write_text(
        ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in all_details), encoding='utf-8')
    text = ['B0第四轮：scope约定×已有goal新原文的2×2开发比较', '',
            '本脚本只读旧300实际轨迹/参考答案与独立验证状态计数、费用；0API/检索。',
            'p02与e03为冻结已完成对照。r04只替换一句scope约定；s04在该替换上加既有e03策略，明确为交互。',
            '保留全部QA及原Top10；不修改原生F1。旧300为开发样本，不是独立验证。',
            '快照：' + snapshot['utc'], '',
            'scope关闭：无新文=p02，有新文=e03；scope开启：无新文=r04，有新文=s04。',
            '版本\t可计分/300\t同题F1\t同题p02\t差值百分点\t升/平/降\t缺失']
    for arm, value in metrics.items():
        text.append(f'{arm}\t{value["scorable"]}/300\t{percent(value["F1"])}\t{percent(value["same_parent_F1"])}\t{percent(value["delta_parent"])}\t{value["up"]}/{value["tie"]}/{value["down"]}\t{value["missing"]}')
        text.append('状态：' + json.dumps(value['states'], ensure_ascii=False))
        for ds, v in value['by_dataset'].items():
            text.append(f'  {ds}：{v["scorable"]}/100，F1={percent(v["F1"])}，同题父={percent(v["same_parent_F1"])}，差值={percent(v["delta_parent"])}。')
        for label, v in value['by_extra_document'].items():
            text.append(f'  {label}：{v["scorable"]}/{v["planned"]}，F1={percent(v["F1"])}，差值={percent(v["delta_parent"])}。')
    text += ['', '同题因子比较：']
    for name, v in contrasts.items():
        text.append(f'{name} = {v["left"]}-{v["right"]}：n={v["scorable"]}/300，均差={percent(v["delta_parent"])}，升平降={v["up"]}/{v["tie"]}/{v["down"]}。')
        for ds, value in v['by_dataset'].items():
            text.append(f'  {ds}：n={value["scorable"]}/100，均差={percent(value["delta_parent"])}。')
    text += ['配对交互：' + json.dumps(interactions, ensure_ascii=False),
             '正交互也可能来自同一个数字/词面失分在组合中不再重复，或来自输出压缩；不能单独解释为事实推理协同。', '',
             '缓存核查：86道无novel-doc题，s04应与r04完全同prompt/答案并复用同一物理request；无基础设施重试时新提交应为300+214=514，不是600。',
             json.dumps(dict(checks), ensure_ascii=False), '', '费用按namespace分别列示：']
    text += ['实际尝试状态（含被成功恢复的旧失败，不改可计分分母）：' + json.dumps(snapshot['actual_attempt_status_counts'], ensure_ascii=False),
             '保留失败尝试：' + json.dumps(snapshot['non_success_attempts_preserved'], ensure_ascii=False),
             f'理想514次提交；当前实计{snapshot["observed_new_submissions"]}次；两臂完整后较理想额外提交={snapshot["extra_submissions_over_ideal_if_complete"]}。',
             '若最终515次且有1次基础设施恢复，应明确为514+1 retry，不能误称86题缓存失效；未知请求预留仍保留。']
    for label in ['old_round3', 'new_round4', 'round2_fresh90', 'round2_confirmation', 'global']:
        v = money[label]
        text.append(f'{label}：已知¥{v["known_cny"]:.6f}，未核销预留¥{v["unreconciled_reserved_cny"]:.6f}，提交{v["provider_submissions"]}次。')
    for arm, v in money['new_round4_by_arm'].items():
        text.append(f'  {arm}：已知¥{v["known_cny"]:.6f}，预留¥{v["unreconciled_reserved_cny"]:.6f}，提交{v["provider_submissions"]}次。')
    text += ['新增调用估算非完整方法冷启动成本；未核对实际账单，不释放未知请求预留。', '',
             '本轮开发选择：' + snapshot['round4_development_decision'], '',
             '独立验证仅状态：' + json.dumps(snapshot['independent_status_only'], ensure_ascii=False), '',
             '实际变化：候选已固定后再审，不按个别AfterHours等例子更改候选；词面命中不等于关系成立。']
    for i, v in enumerate(changed, 1):
        text += ['', f'{i}. {v["arm"]} | {v["id"]}', 'Query: ' + v['question'],
                 '标准（只用于离线分析）：' + v['gold_offline_only'],
                 'p02：' + v['p02_prediction'], 'e03：' + v['e03_prediction'],
                 '候选：' + v['prediction'],
                 f'F1={v["f1"]:.6f}，相对p02={v["delta_parent"]:+.6f}，相对e03={v["delta_vs_e03"]:+.6f}',
                 '未改goal：' + v['unchanged_goal_answer'],
                 '审阅：' + v['review']['classification'], '观察：' + v['review'].get('note', ''),
                 '文件：' + v['case_source']]
        if 'additional_document' in v:
            d = v['additional_document']
            text += [f'实际新增原文：rank={d["rank"]}; source={d["source"]}; doc_id={d["doc_id"]}', d['text']]
    text += ['', '未修复/新失败的目的性代表例：',
             '以下只解释固定候选的实际输出，不据这些例子改规则，不外推总体失败比例。']
    for case in representatives:
        text += ['', case['id'] + ' | ' + case['question']]
        for cell in CELLS:
            value = case['cells'].get(cell)
            text.append(cell + ': ' + (value['prediction'] if value else '尚无实际结果，不推断'))
        review = case['review']
        text += ['审阅：' + review.get('classification', 'unreviewed')]
        text += [str(review[k]) for k in ['scope_observation', 'evidence_observation',
                                         'not_repaired_or_new_failure', 'certainty'] if k in review]
        if 'interaction_observation' in review:
            text.append(str(review['interaction_observation']))
    (OUT / 'REPORT.progress.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'REPORT.progress.txt'),
                      'n': {k: v['scorable'] for k, v in metrics.items()},
                      'changed': len(changed), 'unreviewed': snapshot['unreviewed_changed_records'],
                      'new_API_calls': 0}, ensure_ascii=False))
    return snapshot


if __name__ == '__main__':
    build()
