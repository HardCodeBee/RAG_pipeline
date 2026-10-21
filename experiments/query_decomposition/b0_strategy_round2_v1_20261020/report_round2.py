"""Read-only progress/report reader; no prepare, generation, retrieval or edits to freezes.

Reads current summary and literal selected case traces, keeps shared denominators,
and explicitly separates lexical observations from reviewed factual conclusions.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name


def read(path, optional=False):
    for attempt in range(3):
        try: return json.loads(path.read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            if optional: return None
            raise
        except ValueError:
            if attempt == 2: raise
            time.sleep(.1)


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def result_snapshot():
    path = OUT / 'RESULTS_LATEST.json'
    if not path.exists(): return None, []
    for attempt in range(6):
        try:
            before = path.read_bytes()
            summary = json.loads(before)
            details = rows(OUT / 'PAIRED_DETAILS.jsonl')
            assert before == path.read_bytes()
            count = Counter(r['arm'] for r in details)
            assert all(count[a] == r['scorable'] for a, r in summary['arms'].items())
            return summary, details
        except (FileNotFoundError, ValueError, AssertionError):
            if attempt == 5: raise RuntimeError('Live summary/details were changing; rerun read-only report after current write')
            time.sleep(.2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def selected_case(arm, cid):
    ds, qid = cid.split(':', 1)
    last = None
    for p in sorted((OUT / 'rounds').glob(f'*/cases/{ds}/{arm}/{qid}.json')):
        obj = read(p)
        last = (p, obj)
        if isinstance(obj.get('prediction'), str) and (obj['status'].startswith('success') or obj['status'] == 'algorithm_failure'):
            return p, obj
    return last


def normalize_words(text):
    # Only a report heuristic. Never used to replace official/native scoring.
    return ' '.join(re.findall(r'\w+', text.casefold()))


def lexical_number(text):
    words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
             'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
             'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20}
    x = normalize_words(text)
    return str(words[x]) if x in words else x


def provisional_kind(parent, candidate):
    if parent == candidate: return '最终答案未变；仅观察goal变化，不推断事实收益'
    if normalize_words(parent) == normalize_words(candidate): return '仅大小写/标点/空白变化（文字观察）'
    if lexical_number(parent) == lexical_number(candidate): return '数字拼写形式变化（可影响词面F1；不改动原计分）'
    if normalize_words(parent) in ['yes', 'no'] and normalize_words(candidate) in ['yes', 'no']:
        return '整体判断反转；需依据原文核查，不能只凭F1认定事实改善'
    return '答案内容或范围变化；待事实核查，不能将F1变化直接等同事实改善'


def budget():
    with sqlite3.connect((OUT.parent / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        a, b = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        grouped = db.execute('SELECT namespace,coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ? GROUP BY namespace', (HERE.name + '%',)).fetchall()
    scope = {k: {'known_cny': 0., 'reserved_cny': 0., 'submissions': 0} for k in ['old300_candidates', 'fresh_dev90', 'other_family']}
    for ns, known, reserved, count in grouped:
        normalized = ns.replace('\\', '/')
        if not normalized.startswith(HERE.name + '/'): continue
        suffix = normalized[len(HERE.name) + 1:]
        key = 'old300_candidates' if suffix.startswith('rounds/') else ('fresh_dev90' if suffix.startswith('fresh_dev90/') else 'other_family')
        scope[key]['known_cny'] += known
        scope[key]['reserved_cny'] += reserved
        scope[key]['submissions'] += count
    c = sum(v['known_cny'] for v in scope.values())
    d = sum(v['reserved_cny'] for v in scope.values())
    n = sum(v['submissions'] for v in scope.values())
    return {'known_global_cny': a, 'reserved_global_cny': b, 'occupied_global_cny': a + b,
            'family_known_cny': c, 'family_reserved_cny': d, 'family_submissions': n,
            'by_scope': scope, 'cap_cny': 300}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    plan = read(HERE / 'PROTOCOL.frozen.json')
    summary, details = result_snapshot()
    selection = read(OUT / 'SELECTION.frozen.json', optional=True)
    progress = read(OUT / 'PROGRESS.json', optional=True)
    audit_path = HERE / ('OFFLINE_AUDIT.v002.json' if (HERE / 'OFFLINE_AUDIT.v002.json').exists() else 'OFFLINE_AUDIT.json')
    audit = read(audit_path, optional=True)
    sources = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    cases = {r['id']: r for r in rows(HERE / 'runtime300.frozen.jsonl')}
    # Read scoring-only references only for human posthoc interpretation, never generation.
    reference_path = HERE.parent / 'b0_six_single_changes_v1_20261020/references300.local.frozen.jsonl'
    refs = {r['id']: r for r in rows(reference_path)}
    pilot = {r['id'] for r in rows(HERE / 'pilot100.ids.frozen.jsonl')}
    manual_path = OUT / 'CASE_REVIEWS.progress.json'
    manual = read(manual_path, optional=True) or []
    manual = {(r['arm'], r['id']): r for r in manual}
    changed, review_counts = [], Counter()
    for detail in details:
        arm, cid = detail['arm'], detail['id']
        path, obj = selected_case(arm, cid)
        src = sources[cid]
        parent = read(ROOT / src['control'])
        assert obj['prediction'] == detail['prediction']
        old_goal = parent['nodes'][-1]['answer']
        new_goal = obj['nodes'][-1]['answer'] if obj.get('nodes') else None
        if obj['prediction'] == parent['prediction'] and old_goal == new_goal and abs(detail['delta_parent']) < 1e-12: continue
        item = {**detail, 'query': cases[cid]['question'], 'parent_final': parent['prediction'],
                'candidate_final': obj['prediction'], 'parent_goal': old_goal, 'candidate_goal': new_goal,
                'reference_answer': refs[cid]['answer'], 'candidate_source': str(path.relative_to(ROOT)),
                'parent_source': src['control'], 'provisional_interpretation': provisional_kind(parent['prediction'], obj['prediction'])}
        review = manual.get((arm, cid))
        if review:
            assert review['parent_final'] == item['parent_final'] and review['candidate_final'] == item['candidate_final'], 'Stale manual review'
            item['review'] = review
            review_counts[review['classification']] += 1
        else: review_counts['unreviewed'] += 1
        changed.append(item)
    source_checks = {p: sha(ROOT / p) == h for p, h in plan['source_pins'].items()}
    data_checks = {str(HERE / 'runtime300.frozen.jsonl'): sha(HERE / 'runtime300.frozen.jsonl') == plan['runtime_sha256'],
                   str(HERE / 'SOURCES.frozen.jsonl'): sha(HERE / 'SOURCES.frozen.jsonl') == plan['source_manifest_sha256'],
                   str(HERE / 'pilot100.ids.frozen.jsonl'): sha(HERE / 'pilot100.ids.frozen.jsonl') == plan['pilot_sha256']}
    lineage = {}
    for src in sources.values():
        for p, h in [(src['base'], src['base_sha256']), (src['control'], src['control_sha256'])] + [(r['path'], r['sha256']) for r in src['cache_sources']]:
            lineage[p] = h
    lineage_failures = [p for p, h in lineage.items() if sha(ROOT / p) != h]
    costs = budget()
    report = ['B0第二轮通用策略优化：进度、同题比较及逐条观察', '',
              '本报告只读已有结果，不提交API或检索，不修改任何冻结文件。',
              '本轮是已使用300题上的开发筛选；pilot100是其中的固定子集，不是新留出集。',
              '所有候选以冻结g01（B0+P1+goal一致性）为父对照；保留原B0/S02。',
              'g02：只替换goal约定；p02：只增加中间结果/goal候选角色标题（无普通节点时只加goal标题）；s03：g01上叠加既有f01最终检查。', '']
    if progress: report += ['最近调度进度：' + json.dumps(progress, ensure_ascii=False), '']
    if not summary:
        report += ['尚无完整先导阶段汇总。当前完成数量不作完整100或300题排名。']
    else:
        report += [f"父版完整300题F1：{summary['parent300_F1']*100:.5f}%；原B0：{summary['B0_300_F1']*100:.5f}%。", '',
                   '版本\t可计分题数\t同题F1\t同题g01父F1\t差值百分点\t升/平/降\t先导完成/100\t扩展判定']
        for arm in plan['arms']:
            r = summary['arms'][arm]
            def pct(x): return '-' if x is None else f'{100*x:.5f}'
            report.append(f"{arm}\t{r['scorable']}\t{pct(r['F1'])}\t{pct(r['same_parent_F1'])}\t{pct(r['delta_parent'])}\t{r['up']}/{r['tie']}/{r['down']}\t{r['pilot_scorable']}/100\t{r['expand']}")
            report.append('  状态：' + json.dumps(r['statuses'], ensure_ascii=False))
            for ds, d in r['by_dataset'].items():
                report.append(f"  {ds}：n={d['n']}，F1={pct(d['F1'])}%，相对同题父版={pct(d['delta_parent'])}个百分点。")
        report += ['', '完整300题、先导100题及进行中的部分样本分母不同，不能跨行仅按均分排列。只有完整300题候选可参加本轮开发选择。']
    if selection: report += ['', '冻结开发选择：' + json.dumps(selection, ensure_ascii=False)]
    report += ['', '费用（按namespace分开，避免混入新题父版）：']
    for label, key in [('旧300候选', 'old300_candidates'), ('新dev90各组', 'fresh_dev90'), ('同family其他支出', 'other_family')]:
        scope = costs['by_scope'][key]
        report.append(f"{label}：新增已知¥{scope['known_cny']:.6f}，未核销预留¥{scope['reserved_cny']:.6f}，提交{scope['submissions']}次。")
    report += [f"以上family合计：已知¥{costs['family_known_cny']:.6f}，预留¥{costs['family_reserved_cny']:.6f}，提交{costs['family_submissions']}次。",
               f"累计占用¥{costs['occupied_global_cny']:.6f}/300（已知¥{costs['known_global_cny']:.6f}+预留¥{costs['reserved_global_cny']:.6f}）。",
               '这是本轮增量调用费用，不是重算整个RAG方法的独立运行费用；共享总账可能同时包含其他已授权任务。', '',
               '冻结FINAL_RESULTS中的budget快照保留原样；其new_*按family统计，可能已包含当时fresh_dev90在途，不能直接称为仅旧300候选开销。', '',
               '源及回放审计：', f"代码pins全部一致：{all(source_checks.values())}；运行输入/来源表/先导ID哈希全部一致：{all(data_checks.values())}；{len(lineage)}个冻结轨迹源哈希错误数：{len(lineage_failures)}。"]
    if audit:
        report.append('此前只读重建审计：' + json.dumps(audit['counts'], ensure_ascii=False))
        audit_matches = all(sha(Path(p)) == h for p, h in audit['source_sha256'].items())
        report.append(f'使用审计版本：{audit_path.name}；其执行源码快照与当前冻结代码一致：{audit_matches}。')
        if audit_matches:
            report.append('执行前补充依赖pins的旧快照v001另行完整保留；v002已在当前冻结执行码上重建审计通过，不属于实验执行后的源码漂移。')
        else:
            report.append('此旧审计快照早于执行前补充依赖pins；本报告另按当前PROTOCOL检查所有pins，不将旧快照哈希视为执行漂移。')
    f1_changes = [r for r in changed if abs(r['delta_parent']) > 1e-12]
    reviewed_f1_changes = sum(bool(r.get('review')) for r in f1_changes)
    report += ['', '变化观察：', '记录所有最终答案或goal文本变化，包括F1持平；未审阅项保持未确定，不自动据F1判定事实正确。',
               '仅文字/数字拼写变化与受原文支持的事实修复分开统计；原始标准答案/F1不作任何修改。',
               f'F1变化案例已人工记录{reviewed_f1_changes}/{len(f1_changes)}；其余文本变化仍保留，未完成事实核查则明确未确定。',
               '审阅状态：' + json.dumps(dict(review_counts), ensure_ascii=False), '']
    for i, item in enumerate(sorted(changed, key=lambda r: (r['arm'], -abs(r['delta_parent']), r['id'])), 1):
        report += [f"{i}. {item['arm']} | {item['id']} | pilot={item['pilot']}", 'Query: ' + item['query'],
                   f"F1：{item['parent_f1']:.6f} -> {item['f1']:.6f}；差值{item['delta_parent']*100:+.6f}个百分点。",
                   '标准答案：' + str(item['reference_answer']), '父版最终：' + item['parent_final'], '候选最终：' + item['candidate_final'],
                   '父版goal：' + str(item['parent_goal']), '候选goal：' + str(item['candidate_goal'])]
        if item.get('review'):
            report += ['事实/文字审阅分类：' + item['review']['classification'], '观察：' + item['review']['note']]
            for fact in item['review'].get('evidence', []): report.append('原文依据：' + fact)
        else: report.append('暂定观察：' + item['provisional_interpretation'])
        report += ['候选记录：' + item['candidate_source'], '父记录：' + item['parent_source'], '']
    report += ['后续边界：', '新dev90与sealed150均已单独按元数据封存。当前报告不读取它们的query、分解或结果。',
               '候选需先冻结，再执行新dev验证；封存确认只用于冻结方案的一次性评价，不用其结果继续调规则。']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORT.progress.txt').write_text('\n'.join(report) + '\n', encoding='utf-8', newline='\n')
    (OUT / 'CHANGED_CASES.progress.json').write_text(json.dumps(changed, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'REPORT_AUDIT.progress.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'source_pin_checks': source_checks, 'input_checks': data_checks, 'lineage_sources': len(lineage),
        'lineage_hash_failures': lineage_failures, 'budget': costs, 'changed_cases': len(changed),
        'review_counts': dict(review_counts), 'F1_change_cases': len(f1_changes), 'reviewed_F1_change_cases': reviewed_f1_changes,
        'report_script_sha256': sha(Path(__file__)),
        'manual_reviews_sha256': sha(manual_path) if manual_path.exists() else None,
        'local_reference_sha256': sha(reference_path), 'summary_snapshot_utc': summary.get('utc') if summary else None,
        'offline_audit_file': str(audit_path.relative_to(ROOT)), 'offline_audit_sha256': sha(audit_path) if audit_path.exists() else None,
        'paired_detail_rows': len(details), 'new_API_calls_by_report': 0, 'new_retrieval_calls_by_report': 0}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'REPORT.progress.txt'), 'summary_available': bool(summary),
                      'changed_cases': len(changed), 'review_counts': dict(review_counts),
                      'all_source_hashes_match': all(source_checks.values()) and all(data_checks.values()) and not lineage_failures,
                      'new_API_calls_by_report': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
