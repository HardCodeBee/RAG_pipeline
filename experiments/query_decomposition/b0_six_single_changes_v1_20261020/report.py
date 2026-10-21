"""Native, paired downstream metrics; gold references stay in this scorer."""
from collections import Counter
import json
import sqlite3
import numpy as np
from .prepare import HERE, OUT, OUTPUT, ROOT, rows, read
from .pipeline import ARMS
from .receipts import sources, baseline_case
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric

_BASE = None


def selected(arm, case):
    if arm in list(ARMS)[3:]:
        if not (OUT / 'OFFLINE_VALIDATION.frozen.json').exists(): return None
        return {**_BASE[case['id']], 'arm': arm, 'offline_inactive': True}
    found = None
    for root in sorted((OUT / 'rounds').glob('r[0-9][0-9][0-9]')):
        path = root / 'cases' / case['dataset'] / arm / (case['qid'] + '.summary.json')
        if not path.exists(): continue
        found = read(path)
        if found['status'].startswith('success') or found['status'] == 'algorithm_failure': return found
    return found


def report():
    global _BASE
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {r['id']: r for r in rows(HERE / 'references300.local.frozen.jsonl')}
    scorers = {ds: metric(ds) for ds in ['hotpotqa', '2wiki', 'musique']}
    if _BASE is None:
        _BASE = {}
        for key, source in sources().items():
            b = baseline_case(source)
            _BASE[key] = {**{k: b[k] for k in ['id','qid','dataset','question','prediction','status']},
                          **scorers[b['dataset']](b['prediction'], refs[key]), 'source': source['source']}
    base = {'n': 300, 'F1': sum(r['f1'] for r in _BASE.values()) / 300,
            'by_dataset': {ds: {'n': 100, 'F1': sum(r['f1'] for r in _BASE.values() if r['dataset'] == ds) / 100}
                           for ds in scorers}}
    details, arms = [], {}
    for arm in ARMS:
        values, statuses = [], Counter()
        for c in cases:
            r = selected(arm, c)
            statuses[r['status'] if r else 'unstarted'] += 1
            if r is None or not isinstance(r.get('prediction'), str): continue
            if not r['status'].startswith('success') and r['status'] != 'algorithm_failure': continue
            score = scorers[c['dataset']](r['prediction'], refs[c['id']])
            b = _BASE[c['id']]
            item = {'id': c['id'], 'dataset': c['dataset'], 'question': c['question'], 'arm': arm,
                    'prediction': r['prediction'], 'B0_prediction': b['prediction'], 'gold': refs[c['id']]['answer'],
                    'f1': score['f1'], 'B0_f1': b['f1'], 'delta': score['f1'] - b['f1'],
                    'source': r.get('source'), 'offline_inactive': r.get('offline_inactive', False)}
            details.append(item); values.append(item)
        n = len(values)
        if n:
            delta = np.array([v['delta'] for v in values])
            rng = np.random.default_rng(20261020); boots = np.zeros(5000)
            for ds in scorers:
                d = delta[np.array([v['dataset'] == ds for v in values])]
                if len(d): boots += d[rng.integers(len(d), size=(5000,len(d)))].sum(1) / n
            ci = np.quantile(boots, [.025,.975]).tolist()
        else: ci = None
        arms[arm] = {'planned': 300, 'scorable': n, 'statuses': dict(statuses),
                     'F1': sum(v['f1'] for v in values)/n if n else None,
                     'B0_same_queries_F1': sum(v['B0_f1'] for v in values)/n if n else None,
                     'delta_F1': sum(v['delta'] for v in values)/n if n else None, 'paired_stratified_bootstrap95': ci,
                     'up': sum(v['delta'] > 1e-12 for v in values), 'down': sum(v['delta'] < -1e-12 for v in values),
                     'tie': sum(abs(v['delta']) <= 1e-12 for v in values),
                     'by_dataset': {ds: {'scorable': len(vv := [v for v in values if v['dataset'] == ds]),
                                          'F1': sum(v['f1'] for v in vv)/len(vv) if vv else None,
                                          'B0_F1': sum(v['B0_f1'] for v in vv)/len(vv) if vv else None,
                                          'delta_F1': sum(v['delta'] for v in vv)/len(vv) if vv else None} for ds in scorers}}
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        totals = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges').fetchone()
        local = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT))+'%',)).fetchone()
    costs = {}
    for arm in list(ARMS)[:3]:
        with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
            cost = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                              (str(OUT.relative_to(OUTPUT)) + '%/methods/%/' + arm,)).fetchone()
        costs[arm] = {'new_returned_usage_cny': cost[0], 'new_reserved_cny': cost[1], 'new_submissions': cost[2]}
    value = {'utc': utc(), 'baseline': base, 'arms': arms, 'budget': {'global_returned_cny': totals[0], 'global_reserve_cny': totals[1],
             'experiment_returned_cny': local[0], 'experiment_reserve_cny': local[1], 'experiment_submissions': local[2], 'by_arm': costs},
             'all_complete': all(a['scorable'] == 300 for a in arms.values()),
             'scope': 'Frozen300 diagnostic cohort. Prompt-only interventions are independent; D/E/F are exact full-trajectory offline replays, not API resampling. No new remote/training jobs.',
             'metric': 'Native Answer F1; 2Wiki aliases/demonyms, MuSiQue aliases. Incomplete provider calls excluded and counted. Algorithm failures count as zero.'}
    locked_json(OUT / 'RESULTS_LATEST.json', value)
    (OUT / 'PAIRED_DETAILS_LATEST.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in details),encoding='utf-8')
    text = ['# B0 六项独立改动', '', '同一批300题，每数据集100题；本批为已有开发诊断样本。每组仅修改一项。', '',
            '| 改动 | 可计分/300 | 同题 B0 F1 | 本组 F1 | 差值（百分点） | 升/平/降 |', '|---|---:|---:|---:|---:|---|']
    for arm,a in arms.items():
        if a['scorable']:
            text.append(f"| {arm} | {a['scorable']} | {100*a['B0_same_queries_F1']:.2f}% | {100*a['F1']:.2f}% | {100*a['delta_F1']:+.2f} | {a['up']}/{a['tie']}/{a['down']} |")
        else: text.append(f'| {arm} | 0 | — | — | — | — |')
    text += ['', 'D/E/F 在原始 B0 轨迹均未触发，已验证全部检索、文档顺序、API载荷和最终输入逐项一致，复用原回执，不重复付费。该结果不说明这些机制在触发时没有价值。', '',
             f"本次新增已返回用量估算 ¥{local[0]:.4f}；未核销预留 ¥{local[1]:.4f}；新增提交 {local[2]} 次。累计占用 ¥{totals[0]+totals[1]:.4f}/300。实际账单未核对。", '',
             '逐题配对、置信区间、按数据集结果见 RESULTS_LATEST.json 与 PAIRED_DETAILS_LATEST.jsonl。']
    (OUT / 'REPORT_LATEST.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return value


if __name__ == '__main__':
    print(json.dumps(report(),ensure_ascii=False))
