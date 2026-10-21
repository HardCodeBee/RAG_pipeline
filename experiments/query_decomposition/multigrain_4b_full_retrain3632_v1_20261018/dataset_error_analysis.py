"""Dataset-level comparisons using immutable, already locked decomposition verdicts.

No model calls, regrading, checkpoint selection or writes to frozen reviews.
"""
import collections
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/query_decomposition/multigrain_4b_full_retrain3632_v1_20261018'
OLD = ROOT / 'outputs/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018'
DATASETS = ['hotpotqa', '2wiki', 'musique']
PATHS = {
    'original165': OLD / 'reviews/confirm120/original165.jsonl',
    'T02': OLD / 'reviews/confirm120/p00_parent_t02.jsonl',
    'R01_old40': OLD / 'reviews/confirm120/r01_old_data.jsonl',
    'R02_expanded40': OLD / 'reviews/confirm120/r02_expanded.jsonl',
    'full3632_step228': OUT / 'reviews/confirm120/selected.jsonl',
}


def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def summarize(rows):
    verdicts = collections.Counter(r['verdict'] for r in rows)
    dimensions = {d: [r['index'] for r in rows if d in r['failed_dimensions']]
                  for d in ['F', 'S', 'C', 'U']}
    return {'n': len(rows), 'passes': verdicts['pass'],
            'pass_percent': 100 * verdicts['pass'] / len(rows),
            'verdicts': dict(verdicts), 'nonpass_dimensions': dimensions,
            'FS_risk_indices': [r['index'] for r in rows
                                if {'F', 'S'} & set(r['failed_dimensions'])]}


def paired(base, candidate):
    assert [r['id'] for r in base] == [r['id'] for r in candidate]
    diffs = [int(c['verdict'] == 'pass') - int(b['verdict'] == 'pass')
             for b, c in zip(base, candidate)]
    wins = [b['index'] for b, d in zip(base, diffs) if d == 1]
    losses = [b['index'] for b, d in zip(base, diffs) if d == -1]
    groups = [[d for b, d in zip(base, diffs) if b['dataset'] == ds]
              for ds in DATASETS]
    rng = random.Random(2718)
    boot = sorted(100 * sum(sum(rng.choices(g, k=len(g))) for g in groups if g)
                  / len(base) for _ in range(30000))
    discordant = len(wins) + len(losses)
    p = min(1., 2 * sum(math.comb(discordant, k)
                       for k in range(min(len(wins), len(losses)) + 1)) / 2 ** discordant)
    return {'fixes': wins, 'regressions': losses, 'net_passes': sum(diffs),
            'delta_pp': 100 * sum(diffs) / len(base),
            'paired_bootstrap95_delta_pp': [boot[749], boot[29249]],
            'mcnemar_exact_two_sided_p': p}


def main():
    reviews = {name: load(path) for name, path in PATHS.items()}
    anchor = reviews['T02']
    for name, rows in reviews.items():
        assert len(rows) == 120 and len({r['id'] for r in rows}) == 120
        assert [(r['index'], r['id'], r['dataset'], r['query']) for r in rows] == [
            (r['index'], r['id'], r['dataset'], r['query']) for r in anchor], name
        assert all(r['verdict'] in {'pass', 'fail', 'uncertain'} for r in rows)
        assert all(r['verdict'] != 'pass' or not r['failed_dimensions'] for r in rows)
        for ds in DATASETS:
            assert sum(r['dataset'] == ds for r in rows) == 40
    summaries = {name: {ds: summarize([r for r in rows if r['dataset'] == ds])
                        for ds in DATASETS} for name, rows in reviews.items()}
    comparisons = {}
    for base, candidate in [('original165', 'T02'), ('T02', 'R02_expanded40'),
                            ('R01_old40', 'R02_expanded40'),
                            ('T02', 'full3632_step228'),
                            ('R02_expanded40', 'full3632_step228')]:
        comparisons[f'{candidate}_vs_{base}'] = {
            ds: paired([r for r in reviews[base] if r['dataset'] == ds],
                       [r for r in reviews[candidate] if r['dataset'] == ds])
            for ds in DATASETS}
    selected = [6, 24, 27, 28, 38, 41, 45, 51, 65, 113, 119]
    cases = [{
        'index': idx, 'id': anchor[idx - 1]['id'],
        'dataset': anchor[idx - 1]['dataset'], 'query': anchor[idx - 1]['query'],
        'versions': {name: {key: rows[idx - 1][key]
                           for key in ['raw_output', 'verdict', 'failed_dimensions', 'reason']}
                     for name, rows in reviews.items()}
    } for idx in selected]
    assert all(anchor[i - 1]['index'] == i for i in selected)
    historical_path = OLD / 'HISTORICAL16_REVIEW.json'
    historical = json.loads(historical_path.read_text(encoding='utf-8'))
    result = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'Offline arithmetic on previously locked reviews; no inference or regrading.',
        'sources': {name: {'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in PATHS.items()},
        'summaries': summaries, 'comparisons': comparisons,
        'selected_real_cases_not_random': cases,
        'full_remaining_nonpass': [r for r in reviews['full3632_step228']
                                  if r['verdict'] != 'pass'],
        'historical16_nonrandom_diagnostic': {
            k: historical[k] for k in ['n', 'parent_passes', 'candidate_passes',
                                      'fixes', 'regressions', 'scope']},
        'limits': [
            'Decomposition F/S/C/U pass, not downstream RAG F1. Uncertain counts as nonpass.',
            'Same Codex authored labels and reviewed outputs; not independent human gold.',
            'Mu evaluation is custom native-train held-out with training-primitive exclusion, not official test.',
            'F/S/C/U labels overlap and include uncertain risks; not mutually exclusive root causes.',
            'Paired query bootstrap and McNemar are descriptive, conditional on verdicts and one seed; related Mu rows may share primitives.',
            'R02 versus R01 matches parent, updates and exposures, not target tokens; native96 and synthetic40 effects are not isolated.',
            'Full fresh retraining differs from continuation in initialization, sampling and schedule; no single-factor causal claim.'
        ],
    }
    json_path = OUT / 'DATASET_ERROR_ANALYSIS.json'
    txt_path = OUT / 'DATASET_ERROR_ANALYSIS.txt'
    assert not json_path.exists() and not txt_path.exists(), 'Preserve prior analysis: choose a new version.'
    lines = ['分数据集表现与错误分析（仅复用已冻结复核，不改判）', '',
             '每数据集同40题，存疑不算通过；指标为分解通过率，不是RAG F1。',
             '版本\tHotpotQA\t2Wiki\tMuSiQue\t总计']
    for name, ds in summaries.items():
        scores = [f"{ds[d]['passes']}/40 ({ds[d]['pass_percent']:.2f}%)" for d in DATASETS]
        total = sum(ds[d]['passes'] for d in DATASETS)
        lines.append('\t'.join([name] + scores + [f'{total}/120 ({100*total/120:.2f}%)']))
    lines += ['', 'MuSiQue同题变化（百分点；配对bootstrap 30000次，seed2718）']
    for name, ds in comparisons.items():
        m = ds['musique']
        lines.append(f"{name}: 修复{m['fixes']}；退步{m['regressions']}；净增{m['net_passes']}；"
                     f"差值{m['delta_pp']:+.2f}pp；95%区间{m['paired_bootstrap95_delta_pp']}；"
                     f"McNemar双侧p={m['mcnemar_exact_two_sided_p']:.5f}")
    lines += ['', '各数据集未通过情况；F忠实性，S独立表达，C需求覆盖，U辅助效用。维度可重叠且包括存疑。']
    for name, ds in summaries.items():
        for d in DATASETS:
            s = ds[d]
            lines.append(f"{name}/{d}: {s['verdicts']}；"
                         f"维度计数{ {k: len(v) for k, v in s['nonpass_dimensions'].items()} }；"
                         f"F/S风险{s['FS_risk_indices']}")
    lines += ['', '定向选择的实际案例（非随机抽样）']
    for case in cases:
        lines += ['', f"#{case['index']} {case['id']}", case['query']]
        for name, row in case['versions'].items():
            lines += [f"{name}: {row['verdict']} {row['failed_dimensions']}",
                      row['raw_output'], row['reason']]
    lines += ['', '旧16题为定向困难回归集，不能当新样本总体通过率。',
              f"T02 {historical['parent_passes']}/16，R02 {historical['candidate_passes']}/16。",
              '', '范围限制：'] + result['limits']
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Arithmetic must not change any frozen input review.
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == result['sources'][name]['sha256']
               for name, path in PATHS.items())
    print('\n'.join(lines[:36]))


if __name__ == '__main__':
    main()
