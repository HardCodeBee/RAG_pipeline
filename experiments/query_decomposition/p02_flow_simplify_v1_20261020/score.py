"""Offline native final-answer scoring; reference answers never enter execution.

Missing infrastructure results retain their planned denominator and an [0, 1]
bound. Algorithm failures are explicit zeroes, rather than disappearing from
the comparison. The available-only mean is never labelled a complete score.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name
OLD = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/b0_strategy_round2_v1_20261020'
DATASETS = ('hotpotqa', '2wiki', 'musique')
ARMS = ('timing_only', 'union20', 'discount20', 'single_final')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_paths(root, dataset, arm, qid):
    """Only actual case files; never score previews, partial nodes or exports."""
    root = Path(root)
    direct = root / dataset / arm / (qid + '.json')
    candidates = [direct] if direct.exists() else []
    for pattern in (f'rounds/*/cases/{dataset}/{arm}/{qid}.json',
                    f'*/cases/{dataset}/{arm}/{qid}.json',
                    f'cases/{dataset}/{arm}/{qid}.json'):
        candidates.extend(sorted(root.glob(pattern)))
    return sorted(set(candidates))


def select_case(root, dataset, arm, qid):
    values = [(path, read(path)) for path in case_paths(root, dataset, arm, qid)]
    terminal = [(p, v) for p, v in values if is_scorable(v)]
    if terminal:
        predictions = {(v['status'].startswith('success'), v.get('prediction')) for _, v in terminal}
        if len(predictions) != 1:
            raise AssertionError(f'Conflicting completed results: {dataset}:{qid}/{arm}')
        return terminal[0]
    return values[-1] if values else (None, None)


def is_scorable(value):
    if not value:
        return False
    return value.get('status') == 'algorithm_failure' or (
        value.get('status', '').startswith('success') and isinstance(value.get('prediction'), str))


def _stats(items):
    valid = [item for item in items if item['f1'] is not None]
    n, m = len(items), len(valid)
    total = sum(item['f1'] for item in valid)
    em = sum(item['em'] for item in valid)
    paired = [item for item in valid if item.get('baseline_f1') is not None]
    delta = [item['f1'] - item['baseline_f1'] for item in paired]
    direct = [item for item in valid if item.get('direct_parent_f1') is not None]
    direct_delta = [item['f1'] - item['direct_parent_f1'] for item in direct]
    return {
        'planned': n, 'scorable': m, 'missing': n - m,
        'statuses': dict(Counter(item['status'] for item in items)),
        'F1_available': total / m if m else None,
        'EM_available': em / m if m else None,
        'F1_complete': total / n if n and n == m else None,
        'EM_complete': em / n if n and n == m else None,
        'F1_all_planned_bounds': [total / n, (total + n - m) / n] if n else None,
        'paired_with_frozen_p02': {
            'n': len(paired),
            'F1': sum(item['f1'] for item in paired) / len(paired) if paired else None,
            'baseline_F1': sum(item['baseline_f1'] for item in paired) / len(paired) if paired else None,
            'delta': sum(delta) / len(delta) if delta else None,
            'up': sum(d > 1e-12 for d in delta),
            'tie': sum(abs(d) <= 1e-12 for d in delta),
            'down': sum(d < -1e-12 for d in delta),
        },
        'paired_with_direct_parent': {
            'n': len(direct),
            'F1': sum(item['f1'] for item in direct) / len(direct) if direct else None,
            'parent_F1': sum(item['direct_parent_f1'] for item in direct) / len(direct) if direct else None,
            'delta': sum(direct_delta) / len(direct_delta) if direct_delta else None,
            'up': sum(d > 1e-12 for d in direct_delta),
            'tie': sum(abs(d) <= 1e-12 for d in direct_delta),
            'down': sum(d < -1e-12 for d in direct_delta),
        },
    }


def summarize(cases_root=OUT, samples_path=None, *, arms=ARMS, references=None,
              baseline_root=OLD, output_path=None):
    samples_path = Path(samples_path or HERE / 'runtime300.frozen.jsonl')
    references = Path(references or HERE / 'references300.local.frozen.jsonl')
    samples = rows(samples_path)
    refs = {ref['id']: ref for ref in rows(references)}
    assert len(samples) == len({case['id'] for case in samples}), 'Duplicate sample ID'
    assert all(case['id'] in refs and case['dataset'] in DATASETS for case in samples)
    scorers = {dataset: metric(dataset) for dataset in DATASETS}
    details, results = [], {}
    for arm in arms:
        cohort, logical_usage, new_ids = [], Counter(), set()
        for case in samples:
            cid, ds, qid = case['id'], case['dataset'], case['qid']
            path, value = select_case(cases_root, ds, arm, qid)
            if value:
                assert value.get('id', cid) == cid and value.get('dataset', ds) == ds
                assert value.get('qid', qid) == qid
            item = {'id': cid, 'dataset': ds, 'arm': arm,
                    'status': value.get('status', 'unstarted') if value else 'unstarted',
                    'prediction': value.get('prediction') if value else None,
                    'source': str(path) if path else None, 'f1': None, 'em': None,
                    'baseline_f1': None, 'baseline_em': None,
                    'direct_parent_f1': None, 'direct_parent_arm': None}
            if is_scorable(value):
                if value['status'] == 'algorithm_failure':
                    item.update(f1=0.0, em=0.0)
                else:
                    item.update(scorers[ds](value['prediction'], refs[cid]))
            if baseline_root:
                _, base = select_case(baseline_root, ds, 'p02_role_headers', qid)
                if is_scorable(base):
                    bs = scorers[ds](base['prediction'], refs[cid])
                    item.update(baseline_f1=bs['f1'], baseline_em=bs['em'])
            parent_arm = {'timing_only': 'p02_role_headers', 'union20': 'p02_role_headers',
                          'discount20': 'union20', 'single_final': 'discount20'}.get(arm)
            item['direct_parent_arm'] = parent_arm
            if parent_arm == 'p02_role_headers':
                item['direct_parent_f1'] = item['baseline_f1']
            elif parent_arm:
                _, parent = select_case(cases_root, ds, parent_arm, qid)
                if is_scorable(parent):
                    item['direct_parent_f1'] = 0.0 if parent['status'] == 'algorithm_failure' else scorers[ds](parent['prediction'], refs[cid])['f1']
            cohort.append(item)
            if value:
                for call in value.get('logical_calls', []):
                    if call.get('status') != 'success':
                        continue
                    usage = call.get('usage') or {}
                    logical_usage.update({k: usage.get(k, 0) or 0 for k in ['prompt_tokens', 'completion_tokens']})
                    logical_usage['successful_logical_calls'] += 1
                    if not call.get('reused', False):
                        new_ids.add((call.get('source'), call.get('request_id')))
        details.extend(cohort)
        results[arm] = {**_stats(cohort),
                        'by_dataset': {ds: _stats([item for item in cohort if item['dataset'] == ds]) for ds in DATASETS},
                        'logical_usage_on_selected_case_records': dict(logical_usage),
                        'new_success_request_ids_on_selected_records': len(new_ids),
                        'cost_note': 'Logical usage includes exact cache replays; actual payments and unresolved reservations are reported by the runner budget ledger.'}
    result = {'sample_file': str(samples_path), 'sample_sha256': sha(samples_path),
              'reference_file': str(references), 'reference_sha256': sha(references),
              'planned': len(samples), 'dataset_counts': dict(Counter(c['dataset'] for c in samples)),
              'metric': 'Native Answer F1/EM: HotpotQA; 2Wiki v1.1 entity aliases/demonyms; MuSiQue answer aliases',
              'aggregation': 'Mean over planned query cohort only when complete; available-only mean and missing bounds otherwise; datasets also reported separately.',
              'scores_are_0_to_1': True, 'algorithm_failure_scored_zero': True,
              'infrastructure_missing_not_scored_zero': True,
              'arms': results}
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        output_path.with_suffix('.details.jsonl').write_text(
            ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in details), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['pilot60', '300'], default='300')
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--cases-root', type=Path, default=OUT)
    parser.add_argument('--samples-jsonl', type=Path)
    parser.add_argument('--references', type=Path, default=HERE / 'references300.local.frozen.jsonl')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sample = args.samples_jsonl or HERE / ('pilot60.runtime.frozen.jsonl' if args.sample == 'pilot60' else 'runtime300.frozen.jsonl')
    output = args.output or OUT / ('SCORES_' + args.sample + '_LATEST.json')
    value = summarize(args.cases_root, sample, arms=args.arms, references=args.references, output_path=output)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
