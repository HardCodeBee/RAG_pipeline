"""Offline native final-answer scoring; never imported by method execution."""
from collections import Counter
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from .prepare import HERE, OUT, ROOT, read, rows, sha

METHOD = 'p02_multigrain_timing_cachefix_v1'
DATASETS = ('hotpotqa', '2wiki', 'musique')


def selected(q):
    last = None
    for p in sorted(OUT.glob(f'rounds/r*/cases/{q["dataset"]}/{METHOD}/{q["qid"]}.json')):
        value = read(p); value['_source'] = str(p.relative_to(ROOT)); last = value
        if value['status'].startswith('success') or value['status'] == 'algorithm_failure':
            return value
    return last


def summarize(vv):
    valid = [v for v in vv if v['f1'] is not None]
    n, available = len(vv), len(valid)
    total = sum(v['f1'] for v in valid)
    return {'planned': n, 'scorable': available, 'missing': n-available,
        'successful_answers': sum(v['status'].startswith('success') for v in vv),
        'algorithm_failures': sum(v['status'] == 'algorithm_failure' for v in vv),
        'F1': total/available if available else None,
        'EM': sum(v['em'] for v in valid)/available if available else None,
        'F1_all_planned_bounds': [total/n, (total+n-available)/n],
        'statuses': dict(Counter(v['status'] for v in vv))}


def main():
    refs = rows(HERE/'references1500.local.frozen.jsonl')
    assert len(refs) == len({r['id'] for r in refs}) == 1500
    scores = {ds: metric(ds) for ds in DATASETS}
    details, logical = [], []
    for ref in refs:
        case = selected(ref)
        status = case['status'] if case else 'not_completed'
        valid = case and (status.startswith('success') or status == 'algorithm_failure')
        prediction = (case.get('prediction') or '') if case else None
        native = scores[ref['dataset']](prediction, ref) if valid else {'f1': None, 'em': None}
        details.append({'id': ref['id'], 'qid': ref['qid'], 'dataset': ref['dataset'],
            'type': ref['type'], 'status': status, 'prediction': prediction, 'answer': ref['answer'],
            'source': case.get('_source') if case else None, **native})
        if case:
            logical.extend({**c, 'dataset': ref['dataset']} for c in case.get('logical_calls', []))
    costs = {}
    ledger = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3'
    with sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True) as db:
        for label, where, args in [('global', '', ()), ('experiment', ' WHERE namespace LIKE ?', (HERE.name+'%',))]:
            actual, reserved, calls = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges'+where, args).fetchone()
            costs[label] = {'known_CNY': actual, 'unreconciled_reserved_CNY': reserved,
                'occupied_CNY': actual+reserved, 'submissions': calls, 'actual_bill_verified': False}
    token_totals = Counter()
    for c in logical:
        u = c.get('usage') or {}
        token_totals.update(prompt=u.get('prompt_tokens', 0), completion=u.get('completion_tokens', 0))
    report = {'utc': utc(), 'method': METHOD, 'cohort': 'All original formal1500,500 per dataset; no exclusions',
        'reference_sha256': sha(HERE/'references1500.local.frozen.jsonl'),
        'overall': summarize(details),
        'by_dataset': {ds: summarize([v for v in details if v['dataset'] == ds]) for ds in DATASETS},
        'by_type': {ds: {kind: summarize([v for v in details if v['dataset'] == ds and v['type'] == kind])
            for kind in sorted({v['type'] for v in details if v['dataset'] == ds})} for ds in DATASETS},
        'logical_calls_selected_attempts': len(logical),
        'exact_cache_reuses_selected_attempts': sum(bool(v.get('reused')) for v in logical),
        'logical_tokens_selected_attempts': dict(token_totals),
        'logical_cold_cost_CNY_selected_attempts': token_totals['prompt']*.8e-6+token_totals['completion']*2e-6,
        'costs': costs, 'metric': 'Native final Answer F1/EM;2Wiki aliases/demonyms;MuSiQue aliases',
        'prediction_cleanup': False, 'API_missing_excluded_from_available_F1_only': True,
        'algorithm_failures_included_as_zero': True, 'hallucination_judge_run': False}
    OUT.mkdir(parents=True, exist_ok=True)
    locked_json(OUT/'SCORES.json', report)
    path = OUT/'SCORES.jsonl'
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in details), encoding='utf8')
    temporary.replace(path)
    summary = 'P02 formal1500\n\n'
    for ds in DATASETS:
        s = report['by_dataset'][ds]
        summary += f"{ds}: scorable {s['scorable']}/500; F1 {s['F1']}; EM {s['EM']}; statuses {s['statuses']}\n"
    summary += '\n'+json.dumps(costs, ensure_ascii=False, indent=2)+'\n'
    (OUT/'REPORT.txt').write_text(summary, encoding='utf8')
    print(json.dumps({'overall': report['overall'], 'by_dataset': report['by_dataset'], 'costs': costs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
