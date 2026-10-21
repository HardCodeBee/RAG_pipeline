"""Read-only native baseline comparison and per-dataset expense report. No API."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from .prepare import HERE, ROOT, OUT, read, rows
from .score import selected, summarize, DATASETS


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=30)


def paired(ours, other):
    byds = {ds: [] for ds in DATASETS}
    details = []
    for cid, p in ours.items():
        b = other.get(cid)
        if p.get('f1') is None or b is None or b.get('f1') is None:
            continue
        delta = p['f1']-b['f1']
        byds[p['dataset']].append(delta)
        details.append({'id': cid, 'dataset': p['dataset'], 'P02_F1': p['f1'],
                        'baseline_F1': b['f1'], 'delta': delta})
    flat = [x for vv in byds.values() for x in vv]
    if not flat:
        return {'common_scorable': 0}, details
    rng = np.random.default_rng(1729)
    boot = np.zeros(5000)
    for vv in byds.values():
        if vv:
            a = np.asarray(vv)
            boot += a[rng.integers(0, len(a), size=(5000, len(a)))].sum(axis=1)
    boot /= len(flat)
    return {'common_scorable': len(flat), 'delta_F1': float(np.mean(flat)),
        'paired_stratified_95CI': np.quantile(boot, [.025, .975]).tolist(),
        'up': sum(x > 1e-12 for x in flat), 'equal': sum(abs(x) <= 1e-12 for x in flat),
        'down': sum(x < -1e-12 for x in flat),
        'by_dataset': {ds: {'n': len(vv), 'delta_F1': float(np.mean(vv)) if vv else None}
                       for ds, vv in byds.items()},
        'provisional_until_all1500_complete': len(flat) != 1500}, details


def main():
    refs = rows(HERE/'references1500.local.frozen.jsonl')
    scorers = {ds: metric(ds) for ds in DATASETS}
    ours, cost = {}, {ds: Counter() for ds in DATASETS}
    coverage = {ds: [] for ds in DATASETS}
    for ref in refs:
        case = selected(ref)
        status = case['status'] if case else 'not_completed'
        valid = case and (status.startswith('success') or status == 'algorithm_failure')
        native = scorers[ref['dataset']]((case.get('prediction') or ''), ref) if valid else {'f1': None, 'em': None}
        ours[ref['id']] = {'id': ref['id'], 'dataset': ref['dataset'], 'status': status, **native}
        if not case:
            continue
        c = cost[ref['dataset']]
        for call in case.get('logical_calls', []):
            u = call.get('usage') or {}
            c.update(logical_calls_selected_attempts=1, reused_calls_selected_attempts=int(bool(call.get('reused'))),
                     prompt_tokens_selected_attempts=u.get('prompt_tokens', 0),
                     completion_tokens_selected_attempts=u.get('completion_tokens', 0))
        c['logical_retrievals_selected_attempts'] += len(case.get('logical_retrievals', []))
        if status.startswith('success'):
            gold = set(ref['positive_doc_ids'])
            shown = {d['doc_id'] for n in case['nodes'] for d in n['reader_packet']['docs']}
            shown.update(d['doc_id'] for d in case['original_retrieval']['docs'])
            coverage[ref['dataset']].append({'id': ref['id'],
                'gold_docs': len(gold), 'reader_gold_recall': len(gold & shown)/len(gold) if gold else None,
                'reader_full_gold_coverage': gold.issubset(shown) if gold else None})
    ledger = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3'
    with ro(ledger) as db:
        for namespace, actual, reserved in db.execute(
            'SELECT namespace,actual,reserved FROM charges WHERE namespace LIKE ?', (HERE.name+'%',)):
            parts = namespace.replace('\\', '/').split('/')
            ds = parts[parts.index('methods')+1]
            cost[ds]['new_API_submissions'] += 1
            cost[ds]['known_CNY'] += actual or 0
            cost[ds]['unreconciled_reserved_CNY'] += reserved if actual is None else 0
    cost_report = {}
    for ds, vv in cost.items():
        c = dict(vv)
        c['logical_cold_cost_CNY_selected_attempts'] = vv['prompt_tokens_selected_attempts']*.8e-6+vv['completion_tokens_selected_attempts']*2e-6
        c['planned'] = 500
        c['coverage_successful_cases'] = len(coverage[ds])
        c['reader_full_gold_document_coverage_rate_successful_only'] = (sum(v['reader_full_gold_coverage'] for v in coverage[ds])/len(coverage[ds])) if coverage[ds] else None
        cost_report[ds] = c
    root = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/comprehensive_v1_20261015'
    campaigns = {'original_campaign_v1': root/'campaign_api_v1',
        'integrity_repair_v2_main': root/'baseline_integrity_repair_v2_20261017/main',
        'primary_v2_with_missing_recovery': root/'baseline_integrity_repair_v2_20261017/main'}
    report, pair_details = {}, []
    for label, campaign in campaigns.items():
        methods = sorted({p.name for ds in DATASETS for p in (campaign/ds/'formal').glob('b*') if p.is_dir()})
        for method in methods:
            values, sources, selected_configs = {}, [], {}
            for ds in DATASETS:
                # Parameter IDs differ by method (ReAct uses k10_s5, not fixed).
                # Read the preselected manifest, never select using formal F1.
                if label == 'original_campaign_v1':
                    manifest = campaign/ds/'BASELINE_CONFIGS_FROZEN.json'
                    if not manifest.exists():
                        continue
                    config = read(manifest)['selected'].get(method)
                else:
                    manifest = campaign.parent/'CONFIG_FROZEN.json'
                    config = read(manifest)['native_selected_configs_preserved'].get(ds, {}).get(method)
                if config is None:
                    continue
                if label == 'primary_v2_with_missing_recovery' and method == 'b05_iter_retgen_bge':
                    config = {'id':'k10_t5','k':10,'iterations':5}
                selected_configs[ds] = config
                path = campaign/ds/'formal'/method/config['id']/'checkpoint.sqlite3'
                if label == 'primary_v2_with_missing_recovery' and method == 'b05_iter_retgen_bge':
                    path = campaign.parent/'matched50'/ds/'formal'/method/config['id']/'checkpoint.sqlite3'
                if not path.exists():
                    continue
                sources.append(str(path.relative_to(ROOT)))
                if label == 'primary_v2_with_missing_recovery' and ds == '2wiki':
                    from experiments.query_decomposition.twowiki_missing_recovery_20261021.run import baseline_selected
                    for qid, value in baseline_selected(method,path).items():
                        values[ds+':'+qid] = value
                    sources.extend(str(p.relative_to(ROOT)) for p in sorted(
                        (ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/twowiki_missing_recovery_20261021').glob(
                            f'rounds/r*/methods/{method}/*/checkpoint.sqlite3')))
                    continue
                with ro(path) as db:
                    for qid, raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'"):
                        value = json.loads(raw)
                        values[ds+':'+qid] = value
            aligned = []
            for ref in refs:
                p = values.get(ref['id'], {})
                status = p.get('status', 'not_completed')
                valid = isinstance(p.get('prediction'), str) and (status.startswith('success') or status == 'algorithm_failure')
                s = scorers[ref['dataset']](p['prediction'], ref) if valid else {'f1': None, 'em': None}
                aligned.append({'id': ref['id'], 'dataset': ref['dataset'], 'status': status, **s})
            pair, details = paired(ours, {v['id']: v for v in aligned})
            name = label+'/'+method
            pair_details.extend({'baseline': name, **v} for v in details)
            report[name] = {'overall': summarize(aligned),
                'by_dataset': {ds: summarize([v for v in aligned if v['dataset'] == ds]) for ds in DATASETS},
                'paired_P02_minus_baseline': pair, 'readonly_sources': sources,
                'preselected_configs': selected_configs}
    value = {'utc': utc(), 'planned': 1500, 'per_dataset': 500, 'P02': summarize(list(ours.values())),
        'comparisons': report, 'costs_by_dataset': cost_report,
        'configuration_labels_kept_separate': True, 'new_API_calls': 0,
        'caveats': ['Incomplete cohorts are provisional; paired counts shown explicitly.',
            'Original and integrity-repair baseline versions are separate; no best-answer mixing.',
            'Method-specific readers/prompts are retained; this is not the unexecuted matched-final-prompt control.',
            'Document coverage is over successful reader exposures, not official sentence or joint F1.',
            'Token costs use recorded list-price estimates; actual bills unverified.']}
    locked_json(OUT/'BASELINE_COMPARISONS.json', value)
    locked_json(OUT/'COSTS_BY_DATASET.json', {'utc': utc(), 'by_dataset': cost_report, 'new_API_calls': 0})
    (OUT/'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in pair_details), encoding='utf8')
    (OUT/'READER_COVERAGE.jsonl').write_text(''.join(json.dumps(v)+'\n' for vv in coverage.values() for v in vv), encoding='utf8')
    print(json.dumps({'P02': value['P02'], 'baseline_configurations': len(report), 'new_API_calls': 0}))


if __name__ == '__main__':
    main()
