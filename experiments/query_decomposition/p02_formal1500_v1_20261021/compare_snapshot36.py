"""Freeze the exact36 in the 10:40 status and compare existing native results."""
from collections import Counter
import json
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from .prepare import HERE, OUT, ROOT, read, rows, sha
from .report_comparison import paired
from .run import freeze

CUTOFF = '2026-10-20T23:40:53.039296+00:00'
DEST = OUT/'snapshot36_20261021T104053'


def main():
    cases = []
    for path in OUT.glob('rounds/r*/cases/2wiki/p02_multigrain_timing_cachefix_v1/*.json'):
        value = read(path)
        if value['status'].startswith('success') and value['finished_utc'] <= CUTOFF:
            cases.append((value, path))
    cases.sort(key=lambda cp: (cp[0]['finished_utc'], cp[0]['id']))
    assert len(cases) == len({c['id'] for c, _ in cases}) == 36
    refs = {r['id']: r for r in rows(HERE/'references1500.local.frozen.jsonl')}
    score = metric('2wiki')
    cohort = [{'id': c['id'], 'dataset': c['dataset'], 'qid': c['qid'], 'question': c['question'],
        'P02_source': str(path.relative_to(ROOT)), 'P02_source_sha256': sha(path),
        'finished_utc': c['finished_utc']} for c, path in cases]
    freeze(DEST/'COHORT36.frozen.json', {'cutoff_utc': CUTOFF,
        'selection': 'Exactly the36 successful cases in the previous user-facing report; no new completions included',
        'n': 36, 'reference_sha256': sha(HERE/'references1500.local.frozen.jsonl'), 'cases': cohort})
    ours = {c['id']: {'id': c['id'], 'dataset': '2wiki', 'prediction': c['prediction'],
        'status': c['status'], **score(c['prediction'], refs[c['id']])} for c, _ in cases}
    assert abs(sum(v['f1'] for v in ours.values())/36 - 0.8009259259259259) < 1e-12
    root = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/comprehensive_v1_20261015'
    versions = {'original_campaign_v1': root/'campaign_api_v1',
                'integrity_repair_v2_main': root/'baseline_integrity_repair_v2_20261017/main'}
    table, details = {}, {cid: {'id': cid, 'question': refs[cid]['question'],
        'gold_answer': refs[cid]['answer'], 'P02': value} for cid, value in ours.items()}
    for version, base in versions.items():
        for method in ('b00_rag_top10_v1', 'b03_react_dense'):
            if version == 'original_campaign_v1':
                selected = read(base/'2wiki/BASELINE_CONFIGS_FROZEN.json')['selected'][method]
            else:
                selected = read(base.parent/'CONFIG_FROZEN.json')['native_selected_configs_preserved']['2wiki'][method]
            path = base/'2wiki/formal'/method/selected['id']/'checkpoint.sqlite3'
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
                stored = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
            name = version+'/'+method
            aligned, missing = {}, []
            for c, _ in cases:
                p = stored.get(c['qid'], {})
                status = p.get('status', 'unstarted')
                valid = isinstance(p.get('prediction'), str) and (status.startswith('success') or status == 'algorithm_failure')
                value = {'id': c['id'], 'dataset': '2wiki', 'status': status,
                         'prediction': p.get('prediction'), **(score(p['prediction'], refs[c['id']]) if valid else {'f1': None, 'em': None})}
                aligned[c['id']] = value
                details[c['id']][name] = {**value, 'original_prediction_record': p}
                if not valid:
                    missing.append(c['id'])
            pair, _ = paired(ours, aligned)
            available = [v for v in aligned.values() if v['f1'] is not None]
            table[name] = {'planned': 36, 'scorable': len(available), 'missing': missing,
                'F1': sum(v['f1'] for v in available)/len(available) if available else None,
                'EM': sum(v['em'] for v in available)/len(available) if available else None,
                'statuses': dict(Counter(v['status'] for v in aligned.values())),
                'P02_minus_baseline': pair, 'source': str(path.relative_to(ROOT)),
                'preselected_config': selected}
    assert all(v['scorable'] == 36 for v in table.values())
    result = {'created_utc': utc(), 'cohort_n': 36, 'dataset': '2wiki', 'cutoff_utc': CUTOFF,
        'P02': {'F1': sum(v['f1'] for v in ours.values())/36,
                'EM': sum(v['em'] for v in ours.values())/36, 'n': 36},
        'baselines': table, 'new_API_calls': 0, 'new_API_cost_CNY': 0,
        'scoring': 'Native2Wiki v1.1 Answer F1, official aliases/demonyms; raw final predictions unchanged; algorithm failures included as zero',
        'scope': 'Fixed partial36, not full1500 or a new independent test cohort'}
    target = DEST/'RESULTS.json'
    if target.exists():
        previous = read(target)
        result['created_utc'] = previous['created_utc']
    freeze(target, result)
    freeze(DEST/'PER_QUERY.json', list(details.values()))
    text = 'Fixed36 from the previous 10:40 status;2Wiki native AnswerF1\n\nP02: '+format(result['P02']['F1'], '.8%')+'\n'
    for name, v in table.items():
        delta = v['P02_minus_baseline']['delta_F1']*100
        text += f"{name}: {v['F1']:.8%}, N={v['scorable']}, P02 minus baseline={delta:+.8f} percentage points\n"
    text += '\nReAct algorithm_failure retained as0; no missing item excluded; newAPI calls0, additional cost0.\n'
    (DEST/'REPORT.txt').write_text(text, encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
