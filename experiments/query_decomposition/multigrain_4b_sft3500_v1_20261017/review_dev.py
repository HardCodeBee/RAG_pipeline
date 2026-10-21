"""Display literal dev predictions and assemble explicitly inspected batches."""
import argparse, collections, json
from pathlib import Path
from reaudit import HERE, OUT, rows, sha, write, write_rows

RETURNED = OUT / 'returned_20261017T1353Z'
STEPS = (110, 165, 220)

def predictions(step):
    result = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
        (RETURNED / 'train/dev_predictions' / f'checkpoint-{step:06d}').glob('*.json'))]
    assert len(result) == 150 and len({r['id'] for r in result}) == 150
    return result

def assemble():
    meta = {r['id']: r for r in rows(HERE / 'dev150.questions.frozen.jsonl')}
    all_predictions = {s: predictions(s) for s in STEPS}
    reviewed = {s: {} for s in STEPS}
    for path in sorted((HERE / 'dev_review').glob('batch_*.json')):
        batch = json.loads(path.read_text(encoding='utf-8'))
        lo, hi = batch['reviewed_indices']
        assert batch['reviewed_steps'] == list(STEPS)
        for step in STEPS:
            exceptions = batch.get('exceptions', {}).get(str(step), {})
            assert all(lo <= int(k) <= hi for k in exceptions)
            for index in range(lo, hi + 1):
                assert index not in reviewed[step]
                pred = all_predictions[step][index - 1]
                assert pred['index'] == index and pred['query'] == meta[pred['id']]['query']
                finding = exceptions.get(str(index))
                verdict = finding['verdict'] if finding else 'pass'
                if verdict == 'pass':
                    assert pred['format_valid'] and not pred['truncated'] and not pred['generation_error'] and not pred['explicit_references']
                reviewed[step][index] = {
                    'index': index, 'id': pred['id'], 'dataset': pred['dataset'],
                    'query': pred['query'], 'raw_output': pred['raw_output'],
                    'verdict': verdict, 'failed_dimensions': finding['dimensions'] if finding else [],
                    'reason': finding['reason'] if finding else 'Inspected all literal questions: no definite F/S/C/U defect under frozen v12; useful multiple granularities allowed.',
                    'review_batch': path.name, 'review_batch_sha256': sha(path),
                    'reviewer': batch['reviewer']}
    table = []
    for step in STEPS:
        records = sorted(reviewed[step].values(), key=lambda r: r['index'])
        write_rows(OUT / 'dev_review' / f'checkpoint-{step:06d}.jsonl', records)
        table.append({'step': step, 'reviewed': len(records), 'denominator': 150,
            'verdicts': dict(collections.Counter(r['verdict'] for r in records)),
            'by_dataset': {d: dict(collections.Counter(r['verdict'] for r in records if r['dataset'] == d)) for d in ['hotpotqa', '2wiki', 'musique']}})
    write(OUT / 'dev_review/SUMMARY.progress.json', {'candidates': table,
        'review_limit': 'Current Codex query-only review, not independent human gold; no eval300 used.'})
    print(json.dumps(table, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--start', type=int); ap.add_argument('--end', type=int); ap.add_argument('--audit', action='store_true'); args = ap.parse_args()
    if args.audit:
        assemble()
    else:
        versions = {s: predictions(s) for s in STEPS}
        for i in range(args.start, args.end + 1):
            first = versions[110][i - 1]
            print(f"{i} [{first['dataset']}] {first['query']}")
            previous = {}
            for step in STEPS:
                r = versions[step][i - 1]; assert r['id'] == first['id']
                print(str(step) + ' ' + ('SAME AS ' + str(previous[r['raw_output']]) if r['raw_output'] in previous else str(r['raw_output'])))
                previous[r['raw_output']] = step
