"""Read-only restoration of exact frozen B0 API receipts omitted by v001.

The authoritative call records live in B0_SOURCES.frozen.jsonl, rather than in
the complete B0 case JSON. Payloads/results are loaded in grouped SQL queries
from each record's original SQLite source. No requests or charges are inserted.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from .score import ROOT, HERE, OUT, DATASETS, read, rows, sha

SOURCE_HERE = HERE.parent / 'b0_six_single_changes_v1_20261020'
MANIFEST = SOURCE_HERE / 'B0_SOURCES.frozen.jsonl'


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=20)


def baseline_catalog(dataset, *, model='qwen-plus-2025-12-01'):
    """Return ``(compatible_dictionary, audit_stats)`` without mutation.

    First successful receipt in the frozen manifest's existing case/call order
    wins on duplicate exact payloads. No F1, reference answer or candidate score
    is read. Caller should use ``catalog.setdefault(key, value)`` to preserve
    its already-selected receipts and all outstanding liabilities.
    """
    assert dataset in DATASETS
    plan = read(SOURCE_HERE / 'PROTOCOL.frozen.json')
    assert sha(MANIFEST) == plan['baseline_sources_sha256']
    cases = [case for case in rows(MANIFEST) if case['dataset'] == dataset]
    assert len(cases) == 100
    calls = [call for case in cases for call in case['logical_calls'] if call['status'] == 'success']
    db_ids = defaultdict(set)
    for call in calls:
        db_ids[call['source']].add(call['request_id'])
    stored = {}
    for source, request_ids in db_ids.items():
        values = sorted(request_ids)
        with ro(ROOT / source) as db:
            for offset in range(0, len(values), 400):
                batch = values[offset:offset + 400]
                sql = 'SELECT id,payload,status,result FROM requests WHERE id IN (' + ','.join('?' for _ in batch) + ')'
                for rid, raw_payload, status, raw_result in db.execute(sql, batch):
                    stored[(source, rid)] = (json.loads(raw_payload), status, json.loads(raw_result) if raw_result else None)
    catalog, duplicates = {}, 0
    for call in calls:
        request, status, result = stored[(call['source'], call['request_id'])]
        key = digest(request)
        assert key == call['payload_sha256'], 'Frozen logical hash differs from stored full payload'
        assert status == 'success' and result and result.get('status') == 'success'
        assert isinstance(result.get('answer'), str) and result.get('finish_reason') in ('stop', 'length')
        assert request['model'] == model
        assert request.get('temperature') == 0 and request.get('extra_body', {}).get('enable_thinking') is False
        entry = {'request_id': call['request_id'], 'status': status, 'result': result, 'source': call['source']}
        if key in catalog:
            duplicates += 1
        catalog.setdefault(key, entry)
    stats = {'dataset': dataset, 'manifest': str(MANIFEST.relative_to(ROOT)), 'manifest_sha256': sha(MANIFEST),
             'source_cases': len(cases), 'logical_success_receipts': len(calls),
             'unique_payloads': len(catalog), 'duplicate_exact_payloads': duplicates,
             'original_databases': {source: len(ids) for source, ids in db_ids.items()},
             'selection': 'First successful receipt in frozen B0 manifest order; no answer-quality selection',
             'writes_to_requests_or_budget': 0}
    return catalog, stats


def augment(catalog, dataset, *, model='qwen-plus-2025-12-01'):
    """Return a copy, preserving existing entries; never overwrite unknowns."""
    old, stats = baseline_catalog(dataset, model=model)
    result = dict(catalog)
    for key, entry in old.items():
        result.setdefault(key, entry)
    return result, {**stats, 'existing_entries': len(catalog), 'added_entries': len(result) - len(catalog)}


def cache_reuse_audit(*, output=OUT / 'CACHE_REUSE_AUDIT.json'):
    """Identify new physical requests that had an exact older B0 success.

    Requests remain valid real inference and their actual costs remain charged.
    This audit neither refunds them nor replaces any completed result.
    """
    catalogs, source_stats = {}, []
    for ds in DATASETS:
        catalogs[ds], stats = baseline_catalog(ds)
        source_stats.append(stats)
    budget_path = OUT.parent / 'budget.sqlite3'
    with ro(budget_path) as db:
        charges = {rid: {'namespace': namespace, 'reserved_cny': reserved,
                         'actual_returned_usage_cny': actual, 'status': status}
                   for rid, namespace, reserved, actual, status in db.execute(
                       'SELECT id,namespace,reserved,actual,status FROM charges WHERE namespace LIKE ?', (OUT.name + '%',))}
    all_requests, matching = [], []
    for path in sorted(OUT.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
        ds, arm = path.parent.parent.name, path.parent.name
        if ds not in catalogs:
            continue
        with ro(path) as db:
            requests = list(db.execute('SELECT id,stage,payload,status,result FROM requests'))
        for rid, stage, raw_payload, status, raw_result in requests:
            request = json.loads(raw_payload)
            key = digest(request)
            result = json.loads(raw_result) if raw_result else {}
            item = {'request_id': rid, 'source': str(path.relative_to(ROOT)), 'dataset': ds, 'arm': arm,
                    'stage': stage, 'payload_sha256': key, 'status': status,
                    'usage': result.get('usage', {}), 'charge': charges.get(rid)}
            all_requests.append(item)
            if key in catalogs[ds]:
                old = catalogs[ds][key]
                matching.append({**item, 'old_success_request_id': old['request_id'], 'old_source': old['source']})

    def stats(items):
        ledger = [item['charge'] for item in items if item['charge'] is not None]
        actual = sum(item['actual_returned_usage_cny'] or 0 for item in ledger)
        reserve = sum(item['reserved_cny'] for item in ledger if item['actual_returned_usage_cny'] is None)
        return {'new_physical_request_rows': len(items),
                'statuses': dict(Counter(item['status'] for item in items)),
                'returned_usage_cny': actual, 'unreconciled_reserved_cny': reserve,
                'occupied_cny': actual + reserve,
                'prompt_tokens_known': sum((item['usage'] or {}).get('prompt_tokens', 0) or 0 for item in items),
                'completion_tokens_known': sum((item['usage'] or {}).get('completion_tokens', 0) or 0 for item in items)}
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'finding': 'v001 catalog did not read the original B0 call manifest, so some exact successful legacy prefixes were submitted again.',
              'f1_validity': 'These remain actual recorded inferences with full payload lineage; cached old answers were not spliced into changed downstream inputs.',
              'interpretation': 'The matched cost is an observed avoidable-request estimate for this prefix-cache omission, not a refund or a claim that all total task spending was avoidable.',
              'all_task_new_requests': stats(all_requests), 'new_requests_with_old_exact_success': stats(matching),
              'by_arm': {arm: stats([item for item in matching if item['arm'] == arm])
                         for arm in sorted({item['arm'] for item in all_requests})},
              'by_dataset': {ds: stats([item for item in matching if item['dataset'] == ds]) for ds in DATASETS},
              'source_manifests': source_stats, 'matched_requests': matching,
              'actual_bill_verified': False, 'new_api_calls': 0, 'modified_existing_results': False,
              'cost_accounting_rule': 'Historical seeds retain their original source DB and request ID. Importing a cache dictionary does not insert a task requests row or a task budget charge.',
              'helper_sha256': sha(Path(__file__))}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--dataset', choices=list(DATASETS))
    parser.add_argument('--output', type=Path, default=OUT / 'CACHE_REUSE_AUDIT.json')
    args = parser.parse_args()
    if args.audit:
        value = cache_reuse_audit(output=args.output)
        print(json.dumps({key: value[key] for key in ['utc', 'all_task_new_requests', 'new_requests_with_old_exact_success', 'by_arm', 'by_dataset']}, ensure_ascii=False))
    else:
        datasets = [args.dataset] if args.dataset else DATASETS
        print(json.dumps([baseline_catalog(ds)[1] for ds in datasets], ensure_ascii=False))


if __name__ == '__main__':
    main()
