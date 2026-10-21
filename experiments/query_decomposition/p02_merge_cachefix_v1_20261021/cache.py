"""Full-payload receipt cache with globally mandatory frozen-p02 priority.

This module only reads existing successful receipts. It never constructs an API
client, reads references/scores, reuses a complete case, or writes old artifacts.
The canonical tier is the 1,774 actually replayed p02 logical calls, including
ordinary B0 calls absent from the old complete-case JSON's logical_calls list.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name
OLD_E = HERE.parent / 'p02_flow_simplify_v1_20261020'
OLD_O = OUTPUT / OLD_E.name
DETAILS = OLD_O / 'OFFLINE_AUDIT.details.jsonl'
MODEL = 'qwen-plus-2025-12-01'
DATASETS = ('hotpotqa', '2wiki', 'musique')
PRIORITY = ('canonical_p02', 'historical_union20', 'historical_related', 'current_namespace')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _readonly(path):
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    return sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=20)


def _eligible(request, status, result):
    return (status == 'success' and isinstance(result, dict)
            and result.get('status') == 'success'
            and isinstance(result.get('answer'), str)
            and result.get('finish_reason') in ('stop', 'length')
            and request.get('model') == MODEL
            and request.get('temperature') == 0
            and request.get('extra_body', {}).get('enable_thinking') is False)


def _entry(source, row, tier, *, expected_key=None):
    rid, raw_payload, status, raw_result = row
    request = json.loads(raw_payload)
    result = json.loads(raw_result) if raw_result else None
    key = digest(request)
    if expected_key is not None:
        assert key == expected_key, 'Logical-call hash differs from stored complete payload'
    if not _eligible(request, status, result):
        return None
    return key, {'payload': request, 'payload_sha256': key, 'request_id': rid,
                 'status': status, 'result': result, 'source': _relative(source),
                 'cache_tier': tier, 'result_sha256': digest(result)}


def _insert(target, key, entry, counts, conflicts, *, canonical=False):
    """Lower-priority entries cannot replace an already-selected receipt."""
    if key not in target:
        target[key] = entry
        counts[entry['cache_tier'] + '_selected'] += 1
        return
    prior = target[key]
    assert prior['payload'] == entry['payload'], 'Full-payload digest collision'
    counts[entry['cache_tier'] + '_duplicate_payload'] += 1
    if prior['result'].get('answer') != entry['result'].get('answer') or prior['result'].get('finish_reason') != entry['result'].get('finish_reason'):
        conflict = {'payload_sha256': key, 'selected_tier': prior['cache_tier'],
                    'selected_request_id': prior['request_id'], 'selected_source': prior['source'],
                    'rejected_tier': entry['cache_tier'], 'rejected_request_id': entry['request_id'],
                    'rejected_source': entry['source'], 'canonical_internal_conflict': canonical,
                    'selected_answer': prior['result']['answer'], 'rejected_answer': entry['result']['answer']}
        conflicts.append(conflict)
        counts[entry['cache_tier'] + '_different_response_blocked'] += 1


def _canonical_inputs():
    """Verify pinned baseline manifests; details are additionally bound below."""
    plan = read(OLD_E / 'PROTOCOL.frozen.json')
    pins = {}
    for name in ('runtime300.frozen.jsonl', 'BASELINE_SOURCES.frozen.json'):
        path = OLD_E / name
        assert sha(path) == plan['input_hashes'][name], path
        pins[_relative(path)] = sha(path)
    pins[_relative(OLD_E / 'PROTOCOL.frozen.json')] = sha(OLD_E / 'PROTOCOL.frozen.json')
    for name in ('OFFLINE_AUDIT.json', 'OFFLINE_AUDIT.details.jsonl'):
        path = OLD_O / name
        pins[_relative(path)] = sha(path)
    baseline = read(OLD_E / 'BASELINE_SOURCES.frozen.json')
    assert len(baseline) == 300
    paths = {record['id']: record for record in baseline}
    assert len(paths) == 300
    for record in baseline:
        path = ROOT / record['path']
        assert sha(path) == record['sha256'], path
        pins[_relative(path)] = record['sha256']
    details = rows(DETAILS)
    assert len(details) == 300 and len({r['id'] for r in details}) == 300
    assert Counter(r['dataset'] for r in details) == {ds: 100 for ds in DATASETS}
    assert sum(len(r['timing_only_replay_calls']) for r in details) == 1774
    for row in details:
        assert row['timing_only_exact'] is True
        assert Path(row['p02_source']).resolve() == (ROOT / paths[row['id']]['path']).resolve()
    oldaudit = read(OLD_O / 'OFFLINE_AUDIT.json')
    assert oldaudit['passed'] and oldaudit['timing_only_outputs_equal_frozen_p02']
    b0 = HERE.parent / 'b0_six_single_changes_v1_20261020/B0_SOURCES.frozen.jsonl'
    b0plan = read(b0.parent / 'PROTOCOL.frozen.json')
    assert sha(b0) == b0plan['baseline_sources_sha256']
    pins[_relative(b0)] = sha(b0)
    # This manifest is not used to select answers: the already-verified p02
    # stage receipts include its ordinary-prefix entries and p02's goal/final.
    return details, pins


def _canonical_catalog():
    details, pins = _canonical_inputs()
    source_ids = defaultdict(set)
    for case in details:
        for call in case['timing_only_replay_calls']:
            source_ids[call['source']].add(call['request_id'])
    stored = {}
    for source, ids in source_ids.items():
        values = sorted(ids)
        with _readonly(ROOT / source) as db:
            for start in range(0, len(values), 400):
                batch = values[start:start + 400]
                sql = 'SELECT id,payload,status,result FROM requests WHERE id IN (' + ','.join('?' for _ in batch) + ')'
                for row in db.execute(sql, batch):
                    stored[(source, row[0])] = row
    target, counts, conflicts = {}, Counter(), []
    ds_counts = Counter()
    logical_receipts = []
    for case in details:  # Frozen source order, never outcome/answer selection.
        for call in case['timing_only_replay_calls']:
            value = _entry(ROOT / call['source'], stored[(call['source'], call['request_id'])],
                           'canonical_p02', expected_key=call['payload_sha256'])
            assert value is not None, 'Missing/invalid canonical success receipt'
            key, entry = value
            _insert(target, key, entry, counts, conflicts, canonical=True)
            ds_counts[case['dataset']] += 1
            logical_receipts.append({'id': case['id'], 'dataset': case['dataset'],
                                     'stage': call['stage'], 'payload_sha256': key,
                                     'request_id': entry['request_id'], 'source': entry['source'],
                                     'result_sha256': entry['result_sha256']})
    stats = {'canonical_cases': 300, 'canonical_logical_calls': len(logical_receipts),
             'canonical_unique_payloads': len(target), 'canonical_calls_by_dataset': dict(ds_counts),
             'canonical_internal_response_conflicts': [r for r in conflicts if r['canonical_internal_conflict']],
             'canonical_selection': 'First occurrence in frozen OFFLINE_AUDIT.details case/call order; no score selection',
             'source_pins': pins, 'canonical_receipt_bindings_sha256': digest(logical_receipts),
             'canonical_receipt_bindings': logical_receipts}
    return target, counts, conflicts, stats


def _historical_databases():
    paths = []
    for arm, tier in [('union20', 'historical_union20'), ('discount20', 'historical_related'),
                      ('single_final', 'historical_related')]:
        paths += [(path, tier) for path in sorted(OLD_O.glob(f'rounds/r*/methods/*/{arm}/checkpoint.sqlite3'))]
    # Related terminal-goal runs have distinct output contracts, so only their
    # byte-identical payloads can match. They never override the tiers above.
    terminal = OUTPUT / 'terminal_goal_v1_20261020'
    paths += [(path, 'historical_related') for path in sorted(terminal.glob('rounds/r*/methods/*/*/checkpoint.sqlite3'))]
    return paths


def _add_database(target, path, tier, counts, conflicts, db_stats):
    eligible = 0
    with _readonly(path) as db:
        for row in db.execute("SELECT id,payload,status,result FROM requests WHERE status='success' ORDER BY id"):
            value = _entry(path, row, tier)
            if value is None:
                counts[tier + '_ineligible_receipts'] += 1
                continue
            eligible += 1
            key, entry = value
            _insert(target, key, entry, counts, conflicts)
    db_stats.append({'path': _relative(path), 'tier': tier, 'eligible_success_rows': eligible})


@lru_cache(maxsize=1)
def _historical_catalog():
    """Immutable historical tier, cached in memory; current rounds stay live."""
    target, counts, conflicts, stats = _canonical_catalog()
    db_stats = []
    for path, tier in _historical_databases():
        _add_database(target, path, tier, counts, conflicts, db_stats)
    return target, counts, conflicts, stats, db_stats


def catalog_with_audit(dataset=None, *, include_current=True):
    """Return global ``(full-payload dictionary, audit stats)``.

    ``dataset`` labels the requesting worker; it never restricts cache scope.
    Successful requests from another dataset can match only when their complete
    payload is equal. The caller must assert entry['payload'] == actual_payload.
    Existing in-flight/unknown liabilities belong to the execution store and
    budget ledger; this success-only seed cannot authorize their resubmission.
    """
    assert dataset is None or dataset in DATASETS
    base, base_counts, base_conflicts, stats, base_db_stats = _historical_catalog()
    target, counts, conflicts = dict(base), Counter(base_counts), list(base_conflicts)
    db_stats = list(base_db_stats)
    if include_current:
        for path in sorted(OUT.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
            _add_database(target, path, 'current_namespace', counts, conflicts, db_stats)
    bindings = [{'payload_sha256': key, 'request_id': value['request_id'], 'source': value['source'],
                 'cache_tier': value['cache_tier'], 'result_sha256': value['result_sha256']}
                for key, value in sorted(target.items())]
    report = {**stats, 'requested_dataset': dataset, 'cache_scope': 'all three datasets; exact complete payload only',
              'priority': list(PRIORITY), 'unique_success_payloads': len(target),
              'counts': dict(counts), 'supplementary_databases': db_stats,
              'blocked_different_response_receipts': conflicts,
              'selected_receipt_bindings_sha256': digest(bindings),
              'selected_receipt_bindings': bindings,
              'code_sha256': sha(Path(__file__)), 'new_api_calls': 0,
              'reads_reference_or_scores': False, 'writes_old_results_or_budget': 0}
    return target, report


def catalog(dataset=None, *, include_current=True):
    return catalog_with_audit(dataset, include_current=include_current)[0]


def audit(dataset=None, *, include_current=True):
    return catalog_with_audit(dataset, include_current=include_current)[1]


def refresh_historical_cache():
    """Explicit refresh for a new process phase; never changes frozen sources."""
    _historical_catalog.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    _, report = catalog_with_audit(args.dataset)
    if args.output:
        output = args.output.resolve()
        output.relative_to(OUT.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ('canonical_cases', 'canonical_logical_calls',
          'canonical_unique_payloads', 'canonical_calls_by_dataset', 'canonical_internal_response_conflicts',
          'unique_success_payloads', 'counts', 'new_api_calls')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
