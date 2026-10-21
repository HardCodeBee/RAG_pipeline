"""Offline execution policies; never dispatch requests or load a model."""
from collections import Counter
import hashlib
import math

TIMING_SEED = 20261021
MAX_FAILED_ATTEMPTS = 5  # Includes the initial attempt; persists across restarts.
SINGLE_QUERY_COST_RATIO = 3.0
ROLLING_QUERY_COST_RATIO = 1.5
COST_WINDOW = 20


def _rank(seed, dataset, purpose, qid):
    return hashlib.sha256(f'{seed}:{dataset}:{purpose}:{qid}'.encode('utf8')).hexdigest()


def select_timing_questions(questions, metadata, dataset, n=50, seed=TIMING_SEED):
    """Use only qid/type/group metadata; no outcomes enter sample selection."""
    if any(set(r) != {'qid', 'type', 'group_id'} for r in metadata):
        raise ValueError('Project reference metadata before calling the selector')
    ids = [r['qid'] for r in questions]
    meta = {r['qid']: r for r in metadata}
    if len(ids) != len(set(ids)) or len(meta) != len(metadata) or set(ids) != set(meta):
        raise ValueError('Question and metadata identities differ')
    if not 0 < n <= len(ids) or len({r['group_id'] for r in metadata}) != len(ids):
        raise ValueError('Invalid sample size or duplicate formal groups')
    counts = Counter(r['type'] for r in metadata)
    total = len(ids)
    quota = {kind: n * count // total for kind, count in counts.items()}
    order = sorted(counts, key=lambda k: (-(n * counts[k] % total), k))
    for kind in order[:n - sum(quota.values())]:
        quota[kind] += 1
    chosen = []
    for kind in sorted(quota):
        candidates = [qid for qid in ids if meta[qid]['type'] == kind]
        candidates.sort(key=lambda qid: (_rank(seed, dataset, 'sample', qid), qid))
        chosen.extend(candidates[:quota[kind]])
    chosen.sort(key=lambda qid: (_rank(seed, dataset, 'order', qid), qid))
    return [{'qid': qid, 'stratum': meta[qid]['type']} for qid in chosen], quota


def recovery_action(status, attempts):
    """Count actual wire attempts once, shared by SDK/controller/restarts."""
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError('Invalid persisted attempt count')
    if status == 'success':
        return 'complete'
    if status == 'algorithm_failure':
        return 'preserve_algorithm_failure_zero'
    if status in {'unknown', 'in_flight'}:
        return 'hold_for_receipt_verification'
    if status in {'model_identity_mismatch', 'input_token_limit_violation', 'integrity_failure'}:
        return 'pause_for_integrity_check'
    if attempts >= MAX_FAILED_ATTEMPTS:
        return 'record_and_temporarily_skip'
    return 'retry_confirmed_failure_only'


def expense_signals(reference_mean, query_cost_cny, recent_fresh_complete_costs=()):
    """Notify on increases; these ratios are not spending caps or score filters.

    Single-query cost includes known new paid attempts, including failed ones.
    The rolling window accepts only full fresh flows with complete usage; partial
    cache completions and unknown usage must never be supplied as zero-cost flows.
    """
    if reference_mean is None or reference_mean <= 0:
        return [{'kind': 'missing_historical_cost_reference'}]
    values = list(recent_fresh_complete_costs)[-COST_WINDOW:]
    for value in [query_cost_cny, *values]:
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid observed cost')
    alerts = []
    if query_cost_cny is None:
        alerts.append({'kind': 'unknown_query_usage_not_zero'})
    elif query_cost_cny > SINGLE_QUERY_COST_RATIO * reference_mean:
        alerts.append({'kind': 'single_query_cost_increase', 'ratio': query_cost_cny / reference_mean})
    if len(values) == COST_WINDOW and all(v is not None for v in values):
        mean = sum(values) / COST_WINDOW
        if mean > ROLLING_QUERY_COST_RATIO * reference_mean:
            alerts.append({'kind': 'rolling_query_cost_increase', 'ratio': mean / reference_mean,
                           'window': COST_WINDOW})
    return alerts
