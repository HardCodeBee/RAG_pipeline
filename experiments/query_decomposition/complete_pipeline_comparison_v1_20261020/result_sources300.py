"""Append-only attempt selection: first complete answer, never choose by F1."""
from pathlib import Path
import json
import sqlite3
from .prepare300 import OUT, RUN, ROOT, read

RETRIES = OUT / 'retry_rounds'


def attempt_roots():
    return [RUN] + sorted(p for p in RETRIES.glob('r[0-9][0-9][0-9]') if p.is_dir())


def case_paths(dataset, method, qid):
    return [r / 'cases' / dataset / method / (qid + '.json') for r in attempt_roots()
            if (r / 'cases' / dataset / method / (qid + '.json')).exists()]


def scorable(value):
    return isinstance(value.get('prediction'), str) and (value.get('status', '').startswith('success') or value.get('status') == 'algorithm_failure')


def select_result(dataset, method, qid):
    """Protect the earliest complete result, including wrong/abstaining answers."""
    paths = case_paths(dataset, method, qid)
    values = [(p, read(p)) for p in paths]
    for p, v in values:
        if scorable(v): return p, v
    return values[-1] if values else (None, {'status': 'unstarted', 'prediction': None})


def database_for(case_path):
    return case_path.parents[3] / 'methods' / case_path.parents[1].name / case_path.parent.name / 'checkpoint.sqlite3'


def logical_calls(database, qid=None):
    database = Path(database)
    if not database.exists(): return []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        query = "SELECT value FROM data WHERE kind='logical_call'"
        args = ()
        if qid is not None:
            query += " AND json_extract(value,'$.qid')=?"; args = (qid,)
        return [json.loads(v) for (v,) in db.execute(query, args)]


def retry_reason(path, value):
    if scorable(value): return None
    if 'Three consecutive connection unknowns' in value.get('reason', ''): return 'dispatch_deferred'
    if value.get('status') in {'unknown', 'in_flight'} or value.get('request_status') in {'unknown', 'in_flight'}:
        return 'provider_connection_interrupted'
    if path is None: return None
    for call in logical_calls(database_for(path), value['qid']):
        if call['status'] in {'unknown', 'in_flight'}: return 'provider_connection_interrupted'
        if call['status'] == 'failed':
            source = ROOT / call['source']
            with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute('SELECT result FROM requests WHERE id=?', (call['request_id'],)).fetchone()
            result = json.loads(row[0]) if row and row[0] else {}
            if result.get('http_status') in {408, 409, 425, 429, 500, 502, 503, 504}: return 'retryable_provider_http_failure'
    return None
