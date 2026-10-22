"""Read-only, same-method C1 receipts and unresolved historical submissions.

This adapter never imports predictions, chooses a response by score, mutates an
old database, or considers an unverified ``validated: true`` claim sufficient.
Retrieval is performed by the current runner; only a response whose complete
input and already-executed retrieval agree can become a historical prefix.
"""
from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


P02 = 'p02_multigrain_timing_cachefix_v1'
UNRESOLVED = {'unknown', 'in_flight', 'reserved', 'submitting'}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _records(db, kinds=None, qid=None):
    if 'data' not in _tables(db):
        return []
    if kinds:
        marks = ','.join('?' for _ in kinds)
        query = 'SELECT kind,key,value FROM data WHERE kind IN (' + marks + ')'
        args = tuple(kinds)
        if qid is not None:
            query += " AND json_extract(value,'$.qid')=?"
            args += (qid,)
        rows = db.execute(query, args)
    else:
        rows = db.execute('SELECT kind,key,value FROM data')
    return [(r[0], r[1], json.loads(r[2])) for r in rows]


def _identity(db):
    return next((v for k, key, v in _records(db, ['meta']) if key == 'identity'), {})


def _resource_identity(resources):
    """Use frozen content/model identifiers, ignoring machine-local paths."""
    corpus, bge = resources['corpus'], resources['dense']['bge']
    return {
        'corpus': {k: corpus[k] for k in ('dataset', 'documents', 'manifest_sha256', 'chunks_sha256')},
        'bge': {k: bge[k] for k in ('manifest_sha256', 'query_model', 'query_revision', 'space')},
        'query_prefix': bge.get('query_prefix', ''),
        'embeddings_manifest_sha256': bge.get('embeddings_manifest_sha256'),
        'identical_id_policy': resources.get('identical_id_policy'),
    }


def discover_sources(root, method, dataset):
    """Only the requested method/dataset history; no unrelated probe scan."""
    root = Path(root)
    base = root / 'outputs/query_decomposition/hotpot_formal_eval_v1'
    sources = []
    if method == P02:
        old = base / 'p02_formal1500_v1_20261021'
        protocol = root / 'experiments/query_decomposition/p02_formal1500_v1_20261021/PROTOCOL.frozen.json'
        for path in sorted(old.glob(f'rounds/*/methods/{dataset}/{method}/checkpoint.sqlite3')):
            sources.append({'path': path, 'protocol_file': protocol,
                            'retrieval_paths': [old / f'retrieval/{dataset}/checkpoint.sqlite3']})
    else:
        for campaign in (base / 'comprehensive_v1_20261015', base):
            parent = campaign / 'campaign_api_v1'
            locations = [campaign / f'baseline_integrity_repair_v2_20261017/main/{dataset}/formal/{method}',
                         parent / dataset / 'formal' / method]
            for directory in locations:
                for path in sorted(directory.glob('*/checkpoint.sqlite3')):
                    sources.append({'path': path, 'resource_file': parent / dataset / 'RESOURCE_FROZEN.json'})
    return sources


def _scope(identity, spec, method, dataset):
    """Verify ownership from saved metadata, with the pinned P02 protocol."""
    if identity.get('method') == method and identity.get('dataset') == dataset:
        return True, None
    protocol_path = spec.get('protocol_file')
    if protocol_path and Path(protocol_path).is_file():
        protocol = _read(protocol_path)
        if (identity.get('protocol_sha256') == _sha(protocol_path)
                and identity.get('task') == Path(protocol_path).parent.name
                and identity.get('dataset') == dataset
                and protocol.get('method') == method):
            return True, protocol
    return False, None


def _resources(identity, spec, protocol, dataset):
    if protocol is not None:
        return protocol['resources'][dataset]
    if 'resources' in identity:
        return identity['resources']
    path = spec.get('resource_file')
    if path and Path(path).is_file() and (identity.get('resource_sha256') or identity.get('resources_sha256')) == _sha(path):
        return _read(path)['resources']
    return None


def _packet_identity(packet):
    if packet.get('encoder_truncated') is not False:
        raise ValueError('retrieval_truncation_not_excluded')
    count = packet.get('query_tokens')
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError('retrieval_token_metadata_missing')
    if count > 512:
        windows = packet.get('encoded_window_tokens')
        content = packet.get('content_tokens')
        if (not isinstance(windows, list) or not windows
                or not all(type(n) is int and 0 < n <= 512 for n in windows)
                or type(content) is not int or content < 0
                or packet.get('processed_content_tokens') != content
                or packet.get('window_count') != len(windows) or len(windows) <= 1
                or sum(windows) < count):
            raise ValueError('retrieval_full_input_not_proven')
    docs = packet.get('docs', [])
    if len(docs) != packet.get('k') or len({d['doc_id'] for d in docs}) != len(docs):
        raise ValueError('retrieval_packet_incomplete')
    return {'qid': packet['qid'], 'query': packet['query'], 'k': packet['k'],
            'docs': [{k: d[k] for k in ('rank', 'doc_id', 'source', 'text', 'block')} for d in docs]}


def _binding_rows(db, path, qid, trace_cache):
    bindings = {}
    for kind, _, row in _records(db, ['evidence', 'token_call', 'call_binding', 'logical_call',
                                    'question_request_origin', 'request_origin'], qid=qid):
        if not all(row.get(k) for k in ('qid', 'request_id')):
            continue
        # A logical call may refer to another DB. Do not assign its id locally.
        source = (row.get('source') or row.get('origin_db')) if kind == 'logical_call' else None
        if source and not str(path).replace('\\', '/').endswith(str(source).replace('\\', '/')):
            continue
        bindings.setdefault(row['request_id'], set()).add((row['qid'], row.get('stage')))
    trace = Path(path).with_name('trace.jsonl')
    if trace.is_file():
        stamp = (trace.stat().st_size, trace.stat().st_mtime_ns)
        previous = trace_cache.get(str(trace))
        if previous is None or previous[0] != stamp:
            indexed = {}
            with trace.open(encoding='utf8') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # A final interrupted line does not invalidate DB bindings.
                        continue
                    if row.get('event') in {'logical_api_call', 'api_attempt'} and all(row.get(k) for k in ('qid', 'request_id', 'stage')):
                        indexed.setdefault(row['qid'], set()).add((row['request_id'], row['stage']))
            trace_cache[str(trace)] = (stamp, indexed)
        for rid, stage in trace_cache[str(trace)][1].get(qid, set()):
            bindings.setdefault(rid, set()).add((qid, stage))
    return bindings


def _expense(db, request_id):
    # Historical request tables do not reliably own provider billing. Unknown
    # absence is deliberately None, never zero or a newly imported liability.
    if 'charges' not in _tables(db):
        return None, None
    columns = {r[1] for r in db.execute('PRAGMA table_info(charges)')}
    if not {'id', 'actual', 'reserved'}.issubset(columns):
        return None, None
    row = db.execute('SELECT actual,reserved FROM charges WHERE id=?', (request_id,)).fetchone()
    return (row[0], row[1] if row[0] is None else None) if row else (None, None)


def _value_sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _verified_non_submission(db, path, request_id, attempt_id, resolver):
    """An explicit saved recovery receipt may release only its exact old row."""
    if resolver is None:
        return False
    request = db.execute('SELECT payload,status,result FROM requests WHERE id=?', (request_id,)).fetchone()
    if request is None:
        return False
    raw_result = json.loads(request['result']) if request['result'] else None
    attempts = []
    if 'attempts' in _tables(db):
        attempts = db.execute('SELECT id,status,result FROM attempts WHERE request_id=?', (request_id,)).fetchall()
    if attempt_id is None:
        if attempts or request['status'] not in UNRESOLVED:
            return False
        attempt = None
    else:
        attempt = next((row for row in attempts if row['id'] == attempt_id), None)
        if attempt is None or attempt['status'] not in UNRESOLVED:
            return False
    for overlay in resolver(str(path), request_id) or []:
        evidence = overlay.get('evidence', {})
        try:
            origin = Path(evidence.get('origin_db', '')).resolve()
        except (TypeError, OSError):
            continue
        if (origin != path or evidence.get('request_id') != request_id
                or evidence.get('attempt_id') != attempt_id
                or evidence.get('verified') is not True
                or evidence.get('outcome') != 'verified_not_submitted'
                or evidence.get('provider_confirmed_no_charge') is not True
                or not evidence.get('provider_receipt_id') or not evidence.get('source')
                or evidence.get('payload_sha256') != _value_sha(json.loads(request['payload']))
                or overlay.get('original_request_status') != request['status']
                or overlay.get('original_request_result_sha256') != _value_sha(raw_result)):
            continue
        if attempt is not None and (overlay.get('original_attempt_status') != attempt['status']
                or overlay.get('original_attempt_result_sha256') != _value_sha(json.loads(attempt['result']) if attempt['result'] else None)):
            continue
        return True
    return False


def _unresolved(db, path, request_id, status, result, resolver):
    all_attempts = []
    if 'attempts' in _tables(db):
        all_attempts = db.execute('SELECT id,status FROM attempts WHERE request_id=?', (request_id,)).fetchall()
    unresolved = [r for r in all_attempts if r['status'] in UNRESOLVED]
    if any(not _verified_non_submission(db, path, request_id, r['id'], resolver) for r in unresolved):
        return True
    if status in UNRESOLVED or result.get('status') in UNRESOLVED:
        if unresolved:
            return False  # Every unresolved attempt has its own exact receipt.
        if not all_attempts:
            return not _verified_non_submission(db, path, request_id, None, resolver)
        return True  # Inconsistent request summary still needs review.
    return False


def _without_cap(payload):
    return {k: v for k, v in payload.items() if k != 'max_tokens'}


def _attempt_history(db, path, request_id, payload, root, seen=None, resolver=None):
    """Actual persisted attempts, following mirrored origins without double count."""
    seen = set() if seen is None else seen
    key = (str(path), request_id)
    if key in seen:
        return [], [{'origin_db': str(path), 'request_id': request_id, 'reason': 'historical_origin_cycle'}]
    seen.add(key)
    records = []
    if 'attempts' in _tables(db):
        for row in db.execute('SELECT id,status FROM attempts WHERE request_id=? ORDER BY id', (request_id,)):
            identity = hashlib.sha256(_canonical([str(path), request_id, row[0]]).encode()).hexdigest()
            records.append({'identity': identity, 'origin_db': str(path), 'request_id': request_id,
                            'attempt_id': row[0], 'status': row[1],
                            'payload_sha256': hashlib.sha256(_canonical(payload).encode()).hexdigest(),
                            'equivalent_payload_except_output_cap': True})
    if records:
        return records, []
    origin = next((v for kind, rid, v in _records(db, ['request_origin']) if rid == request_id), None)
    if origin and origin.get('origin_db') and origin.get('origin_request_id'):
        target = Path(origin['origin_db'])
        target = target.resolve() if target.is_absolute() else (root / target).resolve()
        if target.is_file():
            with closing(_connect(target)) as other:
                raw = other.execute('SELECT payload FROM requests WHERE id=?', (origin['origin_request_id'],)).fetchone()
                if raw and _canonical(_without_cap(json.loads(raw[0]))) == _canonical(_without_cap(payload)):
                    return _attempt_history(other, target, origin['origin_request_id'], json.loads(raw[0]), root, seen, resolver)
    if _verified_non_submission(db, path, request_id, None, resolver):
        return [{'identity': _value_sha([str(path), request_id, None]), 'origin_db': str(path),
                 'request_id': request_id, 'attempt_id': None, 'status': 'verified_not_submitted',
                 'payload_sha256': _value_sha(payload), 'equivalent_payload_except_output_cap': True,
                 'count_basis': 'conservative_initial_ordinal_retained_after_verified_no_submission'}], []
    return [], [{'origin_db': str(path), 'request_id': request_id, 'reason': 'historical_actual_attempt_count_unverified'}]


def _external_unknowns(db, path, qid, stage, root, wanted, resolver):
    """A method-owned logical binding can point to a separate network DB."""
    blocks, attempts, unverified = [], [], []
    for _, _, call in _records(db, ['logical_call'], qid=qid):
        if call.get('stage') != stage or not call.get('source') or not call.get('request_id'):
            continue
        source = Path(call['source'])
        source = source.resolve() if source.is_absolute() else (root / source).resolve()
        if source == path:
            continue
        # The binding proves this method/query/stage uses that receipt. It does
        # not prove C1 eligibility; only unresolved submission detection uses it.
        if not source.is_file():
            if call.get('status') in UNRESOLVED:
                blocks.append({'origin_db': str(source), 'request_id': call['request_id'],
                    'qid': qid, 'stage': stage, 'status': 'historical_unknown',
                    'known_expense_cny': None, 'reserved_unknown_cny': None,
                    'reason': 'unresolved_bound_receipt_source_missing'})
            continue
        with closing(_connect(source)) as other:
            req = other.execute('SELECT payload,status,result FROM requests WHERE id=?', (call['request_id'],)).fetchone()
            if req is None:
                if call.get('status') in UNRESOLVED:
                    blocks.append({'origin_db': str(source), 'request_id': call['request_id'],
                        'qid': qid, 'stage': stage, 'status': 'historical_unknown',
                        'known_expense_cny': None, 'reserved_unknown_cny': None,
                        'reason': 'unresolved_bound_receipt_missing'})
                continue
            payload = json.loads(req['payload'])
            if call.get('payload_sha256') and call['payload_sha256'] != hashlib.sha256(_canonical(payload).encode()).hexdigest():
                raise ValueError('Historical logical binding payload hash differs from raw receipt')
            result = json.loads(req['result']) if req['result'] else {}
            if _canonical(_without_cap(payload)) == _canonical(_without_cap(wanted)):
                known, missing = _attempt_history(other, source, call['request_id'], payload, root, resolver=resolver)
                attempts.extend(known)
                unverified.extend(missing)
            if _unresolved(other, source, call['request_id'], req['status'], result, resolver):
                actual, reserved = _expense(other, call['request_id'])
                blocks.append({'origin_db': str(source), 'binding_db': str(path), 'request_id': call['request_id'],
                    'qid': qid, 'stage': stage, 'status': 'historical_unknown',
                    'known_expense_cny': actual, 'reserved_unknown_cny': reserved,
                    'reason': 'receipt_check_required_before_paid_replay'})
    return blocks, attempts, unverified


class ReadOnlyReuseCatalog:
    def __init__(self, sources=None, *, method, dataset, resources, store=None, root=None,
                 reconciliation_resolver=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        if sources is None:
            sources = discover_sources(self.root, method, dataset)
        self.sources = [{'path': Path(s)} if isinstance(s, (str, Path)) else dict(s) for s in sources]
        self.method, self.dataset, self.resources, self.store = method, dataset, deepcopy(resources), store
        self._trace_cache = {}
        if reconciliation_resolver is not None and not callable(reconciliation_resolver):
            raise TypeError('Historical reconciliation resolver must be callable')
        self.reconciliation_resolver = reconciliation_resolver
        self.retrievals = {}  # The new retriever must establish its own packets.
        self.hybrid = {}

    def lookup(self, qid, stage, payload, current_store=None):
        store = current_store if current_store is not None else self.store
        current_packets = []
        if store is not None:
            current_packets = [v for _, _, v in _records(store.db, ['retrieval'], qid=qid)]
        answer = {'candidate': None, 'blockers': [], 'rejected': [], 'historical_attempts': [],
                  'attempt_count_unverified': False, 'attempt_count_unverified_details': []}
        for spec in self.sources:
            path = Path(spec['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError('Historical source disappeared: ' + str(path))
            with closing(_connect(path)) as db:
                identity = _identity(db)
                scoped, protocol = _scope(identity, spec, self.method, self.dataset)
                if not scoped:
                    # A malformed supplied source must not silently clear history.
                    raise ValueError('Historical method/dataset identity is unverified: ' + str(path))
                if 'requests' not in _tables(db):
                    continue
                blocks, attempts, unverified = _external_unknowns(db, path, qid, stage, self.root, payload, self.reconciliation_resolver)
                answer['blockers'].extend(blocks)
                answer['historical_attempts'].extend(attempts)
                answer['attempt_count_unverified_details'].extend(unverified)
                binding = _binding_rows(db, path, qid, self._trace_cache)
                if not binding:
                    continue
                resources = _resources(identity, spec, protocol, self.dataset)
                old_packets = [v for _, _, v in _records(db, ['retrieval'], qid=qid)]
                for other in spec.get('retrieval_paths', []):
                    if not Path(other).is_file():
                        continue
                    with closing(_connect(other)) as rdb:
                        rid = _identity(rdb)
                        valid = protocol is not None and rid.get('protocol_sha256') == identity.get('protocol_sha256') and rid.get('dataset') == self.dataset
                        if not valid:
                            raise ValueError('Historical retrieval resource identity is unverified')
                        old_packets.extend(v for _, _, v in _records(rdb, ['retrieval'], qid=qid))
                marks = ','.join('?' for _ in binding)
                for req in db.execute('SELECT id,stage,payload,status,result FROM requests WHERE id IN (' + marks + ') ORDER BY rowid', tuple(binding)):
                    rid = req['id']
                    uses = {(q, s or req['stage']) for q, s in binding.get(rid, set())}
                    if (qid, stage) not in uses:
                        continue
                    old_payload = json.loads(req['payload'])
                    result = json.loads(req['result']) if req['result'] else {}
                    if _canonical(_without_cap(old_payload)) == _canonical(_without_cap(payload)):
                        attempts, missing = _attempt_history(db, path, rid, old_payload, self.root, resolver=self.reconciliation_resolver)
                        answer['historical_attempts'].extend(attempts)
                        answer['attempt_count_unverified_details'].extend(missing)
                    unresolved = _unresolved(db, path, rid, req['status'], result, self.reconciliation_resolver)
                    if unresolved:
                        actual, reserved = _expense(db, rid)
                        answer['blockers'].append({'origin_db': str(path), 'request_id': rid,
                            'qid': qid, 'stage': stage, 'status': 'historical_unknown',
                            'known_expense_cny': actual, 'reserved_unknown_cny': reserved,
                            'reason': 'receipt_check_required_before_paid_replay',
                            'same_payload_except_output_cap': _canonical({k:v for k,v in old_payload.items() if k != 'max_tokens'}) == _canonical({k:v for k,v in payload.items() if k != 'max_tokens'})})
                        continue
                    reason = self._candidate_error(payload, old_payload, req['status'], result, resources, current_packets, old_packets, store)
                    if reason:
                        answer['rejected'].append({'origin_db': str(path), 'request_id': rid, 'reason': reason})
                        continue
                    if answer['candidate'] is None:
                        answer['candidate'] = {'request_id': rid, 'payload': old_payload, 'result': result,
                            'status': 'success', 'origin_db': str(path),
                            'c1_validation': {'method': self.method, 'dataset': self.dataset, 'qid': qid, 'stage': stage,
                                'full_messages_settings_equal_except_nonbinding_output_cap': True,
                                'complete_stop_usage_model_verified': True, 'resources_verified': True,
                                'current_retrieval_packets_verified': len(current_packets),
                                'source_meta_sha256': hashlib.sha256(_canonical(identity).encode()).hexdigest(),
                                'selection': 'first eligible source in fixed source order; no gold or score inspected'}}
        # An independently successful receipt does not erase another unknown bill.
        if answer['blockers']:
            answer['candidate'] = None
        answer['historical_attempts'] = list({row['identity']: row for row in answer['historical_attempts']}.values())
        answer['attempt_count_unverified_details'] = list({_canonical(row): row for row in answer['attempt_count_unverified_details']}.values())
        answer['attempt_count_unverified'] = bool(answer['attempt_count_unverified_details'])
        if answer['candidate'] is None and answer['attempt_count_unverified']:
            answer['blockers'].extend({**row, 'status': 'historical_attempt_count_unverified',
                'qid': qid, 'stage': stage, 'known_expense_cny': None, 'reserved_unknown_cny': None}
                for row in answer['attempt_count_unverified_details'])
        return answer

    def _candidate_error(self, wanted, old, status, result, resources, current, historical, store):
        if _canonical({k:v for k,v in wanted.items() if k != 'max_tokens'}) != _canonical({k:v for k,v in old.items() if k != 'max_tokens'}):
            return 'complete_payload_changed'
        if status != 'success' or result.get('status') != 'success' or result.get('finish_reason') != 'stop':
            return 'not_complete_success'
        usage = result.get('usage')
        if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) and not isinstance(usage[k], bool) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
            return 'missing_or_invalid_usage'
        if result.get('returned_model') != wanted.get('model') or not isinstance(result.get('answer'), str):
            return 'returned_model_or_answer_unverified'
        cap = old.get('max_tokens')
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= usage['completion_tokens']:
            return 'historical_output_cap_not_proven_nonbinding'
        if resources is None:
            return 'source_resources_unverified'
        try:
            if _canonical(_resource_identity(resources)) != _canonical(_resource_identity(self.resources)):
                return 'retrieval_resources_changed'
        except (KeyError, TypeError):
            return 'source_resource_identity_incomplete'
        if store is None:
            return 'current_store_missing'
        try:
            old_packets = {_canonical(_packet_identity(p)) for p in historical}
            if any(_canonical(_packet_identity(p)) not in old_packets for p in current):
                return 'current_retrieval_differs_or_source_packet_missing'
        except (ValueError, KeyError, TypeError):
            return 'retrieval_complete_input_unverified'
        return None
