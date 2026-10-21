"""Completion guards, truthful per-call tokens, exact private request ownership."""
from threading import Lock
import json
from .common import OUT, MAX_OUTPUT, MAX_INPUT, CONTEXT, PROTECTED_CAP, IntegrityError, digest, canonical, usage_cost
from .transport_base import RawTransport
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scan_api_epoch import CapacityAPI, CapacityBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_rate_policy import RATE_CODES


class Budget(CapacityBudget):
    def reserve(self, request_id, namespace, amount):
        original_cap = self.cap
        self.cap = min(self.cap, PROTECTED_CAP)
        try:
            # Parent reserves under BEGIN IMMEDIATE: concurrent workers cannot overrun it.
            return super().reserve(request_id, namespace, amount)
        finally:
            self.cap = original_cap


class Transport(RawTransport):
    def __init__(self, store, client, budget, model, catalog):
        super().__init__(store, client, budget, model)
        self.catalog = catalog

    def ask(self, stage, prompt, max_tokens):
        if (OUT / 'STOP_REQUESTED').exists():
            raise IntegrityError('User stop requested; no further calls dispatched')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Empty prompt')
        byte_upper = len(prompt.encode('utf8')) + 256
        if byte_upper > MAX_INPUT or byte_upper + MAX_OUTPUT > CONTEXT:
            self.store.put('integrity_violation', digest([self.qid, stage, prompt]), {
                'qid': self.qid, 'stage': stage, 'reason': 'full_input_exceeds_model_context',
                'prompt_utf8_byte_upper': byte_upper, 'input_was_truncated': False})
            raise IntegrityError('Input exceeds verified context; nothing silently discarded')
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0., 'max_tokens': MAX_OUTPUT, 'extra_body': {'enable_thinking': False}}
        desired = digest([self.store.namespace, payload])
        existing = self.store.db.execute('select status,result from requests where id=?', (desired,)).fetchone()
        origin = self.store.get('request_origin', desired)
        if existing:
            if existing[0] in {'in_flight', 'unknown'}:
                result = json.loads(existing[1]) if existing[1] else {'status': 'unknown'}
                key = desired
            else:
                key, result = desired, json.loads(existing[1])
            actual_payload = payload
            mode = 'new_v2_response'
        else:
            inherited = self.catalog.request(self.qid, stage, payload)
            if inherited:
                key = digest([self.store.namespace, 'preserved_old_response', inherited['origin_db'], inherited['request_id']])
                result, actual_payload = inherited['result'], inherited['payload']
                self.store.db.execute('insert or ignore into requests values(?,?,?,?,?)',
                    (key, stage, canonical(actual_payload), inherited['status'], canonical(result)))
                self.store.db.commit()
                origin = {'qid': self.qid, 'origin_db': inherited['origin_db'],
                    'origin_request_id': inherited['request_id'], 'mode': 'complete_historical_prefix'}
                self.store.put('request_origin', key, origin)
                mode = 'complete_historical_prefix' if result['status']=='success' else 'preserved_historical_failure'
            else:
                before = self.store.db.execute('select coalesce(max(id),0) from attempts').fetchone()[0]
                try:
                    key, result = super().ask(stage, prompt, MAX_OUTPUT)
                finally:
                    # Actual retry usage is retained even when a transport guard stops the owner.
                    for aid, rid, raw in self.store.db.execute('select id,request_id,result from attempts where id>? and request_id=?', (before, desired)):
                        if not raw:
                            continue
                        attempt = json.loads(raw)
                        self.store.put('token_attempt', str(aid), {'qid': self.qid, 'stage': stage,
                            'request_id': rid, 'attempt_id': aid, 'status': attempt['status'],
                            'usage': attempt.get('usage'), 'finish_reason': attempt.get('finish_reason'),
                            'usage_missing_is_unknown_not_zero': not bool(attempt.get('usage'))})
                        if attempt.get('http_status') == 429 and attempt.get('error_code') in RATE_CODES:
                            self.budget.note_throttle()
                mode, actual_payload = 'new_v2_response', payload
        usage = result.get('usage')
        record = {'qid': self.qid, 'stage': stage, 'request_id': key,
            'desired_request_id': desired, 'native_requested_max_tokens': max_tokens,
            'allowed_max_output_tokens': MAX_OUTPUT, 'actual_request_max_tokens': actual_payload['max_tokens'],
            'prompt_utf8_bytes': len(prompt.encode('utf8')), 'preflight_token_upper_bound': byte_upper,
            'usage': usage, 'token_count_source': 'provider_usage' if usage else 'unavailable',
            'response_origin': mode, 'finish_reason': result.get('finish_reason'), 'status': result['status']}
        self.store.put('token_call', digest([self.qid, stage, key]), record)
        self.store.event('logical_api_call', qid=self.qid, stage=stage, request_id=key,
                         cache_hit=mode=='complete_historical_prefix' or bool(existing), max_tokens=actual_payload['max_tokens'])
        shown = set()
        for value, in self.store.db.execute("select value from data where kind in ('retrieval','sparse_retrieval','hybrid') and json_extract(value,'$.qid')=?", (self.qid,)):
            for doc in json.loads(value).get('docs', []):
                if doc.get('text') and doc['text'] in prompt:
                    shown.add(doc['doc_id'])
        self.store.put('evidence', digest([self.qid, key, stage]), {'qid': self.qid, 'stage': stage,
            'request_id': key, 'doc_ids': sorted(shown),
            'source_match': 'exact raw corpus text present in this call; generated answers are not source evidence'})
        if result['status'] == 'success':
            if result.get('finish_reason') != 'stop' or not usage:
                self.store.put('integrity_violation', digest([self.qid, stage, key]), {
                    'qid': self.qid, 'stage': stage, 'request_id': key,
                    'reason': 'incomplete_output_or_missing_usage', 'finish_reason': result.get('finish_reason')})
                raise IntegrityError('Incomplete generation or missing usage; cannot score or propagate it')
            if usage.get('prompt_tokens', 0) > MAX_INPUT or usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0) > CONTEXT:
                raise IntegrityError('Provider returned context usage beyond verified model limits')
        return key, result


class API(CapacityAPI):
    def _ask(self, qid, args, kwargs):
        if kwargs:
            raise ValueError('Unexpected transport arguments')
        stage, prompt, maximum = args
        key = digest([self.transport.store.namespace, self.transport.model, prompt, MAX_OUTPUT])
        with self.keys_lock:
            mutex = self.keys.setdefault(key, Lock())
        with mutex:
            store = WorkerStore(self.transport.store)
            budget = Budget(self.transport.budget.path, self.transport.budget.cap, self.interval)
            try:
                transport = Transport(store, self.transport.client, budget, self.transport.model, self.transport.catalog)
                transport.qid = qid
                return transport.ask(stage, prompt, maximum)
            finally:
                store.db.close()
                budget.db.close()
