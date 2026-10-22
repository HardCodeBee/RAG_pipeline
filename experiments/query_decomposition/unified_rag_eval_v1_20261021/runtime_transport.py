"""Guarded request path; no inherited SDK/controller retry loop is used.

One ask invocation submits at most once. Confirmed failures may be retried by
the runner, with a durable five-attempt counter shared by every restart. Unknown
or in-flight submissions are held without any automatic replay.
"""
from __future__ import annotations

import json
import math
import re
import time

from experiments.query_decomposition.baseline_origin_binding_fix_20261017 import OriginSafeTransport
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import (
    MODEL, MAX_INPUT, MAX_OUTPUT, CONTEXT, IntegrityError, canonical, digest, reserve_cost,
)
from .runtime_ledger import RuntimeStageHeld, valid_usage


class RuntimeTransport(OriginSafeTransport):
    def __init__(self, store, client, ledger, model=MODEL, catalog=None, *, guard, dataset, method,
                 sleep=time.sleep, clock=time.time):
        # Deliberately do not initialize the old bounded Budget/Transport path.
        self.store, self.client, self.ledger = store, client, ledger
        self.budget = ledger  # Existing adapters only inspect this attribute.
        self.model, self.catalog, self.guard = model, catalog, guard
        self.dataset, self.method, self.qid = dataset, method, None
        self.sleep, self.clock = sleep, clock
        self.new_requests, self.last_start, self.last_call = 0, 0., None
        if model != MODEL:
            raise IntegrityError('Frozen reader model cannot change in this runtime')
        if not callable(guard):
            raise TypeError('A live execution guard is required')
        if catalog is not None:
            from .runtime_reuse import ReadOnlyReuseCatalog
            if not isinstance(catalog, ReadOnlyReuseCatalog):
                raise TypeError('Historical reuse requires ReadOnlyReuseCatalog C1 validation')
            if catalog.method != method or catalog.dataset != dataset:
                raise IntegrityError('Historical catalog scope differs from the runtime method/dataset')

    def _guard(self):
        if self.guard() is False:
            raise RuntimeStageHeld('Execution guard denied submission')

    def request_id_for(self, stage, prompt, max_tokens=None):
        """Stable pre-dispatch ID for a separately persisted controller receipt."""
        self.ledger.stage_key(self.dataset,self.method,self.qid,stage)
        payload = {'model':self.model,'messages':[{'role':'user','content':prompt}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        return digest([self.store.namespace,self.dataset,self.method,self.qid,stage,payload])

    @staticmethod
    def _validate_result(result, model, stage=''):
        if result.get('status') != 'success':
            return result
        if not valid_usage(result.get('usage')):
            result['status'] = 'unknown'
            result['hold_reason'] = 'missing_or_invalid_usage'
        elif result.get('returned_model') != model:
            result['status'] = 'model_identity_mismatch'
        elif (result['usage']['prompt_tokens'] > MAX_INPUT or
              result['usage']['prompt_tokens'] + MAX_OUTPUT > CONTEXT or
              result['usage']['prompt_tokens'] + result['usage']['completion_tokens'] > CONTEXT or
              result['usage']['completion_tokens'] > MAX_OUTPUT):
            result['status'] = 'input_token_limit_violation'
        elif stage.startswith('qa_raw/') and stage.endswith('/final') and (
                (result.get('finish_reason') == 'stop' and not result.get('answer','').strip()) or
                (result.get('finish_reason') == 'length' and result['usage']['completion_tokens'] == MAX_OUTPUT)):
            result['status'] = 'algorithm_failure'
            result['algorithm_failure_reason'] = 'empty_stop_or_full_cap_final_length'
        elif result.get('finish_reason') != 'stop':
            result['status'] = 'integrity_failure'
            result['hold_reason'] = 'incomplete_generation'
        return result

    def _mirror(self, request_id, stage, payload, result, *, origin, native_maximum, attempt_id=None, ordinal=None):
        db = self.store.db
        db.execute('INSERT OR REPLACE INTO requests VALUES(?,?,?,?,?)',
                   (request_id,stage,canonical(payload),result['status'],canonical(result)))
        db.commit()
        self.store.put('runtime_binding', request_id, {
            'dataset':self.dataset,'method':self.method,'qid':self.qid,'stage':stage,
            'ledger_path':str(self.ledger.path.resolve()),'response_origin':origin})
        if attempt_id is not None:
            self.store.put('token_attempt', str(attempt_id), {
                'qid':self.qid,'stage':stage,'request_id':request_id,'attempt_id':attempt_id,
                'attempt_ordinal':ordinal,'status':result['status'],'usage':result.get('usage'),
                'finish_reason':result.get('finish_reason'),
                'usage_missing_is_unknown_not_zero':not valid_usage(result.get('usage'))})
        if result['status'] in {'success','algorithm_failure'} and valid_usage(result.get('usage')):
            prompt = payload['messages'][0]['content']
            self.store.put('token_call',digest([self.qid,stage,request_id]),{
                'qid':self.qid,'stage':stage,'request_id':request_id,
                'native_requested_max_tokens':native_maximum,'allowed_max_output_tokens':MAX_OUTPUT,
                'actual_request_max_tokens':payload['max_tokens'],'prompt_utf8_bytes':len(prompt.encode('utf8')),
                'preflight_token_upper_bound':len(prompt.encode('utf8'))+256,
                'usage':result['usage'],'token_count_source':'provider_usage','response_origin':origin,
                'finish_reason':result['finish_reason'],'status':result['status']})
        self.store.export()

    def _evidence(self, stage, request_id, prompt):
        shown = set()
        for raw, in self.store.db.execute("SELECT value FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid') AND json_extract(value,'$.qid')=?", (self.qid,)):
            for doc in json.loads(raw).get('docs',[]):
                if doc.get('text') and doc['text'] in prompt:
                    shown.add(doc['doc_id'])
        self.store.put('evidence',digest([self.qid,request_id,stage]),{
            'qid':self.qid,'stage':stage,'request_id':request_id,'doc_ids':sorted(shown),
            'source_match':'exact raw corpus text present in this call; generated answers are not source evidence'})

    @staticmethod
    def _exception_result(exc, elapsed):
        code = getattr(exc,'status_code',None)
        body = getattr(exc,'body',None)
        if not isinstance(body,dict):
            body = {}
        error = body.get('error',body)
        error = error if isinstance(error,dict) else {}
        error_code = error.get('code')
        if not isinstance(error_code,str) or not re.fullmatch(r'[\w.-]{1,100}',error_code):
            error_code = None
        usage = body.get('usage',error.get('usage'))
        confirmed_rejection = code in {400,401,403,404,409,422,429}
        result = {'status':'failed' if confirmed_rejection else 'unknown',
                  'http_status':code,'failure_class':type(exc).__name__,'error_code':error_code,
                  'latency_seconds':elapsed,'usage':usage if valid_usage(usage) else None}
        if error_code in {'data_inspection_failed','DataInspectionFailed'} and code in {None,400,422}:
            result['status'] = 'provider_content_rejected'
            result['retryable'] = False
        headers = getattr(getattr(exc,'response',None),'headers',{}) or {}
        try:
            result['retry_after_seconds'] = max(0.,float(headers.get('retry-after'))) if headers.get('retry-after') else None
        except (TypeError,ValueError):
            result['retry_after_seconds'] = None
        if result['retry_after_seconds'] is not None and not math.isfinite(result['retry_after_seconds']):
            result['retry_after_seconds'] = None
        # A content rejection may be charged; without usage its reserve remains.
        unbilled = code in {400,401,403,404,422,429} and result['status'] != 'provider_content_rejected'
        return result, unbilled

    def ask(self, stage, prompt, max_tokens):
        self._guard()
        self.ledger.stage_key(self.dataset,self.method,self.qid,stage)
        if not isinstance(prompt,str) or not prompt.strip():
            raise ValueError('Empty prompt')
        byte_upper = len(prompt.encode('utf8')) + 256
        if byte_upper > MAX_INPUT or byte_upper + MAX_OUTPUT > CONTEXT:
            raise IntegrityError('Full input exceeds context; no truncation or submission')
        payload = {'model':self.model,'messages':[{'role':'user','content':prompt}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        request_id = self.request_id_for(stage,prompt,max_tokens)
        existing = self.ledger.lookup(self.dataset,self.method,self.qid,stage,payload)
        inspection = {}
        if not existing and self.catalog is not None:
            inspection = self.catalog.lookup(self.qid,stage,payload,self.store)
            self.ledger.register_attempt_history(self.dataset,self.method,self.qid,stage,
                                                  inspection.get('historical_attempts',[]))
            if inspection.get('blockers'):
                self.store.put('historical_stage_hold',digest([self.qid,stage,payload,inspection['blockers']]),{
                    'dataset':self.dataset,'method':self.method,'qid':self.qid,'stage':stage,
                    'blockers':inspection['blockers'],'old_liabilities_remain_in_original_ledger':True})
                raise RuntimeStageHeld('Historical unknown request requires receipt verification for this stage')
            candidate = inspection.get('candidate')
            if candidate:
                result = self._validate_result(dict(candidate['result']),self.model,stage)
                if result['status'] != 'success' or not candidate.get('c1_validation'):
                    raise IntegrityError('C1 catalog supplied an incomplete or unverified response')
                existing = self.ledger.install_reuse(self.dataset,self.method,self.qid,stage,payload,request_id,candidate)
        if existing:
            request_id = existing['request_id']
            status = existing['status']
            if status in {'unknown','in_flight'}:
                raise RuntimeStageHeld('Unknown submission held for receipt verification; no replay')
            if status not in {'success','failed','provider_content_rejected','algorithm_failure'}:
                raise RuntimeStageHeld('Stage paused for integrity review: ' + status)
            if status in {'success','provider_content_rejected','algorithm_failure'}:
                result = existing['result']
                self._mirror(request_id,stage,existing['payload'],result,origin=existing['origin'],native_maximum=max_tokens)
                self._evidence(stage,request_id,prompt)
                self.last_call = {'request_id':request_id,'new_submission':False,'response_origin':existing['origin']}
                return request_id, result
        if inspection.get('attempt_count_unverified'):
            self.store.put('historical_attempt_count_hold',digest([self.qid,stage,payload,inspection.get('attempt_count_unverified_details',[])]),{
                'dataset':self.dataset,'method':self.method,'qid':self.qid,'stage':stage,
                'details':inspection.get('attempt_count_unverified_details',[])})
            raise RuntimeStageHeld('Historical actual-attempt count needs verification before a new submission')
        # Both the SDK and HTTP adapter must have hidden retries disabled by the
        # client factory. One reservation and in-flight row precede the wire.
        retries = getattr(self.client,'max_retries',0)
        if retries != 0:
            raise IntegrityError('SDK hidden retries must be disabled')
        self.ledger.wait_for_start(self._guard,sleep=self.sleep,clock=self.clock)
        self._guard()
        attempt_id, ordinal = self.ledger.begin_attempt(self.dataset,self.method,self.qid,stage,payload,
                                                       request_id,reserve_cost(byte_upper,MAX_OUTPUT))
        self.last_start = time.monotonic()
        started = time.perf_counter()
        confirmed_unbilled = False
        raw = None
        try:
            response = self.client.chat.completions.create(**payload)
            raw = response.model_dump(mode='json')
            choice = response.choices[0]
            result = {'status':'success','answer':choice.message.content or '',
                      'finish_reason':choice.finish_reason,'refusal':getattr(choice.message,'refusal',None),
                      'returned_model':response.model,'usage':raw.get('usage'),
                      'raw_response':raw,'latency_seconds':time.perf_counter()-started}
            result = self._validate_result(result,self.model,stage)
        except Exception as exc:
            result, confirmed_unbilled = self._exception_result(exc,time.perf_counter()-started)
            if isinstance(raw,dict) and valid_usage(raw.get('usage')):
                # A malformed response can still provide genuine billed usage.
                result['usage'] = raw['usage']
                result['raw_response'] = raw
                result['returned_model'] = raw.get('model')
        # BaseException (process interrupt/crash) intentionally leaves in_flight.
        self.ledger.finish_attempt(attempt_id,result,confirmed_unbilled=confirmed_unbilled)
        self.new_requests += 1
        self.last_call = {'request_id':request_id,'new_submission':True,'attempt_id':attempt_id,
                          'attempt_ordinal':ordinal,'response_origin':'new_runtime'}
        self._mirror(request_id,stage,payload,result,origin='new_runtime',native_maximum=max_tokens,
                     attempt_id=attempt_id,ordinal=ordinal)
        self._evidence(stage,request_id,prompt)
        self.store.event('runtime_attempt',qid=self.qid,stage=stage,request_id=request_id,
                         attempt=ordinal,status=result['status'])
        if result['status'] in {'unknown','model_identity_mismatch','input_token_limit_violation','integrity_failure'}:
            raise RuntimeStageHeld('Submission held: ' + result['status'])
        return request_id,result
