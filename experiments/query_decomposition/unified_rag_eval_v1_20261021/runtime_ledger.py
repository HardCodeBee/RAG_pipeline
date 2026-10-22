"""Durable attempt ownership and uncapped accounting for the new runtime only.

This database is authoritative for wire attempts. Historical billing databases
are never opened by this writer. A committed in-flight attempt is a liability,
including after a crash; only an independently checked receipt may resolve it.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import (
    IntegrityError, MAX_INPUT, MAX_OUTPUT, CONTEXT, canonical, digest, usage_cost, utc,
)
from .execution_policy import COST_WINDOW, MAX_FAILED_ATTEMPTS, expense_signals


class RuntimeStageHeld(IntegrityError):
    """The current logical stage is pending, not an algorithm-failure zero."""


def valid_usage(value):
    return (isinstance(value, dict) and all(
        isinstance(value.get(k), int) and not isinstance(value[k], bool) and value[k] >= 0
        for k in ('prompt_tokens', 'completion_tokens')))


class RuntimeLedger:
    def __init__(self, path, *, reference_means=None, notify=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and 'rt_stages' not in tables:
            self.db.close()
            raise IntegrityError('Refusing to mutate a historical or unrelated ledger')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS rt_stages(
            key TEXT PRIMARY KEY,dataset TEXT,method TEXT,qid TEXT,stage TEXT,
            payload TEXT,request_id TEXT UNIQUE,status TEXT,result TEXT,origin TEXT);
          CREATE TABLE IF NOT EXISTS rt_attempts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT,ordinal INTEGER,started TEXT,
            status TEXT,reserved REAL,actual REAL,usage TEXT,result TEXT,
            UNIQUE(key,ordinal));
          CREATE TABLE IF NOT EXISTS rt_alerts(
            id TEXT PRIMARY KEY,created TEXT,payload TEXT,delivered INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS rt_completions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,dataset TEXT,method TEXT,qid TEXT,
            cost REAL,eligible INTEGER,UNIQUE(dataset,method,qid));
          CREATE TABLE IF NOT EXISTS rt_rolling_state(
            dataset TEXT,method TEXT,elevated INTEGER,PRIMARY KEY(dataset,method));
          CREATE TABLE IF NOT EXISTS rt_provenance(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE IF NOT EXISTS rt_scheduler(key TEXT PRIMARY KEY,value REAL);
          CREATE TABLE IF NOT EXISTS rt_reconciliations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,attempt_id INTEGER,created TEXT,
            previous_status TEXT,previous_result TEXT,evidence TEXT,result TEXT);
          CREATE TABLE IF NOT EXISTS rt_historical_attempts(
            key TEXT,identity TEXT,evidence TEXT,PRIMARY KEY(key,identity));
          CREATE TABLE IF NOT EXISTS rt_historical_reconciliations(
            identity TEXT PRIMARY KEY,origin_db TEXT,request_id TEXT,created TEXT,value TEXT);
        ''')
        self.reference_means = reference_means or {}
        self.notify = notify

    def wait_for_start(self, guard, *, sleep=time.sleep, clock=time.time):
        """Keep the existing 2/3-second start interval, shared across restarts."""
        guard()
        now = clock()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute("SELECT value FROM rt_scheduler WHERE key='next_start'").fetchone()
            start = max(now,row[0] if row else now)
            self.db.execute("INSERT OR REPLACE INTO rt_scheduler VALUES('next_start',?)",(start+2/3,))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        remaining = start-now
        while remaining > 0:
            guard()
            step = min(30.,remaining)
            sleep(step)
            remaining -= step
        guard()

    @staticmethod
    def stage_key(dataset, method, qid, stage):
        if not all(isinstance(v, str) and v for v in (dataset, method, qid, stage)):
            raise ValueError('dataset, method, qid and stage must be nonempty strings')
        return digest([dataset, method, qid, stage])

    def receipt(self, request_id):
        row = self.db.execute('SELECT * FROM rt_stages WHERE request_id=?', (request_id,)).fetchone()
        if not row:
            return None
        value = dict(zip(('key','dataset','method','qid','stage','payload','request_id','status','result','origin'), row))
        value['payload'] = json.loads(value['payload'])
        value['result'] = json.loads(value['result']) if value['result'] else None
        value['ledger_path'] = str(self.path.resolve())
        provenance = self.db.execute('SELECT value FROM rt_provenance WHERE key=?',(value['key'],)).fetchone()
        if provenance:
            value['provenance'] = json.loads(provenance[0])
            value['desired_payload'] = value['payload']
            value['payload'] = value['provenance'].get('actual_payload',value['payload'])
        return value

    def install_reuse(self, dataset, method, qid, stage, payload, request_id, candidate):
        """Accept only the transport's already-validated read-only C1 candidate."""
        key = self.stage_key(dataset,method,qid,stage)
        result = candidate['result']
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.lookup(dataset,method,qid,stage,payload):
                raise IntegrityError('Cannot overwrite existing runtime stage with a historical candidate')
            self.db.execute('INSERT INTO rt_stages VALUES(?,?,?,?,?,?,?,?,?,?)',
                (key,dataset,method,qid,stage,canonical(payload),request_id,result['status'],canonical(result),'c1_historical'))
            provenance = {k:v for k,v in candidate.items() if k not in {'result','payload'}}
            provenance['actual_payload'] = candidate['payload']
            self.db.execute('INSERT INTO rt_provenance VALUES(?,?)',(key,canonical(provenance)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return self.receipt(request_id)

    def lookup(self, dataset, method, qid, stage, payload):
        key = self.stage_key(dataset, method, qid, stage)
        row = self.db.execute('SELECT request_id,payload FROM rt_stages WHERE key=?', (key,)).fetchone()
        if not row:
            return None
        if row[1] != canonical(payload):
            raise IntegrityError('Logical stage payload changed; cannot reset its attempts or reuse downstream')
        return self.receipt(row[0])

    def register_attempt_history(self, dataset, method, qid, stage, attempts):
        """Persist read-only catalog evidence; do not import or settle old fees."""
        key = self.stage_key(dataset,method,qid,stage)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for item in attempts:
                identity = item.get('identity')
                if not isinstance(identity,str) or not identity:
                    raise IntegrityError('Historical actual attempt lacks a stable identity')
                old = self.db.execute('SELECT evidence FROM rt_historical_attempts WHERE key=? AND identity=?',
                                      (key,identity)).fetchone()
                if old and old[0] != canonical(item):
                    raise IntegrityError('Historical attempt evidence changed; review its receipt first')
                self.db.execute('INSERT OR IGNORE INTO rt_historical_attempts VALUES(?,?,?)',
                                (key,identity,canonical(item)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def record_historical_reconciliation(self, evidence):
        """Record a checked no-submission/no-charge receipt in a new overlay.

        The source database and its reservations remain byte-preserved. Each
        overlay resolves one identified attempt only; catalogs must recheck the
        original snapshot before letting that resolution affect dispatch.
        """
        if (not isinstance(evidence,dict) or evidence.get('verified') is not True or
            evidence.get('outcome') != 'verified_not_submitted' or
            evidence.get('provider_confirmed_no_charge') is not True or
            not all(evidence.get(k) for k in ('origin_db','request_id','payload_sha256','source','provider_receipt_id')) or
            'attempt_id' not in evidence):
            raise IntegrityError('Historical recovery requires a specific verified provider no-submission receipt')
        path = Path(evidence['origin_db']).resolve()
        if path == self.path.resolve() or not path.is_file():
            raise IntegrityError('Historical receipt source must be a distinct existing database')
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as source:
            source.execute('PRAGMA query_only=ON')
            row = source.execute('SELECT payload,status,result FROM requests WHERE id=?',
                                 (evidence['request_id'],)).fetchone()
            if row is None or digest(json.loads(row[0])) != evidence['payload_sha256']:
                raise IntegrityError('Historical receipt payload/request identity differs from source')
            tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            attempts = source.execute('SELECT id,status,result FROM attempts WHERE request_id=?',
                                      (evidence['request_id'],)).fetchall() if 'attempts' in tables else []
            attempt_id = evidence['attempt_id']
            snapshot = {'original_request_status':row[1],
                        'original_request_result_sha256':digest(json.loads(row[2]) if row[2] else None)}
            if attempt_id is None:
                if attempts or row[1] not in {'unknown','in_flight'}:
                    raise IntegrityError('Request-level recovery requires an unresolved source with no attempt records')
            else:
                if not isinstance(attempt_id,int) or isinstance(attempt_id,bool):
                    raise IntegrityError('Historical attempt identity must be an integer or explicit null')
                matched = next((r for r in attempts if r[0] == attempt_id),None)
                if matched is None or matched[1] not in {'unknown','in_flight'}:
                    raise IntegrityError('Specified historical attempt is not unresolved in the source')
                snapshot.update(original_attempt_status=matched[1],
                    original_attempt_result_sha256=digest(json.loads(matched[2]) if matched[2] else None))
        evidence = {**evidence,'origin_db':str(path)}
        identity = digest([str(path),evidence['request_id'],attempt_id])
        value = {'identity':identity,'evidence':evidence,**snapshot,
                 'original_liability':'unchanged_in_original_database',
                 'overlay_known_CNY':0.,'attempt_counter_reset':False}
        self.db.execute('BEGIN IMMEDIATE')
        try:
            previous = self.db.execute('SELECT value FROM rt_historical_reconciliations WHERE identity=?',(identity,)).fetchone()
            if previous and previous[0] != canonical(value):
                raise IntegrityError('Historical reconciliation evidence changed; preserve the earlier audit')
            self.db.execute('INSERT OR IGNORE INTO rt_historical_reconciliations VALUES(?,?,?,?,?)',
                            (identity,str(path),evidence['request_id'],utc(),canonical(value)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return value

    def historical_reconciliations(self, origin_db, request_id):
        """Read-only catalog resolver; no receipt is inferred from an absent row."""
        return [json.loads(r[0]) for r in self.db.execute(
            'SELECT value FROM rt_historical_reconciliations WHERE origin_db=? AND request_id=? ORDER BY rowid',
            (str(Path(origin_db).resolve()),request_id))]

    def begin_attempt(self, dataset, method, qid, stage, payload, request_id, reserve):
        key = self.stage_key(dataset, method, qid, stage)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.lookup(dataset, method, qid, stage, payload)
            if row:
                if row['status'] in {'unknown','in_flight','success','algorithm_failure',
                                     'provider_content_rejected','model_identity_mismatch',
                                     'input_token_limit_violation','integrity_failure'}:
                    raise RuntimeStageHeld('Logical stage cannot be resubmitted: ' + row['status'])
            count = self.db.execute('SELECT count(*) FROM rt_attempts WHERE key=?', (key,)).fetchone()[0]
            count += self.db.execute('SELECT count(*) FROM rt_historical_attempts WHERE key=?', (key,)).fetchone()[0]
            if count >= MAX_FAILED_ATTEMPTS:
                raise RuntimeStageHeld('Five actual attempts exhausted; temporarily skip this stage')
            self.db.execute('INSERT OR IGNORE INTO rt_stages VALUES(?,?,?,?,?,?,?,?,?,?)',
                (key,dataset,method,qid,stage,canonical(payload),request_id,'in_flight',None,'new_runtime'))
            self.db.execute("UPDATE rt_stages SET status='in_flight',result=NULL WHERE key=?", (key,))
            cursor = self.db.execute('INSERT INTO rt_attempts(key,ordinal,started,status,reserved) VALUES(?,?,?,?,?)',
                (key,count+1,utc(),'in_flight',reserve))
            self.db.commit()
            return cursor.lastrowid, count + 1
        except BaseException:
            self.db.rollback()
            raise

    def finish_attempt(self, attempt_id, result, *, confirmed_unbilled=False):
        usage = result.get('usage')
        actual = usage_cost(usage['prompt_tokens'], usage['completion_tokens']) if valid_usage(usage) else (0. if confirmed_unbilled else None)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT key,status FROM rt_attempts WHERE id=?', (attempt_id,)).fetchone()
            if not row or row[1] != 'in_flight':
                raise IntegrityError('Attempt settlement is missing or already final')
            self.db.execute('UPDATE rt_attempts SET status=?,actual=?,usage=?,result=? WHERE id=?',
                (result['status'],actual,canonical(usage) if valid_usage(usage) else None,canonical(result),attempt_id))
            self.db.execute('UPDATE rt_stages SET status=?,result=? WHERE key=?',
                (result['status'],canonical(result),row[0]))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        owner = self.db.execute('SELECT dataset,method,qid FROM rt_stages WHERE key=?', (row[0],)).fetchone()
        self.check_query_alerts(*owner)

    def reconcile(self, request_id, result, *, evidence):
        """Apply an explicitly verified provider receipt without dispatching.

        The evidence must bind the local request and exact payload hash, identify
        its source, and state completed or verified_not_submitted. This function
        does not infer provider truth from a timeout or elapsed time.
        """
        receipt = self.receipt(request_id)
        if not receipt or receipt['status'] not in {'unknown','in_flight'}:
            raise IntegrityError('Only a pending unknown/in-flight request can be reconciled')
        if (not isinstance(evidence,dict) or evidence.get('verified') is not True or
            evidence.get('request_id') != request_id or not evidence.get('source') or
            evidence.get('payload_sha256') != digest(receipt['payload'])):
            raise IntegrityError('Receipt evidence must bind this request and complete payload')
        outcome = evidence.get('outcome')
        if outcome == 'completed':
            if (not isinstance(result,dict) or result.get('status') != 'success' or
                result.get('returned_model') != receipt['payload']['model'] or
                result.get('finish_reason') != 'stop' or not valid_usage(result.get('usage')) or
                not isinstance(result.get('answer'),str)):
                raise IntegrityError('Receipt must prove complete stop, correct model and actual usage')
            usage = result['usage']
            raw = result.get('raw_response') or {}
            if (not evidence.get('provider_receipt_id') or raw.get('id') != evidence['provider_receipt_id'] or
                raw.get('model') != result['returned_model'] or raw.get('usage') != usage):
                raise IntegrityError('Provider receipt identity/model/usage differs from result')
            if (usage['prompt_tokens'] > MAX_INPUT or usage['completion_tokens'] > MAX_OUTPUT or
                usage['prompt_tokens']+MAX_OUTPUT > CONTEXT or
                usage['prompt_tokens']+usage['completion_tokens'] > CONTEXT):
                raise IntegrityError('Reconciled usage exceeds frozen context limits')
            actual = usage_cost(usage['prompt_tokens'],usage['completion_tokens'])
            next_result = result
        elif outcome == 'verified_not_submitted':
            if evidence.get('provider_confirmed_no_charge') is not True or result is not None:
                raise IntegrityError('No-submission recovery requires explicit provider no-charge confirmation')
            usage, actual = None, 0.
            next_result = {'status':'failed','confirmed_not_submitted':True,'retryable':True,
                           'failure_class':'VerifiedNoSubmission','receipt_evidence_source':evidence['source']}
        else:
            raise IntegrityError('Unsupported receipt disposition; keep liability unresolved')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            current = self.receipt(request_id)
            if current['status'] not in {'unknown','in_flight'}:
                raise IntegrityError('Receipt was reconciled concurrently')
            attempt = self.db.execute('SELECT id,status,result FROM rt_attempts WHERE key=? ORDER BY ordinal DESC LIMIT 1',
                                      (receipt['key'],)).fetchone()
            if not attempt or attempt[1] not in {'unknown','in_flight'}:
                raise IntegrityError('Pending request has no matching unresolved new attempt')
            self.db.execute('INSERT INTO rt_reconciliations(attempt_id,created,previous_status,previous_result,evidence,result) VALUES(?,?,?,?,?,?)',
                (attempt[0],utc(),attempt[1],attempt[2],canonical(evidence),canonical(next_result)))
            self.db.execute('UPDATE rt_attempts SET status=?,actual=?,usage=?,result=? WHERE id=?',
                ('reconciled_'+next_result['status'],actual,canonical(usage) if usage is not None else None,
                 canonical(next_result),attempt[0]))
            self.db.execute('UPDATE rt_stages SET status=?,result=? WHERE key=?',
                (next_result['status'],canonical(next_result),receipt['key']))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.check_query_alerts(receipt['dataset'],receipt['method'],receipt['qid'])
        return self.receipt(request_id)

    def query_summary(self, dataset, method, qid):
        rows = self.db.execute('''SELECT a.actual,a.reserved,a.usage,a.status FROM rt_attempts a
            JOIN rt_stages s ON s.key=a.key WHERE s.dataset=? AND s.method=? AND s.qid=?''',
            (dataset,method,qid)).fetchall()
        stages = self.db.execute('SELECT result,origin,status FROM rt_stages WHERE dataset=? AND method=? AND qid=?',
                                (dataset,method,qid)).fetchall()
        logical_in = logical_out = 0
        logical_unknown = False
        logical_cost = 0.
        # Failed paid attempts belong to full-flow expense; successful cached
        # logical stages count once regardless of repeated reads on restart.
        for actual, reserve, usage, status in rows:
            if usage:
                u = json.loads(usage)
                logical_in += u['prompt_tokens']; logical_out += u['completion_tokens']
                logical_cost += usage_cost(u['prompt_tokens'], u['completion_tokens'])
            elif actual is None:
                logical_unknown = True
        for result, origin, status in stages:
            if origin != 'new_runtime':
                u = (json.loads(result) if result else {}).get('usage')
                if valid_usage(u):
                    logical_in += u['prompt_tokens']; logical_out += u['completion_tokens']
                    logical_cost += usage_cost(u['prompt_tokens'], u['completion_tokens'])
                else:
                    logical_unknown = True
        return {'dataset':dataset,'method':method,'qid':qid,
            'new_known_CNY':sum(r[0] for r in rows if r[0] is not None),
            'unknown_reserved_CNY':sum(r[1] for r in rows if r[0] is None),
            'unknown_usage_attempts':sum(r[0] is None for r in rows),
            'logical_known_CNY':logical_cost,'logical_input_tokens':logical_in,
            'logical_output_tokens':logical_out,'logical_usage_complete':not logical_unknown,
            'all_stages_successful':bool(stages) and all(s[2]=='success' for s in stages),
            'all_stages_complete':bool(stages) and all(s[2] in {'success','algorithm_failure'} for s in stages),
            'all_stages_fresh':bool(stages) and all(s[1]=='new_runtime' for s in stages)}

    def summary(self):
        rows = self.db.execute('SELECT actual,reserved,status FROM rt_attempts').fetchall()
        return {'new_known_CNY':sum(r[0] for r in rows if r[0] is not None),
                'unknown_reserved_CNY':sum(r[1] for r in rows if r[0] is None),
                'unknown_usage_attempts':sum(r[0] is None for r in rows),
                'historical_reconciliation_overlays':self.db.execute('SELECT count(*) FROM rt_historical_reconciliations').fetchone()[0],
                'historical_liabilities_remain_in_original_databases':True,
                'no_fixed_budget_cap':True,
                'pending_notifications':self.db.execute('SELECT count(*) FROM rt_alerts WHERE delivered=0').fetchone()[0]}

    def _reference(self, dataset, method):
        item = self.reference_means.get((dataset,method), self.reference_means.get(dataset,{}).get(method))
        return item.get('mean_cny') if isinstance(item,dict) else item

    def _reference_details(self, dataset, method):
        item = self.reference_means.get((dataset,method), self.reference_means.get(dataset,{}).get(method))
        return item if isinstance(item,dict) else {'mean_cny':item}

    def _alert(self, identity, payload):
        self.db.execute('INSERT OR IGNORE INTO rt_alerts(id,created,payload) VALUES(?,?,?)',
                        (digest(identity),utc(),canonical(payload)))
        self.db.commit()

    def dispatch_notifications(self):
        if self.notify is None:
            return
        for aid, payload in self.db.execute('SELECT id,payload FROM rt_alerts WHERE delivered=0 ORDER BY rowid').fetchall():
            try:
                self.notify(json.loads(payload))
            except Exception:
                # Durable and retried at the next checkpoint; never erase an
                # expense alert because its optional notification sink failed.
                continue
            self.db.execute('UPDATE rt_alerts SET delivered=1 WHERE id=?',(aid,))
            self.db.commit()

    def check_query_alerts(self, dataset, method, qid):
        cost = self.query_summary(dataset,method,qid)
        reference = self._reference(dataset,method)
        for signal in expense_signals(reference,cost['new_known_CNY']):
            self._alert([dataset,method,qid,signal['kind']],
                {**signal,**cost,'historical_mean_CNY':reference,
                 'historical_reference':self._reference_details(dataset,method),
                 'reference_interpretation':'historical configuration; method parameters may differ'})
        if cost['unknown_usage_attempts']:
            self._alert([dataset,method,qid,'unknown_query_usage_not_zero'],
                {'kind':'unknown_query_usage_not_zero',**cost})
        self.dispatch_notifications()

    def complete_query(self, dataset, method, qid, *, complete=True, fresh=True):
        cost = self.query_summary(dataset,method,qid)
        eligible = bool(complete and fresh and cost['logical_usage_complete'] and
                        cost['all_stages_complete'] and cost['all_stages_fresh'])
        previous_completion = self.db.execute('SELECT eligible FROM rt_completions WHERE dataset=? AND method=? AND qid=?',
                                               (dataset,method,qid)).fetchone()
        # A partial/unknown flow can later be completed after receipt recovery.
        # Its position in the window is the time it actually becomes eligible.
        inserted = False
        if complete and (not previous_completion or (eligible and not previous_completion[0])):
            self.db.execute('INSERT OR REPLACE INTO rt_completions(dataset,method,qid,cost,eligible) VALUES(?,?,?,?,?)',
                            (dataset,method,qid,cost['new_known_CNY'],int(eligible)))
            inserted = True
        self.db.commit()
        self.check_query_alerts(dataset,method,qid)
        if not inserted or not eligible:
            return cost
        values = [r[0] for r in self.db.execute('SELECT cost FROM rt_completions WHERE dataset=? AND method=? AND eligible=1 ORDER BY id DESC LIMIT ?',
                                             (dataset,method,COST_WINDOW))]
        signals = expense_signals(self._reference(dataset,method),cost['new_known_CNY'],values)
        elevated = next((s for s in signals if s['kind']=='rolling_query_cost_increase'),None)
        previous = self.db.execute('SELECT elevated FROM rt_rolling_state WHERE dataset=? AND method=?',(dataset,method)).fetchone()
        if elevated and not (previous and previous[0]):
            self._alert([dataset,method,'rolling',qid],{**elevated,'dataset':dataset,'method':method,'qid':qid,
                        'historical_mean_CNY':self._reference(dataset,method),
                        'historical_reference':self._reference_details(dataset,method),
                        'reference_interpretation':'historical configuration; method parameters may differ'})
        self.db.execute('INSERT OR REPLACE INTO rt_rolling_state VALUES(?,?,?)',(dataset,method,int(bool(elevated))))
        self.db.commit()
        self.dispatch_notifications()
        return cost

    def close(self):
        self.db.close()
