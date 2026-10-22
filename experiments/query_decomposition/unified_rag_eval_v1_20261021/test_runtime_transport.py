"""Offline tests of the real ask path; no model, network or old-ledger writes."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from experiments.query_decomposition.baseline_origin_binding_fix_20261017 import OriginSafeTransport
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import (
    MODEL, MAX_OUTPUT, IntegrityError, canonical, digest, usage_cost,
)
from .runtime_ledger import RuntimeLedger, RuntimeStageHeld
from .runtime_transport import RuntimeTransport


class MemoryStore:
    """Minimal actual SQLite checkpoint schema, confined to a disposable directory."""
    def __init__(self, root):
        self.namespace = 'offline/runtime-transport'
        self.db = sqlite3.connect(Path(root)/'checkpoint.sqlite3')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS data(kind TEXT,key TEXT,value TEXT,PRIMARY KEY(kind,key));
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,stage TEXT,payload TEXT,status TEXT,result TEXT);
          CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,request_id TEXT,started TEXT,status TEXT,result TEXT);''')
        self.events = []

    def get(self, kind, key):
        row = self.db.execute('SELECT value FROM data WHERE kind=? AND key=?',(kind,key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        old = self.get(kind,key)
        if old is not None and canonical(old) != canonical(value):
            raise AssertionError('Immutable checkpoint collision: '+kind)
        self.db.execute('INSERT OR IGNORE INTO data VALUES(?,?,?)',(kind,key,canonical(value)))
        self.db.commit()

    def event(self, event, **fields):
        self.events.append((event,fields))

    def export(self):
        pass


class Response:
    def __init__(self, answer='answer', usage=None, finish='stop', model=MODEL, omit_usage=False):
        self.model = model
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=answer,refusal=None),finish_reason=finish)]
        self.usage = None if omit_usage else (usage or {'prompt_tokens':100,'completion_tokens':10})

    def model_dump(self, mode='json'):
        return {'model':self.model,'usage':self.usage,'id':'fake-provider-receipt'}


class HTTPFailure(Exception):
    def __init__(self, code=429, *, usage=None, error_code='Throttling'):
        self.status_code = code
        self.body = {'error':{'code':error_code},'usage':usage}
        self.response = SimpleNamespace(headers={'retry-after':'0'})


class Client:
    max_retries = 0

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.payloads = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.payloads.append(payload)
        outcome = next(self.outcomes)
        if isinstance(outcome,BaseException):
            raise outcome
        return outcome


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='runtime-transport-offline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = MemoryStore(self.root)
        self.addCleanup(self.store.db.close)
        self.ledger = RuntimeLedger(self.root/'runtime.sqlite3',reference_means={('hotpotqa','test_method'):1.})
        self.addCleanup(lambda:self.ledger.close())

    def transport(self, client, **kwargs):
        result = RuntimeTransport(self.store,client,self.ledger,guard=kwargs.pop('guard',lambda:True),
            dataset='hotpotqa',method='test_method',sleep=lambda _:None,**kwargs)
        result.qid = 'q1'
        return result

    def restart(self):
        self.ledger.close()
        self.ledger = RuntimeLedger(self.root/'runtime.sqlite3',reference_means={('hotpotqa','test_method'):1.})

    def test_same_v2_payload_and_success_cache_survive_restart(self):
        client = Client([Response()])
        transport = self.transport(client)
        self.assertIsInstance(transport,OriginSafeTransport)
        expected_id = transport.request_id_for('final','full prompt',16)
        rid, first = transport.ask('final','full prompt',16)
        self.assertEqual(rid,expected_id)
        self.assertEqual(client.payloads,[{'model':MODEL,'messages':[{'role':'user','content':'full prompt'}],
            'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}])
        before = self.ledger.summary()
        self.restart()
        again = self.transport(client)
        rid2, second = again.ask('final','full prompt',16)
        self.assertEqual((rid,first),(rid2,second))
        self.assertEqual(len(client.payloads),1)
        self.assertEqual(self.ledger.summary(),before)
        receipt = self.ledger.receipt(rid)
        self.assertEqual((receipt['dataset'],receipt['method'],receipt['qid'],receipt['stage']),
                         ('hotpotqa','test_method','q1','final'))
        self.assertFalse(again.last_call['new_submission'])

    def test_unknown_is_not_retried_now_or_after_restart_and_other_stage_works(self):
        client = Client([TimeoutError('no receipt'),Response()])
        transport = self.transport(client)
        for _ in range(2):
            with self.assertRaises(RuntimeStageHeld):
                transport.ask('ordinary_1','same prompt',32)
        self.assertEqual(len(client.payloads),1)
        self.restart()
        transport = self.transport(client)
        with self.assertRaises(RuntimeStageHeld):
            transport.ask('ordinary_1','same prompt',32)
        transport.ask('ordinary_2','same prompt',32)
        self.assertEqual(len(client.payloads),2)
        self.assertGreater(self.ledger.summary()['unknown_reserved_CNY'],0)
        self.assertEqual(self.ledger.summary()['unknown_usage_attempts'],1)

    def test_five_actual_attempts_persist_across_restarts(self):
        client = Client([HTTPFailure() for _ in range(7)])
        for _ in range(5):
            transport = self.transport(client)
            _, value = transport.ask('final','full prompt',16)
            self.assertEqual(value['status'],'failed')
            self.restart()
        with self.assertRaisesRegex(RuntimeStageHeld,'Five actual attempts'):
            self.transport(client).ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),5)
        self.assertEqual(self.ledger.summary()['unknown_reserved_CNY'],0.)

    def test_interrupted_process_keeps_inflight_liability(self):
        client = Client([KeyboardInterrupt(),Response()])
        with self.assertRaises(KeyboardInterrupt):
            self.transport(client).ask('final','full prompt',16)
        self.restart()
        with self.assertRaises(RuntimeStageHeld):
            self.transport(client).ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),1)
        self.assertGreater(self.ledger.summary()['unknown_reserved_CNY'],0.)

    def test_payload_change_cannot_reset_stage_attempts(self):
        client = Client([HTTPFailure(),Response()])
        self.transport(client).ask('final','original prompt',16)
        with self.assertRaisesRegex(IntegrityError,'payload changed'):
            self.transport(client).ask('final','changed prompt',16)
        self.assertEqual(len(client.payloads),1)

    def test_returned_failure_usage_settled_without_inventing_zero(self):
        usage = {'prompt_tokens':900,'completion_tokens':20}
        client = Client([HTTPFailure(429,usage=usage)])
        self.transport(client).ask('final','full prompt',16)
        summary = self.ledger.query_summary('hotpotqa','test_method','q1')
        self.assertAlmostEqual(summary['new_known_CNY'],usage_cost(900,20))
        self.assertEqual(summary['logical_input_tokens'],900)
        self.assertEqual(summary['unknown_reserved_CNY'],0.)

    def test_missing_usage_and_incomplete_generation_are_held(self):
        for response in (Response(omit_usage=True),Response(finish='length'),Response(model='wrong-model')):
            client = Client([response])
            transport = self.transport(client)
            transport.qid = 'q'+str(id(response))
            with self.assertRaises(RuntimeStageHeld):
                transport.ask('final','full prompt',16)
            with self.assertRaises(RuntimeStageHeld):
                transport.ask('final','full prompt',16)
            self.assertEqual(len(client.payloads),1)

    def test_malformed_response_keeps_returned_usage(self):
        response = Response(usage={'prompt_tokens':300,'completion_tokens':2})
        response.choices = []
        client = Client([response])
        with self.assertRaises(RuntimeStageHeld):
            self.transport(client).ask('final','full prompt',16)
        self.assertAlmostEqual(self.ledger.summary()['new_known_CNY'],usage_cost(300,2))
        self.assertEqual(self.ledger.summary()['unknown_reserved_CNY'],0.)

    def test_empty_answer_is_not_retried_for_quality(self):
        client = Client([Response(answer='')])
        transport = self.transport(client)
        self.assertEqual(transport.ask('final','full prompt',16)[1]['answer'],'')
        self.assertEqual(transport.ask('final','full prompt',16)[1]['answer'],'')
        self.assertEqual(len(client.payloads),1)

    def test_provider_content_rejection_retained_without_repeat(self):
        client = Client([HTTPFailure(400,error_code='data_inspection_failed')])
        transport = self.transport(client)
        for _ in range(2):
            self.assertEqual(transport.ask('final','full prompt',16)[1]['status'],'provider_content_rejected')
        self.assertEqual(len(client.payloads),1)
        self.assertGreater(self.ledger.summary()['unknown_reserved_CNY'],0.)

    def test_pause_guard_and_sdk_retry_check_prevent_submission(self):
        client = Client([Response()])
        with self.assertRaises(RuntimeStageHeld):
            self.transport(client,guard=lambda:False).ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),0)
        client.max_retries = 2
        with self.assertRaisesRegex(IntegrityError,'hidden retries'):
            self.transport(client).ask('final','full prompt',16)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_attempts').fetchone()[0],0)

    def test_oversize_full_input_is_not_truncated(self):
        client = Client([])
        with self.assertRaisesRegex(IntegrityError,'no truncation'):
            self.transport(client).ask('final','a'*1_000_000,16)
        self.assertEqual(len(client.payloads),0)

    def test_durable_notifications_single_and_rolling_deduplicated(self):
        delivered = []
        self.ledger.reference_means = {('hotpotqa','test_method'):.001}
        self.ledger.notify = delivered.append
        client = Client([Response(usage={'prompt_tokens':5000,'completion_tokens':1})]+[
            Response(usage={'prompt_tokens':2500,'completion_tokens':1}) for _ in range(20)]+[
            Response(usage={'prompt_tokens':100,'completion_tokens':1}) for _ in range(20)]+[
            Response(usage={'prompt_tokens':2500,'completion_tokens':1}) for _ in range(20)])
        transport = self.transport(client)
        transport.ask('final','full prompt',16)
        for _ in range(2):
            self.ledger.complete_query('hotpotqa','test_method','q1')
        for i in range(60):
            transport.qid = 'rolling'+str(i)
            transport.ask('final','full prompt',16)
            self.ledger.complete_query('hotpotqa','test_method',transport.qid)
        self.assertEqual(sum(s['kind']=='single_query_cost_increase' for s in delivered),1)
        self.assertEqual(sum(s['kind']=='rolling_query_cost_increase' for s in delivered),2)
        self.assertEqual(self.ledger.summary()['pending_notifications'],0)

    def test_notification_failure_is_durable_and_not_a_budget_cap(self):
        def broken(_):
            raise RuntimeError('sink unavailable')
        self.ledger.notify = broken
        self.ledger.reference_means = {('hotpotqa','test_method'):1e-12}
        client = Client([Response()])
        self.transport(client).ask('final','full prompt',16)
        self.assertGreater(self.ledger.summary()['pending_notifications'],0)
        received = []
        self.restart()
        self.ledger.notify = received.append
        self.ledger.dispatch_notifications()
        self.assertEqual(len(received),1)
        self.assertTrue(self.ledger.summary()['no_fixed_budget_cap'])

    def test_persisted_rate_slot_rechecks_stop_before_any_attempt(self):
        self.ledger.wait_for_start(lambda:None,sleep=lambda _:None,clock=lambda:100.)
        waited = []
        self.restart()
        self.ledger.wait_for_start(lambda:None,sleep=waited.append,clock=lambda:100.)
        self.assertAlmostEqual(sum(waited),2/3)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_attempts').fetchone()[0],0)

    def test_unrelated_ledger_is_never_modified(self):
        path = self.root/'historical.sqlite3'
        with sqlite3.connect(path) as old:
            old.execute('CREATE TABLE charges(id TEXT,actual REAL)')
            old.execute("INSERT INTO charges VALUES('old',1.2)")
        old.close()
        before = path.read_bytes()
        with self.assertRaisesRegex(IntegrityError,'historical'):
            RuntimeLedger(path)
        self.assertEqual(path.read_bytes(),before)

    def test_qa_raw_terminal_algorithm_failures_are_not_retried(self):
        client = Client([Response(answer=''),Response(finish='length',
                        usage={'prompt_tokens':100,'completion_tokens':MAX_OUTPUT})])
        transport = self.transport(client)
        for stage in ('qa_raw/QA/final','qa_raw/Raw/final'):
            self.assertEqual(transport.ask(stage,'full prompt',16)[1]['status'],'algorithm_failure')
            self.assertEqual(transport.ask(stage,'full prompt',16)[1]['status'],'algorithm_failure')
        self.assertEqual(len(client.payloads),2)
        self.assertEqual(self.ledger.summary()['unknown_usage_attempts'],0)

    def test_receipt_reconciliation_settles_unknown_without_resubmission(self):
        client = Client([TimeoutError()])
        transport = self.transport(client)
        rid = transport.request_id_for('final','full prompt')
        with self.assertRaises(RuntimeStageHeld):
            transport.ask('final','full prompt',16)
        receipt = self.ledger.receipt(rid)
        usage = {'prompt_tokens':120,'completion_tokens':8}
        result = {'status':'success','answer':'answer','finish_reason':'stop','returned_model':MODEL,
                  'usage':usage,'raw_response':{'id':'provider-receipt','model':MODEL,'usage':usage}}
        evidence = {'verified':True,'request_id':rid,'payload_sha256':digest(receipt['payload']),
                    'source':'offline synthetic provider receipt','outcome':'completed',
                    'provider_receipt_id':'provider-receipt'}
        self.ledger.reconcile(rid,result,evidence=evidence)
        self.restart()
        self.assertEqual(self.transport(client).ask('final','full prompt',16)[1]['answer'],'answer')
        self.assertEqual(len(client.payloads),1)
        self.assertAlmostEqual(self.ledger.summary()['new_known_CNY'],usage_cost(120,8))
        self.assertEqual(self.ledger.summary()['unknown_reserved_CNY'],0.)
        row = self.ledger.db.execute('SELECT previous_status,previous_result FROM rt_reconciliations').fetchone()
        self.assertEqual(row[0],'unknown')
        self.assertEqual(json.loads(row[1])['failure_class'],'TimeoutError')
        with self.assertRaises(IntegrityError):
            self.ledger.reconcile(rid,result,evidence=evidence)

    def test_no_submission_receipt_keeps_counter_before_retry(self):
        client = Client([TimeoutError(),Response()])
        transport = self.transport(client)
        rid = transport.request_id_for('final','full prompt')
        with self.assertRaises(RuntimeStageHeld):
            transport.ask('final','full prompt',16)
        receipt = self.ledger.receipt(rid)
        evidence = {'verified':True,'request_id':rid,'payload_sha256':digest(receipt['payload']),
                    'source':'offline synthetic provider acknowledgement','outcome':'verified_not_submitted',
                    'provider_confirmed_no_charge':True}
        bad = {**evidence,'payload_sha256':'different'}
        with self.assertRaises(IntegrityError):
            self.ledger.reconcile(rid,None,evidence=bad)
        self.assertGreater(self.ledger.summary()['unknown_reserved_CNY'],0.)
        self.ledger.reconcile(rid,None,evidence=evidence)
        transport.ask('final','full prompt',16)
        self.assertEqual(transport.last_call['attempt_ordinal'],2)
        self.assertEqual(len(client.payloads),2)

    def test_real_c1_catalog_preserves_old_usage_and_zero_new_expense(self):
        from .runtime_reuse import ReadOnlyReuseCatalog
        from .test_runtime_reuse import database, RESOURCES, PACKET
        path = self.root/'historical.sqlite3'
        payload = {'model':MODEL,'messages':[{'role':'user','content':'Read complete raw evidence. Question?'}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        result = {'status':'success','answer':'answer','finish_reason':'stop','returned_model':MODEL,
                  'usage':{'prompt_tokens':12,'completion_tokens':2}}
        database(path,identity={'method':'test_method','dataset':'hotpotqa','resources':RESOURCES},
                 payload=payload,result=result,qid='q1').close()
        # Historical and current retrieval bind the same actual query and text.
        with sqlite3.connect(path) as old:
            old.execute("UPDATE data SET value=? WHERE kind='retrieval'",(canonical({**PACKET,'qid':'q1'}),))
        old.close()
        self.store.put('retrieval','packet',{**PACKET,'qid':'q1'})
        before = path.read_bytes()
        catalog = ReadOnlyReuseCatalog([path],method='test_method',dataset='hotpotqa',resources=RESOURCES)
        client = Client([])
        transport = self.transport(client,catalog=catalog)
        rid, got = transport.ask('final',payload['messages'][0]['content'],16)
        self.assertEqual(got,result)
        self.assertEqual(len(client.payloads),0)
        self.assertEqual(self.ledger.summary()['new_known_CNY'],0.)
        self.assertAlmostEqual(self.ledger.query_summary('hotpotqa','test_method','q1')['logical_known_CNY'],usage_cost(12,2))
        self.assertEqual(self.ledger.receipt(rid)['payload']['max_tokens'],64)
        transport.ask('final',payload['messages'][0]['content'],16)
        self.assertEqual(path.read_bytes(),before)

    def test_real_historical_unknown_blocks_no_new_liability(self):
        from .runtime_reuse import ReadOnlyReuseCatalog
        from .test_runtime_reuse import database, RESOURCES
        path = self.root/'historical.sqlite3'
        database(path,identity={'method':'test_method','dataset':'hotpotqa','resources':RESOURCES},
                 status='unknown',qid='q1').close()
        catalog = ReadOnlyReuseCatalog([path],method='test_method',dataset='hotpotqa',resources=RESOURCES)
        client = Client([])
        with self.assertRaisesRegex(RuntimeStageHeld,'Historical unknown'):
            self.transport(client,catalog=catalog).ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),0)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_attempts').fetchone()[0],0)

    def test_four_historical_attempts_allow_only_one_new_attempt(self):
        from .runtime_reuse import ReadOnlyReuseCatalog
        from .test_runtime_reuse import database, RESOURCES
        path = self.root/'historical.sqlite3'
        payload = {'model':MODEL,'messages':[{'role':'user','content':'full prompt'}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        old = database(path,identity={'method':'test_method','dataset':'hotpotqa','resources':RESOURCES},
                       status='failed',result={'status':'failed','http_status':429},qid='q1',payload=payload)
        old.execute('UPDATE requests SET payload=?',(canonical(payload),))
        old.execute('DELETE FROM attempts')
        for i in range(1,5):
            old.execute('INSERT INTO attempts VALUES(?,?,?,?)',(i,'receipt','failed',canonical({'status':'failed','http_status':429})))
        old.commit(); old.close()
        catalog = ReadOnlyReuseCatalog([path],method='test_method',dataset='hotpotqa',resources=RESOURCES)
        client = Client([HTTPFailure(),Response()])
        transport = self.transport(client,catalog=catalog)
        transport.ask('final','full prompt',16)
        self.assertEqual(transport.last_call['attempt_ordinal'],5)
        self.restart()
        with self.assertRaisesRegex(RuntimeStageHeld,'Five actual attempts'):
            self.transport(client,catalog=catalog).ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),1)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_historical_attempts').fetchone()[0],4)

    def test_changed_historical_count_evidence_is_not_silently_replaced(self):
        self.ledger.register_attempt_history('hotpotqa','test_method','q1','final',[{'identity':'old','status':'failed'}])
        self.ledger.register_attempt_history('hotpotqa','test_method','q1','final',[{'identity':'old','status':'failed'}])
        with self.assertRaises(IntegrityError):
            self.ledger.register_attempt_history('hotpotqa','test_method','q1','final',[{'identity':'old','status':'unknown'}])
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_historical_attempts').fetchone()[0],1)

    def test_historical_overlay_releases_only_verified_attempt_without_source_writes(self):
        from .runtime_reuse import ReadOnlyReuseCatalog
        from .test_runtime_reuse import database, RESOURCES
        path = self.root/'historical.sqlite3'
        payload = {'model':MODEL,'messages':[{'role':'user','content':'full prompt'}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        old = database(path,identity={'method':'test_method','dataset':'hotpotqa','resources':RESOURCES},
                       status='unknown',result={'status':'unknown'},qid='q1',payload=payload)
        old.execute('UPDATE requests SET payload=?',(canonical(payload),))
        old.execute('DELETE FROM attempts')
        for i in (1,2):
            old.execute('INSERT INTO attempts VALUES(?,?,?,?)',(i,'receipt','unknown',None))
        old.commit(); old.close()
        before = path.read_bytes()
        catalog = ReadOnlyReuseCatalog([path],method='test_method',dataset='hotpotqa',resources=RESOURCES,
                                      reconciliation_resolver=self.ledger.historical_reconciliations)
        client = Client([Response()])
        transport = self.transport(client,catalog=catalog)
        with self.assertRaises(RuntimeStageHeld):
            transport.ask('final','full prompt',16)
        evidence = {'origin_db':str(path),'request_id':'receipt','attempt_id':1,
                    'payload_sha256':digest(payload),'verified':True,'outcome':'verified_not_submitted',
                    'provider_confirmed_no_charge':True,'provider_receipt_id':'checked-one',
                    'source':'offline independent provider confirmation'}
        with self.assertRaises(IntegrityError):
            self.ledger.record_historical_reconciliation({**evidence,'payload_sha256':'wrong'})
        self.ledger.record_historical_reconciliation(evidence)
        self.ledger.record_historical_reconciliation(evidence)
        with self.assertRaises(RuntimeStageHeld):
            transport.ask('final','full prompt',16)
        self.assertEqual(len(client.payloads),0)
        self.ledger.record_historical_reconciliation({**evidence,'attempt_id':2,'provider_receipt_id':'checked-two'})
        transport.ask('final','full prompt',16)
        self.assertEqual(transport.last_call['attempt_ordinal'],3)
        self.assertEqual(len(client.payloads),1)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.ledger.summary()['historical_reconciliation_overlays'],2)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM rt_historical_attempts').fetchone()[0],2)

    def test_request_only_historical_overlay_keeps_conservative_initial_count(self):
        from .runtime_reuse import ReadOnlyReuseCatalog
        from .test_runtime_reuse import database, RESOURCES
        path = self.root/'historical.sqlite3'
        payload = {'model':MODEL,'messages':[{'role':'user','content':'full prompt'}],
                   'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
        old = database(path,identity={'method':'test_method','dataset':'hotpotqa','resources':RESOURCES},
                       status='unknown',result={'status':'unknown'},qid='q1',payload=payload)
        old.execute('UPDATE requests SET payload=?',(canonical(payload),))
        old.execute('DELETE FROM attempts')
        old.commit(); old.close()
        before = path.read_bytes()
        self.ledger.record_historical_reconciliation({'origin_db':str(path),'request_id':'receipt','attempt_id':None,
            'payload_sha256':digest(payload),'verified':True,'outcome':'verified_not_submitted',
            'provider_confirmed_no_charge':True,'provider_receipt_id':'checked-request',
            'source':'offline independent provider confirmation'})
        catalog = ReadOnlyReuseCatalog([path],method='test_method',dataset='hotpotqa',resources=RESOURCES,
                                      reconciliation_resolver=self.ledger.historical_reconciliations)
        transport = self.transport(Client([Response()]),catalog=catalog)
        transport.ask('final','full prompt',16)
        self.assertEqual(transport.last_call['attempt_ordinal'],2)
        self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
