"""Offline checks for shared-layer wiring, content TTFT and denominator truth."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from . import metrics, p02_shared_layer as shared
from .timing import ContentTimedCompletions
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import transport as v2
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import transport_base


class Contracts(unittest.TestCase):
    def test_pause_and_unapproved_revision_block_execution(self):
        with self.assertRaises(shared.ExecutionNotApproved):shared.require_execution_approval()

    def test_real_shared_transport_uses_v2_maximum_and_rejects_length(self):
        class Catalog:
            def request(self,*args):return None
        class Budget:
            def __init__(self,path):self.path=path
            def wait_for_start(self):pass
            def reserve(self,*args):pass
            def settle(self,*args):pass
            def summary(self):return {'offline_fake_budget':True}
        seen=[]
        def create(**payload):
            seen.append(payload)
            return NS(model_dump=lambda **kw:{'usage':{'prompt_tokens':10,'completion_tokens':2}},
                      choices=[NS(message=NS(content='complete',refusal=None),finish_reason='stop')],model=shared.common.MODEL)
        output=shared.ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/unified_eval_offline_checks_20261021'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            store=Store(Path(tmp)/'store',{'task':'offline shared-layer contract'})
            try:
                layer=shared.REQUEST_LAYER(store,NS(chat=NS(completions=NS(create=create))),Budget(Path(tmp)/'budget.sqlite3'),shared.common.MODEL,Catalog())
                layer.qid='qid'
                # Temporary test paths do not alter any real user pause marker.
                with patch.object(v2,'OUT',Path(tmp)),patch.object(transport_base,'OUT',Path(tmp)):
                    _,r=layer.ask('legacy/final','return a short answer',64)
                self.assertEqual(seen[0]['max_tokens'],32768)
                self.assertEqual(r['finish_reason'],'stop')
                token=json.loads(store.db.execute("select value from data where kind='token_call'").fetchone()[0])
                self.assertEqual(token['native_requested_max_tokens'],64)
            finally:store.db.close()

    def test_p02_bridge_refuses_incomplete_final(self):
        callback=object.__new__(shared.P02Callbacks)
        callback.requests=NS(ask=lambda *args:('rid',{'status':'success','answer':'partial','finish_reason':'length','usage':{'prompt_tokens':1}}))
        with patch.object(shared,'require_execution_approval',return_value=None):
            with self.assertRaises(shared.common.IntegrityError):callback.ask('legacy/final','prompt',64,True)

    def test_p02_retrieval_reaches_existing_v2_long_input_layer(self):
        from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.retrieval import Retrieval
        self.assertIs(shared.RETRIEVAL_LAYER,Retrieval)
        callback=object.__new__(shared.P02Callbacks);callback.case={'qid':'q'}
        observed=[]
        callback.retrieval=NS(retrieve_many=lambda cases,k:(observed.append((cases,k)) or [{'docs':[]}]))
        with patch.object(shared,'require_execution_approval',return_value=None):callback.retrieve('long query '*1000)
        self.assertEqual(observed[0][0][0]['query'],'long query '*1000)
        self.assertEqual(observed[0][1],10)

    def test_ttft_ignores_role_and_usage_frames_and_retains_full_output(self):
        def chunk(content=None,finish=None,usage=None):
            return NS(id='id',created=1,model='model',usage=usage,
                      choices=[] if usage else [NS(index=0,delta=NS(content=content,refusal=None),finish_reason=finish)])
        usage=NS(model_dump=lambda **kw:{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7})
        @contextmanager
        def create(**wire):
            self.assertTrue(wire['stream_options']['include_usage'])
            yield iter([chunk(''),chunk('first '),chunk('last'),chunk(finish='stop'),chunk(usage=usage)])
        ticks=iter([10.,13.])
        response=ContentTimedCompletions(NS(create=create),clock=lambda:next(ticks)).create(model='model',messages=[])
        self.assertEqual(response.stream_transport['client_ttft_seconds'],3.)
        self.assertEqual(response.choices[0].message.content,'first last')
        self.assertEqual(response.usage.completion_tokens,2)

    def test_unknown_or_replayed_time_is_never_zero(self):
        result=metrics.timing_for_query([{'kind':'local','processing_seconds':2.},
             {'kind':'api','client_ttft_seconds':0.1,'observation':'historical_replay'}])
        self.assertEqual(result['local_processing_seconds'],2.)
        self.assertIsNone(result['client_ttft_seconds'])
        self.assertNotIn('total_seconds',result)

    def test_partial_quality_is_not_published_as500(self):
        partial=[{'qid':str(i),'f1':1.,'em':1.} for i in range(499)]
        expected={str(i) for i in range(500)}
        r=metrics.quality_for_primary_table(partial,expected_qids=expected)
        self.assertIsNone(r['F1']);self.assertNotIn('scorable',r)
        r=metrics.quality_for_primary_table(partial+[{'qid':'499','f1':0.,'em':0.}],expected_qids=expected)
        self.assertEqual(r['F1'],499/500);self.assertNotIn('scorable',r)
        r=metrics.quality_for_primary_table(partial,expected_qids=expected,confirmed_unrecoverable={'user_decision_id':'user-reviewed','evidence_receipt':'receipt.json','reason':'confirmed blocked'})
        self.assertEqual(r['scorable'],499)
        with self.assertRaises(ValueError):metrics.quality_for_primary_table(partial+[partial[0]],expected_qids=expected)

if __name__=='__main__':unittest.main()
