"""End-to-end controller/transport/store tests with synthetic clients and retrieval."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from . import runtime, runtime_config as cfg
from .runtime_transport import RuntimeTransport
from .runtime_ledger import RuntimeLedger
from .qa_raw_runtime import build_pair


class FakeClient:
    max_retries = 0

    def __init__(self, answers):
        self.answers = iter(answers)
        self.sent = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **payload):
        self.sent.append(payload)
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        usage = {'prompt_tokens': 100, 'completion_tokens': 10}
        return NS(model=cfg.common.MODEL,
                  choices=[NS(message=NS(content=answer, refusal=None), finish_reason='stop')],
                  model_dump=lambda **kw: {'usage': usage})


class FakeRetrieval:
    def __init__(self, store, resources, guard):
        self.store, self.guard = store, guard

    def retrieve(self, qid, query, k):
        return self.retrieve_many([{'qid': qid, 'query': query}], k)[0]

    def retrieve_many(self, cases, k):
        self.guard()
        result = []
        for case in cases:
            docs = [{'doc_id': str(i), 'rank': i, 'source': 'source' + str(i),
                     'text': 'A complete source passage ' + str(i),
                     'block': f'[Chunk {i} | source=source{i}, pages=1-1]\nA complete source passage {i}'}
                    for i in range(1, k + 1)]
            packet = {**case, 'k': k, 'docs': docs, 'encoder_truncated': False,
                      'query_tokens': 10, 'processed_content_tokens': 8,
                      'encoded_window_tokens': [10], 'window_count': 1,
                      'overflow_policy': 'original_short_query'}
            self.store.put('retrieval', cfg.common.digest([case, k]), packet)
            result.append(packet)
        return result


class FakeReranker:
    def __init__(self, store):
        self.store = store

    def rerank(self, question, docs, k, qid):
        return docs[:k], 0.


class RuntimeIntegration(unittest.TestCase):
    def setUp(self):
        cfg.common.OUTPUT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='offline-unified-runtime-', dir=cfg.common.OUTPUT)
        self.root = Path(self.tmp.name)
        self.ledger = RuntimeLedger(self.root / 'ledger.sqlite3')
        self.case = {'qid': 'synthetic', 'question': 'Who is A?'}
        self.resources = cfg.resource('hotpotqa')

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def runner(self, answers):
        client = FakeClient(answers)
        def transport(*a, **kw):
            return RuntimeTransport(*a, **kw, sleep=lambda n: None, clock=lambda: 100000000000.)
        return runtime.Runner(client, self.ledger, lambda: True, output=self.root,
                              transport_factory=transport, retrieval_factory=FakeRetrieval,
                              reuse_factory=lambda *a: None), client

    def test_all_five_frozen_baseline_controllers_through_new_transport(self):
        scripts = {
            'b00_rag_top10_v1': ['some answer'],
            'b01_rag_top10_cot3': ['Final answer: some answer'],
            'b08_qd_rerank': ['[]', 'some answer'],
            'b02_selfask_dense': ['Final answer: some answer'],
            'b06_decomp_bge': ['{"task":"retrieve_odqa","question":"Who is A?"}',
                '{"task":"retrieve","question":"Who is A?"}',
                '{"task":"singlehop_titleqa","titles":["source1"],"question":"Who is A?"}',
                'some answer', '{"task":"EOQ"}', '{"task":"EOQ"}']}
        from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.methods.qd_rerank import method
        for mid, answers in scripts.items():
            with self.subTest(method=mid), patch.object(method, 'Reranker', FakeReranker):
                runner, client = self.runner(answers)
                result = runner.run_case('hotpotqa', mid, self.case, self.resources)
                self.assertEqual(result['status'], 'success')
                self.assertEqual(len(client.sent), len(answers))
                self.assertTrue(all(p['max_tokens'] == 32768 and p['extra_body']['enable_thinking'] is False for p in client.sent))
                again = runner.run_case('hotpotqa', mid, self.case, self.resources)
                self.assertTrue(again['already_terminal'])
                self.assertEqual(len(client.sent), len(answers))

    def test_p02_prefix_survives_unknown_native_final_and_builds_both_arms(self):
        answers = ['A is the named subject.', 'Who is A?', 'A has the described property.', TimeoutError()]
        runner, client = self.runner(answers)
        case = {'id': 'hotpotqa:synthetic', 'dataset': 'hotpotqa', **self.case,
                'supporting_questions': ['What is the subject?']}
        result = runner.run_case('hotpotqa', cfg.P02, self.case, self.resources, p02_case=case)
        self.assertEqual(result['status'], 'technical_pending')
        directory = self.root / 'main/hotpotqa' / cfg.P02
        from .runtime_inputs import prefix_binding
        binding = prefix_binding(directory, 'synthetic')
        prefix = cfg.read(directory / 'prefixes/synthetic.json')
        pair = build_pair(prefix, expected_case=case, expected_namespace=cfg.common.digest(self.resources),
                          expected_prefix_sha256=binding['prefix_sha256'],
                          receipt_resolver=lambda r: self.ledger.receipt(r['request_id']))
        self.assertEqual(pair['status'], 'ready')
        self.assertEqual(set(pair['arms']), {'QA', 'Raw'})
        self.assertFalse(pair['native_final_required'])
        self.assertEqual(len(client.sent), 4)
        runner.run_case('hotpotqa', cfg.P02, self.case, self.resources, p02_case=case)
        self.assertEqual(len(client.sent), 4)

    def test_algorithm_failure_is_terminal_and_not_retried(self):
        runner, client = self.runner(['not a valid continuation'])
        mid = 'b02_selfask_dense'
        self.assertEqual(runner.run_case('hotpotqa', mid, self.case, self.resources)['status'], 'algorithm_failure')
        self.assertTrue(runner.run_case('hotpotqa', mid, self.case, self.resources)['already_terminal'])
        self.assertEqual(len(client.sent), 1)

    def test_cli_execute_refuses_before_client_construction(self):
        with self.assertRaises(cfg.ExecutionHeld):
            runtime.main(['--execute'])

    def test_frozen_mapping_has_exact_six_methods_and_both_decomp_limits(self):
        configs = cfg.method_configs()
        self.assertEqual(tuple(configs), cfg.METHODS)
        self.assertEqual(configs['b08_qd_rerank']['max_queries'], 5)
        self.assertEqual(configs['b02_selfask_dense']['max_search'], 5)
        self.assertEqual(configs['b06_decomp_bge']['max_steps'], 8)
        self.assertEqual(configs['b06_decomp_bge']['k'], 2)


if __name__ == '__main__':
    unittest.main()
