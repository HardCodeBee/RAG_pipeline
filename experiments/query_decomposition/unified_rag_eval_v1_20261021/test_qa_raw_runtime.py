"""Offline tests: synthetic packets and receipts only, no API or retrieval."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.query_decomposition.unified_rag_eval_v1_20261021 import qa_raw_runtime as qr
from src.p02_multigrain_timing_cachefix_v1.core import pipeline


def fixture(*, goal_only=False, formatter=False, conflicting=False, padded=False):
    case = {'id': 'hotpotqa:test-qid', 'dataset': 'hotpotqa', 'qid': 'test-qid',
            'question': 'Original root?', 'supporting_questions': [] if goal_only else ['Auxiliary question?']}
    namespace = 'frozen-resource-namespace'
    calls, nodes, durable = [], [], {}

    def retrieve(query):
        ids = ['d0'] + [f'd{i}' for i in range(10, 19)] if query == 'Auxiliary question?' else (
            [f'g{i}' for i in range(10)] if query == 'Rewritten goal?' else [f'd{i}' for i in range(10)])
        docs = []
        for rank, did in enumerate(ids, 1):
            source = 'Title | [special]\\newline\nsource'
            text = 'Full raw source for ' + did + '.\nLine two; unchanged.'
            if conflicting and query == 'Auxiliary question?' and did == 'd0':
                text += ' Conflicting version.'
            docs.append({'rank': rank, 'doc_id': did, 'source': source, 'text': text,
                         'block': f'[Chunk {rank} | source={source}, pages=1-1]\n{text}'})
        return {'qid': case['qid'], 'query': query, 'k': 10, 'docs': docs,
                'encoder_truncated': False, 'query_tokens': 8, 'content_tokens': 6,
                'processed_content_tokens': 6, 'window_count': 1, 'encoded_window_tokens': [8]}

    class Complete(Exception):
        pass

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        if stage == 'legacy/final':
            raise Complete()
        answer = ('Rewritten goal?' if '/rewrite/' in stage else
                  'Paris' if formatter and stage == 'legacy/answer/1' else
                  'This is an intermediate generated candidate factual answer.')
        if padded:
            answer = ' \n' + answer + '\n '
        result = {'status': 'success', 'finish_reason': 'stop', 'answer': answer,
                  'returned_model': qr.common.MODEL,
                  'usage': {'prompt_tokens': 100, 'completion_tokens': 12}}
        rid = 'rid-' + stage
        call = {'stage': stage, 'request_id': rid, 'payload': qr.payload(prompt), 'result': result,
                'source': {'kind': 'shared_layer', 'dataset': case['dataset'], 'qid': case['qid'],
                           'method': qr.METHOD, 'execution_variant': qr.VARIANT,
                           'resource_namespace': namespace}}
        calls.append(call)
        durable[rid] = {'stage': stage, 'request_id': rid, 'dataset': case['dataset'],
                        'qid': case['qid'], 'method': qr.METHOD,
                        'payload': deepcopy(call['payload']), 'result': deepcopy(result),
                        'status': 'success', 'origin': 'new_runtime'}
        return rid, answer, 'stop'

    try:
        pipeline.execute(case, retrieve, ask, record_node=nodes.append)
    except Complete:
        pass
    prefix = {'schema': qr.SCHEMA, 'case': case, 'method': qr.METHOD,
              'execution_variant': qr.VARIANT, 'resource_namespace': namespace,
              'nodes': nodes, 'original_retrieval': deepcopy(nodes[-1]['original_retrieval']), 'receipts': calls}
    return prefix, durable


def build(prefix, durable):
    return qr.build_pair(prefix, expected_case=prefix['case'],
                         expected_namespace=prefix['resource_namespace'],
                         expected_prefix_sha256=qr.fingerprint(prefix),
                         receipt_resolver=lambda call: durable.get(call['request_id']))


class PrefixTests(unittest.TestCase):
    def test_complete_prefix_does_not_require_native_final(self):
        prefix, durable = fixture()
        pair = build(prefix, durable)
        self.assertEqual(pair['status'], 'ready')
        self.assertFalse(pair['native_final_required'])
        self.assertNotIn('legacy/final', [r['stage'] for r in prefix['receipts']])
        self.assertEqual(pair['upstream']['logical_prompt_tokens'], 300)
        self.assertIsNone(pair['arms']['QA']['input_check']['model_tokens'])
        template = (qr.PROTOCOL_DIR / 'PROMPT.txt').read_bytes().decode('utf8')
        self.assertTrue(pair['arms']['Raw']['prompt'].startswith(template.split('{question}')[0]))

    def test_goal_only_keeps_goal_question_and_reader(self):
        prefix, durable = fixture(goal_only=True)
        pair = build(prefix, durable)
        self.assertEqual(pair['upstream']['ordinary_nodes'], 0)
        self.assertIn('Step 1 (goal)', pair['arms']['QA']['prompt'])
        self.assertIn('Passages: R1, R2', pair['arms']['Raw']['prompt'])

    def test_all_actual_reader_material_and_root_dedup(self):
        prefix, durable = fixture()
        pair = build(prefix, durable)
        raw = pair['arms']['Raw']['prompt']
        for node in prefix['nodes']:
            for doc in node['reader_packet']['docs']:
                self.assertIn(doc['text'], raw)
        for doc in prefix['original_retrieval']['docs']:
            self.assertEqual(raw.count(doc['text']), 1)
        self.assertIn('Full raw source for g4.', raw)
        self.assertNotIn('Full raw source for g5.', raw)
        self.assertNotIn(prefix['nodes'][0]['answer'], raw)
        self.assertIn('\\| \\[special\\]', raw)
        self.assertEqual(pair['reader_associations'][0]['labels'][0], 'R1')

    def test_id_conflict_is_blocked_after_valid_replay(self):
        prefix, durable = fixture(conflicting=True)
        with self.assertRaisesRegex(qr.PrefixBlocked, 'same_document_id'):
            build(prefix, durable)

    def test_wrong_reader_packet_cannot_pass_replay(self):
        prefix, durable = fixture()
        prefix['nodes'][0]['reader_packet']['docs'][0]['text'] += ' changed'
        with self.assertRaisesRegex(qr.PrefixBlocked, 'saved_nodes'):
            build(prefix, durable)

    def test_missing_or_forged_durable_receipt_blocks(self):
        prefix, durable = fixture()
        rid = prefix['receipts'][0]['request_id']
        del durable[rid]
        with self.assertRaisesRegex(qr.PrefixBlocked, 'missing_durable_receipt'):
            build(prefix, durable)
        prefix, durable = fixture()
        prefix['receipts'][0]['result']['answer'] = 'A forged result may look successful.'
        with self.assertRaisesRegex(qr.PrefixBlocked, 'durable_receipt_payload_or_result'):
            build(prefix, durable)

    def test_prefix_hash_or_question_binding_is_required(self):
        prefix, durable = fixture()
        with self.assertRaisesRegex(qr.PrefixBlocked, 'main_prefix_binding'):
            qr.build_pair(prefix, expected_case=prefix['case'], expected_namespace=prefix['resource_namespace'],
                          expected_prefix_sha256='unbound', receipt_resolver=lambda c: durable[c['request_id']])
        case = deepcopy(prefix['case'])
        case['question'] = 'A different original question?'
        with self.assertRaisesRegex(qr.PrefixBlocked, 'frozen_question'):
            qr.build_pair(prefix, expected_case=case, expected_namespace=prefix['resource_namespace'],
                          expected_prefix_sha256=qr.fingerprint(prefix), receipt_resolver=lambda c: durable[c['request_id']])

    def test_missing_goal_or_duplicate_step_blocks(self):
        prefix, durable = fixture()
        prefix['nodes'].pop()
        with self.assertRaisesRegex(qr.PrefixBlocked, 'incomplete_ordinary_goal_prefix'):
            build(prefix, durable)
        prefix, durable = fixture()
        prefix['nodes'][1]['step_id'] = 1
        with self.assertRaisesRegex(qr.PrefixBlocked, 'non_contiguous_steps'):
            build(prefix, durable)

    def test_formatter_receipt_is_required_and_answer_exact(self):
        prefix, durable = fixture(formatter=True)
        pair = build(prefix, durable)
        self.assertEqual(pair['upstream']['logical_prompt_tokens'], 400)
        prefix['receipts'] = [r for r in prefix['receipts'] if '/sentence_format/' not in r['stage']]
        with self.assertRaisesRegex(qr.PrefixBlocked, 'missing_dependency_receipt'):
            build(prefix, durable)

    def test_c1_only_nonbinding_cap_difference(self):
        prefix, durable = fixture()
        for call in prefix['receipts']:
            call['source']['kind'] = 'c1'
            call['payload']['max_tokens'] = 128
            durable[call['request_id']]['payload']['max_tokens'] = 128
            durable[call['request_id']]['origin'] = 'c1_historical'
            durable[call['request_id']]['provenance'] = {'c1_validation': {
                'dataset': prefix['case']['dataset'], 'qid': prefix['case']['qid'],
                'method': qr.METHOD, 'stage': call['stage'],
                'full_messages_settings_equal_except_nonbinding_output_cap': True,
                'complete_stop_usage_model_verified': True, 'resources_verified': True,
                'current_retrieval_packets_verified': 1, 'source_meta_sha256': 'a'*64}}
        self.assertEqual(build(prefix, durable)['status'], 'ready')
        call = prefix['receipts'][0]
        call['payload']['max_tokens'] = 12
        durable[call['request_id']]['payload']['max_tokens'] = 12
        with self.assertRaisesRegex(qr.PrefixBlocked, 'not_proven_nonbinding'):
            build(prefix, durable)

    def test_c1_label_cannot_replace_durable_validation(self):
        prefix, durable = fixture()
        prefix['receipts'][0]['source']['kind'] = 'c1'
        prefix['receipts'][0]['source']['trusted'] = True
        with self.assertRaisesRegex(qr.PrefixBlocked, 'c1_durable_origin_missing'):
            build(prefix, durable)

    def test_provider_whitespace_is_replayed_without_normalizing_raw_rewrite(self):
        prefix, durable = fixture(padded=True)
        self.assertTrue(prefix['nodes'][-1]['raw_rewrite'].startswith(' \n'))
        self.assertEqual(build(prefix, durable)['status'], 'ready')

    def test_usage_unknown_is_not_eligible(self):
        prefix, durable = fixture()
        call = prefix['receipts'][0]
        call['result']['usage'] = None
        durable[call['request_id']]['result']['usage'] = None
        with self.assertRaisesRegex(qr.PrefixBlocked, 'upstream_missing_usage'):
            build(prefix, durable)

    def test_fixed_300_manifest_keeps_missing_questions(self):
        cohort = json.loads((qr.PROTOCOL_DIR / 'COHORT.qid_only.json').read_text(encoding='utf8'))
        manifest = qr.prepare_manifest(cohort, {}, {}, expected_namespace='frozen', bindings={}, receipt_resolver=lambda c: None)
        self.assertEqual(len(manifest['rows']), 300)
        self.assertEqual(manifest['planned_final_conditions'], 600)
        self.assertEqual({r['status'] for r in manifest['rows']}, {'pending_upstream'})
        self.assertEqual(manifest['question_replacements'], 0)

    def test_overflow_is_held_without_truncation(self):
        prefix, durable = fixture()
        pair = qr.build_pair(prefix, expected_case=prefix['case'], expected_namespace=prefix['resource_namespace'],
                             expected_prefix_sha256=qr.fingerprint(prefix), receipt_resolver=lambda c: durable[c['request_id']],
                             token_counter=lambda _: qr.common.CONTEXT)
        self.assertEqual(pair['status'], 'input_overflow')
        self.assertFalse(pair['arms']['Raw']['input_check']['truncated'])


class FakeTransport:
    def __init__(self, unknown_first=False, missing_usage=False):
        self.dataset, self.qid, self.method = 'hotpotqa', 'test-qid', qr.FINAL_METHOD
        self.calls = []
        self.unknown_first = unknown_first
        self.missing_usage = missing_usage

    def request_id_for(self, stage, prompt, max_tokens=None):
        return 'final-rid-' + stage

    def ask(self, stage, prompt, max_tokens):
        self.calls.append((stage, prompt, max_tokens))
        if self.unknown_first and len(self.calls) == 1:
            return 'final-rid-' + stage, {'status': 'unknown', 'usage': None}
        result = {'status': 'success', 'answer': 'A preserved final answer.', 'finish_reason': 'stop',
                  'returned_model': qr.common.MODEL, 'usage': {'prompt_tokens': 200, 'completion_tokens': 8}}
        if self.missing_usage:
            result['usage'] = None
        return 'final-rid-' + stage, result


class ExecutionTests(unittest.TestCase):
    def test_guard_prevents_dispatch(self):
        pair = build(*fixture())
        transport = FakeTransport()
        def guard():
            raise RuntimeError('paused')
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            with self.assertRaisesRegex(RuntimeError, 'paused'):
                qr.run_pair(pair, transport, temp, guard=guard)
        self.assertEqual(transport.calls, [])

    def test_each_arm_checks_guard_and_success_is_not_reissued(self):
        pair = build(*fixture())
        transport = FakeTransport()
        guards = []
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            first = qr.run_pair(pair, transport, temp, guard=lambda: guards.append(True))
            second = qr.run_pair(pair, transport, temp, guard=lambda: guards.append(True))
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(guards), 2)
        self.assertEqual(first, second)
        self.assertTrue(second['complete_pair'])
        for arm in second['arms'].values():
            self.assertEqual(arm['arm_logical_cost_cny'], arm['final_logical_cost_cny'] + pair['upstream']['logical_cost_cny'])
            self.assertEqual(arm['new_upstream_cost_cny'], 0)

    def test_single_arm_unknown_recovery_preserves_other_arm_without_resubmit(self):
        pair = build(*fixture())
        transport = FakeTransport(unknown_first=True)
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            first = qr.run_pair(pair, transport, temp, guard=lambda: None)
            unknown_arm = pair['arm_order'][0]
            good_arm = pair['arm_order'][1]
            saved_good = deepcopy(first['arms'][good_arm])
            self.assertIsNone(first['arms'][unknown_arm]['final_logical_cost_cny'])
            self.assertIsNone(first['arms'][unknown_arm]['score'])
            qr.run_pair(pair, transport, temp, guard=lambda: None)
            self.assertEqual(len(transport.calls), 2)
            result = {'status': 'success', 'answer': 'Recovered provider answer.', 'finish_reason': 'stop',
                      'returned_model': qr.common.MODEL, 'usage': {'prompt_tokens': 201, 'completion_tokens': 9}}
            receipt = {'dataset': pair['dataset'], 'qid': pair['qid'], 'stage': f'qa_raw/{unknown_arm}/final',
                       'method': qr.FINAL_METHOD, 'request_id': first['arms'][unknown_arm]['request_id'],
                       'payload': pair['arms'][unknown_arm]['payload'], 'result': result}
            recovered = qr.run_pair(pair, transport, temp, guard=lambda: None, receipt_lookup=lambda rid: receipt)
        self.assertTrue(recovered['complete_pair'])
        self.assertEqual(recovered['arms'][good_arm], saved_good)
        self.assertEqual(len(transport.calls), 2)

    def test_missing_usage_success_is_held_not_zero(self):
        pair = build(*fixture())
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            state = qr.run_pair(pair, FakeTransport(missing_usage=True), temp, guard=lambda: None)
        for result in state['arms'].values():
            self.assertEqual(result['status'], 'technical_held')
            self.assertIsNone(result['score'])
            self.assertIsNone(result['final_logical_cost_cny'])

    def test_length_is_algorithm_zero_with_known_spend(self):
        pair = build(*fixture())
        result = {'status': 'algorithm_failure', 'answer': 'partial', 'finish_reason': 'length',
                  'returned_model': qr.common.MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': qr.common.MAX_OUTPUT}}
        value = qr._settle(result, pair, 'QA', 'rid', None)
        self.assertEqual(value['status'], 'algorithm_failure')
        self.assertEqual(value['score']['f1'], 0.)
        self.assertGreater(value['final_logical_cost_cny'], 0)

    def test_short_length_is_integrity_hold(self):
        pair = build(*fixture())
        result = {'status': 'integrity_failure', 'answer': 'partial', 'finish_reason': 'length',
                  'returned_model': qr.common.MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': 40}}
        value = qr._settle(result, pair, 'QA', 'rid', None)
        self.assertEqual(value['status'], 'technical_held')
        self.assertIsNone(value['score'])

    def test_false_guard_and_crash_before_return_are_safe(self):
        pair = build(*fixture())
        transport = FakeTransport()
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            with self.assertRaisesRegex(qr.common.IntegrityError, 'denied'):
                qr.run_pair(pair, transport, temp, guard=lambda: False)
        self.assertEqual(transport.calls, [])
        class CrashTransport(FakeTransport):
            def ask(self, stage, prompt, max_tokens):
                super().ask(stage, prompt, max_tokens)
                raise RuntimeError('simulated controller crash after submission')
        crashed = CrashTransport()
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            with self.assertRaisesRegex(RuntimeError, 'simulated controller crash'):
                qr.run_pair(pair, crashed, temp, guard=lambda: None)
            saved = json.loads(next(Path(temp).glob('*.json')).read_text(encoding='utf8'))
            first_arm = pair['arm_order'][0]
            rid = saved['arms'][first_arm]['request_id']
            self.assertTrue(rid)
            recovered = {'dataset': pair['dataset'], 'qid': pair['qid'], 'method': qr.FINAL_METHOD,
                         'stage': f'qa_raw/{first_arm}/final', 'request_id': rid,
                         'payload': pair['arms'][first_arm]['payload'],
                         'result': {'status': 'success', 'answer': 'Recovered.', 'finish_reason': 'stop',
                                    'returned_model': qr.common.MODEL,
                                    'usage': {'prompt_tokens': 100, 'completion_tokens': 4}}}
            state = qr.run_pair(pair, transport, temp, guard=lambda: None, receipt_lookup=lambda _: recovered)
        self.assertTrue(state['complete_pair'])
        self.assertEqual(len(crashed.calls), 1)
        self.assertEqual(len(transport.calls), 1)

    def test_verified_failed_request_retries_are_bounded(self):
        pair = build(*fixture())
        class FailedTransport(FakeTransport):
            def ask(self, stage, prompt, max_tokens):
                super().ask(stage, prompt, max_tokens)
                return self.request_id_for(stage, prompt), {'status': 'failed', 'http_status': 429, 'usage': None}
        transport = FailedTransport()
        with tempfile.TemporaryDirectory(dir=qr.HERE) as temp:
            state = qr.run_pair(pair, transport, temp, guard=lambda: None)
        self.assertEqual(len(transport.calls), 10)
        self.assertFalse(state['complete_pair'])
        self.assertTrue(all(a['score'] is None for a in state['arms'].values()))


if __name__ == '__main__':
    unittest.main()

