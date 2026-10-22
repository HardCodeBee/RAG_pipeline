"""Focused offline report tests; synthetic scores, no inference or retrieval."""
import unittest
import sqlite3
import tempfile
from pathlib import Path

from . import runtime_report as rr
from . import runtime_config as cfg


class ReportTests(unittest.TestCase):
    def test_readonly_prediction_reader_releases_windows_file_handle(self):
        with tempfile.TemporaryDirectory(dir=cfg.HERE) as directory:
            path = Path(directory) / 'checkpoint.sqlite3'
            with rr.closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE data(kind TEXT,key TEXT,value TEXT)')
                db.execute("INSERT INTO data VALUES('prediction','q','{\"status\": \"pending\"}')")
                db.commit()
            self.assertEqual(rr._predictions(directory), {'q': {'status': 'pending'}})
            path.unlink()  # Fails on Windows if the read-only connection is left open.

    def test_technical_missing_is_not_zero_algorithm_failure_is_zero(self):
        refs = {str(i): {'answer': 'right'} for i in range(500)}
        predictions = {'0': {'status': 'success', 'prediction': 'right'},
                       '1': {'status': 'algorithm_failure', 'prediction': None},
                       '2': {'status': 'technical_pending', 'prediction': None}}
        rows = rr.score_rows(list(refs), predictions, refs,
                             lambda p, r: {'f1': float(p == r['answer']), 'em': float(p == r['answer'])})
        self.assertEqual(rows[0]['f1'], 1.)
        self.assertEqual(rows[1]['f1'], 0.)
        self.assertIsNone(rows[2]['f1'])
        summary = rr.quality_summary(rows, planned=500)
        self.assertEqual(summary['scorable'], 2)
        self.assertEqual(summary['technical_missing'], 498)
        self.assertIsNone(summary['quality_for_primary_table'])
        self.assertEqual(summary['fixed_denominator_bounds']['f1'], [.002, .998])
        primary = rr.main_quality_summary(rows, expected_qids=list(refs))
        self.assertEqual(primary['status'], 'not_ready_500')
        self.assertIsNone(primary['F1'])
        self.assertIsNone(primary['EM'])
        for forbidden in ('scorable', 'technical_missing', 'fixed_denominator_bounds', 'quality_for_primary_table'):
            self.assertNotIn(forbidden, primary)

    def test_complete500_populates_table_and_unverified_success_stays_pending(self):
        refs = {str(i): {} for i in range(500)}
        predictions = {qid: {'status': 'success', 'prediction': 'x'} for qid in refs}
        scorer = lambda p, r: {'f1': .5, 'em': 0.}
        rows = rr.score_rows(list(refs), predictions, refs, scorer)
        self.assertEqual(rr.quality_summary(rows, planned=500)['quality_for_primary_table'], {'f1': .5, 'em': 0.})
        self.assertEqual(rr.main_quality_summary(rows, expected_qids=list(refs))['status'], 'complete500')
        self.assertEqual(rr.main_quality_summary(rows, expected_qids=list(refs))['F1'], .5)
        held = rr.score_rows(list(refs), predictions, refs, scorer, verify_success=lambda q, r: False)
        self.assertIsNone(rr.quality_summary(held, planned=500)['quality_for_primary_table'])

    def test_paired_bounds_retain_known_other_arm_and_fixed100(self):
        qa = [{'qid': str(i), 'status': 'technical_pending', 'f1': None, 'em': None} for i in range(100)]
        raw = [dict(r) for r in qa]
        qa[0].update(status='success', f1=1., em=1.)
        raw[0].update(status='algorithm_failure', f1=0., em=0.)
        qa[1].update(status='success', f1=1., em=1.)
        summary = rr.paired_summary(qa, raw, planned=100)
        self.assertEqual(summary['complete_pairs'], 1)
        self.assertEqual(summary['paired_difference_complete_cases']['f1'], 1.)
        self.assertEqual(summary['paired_difference_fixed_denominator_bounds']['f1'], [-.97, 1.])
        self.assertEqual(len(summary['details']), 100)
        self.assertIsNone(summary['details'][1]['Raw_f1'])

    def test_bootstrap_is_frozen_paired_and_equal_dataset_weight(self):
        values = {'hotpotqa': [1.]*100, '2wiki': [-1.]*100, 'musique': [.3]*100}
        result = rr.paired_bootstrap(values)
        self.assertEqual(result['iterations'], 10000)
        self.assertEqual(result['seed'], 20261023)
        self.assertAlmostEqual(result['equal_dataset_mean']['f1_difference'], .1)
        for bound in result['equal_dataset_mean']['percentile_95_ci']:
            self.assertAlmostEqual(bound, .1)
        with self.assertRaises(ValueError):
            rr.paired_bootstrap({ds: [] for ds in cfg.DATASETS})
        with self.assertRaises(ValueError):
            rr.paired_bootstrap(values, seed=1)

    def test_transport_algorithm_failure_receipt_retains_qa_zero(self):
        from .runtime_transport import RuntimeTransport
        for finish, answer, tokens in [('stop', '', 0), ('length', 'partial', cfg.common.MAX_OUTPUT)]:
            with self.subTest(finish=finish):
                result = RuntimeTransport._validate_result(
                    {'status': 'success', 'answer': answer, 'finish_reason': finish,
                     'returned_model': cfg.common.MODEL,
                     'usage': {'prompt_tokens': 100, 'completion_tokens': tokens}},
                    cfg.common.MODEL, 'qa_raw/QA/final')
                self.assertEqual(result['status'], 'algorithm_failure')
                receipt = {'dataset': 'hotpotqa', 'method': rr.FINAL_METHOD, 'qid': 'q',
                           'stage': 'qa_raw/QA/final', 'status': result['status'], 'result': result}
                class Reader:
                    def receipt(self, rid):
                        return receipt
                prediction = {'status': 'algorithm_failure', 'prediction': None, 'request_id': 'r'}
                verify = lambda q, r: rr._verified(Reader(), 'hotpotqa', rr.FINAL_METHOD, q, r,
                    terminal='algorithm_failure', expected_stage='qa_raw/QA/final')
                rows = rr.score_rows(['q'], {'q': prediction}, {'q': {}}, lambda p, r: None,
                                     verify_algorithm_failure=verify)
                self.assertEqual(rows[0]['f1'], 0.)
                receipt['result']['usage'] = None
                held = rr.score_rows(['q'], {'q': prediction}, {'q': {}}, lambda p, r: None,
                                     verify_algorithm_failure=verify)
                self.assertIsNone(held[0]['f1'])


if __name__ == '__main__':
    unittest.main()
