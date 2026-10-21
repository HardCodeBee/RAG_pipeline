"""Offline checks for outcome-independent sampling and recovery/cost safety."""
import unittest
from .execution_policy import select_timing_questions, recovery_action, expense_signals


class ExecutionPolicyTests(unittest.TestCase):
    def test_stratified_sample_is_stable_and_shared_without_reading_answers(self):
        questions = [{'qid': str(i), 'question': f'Q{i}'} for i in range(500)]
        metadata = [{'qid': str(i), 'type': 'bridge' if i < 407 else 'comparison',
                     'group_id': str(i)} for i in range(500)]
        sample, quotas = select_timing_questions(questions, metadata, 'hotpotqa')
        self.assertEqual(quotas, {'bridge': 41, 'comparison': 9})
        self.assertEqual(len({r['qid'] for r in sample}), 50)
        reversed_sample, _ = select_timing_questions(questions[::-1], metadata[::-1], 'hotpotqa')
        self.assertEqual(sample, reversed_sample)
        with self.assertRaises(ValueError):
            select_timing_questions(questions, [{**r, 'answer': 'forbidden'} for r in metadata], 'hotpotqa')

    def test_five_attempt_limit_does_not_hide_algorithm_failures(self):
        self.assertEqual(recovery_action('failed', 4), 'retry_confirmed_failure_only')
        self.assertEqual(recovery_action('failed', 5), 'record_and_temporarily_skip')
        self.assertEqual(recovery_action('failed', 6), 'record_and_temporarily_skip')
        self.assertEqual(recovery_action('algorithm_failure', 1), 'preserve_algorithm_failure_zero')

    def test_unknown_and_integrity_failures_do_not_trigger_paid_retries(self):
        self.assertEqual(recovery_action('unknown', 1), 'hold_for_receipt_verification')
        self.assertEqual(recovery_action('in_flight', 5), 'hold_for_receipt_verification')
        self.assertEqual(recovery_action('model_identity_mismatch', 1), 'pause_for_integrity_check')

    def test_single_outlier_and_sustained_cost_increase_are_distinct(self):
        single = expense_signals(.01, .031, [.01] * 20)
        self.assertEqual([r['kind'] for r in single], ['single_query_cost_increase'])
        rolling = expense_signals(.01, .016, [.016] * 20)
        self.assertEqual([r['kind'] for r in rolling], ['rolling_query_cost_increase'])
        self.assertEqual(expense_signals(.01, .01, [.01] * 20), [])

    def test_unknown_cost_cannot_be_reported_as_zero_or_a_rolling_mean(self):
        self.assertEqual(expense_signals(None, .01)[0]['kind'], 'missing_historical_cost_reference')
        self.assertEqual(expense_signals(.01, None)[0]['kind'], 'unknown_query_usage_not_zero')
        values = [.1] * 19 + [None]
        self.assertEqual(expense_signals(.01, .01, values), [])


if __name__ == '__main__':
    unittest.main()
