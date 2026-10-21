import unittest
from unittest.mock import patch

from . import run_execution_v10_baselines_only as scope


class BaselineScopeTests(unittest.TestCase):
    def setUp(self):
        self.called = []
        called = self.called

        class Owner:
            def execute(self, *args):
                called.append(args)
                return 'original-result'

        self.owner = Owner
        scope.install_baseline_guard(Owner)

    @patch.object(scope, 'require_policy')
    def test_baseline_arguments_are_preserved(self, policy):
        args = ('musique', 'formal', 'b03_react_dense', {'id': 'k10_s5'}, [{'qid': 'example'}])
        self.assertEqual(self.owner().execute(*args), 'original-result')
        self.assertEqual(self.called, [args])

    @patch.object(scope, 'require_policy')
    def test_our_method_control_and_ablations_cannot_reach_original(self, policy):
        for method in ['m00_s02_child4_faithful', 'c01_same_final_reader',
                       'a02_short_qa', 'a03_no_goal_retrieval', 'a04_no_final_constraints']:
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                self.owner().execute('musique', 'formal', method, {}, [])
        self.assertEqual(self.called, [])

    @patch.object(scope, 'require_policy')
    def test_ablation_stage_is_blocked_even_for_baseline_named_method(self, policy):
        with self.assertRaises(RuntimeError):
            self.owner().execute('musique', 'ablation', 'b00_rag_top10_v1', {}, [])
        self.assertEqual(self.called, [])


if __name__ == '__main__':
    unittest.main()
