"""Offline unit contracts for the terminal goal candidate; no model invocation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from . import candidate
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as legacy


class PromptContracts(unittest.TestCase):
    def setUp(self):
        self.question = 'Where was the husband of Sample Person born?'
        self.history = [{'step_id': 1, 'request': 'Who is the husband of Sample Person?',
                         'answer': 'Sample Person was married to Example Husband.'}]
        self.passages = ('[Chunk 1 | source=Example Husband, pages=1-1]\n'
                         'Example Husband\nExample Husband was born in Example Town. Answer: is text in this passage.')
        self.old = legacy.child(self.question, 2, self.question, self.history,
                                self.passages, [], role='goal', combine=True)

    def test_only_header_instruction_and_terminal_readout_change(self):
        previous = deepcopy(self.history)
        new = candidate.render_goal(self.old)
        expected = self.old.replace(candidate.OLD_SENTENCE_INSTRUCTION,
                                    candidate.NEW_SENTENCE_INSTRUCTION, 1)
        expected = expected[:-len('Answer:')] + candidate.OUTPUT_CONTRACT
        self.assertEqual(new, expected)
        self.assertEqual(self.history, previous)
        self.assertIn(self.passages, new)
        self.assertIn(self.question, new)
        self.assertIn(self.history[0]['answer'], new)
        self.assertIn(legacy.GOAL_RULE, new)
        self.assertIn('lack of evidence does not mean no, none, an empty set or zero.', new)
        self.assertNotIn(candidate.OLD_SENTENCE_INSTRUCTION, new[:len(legacy.CHILD_INSTRUCTIONS)])
        self.assertEqual(candidate.MAX_TOKENS, 128)

    def test_refuses_goal_without_consistency_rule(self):
        raw = legacy.child(self.question, 2, self.question, self.history,
                           self.passages, [], role='goal', combine=False)
        with self.assertRaises(ValueError):
            candidate.render_goal(raw)

    def test_refuses_rewriting_a_different_prompt_header(self):
        with self.assertRaises(ValueError):
            candidate.render_goal('New instructions\n\n' + self.old)


class ParsingContracts(unittest.TestCase):
    def test_complete_labels_extract_only_final_short_answer(self):
        raw = ('Conclusion: Example Husband, the husband of Sample Person, was born in Example Town.\n'
               'Final answer: Example Town')
        parsed = candidate.parse_output(raw)
        self.assertTrue(parsed['valid'])
        self.assertEqual(parsed['prediction'], 'Example Town')
        self.assertEqual(parsed['conclusion'], raw.splitlines()[0][len('Conclusion: '):])
        self.assertEqual(parsed['raw'], raw)
        self.assertEqual(parsed['errors'], [])

    def test_crlf_and_outer_whitespace_do_not_change_fact(self):
        parsed = candidate.parse_output(' \r\nConclusion: The count is zero.  \r\nFinal answer: 0 \r\n')
        self.assertTrue(parsed['valid'])
        self.assertEqual(parsed['prediction'], '0')

    def test_no_none_zero_and_abstention_are_not_mechanically_dropped(self):
        for text in ['0', 'No', 'None', 'Cannot be determined']:
            with self.subTest(text=text):
                parsed = candidate.parse_output('Conclusion: The supplied result is ' + text + '.\nFinal answer: ' + text)
                self.assertTrue(parsed['valid'])
                self.assertEqual(parsed['prediction'], text)

    def test_format_failures_never_become_predictions(self):
        outputs = [
            '', 'Example Town',
            'Conclusion: A fact.',
            'Final answer: Example Town\nConclusion: A fact.',
            'Conclusion: A fact.\n\nFinal answer: Example Town',
            'Conclusion: A fact.\nFinal answer: Example Town\nExtra explanation.',
            'Conclusion: A fact.\nFinal answer: Example Town\nFinal answer: Another town',
            '```text\nConclusion: A fact.\nFinal answer: Example Town\n```',
            'Conclusion:\nFinal answer: Example Town',
            'Conclusion: A fact.\nFinal answer:',
            'conclusion: A fact.\nFinal answer: Example Town',
            None,
        ]
        for raw in outputs:
            with self.subTest(raw=raw):
                parsed = candidate.parse_output(raw)
                self.assertFalse(parsed['valid'])
                self.assertIsNone(parsed['prediction'])
                self.assertTrue(parsed['errors'])
                self.assertEqual(parsed['raw'], raw)

    def test_truncated_output_is_not_repaired_or_accepted(self):
        raw = 'Conclusion: A complete fact.\nFinal answer: Example Town'
        parsed = candidate.parse_output(raw, finish_reason='length')
        self.assertFalse(parsed['valid'])
        self.assertIsNone(parsed['prediction'])
        self.assertIn('finish_reason_not_stop', parsed['errors'])
        self.assertEqual(parsed['conclusion'], 'A complete fact.')


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=['']))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    output = root / 'outputs/query_decomposition/hotpot_formal_eval_v1' / here.name / 'OFFLINE_CANDIDATE.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'passed': outcome.wasSuccessful(), 'tests': outcome.testsRun,
              'failures': len(outcome.failures), 'errors': len(outcome.errors),
              'candidate_sha256': hashlib.sha256((here / 'candidate.py').read_bytes()).hexdigest(),
              'new_api_calls': 0, 'new_retrieval_calls': 0, 'gold_read': False,
              'semantic_correctness_not_assessed': True}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
