"""Terminal goal output contract; pure prompt rendering and mechanical parsing.

No API, retrieval, score, gold answer, semantic checker, or repair call belongs
here. The existing goal input and consistency rule remain otherwise exact.
"""
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as legacy


MAX_TOKENS = 128
OLD_SENTENCE_INSTRUCTION = (
    'Return one concise factual sentence preserving the entity, relation and necessary qualifiers.'
)
NEW_SENTENCE_INSTRUCTION = (
    'The Conclusion line must be one concise factual sentence preserving the entity, relation and necessary qualifiers.'
)
OUTPUT_CONTRACT = (
    'Output exactly two lines with these labels:\n'
    'Conclusion: <the complete factual sentence>\n'
    'Final answer: <only the short answer to the Original question>\n'
    'In Final answer, preserve the full identifying name and required qualifiers. '
    'For a supported yes/no answer, use exactly Yes or No. '
    'If the final answer is unsupported, use Cannot be determined. '
    'Do not add any other text.'
)


def render_goal(union20_goal_prompt):
    """Transform the actual goal prompt; never reconstruct its input materials.

    Requires the complete union20 goal prompt, already including GOAL_RULE.
    Only the initial factual-sentence instruction and final ``Answer:`` suffix
    change. Ordinary questions, evidence and prior answers are not rewritten.
    """
    if not isinstance(union20_goal_prompt, str):
        raise TypeError('Expected complete goal prompt string')
    if not union20_goal_prompt.startswith(legacy.CHILD_INSTRUCTIONS + '\n\n'):
        raise ValueError('Expected unchanged legacy goal instructions')
    suffix = legacy.GOAL_RULE + '\n\nAnswer:'
    if not union20_goal_prompt.endswith(suffix):
        raise ValueError('Expected complete g01 GOAL_RULE and final Answer: suffix')
    prefix_length = len(legacy.CHILD_INSTRUCTIONS)
    header = union20_goal_prompt[:prefix_length]
    if header.count(OLD_SENTENCE_INSTRUCTION) != 1:
        raise ValueError('Expected one known factual-sentence instruction in header')
    header = header.replace(OLD_SENTENCE_INSTRUCTION, NEW_SENTENCE_INSTRUCTION, 1)
    return header + union20_goal_prompt[prefix_length:-len('Answer:')] + OUTPUT_CONTRACT


def parse_output(raw, finish_reason='stop'):
    """Return a logged format verdict, not a semantic correctness assessment.

    Leading/trailing whitespace and field-edge whitespace are harmless.
    Otherwise the two labelled lines must appear exactly once, in order.
    Malformed/truncated output is never propagated as a final prediction and
    is not repaired, reformatted, inferred, or compared with a gold answer.
    Supported ``0``, ``No`` and ``None`` remain ordinary valid answer strings.
    """
    errors = []
    if finish_reason != 'stop':
        errors.append('finish_reason_not_stop')
    conclusion, prediction = None, None
    if not isinstance(raw, str):
        errors.append('raw_not_string')
        return {'valid': False, 'prediction': None, 'conclusion': None,
                'errors': errors, 'raw': raw, 'finish_reason': finish_reason}
    lines = raw.strip().splitlines()
    if len(lines) != 2:
        errors.append('expected_exactly_two_lines')
    if not lines or not lines[0].startswith('Conclusion:'):
        errors.append('expected_conclusion_label_on_first_line')
    else:
        conclusion = lines[0][len('Conclusion:'):].strip()
        if not conclusion:
            errors.append('empty_conclusion')
    if len(lines) < 2 or not lines[1].startswith('Final answer:'):
        errors.append('expected_final_answer_label_on_second_line')
    else:
        prediction = lines[1][len('Final answer:'):].strip()
        if not prediction:
            errors.append('empty_final_answer')
    valid = not errors
    return {'valid': valid, 'prediction': prediction if valid else None,
            'conclusion': conclusion, 'errors': errors, 'raw': raw,
            'finish_reason': finish_reason}
