"""Preserve frozen rules/examples except the approved reference notation edits."""
import re
from p02_multigrain_timing_cachefix_v1.core import child_prompts as frozen

RESOLVE = 'Resolve each #N from its numbered prior question and answer. '
CHILD_INSTRUCTIONS = frozen.CHILD_INSTRUCTIONS.replace(RESOLVE, '', 1)
GOAL_RULE = frozen.GOAL_RULE
EXAMPLE_REPLACEMENTS = {
    'What is the length of #1?': 'What is the length of the river into which Pack Creek flows after passing through Spanish Valley?',
    'How many species does #1 contain other than Callicoma serratifolia?': 'How many species does the genus containing Callicoma serratifolia contain other than Callicoma serratifolia?',
}


def edit_example(text):
    for before, after in EXAMPLE_REPLACEMENTS.items():
        text = text.replace(before, after)
    text = re.sub(r'(?m)^#\d+ Question:', 'Question:', text)
    return re.sub(r'(?m)^Current subquestion #\d+:', 'Current subquestion:', text)


def child(question, sid, request, dependencies, context, examples, *, role='ordinary'):
    dependencies = list(dependencies)
    # Keep original validation and ordered full-history contract.
    frozen.child_prompt(question, sid, request, dependencies, context)
    records = frozen._ordered_records(dependencies)
    prior = '\n\n'.join(f'Question: {q}\nAnswer: {a}' for _, q, a in records) if records else 'No prior question-answer pairs are required.'
    body = '\n'.join(['Original question:', question, '', 'Relevant prior question-answer pairs:', prior, '',
        'Current subquestion:', request, '', 'Retrieved passages:', context, '', 'Answer:'])
    prefix = CHILD_INSTRUCTIONS + '\n\n'
    if role == 'ordinary':
        text = '\n\n'.join('Example ' + str(i + 1) + '\n' + edit_example(row['text']) for i, row in enumerate(examples))
        prefix += 'Worked examples:\n\n' + text + '\n\nActual task:\n'
    return prefix + body
