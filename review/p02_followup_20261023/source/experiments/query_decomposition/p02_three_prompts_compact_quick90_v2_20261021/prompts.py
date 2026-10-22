"""Exact rendering of the user's second joint quick90 prompt bundle."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDINARY = (HERE / 'ordinary.frozen.txt').read_text(encoding='utf8').rstrip('\n')
GOAL = (HERE / 'goal.frozen.txt').read_text(encoding='utf8').rstrip('\n')
FINAL = (HERE / 'final.frozen.txt').read_text(encoding='utf8').rstrip('\n')


def prior_block(records):
    return '\n\n'.join(f"#{r['step_id']} Question: {r['request']}\nAnswer: {r['answer']}" for r in records) if records else 'No previous question-answer pairs are available.'


def child(question, sid, request, dependencies, context, examples, *, role='ordinary'):
    values = {'ORIGINAL_QUESTION': question, 'PRIOR_QA_BLOCK': prior_block(dependencies),
              'STEP_ID': sid, 'GOAL_STEP_ID': sid, 'CURRENT_SUBQUESTION': request,
              'STEP_RETRIEVED_PASSAGES': context, 'GOAL_RETRIEVED_PASSAGES': context}
    assert role in {'ordinary', 'goal'}
    if role == 'goal':
        assert request == question
    return (GOAL if role == 'goal' else ORDINARY).format_map(values)


def render_final(question, nodes, passage_text):
    values = {'ORIGINAL_QUESTION': question,
              'INTERMEDIATE_QA_BLOCK': prior_block(nodes[:-1]) if nodes[:-1] else '',
              'GOAL_STEP_ID': nodes[-1]['step_id'], 'GOAL_ANSWER': nodes[-1]['answer'],
              'ORIGINAL_QUERY_RETRIEVED_PASSAGES': passage_text}
    prompt = FINAL.format_map(values)
    body = FINAL.split('Actual task:\n\n', 1)[1]
    context = body.split('\n\nContext:\n', 1)[1].split('\n\nFinal response requirements:\n', 1)[0].format_map(values)
    block = body.split('\n\nContext:\n', 1)[1].split('\n\n{ORIGINAL_QUERY_RETRIEVED_PASSAGES}', 1)[0].format_map(values)
    return prompt, context, block
