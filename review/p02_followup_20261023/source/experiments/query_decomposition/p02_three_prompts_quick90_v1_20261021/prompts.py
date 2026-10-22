"""Render the three user-supplied templates without changing their wording."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDINARY = (HERE / 'ordinary.frozen.txt').read_text(encoding='utf8').rstrip('\n')
GOAL = (HERE / 'goal.frozen.txt').read_text(encoding='utf8').rstrip('\n')
FINAL = (HERE / 'final.frozen.txt').read_text(encoding='utf8').rstrip('\n')
FINAL_PREFIX, FINAL_LIVE = FINAL.split('Actual task:\n\n', 1)
FINAL_BODY, FINAL_READOUT = FINAL_LIVE.split('\n\nFinal response requirements:\n', 1)


def prior_block(records):
    return '\n\n'.join(f"#{r['step_id']} Question: {r['request']}\nAnswer: {r['answer']}" for r in records) if records else 'No previous question-answer pairs are available.'


def child(question, sid, request, dependencies, context, examples, *, role='ordinary'):
    values = {'ORIGINAL_QUESTION': question, 'PRIOR_QA_BLOCK': prior_block(dependencies),
              'STEP_ID': sid, 'CURRENT_SUBQUESTION': request,
              'STEP_RETRIEVED_PASSAGES': context, 'GOAL_RETRIEVED_PASSAGES': context}
    if role == 'goal':
        assert request == question
        return GOAL.format_map(values)
    assert role == 'ordinary'
    return ORDINARY.format_map(values)


def final(question, context, examples, readout):
    return (FINAL_PREFIX + 'Actual task:\n\nOriginal question:\n' + question
            + '\n\nContext:\n' + context + '\n\nFinal response requirements:\n' + FINAL_READOUT)
