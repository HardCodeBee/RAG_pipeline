"""Replace ordinary prompts only; keep the frozen goal renderer exact."""
from pathlib import Path

from p02_multigrain_timing_cachefix_v1.core import child_prompts as frozen

GOAL_RULE = frozen.GOAL_RULE
TEMPLATE = (Path(__file__).parent / 'ordinary_child_prompt.frozen.txt').read_text(encoding='utf8').rstrip('\n')


def child(question, sid, request, dependencies, context, examples, *, role='ordinary'):
    dependencies = list(dependencies)
    if role != 'ordinary':
        return frozen.child(question, sid, request, dependencies, context, examples, role=role)
    # Preserve the frozen input checks and ordered, complete history contract.
    frozen.child_prompt(question, sid, request, dependencies, context)
    records = frozen._ordered_records(dependencies)
    prior = frozen._question_answers(records) if records else 'No previous question-answer pairs are available.'
    return TEMPLATE.format_map({
        'ORIGINAL_QUESTION': question,
        'PRIOR_QA_BLOCK': prior,
        'STEP_ID': sid,
        'CURRENT_SUBQUESTION': request,
        'STEP_RETRIEVED_PASSAGES': context,
    })
