"""Exactly one prompt edit per arm; arms execute sequentially."""
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import child_prompts as parent

GOAL_RULE = parent.GOAL_RULE
ARM = None
RESOLVE = 'Resolve each #N from its numbered prior question and answer. '
ONLY = 'using only the retrieved passages'


def child(*args, role='ordinary', **kwargs):
    prompt = parent.child(*args, role=role, **kwargs)
    if ARM == 'A_no_only' and role == 'ordinary':
        assert prompt.count(ONLY) == 1
        return prompt.replace(ONLY, 'using the retrieved passages', 1)
    if ARM == 'B_no_resolve' and role == 'goal':
        assert prompt.count(RESOLVE) == 1
        return prompt.replace(RESOLVE, '', 1)
    assert ARM in {'A_no_only', 'B_no_resolve'}
    return prompt
