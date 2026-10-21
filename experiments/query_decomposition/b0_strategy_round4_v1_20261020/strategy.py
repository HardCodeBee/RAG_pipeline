"""Scope-rule replacement and its interaction with frozen goal evidence."""
from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy as p02
from experiments.query_decomposition.b0_strategy_round3_v1_20261020 import strategy as evidence

ARMS = ['r04_scope_contract', 's04_scope_plus_evidence']
OLD_SCOPE_RULE = 'Do not transfer a fact about another entity or a different comparison to the Original question.'
NEW_SCOPE_RULE = (
    'Auxiliary results may resolve references in the Original question; '
    'they must not change its requested answer target, relation or conditions.'
)
# Driver-facing names; RULE remains an alias used by the offline audit.
OLD_SCOPE_SENTENCE = OLD_SCOPE_RULE
NEW_SCOPE_SENTENCE = NEW_SCOPE_RULE


def replace_scope(prompt):
    assert prompt.count(OLD_SCOPE_RULE) == 1
    assert NEW_SCOPE_RULE not in prompt
    instructions, marker, task = prompt.partition('\nWorked examples:')
    assert marker and instructions.count(OLD_SCOPE_RULE) == 1
    changed = instructions.replace(OLD_SCOPE_RULE, NEW_SCOPE_RULE, 1) + marker + task
    assert changed.replace(NEW_SCOPE_RULE, OLD_SCOPE_RULE, 1) == prompt
    return changed


def render_final(question, nodes, root_packet, readout, arm):
    if arm not in ['parent_p02', *ARMS]:
        raise ValueError('Unknown round-four final arm: ' + arm)
    prompt, context = p02.render_final(question, nodes, root_packet, readout, 'p02_role_headers')
    if arm == 'parent_p02':
        return prompt, context
    if arm == 's04_scope_plus_evidence':
        prompt, context = evidence.render_final(question, nodes, root_packet, readout, 'e03_goal_evidence')
    return replace_scope(prompt), context
