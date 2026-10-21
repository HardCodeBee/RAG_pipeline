"""Two isolated final-reader successors to p02; no new model/retrieval call."""
from collections import Counter

from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy as p02

ARMS = ['q03_question_last', 'e03_goal_evidence']
UNEXECUTED_ALTERNATIVES = ['e03_evidence_first']
GOAL_EVIDENCE_HEADER = '\n\nAdditional retrieved passage for the Original question\n\n'


def goal_extra_document(nodes, root_packet):
    """First goal-reader rank absent from the original Top10, without scores/gold."""
    assert nodes and nodes[-1]['role'] == 'goal'
    assert sum(n['role'] == 'goal' for n in nodes) == 1
    original_ids = {doc['doc_id'] for doc in root_packet['docs']}
    docs = nodes[-1]['reader_packet']['docs']
    assert len(original_ids) == len(root_packet['docs']) == 10
    assert len(docs) == len({doc['doc_id'] for doc in docs}) == 10
    assert [doc['rank'] for doc in docs] == list(range(1, 11))
    for doc in docs:
        if doc['doc_id'] not in original_ids:
            return doc
    return None


def actual_tail(question, context, readout):
    return ('Actual task:\n\nOriginal question:\n' + question +
            '\n\nContext:\n' + context + '\n\n' + readout)


def render_final(question, nodes, root_packet, readout, arm):
    if arm not in ['parent_p02', *ARMS, *UNEXECUTED_ALTERNATIVES]:
        raise ValueError('Unknown round-three final arm: ' + arm)
    parent_prompt, parent_context = p02.render_final(
        question, nodes, root_packet, readout, 'p02_role_headers')
    if arm == 'parent_p02':
        return parent_prompt, parent_context
    old_tail = actual_tail(question, parent_context, readout)
    assert parent_prompt.endswith(old_tail), 'Frozen actual-task template differs'
    prefix = parent_prompt[:-len(old_tail)]
    if arm == 'q03_question_last':
        context = parent_context
        tail = ('Actual task:\n\nContext:\n' + context +
                '\n\nOriginal question:\n' + question + '\n\n' + readout)
    elif arm == 'e03_evidence_first':
        helper = p02.helper(nodes, 'p02_role_headers')
        passages = p02.b0.passages(root_packet)
        assert helper and passages, 'Full p02 helper and original Top10 must be retained'
        assert parent_context == helper + '\n\n' + passages
        context = passages + '\n\n' + helper
        tail = actual_tail(question, context, readout)
    else:
        extra = goal_extra_document(nodes, root_packet)
        if extra is None:
            return parent_prompt, parent_context
        context = parent_context + GOAL_EVIDENCE_HEADER + extra['block']
        tail = actual_tail(question, context, readout)
    prompt = prefix + tail
    if arm != 'e03_goal_evidence':
        assert len(prompt) == len(parent_prompt)
        assert Counter(prompt) == Counter(parent_prompt), 'Only text order may change'
    return prompt, context
