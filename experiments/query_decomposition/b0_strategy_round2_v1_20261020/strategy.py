"""Minimal round-two strategies. No model calls, routing, or query-specific rules."""
from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.b0_targeted_optimization_v2_20261020.experiment import FINAL_CHECK

ARMS = ['g02_evidence_goal', 'p02_role_headers', 's03_g01_plus_f01']
GOAL_EVIDENCE_RULE = (
    'Base the requested conclusion on the supporting facts, not on agreement with earlier conclusions. '
    'If a required premise remains unsupported, state that gap instead of assigning it a factual value.'
)


def goal_prompt(raw_B0_goal_prompt, arm):
    assert raw_B0_goal_prompt.endswith('Answer:')
    rule = GOAL_EVIDENCE_RULE if arm == 'g02_evidence_goal' else b0.legacy_child.GOAL_RULE
    return raw_B0_goal_prompt[:-len('Answer:')] + rule + '\n\nAnswer:'


def helper(nodes, arm):
    pairs = [{k: n[k] for k in ['step_id', 'request', 'answer']} for n in nodes]
    if arm != 'p02_role_headers':
        return b0.legacy_child.render_helper(pairs, None, 'Answer') if pairs else ''
    ordinary = [p for p, n in zip(pairs, nodes) if n['role'] == 'ordinary']
    goals = [p for p, n in zip(pairs, nodes) if n['role'] == 'goal']
    assert len(goals) == 1 and len(ordinary) + 1 == len(pairs)
    assert pairs == ordinary + goals, 'Do not change any question-answer order'
    sections = ['Auxiliary question-answer pairs']
    if ordinary:
        sections += ['Intermediate candidate results', '\n\n'.join(
            f"#{p['step_id']} Question: {p['request']}\nAnswer: {p['answer']}" for p in ordinary)]
    sections += ['Candidate conclusion for the Original question', '\n\n'.join(
        f"#{p['step_id']} Question: {p['request']}\nAnswer: {p['answer']}" for p in goals)]
    return '\n\n'.join(sections)


def render_final(question, nodes, root_packet, readout, arm):
    import json
    block = helper(nodes, arm)
    context = block + '\n\n' + b0.passages(root_packet) if block else b0.passages(root_packet)
    examples = json.loads((b0.ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json').read_text(encoding='utf8'))
    prompt = b0.final_prompts.final(question, context, examples)
    assert prompt.endswith(b0.final_prompts.FINAL_READOUT)
    suffix = (FINAL_CHECK if arm == 's03_g01_plus_f01' else '') + readout
    return prompt[:-len(b0.final_prompts.FINAL_READOUT)] + suffix, context
