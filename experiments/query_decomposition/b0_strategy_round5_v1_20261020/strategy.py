"""One format factor: worked-example QA role headings match frozen p02."""
from copy import deepcopy
import json

from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy as p02

ARMS = ['e05_example_role_headers']
EXAMPLE_PATH = p02.b0.ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json'
ACTUAL_MARKER = '\n\nActual task:\n\n'


def aligned_examples(examples):
    """Add p02 headings only; keep docs-first order and every example value."""
    assert len(examples) == 4
    changed = deepcopy(examples)
    for old, new in zip(examples, changed):
        qas = old['qas']
        assert qas and qas[-1]['request'] == old['question']
        assert [q['step_id'] for q in qas] == list(range(1, len(qas) + 1))
        generic = p02.b0.legacy_child.render_helper(qas, None, 'Answer')
        assert old['context'].endswith(generic)
        prefix = old['context'][:-len(generic)]
        assert prefix.endswith('\n\n') and prefix.startswith('[Chunk 1 |')
        nodes = [{**qa, 'role': 'goal' if i == len(qas) - 1 else 'ordinary'} for i, qa in enumerate(qas)]
        new['context'] = prefix + p02.helper(nodes, 'p02_role_headers')
        assert {k: v for k, v in new.items() if k != 'context'} == {k: v for k, v in old.items() if k != 'context'}
    return changed


def render_final(question, nodes, root_packet, readout, arm):
    if arm not in ['parent_p02', *ARMS]:
        raise ValueError('Unknown round-five arm: ' + arm)
    parent_prompt, context = p02.render_final(question, nodes, root_packet, readout, 'p02_role_headers')
    if arm == 'parent_p02':
        return parent_prompt, context
    examples = json.loads(EXAMPLE_PATH.read_text(encoding='utf-8'))
    rendered = p02.b0.final_prompts.final(question, context, aligned_examples(examples))
    assert rendered.endswith(p02.b0.final_prompts.FINAL_READOUT)
    rendered = rendered[:-len(p02.b0.final_prompts.FINAL_READOUT)] + readout
    assert rendered.split(ACTUAL_MARKER, 1)[1] == parent_prompt.split(ACTUAL_MARKER, 1)[1]
    return rendered, context
