"""Pure native P02 goal/formatter/final renderers; no model or cache access."""
from copy import deepcopy
from p02_multigrain_timing_cachefix_v1.core.pipeline import passages, render_final

ARMS = ('L0', 'L1')
MARKER = '\n\nOriginal question:\n'
ROLE = 'Before answering, check that each named entity keeps its stated role in every relation and that the final conclusion preserves the original subject, object, relationship direction, generation and requested attribute, including the distinction between a parent and a grandparent.'
GOAL_CAP, FORMATTER_CAP, FINAL_CAP = 128, 128, 64


def goal_payload(record, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    payload = deepcopy(record['original_goal_payload'])
    prompt = payload['messages'][0]['content']
    assert len(payload['messages']) == 1 and payload['messages'][0]['role'] == 'user'
    assert payload['max_tokens'] == GOAL_CAP and prompt.count(MARKER) == 1 and ROLE not in prompt
    if arm == 'L1':
        payload['messages'][0]['content'] = prompt.replace(MARKER, '\n' + ROLE + MARKER, 1)
    return payload


def needs_formatter(answer):
    # Identical to native Pipeline.execute after its answer.strip().
    return len(answer.strip().split()) < 4


def formatter_prompt(record, answer):
    # Keep the native wording, current goal question and exact reader passages.
    return (
        'Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
        'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
        'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
        + 'Question:\n' + record['goal_inputs']['request']
        + '\n\nCandidate answer:\n' + answer.strip()
        + '\n\nContext:\n' + passages(record['goal_inputs']['reader_packet'])
    )


def final_prompt(record, formatted_goal_answer, config, examples):
    goal = deepcopy(record['goal_inputs'])
    goal['answer'] = formatted_goal_answer.strip()
    return render_final(record['question'], deepcopy(record['ordinary_nodes']) + [goal],
                        record['final_root_packet'], config['readout'], examples)
