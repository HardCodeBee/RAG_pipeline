"""Target/scope and lossless candidate-presentation interventions only."""
import re
from experiments.query_decomposition.p02_result_binding_operator_probe_v1_20261022 import templates as prior

ARMS = ('G0', 'T', 'GF', 'GF_SPLIT')
TARGET_GUIDANCE = """Keep the requested relation separate from the description used to identify its
subject. A book, role, employer or event that identifies a person does not by
itself restrict every property of that person to that book, role, employer or
event. Apply a restriction when the question actually states it; do not remove
an explicit restriction. Use prior question-answer pairs to resolve the subject,
then answer the exact requested relation. Do not invent a stronger condition.

Fictional paired examples (all names and facts below are invented):

Example 1
Original question: Which prize did the author of Cloud Lantern receive?
Prior answer: Neri Vale wrote Cloud Lantern.
Passages: Neri Vale received the Silver Reed Prize for River Glass.
Answer: Neri Vale, the author of Cloud Lantern, received the Silver Reed Prize.

Example 2
Original question: Which prize did Neri Vale receive for Cloud Lantern?
Prior answer: Neri Vale wrote Cloud Lantern.
Passages: Neri Vale received the Silver Reed Prize for River Glass; no award for Cloud Lantern is specified.
Answer: The available information does not establish which prize Neri Vale received for Cloud Lantern.

Example 3
Original question: Which university educated the founder of Soralab?
Prior answer: Iven Mora founded Soralab in 2010.
Passages: Iven Mora graduated from Elun University in 1998.
Answer: Soralab's founder Iven Mora graduated from Elun University.

Example 4
Original question: Which university did Iven Mora attend after founding Soralab?
Prior answer: Iven Mora founded Soralab in 2010.
Passages: Iven Mora graduated from Elun University in 1998; later university attendance is not documented.
Answer: The available information does not establish which university Iven Mora attended after founding Soralab.

End of fictional examples. Apply the original factual-sentence requirements to the actual task below."""


def goal_prompt(record, arm):
    if arm in ('GF', 'GF_SPLIT'):
        return prior.goal_prompt(record, 'GF')
    prompt = record['historical_goal_prompt']
    if arm == 'G0':
        return prompt
    if arm != 'T':
        raise ValueError(arm)
    before, after = prompt.split('\n\nOriginal question:\n', 1)
    return before + '\n\n' + TARGET_GUIDANCE + '\n\nOriginal question:\n' + after


BODY_HEADER = 'Candidate synthesis body (unverified):\n'
CLOSE_HEADER = '\n\nClosing sentence from the same candidate synthesis (unverified):\n'


def split_presentation(text):
    """Keep every original character in order; insert two headings or do nothing.

    Last sentence is an operational boundary, not a claim its content is a
    correct or complete conclusion. No model, reference or question access.
    """
    boundaries = []
    for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])', text):
        prefix = text[:match.start()]
        token = prefix.split()[-1]
        if token in ('Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'St.', 'Jr.', 'Sr.', 'vs.', 'U.S.', 'U.K.'):
            continue
        if re.fullmatch(r'(?:[A-Z]\.)+', token):
            continue
        boundaries.append(match.end())
    if not boundaries:
        return text, dict(changed=False, reason='no_unambiguous_sentence_boundary')
    at = boundaries[-1]
    body, closing = text[:at], text[at:]
    assert body + closing == text
    return BODY_HEADER + body + CLOSE_HEADER + closing, dict(changed=True, split_index=at,
        original_characters=len(text), body=body, closing=closing)


def final_prompt(record, goal_text, config, examples, arm):
    presentation, meta = (split_presentation(goal_text) if arm == 'GF_SPLIT'
                          else (goal_text, dict(changed=False, reason='raw_candidate')))
    prompt, context = prior.base.final_prompt(record, presentation, config, examples)
    return prompt, context, meta
