"""Three minimal, parent-matched interventions, with no new demonstrations."""
from experiments.query_decomposition.p02_goal_target_transfer_new60_v1_20261022 import templates as old

ARMS = ('G0', 'A', 'GF', 'P', 'R')
PAIRS = (('G0','A'), ('GF','P'), ('GF','R'))
ALTERNATIVE = """A missing answer to a planned subquestion does not by itself make the
Original question unanswerable. Use another route through the available facts
when they already suffice to answer the Original question, without inventing
missing entities, quantities or relations."""
PRESERVATION = """In the candidate-answer sentence, retain the supported answers and necessary
qualifiers established in the paragraph that still satisfy the Original
question. Exclude one only when the question or evidence rules it out; do not
invent a preference or extra restriction to choose a single answer from
several supported alternatives."""
READOUT = """When using a candidate synthesis, consider its relevant factual statements
throughout the paragraph; its closing answer has no special authority. If that
closing answer drops a supported answer or necessary qualifier from the body,
resolve the choice using the Original question and evidence, without adopting
an extra preference or restriction invented by the synthesis. Keep the
existing final-short-answer requirements."""


def goal_prompt(record, arm):
    if arm in ('G0','A'):
        prompt=old.goal_prompt(record,'G0')
        if arm=='A':
            before,after=prompt.split('\n\nOriginal question:\n',1)
            prompt=before+'\n\n'+ALTERNATIVE+'\n\nOriginal question:\n'+after
        return prompt
    prompt=old.goal_prompt(record,'GF')
    if arm=='P':
        marker='\nFictional worked examples:\n'
        assert prompt.count(marker)==1
        prompt=prompt.replace(marker,'\n'+PRESERVATION+'\n'+marker,1)
    elif arm not in ('GF','R'):
        raise ValueError(arm)
    return prompt


def final_prompt(record, goal, config, examples, arm):
    prompt,context=old.prior.base.final_prompt(record,goal,config,examples)
    if arm=='R':
        marker='\nWorked examples:\n\n'
        assert prompt.count(marker)==1
        prompt=prompt.replace(marker,'\n'+READOUT+'\n'+marker,1)
    return prompt,context
