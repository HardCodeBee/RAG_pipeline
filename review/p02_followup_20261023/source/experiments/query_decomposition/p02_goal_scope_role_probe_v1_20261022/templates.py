"""Single-sentence constraints over the unchanged frozen G2 synthesis prompt."""
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import templates as base

ARMS = ("S0", "S1", "S2")
SCOPE = "Preserve the original question's requested target, predicate and explicit conditions: a work used to identify a person does not require that person's award to be for that work, and asking whether entities share a property does not require their complete property sets to be identical."
ROLE = "Before answering, check that each named entity keeps its stated role in every relation and that the final conclusion preserves the original subject, object, relationship direction, generation and requested attribute, including the distinction between a parent and a grandparent."
SENTENCES = {"S0": "", "S1": SCOPE, "S2": ROLE}


def goal_prompt(record, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    original = base.goal_prompt(record, "G2")
    assert original.startswith(base.G2)
    sentence = SENTENCES[arm]
    # Append exactly one constraint to the instruction prefix; preserve input bytes.
    return original if not sentence else base.G2 + sentence + "\n" + original[len(base.G2):]


final_prompt = base.final_prompt
