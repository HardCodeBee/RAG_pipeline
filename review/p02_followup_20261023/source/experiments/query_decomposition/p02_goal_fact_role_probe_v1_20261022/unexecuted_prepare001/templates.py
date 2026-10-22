"""Original G0 facts sentence with one optional, exactly reused ROLE sentence."""
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import templates as base
from experiments.query_decomposition.p02_goal_scope_role_probe_v1_20261022.templates import ROLE

ARMS = ("T0", "T1")
MARKER = "\n\nWorked examples:"


def goal_prompt(record, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    original = base.goal_prompt(record, "G0")
    if arm == "T0":
        return original
    assert original.count(MARKER) == 1
    # Append to the main instruction prefix, preserving every original byte.
    return original.replace(MARKER, "\n" + ROLE + MARKER, 1)


final_prompt = base.final_prompt
