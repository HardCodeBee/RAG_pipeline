"""The reviewed R15 block, applied without replacing the actual current task."""
from experiments.query_decomposition.p02_actual_child_binding_probe_v1_20261022.templates import HEADER, CAUTION, block

ARMS = ("B0", "B1", "B2")


def prompt(item, arm):
    original = item["control_prompt"]
    if arm == "B0":
        return original
    if arm not in ("B1", "B2"):
        raise ValueError(arm)
    prefix, actual = original.split("\n\nActual task:\n", 1)
    marker = f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\nRetrieved passages:'
    assert actual.count(marker) == 1
    candidate = item["current_question"] if arm == "B1" else item["frozen_rewritten_query"]
    replacement = marker.replace("\n\nRetrieved passages:", "\n\n" + block(candidate) + "\n\nRetrieved passages:")
    return prefix + "\n\nActual task:\n" + actual.replace(marker, replacement, 1)
