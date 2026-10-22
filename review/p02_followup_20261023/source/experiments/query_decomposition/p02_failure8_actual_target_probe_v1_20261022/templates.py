"""Change only the actual ordinary-QA current-question value, never a fewshot."""

ARMS = ("C0", "C1")


def prompt(item, arm):
    original = item["control_prompt"]
    if arm == "C0":
        return original
    if arm != "C1":
        raise ValueError(arm)
    prefix, actual = original.split("\n\nActual task:\n", 1)
    marker = f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\nRetrieved passages:'
    assert actual.count(marker) == 1
    replacement = f'Current subquestion #{item["step_id"]}:\n{item["frozen_rewritten_query"]}\n\nRetrieved passages:'
    return prefix + "\n\nActual task:\n" + actual.replace(marker, replacement, 1)


def evaluation_target(item):
    """The unchanged original raw current remains the target of every label."""
    return item["current_question"]


def model_current(item, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return item["current_question"] if arm == "C0" else item["frozen_rewritten_query"]
