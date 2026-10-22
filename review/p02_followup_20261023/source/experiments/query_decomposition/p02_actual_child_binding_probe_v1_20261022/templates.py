"""Keep actual current subquestion authoritative; add one candidate block."""

ARMS = ("R0", "R1", "R2")
HEADER = "Candidate interpretation (not a replacement for the Current subquestion):\n"
CAUTION = ("Treat this interpretation as a candidate only. Resolve factual or role conflicts using the "
           "Original question, the Current subquestion and retrieved passages about the matching entities.")


def block(candidate):
    return HEADER + candidate + "\n" + CAUTION


def prompt(item, arm):
    original = item["control_prompt"]
    if arm == "R0":
        return original
    if arm not in ("R1", "R2"):
        raise ValueError(arm)
    prefix, actual = original.split("\n\nActual task:\n", 1)
    marker = f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\nRetrieved passages:'
    assert actual.count(marker) == 1
    candidate = item["current_question"] if arm == "R1" else item["frozen_rewritten_query"]
    replacement = marker.replace("\n\nRetrieved passages:", "\n\n" + block(candidate) + "\n\nRetrieved passages:")
    return prefix + "\n\nActual task:\n" + actual.replace(marker, replacement, 1)
