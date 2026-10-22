"""Single literal insertion; preserve all historical inputs and demonstrations."""
SENTENCE = "Before answering, verify which entity and relation the Current subquestion asks about; preserve the relation's direction, requested answer type and explicit conditions, and correct a premise when the passages contradict it rather than restating that premise as fact."
DELIMITER = "\n\nWorked examples:"
ARMS = ("I0", "I1")


def prompt(original, arm):
    assert original.count(DELIMITER) == 1
    assert SENTENCE not in original
    if arm == "I0":
        return original
    if arm != "I1":
        raise ValueError(arm)
    prefix, rest = original.split(DELIMITER)
    return prefix + "\n" + SENTENCE + DELIMITER + rest
