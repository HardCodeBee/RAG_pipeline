"""Only delete the actual ordinary Original question header and value."""

ARMS = ("O0", "O1")
ACTUAL = "\n\nActual task:\n"


def prompt(item, arm):
    original=item["control_prompt"]
    prefix,actual=original.split(ACTUAL,1)
    block="Original question:\n"+item["original_question"]+"\n\n"
    assert actual.startswith(block)
    if arm=="O0":return original
    if arm!="O1":raise ValueError(arm)
    # Preserve all preceding instructions and demonstrations verbatim, including
    # the instruction that the now-absent Original question supplies scope.
    return prefix+ACTUAL+actual[len(block):]


def removed_block(item):
    return "Original question:\n"+item["original_question"]+"\n\n"
