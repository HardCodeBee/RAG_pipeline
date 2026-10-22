"""Native child/goal and short-answer formatter; reader is the only factor."""
from p02_multigrain_timing_cachefix_v1.core import child_prompts
from p02_multigrain_timing_cachefix_v1.core.pipeline import passages
from .selection import ARMS,packet

ACTUAL="\n\nActual task:\n"
READER="\n\nRetrieved passages:\n"


def reader(item,arm):
    return packet(item["original_retrieval"],item["rewritten_retrieval"],arm)


def suffix(item):
    return "\n\n"+(child_prompts.GOAL_RULE+"\n\n"if item["role"]=="goal"else"")+"Answer:"


def prompt(item,arm):
    if arm not in ARMS:raise ValueError(arm)
    original=item["control_prompt"]
    # Native goal has no worked examples/Actual task heading. For ordinary,
    # rsplit selects the actual reader after all demonstration readers.
    if item["role"]=="ordinary":assert original.count(ACTUAL)==1
    else:assert ACTUAL not in original and original.count(READER)==1
    prefix,old_reader=original.rsplit(READER,1)
    assert old_reader==passages(item["reader_packet"])+suffix(item)
    return prefix+READER+passages(reader(item,arm))+suffix(item)


def needs_formatter(answer):return len(answer.strip().split())<4


def formatter_prompt(item,arm,answer):
    return (
        "Express the candidate answer below as a concise complete factual sentence that names its subject and relation. "
        "Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. "
        "If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n"
        +"Question:\n"+item["current_question"]+"\n\nCandidate answer:\n"+answer.strip()
        +"\n\nContext:\n"+passages(reader(item,arm)))
