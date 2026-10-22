"""Four reviewed changes only; preserve frozen goal and all other prompt text."""
from copy import deepcopy
import json

from p02_multigrain_timing_cachefix_v1.core import ROOT
from p02_multigrain_timing_cachefix_v1.core import child_prompts as frozen

GOAL_RULE = frozen.GOAL_RULE
RESOLVE = "Resolve each #N from its numbered prior question and answer."
DESCRIPTIVE = (
    "Use relevant prior answers to interpret descriptive references while preserving "
    "the current subquestion's relation, conditions and scope."
)
INSTRUCTIONS = frozen.CHILD_INSTRUCTIONS.replace(RESOLVE, DESCRIPTIVE, 1).replace(
    "the same entity and scope.", "the same entity, relation and scope.", 1
)
ORIGINAL_EXAMPLES = json.loads((ROOT / "prompts/child_examples.json").read_text(encoding="utf8"))
EXAMPLES = deepcopy(ORIGINAL_EXAMPLES)
EDITS = {
    0: {
        "original_question": "Is the river into which Pack Creek flows after passing through Spanish Valley longer than 1000 mi?",
        "request": "What is the length of the river into which Pack Creek flows after passing through Spanish Valley?",
    },
    2: {
        "original_question": "Does the genus containing Callicoma serratifolia contain any other species?",
        "request": "How many species does the genus containing Callicoma serratifolia contain other than Callicoma serratifolia?",
    },
}
for index, fields in EDITS.items():
    for field, value in fields.items():
        previous = EXAMPLES[index][field]
        assert EXAMPLES[index]["text"].count(previous) == 1
        EXAMPLES[index]["text"] = EXAMPLES[index]["text"].replace(previous, value, 1)
        EXAMPLES[index][field] = value


def child(question, sid, request, dependencies, context, examples, *, role="ordinary"):
    rendered = frozen.child(question, sid, request, dependencies, context,
                            EXAMPLES if role == "ordinary" else examples, role=role)
    if role == "ordinary":
        assert rendered.startswith(frozen.CHILD_INSTRUCTIONS + "\n\n")
        rendered = INSTRUCTIONS + rendered[len(frozen.CHILD_INSTRUCTIONS):]
    return rendered


def template():
    prior = [{"step_id": 1, "request": "{prior_question}", "answer": "{prior_answer}"}]
    text = child("{original_question}", 987654, "{current_request}", prior, "{context}", [])
    return text.replace("#1 Question: {prior_question}\nAnswer: {prior_answer}",
                        "{prior_question_answer_pairs}").replace(
                            "Current subquestion #987654:", "Current subquestion #{current_step_id}:")
