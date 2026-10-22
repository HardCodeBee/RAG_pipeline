"""Goal variants and a common neutral final presentation; no model access."""
from copy import deepcopy

from src.p02_multigrain_timing_cachefix_v1.core import final_prompts
from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages

G1 = """Using the previous question-answer pairs and retrieved passages, write one
short factual summary addressing the Original question. Aim for 80-120 words.
Preserve the entities, relations, requested attributes, conditions and units of
the key facts. Include decisive facts newly obtained from the passages.
Previous answers are candidate information, not verified evidence. If passage
evidence corrects a previous answer about the same entity and scope, state the
correction. Do not turn an incomplete list into a complete set, or missing
information into evidence of absence. State any necessary information that
remains unavailable. Give the supported candidate answer if established.
Report facts and the candidate answer directly; do not describe a step-by-step
derivation, comparison procedure or chain of reasoning. Return one paragraph.
"""

G2 = """Using the previous question-answer pairs and retrieved passages, write one
short synthesis paragraph addressing the Original question. Aim for 80-120 words.
State the key facts needed to answer it, preserving their entities, relations,
requested attributes, conditions and units. Include decisive facts newly
obtained from the passages. Previous answers are candidate information, not
verified evidence. If passage evidence corrects a previous answer about the
same entity and scope, state the correction.
Explain only the essential relation or operation connecting the stated facts
to the answer. Make the comparison direction, relationship composition or set
operation explicit when needed. Ensure that the conclusion follows from the
facts actually stated. Do not treat incomplete lists as complete, or missing
information as evidence of absence. If a necessary fact remains unavailable,
state what is established and what remains unresolved. End with the supported
candidate answer, or explain why it cannot yet be determined. Return one paragraph.
"""

ARMS = ("G0", "G1", "G2")


def qa_block(nodes):
    return "\n\n".join(
        f"#{n['step_id']} Question: {n['request']}\nAnswer: {n['answer']}"
        for n in nodes
    ) or "No previous question-answer pairs are available."


def goal_prompt(record, arm, rewrite_off=False):
    if arm not in ARMS:
        raise ValueError(arm)
    # G0 keeps the exact historical prompt, changing its output cap only.
    if arm == "G0":
        prompt = record["historical_goal_prompt"]
        if rewrite_off:
            original = passages(record["goal_reader_packet"])
            assert original and prompt.count(original) == 1
            prompt = prompt.replace(original, passages(record["goal_original_packet"]), 1)
        return prompt
    packet = record["goal_original_packet"] if rewrite_off else record["goal_reader_packet"]
    body = ("\nOriginal question:\n" + record["question"]
            + "\n\nPrevious question-answer pairs:\n" + qa_block(record["ordinary_nodes"])
            + "\n\nRetrieved passages:\n" + passages(packet)
            + "\n\nResponse:\n")
    instructions = {"G1": G1, "G2": G2}[arm]
    return instructions + body


def final_prompt(record, goal_text, config, examples, *, include_goal=True):
    sections = ["Auxiliary question-answer pairs", "Intermediate candidate results",
                qa_block(record["ordinary_nodes"])]
    if include_goal:
        if not goal_text.strip():
            raise ValueError("Empty candidate goal information")
        sections += ["Candidate information for the Original question", goal_text]
    block = "\n\n".join(sections)
    context = block + "\n\n" + passages(record["final_root_packet"])
    return final_prompts.final(record["question"], context, examples, config["readout"]), context


def clone_without_rewrite(record):
    value = deepcopy(record)
    value["goal_reader_packet"] = deepcopy(record["goal_original_packet"])
    return value
