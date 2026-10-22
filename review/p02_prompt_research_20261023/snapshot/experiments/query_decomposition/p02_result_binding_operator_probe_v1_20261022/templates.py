"""Frozen matched examples. These functions have no model or reference access."""
from experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run import variant_prompt
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import templates as base
from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages

ORDINARY_ARMS = ("D", "B")
GOAL_ARMS = ("G0", "GF", "GO")
NOTES = (
    ("Answer: Pack Creek flows into the Colorado River.",
     "Example interpretation: The prior question asks for the receiving river. "
     "Its returned value is Colorado River; Pack Creek is the starting entity, "
     "not that returned river. The current description denotes this returned "
     "value, so its length must be read for Colorado River."),
    ("Answer: Callicoma contains Callicoma serratifolia.",
     "Example interpretation: The prior question asks for a genus. Its returned "
     "value is Callicoma; Callicoma serratifolia is a member, not the returned "
     "genus. Use Callicoma as the subject of the current count and exclude the "
     "named member Callicoma serratifolia."),
)

def ordinary_prompt(record, arm):
    prompt = variant_prompt(record["control_prompt"], "E2")
    if arm == "D":
        return prompt
    if arm != "B":
        raise ValueError(arm)
    prefix, actual = prompt.split("\n\nActual task:\n", 1)
    for anchor, note in NOTES:
        assert prefix.count(anchor) == 1
        prefix = prefix.replace(anchor, anchor + "\n\n" + note, 1)
    return prefix + "\n\nActual task:\n" + actual

INSTRUCTION = """Answer the Original question by combining the previous question-answer pairs
and retrieved passages. Write one concise paragraph, normally two or three
sentences: decisive facts, the essential operation connecting them, and the
supported candidate answer. Omit background that does not help answer it.

Read each prior answer together with its question: identify the requested
result and use it in the correct entity/relation/attribute slot. Preserve the
Original question's scope; do not add a condition it does not ask for. Previous
answers may supply facts absent from the current passages. Correct a prior
claim when passages directly contradict it about the same entity and scope;
absence of a repeated fact is not itself a contradiction.

Use the operation actually required: follow directed relations, compare the
requested quantities in the right direction, or combine sets or conditions.
Map the result back to the kind of answer requested. The candidate conclusion
must follow from the facts you state. Do not equate missing information with
false, zero or a complete list. If available facts or bounds already decide
the answer, use them; otherwise name only the unresolved fact needed to decide
it. Do not require every planned subanswer when another supported route suffices.
Return only the paragraph, without headings, citations or a numbered analysis.
"""

# Fictional demonstrations: questions, supplied facts and conclusions are
# identical between GF and GO. GO adds only the marked operation sentence.
DEMOS = (
    dict(kind="directed_relation", question="Who is Mira's paternal grandfather?",
         history="Q1: Who is Mira's father? A1: Mira's father is Leto.",
         passages="Leto is a son of Daren. Daren's father is Osric.",
         facts="Mira's father is Leto, and Leto's father is Daren.",
         operation="Following father(Mira) = Leto and father(Leto) = Daren gives the father of Mira's father.",
         conclusion="Mira's paternal grandfather is Daren."),
    dict(kind="comparison_and_return_mapping", question="Which film's director is younger, Silver Orchard or North Ferry?",
         history="Q1: Who directed Silver Orchard? A1: Elia Voss directed Silver Orchard.\nQ2: Who directed North Ferry? A2: Bram Keir directed North Ferry.",
         passages="Elia Voss was born in 1978. Bram Keir was born in 1962.",
         facts="Silver Orchard's director Elia Voss was born in 1978, and North Ferry's director Bram Keir was born in 1962.",
         operation="The later birth year, 1978 > 1962, makes Elia younger; the question asks for her film.",
         conclusion="The answer is Silver Orchard."),
    dict(kind="set_intersection", question="Do Solen and Tavia share a citizenship?",
         history="Q1: What citizenships does Solen have? A1: Solen has Albian and Boran citizenship.",
         passages="Tavia has Boran citizenship.",
         facts="Solen has Albian and Boran citizenship, and Tavia has Boran citizenship.",
         operation="Their citizenship sets intersect at Boran; sharing one citizenship does not require identical sets.",
         conclusion="Yes, they share Boran citizenship."),
    dict(kind="attribute_and_correction", question="How many seated spectators can the stadium in Orin's hometown hold?",
         history="Q1: Where was Orin born? A1: Orin was born in Neral.\nQ2: What is the seated capacity of Neral Stadium? A2: Its seated capacity is 9000.",
         passages="Neral Stadium accommodates 9000 spectators in total, including 6200 seats and 2800 standing places.",
         facts="Orin's hometown is Neral; its stadium has 6200 seats, although total capacity is 9000, correcting the prior seated-capacity answer.",
         operation="The requested attribute is seated capacity, so select seats rather than adding standing places or returning total capacity.",
         conclusion="The answer is 6200 seated spectators."),
    dict(kind="decisive_partial_information", question="Are both Mavren and Ostel native to Iona?",
         history="Q1: Is Mavren native to Iona? A1: Mavren is not native to Iona.\nQ2: Is Ostel native to Iona? A2: Ostel's native range cannot be determined from these passages.",
         passages="Mavren is native only to the continent of Kel, not Iona. Ostel is a flowering shrub.",
         facts="Mavren is not native to Iona, while Ostel's native range remains unknown.",
         operation="The question requires both conditions to hold; one established false condition makes the conjunction false regardless of the unknown one.",
         conclusion="No, they are not both native to Iona."),
    dict(kind="insufficient_comparison", question="Which group has more species, Vesca or Talune?",
         history="Q1: How many species are in Vesca? A1: Vesca has three species.\nQ2: How many species are in Talune? A2: Talune's total species count cannot be determined.",
         passages="Vesca contains exactly three species. One described Talune species is Talune alba; this account is not a complete list.",
         facts="Vesca has three species; Talune has at least one known species but its total is unavailable.",
         operation="The incomplete list supplies only a lower bound; Talune's unknown total could be below, equal to or above three, so this comparison is unresolved.",
         conclusion="The group with more species cannot yet be determined without Talune's total or a decisive bound."),
)

def demonstration_text(arm):
    if arm not in ("GF", "GO"):
        raise ValueError(arm)
    blocks = []
    for i, d in enumerate(DEMOS, 1):
        answer = d["facts"] + " " + (d["operation"] + " " if arm == "GO" else "") + d["conclusion"]
        blocks.append(f"Fictional example {i}\nOriginal question:\n{d['question']}\n"
                      f"Previous question-answer pairs:\n{d['history']}\n"
                      f"Retrieved passages:\n{d['passages']}\nResponse:\n{answer}")
    return "\n\n".join(blocks)

def goal_prompt(record, arm):
    if arm == "G0":
        return base.goal_prompt(record, "G0")
    return (INSTRUCTION + "\nFictional worked examples:\n" + demonstration_text(arm)
            + "\n\nActual task:\nOriginal question:\n" + record["question"]
            + "\n\nPrevious question-answer pairs:\n" + base.qa_block(record["ordinary_nodes"])
            + "\n\nRetrieved passages:\n" + passages(record["goal_reader_packet"])
            + "\n\nResponse:\n")
