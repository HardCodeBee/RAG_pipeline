"""Isolated goal-only candidates; importing this module performs no inference."""
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import templates as old
from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages

INSTRUCTIONS = """Answer the Original question by combining the Previous question-answer
pairs with the Retrieved passages.

Write a concise, evidence-based synthesis showing the essential steps
that connect the available information to the requested answer.

Rules:
1. Keep the Original question as the target. Preserve its requested
   relation, attribute, comparison direction, time and other conditions.
   Do not add a condition that the question does not require.

2. Read each previous answer together with its question. Preserve the
   full entity-relation-entity or entity-attribute-value statement.
   When using one result in a later step, name the entity being passed
   forward and the relation or attribute being followed. Treat # labels
   as identifiers of previous QA pairs, not as factual evidence themselves.

3. Use previous answers as intermediate claims. A fact need not appear
   again in the current passages to remain usable. Correct a previous
   claim when the passages directly contradict it about the same entity,
   relation or attribute, and scope. State the relevant correction briefly.
   Do not merge people or objects merely because their names are similar.

4. Add decisive facts from the passages even when previous answers omitted
   them or could not answer. Use any supported path that answers the
   Original question; not every previous subquestion must be solved.

5. Make the necessary operation explicit: follow each relation separately,
   compare the requested attribute using compatible units, or apply the
   required set operation. Do not skip a relation or substitute a nearby
   attribute. Do not assume a partial list is exhaustive.

6. Report an information gap only when a fact necessary for answering the
   Original question remains unavailable. Missing evidence is not evidence
   that something is false.

Output one short paragraph containing the decisive facts, their essential
combination, and the candidate answer or precise unresolved gap. Refer
briefly to the relevant QA or passage identifiers. End with
"Candidate answer: ...", or "Unresolved: ...".
Avoid unrelated background and repeated facts.
"""

EXAMPLES = """The following examples use fictional facts. Learn their procedure and
format; never use their facts to answer the actual question.

Example 1

Original question:
In which country was the paternal grandfather of Elias Rowan Jr. born?

Previous question-answer pairs:
#1 Question: Who is Elias Rowan Jr.'s father?
Answer: Elias Rowan Jr.'s father is Elias Rowan Sr.
#2 Question: Who is the father of Elias Rowan Jr.'s father?
Answer: Elias Rowan Sr.'s father is Martin Rowan.
#3 Question: Where was Martin Rowan born?
Answer: Martin Rowan was born in Alderport.

Retrieved passages:
[Chunk 1] Elias Rowan Jr. was the son of Elias Rowan Sr.
[Chunk 2] Alderport is a town in Norland.

Response:
Elias Rowan Jr.'s father is Elias Rowan Sr. (#1, Chunk 1), and the father's
father is Martin Rowan (#2). Therefore Martin, rather than Elias Sr., is
the paternal grandfather. Martin was born in Alderport (#3), which is in
Norland (Chunk 2). Following these relations gives Elias Jr. → father
Elias Sr. → paternal grandfather Martin → birthplace Alderport → country
Norland. Chunk 1 confirms the first link; its silence about Martin does
not contradict #2. Candidate answer: Norland.

Example 2

Original question:
What award was the lead actor in Harbor Lights nominated for?

Previous question-answer pairs:
#1 Question: Who was the lead actor in Harbor Lights?
Answer: Mara Ellis was the lead actor in Harbor Lights.
#2 Question: What award nomination did Mara Ellis receive?
Answer: The available material does not identify a nomination.

Retrieved passages:
[Chunk 1] Harbor Lights starred Mara Ellis.
[Chunk 2] Mara Ellis received a Silver Screen Award nomination for Best
Supporting Actress for her performance in Winter Road.

Response:
The lead actor identified by Harbor Lights is Mara Ellis (#1, Chunk 1).
Chunk 2 supplies the nomination that #2 could not establish: a Silver
Screen Award nomination for Best Supporting Actress for Winter Road.
The question uses Harbor Lights to identify the person; it does not
require the nomination to be for Harbor Lights. Connecting the identified
actor to her nomination therefore answers the original question despite
the different nominated film. Candidate answer: Silver Screen Award
for Best Supporting Actress.

Example 3

Original question:
Which has more seats, Northfield Arena or Southmere Arena?

Previous question-answer pairs:
#1 Question: How many seats does Northfield Arena have?
Answer: Northfield Arena has 12,000 seats.
#2 Question: How many seats does Southmere Arena have?
Answer: Southmere Arena has 9,500 seats.

Retrieved passages:
[Chunk 1] Northfield Arena accommodates 12,000 spectators: 8,000 seated
and 4,000 standing.
[Chunk 2] Southmere Arena has 9,500 seats.

Response:
The requested attribute is seats, rather than total spectator capacity.
Chunk 1 corrects #1: Northfield has 8,000 seats; its 12,000 figure includes
standing spectators. Southmere has 9,500 seats (#2, Chunk 2). Comparing
the same attribute gives 9,500 > 8,000, so Southmere has 1,500 more seats.
Northfield's larger total capacity does not reverse this comparison,
because standing places are outside the attribute requested by the
original question. Candidate answer: Southmere Arena.

Example 4

Original question:
Did both Ada Venn and Beno Rell receive the Helios Medal?

Previous question-answer pairs:
#1 Question: Did Ada Venn receive the Helios Medal?
Answer: Ada Venn received the Helios Medal.
#2 Question: What honors did Beno Rell receive?
Answer: Beno Rell received the Lumen Prize.

Retrieved passages:
[Chunk 1] Ada Venn received the Helios Medal.
[Chunk 2] Selected honors of Beno Rell include the Lumen Prize.

Response:
Ada Venn's Helios Medal is established by #1 and Chunk 1. Beno Rell's
Lumen Prize is established by #2 and Chunk 2, but neither establishes
whether he received the Helios Medal. Because Chunk 2 lists selected
honors, omission of the Helios Medal cannot establish that he did not
receive it. Answering whether both received the medal requires resolving
Beno's status as well as Ada's. Unresolved: whether Beno Rell received
the Helios Medal.
"""

ARMS = ("G2", "I", "F")


def goal_prompt(record, arm):
    if arm == "G2":
        return old.goal_prompt(record, "G2")
    if arm not in ("I", "F"):
        raise ValueError(arm)
    prefix = INSTRUCTIONS + ("\n" + EXAMPLES if arm == "F" else "")
    return (prefix + "\nNow answer the actual question.\n\nOriginal question:\n"
            + record["question"] + "\n\nPrevious question-answer pairs:\n"
            + old.qa_block(record["ordinary_nodes"]) + "\n\nRetrieved passages:\n"
            + passages(record["goal_reader_packet"]) + "\n\nResponse:\n")
