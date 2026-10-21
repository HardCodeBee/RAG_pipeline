"""Optional partial binding supplements, never replaces, original-query retrieval."""

from experiments.query_decomposition.multigrain_prompt_v2_20261017.prompts import REWRITE_EXAMPLES
from experiments.query_decomposition.multigrain_prompt_v2_20261017 import prompts as previous

INSTRUCTIONS = """Write one retrieval rewrite of the Current question using the Previous question-answer pairs. The original Current question is always searched separately.
Use only identities or factual relations explicitly supplied by earlier answers. Preserve the requested relation and conditions for every entity you include. Do not answer the Current question, invent missing identities, or paste a full answer sentence into an entity slot.
If an earlier answer supplies only part of an entity set, the additional rewrite may target that known part. It is a partial retrieval path, not a claim that this is the entire set. Do not include an unknown member as an entity. When all relevant entities are known, retain them rather than arbitrarily choosing one.
If no relevant binding is available, return the Current question unchanged. Return only one complete question, without explanation.
"""

PARTIAL_EXAMPLE = """Example 4 (partial known set; original question is also retrieved)
Previous question-answer pairs:
Question: Which plant genera have red flowers?
Answer: Clianthus has red flowers; other genera cannot be determined from these passages.
Current question:
How many species do the genera with red flowers contain?
Rewritten question: How many species does Clianthus contain?"""


def rewrite(question, history):
    prior = '\n\n'.join('Question: '+r['request']+'\nAnswer: '+r['answer'] for r in history)
    return INSTRUCTIONS+'\nWorked examples:\n\n'+REWRITE_EXAMPLES+'\n\n'+PARTIAL_EXAMPLE+'\n\nActual task:\n\nPrevious question-answer pairs:\n'+prior+'\n\nCurrent question:\n'+question+'\n\nRewritten question:'


SUPPORT_SENTENCE = "Auxiliary answers may supply facts from other subquestion retrievals. Absence from the passages shown here is not a contradiction; facts about a related entity or a different event do not conflict."


def final(question, context, examples):
    text = previous.final(question, context, examples)
    return text.replace(previous.FINAL_INSTRUCTIONS, previous.FINAL_INSTRUCTIONS + SUPPORT_SENTENCE + '\n', 1)
