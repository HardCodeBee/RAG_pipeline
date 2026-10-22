"""Focused prompts for rewriting and final reading; no answer verification in rewriting."""

REWRITE_INSTRUCTIONS = """Rewrite the Current question for retrieval using the Previous question-answer pairs.
Replace a descriptive reference only with the matching entity explicitly supplied by an earlier answer. Keep the fact requested by the Current question, its relation, conditions and comparison direction unchanged. Use the meaning of the complete earlier sentence; do not paste that sentence into a noun slot.
Do not answer the Current question or introduce a factual identity that the previous answers have not supplied. If no relevant replacement is available, return the Current question unchanged.
Return only one complete question, without explanation.
"""

REWRITE_EXAMPLES = """Example 1
Previous question-answer pairs:
Question: Which river does Pack Creek flow into?
Answer: Pack Creek flows into the Colorado River.
Current question:
What is the length of the river into which Pack Creek flows?
Rewritten question: What is the length of the Colorado River?

Example 2
Previous question-answer pairs:
Question: Who created the Mona Lisa?
Answer: Leonardo da Vinci created the Mona Lisa.
Current question:
Where did the creator of the Mona Lisa die?
Rewritten question: Where did Leonardo da Vinci die?

Example 3
Previous question-answer pairs:
Question: How many species does Clianthus contain?
Answer: Clianthus contains two species.
Current question:
How many species does Callicoma contain?
Rewritten question: How many species does Callicoma contain?"""

FINAL_INSTRUCTIONS = """Answer the Original question using the retrieved passages and auxiliary question-answer pairs.
Use relevant auxiliary answers as intermediate information. When a retrieved passage gives a conflicting fact about the same entity, relation and conditions, use the passage. Do not transfer a fact about another entity or a different comparison to the Original question.
Complete the relation, comparison or combination requested by the Original question. Missing information does not establish a negative fact, zero or an empty set. If the available information does not support the final answer, return Cannot be determined.
"""


def rewrite(question, history):
    prior = "\n\n".join(
        f"Question: {row['request']}\nAnswer: {row['answer']}" for row in history
    )
    return (
        REWRITE_INSTRUCTIONS
        + "\nWorked examples:\n\n"
        + REWRITE_EXAMPLES
        + "\n\nActual task:\n\nPrevious question-answer pairs:\n"
        + prior
        + "\n\nCurrent question:\n"
        + question
        + "\n\nRewritten question:"
    )
