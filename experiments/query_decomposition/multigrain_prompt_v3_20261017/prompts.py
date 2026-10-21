"""Second round: preserve source names while keeping the v2 rewriter frozen."""

from experiments.query_decomposition.multigrain_prompt_v2_20261017.prompts import rewrite, helper, REWRITE_INSTRUCTIONS, REWRITE_EXAMPLES
from experiments.query_decomposition.multigrain_prompt_v2_20261017.prompts import FINAL_INSTRUCTIONS as BASE_INSTRUCTIONS

FINAL_INSTRUCTIONS = BASE_INSTRUCTIONS + (
    "For a person, place or organization, use the full identifying name stated in the matching passage. "
    "When an auxiliary answer uses a shorter name for the same entity, preserve the fuller source name, including compound surnames and disambiguating suffixes.\n"
)
FINAL_READOUT = """Final response requirements:
Return only a concise answer to the Original question. Preserve full identifying names, necessary qualifiers and units; do not shorten a source name to match an auxiliary answer.
For a supported yes/no answer, return exactly Yes or No.
Do not list intermediate answers, restate the question, or add explanation or citations.
Final short answer:"""


def final(question, context, examples):
    text = "\n\n".join(
        f"Example {i + 1}\nQuestion:\n{row['question']}\n\nContext:\n{row['context']}\n\nFinal short answer: {row['answer']}"
        for i, row in enumerate(examples)
    )
    return FINAL_INSTRUCTIONS + "\nWorked examples:\n\n" + text + "\n\nActual task:\n\nOriginal question:\n" + question + "\n\nContext:\n" + context + "\n\n" + FINAL_READOUT
