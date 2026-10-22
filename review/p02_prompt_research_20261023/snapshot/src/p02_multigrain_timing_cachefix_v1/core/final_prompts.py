"""Second round: preserve source names while keeping the v2 rewriter frozen."""

from .rewrite_prompts import FINAL_INSTRUCTIONS as BASE_INSTRUCTIONS

FINAL_INSTRUCTIONS = BASE_INSTRUCTIONS + (
    "For a person, place or organization, use the full identifying name stated in the matching passage. "
    "When an auxiliary answer uses a shorter name for the same entity, preserve the fuller source name, including compound surnames and disambiguating suffixes.\n"
)


def final(question, context, examples, readout):
    text = "\n\n".join(
        f"Example {i + 1}\nQuestion:\n{row['question']}\n\nContext:\n{row['context']}\n\nFinal short answer: {row['answer']}"
        for i, row in enumerate(examples)
    )
    return (
        FINAL_INSTRUCTIONS
        + "\nWorked examples:\n\n"
        + text
        + "\n\nActual task:\n\nOriginal question:\n"
        + question
        + "\n\nContext:\n"
        + context
        + "\n\n"
        + readout
    )
