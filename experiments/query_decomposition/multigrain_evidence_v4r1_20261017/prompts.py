"""Retain partial factual information; do not turn missing information into zero."""

INSTRUCTION = """Answer only the Current question, using the retrieved passages and relevant prior question-answer pairs. The Original question supplies scope.
Return [complete], [partial], or [none], followed by one or two concise complete factual sentences naming the subject, relation and necessary qualifiers.
[complete]: the requested answer is supported. [partial]: some requested facts are supported, but others are missing. Preserve the supported facts and identify only the missing part. [none]: no requested fact can be established; state the specific unresolved fact.
Do not discard a known parent, place, date, or member merely because the entire requested set is unavailable. Do not invent identities or transfer a property to another entity. Use direct passage evidence when it conflicts with a prior answer about the same entity and relation.
Missing evidence does not establish no, zero or an empty set. A supported negative or zero is a valid complete answer. Do not add reasoning or citations.
"""


def parse_answer(text):
    for state in ("complete", "partial", "none"):
        prefix = "[" + state + "]"
        if text.lower().startswith(prefix) and text[len(prefix):].strip():
            return state, text[len(prefix):].strip()
    return "unparsed", text.strip()


def child(original, question, history, context, examples):
    def prior(records):
        return "\n\n".join("Question: " + r["request"] + "\nAnswer: " + r["answer"] for r in records) or "No prior answers."
    shots = "\n\n".join(
        f"Example {i+1}\nCurrent question: {r['request']}\nPrevious question-answer pairs:\n{prior(r['dependencies'])}\nRetrieved passages:\n{r['context']}\nAnswer: {r['answer']}"
        for i, r in enumerate(examples)
    )
    return (INSTRUCTION + "\nWorked examples:\n" + shots + "\n\nActual task:\nOriginal question:\n" + original
        + "\n\nPrevious question-answer pairs:\n" + prior(history) + "\n\nCurrent question:\n" + question
        + "\n\nRetrieved passages:\n" + context + "\n\nAnswer:")
