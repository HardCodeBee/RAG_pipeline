"""Fact-first child responses and an exact, non-propagating no-answer contract."""
import re

from .prepare import HERE, load


NO_ANSWER = 'The retrieved material does not answer this subquestion.'
INSTRUCTION = f"""Answer only the Current question using the retrieved passages and relevant facts from prior question-answer pairs. The Original question supplies scope.
Return one or two concise factual sentences naming the subject, relation and necessary qualifiers. State supported facts first. If only part of the requested information is available, retain it and identify only the missing part; do not describe the whole question as unanswerable.
Retain a supported intermediate identity needed to resolve the Current question even when its requested attribute is missing. An unknown part of a prior answer does not invalidate facts explicitly stated in that answer.
Keep the direction and meaning of each relation. Do not transfer an attribute between entities or add facts not supplied by the input. Use direct passage evidence when it contradicts a prior answer about the same entity and relation.
If neither the requested answer nor a needed intermediate identity can be established, output exactly: {NO_ANSWER}
Missing evidence does not establish no, zero or an empty set. A supported negative or zero is a valid factual answer. Do not add reasoning, citations or status labels.
"""

# Only WHOLE-response abstentions are removed. A none-marker is not reliable
# evidence of an empty factual answer; zero, No and None are not abstention keys.
LEGACY_ABSTENTIONS = frozenset({
    "i don't know", 'i do not know', 'unknown', 'not known',
    'cannot be determined', 'not enough information', 'insufficient information',
    'the answer cannot be determined from the provided passages',
    'the answer cannot be determined from the retrieved passages',
    'the answer cannot be determined from the provided context',
    'the provided passages do not contain the answer',
    '现有检索材料无法回答该子问题', '不知道', '我不知道', '无法确定', '信息不足',
})


def is_no_answer(answer):
    body = re.sub(r'^\s*\[(?:complete|partial|none)\]\s*', '', str(answer), flags=re.I).strip()
    normalized = body.casefold().replace('’', "'").rstrip('.。!?！？').strip()
    return normalized == NO_ANSWER.casefold().rstrip('.') or normalized in LEGACY_ABSTENTIONS


def usable_history(history):
    """Preserve complete records and partial known facts; remove only explicit abstentions."""
    return [dict(record) for record in history if not is_no_answer(record['answer'])]


def rewrite_inputs(question, history):
    """Program action before calling the existing rewriter; no model-side verification."""
    retained = usable_history(history)
    return {'original_query': question, 'history': retained,
            'invoke_rewriter': bool(retained)}


def rewrite_or_original(question, history, rewrite_call):
    """No rewrite API dispatch when all antecedents abstained; original search survives."""
    plan = rewrite_inputs(question, history)
    if not plan['invoke_rewriter']:
        return question
    return rewrite_call(question, plan['history'])


def auxiliary_pairs(records):
    return usable_history(records)


def parse_response(text):
    return {'answer_text': text.strip(), 'is_no_answer': is_no_answer(text)}


def child(original, question, history, context, examples):
    def prior(records):
        return '\n\n'.join('Question: ' + r['request'] + '\nAnswer: ' + r['answer']
                           for r in usable_history(records)) or 'No prior answers.'
    shots = load(HERE / 'c08_examples.frozen.json')
    demonstrations = '\n\n'.join(
        f"Example {i+1}\nCurrent question: {r['request']}\nPrevious question-answer pairs:\n{prior(r['dependencies'])}\nRetrieved passages:\n{r['context']}\nAnswer: {r['answer']}"
        for i, r in enumerate(shots))
    return (INSTRUCTION + '\nWorked examples:\n' + demonstrations +
            '\n\nActual task:\nOriginal question:\n' + original +
            '\n\nPrevious question-answer pairs:\n' + prior(history) +
            '\n\nCurrent question:\n' + question +
            '\n\nRetrieved passages:\n' + context + '\n\nAnswer:')
