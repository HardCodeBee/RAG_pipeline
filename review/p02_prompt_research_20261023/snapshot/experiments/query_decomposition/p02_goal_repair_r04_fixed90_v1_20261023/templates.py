"""Minimal goal-contract repair; no benchmark entities or answer keys in rules."""
OLD_CONTRACT = ('Return one concise factual sentence preserving the entity, relation and necessary qualifiers. '
    'If a needed fact is missing, state what cannot be determined; lack of evidence does not mean no, none, an empty set or zero. '
    'Do not add reasoning or citations.')
BRIEF_CONTRACT = ('Return one short paragraph, normally two or three sentences. State the relevant facts and the essential '
    'connection needed for this question, then end with a concise factual sentence giving the answer, preserving the entity, '
    'relation and necessary qualifiers. A direct fact needs no additional chain. If a needed fact is missing, state what '
    'cannot be determined; lack of evidence does not mean no, none, an empty set or zero. Do not add citations.')
TARGET = ('Identify the requested relation before combining answers. A description used to identify an entity does not '
    'automatically restrict every fact asked about that entity. Follow the relation chain to the object whose attribute '
    'is requested, and keep conditions that apply to that attribute. Use the question and passages to resolve ordinary '
    'wording; do not reject an otherwise supported relation solely because a broad category word is imprecise. Do not '
    'add a present-day requirement, a missing generation, or an unstated condition. A correction requires a concrete '
    'conflicting fact about the same entity and relation, not an imagined constraint. An identifying clue that is not '
    'repeated in the passages is not itself a contradiction; an actual conflicting clue still matters.')


def goal_prompt(record, arm):
    text = record['historical_goal_prompt']
    assert text.count(OLD_CONTRACT) == 1
    if arm == 'C':
        return text
    assert arm in ('S', 'T')
    text = text.replace(OLD_CONTRACT, BRIEF_CONTRACT, 1)
    if arm == 'T':
        text = text.replace('\n\nOriginal question:\n', '\n\n' + TARGET + '\n\nOriginal question:\n', 1)
    return text
