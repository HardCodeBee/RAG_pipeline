"""Question-only entity-proxy spans and an ``entity`` text preview.

This is a deterministic capitalization/quotation heuristic, not named-entity
recognition. Offsets index the original Python string and have exclusive ends.
The preview is not the encoder input: the experiment runner masks intersecting
original WordPiece positions and constructs its matched random control there.
"""

import unicodedata


QUESTION_OPENERS = frozenset({
    'who', 'what', 'which', 'where', 'when', 'how', 'are', 'do', 'does', 'did',
    'is', 'was', 'were', 'can', 'will',
})
# Only complete connector chains between capitalized words are included.
# Coordination (and), relation words (father), and prepositions such as in/at
# are deliberately absent. An uncompleted chain such as "of film" stays intact.
NAME_CONNECTORS = frozenset({
    'of', 'the', 'de', 'del', 'della', 'da', 'di', 'du', 'van', 'von', 'der',
    'den', 'la', 'le', 'el', 'al', 'bin', 'ibn',
})
_INTERNAL_JOINERS = frozenset("-\u2010\u2011'\u2019")
_QUOTE_PAIRS = {'"': '"', '\u201c': '\u201d', "'": "'", '\u2018': '\u2019'}


def _word_character(character):
    return character.isalnum() or unicodedata.category(character).startswith('M')


def _tokens(question):
    """Unicode letter/number tokens, with combining marks and internal joiners."""
    tokens = []
    index = 0
    while index < len(question):
        if not question[index].isalnum():
            index += 1
            continue
        start = index
        index += 1
        while index < len(question):
            if _word_character(question[index]):
                index += 1
            elif (question[index] in _INTERNAL_JOINERS and index + 1 < len(question)
                  and question[index + 1].isalnum()):
                index += 1
            else:
                break
        tokens.append((start, index, question[start:index]))
    return tokens


def _possessive_length(text):
    return 2 if len(text) >= 2 and text[-2] in "'\u2019" and text[-1].lower() == 's' else 0


def _quoted_spans(question, tokens):
    spans = []
    index = 0
    while index < len(question):
        opening = question[index]
        closing = _QUOTE_PAIRS.get(opening)
        if closing is None or (opening in "'\u2018" and index > 0
                               and _word_character(question[index - 1])):
            index += 1
            continue
        stop = index + 1
        while stop < len(question) and question[stop] not in '\r\n':
            internal_apostrophe = (closing in "'\u2019" and stop > 0
                                   and stop + 1 < len(question)
                                   and _word_character(question[stop - 1])
                                   and _word_character(question[stop + 1]))
            if question[stop] == closing and not internal_apostrophe:
                break
            stop += 1
        if stop == len(question) or question[stop] in '\r\n':
            index += 1
            continue
        start, end = index + 1, stop
        while start < end and question[start].isspace():
            start += 1
        while start < end and question[end - 1].isspace():
            end -= 1
        end -= _possessive_length(question[start:end])
        if any(left < end and right > start for left, right, _ in tokens):
            spans.append((start, end, 'quoted'))
        index = stop + 1
    return spans


def _capitalized_spans(question, tokens):
    def capitalized(index):
        start, _, text = tokens[index]
        prefix = question[:start] if index == 0 else question[tokens[index - 1][1]:start]
        sentence_start = index == 0 or any(char in '.?!' for char in prefix)
        if sentence_start and text.casefold() in QUESTION_OPENERS:
            return False
        return text[0].isupper()

    def space_between(left, right):
        return question[tokens[left][1]:tokens[right][0]].isspace()

    spans = []
    index = 0
    while index < len(tokens):
        if not capitalized(index):
            index += 1
            continue
        last = index
        while last + 1 < len(tokens) and not _possessive_length(tokens[last][2]):
            next_index = last + 1
            if not space_between(last, next_index):
                break
            if capitalized(next_index):
                last = next_index
                continue
            cursor = next_index
            while cursor < len(tokens) and tokens[cursor][2] in NAME_CONNECTORS:
                if cursor + 1 >= len(tokens) or not space_between(cursor, cursor + 1):
                    break
                cursor += 1
            if cursor > next_index and cursor < len(tokens) and capitalized(cursor):
                last = cursor
                continue
            break
        start = tokens[index][0]
        end = tokens[last][1] - _possessive_length(tokens[last][2])
        spans.append((start, end, 'capitalized'))
        index = last + 1
    return spans


def relation_view(question):
    """Return original text, proxy spans, preview, and lexical-token counts.

    ``proxy_spans`` contains dictionaries with start/end/text/source. Outer
    quotation marks and terminal ASCII/curly possessive suffixes stay outside
    spans. Quoted phrases take priority over overlapping capitalized spans.
    ``masked_token_count`` counts original lexical slots intersected by spans,
    not the number of replacement words or WordPieces.
    """
    if not isinstance(question, str):
        raise TypeError('question must be a string')
    tokens = _tokens(question)
    quoted = _quoted_spans(question, tokens)
    capitalized = [span for span in _capitalized_spans(question, tokens)
                   if not any(span[0] < other[1] and span[1] > other[0] for other in quoted)]
    spans = sorted(quoted + capitalized)
    pieces, cursor = [], 0
    for start, end, _ in spans:
        pieces.extend((question[cursor:start], 'entity'))
        cursor = end
    pieces.append(question[cursor:])
    masked = sum(any(left < end and right > start for start, end, _ in spans)
                 for left, right, _ in tokens)
    return {
        'original': question,
        'transformed': ''.join(pieces),
        'proxy_spans': [{'start': start, 'end': end, 'text': question[start:end], 'source': source}
                        for start, end, source in spans],
        'masked_token_count': masked,
        'original_token_count': len(tokens),
    }


def _self_check():
    unicode_question = 'Who was \u00c9mile Zola\u2019s father?'
    unicode_view = relation_view(unicode_question)
    assert unicode_view['transformed'] == 'Who was entity\u2019s father?'
    assert [span['text'] for span in unicode_view['proxy_spans']] == ['\u00c9mile Zola']
    assert unicode_view['masked_token_count'] == 2 and unicode_view['original_token_count'] == 5
    assert relation_view(unicode_question) == unicode_view
    assert all(unicode_question[span['start']:span['end']] == span['text']
               for span in unicode_view['proxy_spans'])

    quoted_question = 'What links Jean-Luc\'s film and \u201cthe silent world\u201d?'
    quoted_view = relation_view(quoted_question)
    assert quoted_view['transformed'] == 'What links entity\'s film and \u201centity\u201d?'
    assert quoted_view['masked_token_count'] == 4 and quoted_view['original_token_count'] == 8

    relation_question = 'Who was Ludwig van Beethoven\u2019s father and director of film?'
    relation_result = relation_view(relation_question)
    assert relation_result['transformed'] == 'Who was entity\u2019s father and director of film?'
    assert relation_result['masked_token_count'] == 3 and relation_result['original_token_count'] == 10
    return {'checks_passed': 3, 'examples': [unicode_view, quoted_view, relation_result]}


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(_self_check(), ensure_ascii=False, indent=2))
    else:
        parser.print_help()
