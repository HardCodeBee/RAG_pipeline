"""Read-only trajectory comparison, plus experiment-local inspection artifacts."""
from . import run as r
from collections import Counter
import json


def ids(packet):
    return [d['doc_id'] for d in packet['docs']] if packet else None


def compare(left, right):
    counts = {role: Counter() for role in ['ordinary', 'goal']}
    changed = []
    for cid, first in left.items():
        last = right[cid]
        assert len(first['nodes']) == len(last['nodes'])
        for a, b in zip(first['nodes'], last['nodes']):
            c = counts[a['role']]
            c['nodes'] += 1
            fields = {
                'raw_question_changed': a['request'] != b['request'],
                'rewrite_changed': a['rewritten_query'] != b['rewritten_query'],
                'original_docs_changed': ids(a['original_retrieval']) != ids(b['original_retrieval']),
                'rewrite_docs_changed': ids(a['rewritten_retrieval']) != ids(b['rewritten_retrieval']),
                'reader_docs_changed': ids(a['reader_packet']) != ids(b['reader_packet']),
                'reader_text_changed': r.parent.baseline.passages(a['reader_packet']) != r.parent.baseline.passages(b['reader_packet']),
                'prior_answers_changed': a['prior_complete_answers'] != b['prior_complete_answers'],
                'answer_changed': a['answer'] != b['answer'],
                'initial_answer_changed': a['initial_answer'] != b['initial_answer'],
            }
            for k, v in fields.items():
                c[k] += int(v)
            if any(fields.values()):
                changed.append({'id': cid, 'original_question': first['question'], 'step_id': a['step_id'], 'role': a['role'],
                    'question': a['request'], **fields, 'before_rewrite': a['rewritten_query'], 'after_rewrite': b['rewritten_query'],
                    'before_answer': a['answer'], 'after_answer': b['answer'],
                    'before_prior': a['prior_complete_answers'], 'after_prior': b['prior_complete_answers'],
                    'before_sources': [d['source'] for d in a['reader_packet']['docs']],
                    'after_sources': [d['source'] for d in b['reader_packet']['docs']]})
    return {'counts': {k: dict(v) for k, v in counts.items()}, 'changed': changed}


def main():
    old = {v['id']: v['trajectory'] for v in r.rows(r.CONTROL / 'control90.jsonl')}
    revised = r.predictions(r.CONTROL)
    a, b = [r.predictions(r.OUT / arm) for arm in r.ARMS]
    result = {'old_to_revised': compare(old, revised), 'revised_to_A': compare(revised, a), 'revised_to_B': compare(revised, b),
        'old_to_A': compare(old, a), 'old_to_B': compare(old, b)}
    r.save(r.OUT / 'STAGE_COMPARISON.json', result)
    r.emit({k: v['counts'] for k, v in result.items()})
    for label in ['revised_to_A', 'revised_to_B']:
        for row in result[label]['changed']:
            if row['answer_changed'] or row['rewrite_changed']:
                r.emit({'contrast': label, **{k: row[k] for k in ['id', 'step_id', 'role', 'question', 'before_answer', 'after_answer', 'before_rewrite', 'after_rewrite', 'reader_docs_changed', 'prior_answers_changed']}})


if __name__ == '__main__':
    main()
