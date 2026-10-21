"""Offline review aid only; never imported by the RAG experiment or strategy.

Print selected-case original passages around literal review terms. The terms are
solely for explaining already completed outcomes, never generation or sampling.
"""
import json
from pathlib import Path
import sys
from .report_round2 import ROOT, HERE, OUT, read, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sources = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    changed = read(OUT / 'CHANGED_CASES.progress.json')
    for r in changed:
        if abs(r['delta_parent']) < 1e-12: continue
        base = read(ROOT / sources[r['id']]['base'])
        candidate = read(ROOT / r['candidate_source'])
        queries = [n['request'] for n in base['nodes']]
        print('\nID', r['id'], 'ARM', r['arm'], 'QUERY', r['query'])
        print('PARENT', r['parent_final'], 'CANDIDATE', r['candidate_final'], 'GOLD', r['reference_answer'])
        print('QUESTIONS', json.dumps(queries, ensure_ascii=False))
        print('ORDINARY ANSWERS', json.dumps([n['answer'] for n in base['nodes'][:-1]], ensure_ascii=False))
        needles = {'2wiki:6d2f94300bb011ebab90acde48001122': ['Timothy', 'Laurence'],
                   'musique:2hop__789565_416401': ['Jipeng', 'Chang', 'Kangzong'],
                   'hotpotqa:5ae0a48155429945ae9593df': ['vigorish', 'bookmaker'],
                   'hotpotqa:5a84ad3a5542992a431d1a8b': ['Back to December', 'forgiveness', 'Lemoine'],
                   'musique:3hop1__109422_720914_27537': ['Roncalli', 'conclave', 'Titian', 'Pius'],
                   'musique:4hop1__94201_642284_131926_13165': ['Rapson', 'Paris', 'Adams', 'Mississippi', 'Pacific'],
                   'hotpotqa:5abec66b5542997ec76fd360': ['Veltjens', 'Göring', 'Pour le'],
                   '2wiki:cdb8b7540bd911eba7f7acde48001122': ['Lanfield', 'Westwood'],
                   'hotpotqa:5a72a9ac5542991f9a20c55f': ['Balthazar', 'fourth hour']}.get(r['id'], [r['reference_answer']])
        docs = {}
        for origin, packet in [('original', base['original_retrieval']), ('goal', candidate['nodes'][-1]['reader_packet'])]:
            for d in packet['docs']:
                text = d['text']
                positions = [text.casefold().find(n.casefold()) for n in needles]
                positions = [p for p in positions if p >= 0]
                if not positions: continue
                key = d['doc_id']
                if key in docs: continue
                snippets = []
                for p in positions:
                    snippet = text[max(0, p-140):min(len(text), p+650)]
                    if snippet not in snippets: snippets.append(snippet)
                docs[key] = {'origin': origin, 'source': d['source'], 'rank': d['rank'], 'snippets': snippets}
        print('EVIDENCE', json.dumps(list(docs.values()), ensure_ascii=False))


if __name__ == '__main__': main()
