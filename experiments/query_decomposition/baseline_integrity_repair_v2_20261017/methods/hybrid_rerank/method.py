"""Original-query BM25+BGE rank fusion, then cross-encoder over the union."""
import hashlib
import json
from pathlib import Path
import time

PROMPT = "Answer the question using only the provided context.\nReturn only the short answer. Do not add citations or explanation.\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:"


def fuse(dense, sparse, constant):
    docs, scores = {}, {}
    for ranking in (dense, sparse):
        for rank, doc in enumerate(ranking, 1):
            docs.setdefault(doc['doc_id'], doc)
            scores[doc['doc_id']] = scores.get(doc['doc_id'], 0) + 1 / (constant + rank)
    return [{**docs[key], 'rrf_score': scores[key]} for key in sorted(docs, key=lambda key: (-scores[key], key))]


from ...reranking import Reranker


def run(cases, retrieval, transport, store, config):
    reranker = None
    for case in cases:
        transport.qid = case['qid']
        if store.get('prediction', case['qid']) is not None:
            continue
        packet = store.get('hybrid', case['qid'])
        if packet is None:
            dense = retrieval.retrieve(case['qid'], case['question'], config['per_retriever_k'])
            sparse = retrieval.retrieve_sparse(case['qid'], case['question'], config['per_retriever_k'])
            union = fuse(dense['docs'], sparse['docs'], config['rrf_constant'])
            if reranker is None:
                reranker = Reranker(store)
            documents, seconds = reranker.rerank(case['question'], union, config['final_k'], qid=case['qid'])
            packet = {'qid': case['qid'], 'docs': documents, 'union_doc_ids': [d['doc_id'] for d in union], 'rerank_seconds': seconds}
            store.put('hybrid', case['qid'], packet)
        context = '\n\n'.join(d['block'] for d in packet['docs'])
        key, result = transport.ask('final', PROMPT.format(question=case['question'], context=context), 64)
        store.put('prediction', case['qid'], {'qid': case['qid'], 'question': case['question'],
                  'prediction': result.get('answer') if result['status'] == 'success' else None,
                  'status': result['status'], 'request_id': key, 'finish_reason': result.get('finish_reason'),
                  'final_doc_ids': [d['doc_id'] for d in packet['docs']]})
        store.export()
