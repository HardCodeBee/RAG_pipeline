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


class Reranker:
    def __init__(self):
        import torch
        from sentence_transformers import CrossEncoder
        spec = json.loads(Path(__file__).with_name('RERANKER_FROZEN.json').read_text(encoding='utf8'))
        snapshot = Path(spec['snapshot'])
        for name, expected in spec['file_sha256'].items():
            with (snapshot / name).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError('Frozen reranker file changed: ' + name)
        self.model = CrossEncoder(str(snapshot), device='cuda', local_files_only=True,
                                  max_length=512, activation_fn=torch.nn.Identity())

    def rerank(self, question, documents, k):
        start = time.perf_counter()
        scores = self.model.predict([(question, d['text']) for d in documents], batch_size=4, show_progress_bar=False)
        ranked = sorted(zip(scores.tolist(), documents), key=lambda item: (-item[0], -item[1]['rrf_score'], item[1]['doc_id']))
        return [{**d, 'rerank_score': float(score)} for score, d in ranked[:k]], time.perf_counter() - start


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
                reranker = Reranker()
            documents, seconds = reranker.rerank(case['question'], union, config['final_k'])
            packet = {'qid': case['qid'], 'docs': documents, 'union_doc_ids': [d['doc_id'] for d in union], 'rerank_seconds': seconds}
            store.put('hybrid', case['qid'], packet)
        context = '\n\n'.join(d['block'] for d in packet['docs'])
        key, result = transport.ask('final', PROMPT.format(question=case['question'], context=context), 64)
        store.put('prediction', case['qid'], {'qid': case['qid'], 'question': case['question'],
                  'prediction': result.get('answer') if result['status'] == 'success' else None,
                  'status': result['status'], 'request_id': key, 'finish_reason': result.get('finish_reason'),
                  'final_doc_ids': [d['doc_id'] for d in packet['docs']]})
        store.export()
