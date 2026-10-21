"""Keep frozen BGE/index/k; encode every token of overflow queries explicitly."""
import time
import numpy as np
from .common import digest, IntegrityError
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval as ParentRetrieval
from experiments.query_decomposition.comprehensive_eval_v1_20261015.exact_parallel_scan_v2 import ExactParallelScan


def windows(token_ids, width):
    if width < 1:
        raise IntegrityError('No content token fits the encoder')
    return [token_ids[i:i+width] for i in range(0, len(token_ids), width)] or [[]]


def encode_complete(encoder, queries):
    import torch
    model = encoder._model
    tokenizer = model[0].tokenizer
    width = min(512, model[0].auto_model.config.max_position_embeddings)
    if encoder.query_prefix:
        raise IntegrityError('Unexpected query prefix under frozen BGE protocol')
    ids = [tokenizer(q, add_special_tokens=False, truncation=False, verbose=False)['input_ids'] for q in queries]
    special = tokenizer.num_special_tokens_to_add(pair=False)
    vector = np.zeros((len(queries), encoder.dimension), dtype=np.float32)
    small = [i for i, token in enumerate(ids) if len(token)+special <= width]
    if small:
        # Byte-identical original encoder path for every short query.
        vector[small] = encoder.encode_queries([queries[i] for i in small])
    metadata = []
    for i, tokens in enumerate(ids):
        chunks = windows(tokens, width-special)
        record = {'query_tokens': len(tokens)+special, 'content_tokens': len(tokens),
            'processed_content_tokens': sum(map(len, chunks)),
            'encoded_window_tokens': [len(c)+special for c in chunks],
            'encoder_truncated': False, 'window_count': len(chunks),
            'overflow_policy': 'original_short_query' if len(chunks)==1 else 'all_nonoverlapping_windows_content_weighted_mean_l2'}
        assert record['processed_content_tokens'] == len(tokens)
        assert all(t <= width for t in record['encoded_window_tokens'])
        if i not in small:
            features = [tokenizer.prepare_for_model(c, add_special_tokens=True, truncation=False,
                        return_attention_mask=True) for c in chunks]
            embeddings = []
            for start in range(0, len(features), encoder.batch_size):
                batch = tokenizer.pad(features[start:start+encoder.batch_size], padding=True, return_tensors='pt')
                batch = {k: value.to(model.device) for k, value in batch.items()}
                with torch.inference_mode():
                    encoded = model(batch)['sentence_embedding']
                    encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
                embeddings.append(encoded.cpu().numpy())
            merged = np.average(np.concatenate(embeddings), axis=0, weights=[max(1, len(c)) for c in chunks])
            norm = np.linalg.norm(merged)
            if not np.isfinite(norm) or norm == 0:
                raise IntegrityError('Invalid pooled query vector')
            vector[i] = merged / norm
        metadata.append(record)
    return np.ascontiguousarray(vector, dtype=np.float32), metadata


class Retrieval(ParentRetrieval):
    def __init__(self, store, resources, catalog):
        super().__init__(store, resources=resources)
        self.catalog = catalog

    def load(self):
        super().load()
        if not isinstance(self.index, ExactParallelScan):
            self.index = ExactParallelScan(self.index, 4, 64)

    def retrieve_many(self, cases, k):
        keys = [digest([c['qid'], c['query'], k, self.resources]) for c in cases]
        pending = []
        for case, key in zip(cases, keys):
            if self.store.get('retrieval', key) is not None:
                continue
            old = self.catalog.retrievals.get(('retrieval', case['qid'], case['query'], k))
            if old:
                packet = {**old, 'processed_content_tokens': max(0, old['query_tokens']-2),
                    'encoded_window_tokens': [old['query_tokens']], 'window_count': 1,
                    'overflow_policy': 'original_short_query', 'encoder_truncated': False,
                    'reused_from_db': str(self.catalog.path)}
                self.store.put('retrieval', key, packet)
                continue
            pending.append((case, key))
        if pending:
            self.load()
            start = time.perf_counter()
            vectors, metadata = encode_complete(self.encoder, [c['query'] for c, _ in pending])
            scores, indices = self.index.search_many(vectors, k+1, query_batch_size=len(pending))
            elapsed = time.perf_counter()-start
            for (case, key), values, ids, meta in zip(pending, scores.tolist(), indices.tolist(), metadata):
                docs = []
                for value, chunk in zip(values, self.chunks.get_many(ids)):
                    if chunk.doc_id == case['qid']:
                        continue
                    rank = len(docs)+1
                    docs.append({'rank': rank, 'score': value, 'doc_id': chunk.doc_id,
                        'vector_id': chunk.vector_id, 'source': chunk.source, 'text': chunk.text,
                        'block': f'[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}'})
                    if len(docs)==k:
                        break
                if len(docs)!=k:
                    raise IntegrityError('Incomplete retrieval; do not silently change k')
                self.store.put('retrieval', key, {'qid': case['qid'], 'query': case['query'], 'k': k,
                    'docs': docs, 'batch_queries': len(pending), 'batch_seconds': elapsed, **meta})
        packets = [self.store.get('retrieval', key) for key in keys]
        for c, p in zip(cases, packets):
            self.store.event('retrieval', qid=c['qid'], query=c['query'], k=k, doc_ids=[d['doc_id'] for d in p['docs']])
        return packets

    def retrieve_sparse(self, qid, query, k):
        key = digest(['bm25', qid, query, k, self.resources])
        if self.store.get('sparse_retrieval', key) is None:
            old = self.catalog.retrievals.get(('sparse_retrieval', qid, query, k))
            if old:
                self.store.put('sparse_retrieval', key, old)
        return super().retrieve_sparse(qid, query, k)
