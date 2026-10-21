"""Frozen cross-encoder; only overflow pairs use full-coverage windows."""
import hashlib
import time
import numpy as np
from .common import HERE, read, digest, IntegrityError
from .retrieval import windows


class Reranker:
    MODEL = None
    TOKENIZER = None
    SNAPSHOT = None

    def __init__(self, store):
        if Reranker.TOKENIZER is None:
            spec = read(HERE / 'methods/hybrid_rerank/RERANKER_FROZEN.json')
            from pathlib import Path
            snapshot = Path(spec['snapshot'])
            for name, expected in spec['file_sha256'].items():
                with (snapshot/name).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest()!=expected:
                        raise IntegrityError('Frozen reranker weight changed')
            from transformers import AutoTokenizer
            Reranker.SNAPSHOT = snapshot
            Reranker.TOKENIZER = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.store = store

    def rerank(self, question, documents, k, qid):
        import torch
        start = time.perf_counter()
        tok = Reranker.TOKENIZER
        q = tok(question, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
        ids = [tok(d['text'], add_special_tokens=False, truncation=False, verbose=False)['input_ids'] for d in documents]
        special = tok.num_special_tokens_to_add(pair=True)
        counts = [len(q)+len(d)+special for d in ids]
        short = [i for i, n in enumerate(counts) if n <= 512]
        old = self.store.get('legacy_hybrid', qid)
        reuse = bool(old and len(short)==len(documents) and
                     set(old.get('union_doc_ids', []))=={d['doc_id'] for d in documents})
        if not reuse and Reranker.MODEL is None:
            from sentence_transformers import CrossEncoder
            Reranker.MODEL = CrossEncoder(str(Reranker.SNAPSHOT), device='cuda', local_files_only=True,
                max_length=512, activation_fn=torch.nn.Identity())
        self.model = Reranker.MODEL
        scores = np.zeros(len(documents), dtype=np.float32)
        if short and not reuse:
            scores[short] = self.model.predict([(question, documents[i]['text']) for i in short],
                batch_size=4, show_progress_bar=False)
        for i, doc in enumerate(documents):
            if i in short:
                windows_count, processed, mode = 1, counts[i], 'original_short_pair'
            else:
                q_parts = windows(q, len(q) if len(q)<512-special else 256)
                features = []
                for part in q_parts:
                    for passage in windows(ids[i], 512-special-len(part)):
                        item = tok.prepare_for_model(part, pair_ids=passage, add_special_tokens=True,
                            truncation=False, return_attention_mask=True)
                        if len(item['input_ids'])>512:
                            raise IntegrityError('Reranker window too long')
                        features.append(item)
                outputs = []
                for j in range(0, len(features), 4):
                    batch = tok.pad(features[j:j+4], padding=True, return_tensors='pt')
                    batch = {name: value.to(self.model.model.device) for name, value in batch.items()}
                    with torch.inference_mode():
                        logits = self.model.model(**batch, return_dict=True).logits
                    if logits.shape[-1]!=1:
                        raise IntegrityError('Unexpected reranker labels')
                    outputs.extend(logits[:, 0].cpu().tolist())
                scores[i] = max(outputs)
                windows_count = len(features)
                processed = sum(len(x['input_ids']) for x in features)
                mode = 'all_full_coverage_pair_windows_max_logit'
            self.store.put('rerank_token', digest([qid, doc['doc_id'], question]), {
                'qid': qid, 'doc_id': doc['doc_id'], 'full_question_tokens': len(q),
                'full_document_tokens': len(ids[i]), 'full_pair_tokens': counts[i],
                'windows': windows_count, 'processed_pair_tokens': processed,
                'input_truncated': False, 'overflow_policy': mode,
                'reused_complete_original_ranking': reuse})
        if reuse:
            return old['docs'][:k], 0.
        ranked = sorted(zip(scores.tolist(), documents), key=lambda item: (-item[0], -item[1]['rrf_score'], item[1]['doc_id']))
        return [{**d, 'rerank_score': float(score)} for score, d in ranked[:k]], time.perf_counter()-start
