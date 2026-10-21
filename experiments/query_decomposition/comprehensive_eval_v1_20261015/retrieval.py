"""Same BGE/BM25 algorithms; generalized corpus-count load boundary only."""
from pathlib import Path
from .common import sha, read
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Retrieval as OriginalRetrieval

class Retrieval(OriginalRetrieval):
    def load(self):
        if self.index is not None:return
        if self.resources['corpus']['dataset']=='hotpotqa':return super().load()
        import torch,faiss
        from src.embedders.text_embedder import TextEmbedder
        from src.indexes.streaming_flat_index import StreamingFlatIPIndex
        from src.retrievers.chunk_store import JsonlOffsetChunkStore
        corpus=self.resources['corpus'];spec=self.resources['dense']['bge'];n=corpus['documents']
        if sha(corpus['manifest'])!=corpus['manifest_sha256']:raise ValueError('Corpus manifest differs')
        if sha(Path(spec['build'])/'DENSE_FROZEN.json')!=spec['manifest_sha256']:raise ValueError('Dense manifest differs')
        pins=read(Path(spec['build'])/'DENSE_FROZEN.json')
        for name,expected in pins['part_sha256'].items():
            if sha(Path(spec['build'])/name)!=expected:raise ValueError('Dense part differs')
        torch.set_num_threads(4);faiss.omp_set_num_threads(4)
        self.index=StreamingFlatIPIndex(spec['build'],corpus_chunk_size=50000)
        if self.index.count!=n or self.index.dimension!=384:raise ValueError('Incomplete dataset index')
        self.encoder=TextEmbedder(backend='sentence_transformers',model_name=spec['query_model'],revision=spec['query_revision'],normalize=True,
          batch_size=16,query_prefix='',document_prefix='',max_sequence_length=512,local_files_only=True,device=self.device)
        if self.encoder.embedding_space('inner_product').to_dict()!=spec['space']:raise ValueError('Shared BGE query space differs')
        self.chunks=JsonlOffsetChunkStore(corpus['chunks'],corpus['offsets'],expected_rows=n)
        self.store.event('dataset_retriever_loaded',dataset=corpus['dataset'],documents=n)
