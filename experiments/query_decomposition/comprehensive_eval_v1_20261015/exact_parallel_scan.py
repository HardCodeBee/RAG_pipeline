"""Parallelize independent original single-query kernels on read-only blocks."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace, FunctionType
import inspect
import textwrap
import numpy as np
import faiss
from src.indexes import streaming_flat_index as original

_NAMESPACE = dict(vars(original))
exec(compile(textwrap.dedent(inspect.getsource(original.StreamingFlatIPIndex.search_many)),
             __file__ + ':original_scan', 'exec'), _NAMESPACE)
_SEARCH_CODE = _NAMESPACE['search_many'].__code__
_FLAT = faiss.IndexFlatIP


class ExactParallelScan:
    def __init__(self, base, threads, query_batch):
        self.base, self.threads, self.query_batch = base, threads, query_batch
    def __getattr__(self, name):
        return getattr(self.base, name)
    def search_many(self, vectors, k, *, query_batch_size=32):
        previous_threads = faiss.omp_get_max_threads()
        try:
            if query_batch_size != 1:
                # Existing grouped calls retain their original Faiss arithmetic.
                faiss.omp_set_num_threads(self.threads)
                return self.base.search_many(vectors,k,query_batch_size=query_batch_size)
            faiss.omp_set_num_threads(1)
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                class ReadOnlyBlock:
                    def __init__(self, dimension):self.index=_FLAT(dimension)
                    def add(self, block):self.index.add(block)
                    def reset(self):self.index.reset()
                    def search(self, queries, count):
                        def single(i):
                            faiss.omp_set_num_threads(1)
                            return self.index.search(queries[i:i+1],count)
                        values=list(pool.map(single,range(len(queries))))
                        return np.concatenate([r[0] for r in values]),np.concatenate([r[1] for r in values])
                namespace=dict(_NAMESPACE)
                namespace['_require_faiss']=lambda:SimpleNamespace(IndexFlatIP=ReadOnlyBlock)
                search=FunctionType(_SEARCH_CODE,namespace)
                return search(self.base,vectors,k,query_batch_size=min(self.query_batch,len(vectors)))
        finally:
            faiss.omp_set_num_threads(previous_threads)
