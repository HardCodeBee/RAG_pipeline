"""Execution-only API slot filling and Hybrid dense-prefetch compatibility."""
import gc
import inspect
from .common import HERE, RUN, read, save, sha, log
from . import campaign, execution, qd_batch_epoch
from experiments.query_decomposition.hotpot_formal_eval_v1 import parallel_execution as original
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.hybrid_rerank import method as hybrid


class SlotAPI(execution.API):
    def ask_group(self, operations, questions):
        # The existing executor has exactly three worker threads. Queued work
        # takes a free slot immediately, rather than waiting for three peers.
        futures = [self.pool.submit(self._ask, questions[op.index]['qid'], op.args, op.kwargs)
                   for op in operations]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(None)
                errors.append(exc)
        if errors:
            raise errors[0]
        return results


def slot_executor():
    source = inspect.getsource(original._execute)
    before = 'ready = [pending[i] for i in calls[:api.concurrency]]'
    after = 'ready = [pending[i] for i in calls]'
    if source.count(before) != 1:
        raise ValueError('Original execution controller changed')
    namespace = dict(vars(original))
    exec(compile(source.replace(before, after), str(HERE / 'throughput_epoch.py') + ':slot_executor', 'exec'), namespace)
    return namespace['_execute']


EXECUTE = slot_executor()


def run_parallel(module, cases, retrieval, transport, store, config):
    api = SlotAPI(transport, 3)
    try:
        return EXECUTE(module, cases, retrieval, transport, store, config, api, 20)
    finally:
        api.close()


_RERANKER = None
_RERANKER_CLASS = hybrid.Reranker


def reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = _RERANKER_CLASS()
    return _RERANKER


ORIGINAL_HYBRID_RUN = hybrid.run


class HybridFinal:
    @staticmethod
    def run(cases, retrieval, transport, store, config):
        ORIGINAL_HYBRID_RUN(cases, retrieval, transport, store, config)


def hybrid_run(cases, retrieval, transport, store, config):
    pending = [c for c in cases if store.get('prediction', c['qid']) is None
               and store.get('hybrid', c['qid']) is None]
    if pending:
        qd_batch_epoch.batch_retrieve(retrieval,
            [{'qid': c['qid'], 'query': c['question']} for c in pending], config['per_retriever_k'])
        model = reranker()
        for case in pending:
            dense = retrieval.retrieve(case['qid'], case['question'], config['per_retriever_k'])
            sparse = retrieval.retrieve_sparse(case['qid'], case['question'], config['per_retriever_k'])
            union = hybrid.fuse(dense['docs'], sparse['docs'], config['rrf_constant'])
            documents, seconds = model.rerank(case['question'], union, config['final_k'])
            store.put('hybrid', case['qid'], {'qid': case['qid'], 'docs': documents,
                      'union_doc_ids': [d['doc_id'] for d in union], 'rerank_seconds': seconds})
    run_parallel(HybridFinal, cases, retrieval, transport, store, config)


def install():
    parity = read(RUN / 'THROUGHPUT_PARITY.json')
    if not parity['passed'] or parity['api_submissions'] != 0 or parity['execution_source_sha256'] != sha(__file__):
        raise ValueError('Execution throughput parity not verified')
    epoch = {'id': 'three_slots_hybrid_prefetch_v1', 'source_sha256': sha(__file__),
             'parity_sha256': sha(RUN / 'THROUGHPUT_PARITY.json'),
             'api_concurrency': 3, 'global_start_rate_policy_changed': False,
             'changes': ['fill the same three API slots from independent ready calls',
                         'prefetch independent Hybrid dense searches with original per-question encoding',
                         'reuse the same immutable reranker model instance'],
             'method_controllers_prompts_parameters_samples_and_native_scores_changed': False,
             'methods_keep_private_state_and_request_cache': True}
    path = campaign.TARGET / 'THROUGHPUT_EXECUTION_EPOCH.json'
    if path.exists() and read(path) != epoch:
        raise ValueError('Throughput epoch changed')
    if not path.exists():
        save(path, epoch)
        log('verified_execution_throughput_epoch_activated', **epoch)
    execution.run_parallel = campaign.run_parallel = qd_batch_epoch.run_parallel = run_parallel
    hybrid.run = hybrid_run
    qd_batch_epoch.qd.Reranker = reranker
    previous = campaign.Campaign.execute

    def execute(self, *args, **kwargs):
        if sha(__file__) != epoch['source_sha256']:
            raise ValueError('Throughput source changed')
        result = previous(self, *args, **kwargs)
        gc.collect()
        return result

    campaign.Campaign.execute = execute
