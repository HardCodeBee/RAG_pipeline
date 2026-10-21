"""Bounded direct-kernel query parallelism and measured API concurrency."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import httpx

from .common import HERE, RUN, read, save, sha, log
from . import campaign, execution, qd_batch_epoch, throughput_epoch
from .retrieval import Retrieval
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest


class ParallelScan:
    def __init__(self, base, threads, query_batch):
        self.base, self.threads, self.query_batch = base, threads, query_batch
    def __getattr__(self, name):
        return getattr(self.base, name)
    def search_many(self, vectors, k, *, query_batch_size=32):
        import faiss
        before = faiss.omp_get_max_threads()
        threshold = faiss.cvar.distance_compute_blas_threshold
        batch = min(self.query_batch, len(vectors))
        # Preserve the original direct-vs-BLAS kernel boundary.
        if query_batch_size >= threshold or batch >= threshold:
            batch = query_batch_size
        try:
            faiss.omp_set_num_threads(self.threads)
            return self.base.search_many(vectors, k, query_batch_size=batch)
        finally:
            faiss.omp_set_num_threads(before)


class CapacityBudget(ParallelBudget):
    def __init__(self, path, cap, interval):
        super().__init__(path, cap)
        self.interval = interval
    def wait_for_start(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            values = dict(self.db.execute('SELECT key,value FROM scheduler'))
            now = time.time()
            start = max(now, values.get('next_start', now))
            if start - now > 300:
                raise RuntimeError('Rate scheduler clock/backlog requires audit')
            previous = values.get('adaptive_minimum_interval', 2)
            interval = max(self.interval, previous if previous > 2 else self.interval)
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('next_start',?)", (start+interval,))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('attempt_count',?)", (int(values.get('attempt_count',0))+1,))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        remaining = max(0, start-time.time())
        while remaining > 0:
            delay = min(60, remaining)
            time.sleep(delay)
            remaining -= delay


class CapacityAPI(throughput_epoch.SlotAPI):
    def __init__(self, transport, slots, interval):
        if not 1 <= slots <= 8 or interval < 2/3:
            raise ValueError('Capacity outside validated upper limits')
        self.transport, self.concurrency, self.interval = transport, slots, interval
        self.pool = ThreadPoolExecutor(max_workers=slots)
        self.keys, self.keys_lock = {}, Lock()
    def _ask(self, qid, args, kwargs):
        stage, prompt, max_tokens = args
        if kwargs:
            raise ValueError('Unexpected transport arguments')
        original = self.transport
        key = digest([original.store.namespace, {'model': original.model,
            'messages': [{'role':'user','content':prompt}], 'temperature':0., 'max_tokens':max_tokens,
            'extra_body':{'enable_thinking':False}}])
        with self.keys_lock:
            mutex = self.keys.setdefault(key, Lock())
        with mutex:
            store = WorkerStore(original.store)
            budget = CapacityBudget(original.budget.path, original.budget.cap, self.interval)
            try:
                transport = execution.ExactTransport(store, original.client, budget, original.model)
                transport.qid = qid
                return transport.ask(stage, prompt, max_tokens)
            finally:
                store.db.close()
                budget.db.close()


SETTINGS = None


def run_parallel(module, cases, retrieval, transport, store, config):
    api = CapacityAPI(transport, SETTINGS['api_slots'], SETTINGS['start_interval'])
    try:
        return throughput_epoch.EXECUTE(module, cases, retrieval, transport, store, config, api, 20)
    finally:
        api.close()


def install():
    global SETTINGS
    settings = read(RUN / 'SCAN_API_SETTINGS_FROZEN.json')
    if settings['source_sha256'] != sha(__file__):
        raise ValueError('Scan/API execution source changed')
    for name, expected in settings['validation_sha256'].items():
        if sha(RUN/name) != expected:
            raise ValueError('Scan/API validation receipt changed')
    SETTINGS = settings['selected']
    previous_load = Retrieval.load
    def load(self):
        previous_load(self)
        if not isinstance(self.index, ParallelScan):
            self.index = ParallelScan(self.index, SETTINGS['scan_threads'], SETTINGS['scan_query_batch'])
    Retrieval.load = load
    execution.run_parallel = campaign.run_parallel = qd_batch_epoch.run_parallel = throughput_epoch.run_parallel = run_parallel
    original_transport = httpx.HTTPTransport
    class Transport(original_transport):
        def __init__(self, *args, **kwargs):
            limits = kwargs.get('limits')
            if limits is not None and limits.max_connections == 3:
                kwargs['limits'] = httpx.Limits(max_connections=SETTINGS['api_slots'],
                    max_keepalive_connections=SETTINGS['api_slots'], keepalive_expiry=10)
            super().__init__(*args, **kwargs)
    httpx.HTTPTransport = Transport
    log('scan_and_api_capacity_epoch_activated', epoch='scan_api_capacity_v1', settings=SETTINGS,
        frozen_method_or_sdk_payload_changed=False, global_budget_cap_cny=200,
        method_private_state_and_unknown_request_policy_preserved=True)
