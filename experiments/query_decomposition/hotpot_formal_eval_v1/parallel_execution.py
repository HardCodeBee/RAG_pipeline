"""Batch immutable retrieval and overlap up to3 independent API calls.

Method controllers remain unchanged. Owner handles local state/retrieval;
API workers own separate SQLite connections and the same global rate/budget.
"""
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread

from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelAPI


class ExecutionAborted(RuntimeError):
    pass


@dataclass
class Operation:
    index: int
    target: str
    name: str
    args: tuple
    kwargs: dict
    future: Future


class Proxy:
    def __init__(self, index, target, queue, aborted, allowed):
        self.index, self.target, self.queue = index, target, queue
        self.aborted, self.allowed = aborted, allowed
        self.qid = None

    def __getattr__(self, name):
        if name not in self.allowed:
            raise AttributeError(name)
        def call(*args, **kwargs):
            if self.aborted.is_set():
                raise ExecutionAborted('Owner stopped; no further operations allowed')
            future = Future()
            self.queue.put(Operation(self.index, self.target, name, args, kwargs, future))
            return future.result()
        return call


def run_parallel(module, cases, retrieval, transport, store, config, workers=20, concurrency=3):
    """Execute unchanged single-question runs; batch only ready searches.

    Fixed input order resolves scheduling ties. A wave never spans methods or
    configurations. Search inputs and outputs retain their original qid.
    """
    api = ParallelAPI(transport, concurrency)
    try:
        return _execute(module, cases, retrieval, transport, store, config, api, workers)
    finally:
        api.close()


def _execute(module, cases, retrieval, transport, store, config, api, workers):
    if workers < 1:
        raise ValueError('Positive worker count required')
    for start in range(0, len(cases), workers):
        wave = cases[start:start+workers]
        queue, aborted = Queue(), Event()
        pending, live = {}, set(range(len(wave)))

        def execute(index):
            error = None
            try:
                module.run([wave[index]],
                           Proxy(index, 'retrieval', queue, aborted, {'retrieve','retrieve_many'}),
                           Proxy(index, 'transport', queue, aborted, {'ask'}),
                           Proxy(index, 'store', queue, aborted, {'get', 'put', 'event', 'export'}),
                           config)
            except BaseException as exc:
                error = exc
            queue.put(Operation(index, 'done', 'done', (error,), {}, Future()))

        threads = [Thread(target=execute, args=(i,), daemon=True) for i in live]
        for thread in threads:
            thread.start()
        try:
            while live:
                # Each live worker is either blocked on its one operation or done.
                # This barrier makes grouping independent of thread timing.
                while len(pending) < len(live):
                    operation = queue.get()
                    if operation.index not in live or operation.index in pending:
                        raise RuntimeError('Invalid worker operation sequence')
                    pending[operation.index] = operation
                done = [i for i in sorted(pending) if pending[i].target == 'done']
                for i in done:
                    operation = pending.pop(i)
                    live.remove(i)
                    if operation.args[0] is not None:
                        raise operation.args[0]
                if not live:
                    break
                local = [i for i in sorted(pending) if pending[i].target == 'store']
                if local:
                    for i in local:
                        operation = pending[i]
                        result = getattr(store, operation.name)(*operation.args, **operation.kwargs)
                        pending.pop(i)
                        operation.future.set_result(result)
                    continue
                calls = [i for i in sorted(pending) if pending[i].target == 'transport']
                if calls:
                    ready = [pending[i] for i in calls[:api.concurrency]]
                    results = api.ask_group(ready, wave)
                    for operation, result in zip(ready, results):
                        pending.pop(operation.index)
                        operation.future.set_result(result)
                    continue
                searches = [pending[i] for i in sorted(pending)]
                normalized = []
                for op in searches:
                    if op.target != 'retrieval' or op.kwargs:
                        raise RuntimeError('Unsupported retrieval operation')
                    if op.name == 'retrieve' and len(op.args) == 3:
                        qid, query, k = op.args; rows = [{'qid': qid, 'query': query}]
                    elif op.name == 'retrieve_many' and len(op.args) == 2:
                        rows, k = op.args
                    else:
                        raise RuntimeError('Unsupported retrieval operation')
                    if len(rows)>1 or any(r['qid']!=wave[op.index]['qid'] for r in rows):
                        raise ValueError('A worker may request only its own question')
                    normalized.append((op, rows, k))
                for k in sorted({item[2] for item in normalized}):
                    group = [item for item in normalized if item[2] == k]
                    inputs = [r for _, rows, _ in group for r in rows]
                    packets = retrieval.retrieve_many(inputs, k) if inputs else []
                    offset = 0
                    for op, rows, _ in group:
                        found = packets[offset:offset+len(rows)]; offset += len(rows)
                        pending.pop(op.index)
                        op.future.set_result(found[0] if op.name=='retrieve' else found)
        finally:
            aborted.set()
            for operation in pending.values():
                if not operation.future.done():
                    operation.future.set_exception(ExecutionAborted('Owner stopped'))
            # Include an operation whose owner-side execution raised after pop.
            for thread in threads:
                thread.join(timeout=.2)
            while not queue.empty():
                operation = queue.get_nowait()
                if not operation.future.done():
                    operation.future.set_exception(ExecutionAborted('Owner stopped'))
