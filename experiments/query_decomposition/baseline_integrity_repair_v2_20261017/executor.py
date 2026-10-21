"""Same asynchronous scheduling, also accepting one-case retrieve_many calls."""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Event, Thread
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_execution import Proxy, Operation, ExecutionAborted


def execute(module, cases, retrieval, transport, store, config, api, workers):
    for start in range(0, len(cases), workers):
        wave = cases[start:start+workers]
        queue, aborted = Queue(), Event()
        live = set(range(len(wave))); outstanding = {}
        def run(index):
            error = None
            try:
                module.run([wave[index]], Proxy(index, 'retrieval', queue, aborted, {'retrieve','retrieve_many'}),
                    Proxy(index, 'transport', queue, aborted, {'ask'}),
                    Proxy(index, 'store', queue, aborted, {'get', 'put', 'event', 'export'}), config)
            except BaseException as exc:
                error = exc
            queue.put(Operation(index, 'done', 'done', (error,), {}, Future()))
        threads = [Thread(target=run, args=(i,), daemon=True) for i in sorted(live)]
        for thread in threads:
            thread.start()
        try:
            while live:
                ops = [queue.get()]
                while True:
                    try: ops.append(queue.get_nowait())
                    except Empty: break
                searches = []
                for op in ops:
                    if op.target == 'done':
                        live.remove(op.index)
                        if op.args[0] is not None: raise op.args[0]
                        continue
                    if op.target == 'api_complete':
                        result, error = op.args
                        outstanding.pop(op.index, None)
                        if error is not None:
                            op.future.set_exception(error)
                            raise error
                        op.future.set_result(result)
                        continue
                    if op.index in outstanding:
                        raise RuntimeError('A case cannot submit two operations simultaneously')
                    outstanding[op.index] = op
                    if op.target == 'store':
                        result = getattr(store, op.name)(*op.args, **op.kwargs)
                        outstanding.pop(op.index); op.future.set_result(result)
                    elif op.target == 'transport':
                        future = api.pool.submit(api._ask, wave[op.index]['qid'], op.args, op.kwargs)
                        def done(future, operation=op):
                            try: result, error = future.result(), None
                            except BaseException as exc: result, error = None, exc
                            queue.put(Operation(operation.index, 'api_complete', 'complete', (result, error), {}, operation.future))
                        future.add_done_callback(done)
                    elif op.target == 'retrieval':
                        if op.kwargs:
                            raise ValueError('Unexpected retrieval keyword argument')
                        if op.name=='retrieve' and len(op.args)==3:
                            qid,query,k=op.args;rows=[{'qid':qid,'query':query}]
                        elif op.name=='retrieve_many' and len(op.args)==2:
                            rows,k=op.args
                        else:raise ValueError('Unexpected retrieval operation')
                        if len(rows)!=1 or rows[0]['qid']!=wave[op.index]['qid']:
                            raise ValueError('Unexpected cross-question retrieval request')
                        searches.append((op,rows[0],k))
                    else:
                        raise ValueError('Unexpected operation')
                for k in sorted({item[2] for item in searches}):
                    group=[item for item in searches if item[2]==k]
                    packets=retrieval.retrieve_many([row for op,row,size in group],k)
                    for (op,row,size),packet in zip(group,packets):
                        outstanding.pop(op.index);op.future.set_result(packet if op.name=='retrieve' else [packet])
        finally:
            aborted.set()
            for op in outstanding.values():
                if not op.future.done(): op.future.set_exception(ExecutionAborted('Owner stopped'))
            for thread in threads: thread.join(timeout=.2)
            while not queue.empty():
                op = queue.get_nowait()
                if not op.future.done(): op.future.set_exception(ExecutionAborted('Owner stopped'))
