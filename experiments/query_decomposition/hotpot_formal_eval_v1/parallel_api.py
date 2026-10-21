"""Bounded API concurrency; unchanged Transport payload/retry/budget policy."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from threading import Lock, RLock

from experiments.query_decomposition.hotpot_formal_eval_v1.common import runtime
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_rate_policy import RateBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport


FILE_LOCK = RLock()


def locked_json(path, value):
    """Atomically export state despite brief Windows reader/scanner locks.

    Retry only local replacement; never repeat an API request. On exhaustion,
    preserve the complete temporary file and raise for infrastructure audit.
    """
    with FILE_LOCK:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(name, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 3.2))


runtime.write_json = locked_json


class ParallelBudget(RateBudget):
    """Smooth starts at1/sec; retain the old6/sec-delay throttle recovery."""
    def wait_for_start(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            values=dict(self.db.execute('SELECT key,value FROM scheduler'))
            now=time.time();start=max(now,values.get('next_start',now))
            if start-now>300:raise RuntimeError('Rate scheduler clock/backlog requires audit')
            previous=values.get('adaptive_minimum_interval',2)
            interval=max(1,previous if previous>2 else 1)
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('next_start',?)",(start+interval,))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('attempt_count',?)",(int(values.get('attempt_count',0))+1,))
            self.db.commit()
        except BaseException:
            self.db.rollback();raise
        remaining=max(0,start-now)
        while remaining>0:
            delay=min(60,remaining);time.sleep(delay);remaining-=delay


class WorkerStore(runtime.Store):
    def __init__(self, original):
        # Opening a worker connection must NEVER run Store's crash recovery
        # UPDATE: another thread may have a genuine request currently in flight.
        self.directory=original.directory
        self.identity=original.identity
        self.namespace=original.namespace
        self.db=sqlite3.connect(self.directory / 'checkpoint.sqlite3',timeout=30)
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute('PRAGMA synchronous=FULL')

    def event(self, event, **fields):
        with FILE_LOCK:
            return super().event(event, **fields)

    def export(self, full=False):
        if full:
            raise ValueError('Only the owner may export full method snapshots')
        with FILE_LOCK:
            return super().export(full=False)


class ParallelAPI:
    def __init__(self, transport, concurrency=3):
        if not 1 <= concurrency <= 4:
            raise ValueError('Bounded concurrency1..4 only')
        self.transport=transport
        self.concurrency=concurrency
        self.pool=ThreadPoolExecutor(max_workers=concurrency)
        self.keys={}
        self.keys_lock=Lock()

    def _ask(self, qid, args, kwargs):
        original=self.transport
        stage,prompt,max_tokens=args
        if kwargs:
            raise ValueError('Unexpected transport arguments')
        payload={'model':original.model,'messages':[{'role':'user','content':prompt}],
                 'temperature':0.0,'max_tokens':max_tokens,'extra_body':{'enable_thinking':False}}
        key=runtime.digest([original.store.namespace,payload])
        with self.keys_lock:
            mutex=self.keys.setdefault(key,Lock())
        # Identical payloads retain the old exact private-cache semantics.
        with mutex:
            store=WorkerStore(original.store)
            budget=ParallelBudget(original.budget.path,original.budget.cap)
            try:
                transport=ResumeTransport(store,original.client,budget,original.model)
                transport.qid=qid
                return transport.ask(stage,prompt,max_tokens)
            finally:
                store.db.close();budget.db.close()

    def ask_group(self, operations, questions):
        if len(operations)>self.concurrency:
            raise ValueError('Too many simultaneous API operations')
        futures=[self.pool.submit(self._ask,questions[op.index]['qid'],op.args,op.kwargs) for op in operations]
        results,errors=[],[]
        # Drain every submitted call before propagating a failure. A successful
        # concurrent call stays cached even if another call is unresolved.
        for future in futures:
            try:results.append(future.result());errors.append(None)
            except BaseException as exc:results.append(None);errors.append(exc)
        for error in errors:
            if error is not None:raise error
        return results

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
