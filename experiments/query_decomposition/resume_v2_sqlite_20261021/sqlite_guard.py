"""Drain read cursors and serialize short local transactions, never API calls.

An active SELECT cursor in WAL mode can keep an old read snapshot. Writing from
that connection after another writer commits may then raise SQLITE_BUSY_SNAPSHOT
without waiting for busy_timeout. Materializing results removes that overlap.
The per-file lock additionally serializes transactions within this process.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import sqlite3
import threading
import time

LOCKS = {}
LOCKS_MUTEX = threading.Lock()
COUNTERS = {'materialized_reads': 0, 'local_busy_retries': 0}
INSTALLED = False


def file_lock(path):
    key = str(Path(path).resolve()).casefold()
    with LOCKS_MUTEX:
        return LOCKS.setdefault(key, threading.RLock())


class Rows:
    def __init__(self, rows, description, lastrowid, rowcount):
        self.rows = iter(rows)
        self.description, self.lastrowid, self.rowcount = description, lastrowid, rowcount

    def fetchone(self):
        return next(self.rows, None)

    def fetchall(self):
        return list(self.rows)

    def fetchmany(self, size=1):
        result = []
        for _ in range(size):
            row = self.fetchone()
            if row is None:
                break
            result.append(row)
        return result

    def __iter__(self):
        return self.rows


class Connection:
    def __init__(self, original, path):
        self.original = original
        self.path = Path(path)
        self.lock = file_lock(path)
        self.held = False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _retry(self, action):
        for attempt in range(4):
            try:
                return action()
            except sqlite3.OperationalError as error:
                code = getattr(error, 'sqlite_errorcode', 0)
                busy = (code & 255) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                # Never roll back a partly completed transaction and replay only
                # its last statement. Never repeat an API call from this layer.
                if not busy or self.original.in_transaction or attempt == 3:
                    raise
                COUNTERS['local_busy_retries'] += 1
                time.sleep(.05 * 2 ** attempt)

    def execute(self, sql, parameters=()):
        self.lock.acquire()
        keep = False
        try:
            cursor = self._retry(lambda: self.original.execute(sql, parameters))
            description, lastrowid, rowcount = cursor.description, cursor.lastrowid, cursor.rowcount
            rows = cursor.fetchall() if description else []
            cursor.close()
            if description:
                COUNTERS['materialized_reads'] += 1
            if self.original.in_transaction and not self.held:
                self.held = keep = True
            return Rows(rows, description, lastrowid, rowcount)
        finally:
            if not keep:
                self.lock.release()

    def executescript(self, sql):
        with self.lock:
            value = self.original.executescript(sql)
            if self.held and not self.original.in_transaction:
                self.held = False
                self.lock.release()
            return value

    def _finish(self, operation):
        with self.lock:
            value = operation()
            if self.held and not self.original.in_transaction:
                self.held = False
                self.lock.release()
            return value

    def commit(self):
        return self._finish(self.original.commit)

    def rollback(self):
        return self._finish(self.original.rollback)

    def close(self):
        try:
            self.original.close()
        finally:
            if self.held:
                self.held = False
                self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        self.rollback() if kind else self.commit()


def install(budget_class):
    """Wrap only this evaluator's checkpoint and budget connections."""
    global INSTALLED
    if INSTALLED:
        return
    from experiments.query_decomposition.hotpot_formal_eval_v1.common import runtime
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import WorkerStore

    def wrap_init(cls, path_for):
        original = cls.__init__

        @wraps(original)
        def initialized(self, *args, **kwargs):
            path = path_for(args, kwargs)
            with file_lock(path):
                original(self, *args, **kwargs)
                if not isinstance(self.db, Connection):
                    self.db = Connection(self.db, path)
        cls.__init__ = initialized

    wrap_init(runtime.Store, lambda a, k: Path(a[0]) / 'checkpoint.sqlite3')
    wrap_init(WorkerStore, lambda a, k: a[0].directory / 'checkpoint.sqlite3')
    wrap_init(budget_class, lambda a, k: Path(a[0]))
    original_put = runtime.Store.put

    @wraps(original_put)
    def put(self, *args, **kwargs):
        with file_lock(self.directory / 'checkpoint.sqlite3'):
            return original_put(self, *args, **kwargs)
    runtime.Store.put = put
    INSTALLED = True
