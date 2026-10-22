"""SQLite records, full-request caching and a single-writer run lock."""

# checkpoint.sqlite3 保存阶段结果和 API 请求记录，是续跑与结果导出的依据。
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import uuid

from . import METHOD


def canonical(value):
    # 按固定键序直接序列化完整内容，用于相等比较和缓存查询。
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def utc():
    return datetime.now(timezone.utc).isoformat()


class Cache:
    """One SQLite checkpoint per run; open it from one process at a time."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        # data 保存各阶段对象；requests 保存实际请求；request_stages 记录请求的各次用途。
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS data(
                kind TEXT, key TEXT, value TEXT, PRIMARY KEY(kind,key));
            CREATE TABLE IF NOT EXISTS requests(
                id TEXT PRIMARY KEY, stage TEXT, payload TEXT, status TEXT,
                started TEXT, completed TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS request_stages(
                request_id TEXT, stage TEXT, PRIMARY KEY(request_id,stage));
            CREATE UNIQUE INDEX IF NOT EXISTS request_payload ON requests(payload);
        """)
        # 上次中断的请求可能已在服务端完成，因此标记为 unknown，不默认当作没发送。
        self.db.execute("UPDATE requests SET status='unknown' WHERE status='in_flight'")
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.db.close()

    def get(self, kind, key):
        row = self.db.execute(
            "SELECT value FROM data WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        # 同一阶段、同一 key 的记录不可覆盖；更改实验输入或设置应使用新的输出目录。
        encoded = canonical(value)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO data VALUES(?,?,?)", (kind, key, encoded)
            )
            saved = self.db.execute(
                "SELECT value FROM data WHERE kind=? AND key=?", (kind, key)
            ).fetchone()[0]
            if saved != encoded:
                raise ValueError(f"Immutable checkpoint changed: {kind}/{key}")

    def all(self, kind):
        return {
            key: json.loads(value)
            for key, value in self.db.execute(
                "SELECT key,value FROM data WHERE kind=? ORDER BY rowid", (kind,)
            )
        }

    def requests(self):
        """Return decoded receipts, including all logical uses of each payload."""
        stages = {}
        for request_id, stage in self.db.execute(
            "SELECT request_id,stage FROM request_stages ORDER BY rowid"
        ):
            stages.setdefault(request_id, []).append(stage)
        return [
            {
                "request_id": rid,
                "stage": stage,
                "stages": stages.get(rid, [stage]),
                "payload": json.loads(payload),
                "status": status,
                "started": started,
                "completed": completed,
                "result": json.loads(result) if result else None,
            }
            for rid, stage, payload, status, started, completed, result in self.db.execute(
                "SELECT * FROM requests ORDER BY rowid"
            )
        ]

    def begin_request(self, stage, payload, *, before_start=None):
        # 完整请求内容相同就复用已有记录；阶段标签不同也可以引用同一次实际调用。
        encoded = canonical(payload)
        saved = self.db.execute(
            "SELECT id,status,result FROM requests WHERE payload=?", (encoded,)
        ).fetchone()
        request_id = saved[0] if saved else uuid.uuid4().hex
        if saved is None and before_start is not None:
            before_start(request_id)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO request_stages VALUES(?,?)", (request_id, stage)
            )
            if saved is None:
                self.db.execute(
                    "INSERT INTO requests VALUES(?,?,?,'in_flight',?,NULL,NULL)",
                    (request_id, stage, encoded, utc()),
                )
        if saved is None:
            # 返回 None 表示本次需要真正请求 API；已有失败或未知状态也不会自动重发。
            return request_id, None
        result = json.loads(saved[2]) if saved[2] else {}
        return request_id, {**result, "status": saved[1]}

    def finish_request(self, request_id, result):
        with self.db:
            self.db.execute(
                "UPDATE requests SET status=?,completed=?,result=? WHERE id=?",
                (result["status"], utc(), canonical(result), request_id),
            )


@contextmanager
def run_lock(output):
    """A single writer owns a run, including its uncommitted API requests."""
    # Windows/Linux 分别使用系统文件锁，同一输出目录只允许一个进程写入。
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            import msvcrt
        except ImportError:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        else:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "This p02 run is already open in another process"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
