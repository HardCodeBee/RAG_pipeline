"""Namespaced storage, budget, transport and immutable corpus retrieval.

No answers, qrels, method prompts or algorithm controllers belong here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PLAN = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1"
SPARSE_ARTIFACTS = {}  # Immutable artifact verification only; no queries/results.


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self, directory: Path, identity: dict):
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(OUTPUT.resolve()):
            raise ValueError("Method outputs must remain inside formal evaluation output root")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "checkpoint.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS data(kind TEXT,key TEXT,value TEXT,PRIMARY KEY(kind,key));
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,stage TEXT,payload TEXT,status TEXT,result TEXT);
          CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT,started TEXT,status TEXT,result TEXT);
          CREATE INDEX IF NOT EXISTS data_qid ON data(kind,json_extract(value,'$.qid'));
        """)
        self.db.execute("UPDATE requests SET status='unknown' WHERE status='in_flight'")
        self.db.execute("UPDATE attempts SET status='unknown' WHERE status='in_flight'")
        self.db.commit()
        self.put("meta", "identity", identity)
        self.identity = identity
        self.namespace = str(self.directory.relative_to(OUTPUT.resolve()))

    def get(self, kind, key):
        row = self.db.execute("SELECT value FROM data WHERE kind=? AND key=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        old = self.get(kind, key)
        if old is not None and canonical(old) != canonical(value):
            raise ValueError(f"Immutable method state differs: {kind}/{key}")
        self.db.execute("INSERT OR IGNORE INTO data VALUES(?,?,?)", (kind, key, canonical(value)))
        self.db.commit()

    def event(self, event, **fields):
        with (self.directory / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"utc": utc(), "event": event, **fields}) + "\n")

    def export(self, full=False):
        write_json(self.directory / "status.json", {"utc": utc(), "namespace": self.namespace,
                   "requests": dict(self.db.execute("SELECT status,count(*) FROM requests GROUP BY status")),
                   "predictions": self.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]})
        if not full:
            return
        for kind, name in [("prediction", "predictions.jsonl"), ("retrieval", "retrieval.jsonl")]:
            values = [json.loads(row[0]) for row in self.db.execute("SELECT value FROM data WHERE kind=? ORDER BY rowid", (kind,))]
            (self.directory / name).write_text("".join(canonical(v) + "\n" for v in values), encoding="utf-8")
        requests = [dict(zip(["request_id", "stage", "payload", "status", "result"], row)) for row in self.db.execute("SELECT * FROM requests ORDER BY rowid")]
        for request in requests:
            request["payload"] = json.loads(request["payload"])
            request["result"] = json.loads(request["result"]) if request["result"] else None
        (self.directory / "requests.jsonl").write_text("".join(canonical(v) + "\n" for v in requests), encoding="utf-8")
        write_json(self.directory / "status.json", {"utc": utc(), "namespace": self.namespace,
                   "requests": dict(self.db.execute("SELECT status,count(*) FROM requests GROUP BY status")),
                   "predictions": self.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]})


class Budget:
    """One global money ledger, containing no prompts or method outputs."""
    def __init__(self, path: Path, cap: float = 200):
        self.path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,namespace TEXT,reserved REAL,actual REAL,status TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS scheduler(key TEXT PRIMARY KEY,value REAL)")
        self.db.commit()
        self.cap = cap

    def reserve(self, request_id, namespace, amount):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0) FROM charges").fetchone()[0]
            previous = self.db.execute("SELECT id FROM charges WHERE id=?", (request_id,)).fetchone()
            if previous:
                raise RuntimeError("An existing submission liability cannot be silently replayed")
            if total + amount > self.cap:
                raise RuntimeError("Authorized API spending cap reached")
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def settle(self, request_id, cost, status):
        self.db.execute("UPDATE charges SET actual=?,status=? WHERE id=?", (cost, status, request_id))
        self.db.commit()

    def wait_for_start(self):
        """Atomic cross-process start slots, without shared prompts or answers."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            values = dict(self.db.execute("SELECT key,value FROM scheduler"))
            count = int(values.get("attempt_count", 0))
            now = time.time()
            start = max(now, values.get("next_start", now)) + (30 if count and count % 64 == 0 else 0)
            if start - now > 300:
                raise RuntimeError("Rate scheduler clock/backlog requires audit")
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('next_start',?)", (start + 3,))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('attempt_count',?)", (count + 1,))
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise
        if start > now:
            remaining = start - now
            while remaining > 0:
                time.sleep(min(60, remaining)); remaining -= min(60, remaining)

    def summary(self):
        row = self.db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges").fetchone()
        return {"cap_cny": self.cap, "returned_usage_list_price_cny": row[0], "unreconciled_reserved_cny": row[1], "submission_attempts": row[2], "actual_bill_verified": False}


class Transport:
    def __init__(self, store: Store, client, budget: Budget, model="qwen-plus-2025-12-01"):
        self.store, self.client, self.budget, self.model = store, client, budget, model
        self.last_start = 0.0
        self.new_requests = 0
        self.qid = None

    def ask(self, stage: str, prompt: str, max_tokens: int):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Empty prompt")
        # Conservative byte upper bound, not a claim of exact tokenizer count.
        byte_upper = len(prompt.encode("utf-8")) + 256
        if byte_upper > 100000:
            raise ValueError("Context byte safety limit exceeded; no silent truncation")
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": max_tokens, "extra_body": {"enable_thinking": False}}
        request_id = digest([self.store.namespace, payload])
        if self.qid is not None:
            shown = {}
            for (raw_packet,) in self.store.db.execute("SELECT value FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid') AND json_extract(value,'$.qid')=?", (self.qid,)):
                packet = json.loads(raw_packet)
                for doc in packet.get("docs", []):
                    if doc.get("text") and doc["text"] in prompt:
                        shown[doc["doc_id"]] = doc
            self.store.put("evidence", digest([self.qid, request_id, stage]),
                           {"qid": self.qid, "stage": stage, "request_id": request_id,
                            "doc_ids": sorted(shown), "source_match": "exact raw corpus text present in this call; generated answers are not source evidence"})
        db = self.store.db
        existing = db.execute("SELECT status,result FROM requests WHERE id=?", (request_id,)).fetchone()
        if existing:
            if existing[0] in {"unknown", "in_flight"}:
                raise RuntimeError("Unknown submission requires reconciliation")
            self.store.event("logical_api_call", qid=self.qid, stage=stage, request_id=request_id, cache_hit=True, max_tokens=max_tokens)
            return request_id, json.loads(existing[1])
        self.store.event("logical_api_call", qid=self.qid, stage=stage, request_id=request_id, cache_hit=False, max_tokens=max_tokens)
        network_retries = 0
        for attempt in range(4):
            self.budget.wait_for_start()
            liability_id = digest([request_id, attempt])
            self.budget.reserve(liability_id, self.store.namespace, byte_upper * .8e-6 + max_tokens * 2e-6)
            db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,'in_flight',NULL)", (request_id, stage, canonical(payload)))
            db.execute("UPDATE requests SET status='in_flight' WHERE id=?", (request_id,))
            cursor = db.execute("INSERT INTO attempts(request_id,started,status) VALUES(?,?,'in_flight')", (request_id, utc()))
            attempt_id = cursor.lastrowid
            db.commit()
            self.last_start = time.monotonic()
            started = time.perf_counter()
            retryable_network_failure = False
            try:
                response = self.client.chat.completions.create(**payload)
                raw = response.model_dump(mode="json")
                choice = response.choices[0]
                usage = raw.get("usage") or {}
                result = {"status": "success", "answer": choice.message.content or "", "finish_reason": choice.finish_reason,
                          "refusal": getattr(choice.message, "refusal", None), "returned_model": response.model,
                          "usage": usage, "raw_response": raw, "latency_seconds": time.perf_counter() - started}
                if usage:
                    self.budget.settle(liability_id, usage.get("prompt_tokens", 0) * .8e-6 + usage.get("completion_tokens", 0) * 2e-6, "returned_usage")
                if response.model != self.model:
                    result["status"] = "model_identity_mismatch"
                if usage.get("prompt_tokens", 0) > 32768:
                    result["status"] = "input_token_limit_violation"
            except Exception as exc:
                import httpx
                from openai import APIConnectionError, APITimeoutError
                retryable_network_failure = isinstance(exc, (
                    APIConnectionError, APITimeoutError, httpx.ConnectError,
                    httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout,
                    httpx.WriteError, httpx.WriteTimeout, httpx.RemoteProtocolError,
                ))
                code = getattr(exc, "status_code", None)
                body = getattr(exc, "body", {})
                body = body.get("error", body) if isinstance(body, dict) else {}
                error_code = body.get("code") if isinstance(body, dict) else None
                error_code = error_code if isinstance(error_code, str) and re.fullmatch(r"[\w.-]{1,100}", error_code) else None
                causes, cause = [], exc
                while cause is not None and len(causes) < 6:
                    causes.append({"type": type(cause).__name__, "errno": getattr(cause, "errno", None)})
                    cause = cause.__cause__ or cause.__context__
                result = {"status": "failed" if code else "unknown", "http_status": code,
                          "failure_class": type(exc).__name__, "error_code": error_code,
                          "failure_causes": causes, "latency_seconds": time.perf_counter() - started}
                if error_code in {"data_inspection_failed", "DataInspectionFailed"} and code in {None, 400, 422}:
                    result["status"] = "provider_content_rejected"
                    result["retryable"] = False
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                try:
                    result["retry_after_seconds"] = max(0, float(retry_after)) if retry_after else None
                except (TypeError, ValueError):
                    result["retry_after_seconds"] = None
                if code in {400, 401, 403, 404, 422, 429} and result['status'] != 'provider_content_rejected':
                    self.budget.settle(liability_id, 0.0, "http_failed_not_returned_usage")
            db.execute("UPDATE attempts SET status=?,result=? WHERE id=?", (result["status"], canonical(result), attempt_id))
            db.execute("UPDATE requests SET status=?,result=? WHERE id=?", (result["status"], canonical(result), request_id))
            db.commit()
            self.store.event("api_attempt", stage=stage, request_id=request_id, attempt=attempt, status=result["status"])
            self.new_requests += 1
            self.store.export()
            write_json(self.budget.path.parent / "budget_status.json", self.budget.summary())
            if (result["status"] == "unknown" and retryable_network_failure
                    and network_retries < 2 and attempt < 3):
                delay = (5, 15)[network_retries]
                network_retries += 1
                self.store.event("unknown_network_retry_scheduled", qid=self.qid,
                                 stage=stage, request_id=request_id, attempt=attempt,
                                 network_retry=network_retries, delay_seconds=delay,
                                 original_liability_id=liability_id,
                                 original_reserve_kept=True,
                                 policy="new in-process requests only; first complete response")
                time.sleep(delay)
                continue
            rate_codes = {"Throttling.RateQuota", "Throttling.RequestQuota", "rate_limit_exceeded", "Throttling"}
            if result.get("http_status") == 429 and result.get("error_code") in rate_codes and attempt < 3:
                delay = max([30, 60, 60][attempt], result.get("retry_after_seconds") or 0)
                if delay > 300:
                    raise RuntimeError("Long provider cooldown; stop instead of busy retrying")
                remaining = delay
                while remaining > 0:
                    time.sleep(min(60, remaining))
                    remaining -= min(60, remaining)
                continue
            if result["status"] in {"unknown", "model_identity_mismatch", "input_token_limit_violation"} or result.get("http_status") in {401, 403, 404, 429}:
                raise RuntimeError("Transport stopped; inspect sanitized method status")
            return request_id, result
        raise AssertionError("Unreachable request state")


class Retrieval:
    def __init__(self, store: Store, resources=None, device="cuda"):
        self.store = store
        self.resources = resources or json.loads((ROOT / "experiments/query_decomposition/query_form_ablation_v1/resources.json").read_text(encoding="utf-8"))
        self.device = device
        self.index = self.encoder = self.chunks = self.sparse = None

    def load(self):
        if self.index is not None:
            return
        import faiss
        import torch
        from src.embedders.text_embedder import TextEmbedder
        from src.indexes.streaming_flat_index import StreamingFlatIPIndex
        from src.retrievers.chunk_store import JsonlOffsetChunkStore
        spec = self.resources["dense"]["bge"]
        for key, expected in [("manifest", self.resources["corpus"]["manifest_sha256"])]:
            if hashlib.sha256(Path(self.resources["corpus"][key]).read_bytes()).hexdigest() != expected:
                raise ValueError("Pinned corpus manifest changed")
        if hashlib.sha256((Path(spec["build"]) / "manifest.json").read_bytes()).hexdigest() != spec["manifest_sha256"]:
            raise ValueError("Pinned dense manifest changed")
        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        self.store.event("retriever_loading")
        self.index = StreamingFlatIPIndex(Path(spec["build"]) / "embeddings", corpus_chunk_size=50000)
        if self.index.count != 5233329 or self.index.dimension != 384:
            raise ValueError("Wrong corpus index")
        self.encoder = TextEmbedder(backend="sentence_transformers", model_name=spec["query_model"], revision=spec["query_revision"], normalize=True, batch_size=16, query_prefix="", document_prefix="", max_sequence_length=512, local_files_only=True, device=self.device)
        if self.encoder.embedding_space("inner_product").to_dict() != spec["space"]:
            raise ValueError("Wrong query embedding space")
        corpus = self.resources["corpus"]
        self.chunks = JsonlOffsetChunkStore(corpus["chunks"], corpus["offsets"], expected_rows=5233329)
        self.store.event("retriever_loaded")

    def retrieve_many(self, cases: list[dict], k: int):
        import numpy as np
        keys = [digest([case["qid"], case["query"], k, self.resources]) for case in cases]
        pending = [(case, key) for case, key in zip(cases, keys) if self.store.get("retrieval", key) is None]
        if pending:
            self.load()
            queries = [case["query"] for case, _ in pending]
            tokenizer = self.encoder._model[0].tokenizer
            counts = [len(tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"]) for query in queries]
            start = time.perf_counter()
            vectors = np.ascontiguousarray(self.encoder.encode_queries(queries), dtype=np.float32)
            scores, ids = self.index.search_many(vectors, k + 1, query_batch_size=len(pending))
            elapsed = time.perf_counter() - start
            for (case, key), values, indices, count in zip(pending, scores.tolist(), ids.tolist(), counts):
                docs = []
                for score, chunk in zip(values, self.chunks.get_many(indices)):
                    if chunk.doc_id == case["qid"]:
                        continue
                    rank = len(docs) + 1
                    docs.append({"rank": rank, "score": score, "doc_id": chunk.doc_id, "vector_id": chunk.vector_id,
                                 "source": chunk.source, "text": chunk.text,
                                 "block": f"[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}"})
                    if len(docs) == k:
                        break
                if len(docs) != k:
                    raise ValueError("Incomplete retrieval")
                packet = {"qid": case["qid"], "query": case["query"], "k": k, "docs": docs, "query_tokens": count,
                          "encoder_truncated": count > 512, "batch_queries": len(pending), "batch_seconds": elapsed}
                self.store.put("retrieval", key, packet)
                self.store.event("retrieval", qid=case["qid"], query=case["query"], k=k, doc_ids=[d["doc_id"] for d in docs])
        self.store.export()
        return [self.store.get("retrieval", key) for key in keys]

    def retrieve(self, qid: str, query: str, k: int):
        return self.retrieve_many([{"qid": qid, "query": query}], k)[0]

    def retrieve_sparse(self, qid, query, k):
        key = digest(["bm25", qid, query, k, self.resources])
        packet = self.store.get("sparse_retrieval", key)
        if packet is not None:
            return packet
        self.load()
        if self.sparse is None:
            from src.retrievers.sqlite_bm25 import SQLiteBM25Retriever, validate_sqlite_bm25_index
            spec = self.resources["bm25"]
            if hashlib.sha256((Path(spec["directory"]) / "manifest.json").read_bytes()).hexdigest() != spec["manifest_sha256"]:
                raise ValueError("Pinned sparse manifest changed")
            database = Path(spec["directory"]) / "index.sqlite3"
            stamp = (database.stat().st_size, database.stat().st_mtime_ns)
            identity = (spec["directory"], spec["manifest_sha256"])
            if identity not in SPARSE_ARTIFACTS:
                self.store.event("bm25_full_validation_started")
                started = time.perf_counter()
                artifact = validate_sqlite_bm25_index(spec["directory"])
                SPARSE_ARTIFACTS[identity] = (artifact, stamp)
                self.store.event("bm25_full_validation_completed", seconds=time.perf_counter() - started)
            artifact, original_stamp = SPARSE_ARTIFACTS[identity]
            if stamp != original_stamp:
                raise ValueError("Immutable sparse database changed during execution")
            self.sparse = SQLiteBM25Retriever(self.chunks, artifact, sparse_index_id=spec["sparse_index_id"])
        trace = self.sparse.retrieve_trace(query, k + 1)
        docs = []
        for hit in trace.results:
            chunk = hit.chunk
            if chunk.doc_id == qid:
                continue
            rank = len(docs) + 1
            docs.append({"rank": rank, "score": hit.score, "doc_id": chunk.doc_id,
                         "vector_id": chunk.vector_id, "source": chunk.source, "text": chunk.text,
                         "block": f"[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}"})
            if len(docs) == k:
                break
        packet = {"qid": qid, "query": query, "k": k, "docs": docs, "timings_ms": trace.timings_ms}
        self.store.put("sparse_retrieval", key, packet)
        return packet
