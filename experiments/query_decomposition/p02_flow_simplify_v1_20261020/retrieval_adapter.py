"""Exact-resource retrieval reuse plus real global BGE retrieval on cache misses.

This adapter never reads references, qrels or answers.  A changed downstream
query is a different cache key and is searched against the full frozen corpus.
The existing BatchSearch retains single-query encoder/FAISS arithmetic while
sharing read-only corpus scans among simultaneous requests.  Only the small
query encoder is kept on its configured device; corpus blocks stay on CPU.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import (
    canonical, digest,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_identity(resources):
    """Use the complete frozen resource object, matching historical cache keys."""
    return digest(resources)


def verify_resource_manifests(resources):
    corpus = resources["corpus"]
    dense = resources["dense"]["bge"]
    if _sha(corpus["manifest"]) != corpus["manifest_sha256"]:
        raise ValueError("Frozen corpus manifest changed")
    manifest = Path(dense["build"]) / (
        "manifest.json" if corpus["dataset"] == "hotpotqa" else "DENSE_FROZEN.json"
    )
    if _sha(manifest) != dense["manifest_sha256"]:
        raise ValueError("Frozen dense manifest changed")
    return {"resource_sha256": resource_identity(resources),
            "corpus_manifest_sha256": corpus["manifest_sha256"],
            "dense_manifest_sha256": dense["manifest_sha256"]}


def _case_packets(case):
    """Only actual route packets, never merged/relabelled reader packets."""
    if isinstance(case.get("original_retrieval"), dict):
        yield case["original_retrieval"]
    for node in case.get("nodes", []):
        for name in ("original_retrieval", "rewritten_retrieval"):
            packet = node.get(name)
            if isinstance(packet, dict):
                yield packet


def _validate_packet(packet, qid=None, query=None, k=None):
    for name, wanted in (("qid", qid), ("query", query), ("k", k)):
        if name not in packet or (wanted is not None and packet[name] != wanted):
            raise ValueError(f"Retrieval packet identity differs: {name}")
    if not isinstance(packet["query"], str) or not packet["query"].strip():
        raise ValueError("Empty retrieval query")
    docs = packet.get("docs", [])
    if not isinstance(packet["k"], int) or packet["k"] <= 0 or len(docs) != packet["k"]:
        raise ValueError("Incomplete retrieval packet")
    for rank, doc in enumerate(docs, 1):
        if doc.get("rank") != rank or not all(
            field in doc for field in ("doc_id", "score", "source", "text", "block")
        ):
            raise ValueError("Invalid ranked corpus document")


class RetrievalAdapter:
    """One dataset/resource, a new output Store and an optional exact live service.

    owner must be the NEW campaign retrieval Store.  Old case files and SQLite
    databases are read-only seed sources.  Case seeding requires the exact
    historical resource object; database seeding accepts only rows whose stored
    key matches digest([qid, query, k, self.resources]).

    get(qid, query, k=10) returns (packet, cache_hit, seconds).
    retrieve(qid, query, k=10) returns the packet only.

    Source metadata never changes an existing successful retrieval result.
    Incoming duplicate identities keep the first source; differences are logged
    rather than selecting by document coverage or downstream answer quality.
    """

    def __init__(self, owner, resources, *, seed_cases=(), source_resources=None,
                 seed_databases=(), device="cuda", max_batch=20,
                 search_service=None, verify_manifests=True):
        self.owner = owner
        self.resources = deepcopy(resources)
        self.resource_sha256 = resource_identity(self.resources)
        self.device = device
        self.max_batch = max_batch
        self.service = search_service
        self._owns_service = search_service is None
        self.retriever = None
        self._closed = False
        self._start_lock = Lock()
        self.seed_summary = {"inserted": 0, "existing": 0,
                             "foreign_resource_rows_ignored": 0,
                             "duplicate_document_differences": 0}
        if verify_manifests:
            identity = verify_resource_manifests(self.resources)
            self.owner.event("retrieval_resources_verified", **identity)
        if seed_cases:
            self.seed_cases(seed_cases, source_resources=source_resources)
        if seed_databases:
            self.seed_databases(seed_databases)

    def _seed(self, packet, source):
        _validate_packet(packet)
        key = digest([packet["qid"], packet["query"], packet["k"], self.resources])
        old = self.owner.get("retrieval", key)
        if old is None:
            self.owner.put("retrieval", key, packet)
            self.owner.put("retrieval_seed_source", key, {
                "source": source, "resource_sha256": self.resource_sha256,
                "qid": packet["qid"], "query": packet["query"], "k": packet["k"],
            })
            self.seed_summary["inserted"] += 1
        else:
            _validate_packet(old, packet["qid"], packet["query"], packet["k"])
            self.seed_summary["existing"] += 1
            if canonical(old["docs"]) != canonical(packet["docs"]):
                self.seed_summary["duplicate_document_differences"] += 1
                self.owner.event("retrieval_seed_duplicate_differs", key=key,
                                 qid=packet["qid"], query=packet["query"],
                                 source=source, action="keep first exact-key result")

    def seed_cases(self, paths, *, source_resources):
        if self.service is not None and self._owns_service:
            raise RuntimeError("Seed before starting the live search worker")
        if source_resources is None or canonical(source_resources) != canonical(self.resources):
            raise ValueError("Case seeding requires exactly matching frozen resources")
        dataset = self.resources["corpus"]["dataset"]
        for source in paths:
            if isinstance(source, dict):
                case = source
                label = "caller supplied recorded case"
            else:
                path = Path(source).resolve()
                case = _read(path)
                label = str(path)
            if case.get("dataset") != dataset:
                raise ValueError("Case dataset differs from the frozen corpus")
            for packet in _case_packets(case):
                if packet["qid"] != case["qid"]:
                    raise ValueError("Case/packet qid mismatch")
                self._seed(packet, label)
        self.owner.event("retrieval_cases_seeded", dataset=dataset,
                         resource_sha256=self.resource_sha256, **self.seed_summary)
        return deepcopy(self.seed_summary)

    def seed_databases(self, paths):
        if self.service is not None and self._owns_service:
            raise RuntimeError("Seed before starting the live search worker")
        for source in paths:
            path = Path(source).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path == (self.owner.directory / "checkpoint.sqlite3").resolve():
                continue
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                for key, raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval' ORDER BY rowid"):
                    packet = json.loads(raw)
                    if not all(name in packet for name in ("qid", "query", "k")):
                        continue
                    expected = digest([packet["qid"], packet["query"], packet["k"], self.resources])
                    if key != expected:
                        self.seed_summary["foreign_resource_rows_ignored"] += 1
                        continue
                    self._seed(packet, str(path))
        self.owner.event("retrieval_databases_seeded",
                         resource_sha256=self.resource_sha256, **self.seed_summary)
        return deepcopy(self.seed_summary)

    def _start(self):
        if self._closed:
            raise RuntimeError("Retrieval adapter is closed")
        with self._start_lock:
            if self.service is not None:
                return
            from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
            from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
            from experiments.query_decomposition.baselines_same300_v1_20261020.batch_search import BatchSearch
            install()
            self.retriever = Retrieval(self.owner, resources=self.resources, device=self.device)
            self.service = BatchSearch(self.owner, self.retriever, max_batch=self.max_batch)
            self.owner.event("exact_global_retrieval_started",
                             dataset=self.resources["corpus"]["dataset"],
                             resource_sha256=self.resource_sha256,
                             backend="original streaming CPU 50000-row FAISS blocks",
                             query_encoder_device=self.device,
                             query_encoding_batch_size=1, faiss_query_batch_size=1,
                             corpus_to_gpu=False, no_approximate_index=True)

    def get(self, qid, query, k=10):
        if self._closed:
            raise RuntimeError("Retrieval adapter is closed")
        if k != 10 or isinstance(k, bool):
            raise ValueError("Frozen experiment requires k=10 per route")
        if not isinstance(qid, str) or not qid or not isinstance(query, str) or not query.strip():
            raise ValueError("qid and exact query must be nonempty strings")
        self._start()
        packet, hit, elapsed = self.service.get(qid, query, k)
        _validate_packet(packet, qid, query, k)
        return deepcopy(packet), hit, elapsed

    def retrieve(self, qid, query, k=10):
        return self.get(qid, query, k)[0]

    def close(self):
        if self._closed:
            return
        if self.service is not None and self._owns_service:
            self.service.close()
        if self.retriever is not None:
            if self.retriever.chunks is not None:
                self.retriever.chunks.close()
            if self.retriever.index is not None:
                self.retriever.index.close()
            self.retriever.encoder = None
        self._closed = True


# Compact alias for runner code.
Adapter = RetrievalAdapter

