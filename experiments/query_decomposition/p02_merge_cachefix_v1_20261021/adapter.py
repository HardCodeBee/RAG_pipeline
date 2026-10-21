"""Exact-key retrieval cache first; frozen full-index retrieval only on misses.

Seed priority is canonical p02 packets, caller supplied packets, all historical
union20 raw packets, then matching-resource historical retrieval database rows.
No case reader/merged packet is treated as a raw retrieval result.
"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import retrieval_adapter as frozen


HERE = Path(__file__).resolve().parent
OLD_HERE = HERE.parent / 'p02_flow_simplify_v1_20261020'
OLD_OUT = OUTPUT / OLD_HERE.name


class RetrievalMiss(RuntimeError):
    def __init__(self, qid, query, k, resource_sha256):
        self.qid, self.query, self.k, self.resource_sha256 = qid, query, k, resource_sha256
        super().__init__('Exact retrieval cache miss; offline mode forbids full-index search')


def canonical_sources(dataset):
    manifest = frozen._read(OLD_HERE / 'BASELINE_SOURCES.frozen.json')
    selected = [row for row in manifest if row['id'].startswith(dataset + ':')]
    if len(selected) != 100:
        raise ValueError('Expected exactly100 canonical p02 sources per dataset')
    paths = []
    for row in selected:
        path = ROOT / row['path']
        if frozen._sha(path) != row['sha256']:
            raise ValueError('Canonical p02 source changed: ' + str(path))
        paths.append(path)
    return paths


class Adapter(frozen.RetrievalAdapter):
    def __init__(self, owner, resources, *, seed_cases=(), source_resources=None,
                 seed_databases=(), offline_only=False, **kwargs):
        dataset = resources['corpus']['dataset']
        expected = frozen._read(OLD_HERE / 'PROTOCOL.frozen.json')['resources'][dataset]
        if frozen.resource_identity(resources) != frozen.resource_identity(expected):
            raise ValueError('Cachefix requires the exact frozen p02 retrieval resource object')
        self.offline_only = offline_only
        canonical = canonical_sources(dataset)
        super().__init__(owner, resources, seed_cases=canonical, source_resources=expected,
                         **kwargs)
        if seed_cases:
            self.seed_cases(seed_cases, source_resources=source_resources)
        historical_union = sorted(OLD_OUT.glob(f'rounds/r*/cases/{dataset}/union20/*.json'))
        if historical_union:
            self.seed_cases(historical_union, source_resources=expected)
        databases = [OLD_OUT / f'retrieval/{dataset}/checkpoint.sqlite3', *map(Path, seed_databases)]
        databases = list(dict.fromkeys(path.resolve() for path in databases if path.is_file()))
        if databases:
            self.seed_databases(databases)
        self.owner.event('cachefix_retrieval_seed_priority', dataset=dataset,
                         canonical_sources=len(canonical), union_case_sources=len(historical_union),
                         database_sources=[str(path.relative_to(ROOT)) for path in databases],
                         resource_sha256=self.resource_sha256, offline_only=offline_only,
                         existing_results_overwritten=False)

    def _cached(self, key):
        path = self.owner.directory / 'checkpoint.sqlite3'
        if path.is_file():
            # Owner Store.db is created on the controller thread. Workers use
            # independent read-only connections instead of crossing threads.
            with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute("SELECT value FROM data WHERE kind='retrieval' AND key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
        return self.owner.get('retrieval', key)  # Pure in-memory offline owner.

    def get(self, qid, query, k=10):
        if self._closed:
            raise RuntimeError('Retrieval adapter is closed')
        if k != 10 or isinstance(k, bool):
            raise ValueError('Frozen experiment requires k=10 per route')
        if not isinstance(qid, str) or not qid or not isinstance(query, str) or not query.strip():
            raise ValueError('qid and exact query must be nonempty strings')
        started = time.perf_counter()
        key = digest([qid, query, k, self.resources])
        packet = self._cached(key)
        if packet is not None:
            frozen._validate_packet(packet, qid, query, k)
            return deepcopy(packet), True, time.perf_counter() - started
        if self.offline_only:
            raise RetrievalMiss(qid, query, k, self.resource_sha256)
        # No new retriever, approximation or batch arithmetic: exactly the old
        # resource-verified CPU streaming full-index backend on genuine misses.
        return super().get(qid, query, k)
