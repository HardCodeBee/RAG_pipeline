"""Carry forward one completed immutable-index validation across the handoff."""
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common import runtime


def seed(resources):
    receipt_file=runtime.OUTPUT / 'parallel_execution_preparation/SPARSE_VALIDATION_REUSE.json'
    if not receipt_file.exists():return False
    receipt=json.loads(receipt_file.read_text(encoding='utf8'))
    spec=resources['bm25']
    directory=Path(spec['directory']).resolve()
    database=directory / 'index.sqlite3'
    manifest_file=directory / 'manifest.json'
    stamp=database.stat()
    if (str(database)!=receipt['database'] or stamp.st_size!=receipt['size_bytes'] or stamp.st_mtime_ns!=receipt['mtime_ns']
            or hashlib.sha256(manifest_file.read_bytes()).hexdigest()!=receipt['manifest_sha256']
            or receipt['manifest_sha256']!=spec['manifest_sha256']):
        return False
    events=[json.loads(s) for s in Path(receipt['prior_trace']).read_text(encoding='utf8').splitlines()]
    if receipt['prior_full_validation_event'] not in events:
        return False
    from src.retrievers.sqlite_bm25 import SQLiteBM25Artifact, _open_read_only, _metadata_values
    manifest=json.loads(manifest_file.read_text(encoding='utf8'))
    connection=_open_read_only(database)
    try:
        metadata=_metadata_values(connection)
    finally:
        connection.close()
    if metadata!=receipt['metadata'] or metadata['identity_sha256']!=manifest['identity_sha256']:
        return False
    artifact=SQLiteBM25Artifact(directory=directory,manifest=manifest,database=database)
    runtime.SPARSE_ARTIFACTS[(spec['directory'],spec['manifest_sha256'])]=(artifact,(stamp.st_size,stamp.st_mtime_ns))
    return True
