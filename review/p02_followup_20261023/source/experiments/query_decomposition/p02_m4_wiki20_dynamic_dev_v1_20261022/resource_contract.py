"""Path-independent resource identity shared by local orchestrator and worker."""
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def contract(manifest, manifest_sha256):
    ds, corpus, bge = manifest["dataset"], manifest["corpus"], manifest["dense"]["bge"]
    files = {f["path"]: f for f in manifest["files"]}
    if ds not in ("hotpotqa", "2wiki") or manifest.get("schema_version") != 1:
        raise ValueError("Unexpected portable resource manifest")
    offsets_sha = files[corpus["offsets"]]["sha256"]
    # Full manifest SHA binds ALL original file SHAs (including embedding parts),
    # so neither a Linux nor a Windows absolute resource path enters this key.
    return dict(schema="p02_full8_resource_v1", dataset=ds,
        portable_manifest_sha256=manifest_sha256, backend="bge",
        corpus_chunks_sha256=corpus["chunks_sha256"], corpus_offsets_sha256=offsets_sha,
        documents=corpus["documents"], query_model=bge["query_model"],
        query_revision=bge["query_revision"], embedding_space=bge["space"],
        query_batch_size=1, search_k_before_filter=11, k=10,
        threads=4, corpus_chunk_size=50000, filter_original_qid=True)


def identity(manifest, manifest_sha256):
    return hashlib.sha256(canonical(contract(manifest, manifest_sha256)).encode("utf-8")).hexdigest()
