"""Build a complete corpus-only BM25/Dense term moment lookup bank.

No questions, labels, encoders, retrieval rankings, or network operations.
Preparation freezes the algorithm and SQL vocabulary counts. A partial build
is deliberately not resumable: preserve it and inspect the failure manually.
"""
from __future__ import annotations

from array import array
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SPARSE = ROOT / "artifacts/_sparse_indexes/sqlite_bm25_04df51906ff9598b"
DENSE = ROOT / "artifacts/_encoded_corpora/encoded_corpus_df8e4c53d3e183b4"
OUT = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/static_pairing_bank_v1"
PROBE = OUT.parent / "lexical_dense_moments_feasibility_v1"
sys.path.insert(0, str(ROOT))
from src.retrievers.sqlite_bm25 import analyze_sqlite_bm25_text

SPEC = {
    "role": "complete_corpus_static_pairing_bank_not_router_effect_experiment",
    "source_build": "build_65e9037e4c1eed7b",
    "sparse_index": SPARSE.name, "encoded_corpus": DENSE.name,
    "documents": 5233329, "dimension": 384, "block_documents": 25000,
    "threads": 2, "vocabulary": "all readonly term_stats rows ordered by term; SQL counts frozen at preparation",
    "document_analysis": "analyze_sqlite_bm25_text(text) followed by Counter; sequential chunk vector_id required",
    "bm25": {"method": "lucene", "k1": 1.5, "b": 0.75, "analyzer": "english_regex_casefold_v1"},
    "impact": "idf * 2.5 * tf / (tf + 1.5 * (0.25 + 0.75 * dl / avgdl))",
    "scalar_moments": "FP64 sum_b, sum_b_squared, sum_b_times_dense_squared_norm",
    "multi_document_terms": "DF>=2; local term-by-document FP32 CSR times existing FP32 Dense; FP32 memmap accumulation; final centroid divided by FP64 mass",
    "singleton_terms": "DF=1; save original vector_id instead of a 384-dimensional copy",
    "corpus_mean": "FP64 vector sum and division by all documents including empty analyzed documents",
    "memory_controls": "25k-document blocks; local CSR; 4096-row memmap updates and normalization; dictionary and scalar arrays resident; approximately 4GB RAM design, not a hard OS limit",
    "source_checks": "existing manifest identities and chunk SHA descriptors; no new source hash scan; integrated exact observed DF and total analyzed length checks",
    "completion_crosscheck": {"reference": str(PROBE / "moments.npz"), "terms": 32,
                              "centroid_max_abs_tolerance": 1e-5, "scalar_relative_tolerance": 5e-12,
                              "source_postings_reread": False},
    "failure": "terminal progress record; no automatic retry or partial-build resume",
    "query_reads": 0, "label_reads": 0, "encoder_calls": 0, "retrieval_calls": 0,
    "model_fits": 0, "paid_calls": 0, "retained_query_reads": 0,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def progress(value):
    path = OUT / "progress.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def source_manifests():
    bm, dm = read(SPARSE / "manifest.json"), read(DENSE / "manifest.json")
    if bm["status"] != "complete" or dm["status"] != "complete":
        raise ValueError("Both existing source artifacts must be complete")
    if bm["identity"]["source_build_id"] != SPEC["source_build"]:
        raise ValueError("Sparse source build differs")
    if bm["identity"]["source_chunks"]["sha256"] != dm["artifacts"]["chunks"]["sha256"]:
        raise ValueError("Sparse and Dense manifests do not describe identical chunks")
    if bm["document_count"] != dm["corpus"]["num_documents"] or bm["document_count"] != SPEC["documents"]:
        raise ValueError("Source document counts differ")
    params = bm["identity"]["bm25"]
    if (params["method"], params["k1"], params["b"], bm["identity"]["analyzer"]["name"]) != ("lucene", 1.5, .75, "english_regex_casefold_v1"):
        raise ValueError("Sparse analysis/scoring configuration differs")
    space = dm["embedding"]["space"]
    if (space["dimension"], space["model_name"], space["revision"], space["max_sequence_length"], space["normalized"], space["document_prefix"], space["similarity"]) != (
            384, "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", 512, True, "", "inner_product"):
        raise ValueError("Existing Dense coordinate space differs")
    parts = dm["artifacts"]["embeddings"]["parts"]
    next_row = 0
    for part in parts:
        if part["start_row"] != next_row or not 0 < part["rows"] <= SPEC["block_documents"] or part["shape"] != [part["rows"], SPEC["dimension"]]:
            raise ValueError("Embedding parts must be contiguous blocks of at most 25k documents")
        next_row += part["rows"]
    if next_row != SPEC["documents"]:
        raise ValueError("Embedding parts do not cover the source document count")
    binding = {"sparse_index_id": bm["sparse_index_id"], "encoded_corpus_id": dm["encoded_corpus_id"],
               "chunks_sha256_from_manifests": dm["artifacts"]["chunks"]["sha256"],
               "sparse_database_sha256_from_manifest": bm["artifacts"]["database"]["sha256"],
               "embeddings_manifest_sha256_from_manifest": dm["artifacts"]["embeddings"]["sha256"],
               "document_count": bm["document_count"], "total_document_length": bm["total_document_length"],
               "average_document_length": bm["average_document_length"], "embedding_parts": len(parts)}
    return bm, dm, binding


def readonly_index():
    connection = sqlite3.connect((SPARSE / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def prepare():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Output directory already contains a preparation or build; inspect it, do not overwrite")
    _, _, binding = source_manifests()
    with readonly_index() as db:
        count, singles, multiples, minimum, maximum = db.execute(
            "SELECT COUNT(*), SUM(CASE WHEN df=1 THEN 1 ELSE 0 END), SUM(CASE WHEN df>=2 THEN 1 ELSE 0 END), MIN(df), MAX(df) FROM term_stats"
        ).fetchone()
    if not count > 0 or singles + multiples != count or minimum < 1 or maximum > SPEC["documents"]:
        raise ValueError("Invalid full-vocabulary SQL counts")
    probe_spec = read(PROBE / "protocol.json")["spec"]
    if any(probe_spec[name] != SPEC[name] for name in ("source_build", "sparse_index", "encoded_corpus")) or not (PROBE / "moments.npz").is_file():
        raise ValueError("The completed 32-term reference must use these same corpus sources")
    OUT.mkdir(parents=True, exist_ok=True)
    value = {"status": "frozen_before_full_vocabulary_and_corpus_build", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "spec": SPEC, "source_binding": binding,
             "vocabulary_counts_sql": {"terms": count, "singleton_terms": singles, "multi_document_terms": multiples,
                                       "minimum_df": minimum, "maximum_df": maximum},
             "centroid_data_bytes": multiples * SPEC["dimension"] * np.dtype(np.float32).itemsize}
    write_new(OUT / "protocol.json", value)
    print(json.dumps({"status": value["status"], "vocabulary_counts_sql": value["vocabulary_counts_sql"],
                      "centroid_data_bytes": value["centroid_data_bytes"]}), flush=True)


def build_vocabulary(counts):
    n = counts["terms"]
    df, idf = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.float64)
    multi_id = np.full(n, -1, dtype=np.int32)
    term_ids = {}
    destination = OUT / "vocabulary.sqlite3"
    with readonly_index() as source, sqlite3.connect(destination) as target:
        target.execute("CREATE TABLE terms (term TEXT PRIMARY KEY, term_id INTEGER NOT NULL, df INTEGER NOT NULL, idf REAL NOT NULL, multi_id INTEGER NOT NULL) WITHOUT ROWID")
        pending, multi_count = [], 0
        for term_id, (term, frequency, inverse_frequency) in enumerate(source.execute("SELECT term, df, idf FROM term_stats ORDER BY term")):
            if term_id >= n or not 0 < frequency <= SPEC["documents"] or not np.isfinite(inverse_frequency) or inverse_frequency <= 0:
                raise ValueError("Invalid or changed term_stats vocabulary")
            term_ids[term], df[term_id], idf[term_id] = term_id, frequency, inverse_frequency
            if frequency >= 2:
                multi_id[term_id] = multi_count
                multi_count += 1
            pending.append((term, term_id, frequency, inverse_frequency, int(multi_id[term_id])))
            if len(pending) == 10000:
                target.executemany("INSERT INTO terms VALUES (?,?,?,?,?)", pending)
                pending.clear()
        if pending:
            target.executemany("INSERT INTO terms VALUES (?,?,?,?,?)", pending)
        if len(term_ids) != n or multi_count != counts["multi_document_terms"] or int(np.sum(df == 1)) != counts["singleton_terms"]:
            raise ValueError("Vocabulary no longer matches the frozen SQL counts")
    return term_ids, df, idf, multi_id


def accumulate_part(stream, part, term_ids, df, idf, multi_id, singleton_doc_id,
                    observed_df, mass, mass_squared, weighted_norm2, centroid_map, avgdl):
    rows, start_row = part["rows"], part["start_row"]
    vectors = np.load(DENSE / "embeddings" / part["file"], mmap_mode="r", allow_pickle=False)
    if vectors.shape != (rows, SPEC["dimension"]) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Invalid source embedding part")
    document_norm2 = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
    vector_sum = vectors.sum(axis=0, dtype=np.float64)
    global_terms, document_ids, frequencies = array("i"), array("i"), array("i")
    lengths = np.empty(rows, dtype=np.int32)
    for local_doc in range(rows):
        line = stream.readline()
        if not line:
            raise ValueError("Chunk stream ended before the embedding sequence")
        record = json.loads(line)
        if type(record.get("vector_id")) is not int or record["vector_id"] != start_row + local_doc:
            raise ValueError("Chunk vector_id differs from the embedding row")
        counts = Counter(analyze_sqlite_bm25_text(record["text"]))
        lengths[local_doc] = sum(counts.values())
        for term, tf in counts.items():
            term_id = term_ids.get(term)
            if term_id is None:
                raise ValueError(f"Analyzed source term absent from frozen term_stats: {term!r}")
            global_terms.append(term_id)
            document_ids.append(local_doc)
            frequencies.append(tf)
    if array("i").itemsize != 4:
        raise RuntimeError("This bounded-memory implementation requires 32-bit array('i')")
    term_indices = np.frombuffer(global_terms, dtype=np.int32)
    doc_indices = np.frombuffer(document_ids, dtype=np.int32)
    tf = np.frombuffer(frequencies, dtype=np.int32)
    if len(term_indices):
        unique_terms, local_terms = np.unique(term_indices, return_inverse=True)
        local_terms = local_terms.astype(np.int32)
        observed_df[unique_terms] += np.bincount(local_terms, minlength=len(unique_terms)).astype(np.int32)
        singleton = df[term_indices] == 1
        singleton_doc_id[term_indices[singleton]] = start_row + doc_indices[singleton]
        weights = idf[term_indices] * 2.5 * tf / (tf + 1.5 * (.25 + .75 * lengths[doc_indices] / avgdl))
        mass[unique_terms] += np.bincount(local_terms, weights=weights, minlength=len(unique_terms))
        mass_squared[unique_terms] += np.bincount(local_terms, weights=weights * weights, minlength=len(unique_terms))
        weighted_norm2[unique_terms] += np.bincount(local_terms, weights=weights * document_norm2[doc_indices], minlength=len(unique_terms))
        matrix = sparse.csr_matrix((weights.astype(np.float32), (local_terms, doc_indices)), shape=(len(unique_terms), rows), dtype=np.float32)
        keep = multi_id[unique_terms] >= 0
        output_rows = multi_id[unique_terms[keep]]
        weighted_vectors = matrix[keep] @ vectors
        for offset in range(0, len(output_rows), 4096):
            selected = output_rows[offset:offset + 4096]
            centroid_map[selected] += weighted_vectors[offset:offset + 4096]
    del vectors
    return int(lengths.sum(dtype=np.int64)), vector_sum, len(term_indices)


def crosscheck_probe(term_ids, df, multi_id, mass, mass_squared, weighted_norm2, centroids):
    with np.load(PROBE / "moments.npz", allow_pickle=False) as reference:
        terms = reference["terms"].astype(str)
        if len(terms) != SPEC["completion_crosscheck"]["terms"]:
            raise ValueError("Expected exactly the existing 32-term quality reference")
        ids = np.asarray([term_ids[term] for term in terms], dtype=np.int64)
        if not np.array_equal(df[ids], reference["df"]) or np.any(multi_id[ids] < 0):
            raise ValueError("Reference document frequencies differ")
        expected = reference["sum_b_times_dense"] / reference["sum_b"][:, None]
        centroid_error = float(np.max(np.abs(centroids[multi_id[ids]].astype(np.float64) - expected)))
        scalar_errors = {}
        for name, actual, saved in (("mass", mass[ids], reference["sum_b"]),
                                    ("mass_squared", mass_squared[ids], reference["sum_b_squared"]),
                                    ("weighted_norm2", weighted_norm2[ids], reference["sum_b_times_dense_squared_norm"])):
            scalar_errors[name] = float(np.max(np.abs(actual - saved) / np.maximum(np.abs(saved), np.finfo(np.float64).tiny)))
    if centroid_error > SPEC["completion_crosscheck"]["centroid_max_abs_tolerance"] or max(scalar_errors.values()) > SPEC["completion_crosscheck"]["scalar_relative_tolerance"]:
        raise ValueError(f"Existing 32-term reference crosscheck failed: centroid={centroid_error}, scalar={scalar_errors}")
    return {"status": "passed_once_against_existing_FP64_32_term_probe", "terms": len(terms),
            "centroid_max_abs_error": centroid_error, "scalar_max_relative_errors": scalar_errors,
            "source_postings_reread": False}


def build():
    protocol = read(OUT / "protocol.json")
    if protocol["spec"] != SPEC:
        raise ValueError("Implementation differs from the frozen preparation")
    existing = {path.name for path in OUT.iterdir()}
    if existing != {"protocol.json"}:
        raise RuntimeError("Build output or partial artifacts already exist; no automatic rerun/resume")
    bm, dm, binding = source_manifests()
    if binding != protocol["source_binding"]:
        raise ValueError("Current source manifests differ from the frozen bindings")
    began = time.perf_counter()
    state = {"status": "building_vocabulary", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "completed_parts": 0, "documents": 0, "analyzed_tokens": 0, "term_document_pairs": 0,
             "resumable": False, "query_reads": 0, "label_reads": 0, "encoder_calls": 0, "paid_calls": 0}
    progress(state)
    try:
        with threadpool_limits(limits=SPEC["threads"]):
            term_ids, df, idf, multi_id = build_vocabulary(protocol["vocabulary_counts_sql"])
            vocabulary_seconds = time.perf_counter() - began
            count = len(df)
            singleton_doc_id = np.full(count, -1, dtype=np.int64)
            observed_df = np.zeros(count, dtype=np.int32)
            mass, mass_squared, weighted_norm2 = (np.zeros(count, dtype=np.float64) for _ in range(3))
            corpus_sum = np.zeros(SPEC["dimension"], dtype=np.float64)
            centroids = np.lib.format.open_memmap(OUT / "centroids.npy", mode="w+", dtype=np.float32,
                shape=(protocol["vocabulary_counts_sql"]["multi_document_terms"], SPEC["dimension"]))
            centroids[:] = 0
            state.update(status="scanning_corpus", vocabulary_terms=count)
            progress(state)
            scan_began = time.perf_counter()
            chunks = DENSE / dm["artifacts"]["chunks"]["file"]
            with chunks.open("r", encoding="utf-8") as stream:
                for part_number, part in enumerate(dm["artifacts"]["embeddings"]["parts"], start=1):
                    tokens, vector_sum, pairs = accumulate_part(stream, part, term_ids, df, idf, multi_id,
                        singleton_doc_id, observed_df, mass, mass_squared, weighted_norm2, centroids,
                        bm["average_document_length"])
                    corpus_sum += vector_sum
                    state["completed_parts"] = part_number
                    state["documents"] += part["rows"]
                    state["analyzed_tokens"] += tokens
                    state["term_document_pairs"] += pairs
                    state["elapsed_seconds"] = time.perf_counter() - began
                    progress(state)
                    if part_number % 10 == 0 or part_number == len(dm["artifacts"]["embeddings"]["parts"]):
                        print(json.dumps(state, allow_nan=False), flush=True)
                if stream.readline():
                    raise ValueError("Chunk stream contains rows beyond the embedding sequence")
            scan_seconds = time.perf_counter() - scan_began
            if state["documents"] != SPEC["documents"] or state["analyzed_tokens"] != bm["total_document_length"]:
                raise ValueError("Integrated document count or analyzed token total differs from BM25 manifest")
            if not np.array_equal(observed_df, df):
                raise ValueError(f"Integrated DF differs for {int(np.count_nonzero(observed_df != df))} terms")
            if np.any(singleton_doc_id[df == 1] < 0) or np.any(singleton_doc_id[df >= 2] != -1):
                raise ValueError("Singleton vector lookup IDs are incomplete or inconsistent")
            if any(not np.isfinite(values).all() or np.any(values <= 0) for values in (mass, mass_squared, weighted_norm2)):
                raise ValueError("Invalid accumulated scalar moments")
            normalize_began = time.perf_counter()
            state["status"] = "normalizing_multi_term_centroids"
            progress(state)
            multi_terms = np.flatnonzero(multi_id >= 0)
            for offset in range(0, len(multi_terms), 4096):
                term_block = multi_terms[offset:offset + 4096]
                rows = multi_id[term_block]
                normalized = centroids[rows].astype(np.float64) / mass[term_block, None]
                if not np.isfinite(normalized).all():
                    raise ValueError("Nonfinite completed centroid block")
                centroids[rows] = normalized.astype(np.float32)
            centroids.flush()
            normalization_seconds = time.perf_counter() - normalize_began
            check = crosscheck_probe(term_ids, df, multi_id, mass, mass_squared, weighted_norm2, centroids)
            corpus_mean = corpus_sum / SPEC["documents"]
            with (OUT / "stats.npz").open("xb") as target:
                np.savez_compressed(target, df=df, idf=idf, multi_id=multi_id,
                    singleton_doc_id=singleton_doc_id, mass=mass, mass_squared=mass_squared,
                    weighted_norm2=weighted_norm2, observed_df=observed_df, corpus_mean=corpus_mean)
            del centroids
        summary = {"status": "complete_corpus_static_pairing_bank_no_router_gain_claim", "complete": True,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_binding": binding,
                   "vocabulary_counts_sql": protocol["vocabulary_counts_sql"],
                   "documents": state["documents"], "analyzed_tokens": state["analyzed_tokens"],
                   "term_document_pairs": state["term_document_pairs"], "completed_parts": state["completed_parts"],
                   "integrated_exact_df_check": True, "integrated_exact_token_total_check": True,
                   "centroid_precision": "FP32 local sparse products and FP32 inter-part accumulation; final FP64 division cast to FP32, not exact FP64 weighted vectors",
                   "scalar_and_corpus_mean_precision": "FP64 accumulation from existing FP32 document vectors",
                   "singleton_storage": "singleton_doc_id uses original fixed Dense vector_id; centroid equals that document vector; multi_id=-1",
                   "quality_crosscheck": check,
                   "timings_seconds": {"vocabulary": vocabulary_seconds, "corpus_scan_and_moments": scan_seconds,
                                       "centroid_normalization": normalization_seconds, "total": time.perf_counter() - began},
                   "files": {name: {"bytes": (OUT / name).stat().st_size} for name in ("vocabulary.sqlite3", "stats.npz", "centroids.npy")},
                   "query_reads": 0, "label_reads": 0, "retained_query_reads": 0, "encoder_calls": 0,
                   "retrieval_calls": 0, "model_fits": 0, "paid_calls": 0,
                   "router_effect_established": False, "novelty_established": False}
        write_new(OUT / "summary.json", summary)
        state.update(status="complete", elapsed_seconds=time.perf_counter() - began)
        progress(state)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as error:
        state.update(status="failed_terminal_no_automatic_resume", error_type=type(error).__name__,
                     error=str(error), elapsed_seconds=time.perf_counter() - began)
        progress(state)
        print(json.dumps(state, ensure_ascii=False, allow_nan=False), flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--build", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else build()
