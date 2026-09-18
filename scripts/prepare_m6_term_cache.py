"""Cache frozen sixth-layer distinct-term vectors for two consumed cohorts.

Only the prior complete-cohort fit/development rows are encoded. Term mapping
reuses E16's character-overlap and repeated-occurrence averaging implementation.
The dense retriever and every RAG artifact remain unchanged. Default invocation
describes the work; --prepare-only does CPU preparation; --execute adds one
fixed-shape six-layer encoder pass and never fits a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "work/router_research"
RUNS = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs"
CACHE = RUNS / "m6_term_aggregation_v1/cache"
ROLES = RUNS / "m6_complete_cohort_extension_v1/source_labels_and_roles.npz"
BEIR = BASE / "m6_beir_validation_v1"
INDEX = ROOT / "artifacts/_sparse_indexes/sqlite_bm25_04df51906ff9598b"
MODEL = Path("C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
import prepare_token_corpus_inputs as term_prep

SPEC = {
    "scope": "consumed_old_fit_cal_and_BEIR_fit_dev_only",
    "row_order": ["old_fit6144", "old_dev1536", "beir_fit4165", "beir_dev1044"],
    "queries": 12889, "encoder_depth": 6, "hidden_size": 384,
    "max_length": 128, "batch_size": 8, "query_prefix": "",
    "encoder_autocast": "cuda_bfloat16", "term_storage": "float32",
    "term_aggregation_accumulator": "float64_then_float32_like_E16",
    "term_mapping": "prepare_token_corpus_inputs.sparse_assembly",
    "empty_terms": "one_existing_M6_vector_with_IDF1",
    "unknown_corpus_term_IDF": 0,
    "all_zero_IDF_nonempty": "retain_terms_and_zero_IDFs_mark_shared_fallback_to_existing_M6",
    "new_encoder_fits": 0, "new_retrieval_calls": 0, "new_answer_calls": 0,
    "old_outer_test_encoder_forwards": 0, "full_M6_replay": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_execution_allowed():
    import yaml
    registry = yaml.safe_load((ROOT / "analysis/hotpotqa_router/registry.yaml").read_text(encoding="utf-8"))
    if registry["research_continuation"]["experiment_execution"] == "paused_by_user":
        raise RuntimeError("Experiments remain paused by the user.")


def questions_by_id(path, selected):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = str(row["query_id"])
            if qid in selected:
                assert qid not in result and isinstance(row["question"], str) and row["question"].strip()
                result[qid] = (str(row["group_id"]), row["question"])
    assert set(result) == selected, f"Missing or extra selected query identities in {path}"
    return result


def load_rows():
    with np.load(ROLES, allow_pickle=False) as saved:
        roles = {name: saved[name].copy() for name in saved.files}
    old_indices = np.r_[roles["old_fit"], roles["old_cal"]]
    new_indices = np.r_[roles["new_fit"], roles["new_dev"]]
    assert tuple(len(roles[key]) for key in ("old_fit", "old_cal", "new_fit", "new_dev")) == (6144, 1536, 4165, 1044)
    assert len(set(old_indices)) == 7680 and len(set(new_indices)) == 5209
    assert np.array_equal(np.sort(new_indices), np.arange(5209))
    with np.load(BASE / "layer_pooling_v1/features.npz", allow_pickle=False) as saved:
        old_qids = saved["query_ids"][old_indices].copy()
        old_groups = saved["group_ids"][old_indices].copy()
        old_x = saved["M6"][old_indices].copy()
    with np.load(BASE / "layer_pooling_v1/predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(old_qids, saved["query_ids"][old_indices])
        assert np.array_equal(old_groups, saved["group_ids"][old_indices])
        old_u = saved["utility"][old_indices].copy()
    with np.load(BEIR / "actions.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["query_ids"], roles["query_ids"])
        assert np.array_equal(saved["group_ids"], roles["group_ids"])
        new_x = saved["M6"][new_indices].copy()
    new_qids, new_groups = roles["query_ids"][new_indices], roles["group_ids"][new_indices]
    old_questions = questions_by_id(BASE / "token_corpus_inputs/queries.jsonl", set(map(str, old_qids)))
    new_questions = questions_by_id(BEIR / "questions.jsonl", set(map(str, new_qids)))
    assert all(old_questions[str(q)][0] == str(g) for q, g in zip(old_qids, old_groups))
    assert all(new_questions[str(q)][0] == str(g) for q, g in zip(new_qids, new_groups))
    query_ids = np.r_[old_qids, new_qids]
    groups = np.r_[roles["old_fit_canonical_keys"], roles["old_cal_canonical_keys"], new_groups]
    source = np.r_[np.full(7680, "old"), np.full(5209, "beir")]
    role = np.r_[np.full(6144, "fit"), np.full(1536, "dev"), np.full(4165, "fit"), np.full(1044, "dev")]
    assert len(query_ids) == len(set(query_ids)) == SPEC["queries"]
    assert all(len(str(group)) == 64 for group in groups)
    assert not set(groups[role == "fit"]) & set(groups[role == "dev"])
    assert not set(groups[source == "old"]) & set(groups[source == "beir"])
    x, utility = np.r_[old_x, new_x], np.r_[old_u, roles["utility"][new_indices]]
    assert x.shape == (SPEC["queries"], 384) and x.dtype == np.float32 and np.isfinite(x).all()
    assert utility.shape == (SPEC["queries"], 2) and np.isfinite(utility).all()
    assert ((utility >= 0) & (utility <= 1)).all()
    questions = [old_questions[str(q)][1] for q in old_qids] + [new_questions[str(q)][1] for q in new_qids]
    return dict(query_ids=query_ids, group_ids=groups, source=source, role=role,
                M6=x, utility=utility.astype(np.float64)), questions


def prepare():
    require_execution_allowed()
    manifest_path = CACHE / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["spec"] == SPEC
        if manifest["status"] not in ("CPU_prepared", "complete"):
            raise RuntimeError("Prior cache preparation or extraction is incomplete; inspect it before any retry.")
        return manifest
    CACHE.mkdir(parents=True, exist_ok=True)
    if any(CACHE.iterdir()):
        raise RuntimeError("Cache directory contains files without its manifest; inspect before retry.")
    started = time.perf_counter()
    manifest = dict(status="preparing_CPU", created_at_utc=datetime.now(timezone.utc).isoformat(), spec=SPEC)
    write_json(manifest_path, manifest)
    rows, questions = load_rows()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast
    encoded = tokenizer(questions, padding="max_length", truncation=True, max_length=SPEC["max_length"],
                        return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors="np")
    token_fields = ("input_ids", "attention_mask", "token_type_ids")
    assert all(encoded[key].shape == (len(questions), 128) for key in token_fields)
    term, diagnostics = term_prep.sparse_assembly(questions, encoded["offset_mapping"],
                                                encoded["special_tokens_mask"], encoded["attention_mask"])
    index = read_json(INDEX / "manifest.json")
    assert index["status"] == "complete" and index["document_count"] == 5233329
    assert index["identity"]["source_build_id"] == "build_65e9037e4c1eed7b"
    assert index["identity"]["analyzer"]["name"] == "english_regex_casefold_v1"
    database = INDEX / index["artifacts"]["database"]["file"]
    statistics = term_prep.read_sqlite_bm25_term_stats(database, set(map(str, term["terms"])))
    # Match E16: corpus-absent terms receive zero; no new estimate or smoothing.
    raw_idf = np.asarray([statistics[str(value)][1] if str(value) in statistics else 0. for value in term["terms"]], dtype=np.float32)
    assert np.isfinite(raw_idf).all() and (raw_idf >= 0).all()
    raw_counts = np.diff(term["query_term_indptr"])
    empty_terms = raw_counts == 0
    fallback = empty_terms.copy()
    counts = np.maximum(raw_counts, 1)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    idf = np.empty(int(indptr[-1]), dtype=np.float32)
    term_strings = []
    all_zero = 0
    for query in range(len(questions)):
        start, stop = term["query_term_indptr"][query:query + 2]
        lo, hi = indptr[query:query + 2]
        if start == stop:
            idf[lo:hi] = 1.
            term_strings.append("<M6_EMPTY_TERMS>")
        else:
            idf[lo:hi] = raw_idf[start:stop]
            term_strings.extend(term["terms"][start:stop].tolist())
            zero = not np.any(raw_idf[start:stop] > 0)
            fallback[query] = zero
            all_zero += int(zero)
    rows.update(term_indptr=indptr, term_idf=idf, fallback=fallback, terms=np.asarray(term_strings))
    for name, values in rows.items():
        np.save(CACHE / f"{name}.npy", values, allow_pickle=False)
    np.savez_compressed(CACHE / "encoder_inputs.npz", **{key: encoded[key] for key in token_fields})
    np.savez_compressed(CACHE / "term_alignment.npz", **{key: value for key, value in term.items() if key != "terms"})
    alignment_counts = Counter()
    for diagnostic in diagnostics:
        alignment_counts.update(diagnostic)
    source_files = [ROLES, BASE / "layer_pooling_v1/features.npz", BASE / "layer_pooling_v1/predictions.npz",
                    BASE / "token_corpus_inputs/queries.jsonl", BEIR / "actions.npz", BEIR / "questions.jsonl",
                    INDEX / "manifest.json", database, BASE / "prepare_token_corpus_inputs.py",
                    BASE / "m6_early_exit.py", Path(__file__)]
    manifest.update(status="CPU_prepared", queries=len(questions), real_terms=int(raw_counts.sum()),
                    cached_terms=int(indptr[-1]), max_terms_per_query=int(counts.max()),
                    empty_term_fallback_queries=int(empty_terms.sum()), all_zero_IDF_nonempty_queries=all_zero,
                    shared_fallback_queries=int(fallback.sum()),
                    unknown_corpus_terms=int(np.sum(raw_idf == 0)), alignment_counts=dict(alignment_counts),
                    encoder_batches=(len(questions) + 7) // 8, encoder_padded_rows=(-len(questions)) % 8,
                    expected_term_hidden_bytes=int(indptr[-1]) * 384 * 4,
                    model_snapshot=str(MODEL), CPU_preparation_seconds=time.perf_counter() - started,
                    utility_columns=["bm25_mean3_F1", "dense_mean3_F1"],
                    source_files={str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} for path in source_files},
                    files={f"{name}.npy": {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in rows.items()})
    write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "queries", "cached_terms", "encoder_batches",
                      "expected_term_hidden_bytes", "empty_term_fallback_queries", "all_zero_IDF_nonempty_queries", "CPU_preparation_seconds")}), flush=True)
    return manifest


def extract():
    manifest = prepare()
    if manifest["status"] == "complete":
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return
    import torch
    from transformers import AutoModel
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with np.load(CACHE / "encoder_inputs.npz", allow_pickle=False) as saved:
        tokens = {key: saved[key] for key in saved.files}
    with np.load(CACHE / "term_alignment.npz", allow_pickle=False) as saved:
        term = {key: saved[key] for key in saved.files}
    indptr = np.load(CACHE / "term_indptr.npy", allow_pickle=False)
    fallback = np.load(CACHE / "fallback.npy", allow_pickle=False)
    existing_m6 = np.load(CACHE / "M6.npy", mmap_mode="r", allow_pickle=False)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    assert model.config.model_type == "bert" and model.config.hidden_size == 384
    assert len(model.encoder.layer) == model.config.num_hidden_layers == 12
    # Identical pruning boundary to FrozenM6Router; no head is loaded or replayed.
    model.encoder.layer = torch.nn.ModuleList(model.encoder.layer[:6])
    model.config.num_hidden_layers = 6
    model.pooler = None
    model.requires_grad_(False).to("cuda").eval()
    manifest.update(status="extracting", encoder_started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(CACHE / "manifest.json", manifest)
    temporary = CACHE / "term_hidden.partial.npy"
    hidden_cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(int(indptr[-1]), 384))
    started, n = time.perf_counter(), manifest["queries"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            batch_ids = list(range(start, stop)) + [stop - 1] * (8 - (stop - start))
            batch = {key: torch.from_numpy(value[batch_ids]).to("cuda") for key, value in tokens.items()}
            hidden = model(**batch).last_hidden_state.float().cpu().numpy()
            assert hidden.shape == (8, 128, 384) and np.isfinite(hidden).all()
            for local, query in enumerate(range(start, stop)):
                lo, hi = indptr[query:query + 2]
                raw_start, raw_stop = term["query_term_indptr"][query:query + 2]
                if raw_start == raw_stop:
                    assert fallback[query] and hi - lo == 1
                    hidden_cache[lo:hi] = existing_m6[query]
                    continue
                assert hi - lo == raw_stop - raw_start
                for local_term, raw_term in enumerate(range(raw_start, raw_stop)):
                    begin, end = term["term_token_indptr"][raw_term:raw_term + 2]
                    index = term["term_token_indices"][begin:end]
                    weights = term["term_token_weights"][begin:end]
                    hidden_cache[lo + local_term] = weights @ hidden[local, index].astype(np.float64)
                assert np.isfinite(hidden_cache[lo:hi]).all()
            if stop % 1024 == 0 or stop == n:
                print(json.dumps(dict(phase="frozen_six_layer_term_extraction", queries=stop,
                                      total_queries=n, elapsed_seconds=time.perf_counter() - started)), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    hidden_cache.flush()
    del hidden_cache
    temporary.replace(CACHE / "term_hidden.npy")
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    encoder_and_term_aggregation_seconds=seconds, encoder_real_query_forwards=n,
                    encoder_total_query_forwards=manifest["encoder_batches"] * 8,
                    encoder_parameters_frozen=True, identity_shape_finite_alignment_checks="passed",
                    new_model_fits=0, new_retrieval_calls=0, new_answer_calls=0,
                    outer_test_encoder_forwards=0, M6_replays=0)
    manifest["files"]["term_hidden.npy"] = {"shape": [int(indptr[-1]), 384], "dtype": "float32"}
    write_json(CACHE / "manifest.json", manifest)
    print(json.dumps(dict(status="complete", queries=n, terms=int(indptr[-1]),
                          encoder_and_term_aggregation_seconds=seconds)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        extract()
    elif args.prepare_only:
        prepare()
    else:
        print(json.dumps(dict(mode="describe_only", cache_directory=str(CACHE), spec=SPEC), indent=2))
