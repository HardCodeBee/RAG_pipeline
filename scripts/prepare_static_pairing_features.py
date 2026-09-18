"""Prepare shared features for the fixed consumed old/BEIR fit/development rows.

--prepare freezes input order and extraction settings without loading an encoder.
--extract requires that protocol and a complete corpus bank. It encodes raw
questions in the actual Dense retrieval space and performs bank lookups only.
Neither command loads outcome labels or fits a model. Existing extraction output,
including a partial/failed attempt, requires inspection rather than automatic retry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
OUT = RUNS / "static_pairing_features_v1"
ROLES = RUNS / "m6_complete_cohort_extension_v1/source_labels_and_roles.npz"
OLD = BASE / "layer_pooling_v1/features.npz"
BEIR = BASE / "m6_beir_validation_v1"
OLD_QUESTIONS = BASE / "token_corpus_inputs/queries.jsonl"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots" / MODEL_REVISION
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from router_static_pairing import BANK, DENSE_QUERY_SPACE, StaticPairingBank

ROLE_KEYS = ("query_ids", "group_ids", "new_fit", "new_dev", "old_fit", "old_cal",
             "old_fit_canonical_keys", "old_cal_canonical_keys")
SCALAR_NAMES = (
    "log_unique_term_count", "corpus_known_fraction", "centroid_available",
    "idf_mean", "idf_std", "idf_max", "log_total_lexical_mass_per_document",
    "centered_centroid_squared_norm", "aligned", "coordinate_control",
)
SPEC = {
    "scope": "consumed_old_fit_dev_and_BEIR_fit_dev_only",
    "row_order": ["old_fit6144", "old_dev1536", "beir_fit4165", "beir_dev1044"],
    "queries": 12889, "fit_queries": 10309, "development_queries": 2580,
    "dense_query_space": DENSE_QUERY_SPACE,
    "model": "BAAI/bge-small-en-v1.5", "revision": MODEL_REVISION,
    "pooling": "model_default", "max_sequence_length": 512,
    "normalize": True, "query_prefix": "", "query_input": "raw_question",
    "batch_size": 64, "device": "cuda", "encoder_dtype": "float32_no_autocast_no_tf32",
    "encoder_wrapper": "src.embedders.text_embedder.TextEmbedder.encode_queries",
    "M6": "reuse_existing_selected_rows_without_reencoding",
    "Dense": "reencode_all_selected_rows_under_one_retrieval_contract",
    "scalar_names": list(SCALAR_NAMES), "scalar_dtype": "float64",
    "bank_directory": str(BANK), "requires_complete_bank": True,
    "outcome_label_fields_read": [], "new_model_fits": 0,
    "new_retrieval_calls": 0, "new_answer_calls": 0,
    "old_outer_test_extractions": 0, "two_wiki_rows_read": 0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def digest_arrays(values):
    result = {}
    for key, value in values.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(json.dumps([str(array.dtype), list(array.shape)]).encode("utf-8"))
        digest.update(array.tobytes())
        result[key] = digest.hexdigest()
    return result


def questions_by_id(path, selected):
    """Same selected-ID projection as prepare_m6_term_cache.questions_by_id."""
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in selected:
                require(query_id not in result, "Duplicate selected query identity")
                require(isinstance(row["question"], str) and row["question"].strip(), "Invalid selected question")
                result[query_id] = (str(row["group_id"]), row["question"])
    require(set(result) == selected, "Selected query identities are incomplete")
    return result


def load_selected_rows():
    # Never iterate saved.files: ROLES also contains utility labels.
    with np.load(ROLES, allow_pickle=False) as saved:
        roles = {key: saved[key].copy() for key in ROLE_KEYS}
    partitions = ("old_fit", "old_cal", "new_fit", "new_dev")
    require(tuple(len(roles[key]) for key in partitions) == (6144, 1536, 4165, 1044), "Unexpected partition sizes")
    require(all(np.issubdtype(roles[key].dtype, np.integer) for key in partitions), "Noninteger row indices")
    old_indices = np.r_[roles["old_fit"], roles["old_cal"]]
    new_indices = np.r_[roles["new_fit"], roles["new_dev"]]
    require(len(set(old_indices)) == 7680 and np.all((old_indices >= 0) & (old_indices < 9600)), "Invalid old selected rows")
    require(np.array_equal(np.sort(new_indices), np.arange(5209)), "Invalid BEIR selected rows")
    with np.load(OLD, allow_pickle=False) as saved:
        old_ids = saved["query_ids"][old_indices].copy()
        old_groups = saved["group_ids"][old_indices].copy()
        old_m6 = saved["M6"][old_indices].copy()
    with np.load(BEIR / "actions.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["query_ids"], roles["query_ids"]), "BEIR query ordering changed")
        require(np.array_equal(saved["group_ids"], roles["group_ids"]), "BEIR group ordering changed")
        new_m6 = saved["M6"][new_indices].copy()
    new_ids, new_groups = roles["query_ids"][new_indices], roles["group_ids"][new_indices]
    old_questions = questions_by_id(OLD_QUESTIONS, set(map(str, old_ids)))
    new_questions = questions_by_id(BEIR / "questions.jsonl", set(map(str, new_ids)))
    require(all(old_questions[str(q)][0] == str(g) for q, g in zip(old_ids, old_groups)), "Old question/group mismatch")
    require(all(new_questions[str(q)][0] == str(g) for q, g in zip(new_ids, new_groups)), "BEIR question/group mismatch")
    require(len(roles["old_fit_canonical_keys"]) == 6144 and len(roles["old_cal_canonical_keys"]) == 1536,
            "Old canonical group keys are incomplete")
    rows = {
        "query_ids": np.r_[old_ids, new_ids],
        "group_ids": np.r_[roles["old_fit_canonical_keys"], roles["old_cal_canonical_keys"], new_groups],
        "source": np.r_[np.full(7680, "old"), np.full(5209, "beir")],
        "role": np.r_[np.full(6144, "fit"), np.full(1536, "dev"), np.full(4165, "fit"), np.full(1044, "dev")],
        "M6": np.r_[old_m6, new_m6],
    }
    require(len(rows["query_ids"]) == len(set(rows["query_ids"])) == SPEC["queries"], "Query identity coverage mismatch")
    groups, source, role = rows["group_ids"], rows["source"], rows["role"]
    require(len(groups) == SPEC["queries"] and all(len(str(g)) == 64 for g in groups), "Invalid canonical groups")
    require(not set(groups[role == "fit"]) & set(groups[role == "dev"]), "Fit/development group overlap")
    require(not set(groups[source == "old"]) & set(groups[source == "beir"]), "Cross-source group overlap")
    require(rows["M6"].shape == (SPEC["queries"], 384) and rows["M6"].dtype == np.float32
            and np.isfinite(rows["M6"]).all(), "Invalid selected M6 rows")
    questions = [old_questions[str(q)][1] for q in old_ids] + [new_questions[str(q)][1] for q in new_ids]
    return rows, questions


def input_binding(rows, questions):
    question_digest = hashlib.sha256()
    for query_id, question in zip(rows["query_ids"], questions):
        question_digest.update((json.dumps([str(query_id), question], ensure_ascii=False) + "\n").encode("utf-8"))
    return {"selected_array_sha256": digest_arrays(rows), "selected_questions_sha256": question_digest.hexdigest()}


def source_binding():
    paths = [ROLES, OLD, BEIR / "actions.npz", OLD_QUESTIONS, BEIR / "questions.jsonl",
             MODEL / "model.safetensors", MODEL / "config.json", MODEL / "1_Pooling/config.json",
             MODEL / "tokenizer.json", MODEL / "tokenizer_config.json"]
    metadata = {str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} for path in paths}
    code = [Path(__file__), ROOT / "scripts/router_static_pairing.py", ROOT / "src/embedders/text_embedder.py",
            ROOT / "src/model_backends/huggingface_snapshot.py", ROOT / "src/retrievers/sqlite_bm25.py"]
    return {"files": metadata, "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not any(OUT.iterdir()), "Output already exists; inspect it rather than preparing or retrying automatically")
    started = time.perf_counter()
    rows, questions = load_selected_rows()
    protocol = {"id": "static_pairing_features_v1", "status": "prepared", "created_at_utc": utc_now(),
                "spec": SPEC, "inputs": input_binding(rows, questions), "sources": source_binding(),
                "model_snapshot": str(MODEL), "CPU_preparation_seconds": time.perf_counter() - started}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print(json.dumps({"status": "prepared", "queries": len(questions), "output": str(OUT)}), flush=True)


def bank_binding():
    summary_path = BANK / "summary.json"
    require(summary_path.is_file() and read_json(summary_path).get("complete") is True, "Full corpus bank is not complete")
    paths = [summary_path, BANK / "stats.npz", BANK / "centroids.npy", BANK / "vocabulary.sqlite3"]
    return {"summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "files": {str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} for path in paths}}


def extract():
    protocol_path = OUT / "protocol.json"
    require(protocol_path.is_file(), "Run --prepare before --extract")
    require({path.name for path in OUT.iterdir()} == {"protocol.json"},
            "Extraction output already exists; partial and completed attempts are not overwritten")
    protocol = read_json(protocol_path)
    require(protocol["id"] == "static_pairing_features_v1" and protocol["status"] == "prepared"
            and protocol["spec"] == SPEC, "Extraction specification differs from the frozen protocol")
    require(protocol["sources"] == source_binding(), "Frozen source metadata or implementation changed")
    bank_record = bank_binding()
    started = time.perf_counter()
    summary = {"status": "extracting", "complete": False, "started_at_utc": utc_now(),
               "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "bank": bank_record}
    write_json(OUT / "summary.json", summary, exclusive=True)
    bank = None
    try:
        rows, questions = load_selected_rows()
        require(input_binding(rows, questions) == protocol["inputs"], "Selected input values or row order changed")
        import torch
        from src.embedders.text_embedder import TextEmbedder
        from src.model_backends.huggingface_snapshot import resolve_hf_snapshot
        require(torch.cuda.is_available(), "The frozen float32 CUDA encoder is unavailable")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(resolve_hf_snapshot(SPEC["model"], revision=MODEL_REVISION, local_files_only=True) == MODEL.resolve(),
                "Resolved model snapshot differs from the prepared local snapshot")
        model_started = time.perf_counter()
        encoder = TextEmbedder(backend="sentence_transformers", model_name=SPEC["model"], revision=MODEL_REVISION,
                              normalize=True, batch_size=64, query_prefix="", document_prefix="",
                              max_sequence_length=512, local_files_only=True, device="cuda")
        encoder._model.requires_grad_(False)
        require(all(p.dtype == torch.float32 for p in encoder._model.parameters() if p.is_floating_point()),
                "Dense encoder must use float32 parameters")
        model_seconds = time.perf_counter() - model_started
        bank = StaticPairingBank(DENSE_QUERY_SPACE)
        n = len(questions)
        dense = np.empty((n, 384), dtype=np.float32)
        scalars = np.empty((n, len(SCALAR_NAMES)), dtype=np.float64)
        encoding_seconds = lookup_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, n, SPEC["batch_size"]):
                stop = min(start + SPEC["batch_size"], n)
                tick = time.perf_counter()
                dense[start:stop] = encoder.encode_queries(questions[start:stop])
                encoding_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                for index in range(start, stop):
                    features = bank.features(questions[index], dense[index])
                    require(set(features) == set(SCALAR_NAMES), "Bank feature schema changed")
                    scalars[index] = [features[name] for name in SCALAR_NAMES]
                lookup_seconds += time.perf_counter() - tick
                if stop % 640 == 0 or stop == n:
                    print(json.dumps({"phase": "Dense_encoding_and_static_lookup", "queries": stop, "total": n,
                                      "encoding_seconds": encoding_seconds, "lookup_seconds": lookup_seconds}), flush=True)
        require(np.isfinite(dense).all() and np.isfinite(scalars).all(), "Nonfinite extracted features")
        require(np.allclose(np.linalg.norm(dense, axis=1), 1.0, atol=1e-5, rtol=1e-4), "Dense vectors are not normalized")
        available = scalars[:, SCALAR_NAMES.index("centroid_available")]
        require(np.isin(available, [0., 1.]).all(), "Invalid centroid support mask")
        require(bank_binding() == bank_record and source_binding() == protocol["sources"], "Inputs changed during extraction")
        temporary = OUT / "features.partial.npz"
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **rows, Dense=dense, scalar_names=np.asarray(SCALAR_NAMES), scalars=scalars)
        temporary.replace(OUT / "features.npz")
        support = {}
        for source_name in ("old", "beir"):
            for role_name in ("fit", "dev"):
                selected = (rows["source"] == source_name) & (rows["role"] == role_name)
                supported = int(available[selected].sum())
                support[f"{source_name}_{role_name}"] = {"queries": int(selected.sum()), "supported": supported,
                                                         "unsupported": int(selected.sum()) - supported}
        summary.update(status="complete", complete=True, completed_at_utc=utc_now(), queries=n,
                       supported_queries=int(available.sum()), unsupported_queries=int(n - available.sum()),
                       support_by_source_role=support, model_load_seconds=model_seconds,
                       encoding_seconds=encoding_seconds, lookup_seconds=lookup_seconds,
                       elapsed_seconds=time.perf_counter() - started, encoder_query_forwards=n,
                       scalar_names=list(SCALAR_NAMES), M6_shape=list(rows["M6"].shape),
                       Dense_shape=list(dense.shape), scalars_shape=list(scalars.shape),
                       outcome_label_fields_read=[], new_model_fits=0, new_retrieval_calls=0, new_answer_calls=0,
                       old_outer_test_extractions=0, two_wiki_rows_read=0)
        write_json(OUT / "summary.json", summary)
        print(json.dumps({"status": "complete", "queries": n, "supported": int(available.sum()),
                          "unsupported": int(n - available.sum()), "output": str(OUT)}), flush=True)
    except BaseException as error:
        summary.update(status="failed_requires_inspection", complete=False, failure_type=type(error).__name__,
                       stopped_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        write_json(OUT / "summary.json", summary)
        raise
    finally:
        if bank is not None:
            bank.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--extract", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.extract:
        extract()
    else:
        print(json.dumps({"mode": "describe_only", "output": str(OUT), "spec": SPEC}, indent=2))
