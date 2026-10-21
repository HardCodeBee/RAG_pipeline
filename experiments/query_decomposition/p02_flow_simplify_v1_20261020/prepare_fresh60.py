"""Prepare, never run, a fixed 60-query RAG confirmation with existing R02 output.

The complete historical decomposer dev60 roster is retained.  Answers are
accessed only AFTER its membership and prediction/source identities are fixed,
then exported separately for offline scoring.  No API, SSH or retrieval runs.
All outputs are proposed preparation artifacts; this script selects no method.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "experiments/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018"
PREDICTIONS = ROOT / (
    "outputs/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018/"
    "returned/snapshot_013/results/v001/training40/r02_expanded/predictions"
)
NATIVE = ROOT / "data/multihop_comprehensive_v1_20261015"
CHECKPOINT_SHA = "728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92"
PROMPT_SHA = "508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8"
DATASETS = ("hotpotqa", "2wiki", "musique")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pin(path):
    return {"path": relative(path), "sha256": sha(path)}


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def primitives(value):
    return set(re.findall(r"\d+", value.split("__", 1)[1])) if "__" in value else set()


def unchanged_or_write(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("Preserve existing proposed artifact: " + str(path))
    else:
        path.write_text(text, encoding="utf-8")


def jsonl(values):
    return "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)


def prior_rag_files(qids):
    """Locate actual case-file exposure by filename; never inspect answer/F1."""
    result = defaultdict(list)
    ignored = {"code_snapshot", "remote_package", "bundle", "code", "vendor",
               "offline_checks", "remote", "returned", "retrieval", "api", "methods",
               "inputs_frozen", "index", "remote_reference_qdmr300"}
    output = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1"
    for current, dirs, files in os.walk(output):
        dirs[:] = [name for name in dirs if name not in ignored]
        if "cases" not in Path(current).parts:
            continue
        for name in files:
            if name.endswith(".json") and name[:-5] in qids:
                result[name[:-5]].append(relative(Path(current) / name))
    return dict(result)


def fixed_runtime():
    """Fix all 60 memberships before touching native reference answer fields."""
    roster_path = SOURCE / "dev60.questions.frozen.jsonl"
    roster = list(rows(roster_path))
    assert len(roster) == len({row["id"] for row in roster}) == 60
    assert Counter(row["dataset"] for row in roster) == dict.fromkeys(DATASETS, 20)
    identity_path = PREDICTIONS / "IDENTITY.json"
    identity = read(identity_path)
    assert identity["checkpoint_manifest_sha256"] == CHECKPOINT_SHA
    assert identity["prompt_sha256"] == PROMPT_SHA
    assert identity["query_sha256"] == sha(roster_path)
    assert identity["do_sample"] is False and identity["num_beams"] == 1
    assert identity["max_new_tokens"] == 1024 and identity["seed"] == 1729
    assert identity["api_calls"] == 0
    exclusions = [HERE / "runtime300.frozen.jsonl",
                  ROOT / "experiments/query_decomposition/b0_strategy_round2_v1_20261020/fresh_validation/dev90.query_only.frozen.jsonl",
                  ROOT / "experiments/query_decomposition/b0_strategy_round2_v1_20261020/fresh_validation/sealed150.query_only.frozen.jsonl"]
    excluded_ids, excluded_queries = set(), set()
    exclusion_pins = []
    for path in exclusions:
        values = list(rows(path))
        excluded_ids.update(row["id"] for row in values)
        excluded_queries.update(norm(row.get("query", row.get("question", ""))) for row in values)
        exclusion_pins.append({**pin(path), "n": len(values)})
    assert len(excluded_ids) == 540
    training_paths = [SOURCE / "DATA3632.frozen.jsonl",
                      ROOT / "experiments/query_decomposition/multigrain_4b_sft3500_v1_20261017/remote_v003/data/MASTER_3500.jsonl",
                      ROOT / "experiments/query_decomposition/multigrain_4b_musique_scope_opt_v2_20261018/continuation/MASTER3500_SCOPE8.candidate.jsonl"]
    train_ids, train_queries, train_primitives, training_pins = set(), set(), set(), []
    for path in training_paths:
        count = 0
        for row in rows(path):
            count += 1
            query = row.get("query", row.get("question"))
            if query:
                train_queries.add(norm(query))
            for value in (row.get("id"), row.get("qid"), row.get("source", {}).get("id")):
                if value:
                    train_ids.add(value)
            train_primitives.update(primitives(str(row.get("source", {}).get("id", ""))))
        training_pins.append({**pin(path), "n": count})
    runtime, bindings = [], []
    for source in roster:
        assert source["id"] == source["dataset"] + ":" + source["qid"]
        assert source["id"] not in excluded_ids and norm(source["query"]) not in excluded_queries
        assert source["id"] not in train_ids and source["qid"] not in train_ids
        assert norm(source["query"]) not in train_queries
        if source["dataset"] == "musique":
            assert source["source_split"] == "train_held_out_query_only"
            assert not primitives(source["qid"]) & train_primitives
        else:
            assert source["source_split"] == "dev"
        path = PREDICTIONS / f"{source['index']:03d}.json"
        prediction = read(path)
        assert prediction["id"] == source["id"] and prediction["query"] == source["query"]
        assert prediction["dataset"] == source["dataset"]
        assert prediction["format_valid"] and not prediction["truncated"]
        assert prediction["generation_error"] is None and not prediction["explicit_references"]
        assert hashlib.sha256(prediction["actual_prompt"].encode("utf-8")).hexdigest() == PROMPT_SHA
        assert json.loads(prediction["raw_output"]) == prediction["parsed"]
        runtime.append({"id": source["id"], "qid": source["qid"], "dataset": source["dataset"],
                        "question": source["query"], "supporting_questions": deepcopy(prediction["parsed"]),
                        "type": source["type"]})
        bindings.append({"id": source["id"], "source_index": source["index"],
                         "source_split": source["source_split"], **pin(path),
                         "identity_sha256_in_prediction": prediction["identity_sha256"]})
    exposure = prior_rag_files({row["qid"] for row in roster})
    # The preparation itself intentionally produces no case trajectory files.
    assert not exposure, "Unexpected prior RAG cases: " + json.dumps(exposure)
    return runtime, {"roster": pin(roster_path), "identity": identity,
                     "identity_file": pin(identity_path), "prediction_bindings": bindings,
                     "excluded_rosters": exclusion_pins, "excluded_unique_ids": len(excluded_ids),
                     "training_sources_checked": training_pins, "prior_RAG_case_files_found": exposure}


def reference_rows(runtime):
    """Only fixed IDs gain access to native answer values in this function."""
    wanted = {ds: {row["qid"]: row for row in runtime if row["dataset"] == ds} for ds in DATASETS}
    native_paths = {"hotpotqa": ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/preparation/offline_gold_source/validation.parquet",
                    "2wiki": NATIVE / "2wiki/native/dev.json",
                    "musique": NATIVE / "musique/native/musique_ans_v1.0_train.jsonl"}
    references = {}

    def add(ds, native, key="id"):
        qid = native[key]
        if qid not in wanted[ds]:
            return
        row = wanted[ds][qid]
        assert native["question"] == row["question"], "Native id/question mismatch"
        assert row["id"] not in references
        references[row["id"]] = {"id": row["id"], "qid": qid, "dataset": ds,
                                 "question": row["question"], "type": row["type"],
                                 "answer": native["answer"],
                                 "aliases": native.get("answer_aliases", []) if ds == "musique" else [],
                                 **({"answer_id": native["answer_id"]} if ds == "2wiki" else {}),
                                 "generation_input_forbidden": True}

    import pyarrow.parquet as pq
    for record in pq.read_table(native_paths["hotpotqa"], columns=["id", "question", "answer"]).to_pylist():
        add("hotpotqa", record)
    from experiments.query_decomposition.multigrain_4b_prompt_train3500_v1_20261017.training_sources import array_stream
    for record in array_stream(native_paths["2wiki"]):
        add("2wiki", record, "_id")
    for record in rows(native_paths["musique"]):
        add("musique", record)
    assert set(references) == {row["id"] for row in runtime}
    return [references[row["id"]] for row in runtime], {ds: pin(path) for ds, path in native_paths.items()}


def prepare():
    runtime, provenance = fixed_runtime()
    # Membership is now fixed without any reference or downstream score access.
    references, native_sources = reference_rows(runtime)
    runtime_path = HERE / "fresh60.runtime.proposed.jsonl"
    reference_path = HERE / "fresh60.references.local.proposed.jsonl"
    unchanged_or_write(runtime_path, jsonl(runtime))
    unchanged_or_write(reference_path, jsonl(references))
    protocol_path = HERE / "PROTOCOL.frozen.json"
    resources = read(protocol_path)["resources"]
    corpus = {}
    for ds, resource in resources.items():
        path = Path(resource["corpus"]["manifest"])
        assert sha(path) == resource["corpus"]["manifest_sha256"]
        manifest = read(path)
        corpus[ds] = {**pin(path), "documents": resource["corpus"]["documents"],
                      "scope": manifest.get("scope", "Frozen complete BEIR HotpotQA corpus"),
                      "new_corpus_or_index_built": False}
    provenance.update({"status": "proposed preparation only; method configuration not selected here",
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "selection": "Complete preexisting decomposer dev60 roster in original order; no answer, F1 or semantic verdict selection",
                       "n": 60, "dataset_counts": dict(Counter(row["dataset"] for row in runtime)),
                       "source_type_counts": {ds: dict(Counter(row["type"] for row in runtime if row["dataset"] == ds)) for ds in DATASETS},
                       "runtime": pin(runtime_path), "references_local_scoring_only": pin(reference_path),
                       "native_sources": native_sources, "corpus_alignment": corpus,
                       "protocol_used_for_resource_identity_only": pin(protocol_path),
                       "training_overlap": {"exact_id": 0, "normalized_query": 0, "musique_primitives": 0},
                       "normalization": "Unicode NFKC + casefold + collapsed whitespace",
                       "native_question_alignment": "All60 exact id and query match; answer fields copied only after membership fixed",
                       "runtime_fields": ["id", "qid", "dataset", "question", "supporting_questions", "type"],
                       "runtime_has_no_reference_fields": all(set(row) == {"id", "qid", "dataset", "question", "supporting_questions", "type"} for row in runtime),
                       "raw_decomposition_rewritten": False,
                       "prior_RAG_exposure_scope": "No actual case files found by filename in scoped formal-eval output trees; excludes remote/code/cache/artifact folders. Not a claim that all external histories are unknowable.",
                       "limitations": [
                           "The decomposer dev60 previously participated in decomposition checkpoint selection. It is exposed for the decomposer and new only to this downstream RAG comparison; not untouched holdout.",
                           "MuSiQue20 are primitive-isolated native train heldout queries, not official dev or test. Existing global corpus includes train/dev/test paragraphs.",
                           "Legacy generation explicitly passed do_sample=False and num_beams=1; source IDENTITY lacks later explicit_generation_config_v002/use_model_defaults=False compatibility. Effective later-contract equivalence is not asserted.",
                           "All60 are retained irrespective of semantic decomposition quality. Format validity is not semantic correctness.",
                           "Future root evaluation must freeze chosen300 candidate and fresh comparison rules before viewing fresh scores. No such choice is made by this script."],
                       "new_api_calls": 0, "new_model_inference": 0, "new_retrieval_calls": 0,
                       "ssh_calls": 0, "script": pin(Path(__file__))})
    target = HERE / "FRESH60.PROVENANCE.proposed.json"
    if target.exists():
        previous = read(target)
        provenance["created_utc"] = previous["created_utc"]
    unchanged_or_write(target, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return {"prepared": 60, "per_dataset": 20, "runtime": relative(runtime_path),
            "reference_records": len(references), "provenance": relative(target),
            "new_api_calls": 0, "new_retrieval_calls": 0}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
