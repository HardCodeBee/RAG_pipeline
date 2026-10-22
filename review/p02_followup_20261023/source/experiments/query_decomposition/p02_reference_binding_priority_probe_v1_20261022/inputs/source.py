"""Freeze existing ordinary-node snapshots, without scores or new inference.

O8 stress snapshots come first, followed by the existing fewshot30 in original
order. Deduplication uses case/sid/raw prompt/history/reader, never quality.
"""
from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from p02_multigrain_timing_cachefix_v1.core.child_prompts import child
from p02_multigrain_timing_cachefix_v1.core.pipeline import passages
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload

PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
EXPERIMENTS = HERE.parent
STRESS = EXPERIMENTS / "p02_ordinary_scope_input_probe8_v1_20261022"
FEWSHOT = EXPERIMENTS / "p02_ordinary_fewshot_probe_v1_20261022"
I8 = EXPERIMENTS / "p02_ordinary_failure_challenge8_v1_20261022"
LINEAGE = EXPERIMENTS / "unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/decomposition_lineage_audit/revision002/W90_LINEAGE.jsonl"
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PRIOR_PUBLIC = EXPERIMENTS / "unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
INPUTS = HERE / "INPUTS.frozen.jsonl"
PROVENANCE = HERE / "SOURCE_PROVENANCE.frozen.jsonl"
METADATA = HERE / "SOURCE_METADATA.json"
UKEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _receipt(receipt, payload, node):
    assert receipt["status"] == "success" and receipt["finish_reason"] == "stop"
    assert not receipt.get("refusal") and isinstance(receipt.get("answer"), str) and receipt["answer"].strip()
    assert receipt["returned_model"] == payload["model"]
    assert receipt.get("requested_model", payload["model"]) == payload["model"]
    value = receipt["usage"]
    assert all(type(value.get(k)) is int and value[k] >= 0 for k in UKEYS)
    assert value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"]
    assert receipt["answer"].strip() == node["initial_answer"].strip()
    assert not node.get("sentence_format_request_id"), "Original reader output must not be substituted by formatter output"


def _build():
    stress, fewshot = rows(STRESS / "INPUTS8.frozen.jsonl"), rows(FEWSHOT / "INPUTS.frozen.jsonl")
    assert len(stress) == 8 and len(fewshot) == 30
    assert all(row["role"] == "ordinary" for row in stress + fewshot)
    assert Counter(row["group"] for row in fewshot) == {"with_history": 15, "no_history": 15}
    generation = read(PACKAGE / "config.json")["generation"]
    examples = read(PACKAGE / "prompts/child_examples.json")
    prior = read(PRIOR_PUBLIC)
    manifest_raw = MANIFEST.read_bytes()
    blob = hashlib.sha1(f"blob {len(manifest_raw)}\0".encode() + manifest_raw).hexdigest()
    assert blob == prior["manifest_git_blob_sha"]
    assert prior["remote_git_blobs_live_verified_by_related_prompt_mechanisms_agent"] is True
    manifest = json.loads(manifest_raw)
    public = {r["source"].replace("\\", "/"): {**r, "container": r.get("bundle")} for r in manifest["source_files"]}
    for archive in manifest["archives"]:
        for member in archive.get("members", []):
            public[member["source"].replace("\\", "/")] = {**member, "container": archive["bundle"]}
    lineage = {row["source_case"].replace("\\", "/"): row for row in rows(LINEAGE)}
    stress_dec = {row["probe_id"]: row for row in rows(STRESS / "DECOMPOSITION_PROOF8.frozen.jsonl")}
    stress_receipts = {row["probe_id"]: row for row in rows(I8 / "HISTORICAL_RECEIPTS8.local.frozen.jsonl")}
    dependencies = {Path(__file__).resolve(), STRESS / "INPUTS8.frozen.jsonl", FEWSHOT / "INPUTS.frozen.jsonl",
        STRESS / "DECOMPOSITION_PROOF8.frozen.jsonl", I8 / "HISTORICAL_RECEIPTS8.local.frozen.jsonl",
        STRESS / "PUBLIC_PAYLOAD_PROOF.json", FEWSHOT / "PUBLIC_PAYLOAD_PROVENANCE.json",
        LINEAGE, MANIFEST, PRIOR_PUBLIC, PACKAGE / "config.json", PACKAGE / "prompts/child_examples.json",
        PACKAGE / "core/child_prompts.py", PACKAGE / "core/pipeline.py", PACKAGE / "core/llm.py"}
    items, proofs, keyed, duplicate_memberships = [], [], {}, []
    for cohort, raw_rows in (("stress8", stress), ("fewshot30", fewshot)):
        for original_index, row in enumerate(raw_rows):
            source_name = row["source"].replace("\\", "/")
            source_path = ROOT / source_name
            assert sha(source_path) == row["source_sha256"] == public[source_name]["sha256"]
            source = read(source_path)
            ident = row.get("case_id", row.get("id"))
            assert source["id"] == ident and source["dataset"] == row["dataset"]
            assert source["question"] == row["original_question"]
            node = next(n for n in source["nodes"] if n["step_id"] == row["step_id"])
            assert node["role"] == row["role"] == "ordinary"
            assert node["request"] == row["current_question"]
            assert node["prior_complete_answers"] == row["history"]
            assert node["reader_packet"] == row["reader_packet"]
            assert node["child_prompt"] == row["control_prompt"]
            assert child(source["question"], node["step_id"], node["request"], node["prior_complete_answers"],
                         passages(node["reader_packet"]), examples, role="ordinary") == row["control_prompt"]
            actual = row["control_prompt"].split("\n\nActual task:\n", 1)[1]
            actual_current = actual.split(f"Current subquestion #{node['step_id']}:\n", 1)[1].split("\n\nRetrieved passages:", 1)[0]
            assert actual_current == node["request"]
            payload = build_payload(generation, row["control_prompt"], 128)
            assert payload == row["control_payload"]
            requested_model = payload["model"]
            if cohort == "stress8":
                evidence = stress_receipts[row["probe_id"]]
                dbpath = ROOT / evidence["source_db"]
                with closing(ro(dbpath)) as db:
                    stored = db.execute("SELECT payload,status,result FROM requests WHERE id=?", (node["answer_request_id"],)).fetchone()
                assert stored and stored[0] == canonical(payload) and stored[1] == "success"
                receipt = json.loads(stored[2])
                assert receipt == evidence["result"]
                dec = stress_dec[row["probe_id"]]
                raw_path = ROOT / dec["raw_file"]
                assert sha(raw_path) == dec["raw_sha256"]
                raw = read(raw_path)
                array = raw["parsed"]
                assert array == source["supporting_questions"] == row["supporting_questions"] == dec["actual_R02_array"]
                decproof = dict(kind="formal1500_actual_R02_v12", raw_file=rel(raw_path), raw_sha256=sha(raw_path),
                    actual_raw_identity_sha256=raw["identity_sha256"], provenance_file=rel(STRESS / "DECOMPOSITION_PROOF8.frozen.jsonl"))
                historical_db = rel(dbpath)
            else:
                dbpath = PACKAGE / "data/replay.sqlite3"
                with closing(ro(dbpath)) as db:
                    stored = db.execute("SELECT result FROM requests WHERE request_id=? AND payload=?", (node["answer_request_id"], canonical(payload))).fetchone()
                assert stored is not None
                receipt = json.loads(stored[0])
                assert receipt == row["control_result"] and row["control_source_request_id"] == node["answer_request_id"]
                lin = lineage[source_name]
                assert lin["source_case_sha256"] == row["source_sha256"] and lin["id"] == ident
                assert lin["question"] == source["question"]
                raw_path = ROOT / lin["raw_record"]
                assert sha(raw_path) == lin["raw_sha256"]
                raw, array = read(raw_path), lin["supporting_questions"]
                assert raw["parsed"] == array
                actual_array = [n["request"] for n in source["nodes"] if n["role"] == "ordinary"]
                assert actual_array == array
                decproof = dict(kind="historical_W90_actual_R02_v12_not_formal1500_decode", raw_file=rel(raw_path),
                    raw_sha256=sha(raw_path), actual_raw_identity_sha256=raw["identity_sha256"],
                    provenance_file=rel(LINEAGE), checkpoint_manifest_sha256=lin["checkpoint_manifest_sha256"],
                    actual_prompt_sha256=lin["actual_prompt_sha256"])
                historical_db = rel(dbpath)
            _receipt(receipt, payload, node)
            assert raw["id"] == ident and raw["query"] == source["question"]
            assert raw["format_valid"] and not raw["truncated"] and not raw["generation_error"]
            assert 1 <= node["step_id"] <= len(array) and node["request"] == array[node["step_id"] - 1]
            if row["history"]:
                rewrite_raw = node["raw_rewrite"]
                assert isinstance(rewrite_raw, str)
                candidate = rewrite_raw.strip()
                accepted = candidate if candidate and candidate.endswith("?") and "\n" not in candidate else node["request"]
                rewrite_basis = "Same source node raw_rewrite, native strip/nonempty/questionmark/no-newline rule; fallback raw current"
            else:
                assert node["raw_rewrite"] is None and node["rewrite_request_id"] is None
                accepted, rewrite_basis = node["request"], "No available prior QA: native rewrite bypass, retain raw current"
            assert accepted == node["rewritten_query"]
            snapshot = dict(case_id=ident, step_id=node["step_id"], control_prompt=row["control_prompt"], history=row["history"], reader_packet=row["reader_packet"])
            snapshot_sha = digest(snapshot)
            member = dict(cohort=cohort, source_index_zero_based=original_index, original_probe_id=row["probe_id"],
                          original_group=row.get("group"), source=source_name, source_sha256=row["source_sha256"])
            if snapshot_sha in keyed:
                index = keyed[snapshot_sha]
                assert items[index]["accepted_rewrite"] == accepted
                items[index]["cohort_memberships"].append(member)
                proofs[index]["cohort_memberships"].append(deepcopy(member))
                duplicate_memberships.append(dict(retained_probe_id=items[index]["probe_id"], duplicate=member))
                continue
            probe_id = row["probe_id"] + "/snapshot-" + snapshot_sha[:16]
            item = dict(probe_id=probe_id, original_probe_id=row["probe_id"], id=ident, case_id=ident,
                qid=source["qid"], dataset=row["dataset"], step_id=node["step_id"], role="ordinary",
                is_stress=cohort == "stress8", cohort=cohort, cohort_memberships=[member],
                original_question=source["question"], current_question=node["request"],
                history=deepcopy(row["history"]), reader_packet=deepcopy(row["reader_packet"]),
                supporting_questions=deepcopy(array), control_prompt=row["control_prompt"], control_payload=payload,
                accepted_rewrite=accepted, source=source_name, source_sha256=row["source_sha256"],
                node_snapshot_sha256=snapshot_sha, historical_request_id=node["answer_request_id"],
                historical_stage_usage_reference=receipt["usage"])
            proof = dict(probe_id=probe_id, case_id=ident, dataset=row["dataset"], step_id=node["step_id"],
                cohort_memberships=[deepcopy(member)], source=source_name, source_sha256=row["source_sha256"],
                node_snapshot_sha256=snapshot_sha, public_repository=prior["public_repository"],
                verified_public_commit=prior["verified_public_commit"], public_manifest=rel(MANIFEST),
                manifest_git_blob_sha=blob, public_container=public[source_name]["container"], public_source_sha256_exact=True,
                source_original_current_history_reader_prompt_exact=True, source_actual_R02_array_exact=True,
                semantic_repairs_applied=False, decomposition=decproof,
                accepted_rewrite=dict(raw_rewrite=node["raw_rewrite"], rewrite_request_id=node["rewrite_request_id"],
                    accepted_query=accepted, acceptance_rule=rewrite_basis, matches_actual_source_rewritten_query=True),
                historical_QA=dict(database=historical_db, request_id=node["answer_request_id"],
                    payload_sha256=digest(payload), result_sha256=digest(receipt), result=receipt,
                    requested_model_from_exact_payload=requested_model, original_requested_model_field_present="requested_model" in receipt,
                    matching_returned_model_stop_three_integer_usage=True),
                no_gold_native_scores_or_diagnostic_labels_used=True,
                source_publicness="Recheck of previously live-verified public manifest and exact local source bytes; not a new network verification")
            keyed[snapshot_sha] = len(items)
            items.append(item)
            proofs.append(proof)
            dependencies.update((source_path, raw_path))
    assert len(items) <= 38 and sum(item["is_stress"] for item in items) == 8
    assert len({item["probe_id"] for item in items}) == len(items)
    assert all(item["is_stress"] for item in items[:8]) and not any(item["is_stress"] for item in items[8:])
    def group_summary(values):
        return dict(nodes=len(values), distinct_queries=len({r["case_id"] for r in values}),
            datasets=dict(Counter(r["dataset"] for r in values)),
            history_nonempty=sum(bool(r["history"]) for r in values), history_empty=sum(not r["history"] for r in values),
            distinct_control_prompts=len({r["control_prompt"] for r in values}),
            accepted_rewrite_differs_current=sum(r["accepted_rewrite"] != r["current_question"] for r in values))
    source_counts = dict(stress8=group_summary(items[:8]), supplemental_after_exact_dedup=group_summary(items[8:]))
    metadata = dict(passed=True, experiment=HERE.name, selection="All original O8 stress8 then original fewshot30 in source order, exact snapshot dedup only; no scores/outputs/status-selection",
        raw_source_cohorts=dict(stress8=dict(nodes=8, datasets=dict(Counter(r["dataset"] for r in stress))),
            fewshot30=dict(nodes=30, datasets=dict(Counter(r["dataset"] for r in fewshot)), history_nonempty=15, history_empty=15)),
        prepared=group_summary(items), cohorts=source_counts, exact_duplicate_memberships_removed=duplicate_memberships,
        snapshot_identity_fields=["case_id", "step_id", "control_prompt", "history", "reader_packet"],
        distinct_snapshot_not_independent_query=True, source_actual_R02_arrays_verified=len(items),
        source_public_manifest_matches=len(items), strict_original_QA_receipts_verified=len(items),
        original_QA_answer_only_in_provenance_not_input=True,
        inference_must_use_original_prompt_with_single_confirmed_intervention_not_whole_provenance=True,
        accepted_rewrite_not_oracle=True, binding_rhs_requires_literal_prior_answer_match="Enforced by S templates; no manual names supplied by this source builder",
        public_repository=prior["public_repository"], verified_public_commit=prior["verified_public_commit"],
        public_manifest=rel(MANIFEST), manifest_git_blob_sha=blob,
        source_dependencies={rel(path): sha(path) for path in sorted(dependencies)},
        no_API_SSH_GPU_or_scoring=True, main_evaluation_remains_paused=True)
    return items, proofs, metadata


def assemble():
    return _build()[0]


def pinned_paths():
    return [ROOT / path for path in _build()[2]["source_dependencies"]]


def materialise():
    assert not (HERE / "RUN_AUTHORIZATION.json").exists(), "Do not materialise after execution authorization"
    items, proofs, metadata = _build()
    outputs = ((INPUTS, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)),
        (PROVENANCE, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in proofs)),
        (METADATA, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"))
    for path, encoded in outputs:
        if path.exists():
            assert path.read_bytes() == encoded.encode("utf-8"), "Preserve prior source bytes: " + str(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded, encoding="utf-8")
    return dict(**metadata["prepared"], files={rel(path): sha(path) for path, _ in outputs})


if __name__ == "__main__":
    print(json.dumps(materialise(), ensure_ascii=True, indent=2))
