"""Read-only source projection for the unchanged, previously exposed W/L90.

No credentials, API, SSH, model inference, scoring or source mutation. Historical
full-flow receipt replay validates the reference; it is never a fresh outcome.
Only the five inference fields leave this module's local provenance boundary.
"""
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments/query_decomposition"
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
BASE = EXPERIMENTS / "p02_noformatter_quick90_v1_20261021"
L_INPUTS = EXPERIMENTS / "p02_goal_native_role_fixed90_v1_20261022/INPUTS90.frozen.jsonl"
W_INPUTS = EXPERIMENTS / "p02_final_target_check_fixed90_v1_20261022/INPUTS90.frozen.jsonl"
CONTROL = ROOT / "outputs/query_decomposition/hotpot_formal_eval_v1/p02_frozen_no_refs_quick90_v1_20261021/control90.jsonl"
REPLAY = PACKAGE / "data/replay.sqlite3"
LINEAGE = EXPERIMENTS / "unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/decomposition_lineage_audit/revision002/W90_LINEAGE.jsonl"
LINEAGE_SUMMARY = LINEAGE.with_name("SUMMARY.json")
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PRIOR_PUBLIC = EXPERIMENTS / "unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
PREFLIGHT = ROOT / "outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
INITIAL_ENGINEERING6 = HERE / "INITIAL_ENGINEERING6.frozen.json"
ENGINEERING_HASH_PREFIX = "p02-shared-context-fixed90-v1|20261022|"
DATASETS = ("hotpotqa", "2wiki", "musique")
FIVE = ("id", "qid", "dataset", "question", "supporting_questions")
UKEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
CHECKPOINT_MANIFEST_SHA = "728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92"
PROMPT_SHA = "508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def usage_ok(u):
    return (all(type(u.get(k)) is int and u[k] >= 0 for k in UKEYS)
            and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"])


def fixed_sample_ids(data=None):
    """Engineering prefix of the SAME90, never a quality-screening cohort."""
    if data is None:
        data = rows(L_INPUTS)
    return [r["id"] for ds in DATASETS for r in sorted(
        (r for r in data if r["dataset"] == ds),
        key=lambda r: (hashlib.sha256((ENGINEERING_HASH_PREFIX + r["id"]).encode()).hexdigest(), r["id"]))[:2]]


def pinned_paths():
    """Direct local proof dependencies; no credentials or global paid ledger."""
    paths = [Path(__file__), L_INPUTS, W_INPUTS, CONTROL, REPLAY, LINEAGE,
        LINEAGE_SUMMARY, MANIFEST, PRIOR_PUBLIC, PREFLIGHT,
        BASE / "runtime90.frozen.jsonl", BASE / "SOURCE_BINDINGS.frozen.jsonl",
        BASE / "references90.local.frozen.jsonl", BASE / "COHORT.frozen.json",
        L_INPUTS.with_name("PROTOCOL.frozen.json"), W_INPUTS.with_name("PROTOCOL.frozen.json"),
        PACKAGE / "config.json", EXPERIMENTS / "multigrain_4b_coverage_retrain_v3_20261018/v12.txt"]
    for row in rows(LINEAGE):
        paths.extend((ROOT / row["source_case"], ROOT / row["raw_record"]))
    return sorted(set(paths), key=lambda q: rel(q))


def public_index():
    raw = MANIFEST.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert blob == "06c67be3c2091ef4066366b8c0c87c2b4d1ece17"
    manifest = json.loads(raw)
    index = {r["source"].replace("\\", "/"): {**r, "container": r["bundle"]} for r in manifest["source_files"]}
    for archive in manifest["archives"]:
        for r in archive.get("members", []):
            index[r["source"].replace("\\", "/")] = {**r, "container": archive["bundle"]}
    prior = read(PRIOR_PUBLIC)
    return index, dict(public_repository=prior["public_repository"],
        verified_public_commit=prior["verified_public_commit"], manifest_git_blob_sha=blob,
        manifest=rel(MANIFEST), manifest_sha256=sha(MANIFEST),
        verification="Local match against previously live-verified public manifest; no new network fetch")


def assemble():
    from . import resource_contract
    left, right, runtime = rows(L_INPUTS), rows(W_INPUTS), rows(BASE / "runtime90.frozen.jsonl")
    controls, bindings = rows(CONTROL), rows(BASE / "SOURCE_BINDINGS.frozen.jsonl")
    assert len(left) == len(right) == len(runtime) == len(controls) == len(bindings) == 90
    ids = [r["id"] for r in left]
    assert len(set(ids)) == 90
    assert all([r["id"] for r in values] == ids for values in (right, runtime, controls, bindings))
    assert Counter(r["dataset"] for r in left) == {ds: 30 for ds in DATASETS}
    # Fix all90 membership and the engineering prefix using only IDs/arrays.
    data = [safe_case({k: deepcopy(r[k]) for k in FIVE}) for r in left]
    assert all(set(r) == set(FIVE) for r in data)
    for clean, r in zip(data, runtime):
        assert clean == {k: r[k] for k in FIVE}
    assert sum(not r["supporting_questions"] for r in data) == 6
    engineering = fixed_sample_ids(data)
    assert len(engineering) == len(set(engineering)) == 6 and set(engineering) <= set(ids)
    lineage = {r["id"]: r for r in rows(LINEAGE)}
    assert set(lineage) == set(ids)
    public, publicproof = public_index()
    config = read(PACKAGE / "config.json")
    costs, provenance, complete_calls = [], [], 0
    with closing(sqlite3.connect(REPLAY.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for clean, l, w, control, binding in zip(data, left, right, controls, bindings):
            i = clean["id"]
            source_path = binding["source"].replace("\\", "/")
            source_sha = sha(ROOT / source_path)
            assert source_sha == binding["source_sha256"] == l["source_sha256"] == w["source_sha256"] == public[source_path]["sha256"]
            assert l["source"].replace("\\", "/") == w["source"].replace("\\", "/") == source_path
            source = read(ROOT / source_path)
            t = control["trajectory"]
            assert {k: t[k] for k in FIVE} == clean
            assert source["id"] == i and source["prediction"] == t["prediction"] == control["prediction"]
            assert [n["request"] for n in source["nodes"] if n["role"] == "ordinary"] == clean["supporting_questions"]
            line = lineage[i]
            assert line["source_case"] == source_path and line["source_case_sha256"] == source_sha
            assert line["supporting_questions"] == clean["supporting_questions"] and all(line["checks"].values())
            assert line["checkpoint_manifest_sha256"] == CHECKPOINT_MANIFEST_SHA
            assert line["actual_prompt_sha256"] == PROMPT_SHA
            raw = read(ROOT / line["raw_record"])
            assert sha(ROOT / line["raw_record"]) == line["raw_sha256"]
            assert raw["id"] == i and raw["query"] == clean["question"] and raw["dataset"] == clean["dataset"]
            assert raw["parsed"] == clean["supporting_questions"] and raw["raw_output"] == line["raw_output"]
            assert raw["format_valid"] and not raw["truncated"] and not raw["generation_error"]
            assert raw["identity_sha256"] == line["raw_identity_sha256"]
            # actual_prompt_sha256 in lineage identifies the template, not each filled prompt.
            assert raw["actual_prompt"].startswith((EXPERIMENTS / "multigrain_4b_coverage_retrain_v3_20261018/v12.txt").read_text(encoding="utf-8").rstrip())
            calls = {c["stage"]: c for c in control["calls"]}
            assert len(calls) == len(control["calls"])
            byquery = {}
            for node in t["nodes"]:
                for key in ("original_retrieval", "rewritten_retrieval"):
                    packet = node[key]
                    if packet is None:
                        continue
                    assert packet["qid"] == clean["qid"]
                    assert len(packet["docs"]) == len({d["doc_id"] for d in packet["docs"]}) == 10
                    if packet["query"] in byquery:
                        assert packet == byquery[packet["query"]]
                    byquery[packet["query"]] = packet
            seen = []

            def ask(stage, prompt, maximum, allow_final_truncation=False):
                call = calls[stage]
                payload = build_payload(config["generation"], prompt, maximum)
                row = db.execute("SELECT payload,result FROM requests WHERE request_id=?", (call["request_id"],)).fetchone()
                assert row and row[0] == canonical(payload), (i, stage, "historical_wire_mismatch")
                receipt = json.loads(row[1])
                assert receipt["status"] == "success" and receipt["finish_reason"] == "stop"
                assert receipt["returned_model"] == payload["model"] and receipt.get("requested_model") in (None, payload["model"])
                assert usage_ok(receipt.get("usage") or {}) and receipt["usage"] == call["usage"]
                assert receipt["answer"].strip() and not receipt.get("refusal")
                seen.append(stage)
                return call["request_id"], receipt["answer"].strip(), receipt["finish_reason"]

            replay = Pipeline(config).execute(clean, lambda q: deepcopy(byquery[q]), ask)
            assert replay == t and len(seen) == len(calls)
            complete_calls += len(seen)
            assert binding["final_request_id"] == t["request_id"] == source["request_id"] == w["historical_final_request_id"]
            cost = sum(c["usage"]["prompt_tokens"] * .8e-6 + c["usage"]["completion_tokens"] * 2e-6 for c in control["calls"])
            costs.append(dict(id=i, dataset=clean["dataset"], reference_CNY=cost,
                reference_kind="same_query_complete_historical_flow", historical_quality_available=True,
                final_prediction_reference_only=t["prediction"], source=source_path,
                reference_source=source_path, source_sha256=source_sha,
                strict_native_messages_receipts_and_retrieval_verified=True,
                reference_round=source.get("arm"), legacy_requested_model_bound_from_exact_payload=True))
            provenance.append(dict(id=i, source=source_path, source_sha256=source_sha,
                public_container=public[source_path]["container"], five_fields_match_W_L_runtime_control=True,
                supporting_questions_exact_raw_R02=True, manual_or_semantic_repairs_applied=False,
                raw_file=line["raw_record"], raw_sha256=line["raw_sha256"], raw_array_format_valid=True,
                checkpoint_manifest_sha256=CHECKPOINT_MANIFEST_SHA, prompt_sha256=PROMPT_SHA,
                lineage_source=rel(LINEAGE), lineage_source_sha256=sha(LINEAGE),
                full_native_trajectory_and_wire_receipts_verified=True, strict_logical_stages=len(seen),
                old_request_ids=[c["request_id"] for c in control["calls"]],
                raw_source_is_formal1500=False, same_effective_decode_contract_as_formal1500_claimed=False,
                prior_exposure="Repeated W/L90 development; not untouched or independent confirmation"))
    assert complete_calls == 532
    # Labels/old predictions never determine membership, order or the engineering gate.
    refindex = {r["id"]: r for r in rows(BASE / "references90.local.frozen.jsonl")}
    assert set(refindex) == set(ids)
    refs = [deepcopy(refindex[i]) for i in ids]
    manifests = read(PREFLIGHT)["manifests"]
    resources = {ds: dict(identity=resource_contract.identity(manifests[ds]["manifest"], manifests[ds]["sha256"]),
        contract=resource_contract.contract(manifests[ds]["manifest"], manifests[ds]["sha256"])) for ds in DATASETS}
    publicproof.update(source_cases_checked=90, source_cases=provenance,
        five_field_arrays_exact_public_source_ordinary_requests_and_raw_R02=True,
        public_formal_runtime_or_same_generation_batch_claimed=False,
        prior_same90_authorized_API_source_inputs_reused=True,
        raw_records_or_all_future_retrieved_text_prior_publication_not_claimed=True,
        dataset_corpus="Three existing frozen public-benchmark portable BGE resources",
        old_preflight_not_a_new_remote_readiness_assertion=True,
        no_credentials_or_reference_answers_in_five_field_inference=True,
        L_inputs_sha256=sha(L_INPUTS), W_inputs_sha256=sha(W_INPUTS),
        original_runtime90_sha256=sha(BASE / "runtime90.frozen.jsonl"),
        original_source_bindings_sha256=sha(BASE / "SOURCE_BINDINGS.frozen.jsonl"),
        all90_preserved_including_empty6=True, initial_engineering6_ids=engineering,
        engineering_prefix_only="First6 of same committed90; protocol-only gate then remaining84, never an F1 screen")
    return data, costs, provenance, refs, resources, {}, publicproof


def growth_estimate():
    """Exact frozen UTF8 bodies; recorded old usage is only a token proxy."""
    values = []
    for c in rows(CONTROL):
        t = c["trajectory"]
        calls = {r["stage"]: r for r in c["calls"]}
        history_bytes = history_tokens = 0
        steps = []
        for node in t["nodes"]:
            u = calls["legacy/answer/" + str(node["step_id"])]["usage"]
            prompt_bytes = len(node["child_prompt"].encode("utf-8"))
            steps.append(dict(step_id=node["step_id"], role=node["role"],
                reset_prompt_UTF8_bytes=prompt_bytes, reset_recorded_prompt_tokens=u["prompt_tokens"],
                shared_content_UTF8_bytes=history_bytes + prompt_bytes,
                shared_prompt_tokens_old_receipt_sum_proxy=history_tokens + u["prompt_tokens"]))
            history_bytes += prompt_bytes + len(node["answer"].encode("utf-8"))
            history_tokens += u["prompt_tokens"] + u["completion_tokens"]
        values.append(dict(id=c["id"], dataset=c["dataset"], ordinary_nodes=len(t["supporting_questions"]), QA_steps=len(steps),
            max_shared_content_UTF8_bytes=max(r["shared_content_UTF8_bytes"] for r in steps),
            max_prompt_tokens_old_receipt_sum_proxy=max(r["shared_prompt_tokens_old_receipt_sum_proxy"] for r in steps),
            reset_QA_prompt_tokens=sum(r["reset_recorded_prompt_tokens"] for r in steps),
            shared_QA_prompt_tokens_old_receipt_sum_proxy=sum(r["shared_prompt_tokens_old_receipt_sum_proxy"] for r in steps), steps=steps))

    def summary(part):
        reset = sum(r["reset_QA_prompt_tokens"] for r in part)
        shared = sum(r["shared_QA_prompt_tokens_old_receipt_sum_proxy"] for r in part)
        return dict(queries=len(part), QA_steps_per_arm=sum(r["QA_steps"] for r in part),
            maximum_QA_steps=max(r["QA_steps"] for r in part),
            maximum_shared_content_UTF8_bytes=max(r["max_shared_content_UTF8_bytes"] for r in part),
            queries_above128000bytes=sum(r["max_shared_content_UTF8_bytes"] > 128000 for r in part),
            queries_above131072bytes=sum(r["max_shared_content_UTF8_bytes"] > 131072 for r in part),
            maximum_prompt_tokens_old_receipt_sum_proxy=max(r["max_prompt_tokens_old_receipt_sum_proxy"] for r in part),
            reset_QA_prompt_tokens=reset, shared_QA_prompt_tokens_old_receipt_sum_proxy=shared,
            shared_to_reset_QA_input_token_proxy_ratio=shared / reset)
    initial = set(fixed_sample_ids())
    return dict(scope="Frozen historical material simulation only; no inference, new tokenizer or true future bound",
        content="Every earlier raw child_prompt plus accepted answer, followed by current raw child_prompt. Never earlier serialized fullwire or transcript recursively.",
        bytes_exclude_new_flat_labels_and_chat_role_framing=True,
        token_proxy_not_new_tokenization_or_provider_usage=True,
        dynamic_future_retrieval_and_answers_may_change_material_sizes=True,
        all90=summary(values), by_dataset={ds: summary([r for r in values if r["dataset"] == ds]) for ds in DATASETS},
        initial_engineering6=summary([r for r in values if r["id"] in initial]), per_query=values)


def materialise():
    """Write only this new module's owned, deterministic preparation artifacts."""
    data, costs, proof, refs, resources, proxies, publicproof = assemble()
    files = {"INPUTS90.jsonl": data, "HISTORICAL_REFERENCES90.local.jsonl": costs,
        "SOURCE_PROVENANCE90.jsonl": proof, "references90.local.jsonl": refs}
    for name, value in files.items():
        content = "".join(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n" for r in value)
        q = HERE / name
        if q.exists():
            assert q.read_text(encoding="utf-8") == content, "Preserve prepared source artifact: " + name
        else:
            q.write_text(content, encoding="utf-8")
    engineering = dict(ids=fixed_sample_ids(data), hash_prefix=ENGINEERING_HASH_PREFIX,
        rule="Dataset order hotpotqa,2wiki,musique; smallest2 SHA256(prefix+canonicalID) per dataset, tiesID",
        cases=6, conditions=18, committed_full_cohort=90, committed_conditions=270,
        only_engineering_gate=True, quality_or_final_F1_not_a_gate=True, remaining84_precommitted=True,
        no_decomposition_or_case_exclusions=True)
    for name, value in {"RESOURCE_CONTRACTS.json": resources, "HISTORICAL_COST_PROXIES.local.json": proxies,
        "PUBLIC_INPUT_PROVENANCE.json": publicproof, INITIAL_ENGINEERING6.name: engineering,
        "FROZEN_CONTEXT_GROWTH_ESTIMATE.json": growth_estimate()}.items():
        content = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        q = HERE / name
        if q.exists():
            assert q.read_text(encoding="utf-8") == content, "Preserve prepared source artifact: " + name
        else:
            q.write_text(content, encoding="utf-8")
    return dict(cases=len(data), planned_conditions=270, initial_engineering6=engineering["ids"],
        ordinary_nodes=sum(len(r["supporting_questions"]) for r in data), QA_nodes_per_arm=266,
        strict_historical_coverage=90, strict_wire_receipts=532, new_API_SSH_GPU=0)


if __name__ == "__main__":
    print(json.dumps(materialise(), ensure_ascii=True))
