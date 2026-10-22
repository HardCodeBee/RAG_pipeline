"""No-history first-node followup; prepare/dry-run never open the provider.

Reuse the frozen parent runner in a separate CLI process with isolated paths.
This changes no parent source, protocol, checkpoint, or running process.
"""
from collections import Counter
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3

from . import run as engine
from .shared_runtime import atomic_json, complete_receipt, utc

ROOT, PACKAGE, COHORT = engine.ROOT, engine.PACKAGE, engine.COHORT
HERE = Path(__file__).resolve().parent / "first_node_followup"
OUT = engine.OUT / "first_node_followup"
PROTOCOL = HERE / "PROTOCOL.frozen.json"
INPUTS = HERE / "PROBES.frozen.jsonl"
MANIFEST = ROOT / "tmp/p02_review_delivery_20261021/snapshot/review/p02_20261021/SOURCE_MANIFEST.json"
PUBLIC_PROOF = ROOT / "experiments/query_decomposition/unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/PUBLIC_PAYLOAD_PROVENANCE.json"
MOTIVATION = engine.OUT / "minimal4_loss_review/REVIEW.md"
read, rows, sha, digest = engine.read, engine.rows, engine.sha, engine.digest


def public_sources():
    """Check the local manifest bytes against its verified public Git blob."""
    proof = read(PUBLIC_PROOF)
    raw = MANIFEST.read_bytes()
    git_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert git_sha == proof["manifest_git_blob_sha"], "Public manifest identity mismatch"
    assert proof["remote_git_blobs_live_verified_by_related_prompt_mechanisms_agent"] is True
    manifest = json.loads(raw)
    sources = {r["source"].replace("\\", "/"): {**r, "public_container": r.get("bundle")}
               for r in manifest["source_files"]}
    for archive in manifest["archives"]:
        for member in archive.get("members", []):
            sources[member["source"].replace("\\", "/")] = {
                **member, "public_container": archive["bundle"]}
    return proof, git_sha, sources


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    HERE.mkdir(parents=True, exist_ok=True)
    config = read(PACKAGE / "config.json")
    runtime = {r["id"]: r for r in rows(COHORT / "runtime90.frozen.jsonl")}
    proof, manifest_git_sha, public = public_sources()
    eligible = []
    with closing(sqlite3.connect((PACKAGE / "data/replay.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as seed:
        for binding in rows(COHORT / "SOURCE_BINDINGS.frozen.jsonl"):
            case = runtime[binding["id"]]
            if not case["supporting_questions"]:
                continue
            source = ROOT / binding["source"]
            assert sha(source) == binding["source_sha256"]
            raw_case = read(source)
            assert raw_case["status"] in ("success", "success_truncated")
            assert raw_case["id"] == case["id"]
            node = raw_case["nodes"][0]
            assert node["step_id"] == 1 and node["role"] == "ordinary"
            assert node["prior_complete_answers"] == []
            assert node["request"] == case["supporting_questions"][0]
            prompt = engine.render(raw_case, node)
            assert prompt == node["child_prompt"]
            payload = engine.build_payload(config["generation"], prompt, 128)
            source_rid = node["answer_request_id"]
            saved = seed.execute("SELECT result FROM requests WHERE request_id=? AND payload=?",
                                 (source_rid, engine.canonical(payload))).fetchone()
            assert saved is not None
            receipt = json.loads(saved[0])
            assert complete_receipt(receipt)
            assert receipt["answer"].strip() == node["initial_answer"].strip()
            assert not node.get("sentence_format_request_id")
            key = hashlib.sha256(f"{engine.SEED}|{case['id']}|first_ordinary_no_history".encode()).hexdigest()
            eligible.append({"probe_id": case["id"] + "/ordinary/1", "case_id": case["id"],
                "qid": case["qid"], "dataset": case["dataset"], "role": "ordinary", "step_id": 1,
                "selection_sha256": key, "source": binding["source"], "source_sha256": binding["source_sha256"],
                "original_question": raw_case["question"], "current_question": node["request"],
                "history": [], "reader_packet": node["reader_packet"], "control_prompt": prompt,
                "control_payload": payload, "control_payload_sha256": digest(payload),
                "control_source_request_id": source_rid, "control_result": receipt})
    groups = {ds: sorted([r for r in eligible if r["dataset"] == ds],
                         key=lambda r: r["selection_sha256"])[:5] for ds in engine.DATASETS}
    assert all(len(g) == 5 for g in groups.values())
    selected = [groups[ds][i] for i in range(5) for ds in engine.DATASETS]
    crosswalk = []
    for item in selected:
        entry = public[item["source"].replace("\\", "/")]
        assert entry["sha256"] == item["source_sha256"]
        crosswalk.append({"probe_id": item["probe_id"], "source": item["source"],
            "source_sha256": item["source_sha256"], "public_container": entry["public_container"],
            "original_child_prompt_and_packet_are_public_source_fields": True})
    INPUTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf8")
    public_record = {"public_repository": proof["public_repository"],
        "verified_public_commit": proof["verified_public_commit"], "public_manifest": proof["public_manifest"],
        "manifest_git_blob_sha": manifest_git_sha, "verified_sources": len(crosswalk), "cases": crosswalk,
        "requested_destination": config["generation"]["base_url"],
        "payload_excludes": ["credentials", "gold answers", "local paths", "billing ledger"],
        "new_instructions_are_nonsensitive": True, "new_provider_submissions": 0}
    atomic_json(HERE / "PUBLIC_PAYLOAD_PROVENANCE.json", public_record)
    paths = [Path(__file__).resolve(), engine.HERE / "run.py", engine.HERE / "shared_runtime.py",
        PACKAGE / "config.json", PACKAGE / "core/child_prompts.py", PACKAGE / "core/pipeline.py",
        PACKAGE / "core/llm.py", PACKAGE / "core/cache.py", PACKAGE / "core/runtime.py",
        PACKAGE / "prompts/child_examples.json", ROOT / "scripts/cli_support.py",
        engine.HERE.parent / "p02_formal1500_v1_20261021/network.py", MANIFEST,
        MOTIVATION, INPUTS, HERE / "PUBLIC_PAYLOAD_PROVENANCE.json",
        COHORT / "runtime90.frozen.jsonl", COHORT / "SOURCE_BINDINGS.frozen.jsonl"]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    pins.update({r["source"]: r["source_sha256"] for r in selected})
    protocol = {"id": "p02_first_ordinary_no_history_followup_v1_20261022", "created_utc": utc(),
        "authority": engine.AUTHORITY,
        "scope": "Fixed-input no-history first-ordinary negative control; not complete RAG, not formal quality evidence",
        "motivation": "Three of four minimal4 F1 harmed cases first changed at an ordinary node with empty history. Two were locally unsupported-identity abstentions followed by coverage changes, one a final exclusion failure; the remaining case was a same-entity score-sensitive name extension. These posthoc observations do not label the prompt changes causally harmful.",
        "selection": "Within the same fixed90, retain nonempty complete RAG controls; SHA256('20261022|case_id|first_ordinary_no_history') ascending, first5 per dataset; do not consult scores or reference answers",
        "planned_nodes": 15, "planned_candidate_tasks": 30, "per_dataset": 5,
        "required_history": [], "required_role": "ordinary", "required_step_id": 1,
        "control": "Same source case/node original answer_request_id AND full payload with stop/usage, exact source hash",
        "changes": {"delete_resolve": {"from": engine.RESOLVE, "to": ""},
                    "describe_binding": {"from": engine.RESOLVE, "to": engine.DESCRIPTIVE}},
        "fixed": "All remaining instructions and few-shot examples (including their #1 references), actual question, empty history wording, reader content/order, model and128 output limit",
        "interpretation": "An effect without actual history concerns global instruction/example priming or generation variation; it is not automatically an entity-binding effect. One historical control draw does not estimate same-input generation noise.",
        "execution": "Reuse parent StrictAsk/gate in a separate CLI process with isolated checkpoint, accounting namespace, hold files and reports; do not execute downstream retrieval/goal/final or formatter",
        "maximum_actual_attempts": 5, "SDK_retries": 0, "HTTP_transport_retries": 0,
        "unknown_with_any_receipt": "hold, do not resend", "unknown_without_receipt": "bounded delegated recovery retaining every old receipt/reservation",
        "shared_gate": "outputs/query_decomposition/p02_prompt_research_v1_20261022/provider_gate",
        "destination": config["generation"]["base_url"], "control_replay_only": True,
        "candidate_seed_reuse": False, "no_gold_or_score_based_selection": True,
        "all15_source_hashes_verified_in_public_manifest": True, "main_evaluation_remains_paused": True,
        "new_provider_submissions_at_preparation": 0, "pins": pins}
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    protocol = read(PROTOCOL)
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    return protocol


def dry_run():
    check()
    proof, manifest_git_sha, public = public_sources()
    probes = rows(INPUTS)
    assert len(probes) == 15
    assert Counter(r["dataset"] for r in probes) == {ds: 5 for ds in engine.DATASETS}
    tasks = []
    for item in probes:
        assert item["history"] == [] and item["step_id"] == 1 and item["role"] == "ordinary"
        assert digest(item["control_payload"]) == item["control_payload_sha256"]
        assert complete_receipt(item["control_result"])
        entry = public[item["source"].replace("\\", "/")]
        assert entry["sha256"] == item["source_sha256"]
        raw_case = read(ROOT / item["source"])
        assert raw_case["nodes"][0]["child_prompt"] == item["control_prompt"]
        assert raw_case["nodes"][0]["reader_packet"] == item["reader_packet"]
        assert raw_case["nodes"][0]["answer_request_id"] == item["control_source_request_id"]
        for variant in engine.VARIANTS:
            prompt = engine.prompt_variant(item["control_prompt"], variant)
            payload = engine.build_payload(read(PACKAGE / "config.json")["generation"], prompt, 128)
            assert {k: v for k, v in payload.items() if k != "messages"} == {
                k: v for k, v in item["control_payload"].items() if k != "messages"}
            tasks.append({"probe_id": item["probe_id"], "variant": variant, "payload_sha256": digest(payload)})
    assert len(tasks) == 30
    value = {"passed": True, "utc": utc(), "planned_nodes": 15, "planned_candidate_tasks": 30,
        "all_history_empty": True, "all_first_ordinary_nodes": True,
        "same_node_original_request_id_stop_usage": True, "only_one_sentence_changed": True,
        "public_manifest_git_sha": manifest_git_sha, "public_source_hashes_verified": 15,
        "no_gold_or_score_selection": True, "new_provider_submissions": 0,
        "new_retrieval_or_GPU_jobs": 0, "isolated_checkpoint": str((OUT / "checkpoint.sqlite3").relative_to(ROOT)),
        "protocol_sha256": sha(PROTOCOL), "main_evaluation_remains_paused": True, "tasks": tasks}
    atomic_json(OUT / "DRY_RUN.json", value)
    return {k: v for k, v in value.items() if k != "tasks"}


def report(cache=None):
    check()
    database = OUT / "checkpoint.sqlite3"
    if cache is None and not database.exists():
        return {"planned_candidate_tasks": 30, "complete": 0}
    db = cache.db if cache is not None else sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        outcomes = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    finally:
        if cache is None:
            db.close()
    complete = [r for r in outcomes if r["status"] == "success"]
    finished = {r["task_id"] for r in outcomes}
    held = [read(p) for p in (OUT / "held").glob("*.json")]
    held = [r for r in held if r["task_id"] not in finished]
    groups = {}
    for ds in (*engine.DATASETS, "overall"):
        for variant in engine.VARIANTS:
            group = [r for r in complete if r["variant"] == variant and (ds == "overall" or r["dataset"] == ds)]
            groups[ds + "/" + variant] = {"planned": 15 if ds == "overall" else 5,
                "complete": len(group), "text_changed": sum(r["answer"] != r["control_answer"] for r in group),
                "input_tokens": sum(r["usage"]["prompt_tokens"] for r in group),
                "output_tokens": sum(r["usage"]["completion_tokens"] for r in group)}
    config = read(PACKAGE / "config.json")
    with closing(sqlite3.connect(Path(config["budget"]["ledger"]).resolve().as_uri() + "?mode=ro", uri=True)) as ledger:
        namespace = "p02_prompt_research_v1_20261022/" + str(OUT.resolve())
        known, reserve = ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?", (namespace,)).fetchone()
    value = {"utc": utc(), "scope": "No-history first-ordinary component followup; no native final F1/EM",
        "planned_candidate_tasks": 30, "complete": len(complete), "sections": groups,
        "terminal_algorithm_failures": sum(r["status"] == "algorithm_failure" for r in outcomes),
        "held_technical_tasks": len(held), "actual_returned_usage_CNY": known, "unresolved_reserved_CNY": reserve,
        "main_evaluation_remains_paused": True}
    atomic_json(OUT / "REPORT.json", value)
    return value


def run(limit=None):
    check()
    # Only this separate CLI process receives new globals; no source is edited.
    engine.OUT, engine.PROTOCOL, engine.INPUTS = OUT, PROTOCOL, INPUTS
    engine.ROLES = ("ordinary",)
    engine.report = report
    return engine.run(limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "run", "report"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.action == "prepare":
        engine.emit(prepare())
    elif args.action == "dry-run":
        engine.emit(dry_run())
    elif args.action == "report":
        engine.emit(report())
    else:
        engine.emit(run(args.limit))


if __name__ == "__main__":
    main()
