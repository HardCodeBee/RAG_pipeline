"""Prepare a27-case challenge offline; root alone may explicitly start --run.

Reuses the frozen pilot's three goal templates and common final presentation.
No decomposer, retrieval, GPU or judge request is available in this module.
"""
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
PILOT = HERE.parent / "p02_goal_synthesis_probe_v1_20261022"
DIAG = ROOT / "outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2"
PLAN = HERE.parent / "unified_rag_eval_v1_20261021"
OUT = ROOT / "outputs/query_decomposition" / HERE.name
PUBLIC_LOCAL = ROOT / "tmp/p02_review_delivery_20261021/snapshot"
DATASETS = ("hotpotqa", "2wiki")
GOAL_CAP, FINAL_CAP = 512, 64
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022 import templates
from experiments.query_decomposition.p02_goal_synthesis_probe_v1_20261022.run import read, rows, save, sha, jsonl


def normalized(path):
    return path.replace("\\", "/")


def git_blob(path):
    raw = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve the frozen challenge")
    labels = rows(DIAG / "QUERY_LABELS_AND_OUTCOMES.jsonl")
    selected = [r for r in labels if r["ordinary_answers_all_correct"] and r["goal_answer_correct"] is False]
    assert len(selected) == 27
    assert Counter(r["dataset"] for r in selected) == {"hotpotqa": 10, "2wiki": 17}
    assert all(r["complete"] and r["ordinary_answer_count"] > 0 for r in selected)
    # No final result is used for selection, ordering, exclusion or replacement.
    selected.sort(key=lambda r: (r["dataset"], r["id"]))
    jsonl(HERE / "SELECTION.frozen.jsonl", [{k: r[k] for k in
          ("id", "qid", "dataset", "type", "source", "source_sha256",
           "ordinary_answers_all_correct", "goal_answer_correct", "review_index")} for r in selected])
    packets = {r["id"]: r for r in rows(DIAG / "STAGE_PACKETS.jsonl") if r.get("complete")}
    manifest_path = PUBLIC_LOCAL / "review/p02_20261021/SOURCE_MANIFEST.json"
    metadata = read(HERE / "PUBLIC_BLOB_METADATA.json")
    assert git_blob(manifest_path) == metadata["public_manifest_blob"]
    archives = read(manifest_path)["archives"]
    member_index = {normalized(m["source"]): (a, m) for a in archives for m in a["members"]}
    opened, inputs, history, references, proof = {}, [], [], [], []
    with ExitStack() as stack:
        for r in selected:
            source_name = normalized(r["source"])
            source_path = ROOT / source_name
            assert sha(source_path) == r["source_sha256"]
            a, m = member_index[source_name]
            assert m["sha256"] == r["source_sha256"]
            expected_archive = "review/p02_20261021/raw_cases/p02_formal1500_v1_20261021/" + r["dataset"] + ".tar.gz"
            assert a["bundle"] == expected_archive
            archive_path = PUBLIC_LOCAL / a["bundle"]
            public_blob = next(f for f in metadata["files"] if f["path"] == r["dataset"] + ".tar.gz")
            if a["bundle"] not in opened:
                assert git_blob(archive_path) == public_blob["sha"] and sha(archive_path) == a["sha256"]
                opened[a["bundle"]] = stack.enter_context(tarfile.open(archive_path, "r:gz"))
            assert opened[a["bundle"]].extractfile(source_name).read() == source_path.read_bytes()
            old = read(source_path)
            nodes = old["nodes"]
            goal = nodes[-1]
            assert old["status"] == "success" and old["finish_reason"] == "stop"
            assert goal["role"] == "goal" and goal["request"].strip() == old["question"].strip()
            assert nodes[:-1] and all(n["role"] == "ordinary" for n in nodes[:-1])
            assert len(goal["reader_packet"]["docs"]) == len(old["original_retrieval"]["docs"]) == 10
            assert packets[r["id"]]["nodes"][-1]["reader_packet"] == goal["reader_packet"]
            ordinary = [{k: n[k] for k in ("step_id", "request", "answer")} for n in nodes[:-1]]
            inputs.append({"id": r["id"], "qid": r["qid"], "dataset": r["dataset"], "type": r["type"],
                "question": old["question"], "ordinary_nodes": ordinary, "goal_step_id": goal["step_id"],
                "goal_rewritten_query": goal["rewritten_query"], "goal_reader_packet": goal["reader_packet"],
                "goal_original_packet": goal["original_retrieval"], "final_root_packet": old["original_retrieval"],
                "historical_goal_prompt": goal["child_prompt"], "source_sha256": r["source_sha256"]})
            history.append({"id": r["id"], "dataset": r["dataset"], "prediction": old["prediction"],
                "goal_answer": goal["answer"], "goal_step_id": goal["step_id"],
                "calls": old["logical_calls"], "source": source_name,
                "selection_reason": "ordinary_answers_all_correct=True and historical goal_answer_correct=False",
                "historical_final_correct": r["final_correct"], "historical_f1": r["f1"], "historical_em": r["em"]})
            ref = packets[r["id"]]["reference"]
            assert ref["id"] == r["id"] and ref["dataset"] == r["dataset"] and ref["qid"] == r["qid"]
            assert ref["question"] == old["question"] and "answer" in ref
            references.append(ref)
            proof.append({"id": r["id"], "source": source_name, "source_sha256": r["source_sha256"],
                "public_archive": a["bundle"], "public_archive_sha256": a["sha256"],
                "public_archive_git_blob": public_blob["sha"], "archive_member": source_name,
                "member_bytes_exact": True, "ordinary_QA_and_reader_from_exact_public_source": True,
                "public_commit": metadata["public_commit"]})
    jsonl(HERE / "inputs27.frozen.jsonl", inputs)
    jsonl(HERE / "historical27.local.jsonl", history)
    jsonl(HERE / "references27.local.frozen.jsonl", references)
    jsonl(HERE / "PUBLIC_SOURCE_PROOF.jsonl", proof)
    for name in ("GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json", "G1.frozen.txt", "G2.frozen.txt"):
        (HERE / name).write_bytes((PILOT / name).read_bytes())
    pinned = [HERE / n for n in ("run.py", "SELECTION.frozen.jsonl", "inputs27.frozen.jsonl",
              "historical27.local.jsonl", "references27.local.frozen.jsonl", "PUBLIC_SOURCE_PROOF.jsonl",
              "PUBLIC_BLOB_METADATA.json", "GENERATION.frozen.json", "FINAL_EXAMPLES.frozen.json",
              "G1.frozen.txt", "G2.frozen.txt")]
    pinned += [PILOT / "templates.py", PILOT / "run.py", PACKAGE / "config.json", PACKAGE / "core/final_prompts.py"]
    protocol = {"id": HERE.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "prepared_only_not_started", "scope": "Diagnostic-selected27-case frozen-prefix challenge; not independent confirmation or full fresh RAG",
        "selection": "All723 diagnostic flows filtered only by ordinary_answers_all_correct and goal_answer_correct=False; includes historical final successes and failures",
        "planned": 27, "per_dataset": {"hotpotqa": 10, "2wiki": 17},
        "historical_final_successes_included": dict(Counter(r["dataset"] for r in selected if r["final_correct"])),
        "arms": ["G0", "G1", "G2"], "goal_cap": GOAL_CAP, "final_cap": FINAL_CAP,
        "protocol_matches_pilot": "same frozen templates, model, no formatter, neutral final presentation and fixed final template/examples/readout; G0 separate from historical128-cap P02",
        "fixed_inputs": "ordinary QA, goal rewrite, dual-route reader_packet and original final root from each exact historical source",
        "gold_boundary": "References and historical outcomes are separate local files; only inputs27.frozen.jsonl enters renderers",
        "cache_and_recovery": "same exact payload stop+usage contract and shared StrictAsk maximum5 technical attempts across restarts; preserve unknown receipts and reservations; SDK/HTTP0; no successful or algorithm rerolls",
        "cost": "No hard legacy budget cap; existing ledger, endpoint and rates; per-query tokens and estimated cost, known charges and unknown reservations separate",
        "provenance": "All27 source bytes match members of already-public formal1500 archives, whose Git blobs were live-verified under ccc12b6",
        "main_evaluation_remains_paused": True, "new_API_requests_during_preparation": 0,
        "pins": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in pinned}}
    save(HERE / "PROTOCOL.frozen.json", protocol)
    return dry_run()


def check():
    protocol = read(HERE / "PROTOCOL.frozen.json")
    for name, expected in protocol["pins"].items():
        assert sha(ROOT / name) == expected, name
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def settings():
    config = read(PACKAGE / "config.json")
    assert config["generation"] == read(HERE / "GENERATION.frozen.json")
    return config, read(HERE / "FINAL_EXAMPLES.frozen.json")


def dry_run():
    check()
    config, examples = settings()
    from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages
    data = rows(HERE / "inputs27.frozen.jsonl")
    contracts = []
    for rec in data:
        assert not {"reference", "gold", "answer", "prediction", "f1", "em", "final_correct"} & set(rec)
        same = []
        for arm in templates.ARMS:
            prompt = templates.goal_prompt(rec, arm)
            assert rec["question"] in prompt and passages(rec["goal_reader_packet"]) in prompt
            assert all(n["answer"] in prompt for n in rec["ordinary_nodes"])
            assert build_payload(config["generation"], prompt, GOAL_CAP)["max_tokens"] == 512
            final, context = templates.final_prompt(rec, "SYNTHETIC SAME CONTENT", config, examples)
            assert passages(rec["final_root_packet"]) in context
            same.append(final)
            contracts.append({"id": rec["id"], "arm": arm,
                 "goal_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        assert len(set(same)) == 1
    assert len(data) == 27 and len(contracts) == 81
    jsonl(OUT / "DRYRUN_CONTRACTS.jsonl", contracts)
    result = {"passed": True, "cases": 27, "goal_conditions_checked": 81,
        "source_public_hash_proofs": 27, "references_outside_model_inputs": True,
        "new_API_requests": 0, "main_evaluation_remains_paused": True}
    save(OUT / "DRYRUN.json", result)
    return result


def execute(arms):
    check()
    authority = read(HERE / "RUN_AUTHORIZATION.json")
    assert authority.get("approved") is True and authority.get("experiment") == HERE.name
    assert any(d.get("id") == authority["decision_id"] for d in rows(PLAN / "DECISIONS.jsonl"))
    from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc
    from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import StrictAsk, NoCapBudget
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    from urllib.parse import urlsplit
    config, examples = settings()
    directory = OUT / "challenge27"
    directory.mkdir(parents=True, exist_ok=True)
    current = {}
    def event(value):
        with (directory / "trace.jsonl").open("a", encoding="utf8") as h:
            h.write(json.dumps({"utc": utc(), **current, **value}, ensure_ascii=False) + "\n")
    class NetworkEvents:
        def event(self, name, **fields): event({"event": name, **fields})
    with run_lock(directory), ExitStack() as stack:
        cache = stack.enter_context(Cache(directory / "checkpoint.sqlite3"))
        cache.put("meta", "identity", {"protocol_sha256": sha(HERE / "PROTOCOL.frozen.json"), "run_id": "challenge27"})
        budget = stack.enter_context(closing(NoCapBudget(config, directory)))
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        stack.enter_context(provider_dns(urlsplit(config["generation"]["base_url"]).hostname, NetworkEvents()))
        live = stack.enter_context(closing(StrictAsk(cache, None, config, budget, event,
                    recover_unknown=True, maximum_attempts=5, stop=lambda: (directory / "STOP_REQUESTED").exists())))
        for rec in rows(HERE / "inputs27.frozen.jsonl"):
            if (directory / "STOP_REQUESTED").exists(): break
            for arm in arms:
                key = arm + "|" + rec["id"]
                if cache.get("prediction", key) is not None: continue
                current.clear(); current.update(id=rec["id"], dataset=rec["dataset"], arm=arm)
                try:
                    gid, goal, _ = live("goal/" + key, templates.goal_prompt(rec, arm), GOAL_CAP)
                    prompt, context = templates.final_prompt(rec, goal, config, examples)
                    fid, prediction, _ = live("final/" + key, prompt, FINAL_CAP)
                    cache.put("prediction", key, {"id": rec["id"], "dataset": rec["dataset"], "arm": arm,
                       "status": "success", "prediction": prediction, "goal_text": goal,
                       "goal_request_id": gid, "final_request_id": fid, "final_prompt": prompt, "final_context": context})
                    print(json.dumps({**current, "status": "success"}), flush=True)
                except Exception as exc:
                    event({"event": "case_held", "error_type": type(exc).__name__, "error": str(exc)[:300]})
                    print(json.dumps({**current, "status": "held"}), flush=True)
    return report()


def report():
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    directory = OUT / "challenge27"
    refs = {r["id"]: r for r in rows(HERE / "references27.local.frozen.jsonl")}
    history = {r["id"]: r for r in rows(HERE / "historical27.local.jsonl")}
    with closing(sqlite3.connect((directory / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        results = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='prediction' ORDER BY rowid")]
        receipts = [{"request_id": rid, "status": status, "result": json.loads(raw) if raw else None}
                    for rid, status, raw in db.execute("SELECT id,status,result FROM requests ORDER BY rowid")]
        retired = [{"request_id": rid, "status": status, "result": json.loads(raw) if raw else None}
                   for rid, status, raw in db.execute("SELECT id,status,result FROM retired_requests ORDER BY rowid")]
    byid = {r["request_id"]: r for r in receipts}
    for r in results:
        r["native_metrics"] = metric(r["dataset"])(r["prediction"], refs[r["id"]])
        r["historical_native_metrics"] = metric(r["dataset"])(history[r["id"]]["prediction"], refs[r["id"]])
        usage = [byid[r[k]]["result"]["usage"] for k in ("goal_request_id", "final_request_id")]
        r["goal_final_input_tokens"] = sum(u["prompt_tokens"] for u in usage)
        r["goal_final_output_tokens"] = sum(u["completion_tokens"] for u in usage)
        r["goal_final_estimated_CNY"] = r["goal_final_input_tokens"] * 0.8e-6 + r["goal_final_output_tokens"] * 2e-6
    groups = {}
    for arm in templates.ARMS:
        groups[arm] = {}
        for ds in DATASETS:
            rr = [r for r in results if r["arm"] == arm and r["dataset"] == ds]
            groups[arm][ds] = {"planned": 10 if ds == "hotpotqa" else 17, "complete": len(rr),
                "F1": sum(r["native_metrics"]["f1"] for r in rr) / len(rr) if rr else None,
                "EM": sum(r["native_metrics"]["em"] for r in rr) / len(rr) if rr else None,
                "goal_final_input_tokens": sum(r["goal_final_input_tokens"] for r in rr),
                "goal_final_output_tokens": sum(r["goal_final_output_tokens"] for r in rr),
                "goal_final_estimated_CNY": sum(r["goal_final_estimated_CNY"] for r in rr)}
    indexed = {(r["arm"], r["id"]): r for r in results}
    paired = {}
    for a, b in (("G0", "G1"), ("G0", "G2"), ("G1", "G2")):
        counts = Counter()
        for ident in refs:
            if (a, ident) in indexed and (b, ident) in indexed:
                delta = indexed[b, ident]["native_metrics"]["f1"] - indexed[a, ident]["native_metrics"]["f1"]
                counts["improved" if delta > 1e-12 else "harmed" if delta < -1e-12 else "tied"] += 1
        paired[a + "_to_" + b] = dict(counts)
    usage = [r["result"]["usage"] for r in receipts + retired if r.get("result") and r["result"].get("usage")]
    result = {"scope": "All27 historical ordinary-correct/goal-wrong diagnostic cases; selected challenge, not generalization evidence",
        "status": "complete" if len(results) == 81 else "partial", "by_arm_dataset": groups,
        "paired_F1": paired, "known_receipt_CNY": sum(u["prompt_tokens"] * 0.8e-6 + u["completion_tokens"] * 2e-6 for u in usage),
        "unresolved_receipts": [r for r in receipts if r["status"] != "success"],
        "retired_receipts_preserved": retired, "historical_P02_separate": True,
        "not_pure_reasoning_causal_effect": True, "main_evaluation_remains_paused": True}
    jsonl(directory / "DETAILS.scored.local.jsonl", results)
    save(directory / "REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--report", action="store_true")
    parser.add_argument("--arms", choices=templates.ARMS, nargs="+", default=list(templates.ARMS))
    args = parser.parse_args()
    value = prepare() if args.prepare else dry_run() if args.dry_run else execute(args.arms) if args.run else report()
    print(json.dumps(value, ensure_ascii=True), flush=True)


if __name__ == "__main__": main()
