"""Isolated v8 simplification study; reuse audited API transport, not old results."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from experiments.query_decomposition.multigrain_decomposer_fresh_eval_v1_20261017 import study as previous
from experiments.query_decomposition.multigrain_decomposer_fresh_eval_v1_20261017 import review

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = previous.OUTPUT / HERE.name
PRIOR = HERE.parent / "multigrain_decomposer_fresh_eval_v1_20261017"
SEED = "20261017-v8-compact-fresh-02"
api = previous.api


def prepare():
    if (HERE / "SELECTION_FROZEN.json").exists():
        return
    prior = json.loads((PRIOR / "SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    paths = {ROOT / r["path"] for r in prior["excluded_query_files"]}
    scope = ROOT / "experiments/query_decomposition"
    paths.update(p for p in scope.rglob("*questions*.jsonl") if HERE not in p.parents)
    paths.update(p for p in scope.rglob("cases.frozen.jsonl") if HERE not in p.parents)
    excluded_ids, excluded_text, sources = set(), set(), []
    for path in sorted(paths):
        if not path.exists():
            continue
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": api.sha(path)})
        for row in api.rows(path):
            qid = row.get("qid", row.get("id"))
            if qid:
                excluded_ids.add(qid)
            question = row.get("question", row.get("query", ""))
            if isinstance(question, str) and question:
                excluded_text.add(api.normalize(question))
    train = ROOT / "outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data"
    manifest = json.loads((train / "split_manifest.json").read_text(encoding="utf-8"))
    for r in manifest["records"]:
        if r["id"] in excluded_ids or r["project_assignment"] in {"project_dev", "project_test", "reserved_test_family", "reserved_dev_family"}:
            excluded_ids.add(r.get("source", {}).get("id", ""))
    screen = previous.OUTPUT / "multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_FROZEN.json"
    excluded_ids.update(json.loads(screen.read_text(encoding="utf-8"))["ids"])
    native = ROOT / "data/multihop_comprehensive_v1_20261015"
    sources_native = {
        "hotpotqa": ROOT / "analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl",
        "2wiki": native / "2wiki/native/dev.json",
        "musique": native / "musique/native/musique_ans_v1.0_dev.jsonl",
    }
    pool, seen = [], set()
    rank = lambda text: hashlib.sha256((SEED + text).encode()).hexdigest()
    for dataset, path in sources_native.items():
        iterator = previous.native_array(path) if dataset == "2wiki" else api.rows(path)
        for row in iterator:
            qid, question = row.get("_id", row.get("id")), row["question"]
            normalized = api.normalize(question)
            if qid in excluded_ids or normalized in excluded_text or normalized in seen:
                continue
            seen.add(normalized)
            kind = str(len(row["question_decomposition"])) + "-hop" if dataset == "musique" else row["type"]
            pool.append({"dataset": dataset, "qid": qid, "question": question, "type": kind})
    used, texts = set(), set()
    for phase in ("dev", "holdout"):
        chosen = []
        for dataset, quotas in previous.QUOTAS.items():
            for kind, count in quotas.items():
                candidates = sorted((r for r in pool if r["dataset"] == dataset and r["type"] == kind and r["qid"] not in used), key=lambda r: rank(phase + r["qid"]))
                assert len(candidates) >= count
                chosen.extend(candidates[:count])
        chosen.sort(key=lambda r: rank("order" + phase + r["qid"]))
        for index, row in enumerate(chosen, 1):
            row.update(index=index, sample_group="v8_simplify_fresh_" + phase)
            assert row["qid"] not in used and api.normalize(row["question"]) not in texts
            used.add(row["qid"])
            texts.add(api.normalize(row["question"]))
        previous.write_rows(HERE / (phase + ".questions.frozen.jsonl"), chosen)
    api.immutable_json(HERE / "SELECTION_FROZEN.json", {
        "utc": previous.utc(), "seed": SEED, "queries_each_phase": 60,
        "quotas_each_phase": previous.QUOTAS,
        "excluded_query_files": sources, "excluded_ids": len(excluded_ids),
        "excluded_normalized_texts": len(excluded_text),
        "native_sources": [{"path": str(p.relative_to(ROOT)), "sha256": api.sha(p)} for p in sources_native.values()],
        "eligible_counts": dict(Counter(r["dataset"] for r in pool)),
        "selection_uses_answers_outputs_or_scores": False,
        "dev_holdout_disjoint_ids_and_normalized_text": True,
        "scope": "query-only; no answer generation, retrieval, new model or training",
        "rubric_sha256": api.sha(previous.OLD / "PILOT_REVIEW_RULES.txt"),
        "optimization": "v8 vs v10 compression first, at most two further targeted revisions on dev60; no inspection of holdout text/output until candidate frozen",
        "selection_rule": "among prompts shorter than v8 with dev pass count >= v8, maximize whole-query pass count, then fewer UTF8 bytes; otherwise keep v8. Full planned60 primary, common available pairs secondary",
        "target": "54/60 pass (90%); not a promised result. Uncertain and unavailable retained. Never redefine rubric to reach target",
        "confirmation": "rerun v8 and selected candidate on sealed fresh holdout60 once, no tuning on confirmation",
        "review": "literal F/S/C/U current-Codex authored review, not independent human gold; no reference answers/docs/plans",
        "global_cap_cny": 200, "stage_max_new_calls": 360,
        "availability": "SDK retry disabled; unknown submissions retained, never replayed",
        "freshness_limit": "excludes known question IDs and normalized exact text; not a complete semantic paraphrase or pretraining audit",
    })


def wire():
    previous.HERE, previous.OUT = HERE, OUT
    review.HERE, review.OUT = HERE, OUT


def record_change(version, parent, purpose):
    prompt, source = HERE / ("TRAINING_PROMPT_" + version + ".txt"), HERE / ("TRAINING_PROMPT_" + parent + ".txt")
    text, old = prompt.read_text(encoding="utf-8"), source.read_text(encoding="utf-8")
    api.immutable_json(HERE / ("CHANGE_" + version + ".json"), {
        "utc": previous.utc(), "version": version, "parent": parent,
        "purpose": purpose, "prompt_sha256": api.sha(prompt),
        "parent_sha256": api.sha(source), "utf8_bytes": len(text.encode()),
        "parent_utf8_bytes": len(old.encode()),
        "v8_utf8_bytes": len((HERE / "TRAINING_PROMPT_v8.txt").read_bytes()),
        "generation": "one same-model call; no external post-editing, evidence or factual answers",
    })
    diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), text.splitlines(keepends=True), fromfile=parent, tofile=version))
    (HERE / ("DIFF_" + parent + "_to_" + version + ".txt")).write_text(diff, encoding="utf-8")


def run(phase, versions):
    """Keep three workers occupied without replaying unknown calls or changing payloads."""
    if phase == "holdout":
        assert (HERE / "FINAL_CANDIDATE_FROZEN.json").exists()
    tasks = []
    for arm in versions:
        cases, prompt, out, protocol = previous.prepare_run(phase, arm)
        for case in cases:
            if not (out / "cases" / arm / case["dataset"] / (case["qid"] + ".json")).exists():
                tasks.append((case, arm, prompt, out, protocol))
    tasks.sort(key=lambda t: (t[0]["index"], (versions.index(t[1]) + t[0]["index"]) % len(versions)))
    owner = previous.Store(OUT / ("transport_window_" + phase), {"task": "fresh_query_only_prompt_comparison", "phase": phase})
    completed, failures, consecutive_unknown = 0, [], 0
    try:
        with ExitStack() as stack:
            stack.enter_context(previous.temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            network = stack.enter_context(previous.httpx.Client(transport=previous.httpx.HTTPTransport(retries=0, limits=previous.httpx.Limits(max_connections=3, max_keepalive_connections=0)), timeout=90))
            options = previous.get_api_option(2)
            raw = stack.enter_context(previous.OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=90, max_retries=0, http_client=network))
            client = previous.StreamingClient(raw)
            stack.enter_context(previous.provider_dns(urlsplit(options["base_url"]).hostname, owner))
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending, iterator = {}, iter(tasks)
                def fill():
                    while len(pending) < 3 and not failures and consecutive_unknown < 6:
                        task = next(iterator, None)
                        if task is None:
                            break
                        c, arm, prompt, out, protocol = task
                        pending[pool.submit(api.run_one, c, arm, protocol, client, pilot_out=out, arm_prompts={arm: str(prompt)})] = task
                fill()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        c, arm, _, out, _ = pending.pop(future)
                        try:
                            future.result()
                            consecutive_unknown = 0
                        except Exception as error:
                            path = out / "cases" / arm / c["dataset"] / (c["qid"] + ".json")
                            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                            if value.get("status") == "unknown":
                                consecutive_unknown += 1
                            else:
                                failures.append(type(error).__name__)
                        completed += 1
                    if completed % 6 == 0 or failures or not pending:
                        print(json.dumps({"phase": phase, "versions": versions, "completed_this_run": completed, "planned_unsubmitted": len(tasks), "hard_failures": failures, "utc": previous.utc()}), flush=True)
                    fill()
                if failures or consecutive_unknown >= 6:
                    raise RuntimeError("Transport stopped; persisted submissions retained without replay: " + str(failures))
    finally:
        owner.db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--phase", choices=["dev", "holdout"], default="dev")
    p.add_argument("--versions", nargs="+", default=["v8", "v10"])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=60)
    args = p.parse_args()
    if args.prepare:
        prepare()
        print("Frozen new dev60 + sealed holdout60; known exposed queries excluded.")
    wire()
    if args.run:
        run(args.phase, args.versions)
    if args.show:
        review.show(args.phase, args.versions, args.start, args.end)
    if args.summarize:
        review.summarize(args.phase, args.versions)
