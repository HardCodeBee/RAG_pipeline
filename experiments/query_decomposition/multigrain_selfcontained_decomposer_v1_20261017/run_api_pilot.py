"""Small paired, query-only prompt diagnostic; no training or downstream QA."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import (
    ROOT, OUTPUT, Store, canonical, digest, utc, write_json,
)
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget


HERE = Path(__file__).resolve().parent
OUT = OUTPUT / "multigrain_decomposer_prompt_pilot_v1_20261017"
SOURCE = ROOT / "experiments/query_decomposition/multigrain_dual_retrieval_expand60_v1_20261017/cases.frozen.jsonl"
ARMS = {"v1": "TRAINING_PROMPT_v1.txt", "v2": "TRAINING_PROMPT_v2.txt"}
SUBCAP = 1.0


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_json(path, value):
    if path.exists():
        if canonical(json.loads(path.read_text(encoding="utf-8"))) != canonical(value):
            raise ValueError("Frozen pilot artifact differs: " + path.name)
    else:
        write_json(path, value)


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip(" \t\n?!？！.")


class PilotBudget(ParallelBudget):
    def __init__(self, path, cap=200, *, pilot_root=None):
        super().__init__(path, cap)
        self.pilot_prefix = str((OUT if pilot_root is None else pilot_root).relative_to(OUTPUT))

    def reserve(self, request_id, namespace, amount):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges").fetchone()[0]
            local = self.db.execute("SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?", (self.pilot_prefix + "%",)).fetchone()[0]
            if total + amount > 195 or local + amount > SUBCAP:
                raise RuntimeError("Protected global budget or one-yuan pilot cap reached")
            if self.db.execute("SELECT 1 FROM charges WHERE id=?", (request_id,)).fetchone():
                raise RuntimeError("Existing submission liability cannot be replayed")
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def prepare():
    selected = [{k: r[k] for k in ("qid", "dataset", "question", "type", "sample_group")}
                for r in rows(SOURCE) if r["sample_group"] == "fresh36_type_stratified_random"]
    assert len(selected) == 36 and len({r["qid"] for r in selected}) == 36
    assert all(sum(r["dataset"] == ds for r in selected) == 12 for ds in ("hotpotqa", "2wiki", "musique"))
    # Pair and interleave datasets before any request; no outcome-based selection.
    buckets = [[r for r in selected if r["dataset"] == ds] for ds in ("hotpotqa", "2wiki", "musique")]
    selected = [buckets[j][i] for i in range(12) for j in range(3)]
    frozen = OUT / "cases.frozen.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    if frozen.exists():
        assert frozen.read_text(encoding="utf-8") == text
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_text(text, encoding="utf-8")
    protocol = {
        "task": "paired_query_only_decomposition_prompt_diagnostic",
        "questions": 36, "datasets": {"hotpotqa": 12, "2wiki": 12, "musique": 12},
        "arms": {arm: {"prompt_file": filename, "sha256": sha(HERE / filename)} for arm, filename in ARMS.items()},
        "source_cases": str(SOURCE.relative_to(ROOT)), "cases_sha256": sha(frozen),
        "selection": "all 36 prior fresh36 diagnostic cases; unchanged IDs, no QA-result selection",
        "prior_exposure": "These questions have been used and inspected in previous local diagnostics; not an independent test set.",
        "inputs": "system=exact versioned prompt; user=original question only; no labels, documents, answers or qrels",
        "model": get_api_option(2)["model"], "temperature": 0, "enable_thinking": False,
        "max_tokens": 1024, "concurrency": 3, "planned_calls": 72,
        "budget_cap_cny": 200, "protected_global_stop_cny": 195, "pilot_subcap_cny": SUBCAP,
        "transport": "streaming with usage; SDK retries disabled; unknown submissions never retried",
        "comparison": "bundle of prompt edits; not a single-sentence causal ablation",
        "semantic_rubric": ["faithfulness_and_unknowns", "self_containment", "supporting_information_coverage", "distinct_usefulness"],
        "original_question_policy": {"v1": "forbidden in model output", "v2": "allowed if naturally generated; no extra mandatory goal"},
        "runtime_compilation": "remove exact normalized original-query duplicates and duplicate strings; append exact original once at end; no semantic rewriting",
        "review": "Current Codex direct inspection of literal query/output pairs; not independent human gold or blind multi-model review.",
        "fourB_claim": "API performance does not establish trained 4B accuracy or downstream RAG gains.",
    }
    immutable_json(OUT / "PROTOCOL_FROZEN.json", protocol)
    return selected, protocol


def parse_output(answer, original):
    try:
        value = json.loads(answer)
        valid = isinstance(value, list) and all(isinstance(s, str) and s.strip() for s in value)
    except (json.JSONDecodeError, TypeError):
        value, valid = None, False
    result = {"array_of_nonempty_strings": valid, "questions": value if valid else None}
    if valid:
        normalized = [normalize(s) for s in value]
        original_key = normalize(original)
        kept, seen = [], set()
        for question, key in zip(value, normalized):
            if key != original_key and key not in seen:
                kept.append(question)
                seen.add(key)
        result.update(
            explicit_references=any(re.search(r"#\d+|\bprevious (?:answer|question)\b", s, re.I) for s in value),
            empty=not value, original_exact_count=normalized.count(original_key),
            duplicate_count=len(normalized) - len(set(normalized)),
            compiled_questions=kept + [original],
        )
    return result


def run_one(case, arm, protocol, client, *, pilot_out=None, arm_prompts=None, user_content=None):
    destination = OUT if pilot_out is None else pilot_out
    prompt_files = ARMS if arm_prompts is None else arm_prompts
    directory = destination / "api" / arm / case["dataset"] / case["qid"]
    payload = {"model": protocol["model"],
               "messages": [{"role": "system", "content": (HERE / prompt_files[arm]).read_text(encoding="utf-8")},
                            {"role": "user", "content": case["question"] if user_content is None else user_content}],
               "temperature": 0.0, "max_tokens": protocol["max_tokens"],
               "extra_body": {"enable_thinking": False}}
    identity = {"arm": arm, "qid": case["qid"], "protocol_sha256": sha(destination / "PROTOCOL_FROZEN.json")}
    store = Store(directory, identity)
    budget = PilotBudget(OUTPUT / "budget.sqlite3", 200, pilot_root=destination)
    request_id = digest([store.namespace, payload])
    try:
        existing = store.db.execute("SELECT status,result FROM requests WHERE id=?", (request_id,)).fetchone()
        if existing:
            if existing[0] in ("unknown", "in_flight"):
                raise RuntimeError("Unknown submission retained; no replay")
            result = json.loads(existing[1])
        else:
            budget.wait_for_start()
            liability = digest([request_id, 0])
            reserve = (sum(len(m["content"].encode("utf-8")) for m in payload["messages"]) + 256) * .8e-6 + payload["max_tokens"] * 2e-6
            budget.reserve(liability, store.namespace, reserve)
            store.db.execute("INSERT INTO requests VALUES(?,?,?,'in_flight',NULL)", (request_id, "decompose", canonical(payload)))
            cursor = store.db.execute("INSERT INTO attempts(request_id,started,status) VALUES(?,?,'in_flight')", (request_id, utc()))
            attempt_id = cursor.lastrowid
            store.db.commit()
            start = time.perf_counter()
            try:
                response = client.chat.completions.create(**payload)
                raw = response.model_dump(mode="json")
                choice, usage = response.choices[0], raw.get("usage") or {}
                result = {"status": "success", "answer": choice.message.content or "", "finish_reason": choice.finish_reason,
                          "returned_model": response.model, "usage": usage, "raw_response": raw,
                          "latency_seconds": time.perf_counter() - start}
                if usage:
                    budget.settle(liability, usage.get("prompt_tokens", 0) * .8e-6 + usage.get("completion_tokens", 0) * 2e-6, "returned_usage")
                if response.model != protocol["model"]:
                    result["status"] = "model_identity_mismatch"
            except Exception as error:
                causes = []
                cause = error.__cause__
                while cause is not None and len(causes) < 5:
                    causes.append(type(cause).__name__)
                    cause = cause.__cause__
                code = getattr(error, "status_code", None)
                body = getattr(error, "body", {})
                body = body.get("error", body) if isinstance(body, dict) else {}
                error_code = body.get("code") if isinstance(body, dict) else None
                error_code = error_code if isinstance(error_code, str) and re.fullmatch(r"[\w.-]{1,100}", error_code) else None
                result = {"status": "failed" if code else "unknown", "http_status": code,
                          "failure_class": type(error).__name__, "error_code": error_code,
                          "failure_causes": causes,
                          "latency_seconds": time.perf_counter() - start}
                if code in (400, 401, 403, 404, 422, 429):
                    budget.settle(liability, 0.0, "http_failed_not_returned_usage")
                if code == 429:
                    budget.note_throttle()
            store.db.execute("UPDATE attempts SET status=?,result=? WHERE id=?", (result["status"], canonical(result), attempt_id))
            store.db.execute("UPDATE requests SET status=?,result=? WHERE id=?", (result["status"], canonical(result), request_id))
            store.db.commit()
        value = {**case, "arm": arm, "request_id": request_id, **{k: v for k, v in result.items() if k != "raw_response"}}
        if result["status"] == "success":
            value["mechanical"] = parse_output(result["answer"], case["question"])
        write_json(destination / "cases" / arm / case["dataset"] / (case["qid"] + ".json"), value)
        store.export(full=True)
        if result["status"] in ("unknown", "model_identity_mismatch") or result.get("http_status") in (401, 403, 404, 429):
            raise RuntimeError("Pilot stopped after sanitized transport failure: " + str(result))
        return {"arm": arm, "qid": case["qid"], "status": result["status"],
                "questions": len(value.get("mechanical", {}).get("questions") or []),
                "schema": value.get("mechanical", {}).get("array_of_nonempty_strings")}
    finally:
        store.db.close()
        budget.db.close()


def summarize(*, pilot_out=None, arm_prompts=None):
    destination = OUT if pilot_out is None else pilot_out
    prompt_files = ARMS if arm_prompts is None else arm_prompts
    outputs = [json.loads(p.read_text(encoding="utf-8")) for p in (destination / "cases").glob("*/*/*.json")]
    metrics = {}
    for arm in prompt_files:
        values = [r for r in outputs if r["arm"] == arm]
        good = [r for r in values if r["status"] == "success" and r.get("finish_reason") == "stop"]
        metrics[arm] = {"planned": 36, "returned": len(values), "complete_success": len(good),
                        "valid_array": sum(r.get("mechanical", {}).get("array_of_nonempty_strings", False) for r in good),
                        "reference_outputs": sum(r.get("mechanical", {}).get("explicit_references", False) for r in good),
                        "empty_outputs": sum(r.get("mechanical", {}).get("empty", False) for r in good),
                        "exact_original_outputs": sum(bool(r.get("mechanical", {}).get("original_exact_count")) for r in good),
                        "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in values),
                        "input_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in values)}
    with sqlite3.connect((OUTPUT / "budget.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        actual, reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) FROM charges WHERE namespace LIKE ?", (str(destination.relative_to(OUTPUT)) + "%",)).fetchone()
    result = {"utc": utc(), "metrics": metrics, "pilot_usage_list_price_cny": actual, "unknown_reserved_cny": reserved,
              "actual_bill_verified": False, "semantic_review": "not inferred from mechanical checks"}
    write_json(destination / "MECHANICAL_SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-unsubmitted", action="store_true")
    args = parser.parse_args()
    cases, protocol = prepare()
    if not args.run:
        print(json.dumps(summarize(), ensure_ascii=False), flush=True)
        return
    tasks = [(case, arm) for index, case in enumerate(cases) for arm in (list(ARMS) if index % 2 == 0 else list(reversed(ARMS)))]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if args.continue_unsubmitted:
        # Retain all prior results, including the unknown request; never replay it.
        tasks = [(case, arm) for case, arm in tasks if not (OUT / "cases" / arm / case["dataset"] / (case["qid"] + ".json")).exists()]
    options = get_api_option(2)
    owner = Store(OUT / "transport", {"task": protocol["task"], "protocol_sha256": sha(OUT / "PROTOCOL_FROZEN.json")})
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=0)), timeout=90))
            raw_client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=90, max_retries=0, http_client=network))
            client = StreamingClient(raw_client)
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, owner))
            with ThreadPoolExecutor(max_workers=3) as pool:
                for index in range(0, len(tasks), 3):
                    futures = [pool.submit(run_one, case, arm, protocol, client) for case, arm in tasks[index:index+3]]
                    errors = []
                    for future in futures:
                        try:
                            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                        except Exception as error:
                            errors.append(type(error).__name__ + ": " + str(error))
                    print(json.dumps(summarize(), ensure_ascii=False), flush=True)
                    if errors:
                        write_json(OUT / "TRANSPORT_STOP.json", {"utc": utc(), "errors": errors})
                        raise RuntimeError("Pilot stopped; review retained transport state before resuming")
    finally:
        owner.db.close()


if __name__ == "__main__":
    main()
