"""Continue2Wiki/Mu with native single-query retrieval; generation unchanged."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import os
import sqlite3
from threading import Lock
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
from openai import OpenAI, BadRequestError

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, digest
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from . import run as frozen
from .prepare import HERE, OUT, OUTPUT, ROOT, CAMPAIGN, INPUTS, PILOT_OUT, rows, load, write


LOCK = Lock()


def guarded(case, protocol, owner, retrieval, client, model, examples, positions):
    errors = []
    try:
        return frozen.run_case(case, protocol, owner, retrieval, client, model, examples, positions)
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "reason": str(exc)[:250]})
    dual = OUT / "cases" / case["dataset"] / "dual_original_rewritten" / (case["qid"] + ".json")
    if not dual.exists():
        try:
            # Only choose the remaining arm. This flag affects orchestration,
            # not prompts, retrieval, history or any response selection.
            frozen.run_case(case, {**protocol, "matched_single_control": False}, owner, retrieval, client, model, examples, positions)
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "reason": str(exc)[:250]})
    with sqlite3.connect((owner.directory / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        attempts = [dict(zip(("id", "stage", "status", "result"), r)) for r in db.execute("SELECT DISTINCT r.id,r.stage,r.status,r.result FROM requests r JOIN data e ON e.kind='evidence' AND json_extract(e.value,'$.request_id')=r.id WHERE json_extract(e.value,'$.qid')=? AND r.status!='success'", (case["qid"],))]
    missing = [arm for arm in ("single_original", "dual_original_rewritten") if not (OUT / "cases" / case["dataset"] / arm / (case["qid"] + ".json")).exists()]
    item = {"dataset": case["dataset"], "qid": case["qid"], "missing_conditions": missing, "status": "provider_or_transport_missing", "errors": errors, "failed_requests": attempts, "changed_prompt_to_bypass_rejection": False}
    with LOCK:
        path = OUT / "TERMINAL_FAILURES.json"
        records = load(path) if path.exists() else []
        records.append(item)
        write(path, records)
    return {"dataset": case["dataset"], "qid": case["qid"], "terminal_missing": missing}


def main():
    protocol = load(HERE / "protocol.json")
    cases = rows(HERE / "cases.frozen.jsonl")
    examples = load(ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json")
    mapping = {"hotpotqa": "api/hotpotqa", "2wiki": "api/2wiki_native_v2", "musique": "api/musique_native_v2"}
    write(OUT / "API_STORES.json", mapping)
    write(OUT / "NATIVE_CONTINUATION_FROZEN.json", {"reason": "2Wiki batch verification swapped two near-tied Top10 documents; no2Wiki generation requests had been sent", "failed_batch_archive": "api/2wiki", "continue_datasets": ["2wiki", "musique"], "native_single_query_retrieval": True, "generation_core": str(frozen.__file__), "generation_core_sha256": hashlib.sha256(__import__("pathlib").Path(frozen.__file__).read_bytes()).hexdigest(), "continuation_sha256": hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(), "hotpot12_preserved": True, "per_case_missingness_does_not_cancel_unaffected_cases": True})
    options = get_api_option(2)
    install()
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3)), timeout=120))
        client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], max_retries=0, http_client=network))
        streaming = StreamingClient(client)
        def create(**payload):
            try:
                return streaming.chat.completions.create(**payload)
            except BadRequestError as exc:
                text = json.dumps(exc.body, ensure_ascii=False).replace(os.environ["DASHSCOPE_API_KEY"], "[redacted]")
                with LOCK:
                    write(OUT / "provider_errors" / (digest(payload) + ".json"), {"http_status": exc.status_code, "redacted_body": text[:2400], "payload_hash": digest(payload), "prompt_changed": False})
                raise
        audited = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        for dataset in ("2wiki", "musique"):
            selected = [c for c in cases if c["dataset"] == dataset]
            resource = load(CAMPAIGN / dataset / "RESOURCE_FROZEN.json")["resources"]
            owner = Store(OUT / mapping[dataset], {"task": "unchanged_expand60_native_retrieval_continuation", "dataset": dataset, "protocol": protocol, "resources": resource})
            historical = OUT / "frozen_s02_musique_new12/checkpoint.sqlite3" if dataset == "musique" else CAMPAIGN / dataset / "formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3"
            with sqlite3.connect(historical.resolve().as_uri() + "?mode=ro", uri=True) as db:
                for case in selected:
                    packets = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='retrieval' AND json_extract(value,'$.qid')=?", (case["qid"],))]
                    packet = next(p for p in packets if p["query"] == case["question"] and p["k"] == 10)
                    owner.put("retrieval", digest([case["qid"], case["question"], 10, resource]), packet)
            retrieval = Retrieval(owner, resources=resource)
            try:
                with provider_dns(urlsplit(options["base_url"]).hostname, owner):
                    retrieval.load()
                    controls = []
                    for path in sorted((PILOT_OUT / "cases" / dataset / "single_original").glob("*.json"))[:2]:
                        old = load(path)["nodes"][0]["original_retrieval"]
                        new = retrieval.retrieve(old["qid"], old["query"], 10)
                        controls.append({"qid": old["qid"], "same_top10_order": [d["doc_id"] for d in old["docs"]] == [d["doc_id"] for d in new["docs"]]})
                    write(OUT / f"NATIVE_EQUIVALENCE_{dataset}.json", controls)
                    assert all(c["same_top10_order"] for c in controls)
                    positions = load(INPUTS / dataset / "formal.positions.json")
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures = [pool.submit(guarded, case, protocol, owner, retrieval, audited, options["model"], examples, positions) for case in selected]
                        for future in as_completed(futures):
                            print(json.dumps(future.result()), flush=True)
            finally:
                if retrieval.chunks is not None:
                    retrieval.chunks.close()
                owner.export()
                owner.db.close()
    write(OUT / "GENERATION_TERMINAL.json", {"new_cases": 36, "original_hotpot12_preserved": True, "remaining24_attempted": True, "failures": str(OUT / "TERMINAL_FAILURES.json")})


if __name__ == "__main__":
    main()
