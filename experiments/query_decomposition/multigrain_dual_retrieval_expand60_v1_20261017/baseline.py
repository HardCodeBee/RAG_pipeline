"""Run unchanged frozen S02 only for twelve new MuSiQue cases."""

from contextlib import ExitStack
import hashlib
import json
import os
import sqlite3
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, digest
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import method
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.comprehensive_eval_v1_20261015.execution import run_parallel
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from .prepare import HERE, OUT, CAMPAIGN, ROOT, OUTPUT, INPUTS, rows, load, write


def main():
    cases = [{"qid": r["qid"], "question": r["question"]} for r in rows(HERE / "cases.frozen.jsonl") if r["dataset"] == "musique"]
    assert len(cases) == 12
    config = load(CAMPAIGN / "CAMPAIGN_FROZEN.json")["protocol"]["methods"]["m00_s02_child4_faithful"]["default"]
    config["helper_positions"] = load(INPUTS / "musique/formal.positions.json")
    resources = load(CAMPAIGN / "musique/RESOURCE_FROZEN.json")["resources"]
    options = get_api_option(2)
    assert options["model"] == load(HERE / "protocol.json")["model"]
    store = Store(OUT / "frozen_s02_musique_new12", {"task": "expand60_unchanged_s02_reference", "query_ids": [c["qid"] for c in cases], "model": options["model"], "config": config, "resources": resources})
    original = CAMPAIGN / "musique/formal/b00_rag_top10_v1/fixed/checkpoint.sqlite3"
    if original.exists():
        with sqlite3.connect(original.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for case in cases:
                packets = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='retrieval' AND json_extract(value,'$.qid')=?", (case["qid"],))]
                for packet in packets:
                    if packet["query"] == case["question"] and packet["k"] == 10:
                        store.put("retrieval", digest([case["qid"], case["question"], 10, resources]), packet)
                        break
    remote = {r["id"]: r for r in rows(OUTPUT / "comprehensive_v1_20261015/remote1500/musique.decompositions.jsonl")}
    for case in cases:
        raw = remote[case["qid"]]
        assert raw["query"] == case["question"]
        qdmr = None if raw.get("generation_error") or raw.get("parse_error") else raw["parsed_target"]
        store.put("decomposition", case["qid"], {"qid": case["qid"], "qdmr": qdmr, "raw": raw})
    write(OUT / "BASELINE_MUSIQUE_FROZEN.json", {"ids": [c["qid"] for c in cases], "config": config, "source_method": str(method.HERE), "method_sha256": hashlib.sha256((method.HERE / "method.py").read_bytes()).hexdigest(), "actual_remote4b_outputs_used": True, "reference_plans_used": False})
    install()
    retrieval = Retrieval(store, resources=resources)
    budget = ParallelBudget(OUTPUT / "budget.sqlite3", 195)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, store))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3)), timeout=120))
            client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=120, max_retries=0, http_client=network))
            transport = ResumeTransport(store, StreamingClient(client), budget, options["model"])
            for start in range(0, 12, 6):
                run_parallel(method, cases[start:start + 6], retrieval, transport, store, config)
                print(json.dumps({"baseline_musique_completed": store.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]}), flush=True)
        reference = {r["qid"]: r for r in rows(INPUTS / "musique/formal.references.local.jsonl")}
        score = metric("musique")
        results = []
        for case in cases:
            pred = store.get("prediction", case["qid"])
            scored = score(pred["prediction"], reference[case["qid"]]) if pred and pred["status"] == "success" else {"f1": None, "em": None}
            results.append({**case, "type": reference[case["qid"]]["type"], "prediction": pred.get("prediction") if pred else None, "status": pred["status"] if pred else "unstarted", **scored})
        write(OUT / "BASELINE_MUSIQUE_RESULTS.json", {"results": results})
    finally:
        if retrieval.chunks is not None:
            retrieval.chunks.close()
        store.export()
        store.db.close()
        budget.db.close()


if __name__ == "__main__":
    main()
