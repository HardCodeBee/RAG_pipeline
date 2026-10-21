"""Complete only missing ReAct references, retaining original k10/s5 policy."""

from contextlib import ExitStack
import hashlib
import os
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.react_dense import method
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.comprehensive_eval_v1_20261015.execution import run_parallel
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from .prepare import HERE, OUT, OUTPUT, ROOT, CAMPAIGN, INPUTS, rows, load, write


def main():
    existing = {(r["dataset"], r["qid"]): r for r in rows(HERE / "historical_references_new36.frozen.jsonl") if r["method"] == "react"}
    cases = [{"qid": r["qid"], "question": r["question"]} for r in rows(HERE / "cases.frozen.jsonl") if r["dataset"] == "musique" and existing[("musique", r["qid"])]["f1"] is None]
    assert len(cases) == 8
    config = load(CAMPAIGN / "CAMPAIGN_FROZEN.json")["protocol"]["methods"]["b03_react_dense"]["default"]
    assert config == {"id": "k10_s5", "k": 10, "max_search": 5}
    resource = load(CAMPAIGN / "musique/RESOURCE_FROZEN.json")["resources"]
    options = get_api_option(2)
    store = Store(OUT / "react_musique_missing8", {"task": "user_requested_react_reference_completion", "qid": [c["qid"] for c in cases], "config": config, "model": options["model"], "resources": resource})
    write(OUT / "REACT_COMPLETION_FROZEN.json", {"cases": cases, "config": config, "method_sha256": hashlib.sha256(__import__("pathlib").Path(method.__file__).read_bytes()).hexdigest(), "prompt_examples_sha256": hashlib.sha256(__import__("pathlib").Path(method.__file__).with_name("examples.json").read_bytes()).hexdigest(), "existing52_scores_preserved": True, "reference_decomposition_used": False})
    budget = ParallelBudget(OUTPUT / "budget.sqlite3", 195)
    retrieval = Retrieval(store, resources=resource)
    install()
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, store))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3)), timeout=120))
            client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], max_retries=0, http_client=network))
            transport = ResumeTransport(store, StreamingClient(client), budget, options["model"])
            run_parallel(method, cases, retrieval, transport, store, config)
        references = {r["qid"]: r for r in rows(INPUTS / "musique/formal.references.local.jsonl")}
        score = metric("musique")
        results = []
        for case in cases:
            pred = store.get("prediction", case["qid"])
            scored = score(pred["prediction"], references[case["qid"]]) if pred and pred["status"] in {"success", "algorithm_failure"} else {"f1": None, "em": None}
            results.append({**case, "prediction": pred.get("prediction") if pred else None, "status": pred["status"] if pred else "unstarted", "searches": pred.get("searches") if pred else None, **scored})
        write(OUT / "REACT_COMPLETION_RESULTS.json", {"results": results})
        print("ReAct missing8 terminal, scored=" + str(sum(r["f1"] is not None for r in results)), flush=True)
    finally:
        if retrieval.chunks is not None:
            retrieval.chunks.close()
        store.export()
        store.db.close()
        budget.db.close()


if __name__ == "__main__":
    main()
