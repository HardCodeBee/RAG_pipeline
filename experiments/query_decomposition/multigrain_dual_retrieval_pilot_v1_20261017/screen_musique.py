"""Run unmodified frozen S02 on a fixed MuSiQue prefix to select diagnostics."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import method
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.comprehensive_eval_v1_20261015.execution import run_parallel
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric


HERE = Path(__file__).resolve().parent
OUT = OUTPUT / "multigrain_dual_retrieval_pilot_v1_20261017"
CAMPAIGN = OUTPUT / "comprehensive_v1_20261015/campaign_api_v1"
INPUTS = ROOT / "experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/musique"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    cases = rows(INPUTS / "formal.questions.jsonl")[:24]
    config = load(CAMPAIGN / "CAMPAIGN_FROZEN.json")["protocol"]["methods"]["m00_s02_child4_faithful"]["default"]
    config["helper_positions"] = load(INPUTS / "formal.positions.json")
    resource = load(CAMPAIGN / "musique/RESOURCE_FROZEN.json")["resources"]
    options = get_api_option(2)
    identity = {"task": "unmodified_frozen_s02_musique_screening", "query_ids": [r["qid"] for r in cases], "model": options["model"], "config": config, "resources": resource, "screening_only": True, "formal_campaign_outputs_modified": False}
    # The sandbox's DNS failures occurred before provider TCP/TLS connection.
    # Preserve the original attempts and reservations; resume in a new network
    # execution namespace only after recording this pre-submission evidence.
    prior = OUT / "musique_baseline_screening/checkpoint.sqlite3"
    if prior.exists():
        failures = []
        with sqlite3.connect(prior.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for request_id, raw in db.execute("SELECT request_id,result FROM attempts"):
                result = json.loads(raw)
                causes = [c["type"] for c in result.get("failure_causes", [])]
                assert result.get("status") == "unknown" and result.get("http_status") is None
                assert "URLError" in causes and "gaierror" in causes
                failures.append({"request_id": request_id, "failure_causes": result["failure_causes"], "latency_seconds": result["latency_seconds"]})
        write(OUT / "DNS_PRE_SUBMISSION_RECONCILIATION.json", {
            "reason": "Provider name and fallback-resolver DNS failed before TCP/TLS; no model request could be transmitted.",
            "attempts": failures,
            "original_checkpoint_or_liabilities_deleted": False,
            "network_probe": "Fresh DNS and verified-hostname TLS1.3 succeeded outside the restricted sandbox.",
            "new_namespace": "musique_baseline_screening_network_v2",
            "model_prompts_and_retrieval_algorithm_changed": False,
        })
    store = Store(OUT / "musique_baseline_screening_network_v2", identity)
    if prior.exists():
        with sqlite3.connect(prior.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for key, value in db.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
                store.put("retrieval", key, json.loads(value))
    remote = {r["id"]: r for r in rows(OUTPUT / "comprehensive_v1_20261015/remote1500/musique.decompositions.jsonl")}
    for case in cases:
        r = remote[case["qid"]]
        assert r["query"] == case["question"]
        target = None if r.get("generation_error") or r.get("parse_error") else r["parsed_target"]
        store.put("decomposition", case["qid"], {"qid": case["qid"], "qdmr": target, "raw": r})
    write(OUT / "MUSIQUE_SCREENING_FROZEN.json", {"ids": [r["qid"] for r in cases], "selection": "first24 in existing frozen formal500 order, before new outputs", "purpose": "obtain actual S02 successes/failures; not a completed formal500 evaluation"})
    print(json.dumps({"screening": "start", "dataset": "musique", "questions": len(cases)}), flush=True)
    install()
    retrieval = Retrieval(store, resources=resource)
    budget = ParallelBudget(OUTPUT / "budget.sqlite3", 200)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, store))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15)), timeout=120))
            client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=120, max_retries=0, http_client=network))
            transport = ResumeTransport(store, StreamingClient(client), budget, options["model"])
            for start in range(0, 24, 8):
                if sum(budget.summary()[k] for k in ("returned_usage_list_price_cny", "unreconciled_reserved_cny")) >= 195:
                    raise RuntimeError("Protected global API budget reached")
                run_parallel(method, cases[start:start + 8], retrieval, transport, store, config)
                store.export()
                print(json.dumps({"screening_block_done": start // 8, "completed": store.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]}), flush=True)
        refs = {r["qid"]: r for r in rows(INPUTS / "formal.references.local.jsonl")}
        score = metric("musique")
        results = []
        for case in cases:
            pred = store.get("prediction", case["qid"])
            value = score(pred["prediction"], refs[case["qid"]]) if pred and pred["status"] == "success" else {"f1": None, "em": None}
            results.append({**case, "type": refs[case["qid"]]["type"], "status": pred["status"] if pred else "unstarted", "prediction": pred.get("prediction") if pred else None, **value})
        write(OUT / "MUSIQUE_SCREENING_RESULTS.json", {"results": results, "budget": budget.summary()})
        print(json.dumps({"screening": "complete", "success_f1_1": sum(r["f1"] == 1 for r in results), "failure_f1_0": sum(r["f1"] == 0 for r in results)}), flush=True)
    finally:
        if retrieval.chunks is not None:
            retrieval.chunks.close()
        store.export()
        store.db.close()
        budget.db.close()


if __name__ == "__main__":
    main()
