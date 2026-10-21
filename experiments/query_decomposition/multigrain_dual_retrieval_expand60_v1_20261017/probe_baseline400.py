"""One unchanged-payload diagnostic of a known HTTP400, never rewrite around it."""

from contextlib import ExitStack
import json
import os
import sqlite3
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
from openai import OpenAI, BadRequestError

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from .prepare import OUT, ROOT, OUTPUT, write
from .run import PilotBudget


def main():
    source = OUT / "frozen_s02_musique_new12/checkpoint.sqlite3"
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        failed = db.execute("SELECT id,payload FROM requests WHERE status='failed' AND json_extract(result,'$.http_status')=400 ORDER BY rowid LIMIT 1").fetchone()
        if not failed:
            print("No known HTTP400 to inspect")
            return
        origin, raw = failed
        payload = json.loads(raw)
        evidence = db.execute("SELECT value FROM data WHERE kind='evidence' AND json_extract(value,'$.request_id')=?", (origin,)).fetchone()
        qid = json.loads(evidence[0])["qid"]
    store = Store(OUT / "baseline400_exact_probe", {"purpose": "one identical-payload retry of known HTTP400 to distinguish transient failure from permanent rejection", "origin": origin, "qid": qid, "prompt_changed": False})
    budget = PilotBudget(OUTPUT / "budget.sqlite3", 195)
    options = get_api_option(2)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, store))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=120))
            client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], max_retries=0, http_client=network))
            streaming = StreamingClient(client)
            def create(**kwargs):
                try:
                    return streaming.chat.completions.create(**kwargs)
                except BadRequestError as exc:
                    text = json.dumps(exc.body, ensure_ascii=False).replace(os.environ["DASHSCOPE_API_KEY"], "[redacted]")
                    write(OUT / "BASELINE400_PROVIDER_BODY.json", {"origin": origin, "qid": qid, "status_code": exc.status_code, "provider_body_redacted": text[:2400], "input_unchanged": True})
                    raise
            wrapped = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            transport = ResumeTransport(store, wrapped, budget, options["model"])
            transport.qid = qid
            rid, result = transport.ask("same_payload400_probe", payload["messages"][0]["content"], payload["max_tokens"])
            write(OUT / "BASELINE400_PROBE_RESULT.json", {"origin": origin, "qid": qid, "probe_request_id": rid, "status": result["status"], "input_unchanged": True})
            print(json.dumps({"probe_status": result["status"], "qid": qid}))
    finally:
        store.export()
        store.db.close()
        budget.db.close()


if __name__ == "__main__":
    main()
