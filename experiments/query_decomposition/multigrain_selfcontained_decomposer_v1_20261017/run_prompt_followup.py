"""One changed prompt per arm, using the exact frozen 36-query diagnostic."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from .run_api_pilot import HERE, OUT as CONTROL, rows, sha, immutable_json, run_one, summarize
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, utc, write_json
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient


def prepare(version):
    assert version in ("v3", "v4", "v5")
    prompt = HERE / f"TRAINING_PROMPT_{version}.txt"
    if not prompt.exists():
        raise RuntimeError("Requested prompt has not been authored and recorded")
    destination = OUTPUT / f"multigrain_decomposer_prompt_{version}_20261017"
    destination.mkdir(parents=True, exist_ok=True)
    cases_file = destination / "cases.frozen.jsonl"
    original = (CONTROL / "cases.frozen.jsonl").read_bytes()
    if cases_file.exists():
        assert cases_file.read_bytes() == original
    else:
        cases_file.write_bytes(original)
    cases = rows(cases_file)
    before = {"v3": "v2", "v4": "v3", "v5": "v3"}[version]
    protocol = {
        "task": "same36_query_only_prompt_followup", "arm": version,
        "prompt_file": str(prompt.relative_to(ROOT)), "prompt_sha256": sha(prompt),
        "parent_prompt": before, "parent_prompt_sha256": sha(HERE / f"TRAINING_PROMPT_{before}.txt"),
        "cases_sha256": sha(cases_file), "control_cases_sha256": sha(CONTROL / "cases.frozen.jsonl"),
        "questions": 36, "planned_calls": 36, "datasets": {"hotpotqa": 12, "2wiki": 12, "musique": 12},
        "selection": "exact unchanged original 36 questions; original frozen v2 responses reused for comparison",
        "model": get_api_option(2)["model"], "temperature": 0, "enable_thinking": False,
        "max_tokens": 1024, "concurrency": 3,
        "inputs": "system=exact versioned prompt; user=original query only; no answers, labels, documents, qrels or prior outputs",
        "component_change": {"v3": "item self-containment only", "v4": "distinct usefulness and input-provided identity only", "v5": "three short illustrative examples added to v3; no rule changes"}[version],
        "semantic_rubric_source": str((HERE / "PILOT_REVIEW_RULES.txt").relative_to(ROOT)),
        "semantic_rubric_sha256": sha(HERE / "PILOT_REVIEW_RULES.txt"),
        "old_control_review_sha256": sha(CONTROL / "AUTHORED_SEMANTIC_REVIEW.jsonl"),
        "original_question_policy": "unchanged from v2: naturally generated original-equivalent questions allowed; no mandatory goal",
        "transport": "same streaming client, proper system/user roles, no SDK retry, unknown submissions never replayed",
        "budget_cap_cny": 200, "protected_global_stop_cny": 195, "pilot_subcap_cny": 1,
        "claim_boundary": "targeted tuning on already inspected cases; no independent-test, trained-4B or RAG-F1 claim",
    }
    immutable_json(destination / "PROTOCOL_FROZEN.json", protocol)
    assert protocol["cases_sha256"] == protocol["control_cases_sha256"] and len(cases) == 36
    return destination, cases, protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--continue-unsubmitted", action="store_true")
    parser.add_argument("--retain-network-failures", action="store_true")
    args = parser.parse_args()
    destination, cases, protocol = prepare(args.version)
    prompts = {args.version: f"TRAINING_PROMPT_{args.version}.txt"}
    if not args.run:
        print(json.dumps(summarize(pilot_out=destination, arm_prompts=prompts), ensure_ascii=False), flush=True)
        return
    if args.continue_unsubmitted:
        cases = [r for r in cases if not (destination / "cases" / args.version / r["dataset"] / (r["qid"] + ".json")).exists()]
    options = get_api_option(2)
    owner = Store(destination / "transport", {"task": protocol["task"], "arm": args.version,
                                               "protocol_sha256": sha(destination / "PROTOCOL_FROZEN.json")})
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt", allowed_root=ROOT, provider="dashscope"))
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=0)), timeout=90))
            client = stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=options["base_url"], timeout=90, max_retries=0, http_client=network))
            streaming = StreamingClient(client)
            stack.enter_context(provider_dns(urlsplit(options["base_url"]).hostname, owner))
            with ThreadPoolExecutor(max_workers=3) as pool:
                all_unknown_batches = 0
                for i in range(0, len(cases), 3):
                    batch = cases[i:i+3]
                    futures = [(case, pool.submit(run_one, case, args.version, protocol, streaming,
                                           pilot_out=destination, arm_prompts=prompts)) for case in batch]
                    errors = []
                    retained_unknown = 0
                    for case, future in futures:
                        try:
                            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                        except Exception as error:
                            path = destination / "cases" / args.version / case["dataset"] / (case["qid"] + ".json")
                            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                            if args.retain_network_failures and value.get("status") == "unknown":
                                retained_unknown += 1
                                print(json.dumps({"qid": case["qid"], "retained_unknown": True, "replayed": False}), flush=True)
                            else:
                                errors.append(type(error).__name__ + ": " + str(error))
                    print(json.dumps(summarize(pilot_out=destination, arm_prompts=prompts), ensure_ascii=False), flush=True)
                    all_unknown_batches = all_unknown_batches + 1 if retained_unknown == len(batch) else 0
                    if all_unknown_batches >= 2:
                        errors.append("Two consecutive batches entirely unknown; stop new submissions")
                    if errors:
                        write_json(destination / "TRANSPORT_STOP.json", {"utc": utc(), "errors": errors})
                        raise RuntimeError("Retain unknown requests; inspect before continuing unsubmitted cases")
    finally:
        owner.db.close()


if __name__ == "__main__":
    main()
