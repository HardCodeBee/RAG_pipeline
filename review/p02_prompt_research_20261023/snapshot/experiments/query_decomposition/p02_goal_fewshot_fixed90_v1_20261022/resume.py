"""Isolated connection diagnostics; generation payloads and old holds unchanged."""
from contextlib import contextmanager
import argparse
import ipaddress
import json
import socket
import httpx
from . import run
from experiments.query_decomposition.p02_formal1500_v1_20261021 import network

REVIEW = run.HERE / "RUNTIME_CONNECT_REVIEW.json"


def connect_proof(exc):
    chain, seen, typed = [], set(), []
    while exc and id(exc) not in seen:
        seen.add(id(exc))
        typed.append(exc)
        chain.append({"type": type(exc).__name__, "errno": getattr(exc, "errno", None)})
        exc = exc.__cause__ or exc.__context__
    before_http = any(isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)) for e in typed)
    uncertain = any(isinstance(e, (httpx.ReadError, httpx.ReadTimeout, httpx.WriteError,
                                 httpx.WriteTimeout, httpx.RemoteProtocolError)) for e in typed)
    return chain, bool(before_http and not uncertain)


class ConnectionEvidenceAsk(run.StrictAsk):
    def _client(self):
        llm = super()._client()
        if getattr(llm, "_connection_evidence_installed", False):
            return llm
        original_create = llm.client.chat.completions.create
        original_once = llm.request_once
        evidence = {}
        def create(*args, **kwargs):
            try:
                return original_create(*args, **kwargs)
            except Exception as exc:
                chain, proof = connect_proof(exc)
                evidence.update(failure_causes=chain, proved_not_submitted=proof)
                raise
        def request_once(payload):
            evidence.clear()
            result = original_once(payload)
            if evidence:
                result.update(evidence)
                if evidence["proved_not_submitted"]:
                    result["reconciliation"] = "Typed connect/TLS/pool failure before HTTP request transmission; bounded technical recovery permitted"
                self.events({"event": "connection_evidence", **evidence})
            return result
        llm.client.chat.completions.create = create
        llm.request_once = request_once
        llm._connection_evidence_installed = True
        return llm

    def _retire(self, rid, stage, payload, attempts, reason):
        raw = self.cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()[0]
        result = json.loads(raw) if raw else {}
        charge = self.budget.db.execute("SELECT actual FROM charges WHERE id=?", (rid,)).fetchone()
        if any(result.get(k) is not None for k in ("response_id", "raw_response", "usage", "returned_model", "refusal", "finish_reason")) or (charge and charge[0] is not None):
            raise RuntimeError("Returned response or charge evidence is held without resubmission")
        return super()._retire(rid, stage, payload, attempts, reason)


def prepare():
    run.check()
    probe = run.read(run.OUT / "NETWORK_PROBE.json")
    candidates = [c for c in probe["checks"] if c.get("reachable") and c.get("public_A_address")]
    preferred = min(candidates, key=lambda c: c["seconds"])["public_A_address"] if candidates else None
    review = {"utc": run.utc(), "scope": "Connection-only implementation refinement inside current explicitly authorized fixed90",
        "generation_payloads_changed": False, "model_endpoint_SNI_TLS_verification_changed": False,
        "SDK_HTTP_retries": 0, "maximum_durable_actual_attempts": 5,
        "recovery": "Only typed HTTPX ConnectError/ConnectTimeout/PoolTimeout before HTTP transmission; Read/Write/RemoteProtocol remain unknown and held",
        "old_unknowns_reclassified_or_reissued": False,
        "preferred_public_A_address": preferred,
        "routing": "Validate preferred address remains in current DNS answers; resolve only the original host inside this process; retain original URL, Host, SNI and TLS certificate checks",
        "base_protocol_sha256": run.sha(run.HERE / "PROTOCOL.frozen.json"),
        "source_sha256": run.sha(__file__), "network_probe_sha256": run.sha(run.OUT / "NETWORK_PROBE.json")}
    if REVIEW.exists():
        raise FileExistsError("Preserve connection review")
    run.save(REVIEW, review)
    run.save(run.REVIEW / "RUNTIME_CONNECT_REVIEW.json", review)
    with (run.PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"id": "P02_GOAL_FEWSHOT90_CONNECT_PHASE_DIAGNOSTICS_20261022", "utc": run.utc(),
            "authority": run.DECISION, "review_sha256": run.sha(REVIEW), "generation_payloads_changed": False,
            "unknown_reissues": False, "main_evaluation_resumed": False}, ensure_ascii=False) + "\n")
    print(json.dumps(review, ensure_ascii=True))


def execute(limit=None):
    review = run.read(REVIEW)
    assert review["source_sha256"] == run.sha(__file__)
    assert review["base_protocol_sha256"] == run.sha(run.HERE / "PROTOCOL.frozen.json")
    preferred = review["preferred_public_A_address"]
    original_context = network.provider_dns
    @contextmanager
    def provider_dns(host, store):
        with original_context(host, store):
            original = socket.getaddrinfo
            addresses = {r[4][0] for r in original(host, 443, type=socket.SOCK_STREAM)}
            use = preferred if preferred in addresses and ipaddress.ip_address(preferred).is_global else None
            def resolve(name, port, *args, **kwargs):
                normalized = name.decode("ascii") if isinstance(name, bytes) else name
                return original(use if use and normalized == host else name, port, *args, **kwargs)
            socket.getaddrinfo = resolve
            store.event("validated_public_A_route", host=host, selected=use,
                        DNS_answers=sorted(addresses), original_URL_SNI_TLS=True, process_only=True)
            try:
                yield
            finally:
                socket.getaddrinfo = original
    run.StrictAsk = ConnectionEvidenceAsk
    network.provider_dns = provider_dns
    try:
        print(json.dumps(run.execute(limit), ensure_ascii=True))
    finally:
        network.provider_dns = original_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare() if args.prepare else execute(args.limit)
