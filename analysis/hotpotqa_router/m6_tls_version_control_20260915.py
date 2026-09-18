"""Six fixed no-auth, non-generation HTTPS checks; retain no response body."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import ssl
import time

import httpx
from httpx._config import create_ssl_context


URL = "https://api.openai.com/v1/models"
ORDER = ("default_tls", "tls12_only") * 3
OUTPUT = Path(__file__).with_suffix(".json")


def check(condition, pair_number):
    context = create_ssl_context(verify=True, trust_env=True)
    if condition == "tls12_only":
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    events = []

    def trace(event, info):
        row = {"event": event}
        if info.get("exception") is not None:
            row["exception_class"] = type(info["exception"]).__name__
        events.append(row)

    def request_hook(request):
        assert "Authorization" not in request.headers
        assert "Proxy-Authorization" not in request.headers
        request.extensions["trace"] = trace

    row = {
        "condition": condition,
        "pair_number": pair_number,
        "certificate_verification": True,
        "hostname_verification": True,
        "tls_minimum": context.minimum_version.name,
        "tls_maximum": context.maximum_version.name,
        "events": events,
    }
    started = time.monotonic()
    try:
        with httpx.Client(
            timeout=12.0,
            verify=context,
            trust_env=True,
            follow_redirects=False,
            event_hooks={"request": [request_hook]},
        ) as client:
            transport = client._transport_for_url(httpx.URL(URL))
            row["effective_pool_class"] = type(transport._pool).__name__
            with client.stream("GET", URL) as response:
                row["http_status"] = response.status_code
                row["negotiated_http_version"] = response.http_version
                stream = response.extensions.get("network_stream")
                tls = stream.get_extra_info("ssl_object") if stream is not None else None
                row["negotiated_tls_version"] = tls.version() if tls is not None else None
    except Exception as error:
        chain, seen = [], set()
        while error is not None and id(error) not in seen and len(chain) < 10:
            seen.add(id(error))
            chain.append(type(error).__name__)
            error = error.__cause__ or error.__context__
        row["error_chain"] = chain
    row["elapsed_seconds"] = time.monotonic() - started
    row["proxy_tls_failed"] = any(event["event"] == "proxy.start_tls.failed" for event in events)
    return row


def main():
    if OUTPUT.exists():
        raise FileExistsError("Fixed six-request result already exists; no requests sent")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": URL,
        "fixed_request_count": 6,
        "order": list(ORDER),
        "model_generation_calls": 0,
        "authorization_headers_sent": False,
        "response_bodies_saved": False,
        "new_client_per_request": True,
        "timeout_seconds": 12,
        "system_proxy_selection": "httpx trust_env=True; unchanged",
        "versions": {name: importlib.metadata.version(name) for name in ("httpx", "httpcore")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [],
    }
    for index, condition in enumerate(ORDER):
        result = check(condition, index // 2 + 1)
        report["results"].append(result)
        print(json.dumps({"completed_request": index + 1, **result}, ensure_ascii=False), flush=True)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"result_file": str(OUTPUT), "requests_completed": 6}), flush=True)


if __name__ == "__main__":
    main()
