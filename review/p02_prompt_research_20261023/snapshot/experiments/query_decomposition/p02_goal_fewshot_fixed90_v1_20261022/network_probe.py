"""Unpaid unauthenticated connectivity check to the already approved endpoint."""
from pathlib import Path
from urllib.parse import urlsplit
import json
import sys
import time
import socket
import ipaddress
import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from . import run
from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns


def main():
    config, _ = run.check()
    events, checks = [], []
    class Events:
        def event(self, name, **fields):
            events.append({"event": name, **fields})
    host = urlsplit(config["generation"]["base_url"]).hostname
    with provider_dns(host, Events()):
        resolved = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(r[4][0] for r in resolved))
        original = socket.getaddrinfo
        for address in addresses:
            assert ipaddress.ip_address(address).is_global
            def pinned(name, port, *args, **kwargs):
                normalized = name.decode("ascii") if isinstance(name, bytes) else name
                return original(address if normalized == host else name, port, *args, **kwargs)
            socket.getaddrinfo = pinned
            start = time.monotonic()
            try:
                with httpx.Client(trust_env=False, timeout=httpx.Timeout(12, connect=10),
                                  transport=httpx.HTTPTransport(retries=0)) as client:
                    response = client.get(config["generation"]["base_url"])
                    checks.append({"public_A_address": address, "HTTP_status": response.status_code, "reachable": True,
                                   "seconds": time.monotonic() - start})
            except Exception as exc:
                causes, seen = [], set()
                while exc and id(exc) not in seen:
                    seen.add(id(exc))
                    causes.append({"type": type(exc).__name__, "message": str(exc),
                                   "errno": getattr(exc, "errno", None)})
                    exc = exc.__cause__ or exc.__context__
                checks.append({"public_A_address": address, "reachable": False, "seconds": time.monotonic() - start,
                               "causes": causes})
            finally:
                socket.getaddrinfo = original
    result = {"utc": run.utc(), "authentication_sent": False, "model_requests": 0,
              "checks": checks, "DNS_events": events}
    run.save(run.OUT / "NETWORK_PROBE.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
