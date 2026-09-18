"""No-auth HTTPS transport checks; no model inference or credential output."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import time
import urllib.parse
import urllib.request

import httpx


def safe_proxy(value):
    parsed = urllib.parse.urlsplit(value if "://" in value else "http://"+value)
    try:
        return {"scheme": parsed.scheme, "host": parsed.hostname, "port": parsed.port}
    except ValueError:
        return {"configured": True, "unparsed": True}


def request(spec):
    label, url, trust_env = spec
    started = time.time()
    try:
        with httpx.Client(timeout=12., trust_env=trust_env, follow_redirects=False) as client:
            response = client.get(url, headers={"User-Agent": "M6-transport-check/1.0"})
        return {"label": label, "http_status": response.status_code, "seconds": time.time()-started}
    except Exception as error:
        chain = []
        seen = set()
        while error is not None and id(error) not in seen and len(chain) < 10:
            seen.add(id(error)); chain.append(type(error).__name__)
            error = error.__cause__ or error.__context__
        return {"label": label, "error_chain": chain, "seconds": time.time()-started}


if __name__ == "__main__":
    report = {"observed_at_unix": time.time(), "model_generation_calls": 0, "authorization_headers_sent": False,
              "proxy_environment_keys_present": [key for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY") if os.environ.get(key)],
              "effective_proxies": {key: safe_proxy(value) for key, value in urllib.request.getproxies().items() if key != "no"}}
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
                report["windows_proxy_enabled"] = winreg.QueryValueEx(key, "ProxyEnable")[0]
                try:
                    value = winreg.QueryValueEx(key, "ProxyServer")[0]
                    report["windows_proxy_server"] = safe_proxy(value) if ";" not in value else {"configured_per_protocol": True}
                except FileNotFoundError:
                    pass
        except OSError as error:
            report["windows_proxy_read_error"] = type(error).__name__
    try:
        report["api_dns_address_count"] = len({item[4][0] for item in socket.getaddrinfo("api.openai.com", 443, type=socket.SOCK_STREAM)})
    except OSError as error:
        report["dns_error"] = type(error).__name__
    specs = [("api_current_settings", "https://api.openai.com/v1/models", True),
             ("api_direct", "https://api.openai.com/v1/models", False),
             ("control_current_settings", "https://www.microsoft.com", True)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        report["https_checks"] = list(pool.map(request, specs))
    path = Path(__file__).resolve().parents[1] / "analysis/hotpotqa_router/m6_transport_check_20260915.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
