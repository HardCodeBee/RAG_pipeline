"""Network-only continuation: pinned API DNS via verified public DoH IP.

System DNS is unavailable in this execution environment. Resolve only the frozen
provider; keep the original API URL/SNI/certificate checks and all old receipts.
"""
from contextlib import contextmanager
import hashlib
import ipaddress
import json
from pathlib import Path
import socket
import ssl
import sys
import threading
import time
import urllib.request
from urllib.parse import urlencode

from . import run as base

BASE_INSTALL = base.install_budget_overlay
base.OUT = base.PARENT / 'resume_v2_primary_network_20261020'
base.MODULE = 'experiments.query_decomposition.resume_v2_primary_20261020.network_resume'
HOST = 'ws-memqyuz7m8e9oxs1.cn-beijing.maas.aliyuncs.com'
RESOLVER = 'https://1.1.1.1/dns-query'


def addresses():
    req = urllib.request.Request(RESOLVER + '?' + urlencode({'name': HOST, 'type': 'A'}),
                                 headers={'Accept': 'application/dns-json'})
    with urllib.request.urlopen(req, timeout=12) as response:
        answer = json.load(response)
    if answer.get('Status') != 0:
        raise RuntimeError('Public DNS did not resolve the frozen provider')
    records = [x for x in answer.get('Answer', []) if x.get('type') == 1]
    ips = list(dict.fromkeys(str(ipaddress.IPv4Address(x['data'])) for x in records))
    if not ips or any(not ipaddress.ip_address(x).is_global for x in ips):
        raise RuntimeError('Provider DNS lacks verified public A records')
    return ips, max(1, min(int(x.get('TTL', 60)) for x in records))


@contextmanager
def provider_dns(host, store):
    if host != HOST:
        raise ValueError('DNS overlay applies only to the frozen provider')
    original = socket.getaddrinfo
    mutex = threading.Lock()
    cache = {'ips': [], 'expires': 0.}

    def resolve(name, port, *args, **kwargs):
        normalized = name.decode('ascii') if isinstance(name, bytes) else name
        if normalized != HOST:
            return original(name, port, *args, **kwargs)
        try:
            return original(name, port, *args, **kwargs)
        except socket.gaierror:
            with mutex:
                if time.monotonic() >= cache['expires']:
                    ips, ttl = addresses()
                    cache.update(ips=ips, expires=time.monotonic()+ttl)
                    store.event('dns_fallback', host=HOST, resolver=RESOLVER,
                                addresses=ips, tls_hostname_verification=True,
                                credentials_sent_to_resolver=False, system_dns_changed=False)
                ips = list(cache['ips'])
            return [v for ip in ips for v in original(ip, port, *args, **kwargs)]

    socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original


def install():
    value = BASE_INSTALL()
    value[1].provider_dns = provider_dns
    return value


base.install_budget_overlay = install


def preflight():
    ips, ttl = addresses()
    last = None
    for ip in ips:
        try:
            with socket.create_connection((ip, 443), timeout=10) as tcp:
                with ssl.create_default_context().wrap_socket(tcp, server_hostname=HOST):
                    return {'api_hostname_unchanged': True, 'verified_tls_to_provider': True,
                            'resolver_tls_verified': True, 'public_A_records': ips,
                            'ttl_seconds': ttl, 'api_model_calls': 0}
        except OSError as e:
            last = e
    raise RuntimeError('Provider TLS preflight failed: ' + type(last).__name__)


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validation = base.OUT/'NETWORK_VALIDATION.json'
    if '--validate-only' in sys.argv:
        base.validate()
        checks = preflight()
        base.write(validation, {'utc': base.utc(), 'passed': True,
                                'source_sha256': signature, 'checks': checks})
        print(json.dumps(checks), flush=True)
        return
    v = base.read(validation)
    if not v.get('passed') or v['source_sha256'] != signature:
        raise RuntimeError('Matching DNS/TLS preflight validation required')
    if '--worker' not in sys.argv:
        preflight()  # Do not start a paid worker if networking is still broken.
        marker = base.V2/'STOP_REQUESTED'
        if marker.exists():
            content = marker.read_text(encoding='utf8')
            if 'Technical network pause' not in content or 'DNS gaierror11001' not in content:
                raise RuntimeError('A separate stop request remains active')
            archived = base.OUT/'ARCHIVED_DNS_TECHNICAL_STOP.txt'
            if archived.exists():
                raise RuntimeError('Do not overwrite an archived technical stop')
            marker.rename(archived)
        base.write(base.OUT/'NETWORK_POLICY.json', {
            'utc': base.utc(), 'source_sha256': signature,
            'change': 'Process-local DNS fallback only, after verified TLS preflight.',
            'resolver': RESOLVER, 'frozen_api_hostname': HOST,
            'api_url_or_model_changed': False, 'system_dns_changed': False,
            'historical_dns_failures_preserved': True, 'unknown_liabilities_released': False,
            'old_attempt_guard': str(base.PARENT/'resume_v2_primary_20261020/STOP_REQUESTED')})
    base.main()


if __name__ == '__main__':
    main()
