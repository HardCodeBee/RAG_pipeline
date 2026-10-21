"""Process-only DNS fallback; original API URL, SNI and TLS checks retained."""
from contextlib import contextmanager
import ipaddress
import json
import socket
import threading
import time
import urllib.request
from urllib.parse import urlencode

HOST = 'ws-memqyuz7m8e9oxs1.cn-beijing.maas.aliyuncs.com'
RESOLVER = 'https://1.1.1.1/dns-query'


@contextmanager
def provider_dns(host, store):
    if host != HOST:
        raise ValueError('DNS fallback is restricted to the frozen provider')
    original = socket.getaddrinfo
    mutex, cache = threading.Lock(), {'ips': [], 'expires': 0.}

    def resolve(name, port, *args, **kwargs):
        normalized = name.decode('ascii') if isinstance(name, bytes) else name
        if normalized != HOST:
            return original(name, port, *args, **kwargs)
        try:
            return original(name, port, *args, **kwargs)
        except socket.gaierror:
            with mutex:
                if time.monotonic() >= cache['expires']:
                    request = urllib.request.Request(RESOLVER+'?'+urlencode({'name': HOST, 'type': 'A'}),
                        headers={'Accept': 'application/dns-json'})
                    with urllib.request.urlopen(request, timeout=12) as response:
                        answer = json.load(response)
                    if answer.get('Status') != 0:
                        raise RuntimeError('Public DNS did not resolve the frozen provider')
                    records = [r for r in answer.get('Answer', []) if r.get('type') == 1]
                    ips = list(dict.fromkeys(str(ipaddress.IPv4Address(r['data'])) for r in records))
                    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
                        raise RuntimeError('Provider DNS lacks verified public A records')
                    ttl = max(1, min(int(r.get('TTL', 60)) for r in records))
                    cache.update(ips=ips, expires=time.monotonic()+ttl)
                    store.event('dns_fallback', host=HOST, resolver=RESOLVER, public_addresses=ips,
                        TLS_hostname_verification=True, credentials_sent_to_resolver=False,
                        system_dns_changed=False)
                ips = list(cache['ips'])
            return [v for ip in ips for v in original(ip, port, *args, **kwargs)]

    socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original
