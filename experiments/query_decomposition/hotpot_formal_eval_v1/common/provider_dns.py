"""Process-local fallback for the pinned provider, with normal TLS verification."""
from contextlib import contextmanager
import ipaddress
import json
import socket
import time
import urllib.request


@contextmanager
def provider_dns(host, store):
    if host != 'ws-memqyuz7m8e9oxs1.cn-beijing.maas.aliyuncs.com':
        raise ValueError('DNS fallback is restricted to the frozen API hostname')
    original = socket.getaddrinfo
    cache = {'addresses': [], 'expires': 0}

    def refresh():
        url = 'https://dns.google/resolve?name=' + host + '&type=A'
        with urllib.request.urlopen(url, timeout=15) as response:
            answer = json.load(response)
        if answer.get('Status') != 0:
            raise OSError('Provider DNS resolution failed')
        records = [item for item in answer.get('Answer', []) if item.get('type') == 1]
        addresses = list(dict.fromkeys(str(ipaddress.IPv4Address(item['data'])) for item in records))
        if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
            raise ValueError('DNS response lacks public provider addresses')
        # Ordering follows the current response; no obsolete address is injected.
        ttl = max(1, min(int(item.get('TTL', 60)) for item in records))
        cache.update(addresses=addresses, expires=time.monotonic() + ttl)
        store.event('dns_fallback', host=host, resolver='https://dns.google/resolve',
                    records=records, tls_hostname_verification=True,
                    credentials_sent_to_resolver=False, system_dns_changed=False)

    def resolve(request_host, port, *args, **kwargs):
        normalized = request_host.decode('ascii') if isinstance(request_host, bytes) else request_host
        if normalized != host:
            return original(request_host, port, *args, **kwargs)
        try:
            return original(request_host, port, *args, **kwargs)
        except socket.gaierror:
            if time.monotonic() >= cache['expires']:
                refresh()
            return [item for address in cache['addresses'] for item in original(address, port, *args, **kwargs)]

    socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original
