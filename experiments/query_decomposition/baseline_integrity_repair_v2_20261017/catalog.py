"""Read-only known-prefix migration, confined to the same method and sample.

Only max_tokens may differ. A complete old response remains an old response,
with its real payload/usage; it is never relabelled as a fresh 32K call.
"""
import json
import sqlite3
from .common import digest, canonical


def fingerprint(payload):
    return digest({k: v for k, v in payload.items() if k != 'max_tokens'})


class Catalog:
    def __init__(self, path):
        self.path = path
        self.requests = {}
        self.retrievals = {}
        self.hybrid = {}
        if not path.exists():
            return
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            bindings = {}
            for value, in db.execute("select value from data where kind='evidence'"):
                row = json.loads(value)
                bindings.setdefault(row['request_id'], set()).add((row['qid'], row['stage']))
            for rid, stage, payload, status, result in db.execute('select * from requests'):
                p = json.loads(payload)
                r = json.loads(result) if result else {'status': status}
                # Length-capped generations must be regenerated, including intermediates.
                if status == 'success' and (r.get('finish_reason') != 'stop' or not r.get('usage')):
                    continue
                if status == 'in_flight':
                    raise RuntimeError('Source worker has not drained')
                if status not in {'success', 'unknown', 'failed', 'provider_content_rejected'}:
                    continue
                for binding in bindings.get(rid, set()):
                    key = (*binding, fingerprint(p))
                    self.requests.setdefault(key, {'request_id': rid, 'payload': p,
                        'result': r, 'status': status, 'origin_db': str(path)})
            for kind in ['retrieval', 'sparse_retrieval']:
                for key, value in db.execute('select key,value from data where kind=?', (kind,)):
                    packet = json.loads(value)
                    if kind == 'retrieval' and (packet.get('encoder_truncated') or packet.get('query_tokens', 513) > 512):
                        continue
                    self.retrievals[kind, packet['qid'], packet['query'], packet['k']] = packet
            for key, value in db.execute("select key,value from data where kind='hybrid'"):
                self.hybrid[key] = json.loads(value)

    def request(self, qid, stage, payload):
        found = self.requests.get((qid, stage, fingerprint(payload)))
        if found:
            # Compare all generation conditions, rather than trusting only a hash.
            a = {k: v for k, v in payload.items() if k != 'max_tokens'}
            b = {k: v for k, v in found['payload'].items() if k != 'max_tokens'}
            if canonical(a) != canonical(b):
                raise ValueError('Historical prefix differs beyond the nonbinding cap')
        return found
