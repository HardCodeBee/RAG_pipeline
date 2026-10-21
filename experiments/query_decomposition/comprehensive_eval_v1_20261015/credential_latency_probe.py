"""Small credential A/B on frozen payloads; separate from formal predictions."""
from contextlib import ExitStack
from types import SimpleNamespace
import json
import sqlite3
import time
import statistics
from urllib.parse import urlsplit
import httpx
from openai import OpenAI

from .common import ROOT, RUN, OUTPUT, read, save, log, utc
from .campaign import TARGET
from .execution import ExactTransport
from .throughput_epoch import SlotAPI
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget


def secret(filename):
    value = (ROOT / filename).read_text(encoding='utf-8-sig').strip()
    if not value.startswith('sk-') or any(c.isspace() for c in value):
        raise ValueError('A separate valid DashScope API key file is required')
    return value


def main():
    frozen = read(TARGET / 'CAMPAIGN_FROZEN.json')
    model, endpoint = frozen['generator'], frozen['base_url']
    baseline = TARGET / 'hotpotqa/formal/b00_rag_top10_v1/fixed/checkpoint.sqlite3'
    cases = []
    with sqlite3.connect(baseline.resolve().as_uri() + '?mode=ro', uri=True) as db:
        for value, in db.execute("SELECT value FROM data WHERE kind='prediction'"):
            prediction = json.loads(value)
            if prediction['status'] != 'success': continue
            payload = json.loads(db.execute('SELECT payload FROM requests WHERE id=?', (prediction['request_id'],)).fetchone()[0])
            assert payload['model'] == model and payload['temperature'] == 0 and payload['max_tokens'] == 64
            cases.append({'qid': prediction['qid'], 'prompt': payload['messages'][0]['content']})
            if len(cases) == 7: break
    result = {'utc': utc(), 'model': model, 'base_url': endpoint, 'formal_outputs_replaced': False,
              'purpose': 'authentication and small latency comparison only; never choose by F1',
              'planned_calls_max': 14, 'concurrency': 3, 'global_cap_cny': 200,
              'rate_policy_unchanged': True, 'new_account_independent_quota_verified': False}
    output = RUN / 'EXTRA_CREDENTIAL_LATENCY_PROBE.json'
    stores = {}
    with ExitStack() as stack:
        budget = ParallelBudget(OUTPUT / 'budget.sqlite3', 200)
        stack.callback(budget.db.close)
        transports = {}
        for alias, filename in [('original', 'api_code_qwen.txt'), ('extra', 'api_code_qwen_extra.txt')]:
            store = Store(RUN / 'credential_probe_v1' / alias,
                          {'model': model, 'base_url': endpoint, 'credential_alias': alias,
                           'qids': [r['qid'] for r in cases], 'usage': 'latency diagnostics, not formal evaluation'})
            stores[alias] = store
            stack.callback(store.db.close)
            network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,
                limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=10)), timeout=60))
            client = stack.enter_context(OpenAI(api_key=secret(filename), base_url=endpoint,
                timeout=60, max_retries=0, http_client=network))
            transports[alias] = ExactTransport(store, StreamingClient(client), budget, model)
        with provider_dns(urlsplit(endpoint).hostname, stores['extra']):
            for alias in ['extra', 'original']:
                transport = transports[alias]
                transport.qid = cases[6]['qid']
                try:
                    rid, response = transport.ask('authentication_warmup', cases[6]['prompt'], 64)
                    result[alias + '_authentication'] = {'status': response['status'], 'returned_model': response.get('returned_model')}
                except Exception as exc:
                    row = stores[alias].db.execute('SELECT status,result FROM requests ORDER BY rowid DESC LIMIT 1').fetchone()
                    detail = json.loads(row[1]) if row and row[1] else {}
                    result.update(status='authentication_or_transport_failed', failed_credential_alias=alias,
                        error_type=type(exc).__name__, http_status=detail.get('http_status'), error_code=detail.get('error_code'))
                    save(output, result)
                    log('extra_credential_probe_stopped', **result)
                    print(json.dumps(result), flush=True)
                    return
                if response['status'] != 'success' or response.get('returned_model') != model:
                    result.update(status='warmup_not_successful', failed_credential_alias=alias)
                    save(output, result)
                    print(json.dumps(result), flush=True)
                    return
            measurements = {'original': [], 'extra': []}
            blocks = []
            # ABBA: each credential answers the same six saved SDK payloads.
            for alias, subset in [('original', cases[:3]), ('extra', cases[:3]),
                                  ('extra', cases[3:6]), ('original', cases[3:6])]:
                api = SlotAPI(transports[alias], 3)
                tick = time.perf_counter()
                operations = [SimpleNamespace(index=i, args=('latency_probe', c['prompt'], 64), kwargs={})
                              for i, c in enumerate(subset)]
                try:
                    values = api.ask_group(operations, subset)
                finally:
                    api.close()
                blocks.append({'credential': alias, 'seconds': time.perf_counter()-tick, 'questions': len(subset)})
                for (_, response), case in zip(values, subset):
                    measurements[alias].append({'qid': case['qid'], 'status': response['status'],
                        'returned_model': response.get('returned_model'), 'latency_seconds': response.get('latency_seconds')})
            summaries = {}
            for alias, values in measurements.items():
                good = [v['latency_seconds'] for v in values if v['status']=='success' and v['returned_model']==model]
                summaries[alias] = {'planned': 6, 'success_same_model': len(good),
                    'median_seconds': statistics.median(good) if good else None,
                    'mean_seconds': statistics.mean(good) if good else None,
                    'block_wall_seconds': sum(b['seconds'] for b in blocks if b['credential']==alias)}
            result.update(status='completed_small_diagnostic', measurements=measurements, summary=summaries, blocks=blocks,
                          budget_after_probe=budget.summary(), not_a_large_concurrency_or_quota_test=True)
            save(output, result)
            log('extra_credential_small_latency_probe_completed', summary=summaries,
                formal_outputs_replaced=False, report=str(output))
            for store in stores.values(): store.export(full=True)
            print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
