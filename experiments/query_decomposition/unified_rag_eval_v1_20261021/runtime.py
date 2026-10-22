"""Six-method formal entry point. Default is offline inspection, never dispatch.

Uses unchanged method controllers; no historical run.main, dev gate, parameter
selection, implicit matched runs, or legacy budget objects are invoked.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, write_json
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.retrieval import Retrieval
from src.p02_multigrain_timing_cachefix_v1.core import pipeline
from . import runtime_config as cfg
from .runtime_transport import RuntimeTransport
from .runtime_ledger import RuntimeLedger, RuntimeStageHeld


class RuntimeStore:
    """Keep terminal predictions immutable; technical failures remain resumable."""
    def __init__(self, store):
        self.original = store
        self.last_pending = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def put(self, kind, key, value):
        if kind == 'prediction' and value.get('status') not in {'success', 'algorithm_failure'}:
            self.last_pending = deepcopy(value)
            return self.original.put('pending_prediction', common.digest(value), value)
        return self.original.put(kind, key, value)


class RetryingTransport(RuntimeTransport):
    """Retry only confirmed retryable rejections, through the one durable ledger."""
    def __init__(self, *args, sleeper=time.sleep, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleeper = sleeper

    def ask(self, stage, prompt, max_tokens):
        for _ in range(5):
            rid, result = super().ask(stage, prompt, max_tokens)
            if result['status'] != 'failed' or result.get('http_status') not in {409, 429}:
                return rid, result
            delay = max(30., result.get('retry_after_seconds') or 0.)
            if delay > 300:
                raise RuntimeStageHeld('Provider cooldown requires later resume')
            while delay:
                self._guard()
                chunk = min(30., delay)
                self.sleeper(chunk)
                delay -= chunk
        raise RuntimeStageHeld('Five attempts exhausted; temporarily held')


class GuardedRetrieval(Retrieval):
    def __init__(self, store, resources, guard):
        from types import SimpleNamespace
        # No unvalidated historical retrieval packet enters the new layer.
        super().__init__(store, resources, SimpleNamespace(retrievals={}, path=None))
        self.guard = guard

    def retrieve_many(self, cases, k):
        if self.guard() is False:
            raise cfg.ExecutionHeld('Retrieval blocked by execution gate')
        return super().retrieve_many(cases, k)


def notify_cost(event):
    # Durable delivery records also remain in RuntimeLedger. No external message.
    print(json.dumps({'event': 'cost_alert', 'details': event}, ensure_ascii=False), flush=True)


def references_for_cost():
    groups = cfg.read(cfg.HERE / 'execution_details/COST_REFERENCE.frozen.json')['groups']
    configs = cfg.method_configs()
    return {(r['dataset'], r['method']): {**r, 'current_configuration': configs.get(r['method']),
             'comparison_note': 'Historical complete-flow reference; current method settings may differ'}
            for r in groups if r.get('mean_cny') is not None}


class Runner:
    def __init__(self, client, ledger, guard, *, output=cfg.OUTPUT,
                 transport_factory=RetryingTransport, retrieval_factory=GuardedRetrieval,
                 reuse_factory=None):
        self.client, self.ledger, self.guard = client, ledger, guard
        self.output = Path(output)
        self.transport_factory, self.retrieval_factory = transport_factory, retrieval_factory
        self.reuse_factory = reuse_factory
        self.configs = cfg.method_configs()
        self.handles = {}

    def identity(self, dataset, method, resources):
        return {'dataset': dataset, 'method': method, 'split': 'formal',
                'execution_variant': 'p02_shared_baseline_v2' if method == cfg.P02 else 'six_method_runtime_v1',
                'configuration': self.configs.get(method, {'interface': 'qa_raw'}),
                'scientific_scope_digest': common.digest(cfg.scope(cfg.read(cfg.HERE / 'PLAN.json'))),
                'resource_namespace': common.digest(resources),
                'resource_sha256': common.sha(common.PARENT / dataset / 'RESOURCE_FROZEN.json'),
                'input_sha256': common.sha(common.INPUTS / dataset / 'formal.questions.jsonl'),
                'model': common.MODEL}

    def _catalog(self, dataset, method, resources, store):
        if self.reuse_factory is not None:
            return self.reuse_factory(dataset, method, resources, store)
        from .runtime_reuse import ReadOnlyReuseCatalog, discover_sources
        return ReadOnlyReuseCatalog(discover_sources(cfg.ROOT, method, dataset),
                                    method=method, dataset=dataset, resources=resources, store=store,
                                    reconciliation_resolver=self.ledger.historical_reconciliations)

    def run_case(self, dataset, method, case, resources, *, p02_case=None):
        self.guard()
        if method not in cfg.METHODS or dataset not in cfg.DATASETS or set(case) != {'qid', 'question'}:
            raise ValueError('Only frozen main methods and gold-free original questions are accepted')
        directory = self.output / 'main' / dataset / method
        store = RuntimeStore(Store(directory, self.identity(dataset, method, resources)))
        retrieval = None
        try:
            terminal = store.get('prediction', case['qid'])
            if terminal is not None:
                return {'qid': case['qid'], 'status': terminal['status'], 'already_terminal': True}
            catalog = self._catalog(dataset, method, resources, store)
            transport = self.transport_factory(store, self.client, self.ledger,
                catalog=catalog, guard=self.guard, dataset=dataset, method=method)
            transport.qid = case['qid']
            retrieval = self.retrieval_factory(store, resources, self.guard)
            if dataset in self.handles and isinstance(retrieval, GuardedRetrieval):
                retrieval.index, retrieval.encoder, retrieval.chunks, retrieval.sparse = self.handles[dataset]
            if method == cfg.P02:
                if p02_case is None:
                    full_case, decomposer = cfg.load_r02_case(dataset, case, with_provenance=True)
                    store.put('decomposition_usage', case['qid'], decomposer)
                else:
                    full_case = p02_case
                if full_case['qid'] != case['qid'] or full_case['question'] != case['question'] or full_case['dataset'] != dataset:
                    raise ValueError('P02 decomposition bound to another original question')
                self._p02(full_case, store, transport, retrieval, common.digest(resources))
            else:
                module = importlib.import_module('experiments.query_decomposition.baseline_integrity_repair_v2_20261017.methods.'
                                                 + common.METHODS[method] + '.method')
                module.run([case], retrieval, transport, store, self.configs[method])
            prediction = store.get('prediction', case['qid'])
            if prediction is not None:
                self.ledger.complete_query(dataset, method, case['qid'], complete=True, fresh=True)
                status = prediction['status']
            else:
                status = 'technical_pending'
            result = {'qid': case['qid'], 'status': status,
                      'cost': self.ledger.query_summary(dataset, method, case['qid'])}
        except cfg.ExecutionHeld:
            raise
        except (RuntimeStageHeld, common.IntegrityError, FileNotFoundError) as exc:
            result = {'qid': case['qid'], 'status': 'technical_pending', 'reason': str(exc),
                      'error_class': type(exc).__name__}
        finally:
            if isinstance(retrieval, GuardedRetrieval) and retrieval.index is not None:
                self.handles[dataset] = (retrieval.index, retrieval.encoder, retrieval.chunks, retrieval.sparse)
            store.export()
            store.db.close()
        write_json(directory / 'query_status' / (case['qid'] + '.json'), result)
        return result

    def _p02(self, case, store, transport, retrieval, namespace):
        nodes, receipts = [], []

        def ask(stage, prompt, maximum, allow_final_truncation=False):
            self.guard()
            rid, result = transport.ask(stage, prompt, maximum)
            if result.get('status') != 'success':
                raise RuntimeStageHeld('P02 stage incomplete: ' + stage)
            # Native final length exception is deliberately not a complete shared-layer result.
            if result.get('finish_reason') != 'stop' or not result.get('usage'):
                raise RuntimeStageHeld('P02 incomplete receipt: ' + stage)
            resolved = self.ledger.receipt(rid)
            if stage != 'legacy/final':
                source = {'kind': 'shared_layer', 'dataset': case['dataset'], 'qid': case['qid'],
                          'method': cfg.P02, 'execution_variant': 'p02_shared_baseline_v2',
                          'resource_namespace': namespace}
                if resolved.get('origin') == 'c1_historical':
                    source['kind'] = 'c1'
                    source['c1_validation'] = resolved.get('provenance', {}).get('c1_validation')
                receipts.append({'stage': stage, 'request_id': rid,
                                 'payload': resolved['payload'], 'result': resolved['result'], 'source': source})
            return rid, result['answer'], result['finish_reason']

        def record(node):
            nodes.append(node)
            store.put('p02_node', case['qid'] + '/' + str(node['step_id']), node)
            if node['role'] == 'goal':
                prefix = {'schema': 'p02_shared_prefix_v1', 'case': case, 'method': cfg.P02,
                          'execution_variant': 'p02_shared_baseline_v2', 'resource_namespace': namespace,
                          'nodes': deepcopy(nodes), 'original_retrieval': node['original_retrieval'],
                          'receipts': deepcopy(receipts)}
                binding = {'qid': case['qid'], 'prefix_sha256': common.digest(prefix),
                           'resource_namespace': namespace}
                store.put('qa_raw_prefix_binding', case['qid'], binding)
                write_json(store.directory / 'prefixes' / (case['qid'] + '.json'), prefix)
        try:
            result = pipeline.execute(case,
                lambda query: retrieval.retrieve_many([{'qid': case['qid'], 'query': query}], 10)[0],
                ask, record_node=record)
        except RuntimeError as exc:
            if isinstance(exc, (cfg.ExecutionHeld, RuntimeStageHeld, common.IntegrityError)):
                raise
            # The frozen pipeline explicitly rejects empty successfully returned child text.
            if str(exc) not in {'Incomplete/empty child answer is not propagated',
                                'Incomplete/empty sentence formatting is not propagated'}:
                raise
            store.put('prediction', case['qid'], {'qid': case['qid'], 'question': case['question'],
                'status': 'algorithm_failure', 'prediction': '', 'reason': str(exc)})
            return
        store.put('p02_trace', case['qid'], result)
        store.put('prediction', case['qid'], {'qid': case['qid'], 'question': case['question'],
            'status': 'success', 'prediction': result['prediction'], 'request_id': result['request_id'],
            'finish_reason': result['finish_reason']})

    def run_main(self):
        questions = cfg.load_questions()
        for dataset in cfg.DATASETS:
            self.handles.clear()
            resources = cfg.resource(dataset)
            for method in cfg.METHODS:
                for case in questions[dataset]:
                    result = self.run_case(dataset, method, case, resources)
                    print(json.dumps({'dataset': dataset, 'method': method, **result}, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Requires explicit matching approval and cleared user holds')
    parser.add_argument('--stage', choices=['main', 'qa_raw'], default='main')
    parser.add_argument('--prepare-qa-raw', action='store_true', help='Offline inputs/status only; never submits finals')
    parser.add_argument('--report', action='store_true', help='Offline native scoring and fixed-denominator report')
    args = parser.parse_args(argv)
    if not args.execute:
        if args.report:
            from .runtime_report import report
            print(json.dumps(report(), ensure_ascii=False, indent=2))
        elif args.prepare_qa_raw:
            from .runtime_inputs import prepare_qa_raw
            print(json.dumps(prepare_qa_raw(), ensure_ascii=False))
        else:
            print(json.dumps(cfg.inspect(), ensure_ascii=False, indent=2))
        return
    if args.prepare_qa_raw or args.report:
        parser.error('Offline preparation/report flags cannot be combined with --execute')
    guard = cfg.ExecutionGate()
    guard()  # Before credentials, clients, model imports, or output creation.
    from .runtime_resources import inspect_resources
    if not inspect_resources()['passed']:
        raise cfg.ExecutionHeld('Frozen resource preflight did not pass')
    import httpx
    from openai import OpenAI
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
    options = get_api_option(2)
    endpoint = cfg.read(cfg.ROOT / 'src/p02_multigrain_timing_cachefix_v1/config.json')['generation']['base_url']
    if options['model'] != common.MODEL or options['base_url'].rstrip('/') != endpoint.rstrip('/'):
        raise cfg.ExecutionHeld('Provider model/endpoint differs from frozen reader')
    install()
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=cfg.ROOT, provider='dashscope'))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=60))
        client = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=options['base_url'],
                                            max_retries=0, timeout=60, http_client=network))
        ledger = RuntimeLedger(cfg.OUTPUT / 'runtime_ledger.sqlite3',
                               reference_means=references_for_cost(), notify=notify_cost)
        try:
            runner = Runner(client, ledger, guard)
            if args.stage == 'main':
                runner.run_main()
            else:
                from .runtime_inputs import execute_qa_raw
                execute_qa_raw(runner)
        finally:
            ledger.close()


if __name__ == '__main__':
    main()
