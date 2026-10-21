"""Actual R02 errors through the unchanged multigrain dual-reading interface.

No repair arm, generation of decompositions, or access to answer references.
Every provider submission is one-shot; unknown requests are never replayed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
from threading import Lock
import time
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import (
    ROOT, OUTPUT, Store, canonical, digest, utc)
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import (
    ParallelBudget, WorkerStore, locked_json)
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as child_prompts
from experiments.query_decomposition.multigrain_prompt_v3_20261017 import prompts

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name / 'run_v001'
CAMPAIGN = OUTPUT / 'comprehensive_v1_20261015/campaign_api_v1'
LOCK = Lock()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(decision, reason, **fields):
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(canonical({'utc': utc(), 'decision': decision, 'reason': reason, **fields}) + '\n')


def prepare():
    from .freeze_rag import prepare as freeze_selected
    return freeze_selected()


class SmallBudget(ParallelBudget):
    def reserve(self, request_id, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            local = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                    (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
            if total + amount > 199.5 or local + amount > 2.5:
                raise RuntimeError('Global remaining allowance or local2.5CNY subcap reached')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (request_id,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


class OneShot:
    def __init__(self, store, budget, client, model, qid):
        self.store, self.budget, self.client, self.model, self.qid = store, budget, client, model, qid

    def ask(self, stage, prompt, maximum, *, allow_final_truncation=False):
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0., 'max_tokens': maximum, 'extra_body': {'enable_thinking': False}}
        rid = digest([self.store.namespace, payload])
        row = self.store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
        if row:
            if row[0] in {'unknown', 'in_flight'}:
                raise RuntimeError('Unknown submission retained; no replay')
            result = json.loads(row[1])
            self.store.event('logical_api_call', qid=self.qid, stage=stage, request_id=rid, cache_hit=True)
        else:
            size = len(prompt.encode('utf-8')) + 256
            if size > 100000:
                raise RuntimeError('Context too large; no truncation or submission')
            self.budget.wait_for_start()
            self.budget.reserve(rid, self.store.namespace, size * .8e-6 + maximum * 2e-6)
            self.store.db.execute("INSERT INTO requests VALUES(?,?,?,'in_flight',NULL)", (rid, stage, canonical(payload)))
            cursor = self.store.db.execute("INSERT INTO attempts(request_id,started,status) VALUES(?,?,'in_flight')", (rid, utc()))
            attempt = cursor.lastrowid
            self.store.db.commit()
            self.store.event('logical_api_call', qid=self.qid, stage=stage, request_id=rid, cache_hit=False)
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**payload)
                raw = response.model_dump(mode='json')
                usage = raw.get('usage') or {}
                choice = response.choices[0]
                result = {'status': 'success', 'answer': choice.message.content or '',
                          'finish_reason': choice.finish_reason, 'returned_model': response.model,
                          'usage': usage, 'raw_response': raw, 'latency_seconds': time.perf_counter() - started}
                if usage:
                    self.budget.settle(rid, usage.get('prompt_tokens', 0) * .8e-6 + usage.get('completion_tokens', 0) * 2e-6, 'returned_usage')
                if response.model != self.model or usage.get('prompt_tokens', 0) > 32768:
                    result['status'] = 'model_or_context_limit_mismatch'
            except Exception as exc:
                code = getattr(exc, 'status_code', None)
                body = getattr(exc, 'body', None)
                error = body.get('error', body) if isinstance(body, dict) else {}
                error_code = error.get('code') if isinstance(error, dict) else None
                result = {'status': 'failed' if code else 'unknown', 'http_status': code,
                          'failure_class': type(exc).__name__, 'error_code': error_code,
                          'latency_seconds': time.perf_counter() - started,
                          'replayed': False, 'unknown_liability_retained': not bool(code)}
                if code in {400, 401, 403, 404, 422, 429}:
                    self.budget.settle(rid, 0., 'http_failed_not_returned_usage')
                if code == 429:
                    self.budget.note_throttle()
            self.store.db.execute('UPDATE attempts SET status=?,result=? WHERE id=?', (result['status'], canonical(result), attempt))
            self.store.db.execute('UPDATE requests SET status=?,result=? WHERE id=?', (result['status'], canonical(result), rid))
            self.store.db.commit()
            self.store.export()
            locked_json(OUTPUT / 'budget_status.json', self.budget.summary())
        self.store.put('call_binding', digest([self.qid, stage, rid]),
                       {'qid': self.qid, 'stage': stage, 'request_id': rid})
        if result['status'] != 'success':
            raise RuntimeError(f"Provider request not successful: {result['status']} {result.get('http_status')}")
        finish = result.get('finish_reason')
        if finish != 'stop' and not (allow_final_truncation and finish == 'length'):
            raise RuntimeError('Incomplete intermediate response retained; not propagated')
        return rid, result['answer'].strip(), finish


def passages(packet):
    return '\n\n'.join(d['block'] for d in packet['docs'])


def merge(original, rewritten):
    if rewritten is None:
        return original
    docs, seen = [], set()
    for rank in range(10):
        for route in (original, rewritten):
            if rank < len(route['docs']):
                d = route['docs'][rank]
                if d['doc_id'] not in seen:
                    seen.add(d['doc_id'])
                    docs.append(d)
    docs = [{**d, 'rank': i, 'block': f'[Chunk {i} | source={d["source"]}, pages=1-1]\n{d["text"]}'}
            for i, d in enumerate(docs[:10], 1)]
    assert len(docs) == len({d['doc_id'] for d in docs}) == 10
    return {'qid': original['qid'], 'query': original['query'], 'k': 10,
            'docs': docs, 'merge_policy': 'interleave_original_first_top10'}


def execute(case, owner, retrieval, client, config, examples):
    store = WorkerStore(owner)
    budget = SmallBudget(OUTPUT / 'budget.sqlite3', 200)
    transport = OneShot(store, budget, client, config['model']['reader'], case['qid'])
    def retrieve(query):
        with LOCK:
            previous = retrieval.store
            retrieval.store = store
            try:
                return retrieval.retrieve(case['qid'], query, 10)
            finally:
                retrieval.store = previous
    result = []
    try:
        root_packet = retrieve(case['question'])
        root_context = passages(root_packet)
        for arm in config['conditions']:
            target = OUT / 'cases' / case['dataset'] / arm / (case['qid'] + '.json')
            if target.exists():
                result.append({'arm': arm, 'status': read(target)['status'], 'reused': True})
                continue
            nodes, history = [], []
            try:
                if arm == 'B_r02_actual':
                    requests = case['supporting_questions'] + [case['question']]
                    for sid, question in enumerate(requests, 1):
                        original = retrieve(question)
                        rewrite, rewrite_id, rejection = question, None, None
                        if history:
                            rewrite_id, candidate, _ = transport.ask(f'{arm}/rewrite/{sid}', prompts.rewrite(question, history), 256)
                            if candidate and candidate.endswith('?') and '\n' not in candidate:
                                rewrite = candidate
                            else:
                                rejection = 'not_one_complete_question; original retained'
                        rewritten = retrieve(rewrite) if rewrite != question else None
                        packet = merge(original, rewritten)
                        is_goal = sid == len(requests)
                        child = child_prompts.child(case['question'], sid, question, history, passages(packet), examples['child'],
                                                    role='goal' if is_goal else 'ordinary', combine=False)
                        answer_id, answer, _ = transport.ask(f'{arm}/answer/{sid}', child, 128)
                        initial_answer, format_id = answer, None
                        if len(answer.split()) < 4:
                            text = ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                                    'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                                    'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                                    + 'Question:\n' + question + '\n\nCandidate answer:\n' + answer + '\n\nContext:\n' + passages(packet))
                            format_id, answer, _ = transport.ask(f'{arm}/sentence_format/{sid}', text, 128)
                        record = {'step_id': sid, 'request': question, 'answer': answer}
                        node = {**record, 'role': 'goal' if is_goal else 'ordinary', 'prior_complete_answers': list(history),
                                'original_retrieval': original, 'rewritten_query': rewrite, 'rewritten_retrieval': rewritten,
                                'reader_packet': packet, 'rewrite_request_id': rewrite_id, 'rewrite_rejection': rejection,
                                'answer_request_id': answer_id, 'initial_answer': initial_answer, 'sentence_format_request_id': format_id,
                                'original_doc_ids_displaced': sorted({d['doc_id'] for d in original['docs']} - {d['doc_id'] for d in packet['docs']})}
                        nodes.append(node)
                        store.put('node', case['qid'] + '/' + str(sid), {'qid': case['qid'], **node})
                        history.append(record)
                helper = child_prompts.render_helper(history, None, 'Answer') if history else ''
                context = helper + '\n\n' + root_context if helper else root_context
                final_prompt = prompts.final(case['question'], context, examples['final'])
                final_id, prediction, finish = transport.ask(arm + '/final', final_prompt, 64, allow_final_truncation=True)
                output = {**case, 'condition': arm, 'status': 'success' if finish == 'stop' else 'success_truncated',
                          'nodes': nodes, 'helper': helper, 'original_retrieval': root_packet,
                          'helper_position': 1 if helper else None, 'final_context': context, 'final_prompt': final_prompt,
                          'prediction': prediction, 'request_id': final_id, 'finish_reason': finish,
                          'original_top10_doc_ids': [d['doc_id'] for d in root_packet['docs']]}
            except Exception as exc:
                output = {**case, 'condition': arm, 'status': 'incomplete', 'nodes': nodes,
                          'error_type': type(exc).__name__, 'reason': str(exc)[:240]}
            locked_json(target, output)
            store.put('prediction', arm + '/' + case['qid'], {'qid': case['qid'], 'condition': arm,
                      'prediction': output.get('prediction'), 'status': output['status'], 'request_id': output.get('request_id')})
            result.append({'arm': arm, 'status': output['status']})
        return {'dataset': case['dataset'], 'qid': case['qid'], 'conditions': result}
    finally:
        store.db.close()
        budget.db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = prepare()
    assert sha(HERE / 'new_errors.runtime.frozen.jsonl') == config['runtime_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in config['source_sha256'].items())
    cases = rows(HERE / 'new_errors.runtime.frozen.jsonl')
    assert all(not {'answer', 'verdict', 'failed_dimensions'} & set(c) for c in cases)
    if args.prepare_only:
        print(json.dumps({'prepared': len(cases), 'conditions': config['conditions'], 'api_calls': 0}))
        return
    options = get_api_option(2)
    assert options['model'] == config['model']['reader']
    install()
    examples = read(ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json')
    examples['final'] = read(ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json')
    locked_json(OUT / 'EXECUTION_IDENTITY.json', {'utc': utc(), 'protocol_sha256': sha(HERE / 'RAG_PROTOCOL.frozen.json'),
                                                'runner_sha256': sha(Path(__file__)), 'gold_in_runtime': False})
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2, keepalive_expiry=15)), timeout=120))
        client = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=options['base_url'],
                                            timeout=120, max_retries=0, http_client=network))
        streaming = StreamingClient(client)
        for dataset in ['hotpotqa', '2wiki', 'musique']:
            selected = [c for c in cases if c['dataset'] == dataset]
            if not selected:
                continue
            owner = Store(OUT / 'api' / dataset, {'task': 'expanded_actual_R02_errors', 'dataset': dataset,
                'protocol_sha256': sha(HERE / 'RAG_PROTOCOL.frozen.json'), 'resources': config['resources'][dataset]})
            retrieval = Retrieval(owner, resources=config['resources'][dataset])
            try:
                with provider_dns(urlsplit(options['base_url']).hostname, owner):
                    # Verified TLS connectivity before any paid request or reservation.
                    host = urlsplit(options['base_url']).hostname
                    started = time.perf_counter()
                    with socket.create_connection((host, 443), timeout=15) as sock:
                        with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                            owner.event('provider_tls_preflight', passed=True, seconds=time.perf_counter() - started,
                                        credentials_sent=False, paid_request=False)
                    print(json.dumps({'dataset': dataset, 'stage': 'load_retrieval', 'n': len(selected)}), flush=True)
                    retrieval.load()
                    fixed = sorted({(c['qid'], q) for c in selected for q in c['supporting_questions'] + [c['question']]})
                    retrieval.retrieve_many([{'qid': qid, 'query': q} for qid, q in fixed], 10)
                    locked_json(OUT / (dataset + '.PREFLIGHT.json'), {'original_and_static_child_packets': len(fixed),
                                'all_complete_top10': True, 'resource_documents': config['resources'][dataset]['corpus']['documents']})
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        jobs = [pool.submit(execute, case, owner, retrieval, streaming, config, examples) for case in selected]
                        for future in as_completed(jobs):
                            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
            finally:
                if retrieval.chunks is not None:
                    retrieval.chunks.close()
                owner.export(full=True)
                owner.db.close()
    locked_json(OUT / 'RUN_COMPLETE.json', {'utc': utc(), 'planned_queries': len(cases),
                'terminal_arm_records': len(list((OUT / 'cases').glob('*/*/*.json'))), 'scoring': 'offline pending'})
    print('All expanded planned queries attempted; run offline scorer next.', flush=True)


if __name__ == '__main__':
    main()
