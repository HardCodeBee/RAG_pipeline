"""Cross optional partial binding and final auxiliary reading; freeze child and bounded reuse."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import json
import os
import sqlite3
from threading import Lock
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, digest, canonical
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as old_prompts
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.run import merge_packets, passages
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write, CAMPAIGN
from experiments.query_decomposition.multigrain_prompt_v3_20261017.run import ask
from experiments.query_decomposition.multigrain_prompt_v2_20261017 import prompts as frozen_prompts
from .prepare import HERE, OUT, ROOT, OUTPUT, OLD_OUT, V2, V3, S02, sha
from experiments.query_decomposition.multigrain_evidence_v4r1_20261017 import prompts as child_prompts
from . import prompts

RETRIEVAL_LOCK = Lock()


class RoundBudget(ParallelBudget):
    def reserve(self, request_id, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            local = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT))+'%',)).fetchone()[0]
            if total+amount > 195 or local+amount > 5:
                raise RuntimeError('Protected global budget or round cap reached')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (request_id,)).fetchone():
                raise RuntimeError('Existing liability must not be resubmitted')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (request_id, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def seed_sources(owner, selected, resources):
    ds = selected[0]['dataset']
    historical = OLD_OUT/'musique_baseline_screening_network_v2/checkpoint.sqlite3' if ds == 'musique' else CAMPAIGN/ds/'formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3'
    sources = [historical, CAMPAIGN/ds/'formal/b00_top10_v1/k10/checkpoint.sqlite3', OLD_OUT/'api'/ds/'checkpoint.sqlite3']
    sources += [OUTPUT/p.name/'api'/ds/'checkpoint.sqlite3' for p in (V2, V3)]
    # Locate the original baseline directory by method identity when its frozen config label differs.
    sources += list((CAMPAIGN/ds/'formal/b00_top10_v1').glob('*/checkpoint.sqlite3'))
    sources.append(OUTPUT/'multigrain_evidence_v4_20261017'/'api'/ds/'checkpoint.sqlite3')
    sources.append(OUTPUT/'multigrain_evidence_v4r1_20261017'/'api'/ds/'checkpoint.sqlite3')
    known = {}
    for path in dict.fromkeys(sources):
        if not path.exists():
            continue
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            for case in selected:
                for key, raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval' AND json_extract(value,'$.qid')=?", (case['qid'],)):
                    packet = json.loads(raw)
                    assert key == digest([packet['qid'],packet['query'],packet['k'],resources])
                    owner.put('retrieval', key, packet)
            for rid, stage, payload, status, raw in db.execute("SELECT * FROM requests WHERE status='success'"):
                parsed_payload = json.loads(payload)
                messages = '\n'.join(m.get('content','') for m in parsed_payload.get('messages',[]))
                if not any(c['question'] in messages for c in selected):
                    continue
                result = json.loads(raw)
                if result.get('status') == 'success' and result.get('finish_reason') == 'stop':
                    known.setdefault(digest(json.loads(payload)), {'id': rid, 'stage': stage, 'payload': payload,
                        'status': status, 'result': raw, 'origin': str(path.relative_to(ROOT))})
    return known


def pooled_packet(current, original, nodes, sid, cap):
    """Retain reader input; append deterministic, unique already-retrieved passages."""
    docs = list(current['docs'])
    seen = {d['doc_id'] for d in docs}
    donors = [original] + [n['reader_packet'] for n in nodes if n['step_id'] != sid]
    added = []
    for rank in range(10):
        for donor in donors:
            if rank < len(donor['docs']):
                doc = donor['docs'][rank]
                if doc['doc_id'] not in seen:
                    seen.add(doc['doc_id'])
                    added.append(doc)
                    if len(added) == cap:
                        break
        if len(added) == cap:
            break
    for doc in added:
        rank = len(docs)+1
        docs.append({**doc, 'rank': rank, 'block': f'[Chunk {rank} | source={doc["source"]}, pages=1-1]\n{doc["text"]}'})
    return {**current, 'k': len(docs), 'docs': docs, 'reuse_added_doc_ids': [d['doc_id'] for d in added]}


def execute_case(case, owner, retrieval, known, bindings, client, protocol, examples):
    store = WorkerStore(owner)
    budget = RoundBudget(OUTPUT/'budget.sqlite3', 200)
    transport = ResumeTransport(store, client, budget, protocol['model'])
    transport.qid = case['qid']
    def query(text):
        with RETRIEVAL_LOCK:
            previous = retrieval.store
            retrieval.store = store
            try:
                return retrieval.retrieve(case['qid'], text, 10)
            finally:
                retrieval.store = previous
    def call(stage, prompt, maximum):
        return ask(transport, known, bindings, stage, prompt, maximum)
    def reread_call(stage, prompt, maximum):
        try:
            rid, answer = call(stage, prompt, maximum)
            return rid, answer, None
        except RuntimeError:
            # A complete provider response ending at its token limit is known,
            # paid, and unusable. Preserve the first pass; never replay it.
            payload = {'model': transport.model, 'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0, 'max_tokens': maximum, 'extra_body': {'enable_thinking': False}}
            matches = store.db.execute("SELECT id,result FROM requests WHERE payload=? AND status='success'", (canonical(payload),)).fetchall()
            for rid, raw in matches:
                result = json.loads(raw)
                if result.get('status') == 'success' and result.get('finish_reason') == 'length':
                    return rid, None, result.get('answer','')
            raise
    try:
        original = query(case['question'])
        position = load(ROOT/f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{case['dataset']}/formal.positions.json")[case['qid']]
        for branch in ('new', 'previous', 'single'):
            for mode in ('partial',):
                label = mode+'_'+branch
                labels = ['reuse_'+branch, 'support_'+branch]
                if all((OUT/'cases'/case['dataset']/l/(case['qid']+'.json')).exists() for l in labels):
                    continue
                history, nodes = [], []
                for step in case['qdmr']['steps']:
                    sid, question = step['id'], step['request']
                    base = query(question)
                    rewrite_id, rewritten, rejection = None, question, None
                    if branch != 'single' and history:
                        rewrite_id, proposed = call(label+f'/rewrite/{sid}', (prompts.rewrite(question, history) if branch == 'new' else frozen_prompts.rewrite(question, history)), protocol['rewrite_tokens'])
                        if proposed and proposed.endswith('?') and '\n' not in proposed:
                            rewritten = proposed
                        else:
                            rejection = 'not_one_complete_question; original retained'
                    changed = query(rewritten) if rewritten != question else None
                    reader = merge_packets(base, changed, 'interleave_top10')
                    prompt = child_prompts.child(case['question'], question, history, passages(reader), examples['child'])
                    answer_id, raw = call(label+f'/answer/{sid}', prompt, protocol['child_tokens'])
                    state, answer = child_prompts.parse_answer(raw)
                    format_id = None
                    record = {'step_id': sid, 'request': question, 'answer': answer}
                    nodes.append({**record, 'prior_complete_answers': list(history), 'rewritten_query': rewritten,
                        'rewrite_request_id': rewrite_id, 'rewrite_rejection': rejection, 'original_retrieval': base,
                        'rewritten_retrieval': changed, 'reader_packet': reader, 'answer_request_id': answer_id,
                        'initial_answer': raw, 'sentence_format_request_id': format_id, 'answer_state': state})
                    history.append(record)
                for result_label in labels:
                    path = OUT/'cases'/case['dataset']/result_label/(case['qid']+'.json')
                    if path.exists():
                        continue
                    result_nodes = deepcopy(nodes)
                    if result_label.startswith(('reuse','support')):
                        for node in result_nodes:
                            if node['answer_state'] == 'complete':
                                node['reread_decision'] = 'self_reported_complete'
                                continue
                            pool = pooled_packet(node['reader_packet'], original, nodes, node['step_id'], protocol['reuse_added_document_cap'])
                            node['reread_decision'] = 'once' if pool['reuse_added_doc_ids'] else 'no_new_material'
                            if pool['reuse_added_doc_ids']:
                                # All donor QA are fixed first-pass outputs. No retry chain or new search is permitted.
                                fixed = [{'step_id': n['step_id'], 'request': n['request'], 'answer': n['answer']} for n in nodes if n['step_id'] != node['step_id']]
                                retry_id, raw, truncated = reread_call(result_label+f"/reread/{node['step_id']}", child_prompts.child(case['question'], node['request'], fixed, passages(pool), examples['child']), protocol['child_tokens'])
                                node['first_pass_answer'] = node['answer']
                                node['first_pass_state'] = node['answer_state']
                                node['reread_request_id'] = retry_id
                                node['reread_packet'] = pool
                                node['reread_raw_answer'] = raw if truncated is None else truncated
                                if truncated is None:
                                    node['answer_state'], node['answer'] = child_prompts.parse_answer(raw)
                                else:
                                    node['reread_decision'] = 'truncated_kept_first_pass'
                                    node['reread_finish_reason'] = 'length'
                    final_history = [{k: n[k] for k in ('step_id', 'request', 'answer')} for n in result_nodes]
                    helper = old_prompts.render_helper(final_history, None, 'Answer')
                    blocks = [d['block'] for d in original['docs']]
                    blocks.insert(position-1, helper)
                    context = '\n\n'.join(blocks)
                    prompt = (prompts.final(case['question'], context, examples['support_final']) if result_label.startswith('support') else frozen_prompts.final(case['question'], context, examples['final']))
                    rid, prediction = call(result_label+'/final', prompt, protocol['final_tokens'])
                    result = {'qid': case['qid'], 'dataset': case['dataset'], 'question': case['question'],
                        'evaluation_group': case['evaluation_group'], 'condition': result_label, 'status': 'success',
                        'nodes': result_nodes, 'helper': helper, 'final_context': context, 'final_prompt': prompt,
                        'prediction': prediction, 'request_id': rid, 'helper_position': position,
                        'original_top10_doc_ids': [d['doc_id'] for d in original['docs']]}
                    write(path, result)
        return {'dataset': case['dataset'], 'qid': case['qid'], 'group': case['evaluation_group'], 'conditions': 6}
    finally:
        store.db.close()
        budget.db.close()


def main():
    protocol = load(HERE/'protocol.json')
    assert all(sha(ROOT/p) == value for p, value in protocol['frozen_files'].items())
    cases = rows(HERE/'cases.frozen.jsonl')
    blocked_qids = {r['qid'] for r in load(OUT/'KNOWN_PROVIDER_REJECTIONS.json')} if (OUT/'KNOWN_PROVIDER_REJECTIONS.json').exists() else set()
    examples = {'old_child': load(S02/'examples.json')['child'], 'child': load(HERE/'child_examples.json'), 'final': load(V2/'final_examples.json'), 'support_final': load(HERE/'final_examples.json')}
    write(OUT/'EXECUTION_FROZEN.json', {'protocol': protocol, 'runner_sha256': sha(HERE/'run.py'), 'gold_in_generation': False})
    options = get_api_option(2)
    assert options['model'] == protocol['model']
    install()
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'))
        network = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0, limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15)), timeout=120))
        client = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=options['base_url'], timeout=120, max_retries=0, http_client=network))
        streaming = StreamingClient(client)
        for ds in ('hotpotqa', '2wiki', 'musique'):
            selected = [c for c in cases if c['dataset'] == ds]
            resources = load(CAMPAIGN/ds/'RESOURCE_FROZEN.json')['resources']
            owner = Store(OUT/'api'/ds, {'task': 'optional_partial_binding', 'dataset': ds, 'protocol_sha256': sha(HERE/'protocol.json'), 'resources': resources})
            known = seed_sources(owner, selected, resources)
            bindings = []
            for c in selected:
                for label in list(protocol['conditions'])+['partial_new','partial_previous','partial_single']:
                    bindings.append({'qid': c['qid'], 'stage': label+'/final'})
                    for n in c['qdmr']['steps']:
                        bindings += [{'qid': c['qid'], 'stage': label+'/'+kind+'/'+str(n['id'])} for kind in ('rewrite', 'answer', 'sentence_format', 'reread')]
            retrieval = Retrieval(owner, resources=resources)
            print(json.dumps({'dataset': ds, 'loading': True, 'questions': len(selected)}), flush=True)
            try:
                with provider_dns(urlsplit(options['base_url']).hostname, owner):
                    retrieval.load()
                    fixed = {(c['qid'],n['request']) for c in selected for n in c['qdmr']['steps']}
                    retrieval.retrieve_many([{'qid':qid,'query':q} for qid,q in sorted(fixed)],10)
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        tasks = {pool.submit(execute_case, c, owner, retrieval, known, bindings, streaming, protocol, examples): c for c in selected if c['qid'] not in blocked_qids}
                        failures = []
                        for future in as_completed(tasks):
                            try:
                                print(json.dumps(future.result()), flush=True)
                            except BaseException as exc:
                                failures.append({'qid': tasks[future]['qid'], 'type': type(exc).__name__, 'reason': str(exc)[:240]})
                                for pending in tasks:
                                    pending.cancel()
                        if failures:
                            write(OUT/'ERRORS.json', failures)
                            raise RuntimeError('Generation stopped; inspect checkpoint')
            finally:
                if retrieval.chunks is not None:
                    retrieval.chunks.close()
                owner.export()
                owner.db.close()
    if (OUT/'ERRORS.json').exists():
        write(OUT/'RECOVERY_COMPLETE.json', {'previous_errors_retained':True, 'optional_length_rule':'keep_first_pass', 'known_rejection_qids':sorted(blocked_qids), 'all_planned_outputs_complete':len(list((OUT/'cases').glob('*/*/*.json')))==len(cases)*len(protocol['conditions'])})
    print('All runnable questions completed; known provider rejections remain in planned denominator.', flush=True)


if __name__ == '__main__':
    main()
