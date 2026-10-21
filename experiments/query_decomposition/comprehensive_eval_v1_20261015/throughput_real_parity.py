"""Actual corpus/reranker parity for execution acceleration; zero paid calls."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Lock
import json
import sqlite3
import time

from .common import ROOT, HERE, RUN, read, rows, save, sha, log
from .campaign import TARGET
from .qd_batch_epoch import batch_retrieve
from .retrieval import Retrieval
from .throughput_epoch import reranker, SlotAPI
from .throughput_validate import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.hybrid_rerank import method as hybrid


def slots_check():
    class Probe(SlotAPI):
        def __init__(self):
            self.pool = ThreadPoolExecutor(max_workers=3)
            self.concurrency = 3
            self.lock = Lock()
            self.active = self.maximum = self.completed = 0
        def _ask(self, qid, args, kwargs):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(.02 if int(qid) % 3 else .05)
            with self.lock:
                self.active -= 1
                self.completed += 1
            return qid
    probe = Probe()
    operations = [SimpleNamespace(index=i, args=(), kwargs={}) for i in range(12)]
    questions = [{'qid': str(i)} for i in range(12)]
    try:
        result = probe.ask_group(operations, questions)
    finally:
        probe.pool.shutdown(wait=True)
    assert result == [str(i) for i in range(12)] and probe.maximum == 3 and probe.completed == 12
    return {'ready_operations': 12, 'maximum_concurrent_calls': probe.maximum,
            'all_drained': probe.active == 0, 'returned_in_controller_order': True}


def compare(actual, expected):
    assert [r['doc_id'] for r in actual['docs']] == [r['doc_id'] for r in expected['docs']], 'Dense rank changed'
    assert [r['block'] for r in actual['docs']] == [r['block'] for r in expected['docs']], 'Dense passage text changed'
    return max(abs(a['score'] - b['score']) for a, b in zip(actual['docs'], expected['docs']))


def main():
    from .offset_compatibility import install
    install()
    started = time.perf_counter()
    offline = read(RUN / 'THROUGHPUT_OFFLINE_PARITY.json')
    assert offline['passed'] and offline['execution_source_sha256'] == sha(HERE / 'throughput_epoch.py')
    directory = TARGET / 'hotpotqa/formal/b07_hybrid_rerank/k20_rrf60'
    selected = []
    with sqlite3.connect((directory / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        for key, raw in db.execute("SELECT key,value FROM data WHERE kind='prediction' LIMIT 8"):
            prediction = json.loads(raw)
            data = {(kind, key): json.loads(value) for kind, key, value in db.execute(
                "SELECT kind,key,value FROM data WHERE json_extract(value,'$.qid')=?", (key,))}
            dense = next(r for (kind, _), r in data.items() if kind == 'retrieval')
            sparse = next(r for (kind, _), r in data.items() if kind == 'sparse_retrieval')
            packet = next(r for (kind, _), r in data.items() if kind == 'hybrid')
            payload = json.loads(db.execute('SELECT payload FROM requests WHERE id=?', (prediction['request_id'],)).fetchone()[0])
            selected.append({'qid': key, 'question': prediction['question'], 'dense': dense, 'sparse': sparse,
                             'hybrid': packet, 'prompt': payload['messages'][0]['content']})
    progress = RUN / 'THROUGHPUT_REAL_PARITY_PROGRESS.json'
    save(progress, {'stage': 'hotpot_dense_prefetch', 'queries': len(selected)})
    cache = Store()
    retrieval = Retrieval(cache, resources=read(ROOT / 'experiments/query_decomposition/query_form_ablation_v1/resources.json'))
    retrieval_started = time.perf_counter()
    packets = batch_retrieve(retrieval, [{'qid': r['qid'], 'query': r['question']} for r in selected], 20)
    load_search_seconds = time.perf_counter() - retrieval_started
    max_diff = max(compare(packet, ref['dense']) for packet, ref in zip(packets, selected))
    # Repeat warm: compare to the already-recorded individual-search wall time.
    retrieval.store = Store()
    warm_start = time.perf_counter()
    warm = batch_retrieve(retrieval, [{'qid': r['qid'], 'query': r['question']} for r in selected], 20)
    warm_seconds = time.perf_counter() - warm_start
    assert all(compare(a, b) <= 1e-6 for a, b in zip(warm, packets))
    save(progress, {'stage': 'hotpot_original_rerank_and_prompt_parity', 'dense_queries_matched': len(packets)})
    model = reranker()
    for packet, ref in zip(packets, selected):
        union = hybrid.fuse(packet['docs'], ref['sparse']['docs'], 60)
        docs, _ = model.rerank(ref['question'], union, 10)
        assert [d['doc_id'] for d in docs] == [d['doc_id'] for d in ref['hybrid']['docs']], 'Rerank order changed'
        assert hybrid.PROMPT.format(question=ref['question'], context='\n\n'.join(d['block'] for d in docs)) == ref['prompt'], 'SDK final prompt changed'
    assert reranker() is model, 'Model was reloaded'
    save(RUN / 'THROUGHPUT_HOTPOT_PARITY_CHECKPOINT.json', {
        'passed': True, 'api_submissions': 0, 'queries': len(selected),
        'dense_order_text_reranked_order_and_final_sdk_prompts_matched': len(selected),
        'search_seconds_including_load': load_search_seconds,
        'warm_batched_search_seconds': warm_seconds,
        'saved_single_search_seconds_sum': sum(r['dense']['batch_seconds'] for r in selected),
        'max_dense_score_absolute_difference': max_diff,
        'execution_source_sha256': sha(HERE / 'throughput_epoch.py')})
    native = {}
    for dataset in ['2wiki', 'musique']:
        save(progress, {'stage': dataset + '_single_vs_batch_dense', 'completed_hotpot_sdk_prompts': len(selected)})
        resources = read(HERE / 'resources_frozen' / (dataset + '.json'))
        cases = rows(HERE / 'inputs_frozen' / dataset / 'formal.questions.jsonl')[:4]
        reference = Retrieval(Store(), resources=resources)
        single = [reference.retrieve(c['qid'], c['question'], 20) for c in cases]
        baseline_seconds = sum(r['batch_seconds'] for r in single)
        reference.store = Store()
        tick = time.perf_counter()
        found = batch_retrieve(reference, [{'qid': c['qid'], 'query': c['question']} for c in cases], 20)
        seconds = time.perf_counter() - tick
        difference = max(compare(a, b) for a, b in zip(found, single))
        native[dataset] = {'queries': 4, 'doc_order_and_prompt_passages_matched': 4,
                           'max_score_absolute_difference': difference, 'single_search_seconds_sum': baseline_seconds,
                           'batched_search_seconds': seconds}
        reference.chunks.close()
        del reference
    result = {'passed': True, 'api_submissions': 0, 'execution_source_sha256': sha(HERE / 'throughput_epoch.py'),
              'validation_source_sha256': sha(__file__), 'offline_parity_sha256': sha(RUN / 'THROUGHPUT_OFFLINE_PARITY.json'),
              'offline_cases': offline['cases'], 'offline_sdk_calls_byte_matched': offline['sdk_calls_byte_matched'],
              'retrieval_grouping_identical_for_all_replayed_methods': all(r['retrieval_grouping_identical'] for r in offline['methods'].values()),
              'hotpot_hybrid_queries': len(packets), 'hotpot_dense_order_and_text_matches': len(packets),
              'hotpot_reranked_order_and_sdk_prompt_matches': len(selected), 'max_dense_score_absolute_difference': max_diff,
              'hotpot_search_seconds_including_load': load_search_seconds, 'hotpot_warm_batched_search_seconds': warm_seconds,
              'hotpot_saved_single_search_seconds_sum': sum(r['dense']['batch_seconds'] for r in selected),
              'native_dataset_dense_parity': native, 'production_api_slot_probe': slots_check(),
              'offset_compatibility_source_sha256': sha(HERE / 'offset_compatibility.py'),
              'model_instance_reused': True, 'total_validation_seconds': time.perf_counter() - started,
              'scope': 'purposive real cached cases; parity is empirical and timings are local microbenchmarks'}
    save(RUN / 'THROUGHPUT_PARITY.json', result)
    save(progress, {'stage': 'completed', **result})
    log('throughput_real_corpus_and_prompt_parity_passed', **result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
