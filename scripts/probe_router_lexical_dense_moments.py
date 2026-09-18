"""Corpus-only feasibility probe for static BM25-weighted Dense term moments.

No task questions, labels, retrieval ranking, model inference, or API calls.
This is not a Router training/effect experiment or a complete feature service.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SPARSE = ROOT/'artifacts/_sparse_indexes/sqlite_bm25_04df51906ff9598b'
DENSE = ROOT/'artifacts/_encoded_corpora/encoded_corpus_df8e4c53d3e183b4'
OUT = ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/lexical_dense_moments_feasibility_v1'
SPEC = {
    'role': 'corpus_only_static_representation_feasibility_no_answer_quality_claim',
    'source_build': 'build_65e9037e4c1eed7b',
    'sparse_index': SPARSE.name, 'encoded_corpus': DENSE.name,
    'df_intervals': [[2,10],[10,50],[50,200],[200,1000]],
    'terms_per_interval': 8, 'selection_seed': 2026091881,
    'selection': 'lowest_SHA256_seed_pipe_term_per_DF_interval_over_existing_term_stats',
    'postings': 'all_postings_of_selected_terms_offline_only_no_top_k',
    'impact': 'existing_lucene_idf_times_tf_times_2.5_div_tf_plus_1.5_times_0.25_plus_0.75_dl_div_avgdl',
    'stored': ['sum_b','sum_b_squared','sum_b_times_dense','sum_b_times_dense_squared_norm'],
    'document_encoding': 'reuse_existing_normalized_model_default_BGE384_max512_no_new_encoder',
    'query_use': 'none; future_dot_products_require_matching_Dense_query_space_not_M6_mean6',
    'reduction_dtype': 'float64', 'threads': 2,
    'limits': '32_terms_DF2_to999_not_vocabulary_coverage_or_whole_corpus_cost_estimate',
    'stop': 'one_pilot_no_term_selection_or_dimension_search',
    'query_reads': 0, 'label_reads': 0, 'model_fits': 0,
    'retained_query_reads': 0, 'paid_calls': 0,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT/'protocol.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spec': SPEC,
        'status': 'frozen_before_term_selection_or_vector_reads',
    })
    print('Frozen corpus-only 32-term feasibility protocol.', flush=True)


def term_moments(weights, vectors):
    """Exact moments for one term over all of its nonzero postings."""
    weights = np.asarray(weights, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    mass = float(weights.sum())
    if weights.ndim != 1 or vectors.ndim != 2 or len(weights) != len(vectors):
        raise ValueError('Expected matched weights and document vectors')
    if not mass > 0 or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isfinite(vectors).all():
        raise ValueError('Expected nonnegative finite impacts with positive mass and finite vectors')
    weighted_vector = weights @ vectors
    squared_mass = float(weights @ weights)
    weighted_norm2 = float(weights @ np.einsum('ij,ij->i', vectors, vectors))
    centroid = weighted_vector/mass
    return mass, squared_mass, weighted_vector, weighted_norm2, centroid


def compose_query(term_ids, masses, weighted_vectors, query_vector, corpus_count, corpus_mean):
    """Lookup-only composition, for complete known term IDs and matching space.

    corpus_mean must be the mean over the same corpus measure as the moments.
    This prototype does not define an incomplete-vocabulary fallback policy.
    """
    ids = np.unique(np.asarray(term_ids, dtype=np.int64))
    if len(ids) == 0:
        raise ValueError('No known query terms')
    mass = float(np.asarray(masses)[ids].sum())
    weighted_vector = np.asarray(weighted_vectors)[ids].sum(axis=0)
    q = np.asarray(query_vector)
    return {
        'lexically_weighted_dense_mean': float(q @ (weighted_vector/mass)),
        'score_covariance': float(q @ (weighted_vector-mass*np.asarray(corpus_mean))/corpus_count),
        'centroid_squared_norm': float((weighted_vector/mass) @ (weighted_vector/mass)),
    }


def math_check():
    """Three one-time mathematical checks, including a limitation example."""
    assert read(OUT/'protocol.json')['spec'] == SPEC
    if (OUT/'math_checks.json').exists():
        raise RuntimeError('Mathematical check already recorded; do not repeat')
    e = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    weights = np.array([[2.,1.,0.,0.],[0.,1.,2.,0.]])
    q = np.array([.6,.8])
    moments = [term_moments(w,e) for w in weights]
    composed = compose_query([0,1,0], [m[0] for m in moments], [m[2] for m in moments], q,4,e.mean(0))
    B, D = weights.sum(0), e@q
    direct = {'lexically_weighted_dense_mean':float(B@D/B.sum()),
              'score_covariance':float(np.mean((B-B.mean())*(D-D.mean()))),
              'centroid_squared_norm':float(np.linalg.norm(B@e/B.sum())**2)}
    errors = {key:abs(composed[key]-value) for key,value in direct.items()}
    assert max(errors.values()) < 1e-12

    # Same lexical marginals and global vector multiset; pairing changes covariance.
    one_term = weights[0]
    flipped = e[[2,1,0,3]]
    original = term_moments(one_term,e)
    changed = term_moments(one_term,flipped)
    marginal_covariances = [compose_query([0],[m[0]],[m[2]],[1.,0.],4,[0.,0.])['score_covariance']
                            for m in [original,changed]]
    assert marginal_covariances == [.5,-.5]
    assert original[0] == changed[0] and original[1] == changed[1] and original[3] == changed[3]

    # Same proposed moments as well, but distinct Dense top1 and overlap with BM25.
    b = np.array([3.,2.,1.,0.])
    second = np.array([[0.,1.],[1.,0.],[0.,-1.],[-1.,0.]])
    left, right = term_moments(b,e), term_moments(b,second)
    for a,c in zip(left,right):
        assert np.array_equal(a,c)
    top1 = [int(np.argmax(x@np.array([1.,0.]))) for x in [e,second]]
    assert top1 == [0,1] and int(np.argmax(b)) == 0
    record = {
        'status':'passed_three_mathematical_checks_once',
        'lookup_vs_direct_full_score_moments_errors':errors,
        'duplicate_query_term_dedup_checked':True,
        'same_marginals_different_pairing_covariances':marginal_covariances,
        'same_joint_moments_top1_counterexample':{
            'b':b.tolist(),'vectors_I':e.tolist(),'vectors_II':second.tolist(),
            'q':[1.,0.], 'sum_b':left[0],'sum_b_squared':left[1],
            'sum_b_times_dense':left[2].tolist(),'sum_b_times_norm_squared':left[3],
            'centroid':left[4].tolist(),'score_covariance':.5,
            'BM25_top1_index':0,'Dense_top1_indices':top1,
        },
        'scope':'synthetic_nonnegative_impact_weights_and_unit_vectors_not_natural_queries_or_a_new_theorem',
        'conclusion':'joint_information_beyond_marginals_but_not_sufficient_for_top_k_or_answer_action',
        'real_query_reads':0,'label_reads':0,'model_fits':0,
    }
    write_new(OUT/'math_checks.json',record)
    print(json.dumps(record),flush=True)


def pilot():
    assert read(OUT/'protocol.json')['spec'] == SPEC
    if (OUT/'summary.json').exists() or (OUT/'moments.npz').exists():
        raise RuntimeError('Pilot already has saved artifacts; do not rerun')
    sparse_meta, dense_meta = read(SPARSE/'manifest.json'), read(DENSE/'manifest.json')
    assert sparse_meta['identity']['source_build_id'] == SPEC['source_build']
    assert sparse_meta['identity']['source_chunks']['sha256'] == dense_meta['artifacts']['chunks']['sha256']
    assert sparse_meta['document_count'] == dense_meta['corpus']['num_documents']
    assert sparse_meta['identity']['bm25']['k1'] == 1.5 and sparse_meta['identity']['bm25']['b'] == .75
    parts = dense_meta['artifacts']['embeddings']['parts']
    began = time.perf_counter()
    heaps = [[] for _ in SPEC['df_intervals']]
    populations = [0]*len(heaps)
    selected, posting_rows = [], []
    with sqlite3.connect((SPARSE/'index.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for term, df, idf in db.execute('SELECT term, df, idf FROM term_stats WHERE df >= 2 AND df < 1000'):
            stratum = next(i for i,(lo,hi) in enumerate(SPEC['df_intervals']) if lo <= df < hi)
            populations[stratum] += 1
            priority = int.from_bytes(hashlib.sha256(f'{SPEC["selection_seed"]}|{term}'.encode()).digest(), 'big')
            item = (-priority, term, df, idf)
            if len(heaps[stratum]) < SPEC['terms_per_interval']:
                heapq.heappush(heaps[stratum], item)
            elif priority < -heaps[stratum][0][0]:
                heapq.heapreplace(heaps[stratum], item)
        for stratum, heap in enumerate(heaps):
            assert len(heap) == SPEC['terms_per_interval']
            for neg_priority, term, df, idf in sorted(heap, key=lambda x: -x[0]):
                selected.append({'term': term, 'df': int(df), 'idf': float(idf), 'stratum': stratum})
                rows = db.execute('SELECT p.vector_id,p.tf,d.length FROM postings p JOIN docs d ON d.vector_id=p.vector_id WHERE p.term=?', (term,)).fetchall()
                assert len(rows) == df
                posting_rows.append(np.asarray(rows, dtype=np.int64))
    selected_seconds = time.perf_counter()-began
    all_ids = np.unique(np.concatenate([row[:,0] for row in posting_rows]))
    dense_rows = np.empty((len(all_ids),384), dtype=np.float32)
    filled = np.zeros(len(all_ids),bool)
    shards_read = 0
    start = time.perf_counter()
    for part in parts:
        row_start, row_end = part['start_row'], part['start_row']+part['rows']
        lo, hi = np.searchsorted(all_ids,[row_start,row_end])
        if lo == hi:
            continue
        block = np.load(DENSE/'embeddings'/part['file'], mmap_mode='r', allow_pickle=False)
        dense_rows[lo:hi] = block[all_ids[lo:hi]-row_start]
        filled[lo:hi] = True
        del block
        shards_read += 1
    assert filled.all() and np.isfinite(dense_rows).all()
    read_seconds = time.perf_counter()-start
    masses, squared_masses, weighted_vectors, weighted_norms, centroids = [], [], [], [], []
    summaries = []
    avgdl = sparse_meta['average_document_length']
    start = time.perf_counter()
    with threadpool_limits(limits=SPEC['threads']):
        for info, rows in zip(selected,posting_rows):
            tf, dl = rows[:,1].astype(np.float64), rows[:,2]
            weights = info['idf']*2.5*tf/(tf+1.5*(.25+.75*dl/avgdl))
            vectors = dense_rows[np.searchsorted(all_ids,rows[:,0])]
            mass, squared_mass, weighted_vector, weighted_norm, centroid = term_moments(weights,vectors)
            masses.append(mass); squared_masses.append(squared_mass)
            weighted_vectors.append(weighted_vector); weighted_norms.append(weighted_norm); centroids.append(centroid)
            squared_centroid = float(centroid @ centroid)
            summaries.append({**info,'centroid_norm': squared_centroid**.5,
                              'effective_weighted_documents': mass*mass/squared_mass,
                              'weighted_dispersion': weighted_norm/mass-squared_centroid,
                              'weighted_mean_document_squared_norm': weighted_norm/mass})
    aggregate_seconds = time.perf_counter()-start
    with (OUT/'moments.npz').open('xb') as stream:
        np.savez_compressed(stream, terms=np.asarray([x['term'] for x in selected]),
                            df=np.asarray([x['df'] for x in selected]),
                            sum_b=np.asarray(masses),sum_b_squared=np.asarray(squared_masses),
                            sum_b_times_dense=np.asarray(weighted_vectors),
                            sum_b_times_dense_squared_norm=np.asarray(weighted_norms))
    result = {'status':'complete_corpus_only_feasibility_not_router_gain',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'eligible_terms_per_interval':populations,'selected_terms':len(selected),
              'postings_read':sum(len(x) for x in posting_rows),'unique_vector_rows_read':len(all_ids),
              'embedding_shards_touched':shards_read, 'moment_file_bytes':(OUT/'moments.npz').stat().st_size,
              'selection_and_postings_seconds':selected_seconds,'vector_read_seconds':read_seconds,
              'moment_reduction_seconds':aggregate_seconds,'total_seconds':time.perf_counter()-began,
              'term_summaries':summaries,'new_encoder_calls':0,'query_reads':0,'label_reads':0,
              'retained_queries_accessed':0,'model_fits':0,'new_answer_calls':0,
              'global_corpus_mean_computed':False,'online_coverage_established':False,
              'query_latency_established':False,'novelty_established':False,'answer_gain_established':False,
              'decision':'static_joint_moments_constructible_next_require_frozen_vocabulary_and_effect_protocol'}
    write_new(OUT/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--math-check',action='store_true')
    args = parser.parse_args()
    freeze() if args.prepare else pilot() if args.pilot else math_check()
