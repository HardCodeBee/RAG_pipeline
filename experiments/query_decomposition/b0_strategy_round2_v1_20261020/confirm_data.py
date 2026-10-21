"""Gated, one-time confirmation150 data preparation and native scoring.

The existing old300/dev90 prerequisites must all pass before any confirmation
query, runtime, reference or result is read. There are no candidate overrides,
new selection gates, API calls, SSH calls, retrieval calls or training here.
All frozen dev90 files and globals remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from . import fresh_confirm as confirm


HERE = Path(__file__).resolve().parent
FRESH = HERE / 'fresh_validation'
OUT = OUTPUT / HERE.name / 'fresh_confirm150'
METADATA = FRESH / 'sealed150.query_only.frozen.jsonl'
REMOTE_QUERIES = FRESH / 'remote_inputs/sealed150.id_query.frozen.jsonl'
RUNTIME = FRESH / 'runtime_confirm150.frozen.jsonl'
REFERENCES = FRESH / 'ref_confirm150.local.frozen.jsonl'
CORE_SOURCE = HERE / 'fresh_data.py'
N = 150
DATASETS = ['hotpotqa', '2wiki', 'musique']


def prerequisites():
    """Always fail closed before looking at a confirmation artifact."""
    checked = confirm.prerequisites()
    assert checked['selected'] in {'g02_evidence_goal', 'p02_role_headers', 's03_g01_plus_f01'}
    return checked


def private_core():
    """Reuse pure dev90 helpers with explicit private path bindings only."""
    from . import fresh_data as original
    spec = importlib.util.spec_from_file_location('_b0_confirmation_data_private', CORE_SOURCE)
    bound = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bound)
    assert bound is not original and bound.__dict__ is not original.__dict__
    old = {name: getattr(original, name) for name in ['OUT', 'METADATA', 'REMOTE_QUERIES', 'RUNTIME', 'REFERENCES']}
    bound.OUT, bound.METADATA, bound.REMOTE_QUERIES = OUT, METADATA, REMOTE_QUERIES
    bound.RUNTIME, bound.REFERENCES = RUNTIME, REFERENCES
    assert all(getattr(original, name) == value for name, value in old.items())
    return bound


def source_pins(core):
    return {core.relative(path): core.sha(path) for path in
            [Path(__file__), CORE_SOURCE, Path(confirm.__file__)]}


def _metadata(checked, core):
    # Private reader: every public caller has already run prerequisites().
    assert checked['selected'] in {'g02_evidence_goal', 'p02_role_headers', 's03_g01_plus_f01'}
    manifest = core.read(FRESH / 'MANIFEST.frozen.json')
    assert core.sha(METADATA) == manifest['files']['sealed150.query_only.frozen.jsonl']['sha256']
    assert core.sha(REMOTE_QUERIES) == manifest['files']['remote_inputs/sealed150.id_query.frozen.jsonl']['sha256']
    qq = core.rows(METADATA)
    assert len(qq) == len({q['id'] for q in qq}) == N
    assert Counter(q['dataset'] for q in qq) == {ds: 50 for ds in DATASETS}
    assert core.rows(REMOTE_QUERIES) == [{'id': q['id'], 'query': q['query']} for q in qq]
    for q in qq:
        assert q['id'] == q['dataset'] + ':' + q['qid']
        assert isinstance(q['query'], str) and q['query'].strip()
        assert not {'answer', 'answers', 'gold', 'prediction', 'supporting_questions'}.intersection(q)
    return qq


def _identity(checked, core):
    path = checked['paths']['candidate_protocol']
    return {'candidate': checked['selected'], 'candidate_protocol_sha256': core.sha(path),
            'prerequisite_hashes': checked['hashes'], 'source_pins': source_pins(core),
            'sample_role': 'one-time confirmation, no tuning', 'not_for_tuning': True}


def ingest(source=None):
    checked = prerequisites()
    core = private_core()
    qq = _metadata(checked, core)
    source = Path(source) if source is not None else OUT / 'remote/results'
    identity_path, state_path = source / 'IDENTITY.json', source / 'STATE.json'
    identity, state = core.read(identity_path), core.read(state_path)
    core.validate_identity(identity, core.sha(REMOTE_QUERIES))
    assert state == {'status': 'completed', 'completed': N, 'planned': N, 'api_calls': 0}, 'Await all150 R02 outputs'
    files = sorted(p for p in source.glob('*.json') if p.stem.isdecimal())
    assert [p.name for p in files] == [f'{i:03d}.json' for i in range(1, N + 1)]
    runtime, failures, counts = [], [], Counter()
    for i, (q, path) in enumerate(zip(qq, files), 1):
        r = core.read(path)
        assert r['index'] == i and r['id'] == q['id'] and r['query'] == q['query']
        assert r['dataset'] == q['dataset']
        assert r['identity_sha256'] == core.object_hash(identity)
        assert hashlib.sha256(r['actual_prompt'].encode('utf-8')).hexdigest() == core.PROMPT
        assert type(r['input_tokens']) is int and r['input_tokens'] == len(r['input_token_ids']) > 0
        assert type(r['output_tokens']) is int and r['output_tokens'] == len(r['output_token_ids'])
        assert r['output_tokens'] <= 1024
        assert all(type(t) is int for t in r['input_token_ids'] + r['output_token_ids'])
        assert isinstance(r.get('truncated'), bool) and 'generation_error' in r
        questions, reasons = core.parse_questions(r)
        if reasons:
            failures.append({'id': q['id'], 'reasons': reasons, 'source_sha256': core.sha(path)})
            counts.update(reasons)
        runtime.append({'id': q['id'], 'qid': q['qid'], 'dataset': q['dataset'],
                        'question': q['query'], 'supporting_questions': questions,
                        **{key: q[key] for key in ['type', 'hops'] if key in q}})
    core.freeze(RUNTIME, runtime, jsonl=True)
    receipt = {**_identity(checked, core), 'status': 'R02_actual_confirmation150_verified_before_RAG',
               'cases': N, 'metadata_sha256': core.sha(METADATA),
               'id_query_sha256': core.sha(REMOTE_QUERIES), 'runtime_sha256': core.sha(RUNTIME),
               'identity': identity, 'identity_sha256': core.sha(identity_path),
               'state_sha256': core.sha(state_path), 'source_directory': core.relative(source),
               'source_files': {p.name: core.sha(p) for p in files},
               'fallback_count': len(failures), 'fallback_reason_counts': dict(counts), 'failures': failures,
               'fallback': 'Empty supporting_questions; execute original-query goal; retain every query in scoring.',
               'raw_arrays_rewritten': False, 'manual_corrected_plans_used': False, 'new_API_calls': 0}
    core.freeze(OUT / 'INGEST.frozen.json', receipt)
    return {'ingested': N, 'fallback_count': len(failures), 'runtime_sha256': core.sha(RUNTIME),
            'sample_role': 'one-time confirmation, no tuning', 'new_API_calls': 0}


def references():
    checked = prerequisites()
    core = private_core()
    qq = _metadata(checked, core)
    values, sources = core.reference_records(qq)
    scorers = core.metrics()
    for ref in values:
        assert isinstance(ref['answer'], str)
        assert scorers[ref['dataset']](ref['answer'], ref) == {'em': 1., 'f1': 1.}
        if ref['dataset'] == '2wiki':
            assert ref['answer_id'] is None or isinstance(ref['answer_id'], str)
        if ref['dataset'] == 'musique':
            assert all(scorers['musique'](alias, ref)['em'] == 1. for alias in ref['aliases'])
    core.freeze(REFERENCES, values, jsonl=True)
    receipt = {**_identity(checked, core), 'cases': N, 'metadata_sha256': core.sha(METADATA),
               'references_sha256': core.sha(REFERENCES), 'native_sources': sources,
               'metric_sources': core.metric_pins(), 'gold_self_score_passed': N,
               'two_wiki_answer_id_preserved': True, 'musique_answer_aliases_preserved': True,
               'generation_input_forbidden': True, 'new_API_calls': 0}
    core.freeze(OUT / 'REFERENCES.frozen.json', receipt)
    return {'references': N, 'gold_self_score_passed': N,
            'sample_role': 'one-time confirmation, no tuning', 'new_API_calls': 0}


def paired_summary(core, details):
    result = core.paired_summary(details)
    result['bootstrap'] = '5000 paired resamples within dataset, seed20261020; one-time confirmation interval'
    result['paired95'] = result.pop('paired95_descriptive')
    return result


def score():
    checked = prerequisites()
    core = private_core()
    qq = _metadata(checked, core)
    plan = core.read(OUT / 'PROTOCOL.frozen.json')
    confirm.verify(plan)
    assert plan['arms_available'] == ['parent_g01', checked['selected']]
    bound = confirm.private_driver()
    cases = bound.runtime_cases()
    assert [c['id'] for c in cases] == [q['id'] for q in qq]
    assert all(c['question'] == q['query'] for c, q in zip(cases, qq))
    ingest_receipt = core.read(OUT / 'INGEST.frozen.json')
    assert core.sha(RUNTIME) == ingest_receipt['runtime_sha256'] == plan['runtime_sha256']
    refs = core.rows(REFERENCES)
    assert [r['id'] for r in refs] == [q['id'] for q in qq]
    ref_receipt = core.read(OUT / 'REFERENCES.frozen.json')
    assert core.sha(REFERENCES) == ref_receipt['references_sha256']
    assert core.metric_pins() == ref_receipt['metric_sources']
    for receipt in [ingest_receipt, ref_receipt]:
        assert receipt['candidate'] == checked['selected']
        assert receipt['candidate_protocol_sha256'] == core.sha(checked['paths']['candidate_protocol'])
        assert receipt['prerequisite_hashes'] == checked['hashes']
        assert receipt['source_pins'] == source_pins(core)
    scorers, details = core.metrics(), []
    for q, case, ref in zip(qq, cases, refs):
        assert ref['question'] == case['question']
        item = {'id': q['id'], 'dataset': q['dataset'], 'query': q['query'], 'gold_answer': ref['answer']}
        for label, arm in [('parent', 'parent_g01'), ('candidate', checked['selected'])]:
            value = bound.selected(case, arm)
            status = value['status'] if value else 'unstarted'
            pred = value.get('prediction') if value else None
            valid = isinstance(pred, str) and (status.startswith('success') or status == 'algorithm_failure')
            m = scorers[q['dataset']](pred, ref) if valid else {'em': None, 'f1': None}
            item[label] = {'arm': arm, 'status': status, 'prediction': pred, **m}
        p, c = item['parent']['f1'], item['candidate']['f1']
        item['delta_F1'] = c - p if p is not None and c is not None else None
        details.append(item)
    report = {**_identity(checked, core), 'cohort': 'sealed confirmation150,50 per dataset',
              'runtime_sha256': core.sha(RUNTIME), 'references_sha256': core.sha(REFERENCES),
              'execution_protocol_sha256': core.sha(OUT / 'PROTOCOL.frozen.json'),
              'metric': 'native Answer EM/F1; 2Wiki v1.1 aliases/demonyms; MuSiQue answer_aliases',
              'parent': core.summarize_scores([d['parent'] for d in details], N),
              'candidate_summary': core.summarize_scores([d['candidate'] for d in details], N),
              'paired': paired_summary(core, details),
              'by_dataset': {ds: {'parent': core.summarize_scores([d['parent'] for d in details if d['dataset'] == ds], 50),
                                  'candidate': core.summarize_scores([d['candidate'] for d in details if d['dataset'] == ds], 50),
                                  'paired': paired_summary(core, [d for d in details if d['dataset'] == ds])} for ds in DATASETS},
              'candidate_reselected': False, 'new_tuning_gate_generated': False, 'new_API_calls': 0}
    complete = report['paired']['n_matched'] == N
    report['complete_150_pairs'] = complete
    locked_json(OUT / 'SCORES.progress.json', {'utc': utc(), **report})
    target = OUT / 'PAIRED_DETAILS.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in details), encoding='utf-8', newline='\n')
    temporary.replace(target)
    if complete:
        core.freeze(OUT / 'SCORES.frozen.json', report)
        core.freeze(OUT / 'PAIRED_DETAILS.frozen.jsonl', details, jsonl=True)
    return report


def self_test():
    """Synthetic checks only; not even metadata hashes open before gates pass."""
    from unittest.mock import patch
    from . import fresh_data as original
    old = {name: getattr(original, name) for name in ['OUT', 'METADATA', 'REMOTE_QUERIES', 'RUNTIME', 'REFERENCES']}
    core = private_core()
    assert core.OUT == OUT and core.RUNTIME == RUNTIME and core.REFERENCES == REFERENCES
    assert all(getattr(original, k) == v for k, v in old.items())
    blocked = []
    with patch.object(confirm, 'prerequisites', side_effect=confirm.GateClosed('synthetic closed gate')):
        # These would fail loudly if an entry point tried to construct the
        # metadata/core binding or read a query before checking the gate.
        with patch(__name__ + '.private_core', side_effect=AssertionError('core opened before gate')):
            for name, action in [('ingest', ingest), ('references', references), ('score', score)]:
                try:
                    action()
                except confirm.GateClosed:
                    blocked.append(name)
                else:
                    raise AssertionError('Closed gate accepted: ' + name)
    assert blocked == ['ingest', 'references', 'score']
    synthetic = [{'dataset': ds, 'parent': {'f1': 0.}, 'candidate': {'f1': 1.}, 'delta_F1': 1.}
                 for ds in DATASETS for _ in range(50)]
    result = paired_summary(core, synthetic)
    assert result['n_matched'] == result['up'] == N and result['delta_F1'] == 1.
    assert result['paired95'] == [1., 1.] and 'development' not in result['bootstrap']
    summary = core.summarize_scores([{'status': 'success', 'f1': 1., 'em': 1.}] * 149 +
                                    [{'status': 'incomplete', 'f1': None, 'em': None}], N)
    assert summary['planned'] == N and summary['scorable'] == 149 and summary['missing'] == 1
    assert summary['F1_full'] is None
    return {'passed': True, 'closed_gate_blocks': blocked, 'private_core_binding': True,
            'full150_paired_contract': True, 'confirmation_queries_opened': False,
            'gold_opened': False, 'new_API_calls': 0, 'new_remote_calls': 0,
            'source_pins': source_pins(core)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--ingest', action='store_true')
    modes.add_argument('--references', action='store_true')
    modes.add_argument('--score', action='store_true')
    parser.add_argument('--remote-results', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.ingest:
        result = ingest(args.remote_results)
    elif args.references:
        result = references()
    else:
        value = score()
        result = {key: value[key] for key in ['candidate', 'sample_role', 'parent', 'candidate_summary',
                                             'paired', 'by_dataset', 'complete_150_pairs']}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
