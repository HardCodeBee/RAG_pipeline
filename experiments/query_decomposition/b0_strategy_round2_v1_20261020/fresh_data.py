"""Offline dev90 ingestion/references/scoring; never opens sealed150 inputs.

    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_data --self-test
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_data --references
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_data --ingest
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_data --score

References are written only after metadata selection and never imported by the
generation runner. No provider, SSH, training, retrieval or remote work occurs.
The scorer is descriptive development evaluation, not confirmation evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json


HERE = Path(__file__).resolve().parent
FRESH = HERE / 'fresh_validation'
OUT = OUTPUT / HERE.name / 'fresh_dev90'
DATA = ROOT / 'data/multihop_comprehensive_v1_20261015'
DATASETS = ['hotpotqa', '2wiki', 'musique']
CHECKPOINT = '728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92'
PROMPT = '508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8'
METADATA = FRESH / 'dev90.query_only.frozen.jsonl'
REMOTE_QUERIES = FRESH / 'remote_inputs/dev90.id_query.frozen.jsonl'
RUNTIME = FRESH / 'runtime_dev90.frozen.jsonl'
REFERENCES = FRESH / 'reference_dev90.local.frozen.jsonl'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def object_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\', '/')


def freeze(path, value, *, jsonl=False):
    path = Path(path)
    text = (''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in value) if jsonl
            else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Frozen artifact changed: ' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')


def metadata():
    """Read/hash only the development inputs; no confirmation file access."""
    manifest = read(FRESH / 'MANIFEST.frozen.json')
    assert sha(METADATA) == manifest['files']['dev90.query_only.frozen.jsonl']['sha256']
    assert sha(REMOTE_QUERIES) == manifest['files']['remote_inputs/dev90.id_query.frozen.jsonl']['sha256']
    qq, remote = rows(METADATA), rows(REMOTE_QUERIES)
    assert len(qq) == len({q['id'] for q in qq}) == 90
    assert Counter(q['dataset'] for q in qq) == {ds: 30 for ds in DATASETS}
    assert remote == [{'id': q['id'], 'query': q['query']} for q in qq]
    for q in qq:
        assert q['id'] == q['dataset'] + ':' + q['qid']
        assert isinstance(q['query'], str) and q['query'].strip()
        assert not {'answer', 'answers', 'gold', 'prediction', 'supporting_questions'}.intersection(q)
    return qq


def validate_identity(identity, query_sha):
    expected = {'checkpoint_manifest_sha256': CHECKPOINT, 'prompt_sha256': PROMPT,
                'query_sha256': query_sha, 'decode_compat': 'explicit_generation_config_v002',
                'effective_do_sample': False, 'effective_num_beams': 1,
                'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024,
                'seed': 1729, 'api_calls': 0}
    for key, value in expected.items():
        assert key in identity and identity[key] == value, 'R02 identity mismatch: ' + key
    # Prevent bool/int equivalence from accepting a different decoding contract.
    assert identity['do_sample'] is False and identity['effective_do_sample'] is False
    for key in ['num_beams', 'effective_num_beams', 'max_new_tokens', 'seed', 'api_calls']:
        assert type(identity[key]) is int, 'R02 identity type mismatch: ' + key


def parse_questions(record):
    """Raw JSON only. Empty/failed outputs keep the question as a goal-only case."""
    reasons = []
    if record.get('generation_error'):
        reasons.append('generation_error')
    if record.get('truncated'):
        reasons.append('truncated')
    value = None
    try:
        value = json.loads(record['raw_output'])
    except (KeyError, TypeError, ValueError):
        reasons.append('raw_output_not_json')
    if value is not None:
        if not isinstance(value, list):
            reasons.append('not_array')
        elif not all(isinstance(q, str) and q.strip() for q in value):
            reasons.append('non_string_or_blank_question')
    elif 'raw_output_not_json' not in reasons:
        reasons.append('not_array')  # JSON null is valid JSON, but not an array.
    return ([], reasons) if reasons else (value, [])


def ingest(source=None):
    qq = metadata()
    source = Path(source) if source is not None else OUT / 'remote/results'
    identity_path, state_path = source / 'IDENTITY.json', source / 'STATE.json'
    identity, state = read(identity_path), read(state_path)
    validate_identity(identity, sha(REMOTE_QUERIES))
    assert state == {'status': 'completed', 'completed': 90, 'planned': 90, 'api_calls': 0}, 'Await all90 real R02 outputs'
    files = sorted(p for p in source.glob('*.json') if p.stem.isdecimal())
    assert [p.name for p in files] == [f'{i:03d}.json' for i in range(1, 91)], 'Require precisely001..090 R02 outputs'
    runtime, failures, counts = [], [], Counter()
    for i, (q, path) in enumerate(zip(qq, files), 1):
        r = read(path)
        assert r['index'] == i and r['id'] == q['id'] and r['query'] == q['query']
        assert r['dataset'] == q['dataset']
        assert r['identity_sha256'] == object_hash(identity), 'Per-item identity mismatch'
        assert hashlib.sha256(r['actual_prompt'].encode('utf-8')).hexdigest() == PROMPT
        assert type(r['input_tokens']) is int and r['input_tokens'] == len(r['input_token_ids']) > 0
        assert type(r['output_tokens']) is int and r['output_tokens'] == len(r['output_token_ids'])
        assert r['output_tokens'] <= 1024
        assert all(type(t) is int for t in r['input_token_ids'] + r['output_token_ids'])
        assert isinstance(r.get('truncated'), bool), 'Missing explicit truncation receipt'
        assert 'generation_error' in r, 'Missing generation outcome receipt'
        questions, reasons = parse_questions(r)
        if reasons:
            failures.append({'id': q['id'], 'reasons': reasons, 'source_sha256': sha(path)})
            counts.update(reasons)
        runtime.append({'id': q['id'], 'qid': q['qid'], 'dataset': q['dataset'],
                        'question': q['query'], 'supporting_questions': questions,
                        **{key: q[key] for key in ['type', 'hops'] if key in q}})
    freeze(RUNTIME, runtime, jsonl=True)
    receipt = {'status': 'R02_actual_dev90_verified_before_RAG', 'cases': 90,
               'metadata_sha256': sha(METADATA), 'id_query_sha256': sha(REMOTE_QUERIES),
               'runtime_sha256': sha(RUNTIME), 'identity': identity,
               'source_directory': relative(source), 'identity_sha256': sha(identity_path),
               'state_sha256': sha(state_path),
               'source_files': {p.name: sha(p) for p in files},
               'fallback_count': len(failures), 'fallback_reason_counts': dict(counts),
               'failures': failures,
               'fallback': 'Empty supporting_questions; execute original-query goal and retain every question in scoring.',
               'raw_arrays_rewritten': False, 'manual_corrected_plans_used': False,
               'sealed_confirmation_opened': False, 'driver_sha256': sha(Path(__file__)),
               'new_API_calls': 0}
    freeze(OUT / 'INGEST.frozen.json', receipt)
    return {'ingested': 90, 'fallback_count': len(failures), 'runtime_sha256': sha(RUNTIME), 'new_API_calls': 0}


def reference_records(qq):
    """Access reference fields only for IDs fixed in dev90 before generation."""
    chosen = {ds: {q['qid']: q for q in qq if q['dataset'] == ds} for ds in DATASETS}
    # Keep source identity and ordering independent of answers and results.
    gold = {}
    source_paths = {'hotpotqa': ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/preparation/offline_gold_source/validation.parquet',
                    '2wiki': DATA / '2wiki/native/dev.json',
                    'musique': DATA / 'musique/native/musique_ans_v1.0_dev.jsonl'}
    import pyarrow.parquet as pq
    native = {'hotpotqa': pq.read_table(source_paths['hotpotqa'], columns=['id', 'question', 'answer']).to_pylist(),
              '2wiki': read(source_paths['2wiki'])}
    for ds in ['hotpotqa', '2wiki']:
        for r in native[ds]:
            qid = r['id'] if ds == 'hotpotqa' else r['_id']
            if qid not in chosen[ds]:
                continue
            q = chosen[ds][qid]
            assert r['question'] == q['query'], 'Native query identity mismatch: ' + q['id']
            assert q['id'] not in gold, 'Duplicate native ID'
            gold[q['id']] = {'id': q['id'], 'qid': qid, 'dataset': ds, 'question': q['query'],
                              'answer': r['answer'], 'aliases': [],
                              **({'answer_id': r['answer_id']} if ds == '2wiki' else {})}
    with source_paths['musique'].open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r['id'] not in chosen['musique']:
                continue
            q = chosen['musique'][r['id']]
            assert r['question'] == q['query'], 'Native query identity mismatch: ' + q['id']
            assert q['id'] not in gold, 'Duplicate native ID'
            gold[q['id']] = {'id': q['id'], 'qid': q['qid'], 'dataset': 'musique',
                              'question': q['query'], 'answer': r['answer'], 'aliases': r.get('answer_aliases', [])}
    assert set(gold) == {q['id'] for q in qq}
    values = [{**gold[q['id']], **{key: q[key] for key in ['type', 'hops'] if key in q},
               'generation_input_forbidden': True} for q in qq]
    return values, {relative(p): sha(p) for p in source_paths.values()}


def metric_pins():
    paths = [ROOT / 'experiments/query_decomposition/comprehensive_eval_v1_20261015/scoring.py',
             ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/common/scoring.py',
             DATA / '2wiki/official/2wikimultihop_evaluate_v1.1.py',
             DATA / '2wiki/native/id_aliases.json',
             DATA / 'musique/official/metrics/answer.py']
    return {relative(p): sha(p) for p in paths}


def metrics():
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    return {ds: metric(ds) for ds in DATASETS}


def references():
    qq = metadata()
    values, sources = reference_records(qq)
    scorers = metrics()
    for ref in values:
        assert isinstance(ref['answer'], str)
        assert scorers[ref['dataset']](ref['answer'], ref) == {'em': 1.0, 'f1': 1.0}
        if ref['dataset'] == '2wiki':
            # Native Boolean/numeric answers can have null answer_id. Keep
            # null as null; entity IDs stay available for official aliases.
            assert ref['answer_id'] is None or isinstance(ref['answer_id'], str)
        if ref['dataset'] == 'musique':
            for alias in ref['aliases']:
                assert scorers['musique'](alias, ref)['em'] == 1.0
    freeze(REFERENCES, values, jsonl=True)
    receipt = {'cohort': 'metadata-fixed dev90 only', 'cases': len(values),
               'metadata_sha256': sha(METADATA), 'references_sha256': sha(REFERENCES),
               'native_sources': sources, 'metric_sources': metric_pins(),
               'two_wiki_answer_id_preserved': True, 'musique_answer_aliases_preserved': True,
               'gold_self_score_passed': 90, 'generation_input_forbidden': True,
               'sealed_confirmation_opened': False, 'driver_sha256': sha(Path(__file__)), 'new_API_calls': 0}
    freeze(OUT / 'REFERENCES.frozen.json', receipt)
    return {'references': 90, 'gold_self_score_passed': 90, 'new_API_calls': 0}


def candidate_identity(candidate=None, protocol_path=None):
    protocol_path = Path(protocol_path) if protocol_path else OUT / 'CANDIDATE.frozen.json'
    protocol = read(protocol_path)
    frozen_arm = next((protocol[k] for k in ['candidate', 'selected', 'arm'] if k in protocol), None)
    if candidate is None:
        selected = read(OUTPUT / HERE.name / 'SELECTION.frozen.json')['selected']
        assert selected == frozen_arm, 'Old300 selection and fresh candidate disagree'
        candidate = selected
    assert candidate == frozen_arm, 'Scoring candidate differs from frozen fresh candidate'
    from .strategy import ARMS
    assert candidate in ['parent_g01', *ARMS]
    return candidate, protocol_path


def summarize_scores(values, planned):
    usable = [v for v in values if v['f1'] is not None]
    n, m = len(usable), planned - len(usable)
    total = sum(v['f1'] for v in usable)
    return {'planned': planned, 'scorable': n, 'missing': m,
            'F1_available': total / n if n else None,
            'EM_available': sum(v['em'] for v in usable) / n if n else None,
            'F1_full': total / planned if n == planned else None,
            'EM_full': sum(v['em'] for v in usable) / planned if n == planned else None,
            'F1_full_bounds': [total / planned, (total + m) / planned],
            'statuses': dict(Counter(v['status'] for v in values))}


def paired_summary(details):
    import numpy as np
    matched = [d for d in details if d['delta_F1'] is not None]
    delta = [d['delta_F1'] for d in matched]
    n = len(matched)
    rng = np.random.default_rng(20261020)
    boot = np.zeros(5000)
    if matched:
        for ds in DATASETS:
            dd = np.array([d['delta_F1'] for d in matched if d['dataset'] == ds])
            if len(dd):
                boot += dd[rng.integers(len(dd), size=(5000, len(dd)))].sum(1) / n
    return {'n_matched': n,
            'parent_F1': sum(d['parent']['f1'] for d in matched) / n if n else None,
            'candidate_F1': sum(d['candidate']['f1'] for d in matched) / n if n else None,
            'delta_F1': sum(delta) / n if n else None,
            'up': sum(v > 1e-12 for v in delta), 'tie': sum(abs(v) <= 1e-12 for v in delta),
            'down': sum(v < -1e-12 for v in delta),
            'paired95_descriptive': np.quantile(boot, [.025, .975]).tolist() if n else None,
            'bootstrap': '5000 paired resamples within dataset, seed20261020; descriptive development interval'}


def score(candidate=None, protocol_path=None):
    qq = metadata()
    candidate, candidate_path = candidate_identity(candidate, protocol_path)
    runtime = rows(RUNTIME)
    assert len(runtime) == 90 and [r['id'] for r in runtime] == [q['id'] for q in qq]
    assert all(r['question'] == q['query'] for r, q in zip(runtime, qq))
    refs = rows(REFERENCES)
    assert [r['id'] for r in refs] == [q['id'] for q in qq]
    ref_receipt = read(OUT / 'REFERENCES.frozen.json')
    assert sha(REFERENCES) == ref_receipt['references_sha256']
    assert metric_pins() == ref_receipt['metric_sources']
    from .fresh_runner import selected
    scorers, details = metrics(), []
    for q, case, ref in zip(qq, runtime, refs):
        assert ref['question'] == case['question']
        item = {'id': q['id'], 'dataset': q['dataset'], 'query': q['query'], 'gold_answer': ref['answer']}
        for label, arm in [('parent', 'parent_g01'), ('candidate', candidate)]:
            value = selected(case, arm)
            status = value['status'] if value else 'unstarted'
            pred = value.get('prediction') if value else None
            valid = isinstance(pred, str) and (status.startswith('success') or status == 'algorithm_failure')
            m = scorers[q['dataset']](pred, ref) if valid else {'em': None, 'f1': None}
            item[label] = {'arm': arm, 'status': status, 'prediction': pred, **m}
        p, c = item['parent']['f1'], item['candidate']['f1']
        item['delta_F1'] = c - p if p is not None and c is not None else None
        details.append(item)
    report = {'utc': utc(), 'cohort': 'fresh development90,30 per dataset',
              'candidate': candidate, 'candidate_protocol_sha256': sha(candidate_path),
              'runtime_sha256': sha(RUNTIME), 'references_sha256': sha(REFERENCES),
              'metric': 'native Answer EM/F1; 2Wiki v1.1 aliases/demonyms; MuSiQue answer_aliases',
              'sample_role': 'Development comparison; not independent confirmation or for tuning sealed150',
              'parent': summarize_scores([d['parent'] for d in details], 90),
              'candidate_summary': summarize_scores([d['candidate'] for d in details], 90),
              'paired': paired_summary(details),
              'by_dataset': {ds: {'parent': summarize_scores([d['parent'] for d in details if d['dataset'] == ds], 30),
                                  'candidate': summarize_scores([d['candidate'] for d in details if d['dataset'] == ds], 30),
                                  'paired': paired_summary([d for d in details if d['dataset'] == ds])} for ds in DATASETS},
              'sealed_confirmation_opened': False, 'new_API_calls': 0}
    complete = report['paired']['n_matched'] == 90
    passed = complete and candidate != 'parent_g01' and report['paired']['delta_F1'] > 1e-12
    report['gate'] = {'complete_90_pairs': complete, 'passed': passed,
                      'rule': 'Both full90 terminal/scorable, fixed candidate mean F1 greater than parent; no CI threshold.'}
    locked_json(OUT / 'SCORES.progress.json', report)
    target = OUT / 'PAIRED_DETAILS.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix('.jsonl.tmp')
    pending.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in details), encoding='utf-8', newline='\n')
    pending.replace(target)
    # A nonpositive or incomplete development result never opens confirmation.
    if passed:
        freeze(OUT / 'GATE.frozen.json', {'passed': True, 'candidate': candidate,
                                         'n_complete': 90, 'delta_F1': report['paired']['delta_F1'],
                                         'candidate_protocol_sha256': sha(candidate_path), 'not_for_tuning': True})
    return report


def self_test():
    """No runtime/gold generation; unit checks for parser, identity and denominators."""
    valid = {'raw_output': '["Who wrote Book R?", "Where was the author of Book R born?"]',
             'generation_error': None, 'truncated': False}
    assert parse_questions(valid) == (["Who wrote Book R?", "Where was the author of Book R born?"], [])
    assert parse_questions({**valid, 'raw_output': '[]'}) == ([], [])
    for raw in ['```json\n[]\n```', '{"steps":[]}', 'null', '[""]', '[1]', 'not JSON']:
        value, reasons = parse_questions({**valid, 'raw_output': raw})
        assert value == [] and reasons
    assert parse_questions({**valid, 'truncated': True}) == ([], ['truncated'])
    assert parse_questions({**valid, 'generation_error': 'oom'}) == ([], ['generation_error'])
    ident = {'checkpoint_manifest_sha256': CHECKPOINT, 'prompt_sha256': PROMPT,
             'query_sha256': 'test', 'decode_compat': 'explicit_generation_config_v002',
             'effective_do_sample': False, 'effective_num_beams': 1, 'do_sample': False,
             'num_beams': 1, 'max_new_tokens': 1024, 'seed': 1729, 'api_calls': 0}
    validate_identity(ident, 'test')
    for key, wrong in [('checkpoint_manifest_sha256', 'x'), ('seed', 0), ('effective_do_sample', True)]:
        try:
            validate_identity({**ident, key: wrong}, 'test')
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid identity accepted: ' + key)
    ss = summarize_scores([{'status': 'success', 'f1': 1., 'em': 1.},
                           {'status': 'incomplete', 'f1': None, 'em': None},
                           {'status': 'algorithm_failure', 'f1': 0., 'em': 0.}], 3)
    assert ss['scorable'] == 2 and ss['missing'] == 1 and ss['F1_available'] == .5
    assert ss['F1_full'] is None and ss['F1_full_bounds'] == [1/3, 2/3]
    ps = paired_summary([{'dataset': 'hotpotqa', 'parent': {'f1': 0.},
                          'candidate': {'f1': 1.}, 'delta_F1': 1.},
                         {'dataset': '2wiki', 'parent': {'f1': 1.},
                          'candidate': {'f1': None}, 'delta_F1': None}])
    assert ps['n_matched'] == ps['up'] == 1 and ps['delta_F1'] == 1.
    assert ps['paired95_descriptive'] == [1., 1.]
    return {'self_test_passed': True, 'metadata_cases': len(metadata()), 'new_API_calls': 0,
            'sealed_confirmation_opened': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ingest', action='store_true')
    parser.add_argument('--references', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--remote-results', type=Path)
    parser.add_argument('--candidate')
    parser.add_argument('--candidate-protocol', type=Path)
    args = parser.parse_args()
    if not any([args.ingest, args.references, args.score, args.self_test]):
        parser.error('Choose --self-test, --references, --ingest or --score')
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
    if args.references:
        print(json.dumps(references(), ensure_ascii=False))
    if args.ingest:
        print(json.dumps(ingest(args.remote_results), ensure_ascii=False))
    if args.score:
        value = score(args.candidate, args.candidate_protocol)
        print(json.dumps({'candidate': value['candidate'], 'parent': value['parent'],
                          'candidate_summary': value['candidate_summary'], 'paired': value['paired'],
                          'gate': value['gate']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
