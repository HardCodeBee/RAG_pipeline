"""Freeze score-blind matched cases with completed S02/ReAct references."""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name
CAMPAIGN = OUTPUT / 'comprehensive_v1_20261015/campaign_api_v1'
INPUTS = ROOT / 'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'
CHAIN = ROOT / 'experiments/query_decomposition/component_chain_c08_c02_v1_20261019'
SEED = 'component-chain-s02-react-matched60-20261019'
QUOTAS = {'hotpotqa': {'bridge': 14, 'comparison': 6},
          '2wiki': {'comparison': 5, 'inference': 5, 'compositional': 5, 'bridge_comparison': 5},
          'musique': {'2-hop': 7, '3-hop': 7, '4-hop': 6}}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).strip().casefold())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (HERE / 'SELECTION.frozen.json').exists():
        manifest = read(HERE / 'SELECTION.frozen.json')
        assert sha(HERE / 'queries60.query_only.frozen.jsonl') == manifest['query_sha256']
        print(json.dumps({'preserved_selection': 60, 'sha256': manifest['query_sha256']}))
        return
    trainfile = ROOT / 'experiments/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018/DATA3632.frozen.jsonl'
    training = {normalize(r['query']) for r in rows(trainfile)}
    childfile = ROOT / 'experiments/query_decomposition/child_answer_accuracy_v1_20261017/round10/cases.inputs.jsonl'
    rewritefile = OUTPUT / 'rewrite_short_answer_ablation_v1_20261017/inputs.frozen.jsonl'
    excluded = {r['qid'] for p in (childfile, rewritefile) for r in rows(p)}
    excluded.update(r.get('source_qid') for r in read(ROOT / 'experiments/query_decomposition/child_answer_accuracy_v1_20261017/c08_examples.frozen.json'))
    selected, baselines, sources, poolcounts = [], [], {}, {}
    def remember(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
    for ds in QUOTAS:
        question_path = INPUTS / ds / 'formal.questions.jsonl'
        reference_path = INPUTS / ds / 'formal.references.local.jsonl'
        questions = {r['qid']: r for r in rows(question_path)}
        # Only type metadata is read for selection. Answer values are not used.
        types = {r['qid']: r['type'] for r in rows(reference_path)}
        remember(question_path); remember(reference_path)
        if ds == 'musique':
            files = [OUTPUT / 'multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_RESULTS.json',
                     OUTPUT / 'multigrain_dual_retrieval_expand60_v1_20261017/BASELINE_MUSIQUE_RESULTS.json']
            s02 = {}
            for f in files:
                remember(f)
                for r in read(f)['results']:
                    s02.setdefault(r['qid'], (r, f))
        else:
            f = CAMPAIGN / ds / 'formal/m00_s02_child4_faithful/frozen_s02_child4/scores.jsonl'
            remember(f)
            s02 = {r['qid']: (r, f) for r in rows(f)}
        f = CAMPAIGN / ds / 'formal/b03_react_dense/k10_s5/scores.jsonl'
        remember(f)
        react = {r['qid']: (r, f) for r in rows(f)}
        if ds == 'musique':
            extra = OUTPUT / 'multigrain_dual_retrieval_expand60_v1_20261017/REACT_COMPLETION_RESULTS.json'
            remember(extra)
            for r in read(extra)['results']:
                # Fill absent results only. Never select the better score.
                if react.get(r['qid'], ({}, None))[0].get('f1') is None:
                    react[r['qid']] = (r, extra)
        eligible = []
        for qid, (s, sf) in s02.items():
            if qid not in questions or qid in excluded or normalize(questions[qid]['question']) in training:
                continue
            if s.get('status') != 'success' or not isinstance(s.get('prediction'), str):
                continue
            rr = react.get(qid)
            if rr is None or rr[0].get('f1') is None:
                continue
            eligible.append(qid)
        poolcounts[ds] = dict(collections.Counter(types[q] for q in eligible))
        for kind, n in QUOTAS[ds].items():
            order = sorted((q for q in eligible if types[q] == kind),
                           key=lambda q: hashlib.sha256((SEED + '|' + ds + '|' + kind + '|' + q).encode()).hexdigest())
            assert len(order) >= n, (ds, kind, len(order), n)
            for qid in order[:n]:
                q = questions[qid]
                selected.append({'id': ds + ':' + qid, 'qid': qid, 'dataset': ds, 'query': q['question'], 'type': kind})
                for label, mapping in [('s02_child4', s02), ('react', react)]:
                    value, src = mapping[qid]
                    baselines.append({'id': ds + ':' + qid, 'qid': qid, 'dataset': ds, 'method': label,
                                      'source': str(src.relative_to(ROOT)), 'source_sha256': sources[str(src.relative_to(ROOT))],
                                      'prediction': value['prediction'], 'status': value['status'],
                                      'f1': value['f1'], 'em': value['em']})
    assert len(selected) == 60 and len({r['id'] for r in selected}) == 60
    assert len({normalize(r['query']) for r in selected}) == 60
    assert all(normalize(r['query']) not in training and r['qid'] not in excluded for r in selected)
    for name, data in [('queries60.query_only.frozen.jsonl', selected), ('baseline_references.local.frozen.jsonl', baselines)]:
        (HERE / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
    resources = {ds: read(CAMPAIGN / ds / 'RESOURCE_FROZEN.json')['resources'] for ds in QUOTAS}
    nominee = read(ROOT / 'experiments/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018/NOMINEE_FROZEN.json')
    manifest = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'planned': 60, 'seed': SEED, 'quotas': QUOTAS,
                'eligible_pool_by_type': poolcounts,
                'selection': 'Hash order within type among completed S02 and scored ReAct availability; no selection on score values.',
                'training_query_overlap': 0, 'C08_c02_component_eval_parent_overlap': 0,
                'availability_conditioned_development_comparison': True,
                'source_files_sha256': {**sources, str(trainfile.relative_to(ROOT)): sha(trainfile),
                                        str(childfile.relative_to(ROOT)): sha(childfile), str(rewritefile.relative_to(ROOT)): sha(rewritefile)},
                'query_sha256': sha(HERE / 'queries60.query_only.frozen.jsonl'),
                'baseline_refs_sha256': sha(HERE / 'baseline_references.local.frozen.jsonl'),
                'decomposer': {'checkpoint': nominee['checkpoint'], 'checkpoint_manifest_sha256': nominee['checkpoint_manifest_sha256'],
                               'prompt_sha256': '508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8',
                               'remote_scope': 'Query decomposition only, no training, retrieval, answering, or gold upload.',
                               'generation': {'do_sample': False, 'num_beams': 1, 'use_model_defaults': False, 'seed': 1729, 'max_new_tokens': 1024}},
                'conditions': ['B0_legacy', 'new_c08_c02', 's02_child4', 'react'],
                'B0_vs_new': 'Same actual R02 output, resources, rootTop10, goal and finalv3; only selected child/rewriter interfaces differ.',
                'external_baselines': 'Reuse frozen unified-Qwen/BGE S02 and ReAct k10_s5 records; their own decomposition, final prompts and search budgets stay intact.',
                'resources': resources, 'model': 'qwen-plus-2025-12-01',
                'generation': {'temperature': 0, 'enable_thinking': False, 'child_tokens': 128, 'rewrite_tokens': 256, 'final_tokens': 64},
                'chain_protocol_sha256': sha(CHAIN / 'PROTOCOL.frozen.json'),
                'global_cap_cny': 200, 'protected_stop_cny': 199.95, 'local_subcap_cny': 2.7,
                'scope': 'Small matched availability-conditioned development comparison; not full-dataset official test or independent SOTA proof.',
                'reporting': 'Native Answer F1/EM; same-query pairs, all-four common set, by dataset, planned denominator, missingness and costs. No missing-as-zero; ReAct algorithm failure follows frozen scoring.'}
    save(HERE / 'SELECTION.frozen.json', manifest)
    save(OUT / 'PREPARATION.json', {'status': 'cohort_frozen_before_R02_inference_and_new_RAG', 'planned': 60,
                                  'baseline_records': 120, 'new_API_calls': 0,
                                  'selection_sha256': sha(HERE / 'SELECTION.frozen.json')})
    print(json.dumps({'planned': 60, 'quotas': QUOTAS, 'eligible_pool_by_type': poolcounts,
                      'training_overlap': 0, 'component_parent_overlap': 0, 'query_sha256': manifest['query_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
