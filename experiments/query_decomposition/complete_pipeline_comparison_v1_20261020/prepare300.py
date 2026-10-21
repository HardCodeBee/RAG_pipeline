"""Freeze the accepted four complete methods on the existing evaluation300.

No API, model inference, semantic plan repair, or outcome-driven sampling.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sqlite3
import zipfile

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
RUN = OUT / 'run_v001'
EVAL = HERE.parent / 'multigrain_4b_frozen_eval300_v1_20261017'
R02 = OUTPUT / 'decomposition_error_expand60_v1_20261019'
ARMS = ['s02_child4', 'B0_legacy_multigrain_R02',
        'R02_C08_c02_r1p1_integrated_candidate', 'c06_comparison_criterion']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if p.exists():
        assert p.read_text(encoding='utf-8') == data, 'Frozen file differs: ' + str(p)
    else:
        p.write_text(data, encoding='utf-8')


def save_rows(p, values):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    data = ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values)
    if p.exists():
        assert p.read_text(encoding='utf-8') == data, str(p)
    else:
        p.write_text(data, encoding='utf-8')


def log(event, **fields):
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def sources(ds):
    """Explicit cache sources; no transfer of algorithm state between methods."""
    roots = ['component_chain_eval60_v1_20261019', 'component_chain_opt_v2_20261019',
             'reader_context_route_ablation_v1_20261019', 'reader_context_route_ablation50_v1_20261020',
             'decomposition_error_expand60_v1_20261019', 'decomposition_error_downstream_v1_20261018',
             'decomposition_corrected_same60_v1_20261019']
    result = []
    for name in roots:
        result += sorted((OUTPUT / name).glob('**/' + ds + '/checkpoint.sqlite3'))
    result += sorted((OUTPUT / 'comprehensive_v1_20261015/campaign_api_v1' / ds / 'formal').glob('m00_s02_child4_faithful/*/checkpoint.sqlite3'))
    return sorted(set(result))


def prepare():
    frozen = HERE / 'EVAL300_PROTOCOL.frozen.json'
    if frozen.exists():
        return read(frozen)
    questions = rows(EVAL / 'questions300.frozen.jsonl')
    assert len(questions) == len({r['id'] for r in questions}) == 300
    assert Counter(r['dataset'] for r in questions) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    ii = read(R02 / 'RAW_IDENTITY.json')
    assert ii['checkpoint_manifest_sha256'] == '728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92'
    raw = {read(p)['id']: (p, read(p)) for p in (R02 / 'raw').glob('[0-9][0-9][0-9].json')}
    assert set(raw) == {r['id'] for r in questions}
    runtime, provenance = [], []
    for q in questions:
        p, r = raw[q['id']]
        assert r['query'] == q['query'] and not r['generation_error'] and not r['truncated']
        ss = json.loads(r['raw_output'])
        assert ss == r['parsed'] and isinstance(ss, list) and all(isinstance(s, str) for s in ss)
        runtime.append({'id': q['id'], 'qid': q['qid'], 'dataset': q['dataset'],
                        'question': q['query'], 'supporting_questions': ss, 'type': q['type']})
        provenance.append({'id': q['id'], 'source': str(p.relative_to(ROOT)), 'sha256': sha(p),
                           'decomposition_seconds': r['elapsed_seconds'],
                           'decomposition_input_tokens': r['input_tokens'],
                           'decomposition_output_tokens': r['output_tokens']})
    save_rows(HERE / 'runtime300.frozen.jsonl', runtime)
    save_rows(HERE / 'r02_provenance300.frozen.jsonl', provenance)
    save_rows(HERE / 'queries300.query_only.frozen.jsonl', [{'id': q['id'], 'query': q['query']} for q in questions])
    # Gold stays in this local scoring-only file, never in runtime/remote packages.
    import pandas as pd
    hot = {r['id']: r for r in pd.read_parquet(ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/preparation/offline_gold_source/validation.parquet').to_dict('records')}
    wiki = {r['_id']: r for r in read(ROOT / 'data/multihop_comprehensive_v1_20261015/2wiki/native/dev.json')}
    mu = {r['id']: r for r in rows(ROOT / 'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_dev.jsonl')}
    refs = []
    for q in questions:
        ds, qid = q['dataset'], q['qid']
        r = {'hotpotqa': hot, '2wiki': wiki, 'musique': mu}[ds][qid]
        assert r['question'] == q['query']
        refs.append({'id': q['id'], 'qid': qid, 'dataset': ds, 'question': q['query'],
                     'type': q['type'], 'answer': r['answer'],
                     'answer_id': r.get('answer_id'), 'aliases': r.get('answer_aliases', []),
                     'generation_input_forbidden': True})
    save_rows(HERE / 'references300.local.frozen.jsonl', refs)
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    for ds in ['hotpotqa', '2wiki', 'musique']:
        score = metric(ds)
        assert all(score(r['answer'], r)['f1'] == 1 for r in refs if r['dataset'] == ds)
    resources = read(HERE.parent / 'component_chain_eval60_v1_20261019/SELECTION.frozen.json')['resources']
    # Same deterministic random insertion policy for the two historical QA methods.
    positions = {q['qid']: int(hashlib.sha256(('complete-eval300-position-v1|' + q['id']).encode()).hexdigest(), 16) % 10 + 1 for q in questions}
    save(HERE / 'helper_positions300.frozen.json', positions)
    # Only pre-existing literal reference-style plans may be reused for S02/c06.
    old = {}; audit = []
    for ds in resources:
        wanted = {q['qid']: q for q in questions if q['dataset'] == ds}
        for p in sources(ds):
            with sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True) as db:
                for key, value in db.execute("SELECT key,value FROM data WHERE kind='decomposition'"):
                    if key not in wanted: continue
                    r = json.loads(value)
                    if not isinstance(r.get('qdmr'), dict): continue
                    rid = wanted[key]['id']
                    if rid in old: assert old[rid]['qdmr'] == r['qdmr'], 'Conflicting old plans; do not select by quality'
                    old[rid] = {'id': rid, 'qid': key, 'question': wanted[key]['query'], 'qdmr': r['qdmr'], 'source': str(p.relative_to(ROOT))}
    save_rows(HERE / 'reference_qdmr.cached.frozen.jsonl', sorted(old.values(), key=lambda r: r['id']))
    for name in ARMS:
        audit.append({'method': name, 'planned': 300, 'ready_decompositions': 300 if name.startswith(('B0', 'R02')) else len(old)})
    pins = {}
    roots = [HERE.parent / 'component_chain_c08_c02_v1_20261019', HERE.parent / 'component_chain_opt_v2_20261019',
             HERE.parent / 'hotpot_formal_eval_v1/methods/s02_child4_faithful',
             HERE.parent / 'contextual_auxiliary_rag_v1', HERE.parent / 'multigrain_prompt_v3_20261017',
             HERE.parent / 'rewrite_api_baseline_c02_v1_20261019', HERE.parent / 'child_answer_accuracy_v1_20261017']
    for d in roots:
        for p in d.glob('*'):
            if p.is_file() and (p.suffix == '.py' or p.name in ['examples.json', 'c08_examples.json', 'final_examples.json', 'SELECTED.frozen.json', 'PROTOCOL.frozen.json']):
                pins[str(p.relative_to(ROOT))] = sha(p)
    plan = {'frozen_utc': utc(), 'planned': 300, 'datasets': dict(Counter(q['dataset'] for q in questions)),
            'methods': ARMS, 'sample': str((EVAL / 'questions300.frozen.jsonl').relative_to(ROOT)),
            'sample_sha256': sha(EVAL / 'questions300.frozen.jsonl'), 'runtime_sha256': sha(HERE / 'runtime300.frozen.jsonl'),
            'references_sha256': sha(HERE / 'references300.local.frozen.jsonl'), 'r02_identity': ii,
            'model': 'qwen-plus-2025-12-01', 'temperature': 0, 'enable_thinking': False,
            'resources': resources, 'helper_positions_sha256': sha(HERE / 'helper_positions300.frozen.json'),
            'common_controls': 'same300 query IDs, generator snapshot/decoding, per-dataset corpus/BGE and native F1/EM scoring',
            'method_identity': 'Each whole method retains its own decomposition, prompts, goal handling, search counts and token limits. Not a same-final-prompt ablation.',
            'primary_ranking': 'Native final Answer F1 on four-way common scorable queries; also report planned300 missingness/bounds and per-dataset results. No pooled unmatched means.',
            'failure_policy': 'Known intermediate-length algorithm failures score0; provider/transport/budget missingness stays missing. Unknown receipts retained without replay.',
            'cache_policy': 'Exact full API payload only; immutable resource/query retrieval keys. Re-execute controller, never select answers by gold or transplant another method state.',
            'cost_policy': 'Logical standalone calls/tokens/cost per full method and actual incremental submissions/usage/unknown reserves separately; remote decomposition GPU seconds separately, no invented GPU monetary price.',
            'budget': {'cumulative_authorized_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'this_campaign_subcap_cny': 15,
                       'authorization': 'User explicitly raised cumulative budget to300 in current evaluation turn; not a new300 allowance.'},
            'source_pins': pins, 'availability': audit,
            'limitations': ['This pre-existing300 was previously used for decomposition evaluation and natural-error diagnostics; not untouched new holdout.',
                           'c06 is a unified-resource transfer of its full packet-building+contextualQA+criterion-final recipe, not historical600 cached-score replay.',
                           'No retuning on these300 outcomes, no hallucination judge calls, no component/ablation arms in main ranking.']}
    save(frozen, plan)
    save(OUT / 'PREPARATION.json', {'phase': 'offline_frozen', 'availability': audit, 'new_api_calls': 0, 'budget_cny': 300})
    log('freeze_complete_eval300', methods=ARMS, queries=300, budget_cny=300, old_reference_plans=len(old), new_api_calls=0)
    return plan


def package_remote(plan):
    """Only missing old-style QDMR plans; R02 already has all300 actual outputs."""
    cached = {q['id'] for q in rows(HERE / 'reference_qdmr.cached.frozen.jsonl')}
    missing = [q for q in rows(HERE / 'queries300.query_only.frozen.jsonl') if q['id'] not in cached]
    if not missing: return None
    template = OUTPUT / 'dev200_remote_20261014/remote_package'
    before = read(template / 'PACKAGE_MANIFEST.json')
    for name, expected in before['files'].items(): assert sha(template / name) == expected
    parent = OUT / 'remote_reference_qdmr300'; package = parent / 'package'; package.mkdir(parents=True, exist_ok=True)
    run_name = 'complete_pipeline_eval300_reference_qdmr_20261020'
    cfg = read(template / 'predict200.proposed.json')
    for key in ['input_file', 'output_dir']: cfg[key] = cfg[key].replace(before['run_name'], run_name)
    cfg['scope'] = 'accepted_complete_methods_eval300_reference_qdmr_only'
    save(package / 'predict200.proposed.json', cfg)
    save_rows(package / 'queries200.query_only.jsonl', missing)
    text = (template / 'prepare_remote.py').read_text(encoding='utf-8')
    text = text.replace('len(rows) != 200', f'len(rows) != {len(missing)}').replace('}) != 200', f'}}) != {len(missing)}').replace('exactly200', f'exactly{len(missing)}').replace('"inputs": 200', f'"inputs": {len(missing)}').replace('qdmr4b-dev200', 'qdmr4b-eval300')
    compile(text, 'prepare_remote.py', 'exec')
    (package / 'prepare_remote.py').write_text(text, encoding='utf-8', newline='\n')
    task = f'''Execute ONLY old-style query decomposition for {len(missing)} records.
Use unchanged checkpoint-000148, original instruction and decoding. Do not use R02 or convert its arrays.
No retrieval, answering, training, gold access, scoring, or API calls.
Preserve IDs verbatim. Historical filenames contain200; actual frozen count is {len(missing)}.
Verify PACKAGE_MANIFEST.json; then run python prepare_remote.py --submit on the existing deployment.
If needed use --route-root to change deployment paths only. Reconcile Slurm state before resuming.
Run name: {run_name}
Return the complete runs/{run_name}/ folder including raw predictions, runtime config, identities, diagnostics, elapsed/token logs;
also execution/package/ (predictor source and its identity), model_identity/BASE_MODEL_IDENTITY.json,
checkpoint_identity/checkpoint-000148/manifest.json, and a SHA256 RETURN_FILE_MANIFEST.json.
Zip as {run_name}_returned.zip. Return no weights or credentials.
'''
    (package / 'TASK_FOR_REMOTE_CODEX.txt').write_text(task, encoding='utf-8')
    manifest = {k: v for k, v in before.items() if k != 'files'}
    manifest.update(inputs=len(missing), queries=len(missing), run_name=run_name, scope=cfg['scope'],
                    task='complete_eval300_missing_reference_qdmr', local_source_sha256=sha(HERE / 'queries300.query_only.frozen.jsonl'))
    manifest['files'] = {p.name: sha(p) for p in sorted(package.iterdir()) if p.name != 'PACKAGE_MANIFEST.json'}
    save(package / 'PACKAGE_MANIFEST.json', manifest)
    archive = parent / 'reference_qdmr300_remote_package.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.iterdir()): z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in manifest['files'].items(): assert hashlib.sha256(z.read(name)).hexdigest() == expected
        assert all(set(r) == {'id', 'query'} for r in rows(package / 'queries200.query_only.jsonl'))
    upload = f"$ErrorActionPreference = 'Stop'\n$eval300Package = '{archive.resolve()}'\n& scp $eval300Package 'u7830488@cluster1.comp.anu.edu.au:/home/remote/u7830488/densex_workspace/'\nif ($LASTEXITCODE -ne 0) {{ throw 'scp failed' }}\n"
    (parent / 'UPLOAD.ps1').write_text(upload, encoding='utf-8')
    save(parent / 'PACKAGE_AUDIT.json', {'count': len(missing), 'archive': str(archive.relative_to(ROOT)), 'sha256': sha(archive),
                                        'query_only': True, 'old_model': 'checkpoint-000148', 'remote_executed': False})
    return archive


if __name__ == '__main__':
    p = prepare(); archive = package_remote(p)
    print(json.dumps({'planned': p['planned'], 'availability': p['availability'], 'remote_package': str(archive) if archive else None, 'new_api_calls': 0}, ensure_ascii=False))
