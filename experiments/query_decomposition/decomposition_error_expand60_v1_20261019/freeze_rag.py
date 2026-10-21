"""Freeze definite R02 failures before any new downstream outcomes are read."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    final = HERE / 'RAG_PROTOCOL.frozen.json'
    if final.exists():
        return json.loads(final.read_text(encoding='utf-8'))
    plan = json.loads((HERE / 'PLAN_FROZEN.json').read_text(encoding='utf-8'))
    olddir = HERE.parent / 'decomposition_error_downstream_v1_20261018'
    old = json.loads((olddir / 'DIRECT11_PROTOCOL.frozen.json').read_text(encoding='utf-8'))
    reviews_path = HERE / 'reviews/ALL_REVIEWED.frozen.jsonl'
    reviewed = rows(reviews_path)
    candidates = rows(HERE / 'candidates300.query_only.frozen.jsonl')
    assert len(reviewed) >= len(candidates) == 300
    assert len({r['id'] for r in reviewed}) == len(reviewed)
    assert all(r['verdict'] in {'pass', 'fail', 'uncertain'} for r in reviewed)
    errors = [r for r in reviewed if r['verdict'] == 'fail']
    if len(errors) < 49:
        raise RuntimeError(f'Only{len(errors)} new definite errors; freeze/infer/review another query-only batch before RAG.')
    seed = plan['selection_after_review']['rank_seed']
    order = lambda r: hashlib.sha256((seed + r['id']).encode()).hexdigest()
    rare = sorted([r for r in errors if r['dataset'] == '2wiki'], key=order)
    common = sorted([r for r in errors if r['dataset'] != '2wiki'], key=order)
    selected = (rare + common)[:49]
    assert len(selected) == 49
    runtime = []
    for r in selected:
        questions = json.loads(r['raw_output'])
        assert isinstance(questions, list) and all(isinstance(q, str) for q in questions)
        runtime.append({'id': r['id'], 'qid': r['id'].split(':', 1)[1], 'dataset': r['dataset'],
                        'question': r['query'], 'supporting_questions': questions})
    assert not {r['id'] for r in runtime} & {r['id'] for r in rows(olddir / 'direct11.runtime.frozen.jsonl')}
    for name, values in [('new_errors.runtime.frozen.jsonl', runtime), ('selected_quality.frozen.jsonl', selected)]:
        p = HERE / name
        assert not p.exists()
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8')
    config = {k: old[k] for k in ['model', 'conditions', 'helper_position', 'retrieval_k', 'merge', 'history',
                                'goal', 'duplicate_policy', 'sentence_format', 'final_reader', 'network']}
    config.update(utc=datetime.now(timezone.utc).isoformat(), status='frozen_before_new_RAG',
                  planned_queries=49, old_queries_reused=11, total_planned=60,
                  runtime_sha256=sha(HERE / 'new_errors.runtime.frozen.jsonl'),
                  max_workers=2, local_api_subcap_cny=2.5, global_cap_cny=200, isolated_protected_cutoff_cny=199.5)
    config['resources'] = {ds: json.loads((OUTPUT / 'comprehensive_v1_20261015/campaign_api_v1' / ds / 'RESOURCE_FROZEN.json').read_text(encoding='utf-8'))['resources']
                           for ds in ['hotpotqa', '2wiki', 'musique']}
    source = dict(old['source_sha256'])
    for p in [HERE / 'PLAN_FROZEN.json', reviews_path, HERE / 'selected_quality.frozen.jsonl', HERE / 'run.py',
              HERE / 'freeze_rag.py', HERE / 'new_errors.runtime.frozen.jsonl']:
        source[str(p.relative_to(ROOT))] = sha(p)
    config['source_sha256'] = source
    final.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    event = {'utc': config['utc'], 'decision': 'Freeze49 new natural R02 errors and reuse old11 unchanged',
             'reason': 'Predetermined label-only/hash-order selection; no new RAG outputs observed.',
             'candidate_reviews': len(reviewed), 'available_definite_errors': len(errors),
             'selected_by_dataset': {ds: sum(r['dataset'] == ds for r in runtime) for ds in config['resources']},
             'protocol_sha256': sha(final)}
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return config


if __name__ == '__main__':
    prepare()
