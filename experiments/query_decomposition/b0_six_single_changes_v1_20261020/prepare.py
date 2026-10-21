"""Isolate six new experiments; parent results and controllers remain read-only."""
from collections import Counter
from pathlib import Path
import json
import hashlib
import sqlite3
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc
from experiments.query_decomposition.complete_pipeline_comparison_v1_20261020 import result_sources300 as parent
from experiments.query_decomposition.child_answer_accuracy_v1_20261017.candidate_c08 import is_no_answer
from .pipeline import ARMS

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
PARENT_HERE = HERE.parent / 'complete_pipeline_comparison_v1_20261020'
PARENT_OUT = OUTPUT / PARENT_HERE.name
BASELINE = 'B0_legacy_multigrain_R02'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if p.exists(): assert p.read_text(encoding='utf-8') == text, str(p)
    else: p.write_text(text, encoding='utf-8')
def save_rows(p, values):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values)
    if p.exists(): assert p.read_text(encoding='utf-8') == text, str(p)
    else: p.write_text(text, encoding='utf-8')
def log(event, **fields):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def prepare():
    if (HERE / 'PROTOCOL.frozen.json').exists(): return read(HERE / 'PROTOCOL.frozen.json')
    plan = read(PARENT_HERE / 'EVAL300_PROTOCOL.frozen.json')
    runtime = rows(PARENT_HERE / 'runtime300.frozen.jsonl')
    assert len(runtime) == len({c['id'] for c in runtime}) == 300
    assert Counter(c['dataset'] for c in runtime) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    assert sha(PARENT_HERE / 'runtime300.frozen.jsonl') == plan['runtime_sha256']
    manifest, triggers, counts = [], {}, Counter()
    for c in runtime:
        p, r = parent.select_result(c['dataset'], BASELINE, c['qid'])
        assert r['status'].startswith('success') and isinstance(r['prediction'], str)
        calls = parent.logical_calls(parent.database_for(p), c['qid'])
        assert calls and all(v['status'] == 'success' for v in calls)
        counts['nodes'] += len(r['nodes'])
        flags = {'D_drop_whole_abstentions': any(is_no_answer(n['answer']) for n in r['nodes']),
                 'E_remove_sentence_formatter': any(n.get('sentence_format_request_id') for n in r['nodes']),
                 'F_child_length_fallback': any('/answer/' in v['stage'] and v.get('finish_reason') == 'length' for v in calls)}
        for arm, flag in flags.items():
            if flag: triggers.setdefault(arm, []).append(c['id'])
        manifest.append({'id': c['id'], 'qid': c['qid'], 'dataset': c['dataset'],
                         'source': str(p.relative_to(ROOT)), 'source_sha256': sha(p),
                         'logical_calls': calls, 'program_triggers': flags})
    save_rows(HERE / 'runtime300.frozen.jsonl', runtime)
    save_rows(HERE / 'B0_SOURCES.frozen.jsonl', manifest)
    references = rows(PARENT_HERE / 'references300.local.frozen.jsonl')
    save_rows(HERE / 'references300.local.frozen.jsonl', references)
    pins = dict(plan['source_pins'])
    for f in [PARENT_HERE / 'EVAL300_PROTOCOL.frozen.json', PARENT_HERE / 'FINAL_RESULTS.frozen.json',
              HERE / 'pipeline.py', HERE / 'prepare.py', PARENT_HERE / 'result_sources300.py']:
        if f.exists(): pins[str(f.relative_to(ROOT))] = sha(f)
    spec = {'frozen_utc': utc(), 'task': 'Six independent one-factor interventions on frozen B0',
            'authorization': 'User in this side conversation explicitly requested testing each of these six edits on B0.',
            'arms': ARMS, 'baseline': BASELINE, 'planned_queries': 300, 'datasets': dict(Counter(c['dataset'] for c in runtime)),
            'runtime_sha256': sha(HERE / 'runtime300.frozen.jsonl'), 'baseline_sources_sha256': sha(HERE / 'B0_SOURCES.frozen.jsonl'),
            'references_sha256': sha(HERE / 'references300.local.frozen.jsonl'), 'model': plan['model'],
            'temperature': 0, 'enable_thinking': False, 'resources': plan['resources'],
            'preserved': ['same actual R02/v12 arrays', 'original-first dual retrieval and dedupTop10',
                          'original query appended as goal and answered before separate final', 'B0 formatter unless factor E',
                          'all originalTop10 plus QA helper at front', 'B0 v3 final unless factor C',
                          'rewrite256 child128 formatter128 final64', 'no new training or remote job'],
            'factor_A_boundary': 'C08 instructions/examples/layout, with ALL B0 previous QA preserved. Strip renderer-embedded runtime history filtering to avoid mixing factor D.',
            'factor_D_boundary': 'Use the frozen C08 exact whole-response predicate on B0 outputs, at rewrite/child/final interfaces; keep partial facts and supported No/0/None.',
            'factor_F_boundary': 'Only known child-answer length termination. Preserve B0 completed QA without C08 filtering and use unchanged B0 final prompt; do not include P1.',
            'offline_baseline_trigger_audit': {'nodes': counts['nodes'], 'triggered_queries': {arm: len(triggers.get(arm, [])) for arm in ['D_drop_whole_abstentions','E_remove_sentence_formatter','F_child_length_fallback']}},
            'evaluation': 'Final native Answer F1 primary. Compare each independent arm with B0 on the SAME common questions; full300 whenever available. Dataset breakdown, paired bootstrap95, gains/ties/losses, final abstentions and cost.',
            'sample_role': 'Previously used frozen300 development/diagnostic cohort; not a fresh holdout and not selected on failures.',
            'cache': 'Baseline own exact-payload successful receipts first, then exact successful cache. Never change existing successes or choose by F1. Inactive factors require complete byte-identical offline replay before reuse.',
            'execution': {'max_workers': 3, 'max_rounds': 4, 'retry_eligibility': 'Provider connection/timeout or transient HTTP/dispatch only; no retries of scorable answers or known algorithmic truncation.',
                          'retry_identity': 'New immutable round namespace and financial liability; old unknown rows/reserves retained. Reuse successful prefix payloads.'},
            'budget': {'cumulative_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'experiment_subcap_cny': 15,
                       'scope': 'Existing cumulative project ceiling retained as a conservative limit; not an extra300 allowance.'},
            'source_pins': pins, 'parent_mutations': False}
    save(HERE / 'PROTOCOL.frozen.json', spec)
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        actual, reserve = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    save(OUT / 'BUDGET_START.json', {'utc': utc(), 'usage_cny': actual, 'reserved_cny': reserve, 'occupied_cny': actual + reserve, 'cap_cny': 300})
    log('freeze_six_independent_changes', queries=300, factors=ARMS, initial_program_triggers=spec['offline_baseline_trigger_audit'], new_api_calls=0)
    return spec


if __name__ == '__main__':
    x = prepare(); print(json.dumps({'queries': 300, 'arms': list(ARMS), 'trigger_audit': x['offline_baseline_trigger_audit'], 'new_api_calls': 0}))
