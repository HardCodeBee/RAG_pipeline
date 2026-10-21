"""Frozen extension of complete300: ordinary RAG and native ReAct."""
from pathlib import Path
import json
import sqlite3
from experiments.query_decomposition.complete_pipeline_comparison_v1_20261020.prepare300 import ROOT, OUTPUT, HERE as PREVIOUS, OUT as OLD_OUT, rows, read, sha, save
from experiments.query_decomposition.complete_pipeline_comparison_v1_20261020.result_sources300 import attempt_roots, select_result, database_for
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
ARMS = ['b00_rag_top10_v1', 'b03_react_dense']
OLD_ARMS = ['B0_legacy_multigrain_R02', 'R02_C08_c02_r1p1_integrated_candidate', 's02_child4']
MODULES = {'b00_rag_top10_v1': 'rag_top10_v1', 'b03_react_dense': 'react_dense'}


def log(event, **fields):
    HERE.mkdir(exist_ok=True, parents=True)
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def attempts():
    return sorted((OUT / 'attempts').glob('r[0-9][0-9][0-9]'))


def scorable(x):
    return isinstance(x.get('prediction'), str) and (x.get('status', '').startswith('success') or x.get('status') == 'algorithm_failure')


def select(ds, arm, qid):
    values = [(p, read(p)) for root in attempts() if (p := root / 'cases' / ds / arm / (qid + '.json')).exists()]
    correction = (HERE / 'PROTOCOL.v002.frozen.json').exists()
    def wrapper_rejected(x):
        return correction and arm == 'b03_react_dense' and x.get('reason') == 'Incomplete response retained, not parsed or propagated'
    for p, x in values:
        if scorable(x) and not wrapper_rejected(x): return p, x
    if values:
        p, x = values[-1]
        if wrapper_rejected(x):
            return p, {**x, 'status': 'controller_replay_required', 'prediction': None, 'original_status': x['status']}
        return p, x
    return None, {'status': 'unstarted', 'prediction': None}


def protocol_path():
    v3 = HERE / 'PROTOCOL.v003.frozen.json'
    if v3.exists(): return v3
    v2 = HERE / 'PROTOCOL.v002.frozen.json'
    return v2 if v2.exists() else HERE / 'PROTOCOL.frozen.json'


def source_databases(ds):
    from experiments.query_decomposition.complete_pipeline_comparison_v1_20261020.prepare300 import sources
    paths = sources(ds)
    for root in attempt_roots() + attempts():
        paths += sorted((root / 'methods' / ds).glob('*/checkpoint.sqlite3'))
        p = root / 'retrieval' / ds / 'checkpoint.sqlite3'
        if p.exists(): paths.append(p)
    campaign = OUTPUT / 'comprehensive_v1_20261015'
    paths += sorted((campaign / 'campaign_api_v1' / ds / 'formal').glob('*/*/checkpoint.sqlite3'))
    paths += sorted((campaign / 'baseline_integrity_repair_v2_20261017/main' / ds).glob('**/checkpoint.sqlite3'))
    return sorted(set(paths))


def prepare():
    target = HERE / 'PROTOCOL.frozen.json'
    if target.exists(): return read(protocol_path())
    old = read(PREVIOUS / 'EVAL300_PROTOCOL.frozen.json')
    cases = rows(PREVIOUS / 'runtime300.frozen.jsonl')
    assert len(cases) == 300 and all('answer' not in c for c in cases)
    methods = ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods'
    pins = {}
    for folder in MODULES.values():
        for p in (methods / folder).iterdir():
            if p.is_file(): pins[str(p.relative_to(ROOT))] = sha(p)
    configs = {}
    campaign = OUTPUT / 'comprehensive_v1_20261015/campaign_api_v1'
    for ds in old['resources']:
        cfg = read(campaign / ds / 'BASELINE_CONFIGS_FROZEN.json')['selected']
        configs[ds] = {arm: cfg[arm] for arm in ARMS}
        assert configs[ds]['b03_react_dense'] == {'id': 'k10_s5', 'k': 10, 'max_search': 5}
    protected = []
    for c in cases:
        for arm in OLD_ARMS + ['c06_comparison_criterion']:
            p, x = select_result(c['dataset'], arm, c['qid'])
            assert scorable(x)
            protected.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    save(OUT / 'EXISTING_SUCCESS_PROTECTION.frozen.json', protected)
    plan = {'utc': utc(), 'sample': str((PREVIOUS / 'runtime300.frozen.jsonl').relative_to(ROOT)),
            'sample_sha256': sha(PREVIOUS / 'runtime300.frozen.jsonl'),
            'references': str((PREVIOUS / 'references300.local.frozen.jsonl').relative_to(ROOT)),
            'references_sha256': sha(PREVIOUS / 'references300.local.frozen.jsonl'),
            'methods': ARMS, 'comparison_existing_methods': OLD_ARMS, 'configs': configs,
            'model': old['model'], 'temperature': 0, 'enable_thinking': False,
            'resources': old['resources'], 'source_pins': pins,
            'generation_limits': {'Top10_final': 64, 'ReAct_action': 512},
            'cache_policy': 'Exact complete payload including generation limit; successful first stop response only. Never select by F1. Immutable resource/query retrieval keys.',
            'retry_policy': 'At most two recovery rounds after first pass; retry infrastructure failures only in fresh attempt namespaces; preserve all successful trajectory receipts and all unknown liabilities. No content-rejection or algorithm-failure retry.',
            'failure_policy': 'Known incomplete ReAct action/invalid action/no Finish = algorithm_failure scored0. Top10 final length response is retained and explicitly marked success_truncated, matching existing final-reader handling. Provider missing stays missing. First scorable trajectory retained.',
            'comparison': 'Same300 native final Answer F1/EM; five-way shared scorable cohort, each pair common cohort and missing bounds. Per-dataset100. Not equal search/call budgets or same-final-prompt ablation.',
            'budget': {'cumulative_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'extension_subcap_cny': 12},
            'concurrency': 3, 'new_training': False, 'new_remote_work': False,
            'limitations': ['Existing300 has already been used for decomposition diagnostics; not untouched holdout.', 'Shared-resource local ReAct adaptation with BGE Search and no Wiki Lookup; not original-paper reproduction.', 'Frozen64/512 generation limits are retained; length failures are disclosed, not silently propagated.']}
    save(target, plan)
    log('freeze', methods=ARMS, queries=300, existing_successes_protected=len(protected), budget=plan['budget'])
    return plan


def verify(plan):
    assert sha(ROOT / plan['sample']) == plan['sample_sha256']
    for p, h in plan['source_pins'].items(): assert sha(ROOT / p) == h, p
    for x in read(OUT / 'EXISTING_SUCCESS_PROTECTION.frozen.json'):
        assert sha(ROOT / x['path']) == x['sha256'], x['path']


def budget_status():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        a, r = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        local = db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
    return {'known_usage_cny': a, 'unknown_reserves_cny': r, 'occupied_cny': a+r, 'remaining_cny': 300-a-r, 'extension_occupied_cny': local, 'cap_cny': 300, 'invoice_verified': False}


if __name__ == '__main__':
    p = prepare(); verify(p)
    print(json.dumps({'planned': 300, 'methods': ARMS, 'budget': budget_status(), 'new_api_calls': 0}))
