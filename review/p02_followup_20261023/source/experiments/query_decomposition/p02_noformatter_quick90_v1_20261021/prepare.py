"""Freeze a score-blind 30/dataset development subset and a separate variant."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'src/p02_multigrain_timing_cachefix_v1'
PLAN_DIR = HERE.parent / 'unified_rag_eval_v1_20261021'
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name
DATASETS = ('hotpotqa', '2wiki', 'musique')
SEED = 20261021
METHOD = 'p02_multigrain_timing_cachefix_noformatter_v1'
DECISION_ID = 'P02_NOFORMATTER_QUICK90_V1_20261021'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')


def order(label):
    return hashlib.sha256(f'{SEED}|quick90|{label}'.encode('utf-8')).hexdigest()


def select(samples):
    # Uses only dataset, type and ID; no predictions, scores, runtime or success state.
    selected, allocation = [], {}
    for dataset in DATASETS:
        population = [v for v in samples if v['dataset'] == dataset]
        assert len(population) == 100
        strata = defaultdict(list)
        for item in population:
            strata[item['type']].append(item)
        quota = {kind: 30 * len(v) // 100 for kind, v in strata.items()}
        remaining = 30 - sum(quota.values())
        priority = sorted(strata, key=lambda kind: (
            -(30 * len(strata[kind]) % 100), order(f'{dataset}|quota|{kind}')))
        for kind in priority[:remaining]:
            quota[kind] += 1
        allocation[dataset] = quota
        for kind in sorted(strata):
            selected += sorted(strata[kind], key=lambda v: order(f'{dataset}|{kind}|{v["id"]}'))[:quota[kind]]
    assert len(selected) == len({v['id'] for v in selected}) == 90
    assert Counter(v['dataset'] for v in selected) == {d: 30 for d in DATASETS}
    return selected, allocation


def build_variant():
    source = (PACKAGE / 'core/pipeline.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'len(answer.split()) < 4']
    assert len(targets) == 1
    target = targets[0]
    lines = source.splitlines(keepends=True)
    del lines[target.lineno - 1:target.end_lineno]
    variant = ''.join(lines)
    variant = variant.replace('from . import METHOD, ROOT',
        'from src.p02_multigrain_timing_cachefix_v1.core import ROOT\n' + f'METHOD = {METHOD!r}', 1)
    variant = variant.replace('from . import child_prompts, rewrite_prompts, final_prompts',
        'from src.p02_multigrain_timing_cachefix_v1.core import child_prompts, rewrite_prompts, final_prompts', 1)
    assert 'legacy/sentence_format/' not in variant
    assert 'Express the candidate answer below' not in variant
    ast.parse(variant)
    (HERE / 'pipeline.py').write_text(variant, encoding='utf-8')
    (HERE / 'PIPELINE_CHANGE.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), variant.splitlines(True), fromfile='frozen_P02/core/pipeline.py',
        tofile='quick90_noformatter/pipeline.py')), encoding='utf-8')


def main():
    if (HERE / 'PROTOCOL.frozen.json').exists():
        raise FileExistsError('Keep this frozen cohort; use the recorded run command.')
    samples = rows(PACKAGE / 'data/runtime300.jsonl')
    selected, allocation = select(samples)
    # The cohort is fixed before references, existing predictions or receipts are opened.
    jsonl(HERE / 'runtime90.frozen.jsonl', selected)
    jsonl(HERE / 'quick90.questions.frozen.jsonl',
          [{k: v[k] for k in ['id', 'qid', 'dataset', 'question', 'type']} for v in selected])
    write(HERE / 'COHORT.frozen.json', {
        'id': 'p02_development_quick90_v1_20261021', 'seed': SEED,
        'sampling': 'per-dataset proportional type/hop stratification; largest remainder; SHA256 deterministic ties and ordering',
        'sampling_fields': ['dataset', 'type', 'id'], 'answer_or_score_used': False,
        'source': str((PACKAGE / 'data/runtime300.jsonl').relative_to(ROOT)).replace('\\', '/'),
        'source_sha256': sha(PACKAGE / 'data/runtime300.jsonl'), 'planned': 90,
        'per_dataset': {ds: 30 for ds in DATASETS}, 'strata': allocation,
        'selected_ids': [v['id'] for v in selected],
        'scope': 'Previously exposed development300 subset; not formal, independent or timing50 cohort'})

    refs_path = HERE.parent / 'p02_flow_simplify_v1_20261020/references300.local.frozen.jsonl'
    references = {v['id']: v for v in rows(refs_path)}
    jsonl(HERE / 'references90.local.frozen.jsonl', [references[v['id']] for v in selected])
    provenance = {v['id']: v for v in read(PACKAGE / 'SOURCE_PROVENANCE.json')['sources']}
    expected = {v['id']: v for v in rows(PACKAGE / 'data/expected300.jsonl')}
    bindings = []
    for item in selected:
        p = ROOT / provenance[item['id']]['source'].replace('\\', '/')
        source = read(p)
        assert source['id'] == item['id'] and sha(p) == provenance[item['id']]['sha256']
        bindings.append({'id': item['id'], 'source': str(p.relative_to(ROOT)).replace('\\', '/'),
                         'source_sha256': sha(p), 'final_request_id': source['request_id'],
                         'historical_formatter_triggers': sum(bool(n.get('sentence_format_request_id'))
                             for n in expected[item['id']]['nodes'])})
    jsonl(HERE / 'SOURCE_BINDINGS.frozen.jsonl', bindings)
    build_variant()
    package_files = read(PACKAGE / 'FROZEN.json')['files'] + ['FROZEN.json']
    package_hashes = {str((PACKAGE / p).relative_to(ROOT)).replace('\\', '/'): sha(PACKAGE / p)
                      for p in package_files}
    holds = [ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
             PLAN_DIR / 'STOP_REQUESTED',
             ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol = {'id': HERE.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'User requested 30 queries from each dataset of development300 and a quick test of P02 without the extra formatter.',
        'baseline': 'p02_multigrain_timing_cachefix_v1', 'variant': METHOD,
        'only_algorithm_change': 'Remove the answer-word-count conditional and extra sentence-format generation; retain all other prompts, stages and limits.',
        'execution': 'paired complete-pipeline exact-payload and exact-retrieval recorded replay',
        'execution_reason': 'No formatter trigger in the selected historical trajectories; unchanged inputs must reuse matching successes.',
        'new_inference': False, 'paid_fallback': False, 'main_evaluation_resume': False,
        'quality_metrics': ['native Answer F1', 'EM'],
        'cost_scope': 'Recorded hosted-LLM logical usage and historical frozen rates; precomputed local decomposition excluded and labelled. New paid API usage is zero.',
        'timing_scope': 'No fresh component or TTFT measurement; replay wall time is not a method efficiency result.',
        'cohort': 'COHORT.frozen.json', 'same_inputs': True,
        'inputs': {n: sha(HERE / n) for n in ['runtime90.frozen.jsonl', 'quick90.questions.frozen.jsonl',
                    'references90.local.frozen.jsonl', 'COHORT.frozen.json', 'SOURCE_BINDINGS.frozen.jsonl']},
        'source_pins': {n: sha(HERE / n) for n in ['prepare.py', 'pipeline.py', 'run.py']},
        'original_package_pins': package_hashes,
        'hold_pins': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in holds},
        'formatter_triggered_in_selected_original_nodes': sum(v['historical_formatter_triggers'] for v in bindings)}
    write(HERE / 'PROTOCOL.frozen.json', protocol)
    decision = {'id': DECISION_ID, 'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'approved_scoped_quick_test', 'authority': protocol['authority'],
        'scope': 'Development300 subset: HotpotQA30,2Wiki30,MuSiQue30; frozen original vs no-formatter variant; exact complete-flow cache replay.',
        'cohort': str((HERE / 'COHORT.frozen.json').relative_to(ROOT)).replace('\\', '/'),
        'protocol': str((HERE / 'PROTOCOL.frozen.json').relative_to(ROOT)).replace('\\', '/'),
        'seed': SEED, 'strata': allocation, 'new_API_requests': 0,
        'main_evaluation_remains_paused': True, 'frozen_primary_method_overwritten': False,
        'main_plan_revision': 6}
    # Append the scoped authorization while preserving concurrent plan additions.
    with (PLAN_DIR / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(decision, ensure_ascii=False) + '\n')
    for attempt in range(3):
        p = PLAN_DIR / 'PLAN.json'; old = p.read_bytes(); current = json.loads(old.decode('utf-8-sig'))
        assert current['execution_approval']['approved'] is False
        current.setdefault('standalone_diagnostics', []).append({
            'decision_id': DECISION_ID, 'scope': decision['scope'], 'protocol': decision['protocol'],
            'campaign_resume_authorized': False, 'new_inference': False})
        if p.read_bytes() != old:
            continue
        temporary = PLAN_DIR / (DECISION_ID + '.tmp')
        write(temporary, current); temporary.replace(p); break
    else:
        raise RuntimeError('Concurrent plan update; retain frozen experiment and review before execution.')
    with (PLAN_DIR / 'PLAN.txt').open('a', encoding='utf-8') as f:
        f.write('\n独立授权：P02去formatter快速90题（2026-10-21）\n'
                '从开发300题按原分层固定每集30题，仅比较冻结原版与独立去formatter变体。\n'
                '原轨迹未触发formatter，执行完整流程严格同输入缓存回放；不产生新推理、不恢复主评。\n'
                '名单、协议和结果见 ../p02_noformatter_quick90_v1_20261021/。冻结主方法不覆盖。\n')
    files = ['__init__.py', 'prepare.py', 'pipeline.py', 'run.py']
    write(HERE / 'experiment.json', {
        'schema_version': 1, 'id': 'query_decomposition/' + HERE.name,
        'protocol_version': 'paired_exact_replay_quick90_noformatter_v1', 'state': 'active',
        'files': [{'path': str((HERE / n).relative_to(ROOT)).replace('\\', '/'),
                   'role': 'Scoped quick90 preparation, separate no-formatter variant and offline full-flow replay',
                   'sha256': sha(HERE / n)} for n in files],
        'commands': {'run': str((HERE / 'run.py').relative_to(ROOT)).replace('\\', '/')},
        'status_source': str((OUT / 'REPORT.json').relative_to(ROOT)).replace('\\', '/'),
        'evidence_paths': [str((HERE / n).relative_to(ROOT)).replace('\\', '/')
                           for n in ['PROTOCOL.frozen.json', 'COHORT.frozen.json', 'PIPELINE_CHANGE.diff']],
        'depends_on': ['rag/p02_multigrain_timing_cachefix_v1']})
    print(json.dumps({'frozen': True, 'planned': 90, 'strata': allocation,
                      'old_formatter_triggers': protocol['formatter_triggered_in_selected_original_nodes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
