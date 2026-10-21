"""Freeze a development300 winner, then confirm it against p02 on fixed60.

Only --run enables actual RAG. This reuses the frozen run.py transport and
run_case; its scoped adapters change paths, sample membership and baseline
label only. It never changes original300 inputs, frozen source, or old pins.
The60 were exposed to decomposer selection; they are new downstream queries,
not an untouched holdout. MuSiQue20 come from native train heldout queries.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sqlite3
from threading import Lock

from . import pipeline as frozen_pipeline
from . import run as v1
from . import score
from experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_runner import execute_fresh


HERE = Path(__file__).resolve().parent
FRESH_HERE = HERE / 'fresh60'
PROJECT_OUT = v1.OUT
OUT = PROJECT_OUT / 'fresh60'
RUNTIME = FRESH_HERE / 'runtime60.frozen.jsonl'
REFERENCES = FRESH_HERE / 'references60.local.frozen.jsonl'
PROTOCOL = FRESH_HERE / 'PROTOCOL.frozen.json'
RULE = FRESH_HERE / 'SELECTION_RULE.frozen.json'
CHOICE = OUT / 'SELECTION_FROM300.frozen.json'
BASELINE = 'p02_role_headers'
CANDIDATES = ('union20', 'discount20', 'single_final')
_PATCH_LOCK = Lock()
_PARENT_BUDGET = v1.Budget


def _copy_bytes(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), str(target)
    else:
        target.write_bytes(source.read_bytes())


def _fresh_cases_exist():
    return any(OUT.glob('rounds/*/cases/*/*/*.json'))


def prepare():
    """Freeze selection BEFORE accessing new60 scores; no generator runs."""
    parent = v1.prepare()
    if not RULE.exists() and _fresh_cases_exist():
        raise RuntimeError('Cannot create a retrospective selection rule after fresh case generation')
    FRESH_HERE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rule = {
        'eligible_candidates': list(CANDIDATES), 'cohort': 'complete original development300',
        'metric': 'Native final Answer F1 over all300, algorithm failures zero, no infrastructure missing',
        'eligibility': 'All3 candidates must each have300 scorable cases before choosing any one',
        'ordering': ['descending native F1_complete', 'ascending successful logical generation calls',
                     'ascending logical prompt_tokens + completion_tokens', 'ascending arm name'],
        'call_cost_tiebreak': 'Logical calls/tokens include exact successful cache reuse; not actual new billed requests',
        'fresh_answers_or_scores_used_for_selection': False,
        'new60_membership_changes_allowed': False, 'new60_prompt_tuning_allowed': False,
        'baseline': BASELINE, 'baseline_execution': 'timing_only; all model payloads byte-equivalent to frozen p02',
        'candidate_execution': 'Unchanged frozen isolated pipeline.execute for the selected arm',
        'scope': 'New downstream RAG comparison only; previously exposed decomposer dev60, not unseen holdout',
    }
    v1.freeze(RULE, rule)
    provenance = v1.read(HERE / 'FRESH60.PROVENANCE.proposed.json')
    proposed_runtime = HERE / 'fresh60.runtime.proposed.jsonl'
    proposed_refs = HERE / 'fresh60.references.local.proposed.jsonl'
    assert provenance['runtime']['sha256'] == v1.sha(proposed_runtime)
    assert provenance['references_local_scoring_only']['sha256'] == v1.sha(proposed_refs)
    samples = v1.rows(proposed_runtime)
    assert len(samples) == len({row['id'] for row in samples}) == 60
    assert Counter(row['dataset'] for row in samples) == {'hotpotqa': 20, '2wiki': 20, 'musique': 20}
    for row in samples:
        frozen_pipeline.safe_case(row)
    parent_ids = {row['id'] for row in v1.rows(HERE / 'runtime300.frozen.jsonl')}
    assert not parent_ids & {row['id'] for row in samples}
    for binding in provenance['prediction_bindings']:
        assert v1.sha(v1.ROOT / binding['path']) == binding['sha256']
    _copy_bytes(proposed_runtime, RUNTIME)
    _copy_bytes(proposed_refs, REFERENCES)
    _copy_bytes(HERE / 'FRESH60.PROVENANCE.proposed.json', FRESH_HERE / 'PROVENANCE.frozen.json')
    # The reused runner's seed-manifest interface is intentionally empty here:
    # these60 have no historical p02 RAG packets. Live cache keys remain qid,
    # exact query, k and the complete resource object. No gold packet seeding.
    v1.freeze(FRESH_HERE / 'BASELINE_SOURCES.frozen.json', [])
    pins = dict(parent['source_pins'])
    for name in ('fresh_evaluation.py', 'prepare_fresh60.py'):
        path = HERE / name
        pins[str(path.relative_to(v1.ROOT))] = v1.sha(path)
    files = [RULE, RUNTIME, REFERENCES, FRESH_HERE / 'PROVENANCE.frozen.json',
             FRESH_HERE / 'BASELINE_SOURCES.frozen.json']
    plan = {
        **{key: deepcopy(parent[key]) for key in ('model', 'temperature', 'enable_thinking', 'resources', 'readout')},
        'task': HERE.name + '/fresh60', 'baseline': BASELINE, 'planned': 60,
        'candidate': 'Read only the frozen SELECTION_FROM300.frozen.json at dispatch',
        'source_pins': pins,
        'input_pins': {str(path.relative_to(v1.ROOT)): v1.sha(path) for path in files},
        'selection_rule_sha256': v1.sha(RULE), 'parent_protocol_sha256': v1.sha(HERE / 'PROTOCOL.frozen.json'),
        'runtime': str(RUNTIME.relative_to(v1.ROOT)), 'runtime_sha256': v1.sha(RUNTIME),
        'reference_file_scoring_only': str(REFERENCES.relative_to(v1.ROOT)),
        'sample_mask': [row['id'] for row in samples],
        'dataset_counts': dict(Counter(row['dataset'] for row in samples)),
        'workers': 3, 'max_rounds': 2, 'global_cap_cny': 300,
        'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 25,
        'local_subcap_scope': str(PROJECT_OUT.relative_to(v1.OUTPUT)),
        'cache': 'Exact successful full payloads from this fresh campaign with verified identical protocol resources only',
        'recovery': 'At most one immutable r001 infrastructure recovery; no successful or algorithm-failed rerun',
        'baseline_adapter': 'p02_role_headers -> timing_only, no goal/final prompt changes',
        'limitations': provenance['limitations'],
    }
    v1.freeze(PROTOCOL, plan)
    verify(plan)
    return plan


def verify(plan):
    # Resolve original inputs against original HERE explicitly. The reused
    # runner's globals are temporarily scoped to FRESH_HERE during dispatch.
    parent = v1.read(HERE / 'PROTOCOL.frozen.json')
    for path, expected in parent['source_pins'].items():
        assert v1.sha(v1.ROOT / path) == expected, path
    for path, expected in parent['input_hashes'].items():
        assert v1.sha(HERE / path) == expected, path
    for path, expected in {**plan['source_pins'], **plan['input_pins']}.items():
        assert v1.sha(v1.ROOT / path) == expected, path
    assert plan['resources'] == v1.read(HERE / 'PROTOCOL.frozen.json')['resources']
    assert plan['selection_rule_sha256'] == v1.sha(RULE)
    assert [row['id'] for row in v1.rows(RUNTIME)] == plan['sample_mask']


def select_candidate(plan=None):
    """Choose using only complete300 scores; never read fresh references."""
    plan = prepare() if plan is None else plan
    verify(plan)
    if CHOICE.exists():
        value = v1.read(CHOICE)
        assert value['selection_rule_sha256'] == v1.sha(RULE)
        assert value['selected_arm'] in CANDIDATES
        assert v1.sha(OUT / 'SELECTION_SCORES300.frozen.json') == value['scores300_sha256']
        return value
    if _fresh_cases_exist():
        raise RuntimeError('Cannot select a new method after fresh generation has started')
    # This scorer is explicitly bound to original300, not new60.
    summary = score.summarize(
        PROJECT_OUT, HERE / 'runtime300.frozen.jsonl', arms=CANDIDATES,
        references=HERE / 'references300.local.frozen.jsonl', baseline_root=None)
    incomplete = {arm: value['scorable'] for arm, value in summary['arms'].items()
                  if value['planned'] != 300 or value['scorable'] != 300 or value['F1_complete'] is None}
    if incomplete:
        raise RuntimeError('Wait for all3 complete300 candidates; scorable counts=' + json.dumps(incomplete))
    candidates = []
    for arm in CANDIDATES:
        value = summary['arms'][arm]
        usage = value['logical_usage_on_selected_case_records']
        candidates.append({'arm': arm, 'native_F1': value['F1_complete'],
                           'logical_generation_calls': usage.get('successful_logical_calls', 0),
                           'logical_tokens': usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0),
                           'scorable': value['scorable']})
    candidates.sort(key=lambda row: (-row['native_F1'], row['logical_generation_calls'],
                                    row['logical_tokens'], row['arm']))
    v1.freeze(OUT / 'SELECTION_SCORES300.frozen.json', summary)
    chosen = {'selected_arm': candidates[0]['arm'], 'ordered_candidates': candidates,
              'selection_rule_sha256': v1.sha(RULE),
              'scores300_sha256': v1.sha(OUT / 'SELECTION_SCORES300.frozen.json'),
              'fresh_runtime_sha256': v1.sha(RUNTIME),
              'fresh_results_read': False, 'fresh_reference_values_read': False}
    v1.freeze(CHOICE, chosen)
    return chosen


class _PipelineProxy:
    safe_case = staticmethod(frozen_pipeline.safe_case)

    @staticmethod
    def execute(case, retrieve, ask, *, arm, **kwargs):
        actual_arm = 'timing_only' if arm == BASELINE else arm
        result = frozen_pipeline.execute(case, retrieve, ask, arm=actual_arm, **kwargs)
        result['arm'] = arm
        result['fresh_execution_arm'] = actual_arm
        return result


class _ProjectBudget(_PARENT_BUDGET):
    def reserve(self, rid, namespace, amount):
        """Keep25 across original campaign AND this fresh subdirectory."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            own = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                  (str(PROJECT_OUT.relative_to(v1.OUTPUT)) + '%',)).fetchone()[0]
            if total + amount > 299.9 or own + amount > 25:
                raise RuntimeError('Cumulative300 or shared project25 dispatch ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def _snapshot():
    with sqlite3.connect((v1.OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        def sums(prefix=None):
            query = 'SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges'
            return db.execute(query + (' WHERE namespace LIKE ?' if prefix else ''), (prefix + '%',) if prefix else ()).fetchone()
        global_cost = sums()
        project = sums(str(PROJECT_OUT.relative_to(v1.OUTPUT)))
        fresh = sums(str(OUT.relative_to(v1.OUTPUT)))
    return {'global_known_cny': global_cost[0], 'global_reserved_cny': global_cost[1],
            'global_occupied_cny': sum(global_cost[:2]), 'global_cap_cny': 300,
            'experiment_known_cny': project[0], 'experiment_reserved_cny': project[1],
            'experiment_submissions': project[2], 'local_subcap_cny': 25,
            'fresh_known_cny': fresh[0], 'fresh_reserved_cny': fresh[1],
            'fresh_submissions': fresh[2], 'actual_bill_verified': False}


@contextmanager
def _adapt_runner(plan, arms):
    """Process-local paths only; frozen files and source stay untouched."""
    _PATCH_LOCK.acquire()
    names = ('HERE', 'OUT', 'ARMS', 'pipeline', 'rows', 'verify', 'Budget', 'budget_snapshot', 'source_catalog')
    original = {name: getattr(v1, name) for name in names}
    original_rows = original['rows']
    def masked_rows(path):
        if Path(path).resolve() == (FRESH_HERE / 'runtime300.frozen.jsonl').resolve():
            values = original_rows(RUNTIME)
            assert [row['id'] for row in values] == plan['sample_mask']
            return values
        return original_rows(path)
    def fresh_catalog(dataset):
        # Never mix same-looking prompts from another retrieval resource or
        # another experiment. Every incoming database must prove this exact
        # immutable fresh protocol identity before the old catalog reads it.
        for path in sorted(OUT.glob(f'rounds/r*/methods/{dataset}/*/checkpoint.sqlite3')):
            with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
                assert row is not None, str(path)
                identity = json.loads(row[0])
                assert identity['protocol_sha256'] == v1.sha(PROTOCOL), str(path)
                assert identity['dataset'] == dataset, str(path)
        return original['source_catalog'](dataset)
    try:
        v1.HERE, v1.OUT, v1.ARMS = FRESH_HERE, OUT, list(arms)
        v1.pipeline, v1.rows = _PipelineProxy, masked_rows
        v1.verify, v1.Budget, v1.budget_snapshot = verify, _ProjectBudget, _snapshot
        v1.source_catalog = fresh_catalog
        yield
    finally:
        for name, value in original.items():
            setattr(v1, name, value)
        _PATCH_LOCK.release()


def self_test(plan=None, *, save=True):
    """Only synthetic packets/callbacks; checks real frozen prompt renderers."""
    plan = prepare() if plan is None else plan
    case = {'id': '2wiki:fresh-contract', 'qid': 'fresh-contract', 'dataset': '2wiki',
            'question': 'Where was the husband of Sample Person born?',
            'supporting_questions': ['Who is the husband of Sample Person?']}
    def exercise(executor, arm):
        calls, queries, nodes = [], [], []
        def get(query):
            queries.append(query)
            docs = [{'doc_id': query + str(i), 'source': f'Source{i}', 'text': f'Source{i}\nSome actual text {i}.',
                     'score': 1 / i, 'rank': i,
                     'block': f'[Chunk {i} | source=Source{i}, pages=1-1]\nSource{i}\nSome actual text {i}.'}
                    for i in range(1, 11)]
            return {'qid': case['qid'], 'query': query, 'k': 10, 'docs': docs}
        def ask(stage, prompt, limit, **kwargs):
            calls.append((stage, prompt, limit, kwargs))
            answer = 'Where was Sample Husband born?' if '/rewrite/' in stage else (
                'Example Town' if stage == 'legacy/final' else 'The husband of Sample Person is Sample Husband.')
            return str(len(calls)), answer, 'stop'
        result = executor(case, get, ask, arm=arm, readout=plan['readout'], record_node=nodes.append)
        return result, calls, queries, nodes
    historical, old_calls, old_queries, old_nodes = exercise(execute_fresh, BASELINE)
    adapted, calls, queries, nodes = exercise(_PipelineProxy.execute, BASELINE)
    assert calls == old_calls and nodes == old_nodes
    assert historical['final_prompt'] == adapted['final_prompt'] and historical['helper'] == adapted['helper']
    assert old_queries[1:] == queries
    contracts = {}
    for arm in CANDIDATES:
        result, observed, _, actual_nodes = exercise(_PipelineProxy.execute, arm)
        assert result['arm'] == arm and result['status'] == 'success'
        if arm == 'single_final':
            assert all(node['role'] == 'ordinary' for node in actual_nodes)
            assert not any(stage == 'legacy/answer/2' for stage, *_ in observed)
            assert 'Candidate conclusion for the Original question' not in result['helper']
            assert len(result['final_reader_packet']['docs']) == 20
        else:
            assert actual_nodes[-1]['role'] == 'goal'
            assert len(actual_nodes[-1]['reader_packet']['docs']) == 20
        contracts[arm] = {'logical_calls': len(observed), 'status': result['status']}
    # The budget must not reset when the output directory moves into fresh60.
    ledger = sqlite3.connect(':memory:')
    ledger.execute('CREATE TABLE charges(id TEXT PRIMARY KEY,namespace TEXT,reserved REAL,actual REAL,status TEXT)')
    ledger.execute('INSERT INTO charges VALUES(?,?,?,?,?)', (
        'old', str(PROJECT_OUT.relative_to(v1.OUTPUT)) + '/rounds/old', 24.99, 24.99, 'returned_usage'))
    ledger.commit()
    budget = object.__new__(_ProjectBudget)
    budget.db = ledger
    try:
        budget.reserve('new', str(OUT.relative_to(v1.OUTPUT)), .02)
    except RuntimeError as error:
        assert 'project25' in str(error)
    else:
        raise AssertionError('Fresh campaign reset the25 shared project cap')
    ledger.close()
    original_runtime_sha = v1.sha(HERE / 'runtime300.frozen.jsonl')
    original_pipeline = v1.pipeline
    with _adapt_runner(plan, (BASELINE, 'union20')):
        assert v1.rows(FRESH_HERE / 'runtime300.frozen.jsonl') == v1.rows(RUNTIME)
        assert v1.OUT == OUT and v1.HERE == FRESH_HERE
    assert v1.pipeline is original_pipeline and v1.sha(HERE / 'runtime300.frozen.jsonl') == original_runtime_sha
    result = {'passed': True, 'baseline_full_payloads_and_nodes_exact': True,
              'baseline_only_root_get_timing_changes': True,
              'candidate_contracts': contracts, 'fixed60_sample_mask_exact': True,
              'shared_project25_budget_contract': True, 'scoped_adapter_restored': True,
              'frozen_runtime300_unchanged': True, 'new_api_calls': 0, 'new_retrieval_calls': 0,
              'source_sha256': v1.sha(Path(__file__)), 'protocol_sha256': v1.sha(PROTOCOL)}
    if save:
        v1.freeze(OUT / 'OFFLINE_AUDIT.json', result)
    return result


def run(*, round_id=0, dataset=None):
    plan = prepare()
    chosen = select_candidate(plan)
    self_test(plan)
    arms = (BASELINE, chosen['selected_arm'])
    assert round_id in (0, 1)
    samples = v1.rows(RUNTIME)
    assert [row['id'] for row in samples] == plan['sample_mask']
    # Reject reopening an owner under a different phase name/selection mask.
    for case in samples:
        if dataset and case['dataset'] != dataset:
            continue
        for arm in arms:
            _, old = score.select_case(OUT, case['dataset'], arm, case['qid'])
            pending = old is None or (old['status'] == 'incomplete' and old.get('retryable_infrastructure', False))
            owner = OUT / f'rounds/r{round_id:03d}/methods/{case["dataset"]}/{arm}'
            if pending and owner.exists():
                raise RuntimeError('Existing fresh method owner; use the unused immutable recovery round')
    mask_path = OUT / f'DISPATCH_MASK_r{round_id:03d}_{dataset or "all"}.frozen.json'
    v1.freeze(mask_path, {'sample': 'fresh60', 'runtime_sha256': v1.sha(RUNTIME),
                          'full_sample_ids': plan['sample_mask'], 'dataset_filter': dataset,
                          'arms': list(arms), 'selection_sha256': v1.sha(CHOICE),
                          'output_root': str(OUT.relative_to(v1.ROOT))})
    with _adapt_runner(plan, arms):
        v1.log('fresh60_dispatch', sample='fresh60', round=round_id,
               sample_mask_sha256=v1.sha(mask_path), runtime_sha256=v1.sha(RUNTIME),
               selection_sha256=v1.sha(CHOICE), arms=list(arms), reference_values_read=False)
        return v1.run(plan, sample='fresh60', arms=list(arms), round_id=round_id, dataset=dataset)


def summarize():
    plan = prepare()
    chosen = select_candidate(plan)
    verify(plan)
    # Explicit all4 bindings: never default to original300 refs or old p02.
    return score.summarize(OUT, RUNTIME, arms=(BASELINE, chosen['selected_arm']),
                           references=REFERENCES, baseline_root=OUT,
                           output_path=OUT / 'SCORES_fresh60_LATEST.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'select', 'self-test', 'run', 'score'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--round', type=int, default=0, choices=(0, 1))
    parser.add_argument('--dataset', choices=('hotpotqa', '2wiki', 'musique'))
    args = parser.parse_args()
    if args.prepare:
        plan = prepare()
        value = {'prepared': True, 'planned': plan['planned'], 'selection_rule_frozen': True,
                 'candidate_selected': CHOICE.exists(), 'new_api_calls': 0, 'new_retrieval_calls': 0}
    elif args.select:
        value = select_candidate()
    elif args.self_test:
        value = self_test()
    elif args.score:
        value = summarize()
    else:
        value = run(round_id=args.round, dataset=args.dataset)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
