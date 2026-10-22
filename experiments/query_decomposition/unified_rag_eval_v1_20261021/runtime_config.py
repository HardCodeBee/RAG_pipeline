"""Frozen six-method binding and fail-closed execution gate; no model imports."""
import json
from pathlib import Path

from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common

ROOT = common.ROOT
HERE = Path(__file__).resolve().parent
P02 = 'p02_multigrain_timing_cachefix_v1'
METHODS = (P02, 'b00_rag_top10_v1', 'b01_rag_top10_cot3', 'b08_qd_rerank',
           'b02_selfask_dense', 'b06_decomp_bge')
DATASETS = ('hotpotqa', '2wiki', 'musique')
OUTPUT = common.OUTPUT / 'unified_rag_revision009_20261023'
REVIEW = HERE / 'reviews/20261023_narrative_revision007'
R02_HERE = HERE.parent / 'p02_formal1500_v1_20261021'
R02_OUTPUT = common.OUTPUT / 'p02_formal1500_v1_20261021'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def scope(plan):
    """Execution authorization may change; scientific settings cannot drift."""
    return {key: plan[key] for key in ('main_table', 'input_directory', 'generation',
                                     'sample_policy', 'statistics', 'qa_raw')}


def method_configs(manifest=None):
    manifest = manifest or read(REVIEW / 'METHODS.json')
    if tuple(row['method'] for row in manifest['methods']) != METHODS:
        raise ValueError('Expected exactly the frozen six methods, in frozen order')
    result = {}
    for row in manifest['methods']:
        config = {k: v for k, v in row['config'].items() if k != 'source'}
        if row['method'] == 'b08_qd_rerank':
            config['max_queries'] = config.pop('max_generated_subqueries')
            if config['k'] != 10 or config['final_k'] != 10 or not config['retain_original']:
                raise ValueError('QD fixed implementation/configuration mismatch')
        if row['method'] == 'b06_decomp_bge':
            if not config['max_steps'] == config['outer_max_steps'] == config['inner_max_steps'] == 8:
                raise ValueError('Both frozen DecomP controllers must use eight steps')
        result[row['method']] = config
    return result


def load_questions(plan=None):
    plan = plan or read(HERE / 'PLAN.json')
    result = {}
    for dataset in DATASETS:
        cases = rows(ROOT / plan['input_directory'] / dataset / 'formal.questions.jsonl')
        if len(cases) != 500 or len({c['qid'] for c in cases}) != 500:
            raise ValueError('Formal sample is not the unchanged unique500: ' + dataset)
        if any(set(c) != {'qid', 'question'} for c in cases):
            raise ValueError('Only original qid/question may enter inference')
        result[dataset] = cases
    return result


def resource(dataset):
    return read(common.PARENT / dataset / 'RESOURCE_FROZEN.json')['resources']


def load_r02_case(dataset, case, *, with_provenance=False):
    """Read and verify an existing query-only R02 receipt; never regenerate it."""
    from src.p02_multigrain_timing_cachefix_v1.core.pipeline import safe_case
    questions = rows(R02_HERE / 'questions1500.frozen.jsonl')
    matches = [(i, q) for i, q in enumerate(questions, 1)
               if q['dataset'] == dataset and q['qid'] == case['qid']]
    if len(matches) != 1 or matches[0][1]['question'] != case['question']:
        raise ValueError('R02 original question binding mismatch')
    index, q = matches[0]
    identity = read(R02_OUTPUT / 'remote_raw/IDENTITY.json')
    expected = {'decode_compat': 'explicit_generation_config_v002',
                'effective_do_sample': False, 'effective_num_beams': 1,
                'checkpoint_manifest_sha256': read(R02_HERE / 'REMOTE_INPUTS.frozen.json')['checkpoint_manifest_sha256'],
                'query_sha256': common.sha(R02_HERE / 'queries1500.id_query.frozen.jsonl'),
                'prompt_sha256': common.sha(R02_HERE / 'v12.txt'), 'do_sample': False,
                'num_beams': 1, 'max_new_tokens': 1024, 'seed': 1729, 'api_calls': 0}
    if identity != expected:
        raise ValueError('R02 identity mismatch')
    raw = read(R02_OUTPUT / f'remote_raw/{index:03d}.json')
    if (raw['id'] != q['id'] or raw['dataset'] != dataset or raw['index'] != index
            or raw['query'] != q['question'] or raw['identity_sha256'] != common.digest(identity)
            or raw['actual_prompt'] != (R02_HERE / 'v12.txt').read_text(encoding='utf8')):
        raise ValueError('R02 raw receipt not bound to this question/model/prompt')
    try:
        support = json.loads(raw['raw_output'])
        valid = (not raw.get('generation_error') and not raw.get('truncated')
                 and isinstance(support, list) and all(isinstance(s, str) and s.strip() for s in support))
    except (ValueError, TypeError):
        valid = False
    # Same pre-existing mechanical fallback; never a new semantic repair.
    if not valid:
        support = []
    result = safe_case({**q, 'supporting_questions': support})
    if result != read(R02_OUTPUT / f'runtime/{dataset}/{case["qid"]}.json'):
        raise ValueError('Accepted R02 case differs from raw receipt')
    if with_provenance:
        return result, {'model': 'Qwen/Qwen3-4B-Instruct-2507 plus r02_expanded LoRA, v12',
                        'input_tokens': raw.get('input_tokens'), 'output_tokens': raw.get('output_tokens'),
                        'source': (R02_OUTPUT / f'remote_raw/{index:03d}.json').relative_to(ROOT).as_posix(),
                        'source_sha256': common.sha(R02_OUTPUT / f'remote_raw/{index:03d}.json'),
                        'new_inference': False, 'gpu_CNY': None,
                        'logical_token_source': 'frozen query-only R02 generation receipt'}
    return result


class ExecutionHeld(RuntimeError):
    pass


class ExecutionGate:
    def __init__(self, plan_path=HERE / 'PLAN.json', binding_path=HERE / 'RUNTIME_BINDING.json'):
        self.plan_path, self.binding_path = Path(plan_path), Path(binding_path)
        self.binding = read(self.binding_path)
        plan = read(self.plan_path)
        if common.digest(scope(plan)) != self.binding['scientific_scope_digest']:
            raise ExecutionHeld('Scientific configuration differs from runtime binding')
        if tuple(plan['main_table']['methods']) != METHODS or plan['main_table']['method_query_results'] != 9000:
            raise ExecutionHeld('Runtime is bound exclusively to the frozen six-method matrix')
        for name, sha in self.binding['source_input_pins'].items():
            if not (ROOT / name).is_file() or common.sha(ROOT / name) != sha:
                raise ExecutionHeld('Frozen source/input mismatch: ' + name)

    def __call__(self):
        plan = read(self.plan_path)
        approval = plan.get('execution_approval', {})
        if not approval.get('approved') or approval.get('plan_revision') != plan['revision']:
            raise ExecutionHeld('Formal execution remains unapproved')
        if common.digest(scope(plan)) != self.binding['scientific_scope_digest']:
            raise ExecutionHeld('Scientific scope changed after startup')
        holds = (HERE / 'STOP_REQUESTED', common.OUTPUT / 'STOP_REQUESTED',
                 common.OUT / 'STOP_REQUESTED', R02_OUTPUT / 'STOP_REQUESTED', OUTPUT / 'STOP_REQUESTED')
        if any(p.exists() for p in holds):
            raise ExecutionHeld('Explicit user pause remains; no automatic removal')
        return True


def inspect():
    """Offline configuration/sample inspection; does not instantiate runtime resources."""
    plan = read(HERE / 'PLAN.json')
    ExecutionGate()
    configs, questions = method_configs(), load_questions(plan)
    return {'plan_revision': plan['revision'], 'experimental_design_revision': 7,
            'methods': list(configs), 'counts': {d: len(v) for d, v in questions.items()},
            'main_outcomes': 9000, 'qa_raw_pairs': 300, 'qa_raw_finals': 600,
            'output': OUTPUT.relative_to(ROOT).as_posix(),
            'execution_approved': plan['execution_approval']['approved'],
            'inspection_only': True}
