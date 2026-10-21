"""Read-only budget/fidelity audit; no scoring, index search, GPU or API work."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/comprehensive_v1_20261015'
DS = ('hotpotqa', '2wiki', 'musique')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    plan = read(HERE/'PLAN.json')
    assert not plan['execution_approval']['approved'] and (HERE/'STOP_REQUESTED').exists()
    out = HERE/'reviews/20261021_fairness_audit'
    out.mkdir(exist_ok=True)
    pins = {}
    def source(path):
        pins[path.relative_to(ROOT).as_posix()] = sha(path)
        return path
    native, budgets, p02 = {}, {}, {}
    matched = read(source(CAMPAIGN/'baseline_integrity_repair_v2_20261017/matched50/CONFIG_FROZEN.json'))
    for ds in DS:
        native[ds] = read(source(CAMPAIGN/f'campaign_api_v1/{ds}/BASELINE_CONFIGS_FROZEN.json'))['selected']
        config = native[ds]
        budgets[ds] = {
            'b00_rag_top10_v1': {'candidate_slots_max': 10},
            'b01_rag_top10_cot3': {'candidate_slots_max': 10},
            'b02_selfask_dense': {'candidate_slots_max': config['b02_selfask_dense']['k']*config['b02_selfask_dense']['max_search']},
            'b03_react_dense': {'candidate_slots_max': matched['react_config']['k']*matched['react_config']['max_search']},
            'b04_ircot_bge': {'candidate_slots_max': config['b04_ircot_bge']['k']*config['b04_ircot_bge']['max_reason_sentences'], 'retained_document_pool_max': config['b04_ircot_bge']['max_document_pool']},
            'b05_iter_retgen_bge': {'candidate_slots_max': matched['iter_config']['k']*matched['iter_config']['iterations'], 'active_override': matched['iter_config']},
            'b06_decomp_bge': {'candidate_slots_max': None, 'reason': 'No explicit shared original-query retrieval counter; outer and inner task loops each max_steps8. Task steps are not retrieval operations.', 'native_k': config['b06_decomp_bge']['k']},
            'b07_hybrid_rerank': {'candidate_slots_max': 2*config['b07_hybrid_rerank']['per_retriever_k'], 'two_retrieval_backends': True, 'final_k': 10},
            'b08_qd_rerank': {'candidate_slots_max': 10*(1+config['b08_qd_rerank']['max_queries']), 'original_query_included': True, 'final_k': 10}}
        runtime = ROOT/f'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/runtime/{ds}'
        counts = Counter()
        for path in sorted(runtime.glob('*.json')):
            counts[len(read(source(path))['supporting_questions'])] += 1
        assert sum(counts.values()) == 500
        p02[ds] = {'supporting_question_count_distribution': dict(sorted(counts.items())),
                   'potential_candidate_slots_formula': '10*(2*m+1), m=supporting questions, goal already included; changed rewrites only',
                   'potential_slots_max_in_frozen_cohort': max(10*(2*m+1) for m in counts),
                   'cases_with_potential_slots_above50': sum(n for m,n in counts.items() if 10*(2*m+1)>50),
                   'interpretation': 'Potential upper bounds from already frozen decomposition lengths, not observed fresh retrieval counts and not a quality-tuning signal'}
    exposure_path = source(ROOT/'experiments/query_decomposition/p02_formal1500_v1_20261021/EXPOSURE_AUDIT.frozen.json')
    exposure = read(exposure_path)
    train_path = source(ROOT/exposure['training_source'])
    assert sha(train_path) == exposure['training_sha256']
    train = {norm(r['query']) for r in rows(train_path)}
    overlap = {}
    for ds in DS:
        questions = rows(source(ROOT/f'{plan["input_directory"]}/{ds}/formal.questions.jsonl'))
        overlap[ds] = [r['qid'] for r in questions if norm(r['question']) in train]
    assert {ds:len(ids) for ds,ids in overlap.items()} == exposure['training_duplicates_by_dataset']
    prior = read(source(HERE/'reviews/20261021T051154Z/REUSE_AUDIT.json'))
    failures = {}
    for record in prior['baseline_cost_and_timing_inventory']:
        item = failures.setdefault(record['method'], {'recorded_predictions': 0, 'algorithm_failures': 0, 'reasons': Counter()})
        with sqlite3.connect((ROOT/record['source']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            for (raw,) in db.execute("select value from data where kind='prediction'"):
                prediction = json.loads(raw)
                item['recorded_predictions'] += 1
                if prediction.get('status') == 'algorithm_failure':
                    item['algorithm_failures'] += 1
                    item['reasons'][prediction.get('reason', 'unspecified')] += 1
    assert sum(r['algorithm_failures'] for r in failures.values()) == 341
    method_root = ROOT/'experiments/query_decomposition/baseline_integrity_repair_v2_20261017'
    for path in sorted((method_root/'methods').glob('*/method.py')):
        source(path)
    for path in [method_root/'prepare.py', method_root/'retrieval.py', method_root/'transport.py',
                 method_root/'reranking.py', HERE/'p02_shared_layer.py',
                 ROOT/'src/p02_multigrain_timing_cachefix_v1/core/pipeline.py']:
        source(path)
    for path,expected in read(HERE/'PROTECTED_FILES_BEFORE.json').items():
        assert sha(ROOT/path) == expected, path
    with sqlite3.connect((ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        charged = db.execute('select count(*) from charges').fetchone()[0]
    assert charged == 120094
    result = {'utc':datetime.now(timezone.utc).isoformat(), 'plan_revision_reviewed':plan['revision'],
              'status':'read_only_findings_proposals_not_adopted', 'baseline_native_selected_configs':native,
              'active_budget_matrix':budgets, 'p02_potential_budget_audit':p02,
              'training_question_text_overlap':overlap,
              'training_overlap_note':'Question-text overlap, not proof of answer leakage. Do not delete or replace the planned500 questions without user approval.',
              'baseline_failure_diagnostics':failures,
              'parameter_selection':'v2 preserves pre-repair dev100 configuration selection; v2 dev20 checks integrity only. matched50 T5 is inherited from ReAct allowance, not a new F1 selection.',
              'budget_definition':'Sum requested top-k per logical retrieval before dedup/merge, including cache replay; retained context size, input/output tokens and expense are separate quantities.',
              'source_sha256':pins, 'protected_frozen_sources_unchanged':86,
              'API_charge_rows_unchanged':charged, 'new_API_requests':0,'new_GPU_jobs':0,
              'new_index_searches':0,'quality_rescoring':False,'protocol_changes_adopted':False}
    (out/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out),'matrix':budgets,'P02_potential':p02,'training_overlap_counts':{ds:len(ids) for ds,ids in overlap.items()},'algorithm_failures':341,'new_API_requests':0,'protocol_changes_adopted':False},ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
