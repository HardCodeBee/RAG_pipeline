"""Create an additive candidate snapshot. Never modify prior experiments.

Only small identity, prompt, and reference records are written. Existing case
trajectories, request databases, corpora, and model weights stay at their paths.
No inference or network call is available from this script.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E = ROOT / 'experiments/query_decomposition'
O = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_write(name, value):
    path = HERE / name
    if path.exists():
        assert read(path) == value, ('Do not overwrite frozen record', name)
    else:
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def freeze():
    from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline as flow
    from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import run as executor
    from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy

    # Verify original pins before using any source; this is read-only.
    protocol = read(executor.HERE / 'PROTOCOL.frozen.json')
    executor.verify(protocol)
    decomp = read(HERE / 'DECOMPOSER_AUDIT.json')
    archive = read(HERE / 'SIMPLIFICATION_ARCHIVE.json')
    prior = read(E / 'p02_simplification_review_20261020/ADOPTION.user_20261020.frozen.json')
    cache_decision = read(executor.HERE / 'CACHE_EXECUTOR_SELECTED.user_20261021.frozen.json')
    verified = read(O / 'p02_merge_cachefix_v1_20261021/reports/FINAL_VERIFICATION.json')
    assert prior['selected_arm'] == cache_decision['adopted_method'] == 'timing_only'
    assert verified['passed'] and verified['complete_case_counts']['timing_only'] == 300
    assert decomp['checked_raw_records'] == 300 and decomp['all300_raw_id_query_array_prompt_and_sha_exact']

    history = [{'step_id':1,'request':'<PREVIOUS_QUESTION>','answer':'<PREVIOUS_COMPLETE_FACTUAL_SENTENCE>'}]
    examples_path = E / 'hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json'
    examples = read(examples_path)
    raw_goal = flow.b0.legacy_child.child('<ORIGINAL_QUERY>',2,'<ORIGINAL_QUERY>',history,
        '<CURRENT_MERGED_TOP10_PASSAGES>',examples['child'],role='goal',combine=False)
    nodes = [dict(history[0], role='ordinary'),
             {'step_id':2,'request':'<ORIGINAL_QUERY>','answer':'<GOAL_COMPLETE_FACTUAL_SENTENCE>','role':'goal'}]
    packet = {'docs':[{'block':'<ORIGINAL_QUERY_ORIGINAL_ROUTE_TOP10_PASSAGES>'}]}
    final_prompt, _, helper, _ = flow.render_final('<ORIGINAL_QUERY>',nodes,packet,protocol['readout'])
    prompt_snapshot = {
        'scope':'Exact renderer outputs with explicit placeholders; step2 and one prior QA show the variable history shape. Actual runs can contain more prior QA. All worked examples are retained verbatim.',
        'decomposition_prompt_audit':relative(HERE / 'DECOMPOSER_AUDIT.json'),
        'decomposition_v12_prompt':(ROOT / decomp['prompt']['path']).read_text(encoding='utf8'),
        'child_instructions':flow.b0.legacy_child.CHILD_INSTRUCTIONS,
        'ordinary_child_rendered_shape_step2':flow.b0.legacy_child.child(
            '<ORIGINAL_QUERY>',2,'<CURRENT_SUBQUESTION>',history,'<CURRENT_MERGED_TOP10_PASSAGES>',
            examples['child'],role='ordinary',combine=False),
        'goal_rendered_shape_step2':strategy.goal_prompt(raw_goal,'p02_role_headers'),
        'rewrite_rendered_shape':flow.b0.legacy_rewrite.rewrite('<CURRENT_QUESTION>',history),
        'helper_rendered_shape':helper,
        'final_rendered_shape':final_prompt,
        'child_examples':examples['child'],
        'final_examples':read(E / 'multigrain_prompt_v3_20261017/final_examples.json'),
        'goal_rule':flow.b0.legacy_child.GOAL_RULE,
        'final_instructions':flow.b0.final_prompts.FINAL_INSTRUCTIONS,
        'P1_readout':protocol['readout'],
        'legacy_sentence_format_contract':'The exact conditional formatter remains in the pinned flow pipeline; initial answers with fewer than four whitespace words trigger it. It was not triggered by the recorded 300 cases.',
    }
    frozen_write('PROMPTS.frozen.json',prompt_snapshot)
    frozen_write('RETRIEVAL_RESOURCES.frozen.json',{
        'active_retriever':'bge',
        'query_model':'BAAI/bge-small-en-v1.5',
        'route_top_k':10,
        'resources':protocol['resources'],
        'unused_resource_entries':'bm25/contriever/dpr entries are inherited resource identities, not selected retrieval routes.',
    })

    # Direct execution/rendering and execution-policy dependencies, plus the
    # already frozen source closure. This also pins previously omitted examples.
    names = set(protocol['source_pins'])
    names.update([
        'p02_simplification_review_20261020/selected_pipeline.py',
        'p02_simplification_review_20261020/ADOPTION.user_20261020.frozen.json',
        'p02_merge_cachefix_v1_20261021/PROTOCOL.frozen.json',
        'p02_merge_cachefix_v1_20261021/HISTORICAL_SUCCESS_INDEX.frozen.json',
        'p02_merge_cachefix_v1_20261021/CACHE_EXECUTOR_SELECTED.user_20261021.frozen.json',
        'p02_merge_cachefix_v1_20261021/offline_audit.py',
        'p02_flow_simplify_v1_20261020/PROTOCOL.frozen.json',
        'b0_strategy_round2_v1_20261020/fresh_runner.py',
        'hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json',
        'multigrain_prompt_v2_20261017/prompts.py',
        'component_chain_opt_v2_20261019/run.py',
        'component_chain_opt_v2_20261019/policy.py',
        'hotpot_formal_eval_v1/common/runtime.py',
        'hotpot_formal_eval_v1/parallel_api.py',
        'b0_strategy_round4_v1_20261020/PROTOCOL.frozen.json',
        'b0_strategy_round4_v1_20261020/strategy.py',
        'b0_strategy_round4_v1_20261020/SOURCES.frozen.jsonl',
    ])
    paths = []
    for name in names:
        p = ROOT / name
        if not p.exists():
            p = E / name
        assert p.is_file(), str(p)
        paths.append(p)
    for name in ['runtime300.frozen.jsonl','BASELINE_SOURCES.frozen.json']:
        paths.append(executor.HERE / name)
    for binding in decomp['minimal_freeze_pins']:
        p = ROOT / binding['path']
        assert sha(p) == binding['sha256'], ('Decomposer provenance changed', str(p))
        paths.append(p)
    for name in ['pipeline.py','freeze.py','verify.py','FLOW.txt','PROMPTS.frozen.json',
                 'RETRIEVAL_RESOURCES.frozen.json','DECOMPOSER_AUDIT.json',
                 'SIMPLIFICATION_ARCHIVE.json','SIMPLIFICATION_ARCHIVE.txt']:
        paths.append(HERE / name)
    evidence = [
        O / 'p02_merge_cachefix_v1_20261021/RESULTS300.frozen.json',
        O / 'p02_merge_cachefix_v1_20261021/reports/FINAL_VERIFICATION.json',
        O / 'p02_merge_cachefix_v1_20261021/reports/PAIRED_DETAILS.jsonl',
        O / 'b0_strategy_round4_v1_20261020/FINAL_RESULTS.frozen.json',
        O / 'b0_strategy_round4_v1_20261020/SELECTION.frozen.json',
    ]
    paths.extend(evidence)
    existing = read(HERE / 'CANDIDATE.frozen.json') if (HERE / 'CANDIDATE.frozen.json').exists() else {}
    record = {
        'schema_version':1,
        'candidate_id':'p02_multigrain_timing_cachefix_v1',
        'status':'frozen_candidate',
        'created_utc':existing.get('created_utc',datetime.now(timezone.utc).isoformat()),
        'client_date':'2026-10-21',
        'authorization':'User requested preserving the simplification series and confirming/freezing the current multigrain method as another candidate alongside s04.',
        'candidate_slot':2,
        'candidate_slots_are_not_quality_rankings':True,
        'coexisting_candidate':{
            'user_alias':'s04','resolved_id':'s04_scope_plus_evidence',
            'protocol':relative(E / 'b0_strategy_round4_v1_20261020/PROTOCOL.frozen.json'),
            'historical_selection':relative(O / 'b0_strategy_round4_v1_20261020/SELECTION.frozen.json'),
            'identity_note':'The recent local S04 complete-flow candidate; not the old s04_balanced_demos variant. Original experiment and selection records are retained unchanged.',
        },
        'entrypoint':'experiments.query_decomposition.multigrain_candidate_p02_v1_20261021.pipeline.execute',
        'execution_runner':cache_decision['execution_runner'],
        'selected_runner_arguments':{'sample':'300','arms':['timing_only']},
        'method':{
            'parent':'p02_role_headers','arm':'timing_only',
            'decomposer':'R02 expanded3632 full retrain, v12; exact provenance in DECOMPOSER_AUDIT.json',
            'remote_checkpoint':decomp['remote_checkpoint_directory'],
            'remote_weight_bytes_reverified_this_freeze':False,
            'decomposition_interface':'Ordered self-contained supporting_questions; no cross-question #N references; append original query verbatim as final goal.',
            'root_retrieval':'No initial original-query retrieval; get it in goal original route, reuse the same packet for final.',
            'rewrite':'legacy multigrain_prompt_v2, all earlier complete QA; rewrite only when history exists',
            'reader_merge':{'routes':'current original Top10 + rewritten Top10 when query differs',
                            'order':'rank interleave, original first','deduplication':'doc_id','keep':10},
            'child_reader_inputs':['original query scope','current question','all prior complete QA','current merged Top10'],
            'child_prompt':'legacy S02 child instructions, ordinary4shot/goal0shot; full sentence, no C08 selection',
            'short_answer_formatter':'Retained: <4 whitespace words triggers legacy complete-sentence call',
            'history':'Full factual-sentence QA, no mechanical #N binding, no sentinel-only deletion/filtering added',
            'goal_preanswer':True,'goal_routing':'Every goal retrieved and answered; no retrieve/combine/skip routing',
            'goal_rule':'Existing g01 consistency instruction',
            'auxiliary':'p02 role headers; ordinary QA followed by goal candidate conclusion',
            'final_overall_answer':True,
            'final_reader_inputs':['original query','all ordinary+goal QA with role headers','goal original-route original-query Top10'],
            'final_context_order':'helper before original-query raw Top10',
            'final_prompt':'multigrain v3 instructions, four final examples, P1 readout',
            'excluded_candidates':['union20','discount20','single_final','terminal_goal','C08+c02 integration','S04 scope/evidence edits'],
        },
        'generation':{'model':protocol['model'],'temperature':0,'enable_thinking':False,
                      'max_tokens':{'rewrite':256,'child_or_goal':128,'sentence_format':128,'final':64}},
        'cache':{'provider':cache_decision['cache_provider'],
                 'key':'full API payload hash AND complete payload equality',
                 'priority':cache_decision['cache_priority'],
                 'success_reuse':'Never repay an identical successful input; per-payload concurrency lock',
                 'retrieval_key':'qid, exact query, k, full frozen resource identity',
                 'unknown_requests':'Retain liability; do not silently replay unknown requests or erase reserves'},
        'evaluation':{'scope':'Previously exposed development300, 100 per dataset; not a new independent test',
                      'runtime':relative(executor.HERE / 'runtime300.frozen.jsonl'),
                      'complete_cases':300,
                      'native_final_Answer_F1':verified['independent_native_scores']['timing_only'],
                      'logical_generation_usage':verified['independent_cold_logical_usage']['timing_only'],
                      'cost_scope':'Historical model/token-based cold estimate; no claim this is newly spent or total historical spend. Decomposition/retrieval compute is not priced in this API estimate.',
                      'cache_fixed_control':relative(O / 'p02_merge_cachefix_v1_20261021/RESULTS300.frozen.json')},
        'preservation':{'index':relative(HERE / 'SIMPLIFICATION_ARCHIVE.json'),
                        'policy':'Retain all original protocols, prompts, cases, request SQLite databases, failed/unknown attempts, scoring and costs in place. No deletion, renaming or replacement.',
                        'ablation_limit':'single_final and terminal_goal are bundled variants; old union results keep their historical cache caveat; repaired union20 is the controlled retention comparison.'},
        'file_pins':{relative(p):sha(p) for p in sorted(set(paths))},
        'new_API_calls':0,'new_index_searches':0,'new_remote_calls':0,'new_training_jobs':0,
        'global_default_switched':False,
    }
    frozen_write('CANDIDATE.frozen.json',record)
    print(json.dumps({'candidate_id':record['candidate_id'],'status':record['status'],
                      'pinned_files':len(record['file_pins']),'new_API_calls':0},ensure_ascii=False))


if __name__ == '__main__':
    freeze()
