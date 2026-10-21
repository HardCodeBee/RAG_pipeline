"""Local plan and archive inventory only. No inference, move, delete or resume."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
Q=ROOT/'experiments/query_decomposition'
H=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()

def local_dependencies(seeds):
    """Conservative import closure; provenance pins supplement runtime imports."""
    todo=list(seeds);seen=set()
    while todo:
        p=todo.pop()
        if p in seen or not p.is_file():continue
        seen.add(p)
        if p.suffix!='.py':continue
        try:tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        except (SyntaxError,UnicodeError):continue
        module=rel(p).removesuffix('.py').split('/')
        if module[-1]=='__init__':module=module[:-1]
        else:module=module[:-1]
        for n in ast.walk(tree):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                base=module[:len(module)-n.level+1] if n.level else []
                base+=n.module.split('.') if n.module else []
                names=['.'.join(base),*['.'.join(base+[a.name]) for a in n.names]]
            for name in names:
                if not name.startswith(('experiments.','src.','scripts.')):continue
                path=ROOT.joinpath(*name.split('.'))
                for candidate in [path.with_suffix('.py'),path/'__init__.py']:
                    if candidate.is_file() and candidate not in seen:todo.append(candidate)
    return seen

def main():
    if (HERE/'PLAN.json').exists():
        raise RuntimeError('Current plan already exists; update it only through user-approved revisions, never regenerate it')
    now=datetime.now(timezone.utc).isoformat()
    oldp=read(Q/'p02_formal1500_v1_20261021/PROTOCOL.frozen.json')
    baseline=read(H/'comprehensive_v1_20261015/baseline_integrity_repair_v2_20261017/CONFIG_FROZEN.json')
    pins={ROOT/p for p in oldp['source_pins']}|{ROOT/p for p in baseline['source_and_input_sha256']}
    seeds=pins|{HERE/'p02_shared_layer.py',ROOT/'src/s02/core/pipeline.py'}
    closure=local_dependencies(seeds)
    fingerprints={rel(p):sha(p) for p in sorted(pins) if p.is_file()}
    if not (HERE/'PROTECTED_FILES_BEFORE.json').exists():write(HERE/'PROTECTED_FILES_BEFORE.json',fingerprints)
    for p in [ROOT/'README.md',ROOT/'experiments/README.md',ROOT/'configs/active_rag_candidates.json']:
        backup=HERE/'before'/p.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():backup.write_bytes(p.read_bytes())
    decisions=[
        ('U001','只关注完整RAG流程最终答案评价；P02为主方法，S02仅为另一条路线参考'),
        ('U002','基线采用完整修复v2，原始版只作历史参考'),
        ('U003','P02接入现有基线统一请求与长输入处理层'),
        ('U004','与主方法无关的组件消融或测试实验归档；有关材料不进入主结果表'),
        ('U005','同最终提示词控制和四项机制消融搁置'),
        ('U006','逐数据集记录每个原始query的时间、tokens和LLM开销，使用简明合理指标'),
        ('U007','多模型多检索器扩展在主评完成后，选择主方法及平均表现最佳前四种方法'),
        ('U008','后续约定和修正写入同一评估方案；任何新决策必须经用户允许'),
        ('U009','此前全部评估暂停继续有效；本轮方案整理和离线适配不构成恢复授权'),
    ]
    plan={
        'id':'unified_rag_eval_v1_20261021','revision':1,'created_utc':now,
        'status':'paused_pending_user_decisions','execution_approval':{'approved':False,'plan_revision':None},
        'primary_method':{'id':'p02_multigrain_timing_cachefix_v1','package':'src/p02_multigrain_timing_cachefix_v1','execution_variant':'p02_shared_baseline_v2','shared_layer':'existing baseline_integrity_repair_v2_20261017 with origin binding fix','integration_status':'offline_callbacks_implemented_not_execution_ready'},
        'reference_route':{'id':'s02_child4','package':'src/s02','role':'route_reference_only','historical_or_unified_rerun':'pending_user_decision'},
        'baselines':{'primary':'baseline_integrity_repair_v2_20261017','historical_reference':'campaign_api_v1','methods':list(baseline['native_selected_configs_preserved']['hotpotqa']),'iter_primary_config':'matched50 k10/T5','original_iter':'historical reference only'},
        'datasets':{'hotpotqa':{'formal':500,'dev':100},'2wiki':{'formal':500,'dev':100},'musique':{'formal':500,'dev':100}},
        'input_directory':'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen',
        'planned_core_method_query_results':15000,'reference_route_additional_runs':'not approved',
        'sample_policy':'Existing frozen questions; no replacement or result-based filtering; dev/formal/history kept separate',
        'generation':{'model':baseline['model'],'temperature':0,'enable_thinking':False,'new_request_max_tokens':baseline['effective_llm_max_output_tokens'],'max_input':baseline['max_model_input_tokens'],'context':baseline['context_window']},
        'shared_input_policy':'Existing complete BGE windows; existing complete reranker windows only for methods that already rerank; no new reranker in P02',
        'metrics':{'required_by_user':['native Answer F1','native Answer EM','scorable/planned','per-query time','LLM calls','input/output tokens','LLM expense'],
                   'proposed_summary':['mean query seconds','optional P50/P95','mean logical API calls','mean API input/output tokens','mean logical API cost CNY'],
                   'status':'specific definitions pending user approval','decomposition_receipts':'P02 remote_raw already contains elapsed_seconds/input_tokens/output_tokens; cannot omit upstream decomposition overhead or treat offline replay as zero'},
        'extensions':{'status':'deferred_until_primary_evaluation_complete_and_user_approval','groups':'P02 plus top4 by average performance','eligible_pool_proposed':'nine v2 baselines; S02 excluded as route reference unless user approves otherwise','ranking_proposed':'equal-weight mean of dataset-native F1 across the three datasets','ranking_formula_approved':False,'missingness_and_ties':'pending approval','models_retrievers':'previous five combinations retained as historical design; final matrix to be approved after main evaluation'},
        'deferred':['same_final_prompt_control','four_mechanism_ablations','new_evidence_judge_calls','remote_resource_extensions'],
        'archive':{'strategy':'pending_user_choice','inventory':'ARCHIVE_INVENTORY.json','physical_moves_authorized_for_this_run':False,'delete_originals':False},
        'budget':{'cumulative_cap_cny':300,'submission_ceiling_cny':299.9,'known_cny_at_pause':219.4958576,'unknown_reserved_cny_at_pause':54.4043504,'occupied_cny_at_pause':273.900208,'remaining_cny_at_pause':26.099792,'new_API_calls_authorized':False,'previous_p02_subcap_cny':20,'new_unified_variant_allocation':'requires explicit user approval'},
        'approved_decisions':[{'id':k,'status':'approved','authority':'latest user instruction in this chat','text':v} for k,v in decisions],
        'pending_decisions':[
            {'id':'P001','topic':'归档采用索引保留路径还是物理移动','status':'awaiting_user'},
            {'id':'P002','topic':'S02仅引用历史成绩还是接入统一层重跑','status':'awaiting_user'},
            {'id':'P003','topic':'指标汇总和三数据集F1等权平均排名原则','status':'awaiting_user'},
            {'id':'P004','topic':'计时边界、分解器纳入方式、缓存/冷运行、硬件并发与计时样本','status':'awaiting_user'},
            {'id':'P005','topic':'旧P02响应迁移、受长输入变化影响下游的重算范围及预算','status':'awaiting_user'},
            {'id':'P006','topic':'扩展排名候选池、缺失可比口径及并列处理','status':'awaiting_user'},
            {'id':'P007','topic':'最终新协议版本、预算分配与恢复执行','status':'awaiting_user'},
        ],
        'change_policy':'Every methodological, sample, metric, cache, budget, ranking, archive, or execution decision requires explicit user approval recorded in DECISIONS.jsonl; no timeout/default approval; no retroactive rewriting of frozen protocols',
    }
    if not (HERE/'PLAN.json').exists():write(HERE/'PLAN.json',plan)
    if not (HERE/'DECISIONS.jsonl').exists():
        (HERE/'DECISIONS.jsonl').write_text(''.join(json.dumps({'utc':now,**v},ensure_ascii=False)+'\n' for v in plan['approved_decisions']),encoding='utf8')
    (HERE/'STOP_REQUESTED').write_text('Explicit user pause active. No execution without a subsequent user-approved plan revision and resume instruction.\n',encoding='utf8')
    needed_dirs={p.relative_to(Q).parts[0] for p in closure if p.is_relative_to(Q) and len(p.relative_to(Q).parts)>1}
    operational={'main_eval_guard_20261021','resume_v2_primary_20261020','resume_v2_sqlite_20261021','resume_v2_trace_20261021','twowiki_missing_recovery_20261021','evaluation_catalog_20261021'}
    active={'baseline_integrity_repair_v2_20261017','comprehensive_eval_v1_20261015',HERE.name}
    deferred={'robustness_models_retrievers_v1_20261017','aux_representation_formal200_v1','reader_context_route_ablation_v1_20261019','reader_context_route_ablation50_v1_20261020'}
    related_prefixes=('p02_','multigrain_','b0_','rewrite_','child_answer_','decomposition_error_','component_chain_','qdmr_','retrieval_propagation_','terminal_goal_')
    inventory=[]
    for p in sorted(Q.iterdir()):
        if not p.is_dir() or p.name=='__pycache__':continue
        n=p.name
        if n in active:group='current_plan_or_primary_baseline'
        elif n in operational:group='preserved_operational_support_paused'
        elif n in deferred:group='deferred_not_current_evaluation'
        elif n in needed_dirs:group='protected_runtime_or_provenance_dependency'
        elif n.startswith(related_prefixes):group='related_history_not_in_primary_table'
        elif n in {'hotpot_formal_eval_v1','ircot_prompt_stage2_v1'}:group='reference_route_and_shared_history'
        else:group='unrelated_archive_candidate'
        outputs=[x for x in [H/n,ROOT/'outputs/query_decomposition'/n] if x.exists()]
        inventory.append({'path':rel(p),'classification':group,'archive_status':'pending_strategy_approval' if group=='unrelated_archive_candidate' else 'not_moved','physical_move':False,'referenced_by_current_import_or_pin':n in needed_dirs,'outputs':[rel(x) for x in outputs]})
    write(HERE/'ARCHIVE_INVENTORY.json',{'utc':now,'strategy':'pending_user_approval','scope':'query_decomposition family and related outputs; other research families not reassigned without user instruction','classification_is_reviewable_not_a_new_scientific_conclusion':True,'directories':inventory,'protected_sources':[rel(p) for p in sorted(closure)]})
    summary=Counter(x['classification'] for x in inventory)
    print(json.dumps({'plan':str(HERE/'PLAN.json'),'inventory_directories':len(inventory),'classification_counts':dict(summary),'protected_frozen_files':len(fingerprints),'new_API_calls':0,'physical_moves':0},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
