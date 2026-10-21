"""Freeze the isolated correction without modifying the original campaign."""
import difflib
from .common import *
from src.generators.api_options import get_api_option


def main():
    if (OUT/'CONFIG_FROZEN.json').exists():
        return verify()
    stopped=read(OUT/'STOP_COMPLETED.json')
    if stopped['in_flight_requests']!=0:
        raise ValueError('Prior workers must be stopped and drained first')
    options=get_api_option(2)
    if options['model']!=MODEL:
        raise ValueError('Frozen model changed')
    paths=list(HERE.rglob('*.py'))+list((HERE/'methods').rglob('*.json'))
    configs={}
    for dataset in DATASETS:
        configs[dataset]=read(PARENT/dataset/'BASELINE_CONFIGS_FROZEN.json')['selected']
        paths += [PARENT/dataset/'BASELINE_CONFIGS_FROZEN.json', PARENT/dataset/'RESOURCE_FROZEN.json']
        paths += [INPUTS/dataset/(split+'.'+suffix) for split in ['dev','formal'] for suffix in ['questions.jsonl','references.local.jsonl']]
    paths += [ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015'/name for name in
              ['retrieval.py','offset_compatibility.py','exact_parallel_scan_v2.py','scoring_safe_trace.py']]
    value={'id':'baseline_integrity_repair_v2_20261017','created_utc':utc(),
        'model':MODEL,'base_url':options['base_url'],'global_cap_cny':200,'submission_guard_cny':PROTECTED_CAP,
        'effective_llm_max_output_tokens':MAX_OUTPUT,'max_model_input_tokens':MAX_INPUT,'context_window':CONTEXT,
        'official_model_limits_and_prices':'https://help.aliyun.com/zh/model-studio/qwen-plus',
        'preflight':'UTF8 byte upper bound plus256 framing; reject instead of truncate if full output allowance does not fit',
        'api_slots':8,'start_interval_seconds':2/3,'ready_cases':60,
        'datasets':DATASETS,'formal_questions_per_dataset':500,'dev_integrity_questions_per_method':20,
        'native_selected_configs_preserved':configs,
        'matched25':{'dataset':'2wiki','k':5,'max_search':5,'iterations':5,'methods':['b05_iter_retgen_bge','b03_react_dense']},
        'changes':['new API calls allow32768 output tokens and verified full context instead of old64/512/4096 ceilings',
            'unparsed multiline reasoning is a scored format failure, never a short-answer fallback',
            'any new incomplete output, missing usage or input overflow blocks normal scoring and pauses the run',
            'BGE overflow queries use all510-content-token windows; weighted-mean normalized vector; one native k search',
            'cross-encoder overflow pairs use all full-coverage windows; max logit per document; short pairs unchanged',
            'provider prompt/completion usage and local encoder/reranker counts recorded for each query and stage'],
        'retained':['exact prompt strings and demonstrations','generator identity/temperature/disable-thinking',
            'corpus,index,embedding and reranker weights','all native selected k/search/iteration/pool limits',
            'original method controllers except invalid-answer extraction','original samples/references/native scorers'],
        'completed_response_migration':'same method,qid,stage,model and complete message payload; only max_tokens may differ. Successful source finish_reason must be stop with usage; original request payload/usage retained. Not a fresh full rerun.',
        'failure_migration':'historical unknown/rejected/failed requests stay missing; no blind replay',
        'not_reused':['length-capped source responses','source overflow/truncated retrieval packets','old reranking if any candidate pair exceeds512'],
        'selection':'no new parameter/F1 selection; dev20 checks technical integrity only; same inspected formal samples remain diagnostic evidence',
        'source_and_input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},
        'prior_frozen_checks':audit_prior()}
    once(OUT/'CONFIG_FROZEN.json',value)
    diffs=[]
    for mid,name in METHODS.items():
        parent=(ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/methods' if name=='qd_rerank' else ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/methods')/name/'method.py'
        private=HERE/'methods'/name/'method.py'
        diffs.extend(difflib.unified_diff(parent.read_text(encoding='utf8').splitlines(True),private.read_text(encoding='utf8').splitlines(True),fromfile=str(parent.relative_to(ROOT)),tofile=str(private.relative_to(ROOT))))
    (OUT/'METHOD_CHANGES.diff').write_text(''.join(diffs),encoding='utf8')
    journal('token_integrity_repair_frozen',new_api_calls=0,source_receipt_sha256=sha(OUT/'CONFIG_FROZEN.json'),originals_preserved=True)
    return value


if __name__=='__main__':
    main()
