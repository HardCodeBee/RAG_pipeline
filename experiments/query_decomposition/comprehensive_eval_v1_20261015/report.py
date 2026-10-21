"""Complete-denominator tables and predeclared paired comparisons, offline only."""
from collections import defaultdict
import json
from .common import HERE,RUN,read,rows,save,utc
from .campaign import TARGET,DATASETS,OUR,CONTROL,ABLATIONS
from experiments.query_decomposition.hotpot_formal_eval_v1.compare_scores import paired,holm

def main():
    result={'utc':utc(),'datasets':{},'all_requested_evaluations_complete':True,
      'scope':'shared-resource method adaptation; native answer metrics; not a paper-specific replication or SOTA claim',
      'f1_is_hallucination_rate':False,'missing_answers_imputed':False}
    lines=['三数据集 API 评估进度及结果','生成模型：qwen-plus-2025-12-01；总预算上限200元','']
    macro=defaultdict(list)
    for dataset in DATASETS:
        selected_file=TARGET/dataset/'BASELINE_CONFIGS_FROZEN.json'
        if not selected_file.exists():
            result['datasets'][dataset]={'status':'awaiting_resources_or_dev_selection'};result['all_requested_evaluations_complete']=False
            lines.append(dataset+'：尚未完成开发参数选择。');continue
        selected=read(selected_file)['selected'];methods={};scored={};dirs={}
        configs={**selected,OUR:{'id':'frozen_s02_child4'},CONTROL:{'id':'matched_final_reader'}}
        lines.extend(['',dataset+'：方法 | 已完成/计划 | 可评分 | F1 | EM'])
        for mid,cfg in configs.items():
            directory=TARGET/dataset/'formal'/mid/cfg['id'];dirs[mid]=directory
            file=directory/'scores.json'
            if not file.exists():
                methods[mid]={'status':'not_started','planned':500};result['all_requested_evaluations_complete']=False
                lines.append(mid+' | 0/500 | 0 | — | —');continue
            s=read(file);methods[mid]=s;stats=s['overall'];scored[mid]={r['qid']:r for r in rows(directory/'scores.jsonl')}
            done=stats['terminal_records']==500
            if not done:result['all_requested_evaluations_complete']=False
            f1='—' if stats['f1_available'] is None else f"{100*stats['f1_available']:.2f}%"
            em='—' if stats['em_available'] is None else f"{100*stats['em_available']:.2f}%"
            lines.append(f"{mid} | {stats['terminal_records']}/500 | {stats['available']} | {f1} | {em}")
            if done and stats['f1_available'] is not None:macro[mid].append(stats['f1_available'])
        comparisons={};pvalues={}
        if OUR in scored and methods[OUR]['overall']['terminal_records']==500:
            for control in ['b00_rag_top10_v1','b03_react_dense','b08_qd_rerank',CONTROL]:
                if control not in scored or methods[control]['overall']['terminal_records']!=500:continue
                comparison=paired(scored[OUR],scored[control],seed=2026101501)
                comparisons[control]=comparison
                if 'two_sided_group_sign_flip_p' in comparison:pvalues[control]=comparison['two_sided_group_sign_flip_p']
        adjusted=holm(pvalues)
        for name,p in adjusted.items():comparisons[name]['holm_p_four_predeclared_primary_contrasts']=p
        if len(comparisons)!=4:result['all_requested_evaluations_complete']=False
        ablations={};subset={r['qid'] for r in rows(HERE/'inputs_frozen'/dataset/'ablation.questions.jsonl')}
        if OUR in scored and methods[OUR]['overall']['terminal_records']==500:
            full={q:scored[OUR][q] for q in subset}
            if CONTROL in scored and methods[CONTROL]['overall']['terminal_records']==500:
                ablations['a01_same_reader_no_helper']=paired(full,{q:scored[CONTROL][q] for q in subset},seed=2026101501)
            for mid in ABLATIONS:
                directory=TARGET/dataset/'ablation'/mid/'fixed_packets';file=directory/'scores.json'
                if not file.exists() or read(file)['overall']['terminal_records']!=200:continue
                values={r['qid']:r for r in rows(directory/'scores.jsonl')}
                ablations[mid]={'scores':read(file),**paired(full,values,seed=2026101501)}
        if len(ablations)!=4:result['all_requested_evaluations_complete']=False
        ap={k:v['two_sided_group_sign_flip_p'] for k,v in ablations.items() if 'two_sided_group_sign_flip_p' in v}
        for key,p in holm(ap).items():ablations[key]['holm_p_four_ablations']=p
        result['datasets'][dataset]={'main_methods':methods,'primary_comparisons':comparisons,'core_ablations':ablations}
        for control,s in comparisons.items():
            ci=s.get('paired_95_ci_common');delta=s.get('mean_difference_common')
            if ci is not None:lines.append(f"s02 vs {control}：同题{ s['common_scored'] }/500，F1差{100*delta:+.2f}个百分点，95%CI[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]；增/持/降={s['gain']}/{s['tie']}/{s['loss']}。")
        for arm,s in ablations.items():
            if s.get('mean_difference_common') is not None:lines.append(f"完整方法 vs {arm}：同题{s['common_scored']}/200，F1差{100*s['mean_difference_common']:+.2f}个百分点。")
    result['three_dataset_equal_weight_macro_f1']={k:sum(v)/3 for k,v in macro.items() if len(v)==3}
    legacy=TARGET/'legacy200_diagnostics/RESULTS.json'
    if legacy.exists():result['legacy200_post_hoc_diagnostics']=read(legacy)
    else:result['all_requested_evaluations_complete']=False
    lines.extend(['','未完成的方法不进行完整排名。API失败保留在计划分母中，报告可评分均值及全部计划题的上下界。',
      '主表与消融均不用于改动冻结的s02_child4。答案F1不代表幻觉率，自动证据审查尚未执行。'])
    save(TARGET/'RESULTS.json',result)
    (TARGET/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result

if __name__=='__main__':main()
