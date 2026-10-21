"""Offline final statistics; never called by generation or used for selection."""
import argparse
from collections import Counter
import json
import sqlite3
import time
import numpy as np
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import OUT,PARENT,INPUTS,METHODS,DATASETS,read,rows,save,utc,verify,canonical
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.answers import extract
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.compare_scores import paired,holm


def score_rows(path):
    return {x['qid']:x for x in rows(path)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        deadline=time.monotonic()+24*3600
        while time.monotonic()<deadline:
            try:p=read(OUT/'progress.json')
            except (ValueError,FileNotFoundError):time.sleep(30);continue
            if p.get('status')=='completed' and p.get('stage')=='all':break
            if (OUT/'STOP_REQUESTED').exists():return
            if p.get('status')=='stopped_for_audit' and not p.get('reason','').startswith('Transport stopped;'):
                save(OUT/'FINAL_ANALYSIS_STATUS.json',{'utc':utc(),'status':'waiting_required_repair_or_budget','progress':p,'new_api_calls':0});return
            time.sleep(30)
        else:return
    verify()
    result={'utc':utc(),'new_api_calls':0,'old_new':{},'common_all_methods':{},'matched25':{},
        'interpretation':'Frozen inspected samples; integrity repair including retained completed responses; no new parameter or round selection.'}
    cases=[]
    for dataset in DATASETS:
        selected=read(PARENT/dataset/'BASELINE_CONFIGS_FROZEN.json')['selected']
        current={};old_new={}
        for mid,cfg in selected.items():
            base=OUT/'main'/dataset/'formal'/mid/cfg['id']
            if not (base/'scores.jsonl').exists():continue
            current[mid]=score_rows(base/'scores.jsonl')
            oldpath=PARENT/dataset/'formal'/mid/cfg['id']/'scores.jsonl'
            if not oldpath.exists():continue
            old=score_rows(oldpath)
            old_new[mid]=paired(current[mid],old,seed=2026101706)
            for qid,new in current[mid].items():
                before=old[qid]
                cases.append({'dataset':dataset,'method':mid,'qid':qid,
                    'old_prediction':before['prediction'],'new_prediction':new['prediction'],
                    'old_f1':before['f1'],'new_f1':new['f1'],
                    'delta':new['f1']-before['f1'] if new['f1'] is not None and before['f1'] is not None else None})
        adjusted=holm({m:c['two_sided_group_sign_flip_p'] for m,c in old_new.items() if 'two_sided_group_sign_flip_p' in c})
        for mid,p in adjusted.items():old_new[mid]['holm_p_within_dataset_diagnostic_family']=p
        result['old_new'][dataset]=old_new
        if len(current)==9:
            common=set.intersection(*({q for q,r in data.items() if r['f1'] is not None} for data in current.values()))
            result['common_all_methods'][dataset]={'common':len(common),'planned':500,
                'all_methods_terminal500':all(all(r['status']!='unstarted' for r in data.values()) for data in current.values()),
                'f1':{mid:float(np.mean([data[q]['f1'] for q in common])) for mid,data in current.items()},
                'cohort_qids':sorted(common)}
    refs={r['qid']:r for r in rows(INPUTS/'2wiki/formal.references.local.jsonl')}
    scores=metric('2wiki')
    stores={}
    for mid,name in [('b05_iter_retgen_bge','k5_t5'),('b03_react_dense','k5_s5')]:
        path=OUT/'matched25/2wiki/formal'/mid/name/'checkpoint.sqlite3'
        if path.exists():
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                stores[mid]={(kind,key):json.loads(value) for kind,key,value in db.execute('select kind,key,value from data')}
    if len(stores)==2:
        curves={};fixed_common=set(refs)
        for budget in range(1,6):
            values={mid:{} for mid in stores}
            for qid,ref in refs.items():
                row=stores['b05_iter_retgen_bge'].get(('iteration',f'{qid}/{budget}'))
                pred=extract(row['result'].get('answer','')) if row and row['result']['status']=='success' else None
                values['b05_iter_retgen_bge'][qid]={'qid':qid,'type':ref['type'],'group_id':ref['group_id'],
                    'f1':scores(pred,ref)['f1'] if pred is not None else None}
                searches=0;answer=None
                for turn in range(1,7):
                    action=stores['b03_react_dense'].get(('action',f'{qid}/{turn}'))
                    if not action:break
                    parsed=action['parsed']
                    if parsed is None:answer='';break
                    op,arg,_=parsed
                    if op=='finish':answer=arg;break
                    if searches==budget:answer='';break
                    searches+=1
                values['b03_react_dense'][qid]={'qid':qid,'type':ref['type'],'group_id':ref['group_id'],
                    'f1':scores(answer,ref)['f1'] if answer is not None else None}
            common=set.intersection(*({q for q,r in v.items() if r['f1'] is not None} for v in values.values()))
            fixed_common &= common
            curves[str(budget)]={'maximum_document_slots':budget*5,'paired':paired(values['b05_iter_retgen_bge'],values['b03_react_dense'],seed=2026101706),
                'common_f1':{mid:float(np.mean([v[q]['f1'] for q in common])) for mid,v in values.items()},'scores':values}
        for curve in curves.values():
            curve['fixed_intersection_cohort_n']=len(fixed_common)
            curve['fixed_intersection_cohort_f1']={mid:float(np.mean([v[q]['f1'] for q in fixed_common])) for mid,v in curve['scores'].items()}
        result['matched25']={'main_budget_calls':5,'budget_curves':curves,
            'fixed_intersection_qids':sorted(fixed_common),'native_early_finish_retained':True,'round_not_selected_by_f1':True}
    save(OUT/'FINAL_STATISTICS.json',result)
    with (OUT/'OLD_NEW_CASES.jsonl').open('w',encoding='utf8') as f:
        for case in cases:f.write(canonical(case)+'\n')
    lines=['完整性修复后统计（仅使用已保存结果，无API调用）','相同样本、原参数；原结果保留；共同可评分分母和缺失上下界分别报告。','']
    for dataset,methods in result['old_new'].items():
        for mid,c in methods.items():
            if 'mean_difference_common' not in c:continue
            lo,hi=c['paired_95_ci_common']
            lines.append(f"{dataset} {mid}: 新-旧F1 {100*c['mean_difference_common']:+.2f}个百分点，同题{c['common_scored']}/500，95%CI[{100*lo:+.2f},{100*hi:+.2f}]；增/持/降={c['gain']}/{c['tie']}/{c['loss']}。")
    if result['matched25']:
        main=result['matched25']['budget_curves']['5']['paired']
        lines.append('matched25 Iter-ReAct: '+json.dumps(main,ensure_ascii=False))
    (OUT/'FINAL_STATISTICS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(OUT/'FINAL_ANALYSIS_STATUS.json',{'utc':utc(),'status':'completed','new_api_calls':0})
    print('Offline final statistics saved')


if __name__=='__main__':main()
