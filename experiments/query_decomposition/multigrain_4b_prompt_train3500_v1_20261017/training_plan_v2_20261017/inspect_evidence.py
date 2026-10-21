"""Read-only historical training and current data checks for the training proposal."""
from collections import Counter
import hashlib,json,re,statistics,unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'outputs/query_decomposition/multigrain_4b_prompt_train3500_v1_20261017/training'
OLD=ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer'
OUT=ROOT/'outputs/query_decomposition/multigrain_4b_training_plan_v2_20261017'
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' ?!.？！')
def write(n,r):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    master=rows(DATA/'MASTER_3500.jsonl');sft=rows(DATA/'train3500.messages.jsonl')
    assert len(master)==len(sft)==3500
    prompt=(ROOT/'experiments/query_decomposition/multigrain_4b_prompt_train3500_v1_20261017/FINAL_PROMPT.txt').read_text(encoding='utf-8').strip()
    errors=[]
    for r,s in zip(master,sft):
        if s['messages']!=[{'role':'system','content':prompt},{'role':'user','content':r['query']},{'role':'assistant','content':json.dumps(r['target'],ensure_ascii=False,separators=(',',':'))}]:errors.append([r['index'],'message_mismatch'])
        if not isinstance(r['target'],list) or any(not isinstance(q,str) or not q.strip() or not q.endswith('?') for q in r['target']):errors.append([r['index'],'invalid_target'])
        if any(re.search(r'#\d+|UNRESOLVED|the result of \(',q) for q in r['target']):errors.append([r['index'],'unresolved_placeholder'])
        if len(r['target'])!=len(set(map(norm,r['target']))):errors.append([r['index'],'duplicate_subquestion'])
    evalpath=ROOT/'experiments/query_decomposition/multigrain_4b_frozen_eval300_v1_20261017/questions300.frozen.jsonl'
    promptpath=ROOT/'experiments/query_decomposition/multigrain_4b_prompt_train3500_v1_20261017/questions100.frozen.jsonl'
    traintexts=set(norm(r['query']) for r in master)
    overlaps={name:sum(norm(r['query']) in traintexts for r in rows(path)) for name,path in [('eval300',evalpath),('prompt100',promptpath)]}
    assert len(traintexts)==len({r['id'] for r in master})==3500 and not errors and not any(overlaps.values())
    report={'records':3500,'unique_normalized_queries':3500,'unique_master_ids':3500,'message_consistency_checked':3500,'mechanical_errors':errors,'protected_text_overlaps':overlaps,'master_sha256':sha(DATA/'MASTER_3500.jsonl'),'sft_sha256':sha(DATA/'train3500.messages.jsonl'),'full_second_semantic_review_completed':False,'status':'mechanical_recheck_completed; semantic_recheck_required_before_training','api_calls':0,'training_started':False}
    write('DATA_RECHECK.json',report)
    metrics=rows(OLD/'runs/route2_train/metrics.jsonl')
    config=json.loads((OLD/'checkpoints/route2_lora/checkpoint-000148/training_config.json').read_text(encoding='utf-8'))
    summary=json.loads((OLD/'runs/route2_train/SUMMARY.json').read_text(encoding='utf-8'))
    common=json.loads((ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/results/COMMON_METRICS.json').read_text(encoding='utf-8'))
    windows=[]
    for lo,hi in [(1,10),(11,37),(38,74),(75,111),(112,148),(139,148)]:
        segment=[r for r in metrics if lo<=r['global_step']<=hi]
        windows.append({'updates':[lo,hi],'mean_training_target_token_loss':statistics.mean(r['loss'] for r in segment)})
    history={'training_records':2368,'epochs':config['epochs'],'updates':summary['global_step'],'consumed_examples':summary['consumed_examples'],'learning_rate':config['learning_rate'],'effective_batch':config['effective_batch_size'],'micro_batch':config['micro_batch'],'lora':config['lora'],'warmup_ratio':config['warmup_ratio'],'weight_decay':config['weight_decay'],'first_training_loss':metrics[0]['loss'],'last_training_loss':metrics[-1]['loss'],'loss_windows':windows,'best_validation_loss':summary['best_validation_loss'],'validation_loss_recorded':(OLD/'runs/route2_train/validation_metrics.jsonl').exists(),'historical_semantic_dev':[r for r in common if r['model_key'].startswith('route2:')],'optimizer_update_seconds':sum(r['elapsed_seconds'] for r in metrics),'reported_training_seconds':summary['training_elapsed_seconds'],'wall_seconds':summary['wall_elapsed_seconds'],'peak_allocated_GiB':summary['peak_allocated_bytes']/1024**3,'peak_reserved_GiB':summary['peak_reserved_bytes']/1024**3,'interpretation':'Training loss flattened late, while dev semantic pass still improved at epoch2. No held-out validation loss was recorded; cannot establish absence of overfitting or full semantic convergence. Old steps/goal task differs from current standalone-support-array task.','source_files':{str(p.relative_to(ROOT)):sha(p) for p in [OLD/'runs/route2_train/metrics.jsonl',OLD/'runs/route2_train/SUMMARY.json',OLD/'checkpoints/route2_lora/checkpoint-000148/training_config.json',ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/results/COMMON_METRICS.json']}}
    write('HISTORICAL_TRAINING_EVIDENCE.json',history)
    findings=[{'index':717,'status':'confirmed_minor_semantic_issue','query':next(r['query'] for r in master if r['index']==717),'current_target':next(r['target'] for r in master if r['index']==717),'issue':'Gendered his/her were added although original only says their; v12 forbids adding gender.','proposed_target':['How many Grand Slam singles and doubles titles combined did Jim Pugh win during Jim Pugh\'s tennis career?','How many Grand Slam singles and doubles titles combined did Iva Majoli win during Iva Majoli\'s tennis career?']},{'index':612,'status':'confirmed_modifier_omission','query':next(r['query'] for r in master if r['index']==612),'current_target':next(r['target'] for r in master if r['index']==612),'issue':'The intermediate hurricane question omitted the identifying most-destructive-of-the-season condition.','proposed_target':['Which hurricane that hit the northeastern United States did Stephen Farrell report on as the most destructive of its season?']}]
    write('PRELIMINARY_SEMANTIC_FINDINGS.json',{'full3500_review_completed':False,'confirmed_findings':findings,'original_master_modified':False,'note':'These preliminary findings are not a final whole-data error count. Preserve v001, complete second semantic review and export a new3500 revision before training.'})
    print(json.dumps({'data':report,'history':{'epochs':history['epochs'],'updates':history['updates'],'loss_windows':windows,'semantic_pass_counts':[r['semantic_pass'] for r in history['historical_semantic_dev']],'denominator':500,'recorded_validation_loss':history['validation_loss_recorded']},'preliminary_semantic_findings':len(findings)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
