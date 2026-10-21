"""Persist manually reviewed decisions, with source hashes and explicit coverage."""
import argparse
import json
from datetime import datetime, timezone
from prepare_inventory import OUT, rows, read, write, sha

run=OUT/'offline_review_v2'
parser=argparse.ArgumentParser()
parser.add_argument('input_file')
args=parser.parse_args()
batch=read(__import__('pathlib').Path(args.input_file))
packets={p['review_index']:p for p in rows(run/'DECOMPOSITION_PACKETS.jsonl') if p.get('complete')}
stages={p['review_index']:p for p in rows(run/'STAGE_PACKETS.jsonl') if p.get('complete')}
labels=[]
items=list(batch.get('items',[]))
for group in batch.get('manually_reviewed_pass_groups',[]):
    items.extend([i,[],group['reason']] for i in group['indices'])
for flow in batch.get('manually_reviewed_flows',[]):
    assert batch['axis']=='stage_node' and flow['all_nodes_inspected'] is True
    i=flow['i']
    for n in stages[i]['nodes']:
        s=n['step_id'];called=bool(n['rewrite_request_id'])
        bad_answer=s in flow.get('incorrect_answers',[])
        pending_answer=s in flow.get('pending_answer_facts',[])
        bad_rf=s in flow.get('incorrect_fidelity',[])
        bad_rt=s in flow.get('incorrect_targets',[])
        subtype=flow.get('answer_subtypes',{}).get(str(s),'incorrect_fact' if bad_answer else 'factual')
        items.append([i,s,None if pending_answer else not bad_answer,subtype,not bad_rf if called else None,
                      not bad_rt if called else None,flow.get('bindings',{}).get(str(s),[]),flow['reason']])
for item in sorted(items,key=lambda r:r[0]):
    i=item[0];p=packets[i]
    if batch['axis']=='decomposition':
        failed=item[1];reason=item[2]
        label={'axis':'decomposition','correct':not failed,'dimensions':{k:k not in failed for k in 'FSCU'},
               'reason':reason,'reviewed_question':p['question'],'reviewed_children':p['supporting_questions']}
    elif batch['axis']=='stage_node':
        step,answer_correct,subtype,rf,rt,used,reason=item[1:]
        n=next(n for n in stages[i]['nodes'] if n['step_id']==step)
        assert bool(n['rewrite_request_id'])==(rf is not None)==(rt is not None)
        assert (answer_correct is None or isinstance(answer_correct,bool)) and all(s<step for s in used)
        label={'axis':'stage_node','step_id':step,'role':n['role'],
            'answer_correct':answer_correct,'answer_subtype':subtype,
            'answer_review_status':'fact_pending_no_correctness_label' if answer_correct is None else 'reviewed',
            'rewrite_fidelity_correct':rf,'rewrite_target_correct':rt,
            'explicit_binding_answer_step_ids':used,'reason':reason,
            'reviewed_request':n['request'],'reviewed_answer':n['answer'],
            'effective_rewrite':n['rewritten_query'],'raw_rewrite':n['raw_rewrite'],
            'rewrite_rejection':n['rewrite_rejection'],
            'fact_reference_packet':'FACT_REFERENCE_PACKETS.jsonl#'+p['id'],
            'reader_evidence_packet':'STAGE_PACKETS.jsonl#'+p['id']+':'+str(step)}
    else:
        raise ValueError('Unknown axis')
    labels.append({**label,'id':p['id'],'review_index':i,'dataset':p['dataset'],
        'source_sha256':p['source_sha256'],'reviewer':'current_Codex_offline_single_reviewer',
        'final_prediction_or_score_shown':False,'protocol_revision':2})
dest=run/'reviews';dest.mkdir(exist_ok=True)
path=dest/(batch['batch_id']+'.json')
assert not path.exists(),path
write(path,{'batch_id':batch['batch_id'],'reviewed_utc':datetime.now(timezone.utc).isoformat(),
            'axis':batch['axis'],'labels':labels,'manual_decisions':True})
all_labels=[]
for f in sorted(dest.glob('*.json')):all_labels.extend(read(f)['labels'])
keys=[(r['axis'],r['id'],r.get('step_id')) for r in all_labels]
assert len(keys)==len(set(keys)), 'Duplicate semantic review'
write(run/'REVIEW_PROGRESS.json',{'label_count':len(all_labels),'by_axis':{
    axis:sum(r['axis']==axis for r in all_labels) for axis in sorted({r['axis'] for r in all_labels})},
    'complete_cohort':len(packets),'all_stages_completed':False,
    'reviewed_batch_files':[{'path':f.name,'sha256':sha(f)} for f in sorted(dest.glob('*.json'))]})
print(json.dumps({'saved':str(path),'labels':len(labels),'total':len(all_labels)}))
