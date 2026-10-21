"""Read-only endpoint join after blinded manual annotation and recheck.

No semantic labels are assigned here. All proportions come from saved labels.
Cluster bootstrap resamples original queries, preserving all nodes/events.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import numpy as np
from prepare_inventory import ROOT,OUT,read,rows,write,sha

run=OUT/'offline_review_v2'
cohort=rows(run/'COHORT.jsonl')
packets={p['id']:p for p in rows(run/'STAGE_PACKETS.jsonl') if p.get('complete')}
dlabels=read(run/'DECOMPOSITION_LABELS_LOCKED.json')['labels']
dl={l['id']:l for l in dlabels}
stage=[l for f in sorted((run/'reviews').glob('*.json')) for l in read(f)['labels'] if l['axis']=='stage_node']
amendments=read(run/'STAGE_AMENDMENTS.json')['amendments']
for l in stage:
 for a in amendments:
  if (l['review_index'],l['step_id'])==(a['review_index'],a['step_id']):
   l.update({k:a[k] for k in ('answer_correct','answer_subtype') if k in a})
sl={(l['id'],l['step_id']):l for l in stage}
assert len(sl)==len(stage)==sum(len(p['nodes']) for p in packets.values())==2084
assert len(dl)==len(packets)==723
pending={(v['review_index'],v['step_id']) for v in read(run/'PENDING_FACTS.json')['items']}
assert pending=={(l['review_index'],l['step_id']) for l in stage if l['answer_correct'] is None}
assert len(read(run/'STAGE_RECHECK_COMPLETED.json')['amendments'])==len(amendments)
write(run/'STAGE_LABELS_LOCKED.json',{'status':'all723_initial_stage_review_and_second_pass_completed','utc':datetime.now(timezone.utc).isoformat(),'node_count':len(stage),'reviewed_binary_answer_labels':sum(l['answer_correct'] is not None for l in stage),'pending_factual_workitems_without_truth_labels':len(pending),'stage_annotation_final_outcomes_hidden':True,'reviewer':'single current Codex reviewer, not independent human gold','labels':stage,'source_reviews':read(run/'REVIEW_PROGRESS.json')['reviewed_batch_files'],'amendments_sha256':sha(run/'STAGE_AMENDMENTS.json')})

# Endpoints are first joined only after labels above are fixed.
ys={y['id']:y for y in rows(run/'OUTCOMES_SEPARATE.jsonl')}
assert set(ys)==set(packets)
records=[]
def all_or_bad(values):
 if any(v is False for v in values):return False
 if values and all(v is True for v in values):return True
 return None

for c in cohort:
 p=packets[c['id']];labs=[sl[c['id'],n['step_id']] for n in p['nodes']]
 ordinary=[l for l in labs if l['role']=='ordinary'];goals=[l for l in labs if l['role']=='goal']
 rr=[l for l in labs if l['rewrite_fidelity_correct'] is not None]
 rec={**c,'question':p['question'],'final_correct':bool(ys[c['id']]['final_correct']),'f1':ys[c['id']]['f1'],'em':ys[c['id']]['em'],
 'decomposition_correct':dl[c['id']]['correct'],'decomposition_utility_only_failure':dl[c['id']]['utility_only_failure'],
 'ordinary_answers_all_correct':all_or_bad([l['answer_correct'] for l in ordinary]),
 'goal_answer_correct':goals[0]['answer_correct'],
 'rewrites_all_faithful':all(l['rewrite_fidelity_correct'] for l in rr),
 'rewrites_all_valid_targets':all(l['rewrite_target_correct'] for l in rr),
 'rewrites_all_correct':all(l['rewrite_fidelity_correct'] and l['rewrite_target_correct'] for l in rr),
 'ordinary_rewrites_all_correct':all(l['rewrite_fidelity_correct'] and l['rewrite_target_correct'] for l in rr if l['role']=='ordinary'),
 'goal_rewrite_correct':all(l['rewrite_fidelity_correct'] and l['rewrite_target_correct'] for l in rr if l['role']=='goal'),
 'ordinary_answer_count':len(ordinary),'rewrite_count':len(rr),'reasonable_ordinary_abstention':any(l['answer_correct'] is True and 'abstention' in l['answer_subtype'] for l in ordinary),
 'ordinary_error_steps':[l['step_id'] for l in ordinary if l['answer_correct'] is False],
 'rewrite_error_steps':[l['step_id'] for l in rr if not(l['rewrite_fidelity_correct'] and l['rewrite_target_correct'])]}
 records.append(rec)
(run/'QUERY_LABELS_AND_OUTCOMES.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')

events=[]
for c in cohort:
 p=packets[c['id']]
 for n in p['nodes']:
  if not n['rewrite_request_id']:continue
  l=sl[c['id'],n['step_id']]
  prev=[sl[c['id'],a['step_id']] for a in n['prior_complete_answers'] if sl[c['id'],a['step_id']]['role']=='ordinary']
  assert prev and all(a['step_id']<n['step_id'] for a in prev)
  used=[sl[c['id'],s] for s in l['explicit_binding_answer_step_ids']]
  events.append({'id':c['id'],'dataset':c['dataset'],'review_index':c['review_index'],'step_id':n['step_id'],'role':n['role'],'prefix':False,
   'prior_ordinary_answers_all_correct':all_or_bad([v['answer_correct'] for v in prev]),
   'explicitly_bound_prior_all_correct':all_or_bad([v['answer_correct'] for v in used]),
   'rewrite_fidelity_correct':l['rewrite_fidelity_correct'],'rewrite_target_correct':l['rewrite_target_correct'],
   'rewrite_correct':l['rewrite_fidelity_correct'] and l['rewrite_target_correct'],
   'effective_query_changed':n['rewritten_query']!=n['request'],'rejected_raw':bool(n['rewrite_rejection']),
   'final_correct':bool(ys[c['id']]['final_correct']), 'current_request':n['request'],'effective_rewrite':n['rewritten_query'],'raw_rewrite':n['raw_rewrite'],'reason':l['reason']})
prefix=read(run/'PREFIX_REWRITE_LABELS.json')['labels']
for p in prefix:
 p.update(prefix=True,rewrite_correct=p['rewrite_fidelity_correct'] and p['rewrite_target_correct'])
(run/'REWRITE_EVENTS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in events),encoding='utf-8')

tables=[]
def table(dataset,name,source,x,y='final_correct',filter_fn=None):
 data=[r for r in source if r['dataset']==dataset and (filter_fn is None or filter_fn(r))]
 eligible=[r for r in data if r.get(x) is not None and r.get(y) is not None]
 ids=sorted({r['id'] for r in data});idx={k:j for j,k in enumerate(ids)}
 if not ids:return
 rng=np.random.default_rng(20261021)
 weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
 res={'dataset':dataset,'table':name,'unit':'original query' if source is records else 'stage node or rewrite event','condition_field':x,'success_field':y,'eligible_rows':len(eligible),'excluded_missing_rows':len(data)-len(eligible),'original_query_clusters':len(ids),'groups':{}}
 rates={}
 for flag,key in [(True,'correct'),(False,'incorrect')]:
  group=[r for r in eligible if r[x] is flag]
  den=np.zeros(len(ids),dtype=int);num=np.zeros(len(ids),dtype=int)
  for r in group:den[idx[r['id']]]+=1;num[idx[r['id']]]+=int(r[y])
  ds=weights@den;ns=weights@num
  bs=np.divide(ns,ds,out=np.full(2000,np.nan),where=ds>0)
  finite=bs[np.isfinite(bs)];rates[key]=bs
  res['groups'][key]={'n':int(den.sum()),'successes':int(num.sum()),'probability':float(num.sum()/den.sum()) if den.sum() else None,'cluster_bootstrap_95_ci':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,'nonempty_bootstrap_replicates':len(finite)}
 delta=rates['correct']-rates['incorrect'];finite=delta[np.isfinite(delta)]
 p1=res['groups']['correct']['probability'];p0=res['groups']['incorrect']['probability']
 res['correct_minus_incorrect']={'probability_difference':p1-p0 if p1 is not None and p0 is not None else None,'cluster_bootstrap_95_ci':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,'nonempty_bootstrap_replicates':len(finite)}
 tables.append(res)

nodes=[]
for l in stage:
 nodes.append({**l,'final_correct':bool(ys[l['id']]['final_correct'])})
for ds in ['2wiki','hotpotqa']:
 table(ds,'D_to_Y',records,'decomposition_correct')
 table(ds,'D_semantic_to_Y',records,'decomposition_correct',filter_fn=lambda r:not r['decomposition_utility_only_failure'])
 table(ds,'A_history_to_R',events,'prior_ordinary_answers_all_correct','rewrite_correct')
 table(ds,'A_history_to_R_fidelity',events,'prior_ordinary_answers_all_correct','rewrite_fidelity_correct')
 table(ds,'A_history_to_R_target',events,'prior_ordinary_answers_all_correct','rewrite_target_correct')
 for role in ['ordinary','goal']:
  table(ds,'A_history_to_R_'+role,events,'prior_ordinary_answers_all_correct','rewrite_correct',lambda r:r['role']==role)
 table(ds,'A_bound_to_R',events,'explicitly_bound_prior_all_correct','rewrite_correct',lambda r:r['explicitly_bound_prior_all_correct'] is not None)
 table(ds,'A_history_to_R_with_prefix',events+prefix,'prior_ordinary_answers_all_correct','rewrite_correct')
 table(ds,'A_to_Y_query',records,'ordinary_answers_all_correct')
 table(ds,'A_to_Y_query_excluding_reasonable_abstentions',records,'ordinary_answers_all_correct',filter_fn=lambda r:not r['reasonable_ordinary_abstention'])
 table(ds,'A_to_Y_node',nodes,'answer_correct',filter_fn=lambda r:r['role']=='ordinary')
 table(ds,'GoalA_to_Y',records,'goal_answer_correct')
 table(ds,'R_to_Y_query',records,'rewrites_all_correct')
 table(ds,'R_fidelity_to_Y_query',records,'rewrites_all_faithful')
 table(ds,'R_target_to_Y_query',records,'rewrites_all_valid_targets')
 table(ds,'R_to_Y_event',events,'rewrite_correct')
 table(ds,'R_to_Y_changed_event',events,'rewrite_correct',filter_fn=lambda r:r['effective_query_changed'])
 for role in ['ordinary','goal']:
  table(ds,'R_to_Y_event_'+role,events,'rewrite_correct',filter_fn=lambda r:r['role']==role)

summary={}
for ds in ['2wiki','hotpotqa']:
 rr=[r for r in records if r['dataset']==ds];nn=[n for n in nodes if n['dataset']==ds];ee=[e for e in events if e['dataset']==ds]
 summary[ds]={'complete_eligible_queries':len(rr),'final_correct':sum(r['final_correct'] for r in rr),'mean_native_F1':float(np.mean([r['f1'] for r in rr])),'mean_native_EM':float(np.mean([r['em'] for r in rr])),
 'decomposition_correct':sum(r['decomposition_correct'] for r in rr),'decomposition_utility_only_failure':sum(r['decomposition_utility_only_failure'] for r in rr),'ordinary_nodes':sum(n['role']=='ordinary' for n in nn),'goal_nodes':sum(n['role']=='goal' for n in nn),'rewrite_events':len(ee),'rewrite_event_errors':sum(not e['rewrite_correct'] for e in ee),'rewrite_error_queries':sum(not r['rewrites_all_correct'] for r in rr),'effective_changed':sum(e['effective_query_changed'] for e in ee),'effective_unchanged':sum(not e['effective_query_changed'] for e in ee),'raw_rewrite_rejected':sum(e['rejected_raw'] for e in ee),
 'by_role':{role:{'answer_correct':sum(n['answer_correct'] is True for n in nn if n['role']==role),'answer_incorrect':sum(n['answer_correct'] is False for n in nn if n['role']==role),'answer_pending':sum(n['answer_correct'] is None for n in nn if n['role']==role),'correct_reasonable_abstentions':sum(n['answer_correct'] is True and 'abstention' in n['answer_subtype'] for n in nn if n['role']==role),'incorrect_subtypes':dict(Counter(n['answer_subtype'] for n in nn if n['role']==role and n['answer_correct'] is False))} for role in ['ordinary','goal']}}
write(run/'CONDITIONAL_PROBABILITIES.json',{'status':'computed_from_locked_offline_annotations_and_existing_endpoints','utc':datetime.now(timezone.utc).isoformat(),'final_correctness':'nativeF1==1 OR nativeEM==1','bootstrap':{'unit':'original query cluster','replicates':2000,'seed':20261021,'confidence':.95,'interval':'percentile; NaN draws with empty conditioning groups excluded and counted; singleton/all-success strata may have degenerate bootstrap intervals'},'association_not_causal':True,'summary':summary,'tables':tables,'annotation_files':{'stage':sha(run/'STAGE_LABELS_LOCKED.json'),'decomposition':sha(run/'DECOMPOSITION_LABELS_LOCKED.json'),'endpoints':sha(run/'OUTCOMES_SEPARATE.jsonl')}})
print(json.dumps({'summary':summary,'main_tables':[t for t in tables if t['table'] in ['D_to_Y','A_history_to_R','A_to_Y_query','R_to_Y_query','GoalA_to_Y']]},ensure_ascii=False,indent=2))
