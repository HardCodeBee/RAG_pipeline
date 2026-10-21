"""Offline actual receipt audit. No API or index calls; frozen inputs are read only."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from experiments.query_decomposition.p02_flow_simplify_v1_20261020.score import ROOT,HERE,OUT,read,rows,metric,is_scorable,sha
from experiments.query_decomposition.p02_flow_simplify_v1_20261020.offline_audit import source_catalog,stored_request
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

DEST=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/p02_simplification_review_20261020'
bases={r['id']:r for r in read(HERE/'BASELINE_SOURCES.frozen.json')}
refs={r['id']:r for r in rows(HERE/'references300.local.frozen.jsonl')}
manifests={r['id']:r for r in rows(HERE.parent/'b0_strategy_round2_v1_20261020/SOURCES.frozen.jsonl')}
b0man={r['id']:r for r in rows(HERE.parent/'b0_six_single_changes_v1_20261020/B0_SOURCES.frozen.jsonl')}
unions={}
for p in sorted(OUT.glob('rounds/r*/cases/*/union20/*.json')):
    x=read(p)
    if is_scorable(x): unions.setdefault(x['id'],(p,x))
scorers={ds:metric(ds) for ds in ('hotpotqa','2wiki','musique')}
all_scores=[];changed=[]
for identity,(upath,u) in unions.items():
    b=read(ROOT/bases[identity]['path']);ref=refs[identity];ds=ref['dataset']; score=scorers[ds]
    bf=score(b['prediction'],ref)['f1'];uf=score(u['prediction'],ref)['f1'];delta=uf-bf
    row={'id':identity,'dataset':ds,'question':ref['question'],'reference':ref['answer'],'old_prediction':b['prediction'],'new_prediction':u['prediction'],'old_f1':bf,'new_f1':uf,'delta':delta,'direction':'up' if delta>1e-12 else 'down' if delta< -1e-12 else 'tie','old_source':bases[identity]['path'],'new_source':str(upath.relative_to(ROOT))}
    all_scores.append(row)
    if abs(delta)<=1e-12:continue
    catalog=source_catalog(manifests[identity],b0man[identity],b)
    newcalls={c['stage']:c for c in u['logical_calls']}
    def receipt(prompt,maximum,stage):
        pay=payload('qwen-plus-2025-12-01',prompt,maximum);key=digest(pay);oldcall=catalog[key]
        oc=stored_request(oldcall['source'],oldcall['request_id']);nc=newcalls[stage];ns=stored_request(nc['source'],nc['request_id'])
        assert oc['payload']==pay and digest(ns['payload'])==nc['payload_sha256']
        return {'old_call':oldcall,'new_call':nc,'old_payload_sha256':key,'new_payload_sha256':digest(ns['payload']),'payload_exact_equal':oc['payload']==ns['payload'],'old_response':oc['result']['answer'],'new_response':ns['result']['answer']}
    changes=[];stages=[]
    for bn,un in zip(b['nodes'],u['nodes']):
        assert bn['request']==un['request'] and bn['step_id']==un['step_id']
        sid=bn['step_id']
        for field,pf,limit,kind in [('raw_rewrite','rewrite_prompt',256,'rewrite'),('initial_answer','child_prompt',128,'answer')]:
            if bn.get(pf) is None:continue
            stage=f'legacy/{kind}/{sid}';r=receipt(bn[pf],limit,stage)
            r.update(stage=stage,step_id=sid,role=bn['role'],question=bn['request'])
            stages.append(r)
            if r['old_response']!=r['new_response']:
                olddocs=bn['reader_packet']['docs'];newdocs=un['reader_packet']['docs'];oldids={d['doc_id'] for d in olddocs};newids={d['doc_id'] for d in newdocs}
                r.update(old_answer=bn['answer'],new_answer=un['answer'],old_rewrite=bn.get('rewritten_query'),new_rewrite=un.get('rewritten_query'),prior_qa_equal=bn['prior_complete_answers']==un['prior_complete_answers'],old_doc_ids=[d['doc_id'] for d in olddocs],new_doc_ids=[d['doc_id'] for d in newdocs],added_documents=[d for d in newdocs if d['doc_id'] not in oldids],removed_documents=[d for d in olddocs if d['doc_id'] not in newids],old_prompt=bn[pf],new_prompt=un[pf])
                changes.append(r)
    fr=receipt(b['final_prompt'],64,'legacy/final');fr.update(stage='legacy/final',role='final');stages.append(fr)
    if fr['old_response']!=fr['new_response']:changes.append(fr)
    first=changes[0] if changes else None
    first_answer=next((c for c in changes if c['stage'].startswith('legacy/answer/')),None)
    category='final_only' if first and first['stage']=='legacy/final' else ('same_full_payload_different_response' if first and first['payload_exact_equal'] else 'changed_payload_before_first_response_change')
    row.update(first_divergence=first,first_answer_divergence=first_answer,classification=category,changed_stages=changes,all_stage_payload_comparison=[{k:v for k,v in s.items() if k in ('stage','step_id','role','old_payload_sha256','new_payload_sha256','payload_exact_equal')} for s in stages],complete_decomposition=[{'id':n['step_id'],'request':n['request'],'role':n['role']} for n in b['nodes']],old_goal=b['nodes'][-1]['answer'],new_goal=u['nodes'][-1]['answer'])
    changed.append(row)
summary={}
for ds in ('overall','hotpotqa','2wiki','musique'):
    rr=[r for r in all_scores if ds=='overall' or r['dataset']==ds];cr=[r for r in changed if ds=='overall' or r['dataset']==ds]
    summary[ds]={'n':len(rr),'old_f1':sum(r['old_f1'] for r in rr)/len(rr),'new_f1':sum(r['new_f1'] for r in rr)/len(rr),'delta_pp':100*sum(r['delta'] for r in rr)/len(rr),'directions':dict(Counter(r['direction'] for r in rr)),'first_divergence_classes':dict(Counter(r['classification'] for r in cr)),'first_divergence_by_direction':{d:dict(Counter(r['classification'] for r in cr if r['direction']==d)) for d in ('up','down')}}
result={'scope':'Actual same300 union20 vs frozen p02; native final Answer F1; full-payload receipt audit of all 24 score-changing cases; no new calls or searches','summary':summary,'changed_cases':changed,'all_scores':all_scores,'timing_only_offline_audit':read(OUT/'OFFLINE_AUDIT.json'),'control_limitation':'Identical full payloads were sometimes submitted again in cache v001, producing different responses at temperature 0. This is a confound, not a demonstrated added-document effect. cache_v002 adds missing B0 receipts without replacing already completed cases.'}
(DEST/'union_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=True,indent=2))
for r in changed:
    f=r['first_divergence'];print(json.dumps({'id':r['id'],'direction':r['direction'],'class':r['classification'],'first_stage':f['stage'],'firstq':f.get('question'),'old':f['old_response'],'new':f['new_response'],'added_sources':[d['source'] for d in f.get('added_documents',[])]},ensure_ascii=True))
