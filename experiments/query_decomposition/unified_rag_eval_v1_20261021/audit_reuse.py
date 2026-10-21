"""Read-only migration review. No API, GPU inference, index search or migration.

Replays saved P02 packets and raw receipts solely to establish whether unchanged
messages survive the approved v2 integrity rules. No references or F1 are read.
"""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3

from .prepare_plan import ROOT,HERE,read,write,rel
from experiments.query_decomposition.main_eval_guard_20261021 import watch,watch_p02
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import usage_cost,MAX_OUTPUT,MAX_INPUT,CONTEXT
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import canonical,digest
from src.p02_multigrain_timing_cachefix_v1.core import pipeline

P02=watch_p02.RUN
DS=('hotpotqa','2wiki','musique')

class Blocked(RuntimeError):pass

def main():
    start=datetime.now(timezone.utc)
    out=HERE/'reviews'/start.strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    inventory,_=watch.telemetry()
    config=read(ROOT/'src/p02_multigrain_timing_cachefix_v1/config.json')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(config['resources']['hotpotqa']['dense']['bge']['query_snapshot'],local_files_only=True)
    special=tok.num_special_tokens_to_add(pair=False)
    selected=watch_p02.selected_cases()
    dbs={};cache={};token_cache={}
    receipt_counts=Counter();finish_counts=Counter();firstchunk=content_ttft=0
    case_details=[];queries={};bindings={}
    def request(call):
        nonlocal firstchunk,content_ttft
        path=(ROOT/call['source']).resolve();key=(str(path),call['request_id'])
        if key not in cache:
            if path not in dbs:dbs[path]=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
            row=dbs[path].execute('select payload,status,result from requests where id=?',(call['request_id'],)).fetchone()
            if row is None:raise Blocked('missing_raw_receipt')
            payload,status,result=row;payload=json.loads(payload);result=json.loads(result) if result else {}
            reasons=[]
            if status!='success' or result.get('status','success')!='success':reasons.append('not_success')
            if result.get('finish_reason')!='stop':reasons.append('not_complete_stop')
            usage=result.get('usage')
            if not usage or not all(isinstance(usage.get(k),int) for k in ['prompt_tokens','completion_tokens']):reasons.append('missing_usage')
            if payload.get('model')!=config['generation']['model'] or result.get('returned_model')!=payload.get('model'):reasons.append('model_mismatch')
            if payload.get('temperature')!=0 or payload.get('extra_body',{}).get('enable_thinking') is not False:reasons.append('generation_settings_mismatch')
            if call.get('payload_sha256')!=digest(payload):reasons.append('payload_hash_mismatch')
            cache[key]=(payload,status,result,reasons)
            receipt_counts['unique_receipts']+=1;receipt_counts['eligible' if not reasons else 'ineligible']+=1
            for reason in reasons:receipt_counts['reason_'+reason]+=1
            finish_counts[str(result.get('finish_reason'))]+=1
            timing=result.get('raw_response',{}).get('stream_transport',{})
            firstchunk+=timing.get('first_chunk_seconds') is not None
            content_ttft+=timing.get('client_ttft_seconds') is not None
        return cache[key]
    try:
        for cid,case in sorted(selected.items()):
            calls=case.get('logical_calls',[])
            eligible_prefix=0;all_receipts_eligible=True;logical_cost=0.
            for call in calls:
                try:
                    payload,status,result,reasons=request(call)
                    if not reasons:
                        eligible_prefix+=1
                        u=result['usage'];logical_cost+=usage_cost(u['prompt_tokens'],u['completion_tokens'])
                        bindings[(case['dataset'],case['qid'],call['stage'],digest({k:v for k,v in payload.items() if k!='max_tokens'}))]={'source':call['source'],'request_id':call['request_id']}
                    else:all_receipts_eligible=False
                except Blocked:all_receipts_eligible=False
            packets={}
            for node in case.get('nodes',[]):
                for name in ['original_retrieval','rewritten_retrieval']:
                    packet=node.get(name)
                    if packet:packets.setdefault(packet['query'],packet)
            if case.get('original_retrieval'):packets.setdefault(case['original_retrieval']['query'],case['original_retrieval'])
            overflow=[];inconsistent=[]
            for query,packet in packets.items():
                if query not in token_cache:token_cache[query]=len(tok(query,add_special_tokens=False,truncation=False,verbose=False)['input_ids'])+special
                full=token_cache[query]
                key=(case['dataset'],case['qid'],query)
                queries[key]={'tokens_with_special':full,'legacy_truncated':packet.get('encoder_truncated'),
                              'packet_query_tokens':packet.get('query_tokens'),'needs_complete_window_policy':full>512 or packet.get('encoder_truncated') is True}
                if full>512 or packet.get('encoder_truncated') is True:overflow.append(query)
                if packet.get('query_tokens')!=full:inconsistent.append(query)
            migrated=False;reason=None
            scorable=isinstance(case.get('prediction'),str) and (case['status'].startswith('success') or case['status']=='algorithm_failure')
            if not scorable:reason='old_case_incomplete_prefixes_only'
            elif case['status']=='algorithm_failure':reason='requires_algorithm_failure_scope_review'
            else:
                bystage=defaultdict(list)
                for call in calls:bystage[call['stage']].append(call)
                def retrieve(query):
                    packet=packets.get(query)
                    if packet is None:raise Blocked('packet_missing')
                    if token_cache[query]>512 or packet.get('encoder_truncated') is True:raise Blocked('query_needs_v2_windows')
                    if packet.get('query_tokens')!=token_cache[query]:raise Blocked('query_token_metadata_mismatch')
                    if packet['qid']!=case['qid'] or packet.get('k')!=10 or len({d['doc_id'] for d in packet['docs']})!=10:raise Blocked('packet_identity_mismatch')
                    return deepcopy(packet)
                def ask(stage,prompt,maximum,allow_final_truncation=False):
                    wanted={'model':config['generation']['model'],'messages':[{'role':'user','content':prompt}],
                            'temperature':0.,'max_tokens':MAX_OUTPUT,'extra_body':{'enable_thinking':False}}
                    if len(prompt.encode('utf8'))+256>MAX_INPUT or len(prompt.encode('utf8'))+256+MAX_OUTPUT>CONTEXT:raise Blocked('full_model_input_overflow')
                    for call in bystage.get(stage,[]):
                        payload,status,result,reasons=request(call)
                        if reasons:continue
                        a={k:v for k,v in payload.items() if k!='max_tokens'};b={k:v for k,v in wanted.items() if k!='max_tokens'}
                        if canonical(a)==canonical(b):return call['request_id'],result['answer'],result['finish_reason']
                    raise Blocked('no_same_stage_complete_identical_prompt')
                runtime=read(P02/f'runtime/{case["dataset"]}/{case["qid"]}.json')
                try:
                    replay=pipeline.execute(runtime,retrieve,ask)
                    if replay['prediction']!=case['prediction'] or replay['final_prompt']!=case['final_prompt']:raise Blocked('replay_not_identical')
                    migrated=True
                except (Blocked,RuntimeError,AssertionError,ValueError) as error:reason=str(error)
            case_details.append({'id':cid,'dataset':case['dataset'],'old_status':case['status'],'old_scorable':scorable,
                                 'identical_complete_case_candidate':migrated,'blocked_reason':reason,
                                 'source_receipts_eligible':all_receipts_eligible,'eligible_receipt_bindings':eligible_prefix,
                                 'retrieval_queries_needing_v2':len(overflow),'query_token_metadata_mismatches':len(inconsistent),
                                 'logical_usage_estimate_cny':logical_cost,'selection':'first scorable old trajectory; never by answer score'})
    finally:
        for db in dbs.values():db.close()
    byds={}
    for ds in DS:
        data=[r for r in case_details if r['dataset']==ds]
        good=[r for r in data if r['identical_complete_case_candidate']]
        byds[ds]={'processed_old':len(data),'scorable_old':sum(r['old_scorable'] for r in data),
                  'whole_case_candidates':len(good),'remaining_if_candidates_approved':500-len(good),
                  'candidate_mean_logical_usage_cny':sum(r['logical_usage_estimate_cny'] for r in good)/len(good) if good else None,
                  'blocking_reasons':dict(Counter(r['blocked_reason'] for r in data if not r['identical_complete_case_candidate']))}
    costs=[]
    for dataset,method,cfg,track,dbpath in watch.checkpoints():
        with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True) as db:
            preds={k:json.loads(v) for k,v in db.execute("select key,value from data where kind='prediction'")}
            valid={k for k,r in preds.items() if isinstance(r.get('prediction'),str) and (r['status'].startswith('success') or r['status']=='algorithm_failure')}
            sums=defaultdict(float);missing=set();observed=set();exactttft=oldchunk=0
            for (raw,) in db.execute("select value from data where kind='token_call'"):
                r=json.loads(raw)
                if r.get('qid') not in valid:continue
                observed.add(r['qid']);u=r.get('usage')
                if not u:missing.add(r['qid']);continue
                sums[r['qid']]+=usage_cost(u['prompt_tokens'],u['completion_tokens'])
            complete=valid-missing
            mean=sum(sums[q] for q in complete)/len(complete) if complete else None
            # Count exact TTFT fields; first-chunk observations cannot be promoted.
            a,b=db.execute("select coalesce(sum(json_extract(result,'$.raw_response.stream_transport.client_ttft_seconds') is not null),0),coalesce(sum(json_extract(result,'$.raw_response.stream_transport.first_chunk_seconds') is not null),0) from requests where status='success'").fetchone()
            costs.append({'dataset':dataset,'method':method,'valid_primary_checkpoint':len(valid),
                          'complete_usage_cases':len(complete),'mean_logical_usage_cny':mean,
                          'first_content_ttft_receipts':a,'first_chunk_only_receipts':b,
                          'source':rel(dbpath)})
    with sqlite3.connect((watch.base.PARENT/'budget.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        n,a,b=db.execute('select count(*),coalesce(sum(actual),0),coalesce(sum(case when actual is null then reserved else 0 end),0) from charges').fetchone()
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'review_only_no_migration_no_execution','new_API_requests':0,'new_GPU_jobs':0,'new_index_searches':0,
            'baseline_inventory':inventory,'p02_by_dataset':byds,'p02_receipts':dict(receipt_counts),'p02_finish_reasons':dict(finish_counts),
            'p02_first_chunk_observations':firstchunk,'p02_exact_first_content_TTFT_observations':content_ttft,
            'p02_eligible_binding_keys':len(bindings),'p02_unique_saved_queries':len(queries),'p02_overflow_saved_queries':sum(v['needs_complete_window_policy'] for v in queries.values()),
            'baseline_cost_and_timing_inventory':costs,'budget':{'submissions':n,'known_cny':a,'unknown_reserved_cny':b,'occupied_cny':a+b,'remaining_to300':300-a-b,'dispatch_headroom_to299_9':299.9-a-b},
            'limitations':['Whole-case candidates require user-approved migration and concrete v2 binding validation before publication.','No real inference/index search; overflow cases are deferred instead of guessed.','No F1/gold read and no result selected by score.','Old first-chunk timing cannot reconstruct first-content TTFT.']}
    write(out/'REUSE_AUDIT.json',result)
    (out/'P02_CASE_REVIEW.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in case_details),encoding='utf8')
    write(out/'RECEIPT_BINDING_CANDIDATES.json',{'status':'not_approved_not_imported','bindings':[{'dataset':k[0],'qid':k[1],'stage':k[2],'fingerprint_without_output_cap':k[3],**v} for k,v in sorted(bindings.items())]})
    print(json.dumps({'directory':str(out),'P02':byds,'receipts':dict(receipt_counts),'exact_TTFT':content_ttft,'baseline_exact_TTFT':sum(r['first_content_ttft_receipts'] for r in costs),'budget':result['budget']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
