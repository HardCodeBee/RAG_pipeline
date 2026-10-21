"""Read-only full-method packets and exact known-success response borrowing."""
import json
import sqlite3
from pathlib import Path
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest,canonical
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts,mechanics

def load(source,qid):
    with sqlite3.connect('file:'+str((Path(source)/'checkpoint.sqlite3').resolve())+'?mode=ro',uri=True) as db:
        records={}
        for kind,key,value in db.execute("SELECT kind,key,value FROM data WHERE key=? OR json_extract(value,'$.qid')=?",(qid,qid)):
            records[(kind,key)]=json.loads(value)
    return records

def final(case,context,nodes,original,transport,store,prompt=None):
    qid=case['qid'];transport.qid=qid
    examples=json.loads(Path(prompts.__file__).with_name('examples.json').read_text(encoding='utf8'))['final']
    prompt=prompt if prompt is not None else prompts.final(case['question'],context,examples) if nodes else prompts.V1.format(question=case['question'],context=context)
    store.put('context',qid,{'qid':qid,'context':context,'helper_step_ids':[n['step_id'] for n in nodes],
                          'original_doc_ids':[d['doc_id'] for d in original['docs']]})
    key,raw=transport.ask('final',prompt,64)
    store.put('prediction',qid,{'qid':qid,'question':case['question'],'prediction':raw.get('answer') if raw['status']=='success' else None,
       'status':raw['status'],'request_id':key,'finish_reason':raw.get('finish_reason'),'final_doc_ids':[d['doc_id'] for d in original['docs']],
       'fallback':not bool(nodes),'helper_step_ids':[n['step_id'] for n in nodes]})

def packet_context(packet):return '\n\n'.join(d['block'] for d in packet['docs'])

def original_packet(case,source):
    candidates=[v for (kind,key),v in source.items() if kind=='retrieval' and v['query']==case['question'] and v['k']==10]
    if len(candidates)!=1:raise ValueError('Expected one fixed original packet')
    return candidates[0]

def skip_failed_source(case,source,store):
    prediction=source.get(('prediction',case['qid']))
    if prediction is None:raise ValueError('Full-method case must be terminal before ablation')
    if prediction.get('prediction') is None:
        store.put('prediction',case['qid'],{**prediction,'ablation_not_executed':'source_infrastructure_missing'})
        return True
    return False

def seed_exact_responses(source_dir,store,cases):
    """Private copies from the full method; never replay unknown requests."""
    allowed={c['qid'] for c in cases};count=0
    with sqlite3.connect('file:'+str((Path(source_dir)/'checkpoint.sqlite3').resolve())+'?mode=ro',uri=True) as db:
        calls=[]
        for (value,) in db.execute("SELECT value FROM data WHERE kind='evidence'"):
            r=json.loads(value)
            if r['qid'] in allowed:calls.append(r)
        aliases={};origins={}
        for evidence in calls:
            origin=evidence['request_id']
            old=db.execute('SELECT id,stage,payload,status,result FROM requests WHERE id=?',(origin,)).fetchone()
            if old is None or old[3]!='success' or json.loads(old[4]).get('status')!='success':continue
            newkey=digest([store.namespace,json.loads(old[2])])
            binding={'qid':evidence['qid'],'stage':evidence['stage']}
            value=aliases.setdefault(newkey,{'origin_request_id':origin,'allowed_bindings':[]})
            if value['origin_request_id']!=origin:raise ValueError('Exact payload has different source response IDs')
            if binding not in value['allowed_bindings']:value['allowed_bindings'].append(binding)
            origins[origin]=old
        for origin,old in origins.items():store.db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?)',old)
        store.db.commit()
        for key,value in aliases.items():
            value['allowed_bindings'].sort(key=lambda r:(r['qid'],r['stage']))
            store.put('resume_request_alias',key,value);count+=1
        # Packets are copied as immutable input material, not as an evidence
        # claim. Transport records only passages actually present in each call.
        for kind,key,value in db.execute("SELECT kind,key,value FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid')"):
            packet=json.loads(value)
            if packet['qid'] in allowed:store.put(kind,key,packet)
    store.event('borrow_exact_known_success_payloads',source=str(source_dir),allowed_questions=len(cases),aliases=count,
                new_api_liabilities=0,unknown_results_borrowed=False)
