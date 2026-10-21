"""Freeze query-only labeling inputs. No API, model inference or target acceptance."""
from collections import Counter,defaultdict
import hashlib,json,re,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/query_decomposition'/HERE.name/'training'
OLD=ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data'
NATIVE=ROOT/'data/multihop_comprehensive_v1_20261015'
SEED='20261017-selfcontained3500-v12-01'
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_rows(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value),encoding='utf-8')
def array_stream(path):
 decoder=json.JSONDecoder();buffer='';started=False;eof=False
 with path.open(encoding='utf-8') as stream:
  while True:
   if not eof and len(buffer)<1048576:
    chunk=stream.read(1048576);buffer+=chunk;eof=not chunk
   buffer=buffer.lstrip()
   if not started:
    assert buffer and buffer[0]=='[';buffer=buffer[1:];started=True;continue
   buffer=buffer.lstrip(' \r\n\t,')
   if not buffer:
    if eof:raise ValueError('Incomplete source')
    continue
   if buffer[0]==']':return
   try:r,end=decoder.raw_decode(buffer)
   except json.JSONDecodeError:
    if eof:raise
    chunk=stream.read(1048576);buffer+=chunk;eof=not chunk;continue
   yield r;buffer=buffer[end:]
def project(dataset,split):
 if dataset=='2wiki':
  for r in array_stream(NATIVE/'2wiki/native'/f'{split}.json'):
   yield {'id':r['_id'],'query':r['question'],'type':r['type']}
 else:
  with (NATIVE/'musique/native'/f'musique_ans_v1.0_{split}.jsonl').open(encoding='utf-8') as stream:
   for line in stream:
    r=json.loads(line);yield {'id':r['id'],'query':r['question'],'type':str(len(r['question_decomposition']))+'-hop'}
def main():
 if (OUT/'INPUTS_FROZEN.json').exists():print((OUT/'INPUTS_FROZEN.json').read_text());return
 OUT.mkdir(parents=True,exist_ok=True)
 old=rows(OLD/'accepted_master.jsonl');protected={r['query_sha256'] for r in rows(OLD/'training_exclusions.hashes.jsonl')}
 protected_ids=set();provenance=[]
 for ds in ['2wiki','musique']:
  for r in project(ds,'dev'):protected.add(digest(norm(r['query'])));protected_ids.add(r['id'])
 hp=ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl'
 for r in rows(hp):protected.add(digest(norm(r['question'])));protected_ids.add(r['id'])
 files=set()
 for pattern in ['*questions*.jsonl','cases.frozen.jsonl','queries*.jsonl']:
  files.update((ROOT/'experiments/query_decomposition').rglob(pattern))
 for p in sorted(files):
  if any(v in p.parts for v in ['vendor','source','data','remote']):continue
  n=0
  for r in rows(p):
   q=r.get('query',r.get('question',r.get('question_text')))
   if isinstance(q,str):protected.add(digest(norm(q)));n+=1
  if n:provenance.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'questions':n})
 # Current100 is explicitly protected, including if traversal exclusions change.
 for r in rows(HERE/'EXCLUDE_FROM_TRAINING.jsonl'):protected.add(digest(norm(r['query'])));protected_ids.add(r['qid'])
 hard_path=HERE.parent/'multigrain_decomposer_fresh_eval_v1_20261017/hard_cases_for_4b/CANDIDATES.jsonl'
 hard=rows(hard_path);records=[];seen=set()
 for i,r in enumerate(old,1):
  records.append({'index':i,'id':r['id'],'query':r['query'],'source':r['source'],'old_target':r['target'],'cohort':'old2368','group_id':r['group_id'],'source_line':i,'protected_text_overlap':digest(norm(r['query'])) in protected})
  seen.add(norm(r['query']))
 for r in hard:
  if norm(r['query']) in seen:continue
  s=r['source'];records.append({'index':len(records)+1,'id':r['id'],'query':r['query'],'source':{'dataset':s['dataset'],'id':s['native_id'],'original_split':s['native_split'],'type':s['query_type']},'cohort':'user_designated_hard','old_target':None,'training_exception':'User designated exposed hard cases for training; original dev source retained; exclude future evaluation','source_label_status':r['training_preparation']['label_status'],'protected_project_split_conflict':r['training_preparation']['protected_metadata_overlap']});seen.add(norm(r['query']))
 counts={};excluded=Counter()
 quotas={'2wiki':{'comparison':220,'inference':220,'compositional':220,'bridge_comparison':220},'musique':{'2-hop':250,'3-hop':350,'4-hop':350}}
 for ds,qs in quotas.items():
  buckets=defaultdict(list)
  for r in project(ds,'train'):
   key=norm(r['query'])
   if key in seen:excluded['existing_query']+=1;continue
   if digest(key) in protected or r['id'] in protected_ids:excluded['protected_query_or_id']+=1;continue
   buckets[r['type']].append(r)
  for kind,n in qs.items():
   pool=sorted(buckets[kind],key=lambda r:digest(SEED+ds+r['id']))
   taken=0
   for r in pool:
    if norm(r['query']) in seen:continue
    records.append({'index':len(records)+1,'id':ds+':'+r['id'],'query':r['query'],'source':{'dataset':ds,'id':r['id'],'original_split':'train','type':kind},'cohort':'new_train_pool','old_target':None});seen.add(norm(r['query']));taken+=1
    if taken==n:break
   assert taken==n,(ds,kind,taken)
  counts[ds]=sum(qs.values())
 write_rows(OUT/'LABELING_INPUTS.frozen.jsonl',records)
 write(OUT/'INPUTS_FROZEN.json',{'seed':SEED,'candidate_count':len(records),'old_source_count':len(old),'hard_count':len(hard),'new_candidate_pool':counts,'old_source_sha256':sha(OLD/'accepted_master.jsonl'),'hard_source_sha256':sha(hard_path),'prompt_sha256':sha(HERE/'FINAL_PROMPT.txt'),'selected_prompt':'v12','protected_query_hashes':len(protected),'exclusion_sources':provenance,'excluded_new_candidates':dict(excluded),'old_protected_overlap_count':sum(r.get('protected_text_overlap',False) for r in records),'accepted_training_count':0,'labels_authored_count':0,'api_calls':0,'note':'This is a candidate pool, not a3500 accepted training set. Current Codex must author/review each retained target. Raw answers/passages/native decomposition targets do not cross the query-only projection.'})
 print(json.dumps({'candidates':len(records),'old2368_protected_overlap':sum(r.get('protected_text_overlap',False) for r in records),'hard':len(hard),'new_pool':counts},indent=2))
if __name__=='__main__':main()
