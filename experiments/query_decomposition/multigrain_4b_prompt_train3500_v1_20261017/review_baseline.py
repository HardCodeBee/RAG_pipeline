"""Display literal outputs and aggregate explicitly authored semantic judgments."""
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/query_decomposition'/HERE.name
RETURN=OUT/'returned_20261017T094753Z'
ARMS=['v6','v8','v11','v12']
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=100);p.add_argument('--finish',action='store_true');a=p.parse_args()
 raw=rows(RETURN/'predictions.jsonl');group=defaultdict(list)
 for r in raw:group[r['index']].append(r)
 if not a.finish:
  for i,rs in sorted(group.items()):
   if not a.start<=i<=a.end:continue
   print(str(i)+' '+rs[0]['query']);variants=defaultdict(list)
   for r in rs:variants[json.dumps(r['parsed_questions'],ensure_ascii=False)].append(r['arm'])
   for v,arms in variants.items():print('/'.join(arms)+': '+v)
  return
 authored=json.loads((HERE/'BASELINE_AUTHORED.json').read_text(encoding='utf-8'))
 assert len(authored)==100 and len({r[0] for r in authored})==100
 metadata={r['index']:r for r in rows(HERE/'questions100.frozen.jsonl')}
 reviewed=[];stats={arm:{'planned':100,'pass':0,'fail':0,'uncertain':0,'by_dataset':{},'checks_overlapping':Counter(),'F_or_S_union':0} for arm in ARMS}
 for index,codes,reason in authored:
  assert len(codes)==4
  for arm,code in zip(ARMS,codes):
   assert code=='P' or (set(code.lstrip('?'))<=set('FSCU') and code.lstrip('?'))
   r=next(r for r in group[index] if r['arm']==arm)
   verdict='pass' if code=='P' else 'uncertain' if code.startswith('?') else 'fail'
   s=stats[arm];s[verdict]+=1;s['checks_overlapping'].update(set(code.lstrip('?'))&set('FSCU'));s['F_or_S_union']+=int(bool(set(code)&set('FS')))
   ds=metadata[index]['dataset'];d=s['by_dataset'].setdefault(ds,{'planned':0,'pass':0,'fail':0,'uncertain':0});d['planned']+=1;d[verdict]+=1
   reviewed.append({'index':index,'arm':arm,'dataset':ds,'id':r['id'],'query':r['query'],'questions':r['parsed_questions'],'code':code,'verdict':verdict,'reason':reason,'raw_output_sha256':hashlib.sha256(r['raw_output'].encode()).hexdigest(),'reviewer':'current Codex direct literal-output review, nonblind same context; not human gold'})
 for arm,s in stats.items():
  s['minimum_dataset_pass_rate']=min(v['pass']/v['planned'] for v in s['by_dataset'].values());s['mean_input_tokens']=sum(r['input_tokens'] for r in raw if r['arm']==arm)/100
 order=sorted(ARMS,key=lambda arm:(-stats[arm]['pass'],-stats[arm]['minimum_dataset_pass_rate'],stats[arm]['F_or_S_union'],stats[arm]['mean_input_tokens']))
 result={'stats':stats,'selection_order':order,'selected':order[0],'denominator':100,'api_calls':0,'review_scope':'F/S/C/U direct current Codex nonblind review of all400 literal outputs; not independent human gold','checks':'F faithful relations/conditions/unknowns; S standalone; C useful bridge/property coverage; U distinct supporting purpose'}
 (OUT/'BASELINE_SEMANTIC_REVIEW.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed),encoding='utf-8')
 (OUT/'BASELINE_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (HERE/'FINAL_PROMPT.txt').write_bytes((HERE/'remote/prompts'/(order[0]+'.txt')).read_bytes())
 (HERE/'FINAL_PROMPT_SELECTION.json').write_text(json.dumps({**result,'prompt_sha256':hashlib.sha256((HERE/'FINAL_PROMPT.txt').read_bytes()).hexdigest(),'raw_return_sha256':hashlib.sha256((OUT/'base4b_prompt100_return.zip').read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
