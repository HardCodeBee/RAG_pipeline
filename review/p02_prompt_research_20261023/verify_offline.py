"""Verify this review package with the Python standard library, without API calls.

python -B verify_offline.py
python -B verify_offline.py --write-result  # writes only VALIDATION.json here
"""
from pathlib import Path
from collections import Counter, defaultdict
import ast, collections, gzip, hashlib, json, math, re, string, sys

HERE=Path(__file__).resolve().parent
SNAP=HERE/'snapshot'
EX=SNAP/'experiments/query_decomposition'
OUT=SNAP/'outputs/query_decomposition'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,abs_tol=1e-10),(a,b)

def functions(path,names):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'))
 body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in body}==set(names)
 env={'re':re,'string':string,'collections':collections,'Counter':Counter}
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env)
 return env

def scorers():
 h=functions(EX/'hotpot_formal_eval_v1/common/scoring.py',['normalize_answer','f1_score','exact_match_score'])
 w=functions(SNAP/'data/multihop_comprehensive_v1_20261015/2wiki/official/2wikimultihop_evaluate_v1.1.py',
   ['normalize_answer','f1_score','exact_match_score','eval_answer','update_answer'])
 m=functions(SNAP/'data/multihop_comprehensive_v1_20261015/musique/official/metrics/answer.py',
   ['normalize_answer','get_tokens','compute_exact','compute_f1','metric_max_over_ground_truths'])
 aliases={x['Q_id']:x['aliases']+x['demonyms'] for x in read(HERE/'scoring/2wiki_aliases.selected.json')['records']}
 def wiki(pred,ref):
  out={'em':0,'f1':0,'prec':0,'recall':0}
  w['update_answer'](out,pred,[ref['answer']]+aliases.get(ref.get('answer_id'),[]))
  return {k:out[k] for k in ('em','f1')}
 return {
  'hotpotqa':lambda p,r:{'em':float(h['exact_match_score'](p,r['answer'])),'f1':h['f1_score'](p,r['answer'])[0]},
  '2wiki':wiki,
  'musique':lambda p,r:{'em':max(m['compute_exact'](a,p) for a in [r['answer']]+r['aliases']),
    'f1':max(m['compute_f1'](a,p) for a in [r['answer']]+r['aliases'])},
 }

def main():
 manifest=read(HERE/'MANIFEST.json')
 for x in manifest['files']:
  p=HERE/x['path'];assert p.is_file(),x['path']
  assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
 inventory=read(HERE/'INVENTORY.json')
 for x in inventory['source_files']:assert sha(HERE/x['path'])==x['sha256']
 for x in inventory['pin_checks']:
  assert x['status']=='matched' and sha(SNAP/x['source'])==x['expected'],x
 receipt_rows=0;usage_rows=0;outcome_rows=0;export_rows=0
 for e in inventory['checkpoints']:
  info=read(HERE/'receipts'/e['experiment']/'SCHEMA.json');tables={}
  for t in info['tables']:
   raw=gzip.decompress((HERE/t['path']).read_bytes())
   assert hashlib.sha256(raw).hexdigest()==t['uncompressed_sha256']
   records=[json.loads(s) for s in raw.decode('utf8').splitlines()]
   assert len(records)==t['rows'];tables[t['table']]=records;export_rows+=len(records)
  requests={x['id']:x for x in tables['requests']}
  assert len(requests)==len(tables['requests'])
  assert dict(Counter(x['status'] for x in requests.values()))==e['request_statuses']
  for x in tables['request_stages']:assert x['request_id'] in requests
  for x in requests.values():
   assert isinstance(x['payload']['messages'],list)
   u=(x.get('result') or {}).get('usage')
   if u:
    assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens'];usage_rows+=1
  outcomes=[x['value'] for x in tables['data'] if x['kind'] in ('prediction','outcome')]
  assert len(outcomes)==e['conditions']
  assert dict(Counter(x.get('status','unspecified') for x in outcomes))==e['statuses']
  for x in outcomes:
   for s in x.get('stages',[]):assert s['request_id'] in requests
   for s in x.get('usage_by_stage',{}).values():
    if s.get('request_id'):assert s['request_id'] in requests
   for field in ('goal_request_id','final_request_id','request_id'):
    if x.get(field):assert x[field] in requests
   if x.get('final_request_id') and x.get('status')=='success':
    assert requests[x['final_request_id']]['result']['answer'].strip()==x['prediction'].strip()
  receipt_rows+=len(requests);outcome_rows+=len(outcomes)
 refs=read(HERE/'scoring/references.json');score=scorers();score_checks=0;cells=0;paired_checks=0;computed={}
 for name in inventory['experiments']:
  if 'design' in name or 'fewshot_synthesis' in name:continue
  detail=OUT/name/'DETAILS.local.jsonl'
  if not detail.exists():detail=OUT/name/'DETAILS.scored.local.jsonl'
  goal=[x for x in rows(detail) if x.get('suite','goal')=='goal']
  idx={(x['arm'],x['id']):x for x in goal}
  assert len(idx)==len(goal)
  for x in goal:
   metrics=x.get('metrics',x.get('native_metrics'))
   if metrics and metrics.get('f1') is not None:
    assert x['status'] in ('success','algorithm_failure')
    actual=score[x['dataset']](x['prediction'],refs[x['id']])
    for k in ('f1','em'):close(actual[k],metrics[k])
    score_checks+=1
  def aggregate(arm,ids):
   vals=[idx[(arm,i)] for i in ids]
   mm=[x.get('metrics',x.get('native_metrics')) for x in vals]
   return {'f1':sum(x['f1'] for x in mm)/len(mm),'em':sum(x['em'] for x in mm)/len(mm)}
  report_file=OUT/name/'ANALYSIS.json'
  if report_file.exists():
   report=read(report_file)
   all_ids=sorted({x['id'] for x in goal})
   for ds,arms in report['by_dataset_arm'].items():
    ids=[i for i in all_ids if ds=='all' or refs[i]['dataset']==ds]
    for arm,v in arms.items():
     a=aggregate(arm,ids)
     for k in ('F1','EM'):close(a[k.lower()],v[k]);cells+=1
   for ds,contrasts in report['paired'].items():
    ids=[i for i in all_ids if ds=='all' or refs[i]['dataset']==ds]
    for contrast,v in contrasts.items():
     a,b=contrast.split('->')
     delta=[idx[(b,i)]['metrics']['f1']-idx[(a,i)]['metrics']['f1'] for i in ids]
     for k,actual in [('improved',sum(x>1e-12 for x in delta)),('harmed',sum(x< -1e-12 for x in delta)),('tied',sum(abs(x)<=1e-12 for x in delta))]:
      if k in v:assert actual==v[k],(name,ds,contrast,k)
     if 'F1_delta' in v:close(sum(delta)/len(delta),v['F1_delta'])
     paired_checks+=1
   computed[name]=report['by_dataset_arm']['all']
  else:
   report=read(OUT/name/'SUMMARY.json')
   ids=report.get('common_ids')
   if ids is None:
    arms=('G2','I','F')
    sets=[{x['id'] for x in goal if x['arm']==arm and x.get('native_metrics',{}).get('f1') is not None} for arm in arms]
    ids=sorted(set.intersection(*sets))
   for table in report['table']:
    sub=[i for i in ids if table['dataset']=='all' or refs[i]['dataset']==table['dataset']]
    assert len(sub)==table['n']
    for arm,v in table['scores'].items():
     if 'historical' in arm:continue
     a=aggregate(arm,sub)
     for k,expected in v.items():close(a[k.lower()],expected);cells+=1
   computed[name]=report['table'][0]
 result={'passed':True,'manifest_files':len(manifest['files']),'historical_pins_verified':len(inventory['pin_checks']),
  'checkpoint_export_rows':export_rows,'terminal_condition_records':outcome_rows,'request_records':receipt_rows,
  'usage_totals_checked':usage_rows,'native_scored_goal_rows':score_checks,'reported_metric_cells_verified':cells,
  'paired_comparisons_checked':paired_checks,'new_API_requests':0,'source_files_modified':False,
  'limitations':['Does not rejudge semantic correctness or rerun models.','Initial42 was blocked; historical cached reference scores are preserved, not counted as newly run evidence.','Bootstrap intervals and manual labels are preserved with source analysis code, not recalculated by this standard-library verifier.'],
  'all_dataset_results':computed}
 if '--write-result' in sys.argv:
  (HERE/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
 print(json.dumps({k:v for k,v in result.items() if k!='all_dataset_results'},ensure_ascii=False))

if __name__=='__main__':main()
