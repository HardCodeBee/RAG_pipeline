"""Audit explicit revision batches; incomplete review is never exported as SFT."""
from collections import Counter
import hashlib,json,re
from training_sources import HERE,OUT,OLD,rows,sha,write,write_rows
def materialize():
 originals=rows(OLD/'accepted_master.jsonl');drafts={r['index']:r for r in rows(OUT/'OLD_DRAFTS.unreviewed.jsonl')}
 frozen={r['index']:r for r in rows(OUT/'LABELING_INPUTS.frozen.jsonl')};decisions={};sources={}
 for p in sorted((HERE/'training_reviews').glob('old_*.json')):
  doc=json.loads(p.read_text(encoding='utf-8'));lo,hi=doc['reviewed_range']
  assert lo<=hi and not set(range(lo,hi+1))&set(decisions)
  assert all(lo<=int(i)<=hi for i in doc['revisions']) and all(lo<=int(i)<=hi for i in doc['quarantine'])
  assert not set(doc['revisions'])&set(doc['quarantine'])
  sources[p.name]=sha(p)
  for i in range(lo,hi+1):
   old=originals[i-1];draft=drafts[i];assert old['id']==draft['id']==frozen[i]['id'] and old['query']==draft['query']==frozen[i]['query']
   target=doc['revisions'].get(str(i),draft['draft']);reason=doc['quarantine'].get(str(i));overlap=frozen[i].get('protected_text_overlap',False)
   if overlap:reason=(reason+'; ' if reason else '')+'Protected evaluation text overlap; excluded from training.'
   decisions[i]={'index':i,'id':old['id'],'query':old['query'],'source':old['source'],'group_id':old['group_id'],'cohort':'old2368','old_target':old['target'],'draft_target':draft['draft'],'target':target,'status':'quarantined' if reason else 'accepted','review_method':'current Codex direct all-record query/draft review with targeted authored correction; same-context nonblind, not human gold','review_batch':str(p.relative_to(HERE)),'review_batch_sha256':sha(p),'change_kind':'authored_revision' if str(i) in doc['revisions'] else 'reviewed_draft_retained','reason':reason or doc['note']}
 accepted=[r for r in decisions.values() if r['status']=='accepted'];errors=[]
 for r in accepted:
  q=r['target']
  if not isinstance(q,list) or any(not isinstance(s,str) or not s.strip() or not s.endswith('?') for s in q):errors.append([r['index'],'non-question-string-array'])
  if len(q)!=len(set(q)):errors.append([r['index'],'duplicate-question'])
  for s in q:
   if re.search(r'#\d+|UNRESOLVED|the result of \(',s):errors.append([r['index'],'unresolved-draft-expression'])
 report={'reviewed_old_count':len(decisions),'old_total':len(originals),'old_remaining':len(originals)-len(decisions),'accepted_reviewed_old':len(accepted),'quarantined_reviewed_old':len(decisions)-len(accepted),'authored_revisions':sum(r['change_kind']=='authored_revision' for r in decisions.values()),'accepted_by_source':dict(Counter(r['source']['dataset'] for r in accepted)),'mechanical_errors':errors,'old_source_sha256':sha(OLD/'accepted_master.jsonl'),'draft_sha256':sha(OUT/'OLD_DRAFTS.unreviewed.jsonl'),'review_batch_hashes':sources,'training_export_ready':False,'api_calls':0}
 write_rows(OUT/'OLD_REVIEWED.progress.jsonl',sorted(decisions.values(),key=lambda r:r['index']));write(OUT/'REVIEW_PROGRESS.json',report)
 return report
if __name__=='__main__':print(json.dumps(materialize(),ensure_ascii=False,indent=2))
