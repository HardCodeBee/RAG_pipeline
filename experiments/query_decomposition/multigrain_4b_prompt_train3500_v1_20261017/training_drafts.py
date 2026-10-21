"""Deterministic OLD-label drafts only. Every result still requires Codex review."""
import argparse,json,re
from pathlib import Path
from training_sources import HERE,OUT,OLD,rows,write_rows
def clean(q):return re.sub(r'\s+',' ',q).strip().rstrip('?,. ')
def np(q):
 q=clean(q)
 for pattern,form in [
  (r'Who (?:wrote|authored) (.+)',lambda m:'the author of '+m[1]),
  (r'Who (?:directed) (.+)',lambda m:'the director of '+m[1]),
  (r'Who (?:produced) (.+)',lambda m:'the producer of '+m[1]),
  (r'Who (?:hosted) (.+)',lambda m:'the host of '+m[1]),
  (r'Who (?:created) (.+)',lambda m:'the creator of '+m[1]),
  (r'(?:In what year|When) (?:was|were) (.+) born',lambda m:'the birth date of '+m[1]),
  (r'(?:In what year|When) did (.+) die',lambda m:'the date of death of '+m[1]),
  (r'(?:In what city|Where) (?:was|were) (.+) born',lambda m:'the birthplace of '+m[1]),
  (r'(?:In what city|Where) did (.+) die',lambda m:'the place where '+m[1]+' died'),
  (r'(?:Who|What) (?:is|are|was|were) (.+)',lambda m:m[1] if m[1].lower().startswith(('the ','a ','an ')) else 'the '+m[1]),
  (r'Which (.+)',lambda m:'the '+m[1]),
  (r'What (.+)',lambda m:'the '+m[1]),
  (r'Where (?:is|are|was|were) (.+)',lambda m:'the location of '+m[1]),
  (r'Who (.+)',lambda m:'the person who '+m[1]),
  (r'When (.+)',lambda m:'the date when '+m[1]),
  (r'In what (\w+) (.+)',lambda m:'the '+m[1]+' where '+m[2]),
 ]:
  m=re.fullmatch(pattern,q,re.I)
  if m:return form(m)
 return 'the result of ('+q+')'
def count_question(q):
 q=clean(q)
 m=re.fullmatch(r'Who (?:are|were) the members of (.+)',q,re.I)
 if m:return 'How many members does '+m[1]+' have?'
 m=re.fullmatch(r'Who (are|were) (?:the )?(.+)',q,re.I)
 if m:return 'How many '+m[2]+' '+m[1].lower()+' there?'
 if re.match(r'(Which|What) ',q,re.I):return re.sub(r'^(Which|What) ','How many ',q,flags=re.I)+'?'
 return 'How many items satisfy the following description: '+q+'?'
def old_draft(r):
 steps=r['target']['steps'];goal=r['target']['goal'];expanded={};kept=[];consumed=set();flags=[]
 for s in steps:
  i,q=s['id'],s['request']
  expanded[i]=re.sub(r'#(\d+)',lambda m:np(expanded.get(int(m[1]),'UNRESOLVED')),q)
  if i==goal:continue
  if re.match(r'^(?:What is|Is|Are|Which).*#\d+',q) and re.search(r'\b(?:minus|plus|divided|multiplied|sum|difference|greater|less|equal|both|either|common|in both)\b',q,re.I):
   flags.append('arithmetic_or_combination_omitted');continue
  m=re.fullmatch(r'How many .+ (?:are|is) in #(\d+)\?',q,re.I)
  if m:
   ref=int(m[1]);kept.append((i,count_question(expanded[ref])));consumed.add(ref);continue
  kept.append((i,expanded[i]))
 questions=[]
 for i,q in kept:
  if i in consumed:continue
  if q not in questions:questions.append(q)
 if len(steps)==1:flags.append('single_old_goal_requires_query_semantic_review')
 if any('UNRESOLVED' in q or 'the result of (' in q for q in questions):flags.append('nominalization_requires_revision')
 if any(re.search(r'\b(that|those|these|previous)\b',q,re.I) for q in questions):flags.append('possible_anaphora')
 return {'index':r['source_line'] if 'source_line' in r else None,'id':r['id'],'query':r['query'],'draft':questions,'draft_flags':flags,'status':'unreviewed_deterministic_old_QDMR_draft_not_training_gold'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=50);p.add_argument('--build',action='store_true');a=p.parse_args()
 if a.build:
  old=rows(OLD/'accepted_master.jsonl');drafts=[]
  for i,r in enumerate(old,1):
   d=old_draft(r);d['index']=i;drafts.append(d)
  write_rows(OUT/'OLD_DRAFTS.unreviewed.jsonl',drafts)
  print('Unreviewed drafts='+str(len(drafts)));return
 for r in rows(OUT/'OLD_DRAFTS.unreviewed.jsonl'):
  if a.start<=r['index']<=a.end:print(str(r['index'])+' '+r['query']+'\n'+json.dumps(r['draft'],ensure_ascii=False))
if __name__=='__main__':main()
