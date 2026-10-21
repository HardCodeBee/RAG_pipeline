"""Query-only template drafts for native 2Wiki training inputs; never auto-accepted."""
import argparse,json,re
from training_sources import HERE,OUT,rows,write,write_rows

def clean(x):return x.strip(' ,:?')

def comparison(q,bridge=False):
    text=q.strip().rstrip('?').strip()
    if bridge:
        m=re.match(r'Which film (?:has the director(?: who)?(?: was| is)?|whose director was) (born earlier|born first|born later|died earlier|died first|who is older than the other|older than the other), (.+) or (.+)$',text)
        if m:
            kind,a,b=m.groups();prop='death' if 'died' in kind else 'birth'
        else:
            prop='nationality' if 'nationality' in text else 'country'
            raw=re.sub(r'^(?:Are|Do) (?:both )?(?:the )?(?:directors of (?:both )?films |films[:,]? |director of film )','',text)
            raw=re.sub(r'(?:,? have the directors(?: who are| that share| that| who)? (?:are )?(?:from|share|have) the same (?:country|nationality)| (?:share|have) the same nationality| (?:both )?from the same country)$','',raw)
            raw=re.sub(r' films have the directors from the same country$','',raw)
            raw=raw.rstrip(' ,')
            if ' and director of film ' in raw:a,b=raw.split(' and director of film ',1)
            elif ' and ' in raw:a,b=raw.split(' and ',1)
            else:return None
            a,b=clean(a),clean(b)
        a,b=clean(a),clean(b)
        if prop=='birth':fn=lambda t:f'When was the director of the film {t} born?'
        elif prop=='death':fn=lambda t:f'When did the director of the film {t} die?'
        elif prop=='nationality':fn=lambda t:f'What is the nationality of the director of the film {t}?'
        else:fn=lambda t:f'What country is the director of the film {t} from?'
        return {'entities':[a,b],'operation':prop,'target':[f'Who directed the film {a}?',f'Who directed the film {b}?',fn(a),fn(b)]}
    for pat,kind in [
      (r'^Who (?:is younger|was born (?:earlier|first|later)|lived longer), (.+) or (.+)$','birth'),
      (r'^Was (.+) or (.+) born first$','birth'),
      (r'^Who was born first out of (.+) and (.+)$','birth'),
      (r'^Which film (?:was released|came out) (?:earlier|first|more recently), (.+) or (.+)$','release')]:
        m=re.match(pat,text)
        if m:
            a,b=map(clean,m.groups())
            if 'lived longer' in text:target=[f'When was {a} born?',f'When did {a} die?',f'When was {b} born?',f'When did {b} die?'];kind='lifespan'
            elif kind=='birth':target=[f'When was {a} born?',f'When was {b} born?']
            else:target=[f'When was the film {a} released?',f'When was the film {b} released?']
            return {'entities':[a,b],'operation':kind,'target':target}
    raw=re.sub(r'^(?:Are|Were|Do|Did|Does) (?:both )?(?:(?:businesses|movies|villages|bands|rivers),? |the movies )?','',text)
    raw=re.sub(r'(?:,? (?:both )?(?:located in|from|originate from) the same country| (?:both )?(?:have|had|share|shared) the same nationality| (?:of )?the same nationality)$','',raw)
    raw=re.sub(r' have the same nationality as ',' and ',raw)
    kind='nationality' if 'nationality' in text else ('location' if 'located' in text else 'origin')
    if ' and ' not in raw:return None
    a,b=map(clean,raw.split(' and ',1))
    fn=(lambda t:f'What is {t}\'s nationality?') if kind=='nationality' else ((lambda t:f'What country is {t} located in?') if kind=='location' else (lambda t:f'What country is {t} from?'))
    return {'entities':[a,b],'operation':kind,'target':[fn(a),fn(b)]}

def inference(q):
    rels='paternal grandfather|paternal grandmother|maternal grandfather|maternal grandmother|mother-in-law|father-in-law|sibling-in-law|uncle|father|mother'
    m=re.match(rf'^Who is the ({rels}) of (.+)\?$',q)
    if m:rel,ent=m.groups()
    else:
        m=re.match(rf"^Who is (.+)'s ({rels})\?$",q)
        if not m:return None
        ent,rel=m.groups()
    if rel in ['uncle','father','mother']:target=[]
    elif '-in-law' in rel:
        if rel=='sibling-in-law':target=[f"Who is {ent}'s spouse?",f"Who are the siblings of {ent}'s spouse?",f"Who are {ent}'s siblings?",f"Who are the spouses of {ent}'s siblings?"]
        else:
            parent=rel.split('-')[0];target=[f"Who is {ent}'s spouse?",f"Who is the {parent} of {ent}'s spouse?"]
    else:
        side,parent=rel.split();base='father' if side=='paternal' else 'mother';parent=parent.replace('grand','')
        target=[f"Who is {ent}'s {base}?",f"Who is the {parent} of {ent}'s {base}?"]
    return {'entities':[ent],'operation':rel,'target':target}

def compositional(q):
    m=re.search(r'(?:the )?(director|composer|performer) of (film|song) (.+)',q)
    if m:
        role,kind,title=m.groups();title=title.rstrip('?').strip()
        title=re.sub(r"\s*'s (?:birthday|father|mother)$| (?:graduate from|die|born|is from|got|won|received)$",'',title)
        title=clean(title)
        # Prefix requests put the whole identifying description at the end;
        # suffix requests are reduced only by explicit grammar, not knowledge.
        return {'entities':[title],'operation':role,'target':[f'Who is the {role} of the {kind} {title}?']}
    m=re.search(r"(.+)'s (father|mother|husband|wife)(?:'s (?:birthday)| (?:born|die)|\?)",q)
    if not m:
        m=re.search(r"(.+)'s (father|mother|husband|wife)\?",q)
    if not m:return None
    ent,rel=m.groups();ent=re.sub(r'^(?:When was|When is|When did|Where was|Where did|What nationality is|What is the (?:date of (?:death|birth)|place of birth|cause of death) of|Which country) ','',ent)
    return {'entities':[ent],'operation':rel,'target':[f"Who is {ent}'s {rel}?"]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);args=ap.parse_args()
    path=OUT/'NEW_2WIKI_DRAFTS.unreviewed.jsonl'
    if args.build:
        data=[];counts={}
        for r in rows(OUT/'LABELING_INPUTS.frozen.jsonl'):
            if r['cohort']!='new_train_pool' or r['source']['dataset']!='2wiki':continue
            kind=r['source']['type'];counts.setdefault(kind,0)
            if counts[kind]>=175:continue
            counts[kind]+=1
            result=inference(r['query']) if kind=='inference' else compositional(r['query']) if kind=='compositional' else comparison(r['query'],kind=='bridge_comparison')
            data.append({'index':r['index'],'query':r['query'],'draft':result['target'] if result else None,'parse':{k:v for k,v in (result or {}).items() if k!='target'},'status':'unreviewed'})
        write_rows(path,data);write(OUT/'NEW_SELECTION.json',{'2wiki':counts,'rule':'First175 frozen candidates per type, before target review. No score-based filtering.','api_calls':0})
        print(json.dumps({'drafts':len(data),'unmatched':[r['index'] for r in data if r['draft'] is None]}))
    else:
        for r in rows(path):
            if args.start<=r['index']<=args.end:print(str(r['index'])+' '+r['query']+'\n'+json.dumps(r['draft'],ensure_ascii=False))
if __name__=='__main__':main()
