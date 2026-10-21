"""Prepare two example-only variants after an explicit80-record label inspection."""
import collections, difflib, json, math, re
from prepare import HERE,ROOT,OUT,OLD,rows,sha,write,freeze

EXCEPTIONS={
 3:('uncertain',['U'],'The second Kanye-genre question may be an original-target paraphrase rather than a distinct prerequisite; keep uncertainty separate from definite fact distortion.'),
 20:('fail',['U'],'Anne is already named and described as duchess of the county. Reasking her title adds no new prerequisite beyond the county identity.'),
 25:('uncertain',['U'],'The final generic Where question for named Walnut Corner may duplicate the country target rather than expose a distinct intermediate; retain original label unchanged.'),
 33:('uncertain',['U'],'The final generic Where question for named Midway may duplicate the country target; unclear added supporting role.'),
 51:('fail',['F'],'The birthplace question loses the New England Patriots state condition, although a separate state question is present; the city is no longer jointly restricted.'),
 54:('fail',['F'],'Changes an unspecified intelligence-gathering target into a nation; that entity type is not supplied in the original wording.'),
 58:('fail',['F'],'The author-birthplace question loses the Charles River Peninsula state condition; asking the state separately does not bind it to that birthplace question.'),
 73:('uncertain',['F'],'The first concert-artists question drops contemporaneity with a band named after an American city. It might retrieve a broad candidate set, but full joint identification is not preserved.'),
}

def tokens(q):
    words=re.findall(r"[\w]+(?:['’-][\w]+)*",q.casefold())
    return words+['b:'+a+' '+b for a,b in zip(words,words[1:])]

def retrieve(query,bank):
    docs=[collections.Counter(tokens(r['query'])) for r in bank]
    df=collections.Counter(t for doc in docs for t in doc)
    avg=sum(sum(doc.values()) for doc in docs)/len(docs)
    scored=[]
    for r,doc in zip(bank,docs):
        length=sum(doc.values());score=0.
        for t in set(tokens(query)):
            tf=doc.get(t,0)
            if tf:score+=math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*tf*2.2/(tf+1.2*(.25+.75*length/avg))
        scored.append((score,r))
    scored.sort(key=lambda pair:(-pair[0],pair[1]['id']))
    selected=[];targets=set()
    for score,r in scored:
        key=json.dumps(r['target'],ensure_ascii=False)
        if key in targets:continue
        targets.add(key);selected.append({'id':r['id'],'audit_index':r['audit_index'],'score':score,'query':r['query'],'target':r['target']})
        if len(selected)==2:break
    assert len(selected)==2
    return selected

def main():
    audit=rows(HERE/'TRAIN_AUDIT80.frozen.jsonl');judged=[]
    for r in audit:
        verdict,dimensions,reason=EXCEPTIONS.get(r['audit_index'],('pass',[],'Current Codex inspected original question and every target; no definite F/S/C/U defect, retaining source ambiguity without assumed answers.'))
        judged.append({**r,'verdict':verdict,'failed_dimensions':dimensions,'reason':reason,'reviewer':'Current Codex query/target-only, same reviewer as prior training; not independent human gold.'})
    freeze(HERE/'TRAIN_AUDIT80.review.jsonl',judged,True)
    bank=[r for r in judged if r['verdict']=='pass']
    assert len(bank)==72
    freeze(HERE/'EXAMPLE_BANK72.frozen.jsonl',bank,True)
    write(OUT/'TRAIN_AUDIT_SUMMARY.json',{'n':80,'verdicts':dict(collections.Counter(r['verdict'] for r in judged)),
      'by_cohort':{c:dict(collections.Counter(r['verdict'] for r in judged if r['cohort']==c)) for c in sorted({r['cohort'] for r in judged})},
      'original_training_labels_changed':0,'in_context_bank_n':72,
      'limitation':'60 stratified random records20/hop plus20 purposive long/nested; not an unweighted random estimate of all509 label error or an independent gold audit. Questionable records excluded only from this example bank.'})
    original=(HERE.parent/'multigrain_4b_post_sft_opt_v1_20261018/prompts/b00_v12.txt').read_text(encoding='utf-8')
    prefix,examples=original.split('Input: ',1)
    blocks=['Input: '+s for s in examples.split('\nInput: ')]
    assert len(blocks)==5
    suffix=blocks[-1][blocks[-1].index('\nTreat the input'):]
    blocks[-1]=blocks[-1][:blocks[-1].index('\nTreat the input')]
    # Replace two demonstrations, keeping every rule and the last three demonstrations.
    def render(two):
        return prefix+''.join('Input: '+r['query']+'\nOutput: '+json.dumps(r['target'],ensure_ascii=False)+'\n\n' for r in two)+''.join(blocks[2:])+suffix
    static=[next(r for r in bank if r['audit_index']==i) for i in [62,47]]
    static_prompt=render(static)
    variants={'b00_v12':original,'m01_static_mu':static_prompt}
    for name,prompt in variants.items():
        path=HERE/'prompts'/(name+'.txt');path.parent.mkdir(exist_ok=True)
        if path.exists():assert path.read_text(encoding='utf-8')==prompt
        else:path.write_text(prompt,encoding='utf-8')
    (OUT/'STATIC_PROMPT_DIFF.txt').write_text(''.join(difflib.unified_diff(original.splitlines(True),static_prompt.splitlines(True),fromfile='b00_v12',tofile='m01_static_mu')),encoding='utf-8')
    specs={
      'b00_v12':{'mode':'fixed','prompt_sha256':sha(HERE/'prompts/b00_v12.txt'),'purpose':'Unchanged trained baseline.'},
      'm01_static_mu':{'mode':'fixed','prompt_sha256':sha(HERE/'prompts/m01_static_mu.txt'),'purpose':'Replace only the first two generic demonstrations with verified training62 joint-constraint island and47 institutional-role chain; no additional rules.'},
      'm02_adaptive_mu':{'mode':'adaptive_examples','purpose':'Same instruction and last three examples as m01; select first two from72 reviewed Mu training examples by fixed BM25 word+bigram similarity. No dataset metadata used for selection.'},
    }
    for phase in ['screen90','confirm120']:
        queries=rows(HERE/f'{phase}.questions.frozen.jsonl');payload=[]
        for q in queries:
            selected=retrieve(q['query'],bank);prompt=render(selected)
            payload.append({'id':q['id'],'index':q['index'],'prompt':prompt,'prompt_sha256':__import__('hashlib').sha256(prompt.encode()).hexdigest(),'examples':selected})
        freeze(HERE/f'{phase}.adaptive_prompts.frozen.jsonl',payload,True)
    freeze(HERE/'PROTOCOL_FROZEN.json',{
      'baseline':'checkpoint165/v12, same native template,greedy,seed1729,max_new_tokens1024',
      'variants':specs,'bank_sha256':sha(HERE/'EXAMPLE_BANK72.frozen.jsonl'),'n_screen':90,'n_confirmation':120,
      'screen':{'roster_sha256':sha(HERE/'screen90.questions.frozen.jsonl'),'macro':'Unweighted mean of three dataset pass rates; raw90 is Mu-enriched, not comparable to old balanced samples.',
        'candidate_gate':'MuSiQue passes strictly exceed baseline, macro pass rate exceeds baseline, and combined Hotpot+2Wiki regress by at most1 net pass.',
        'selection':'Eligible candidate with highest Mu pass count, then macro pass rate, then lower mean input tokens; otherwise retain baseline and do not run confirmation.'},
      'confirmation':{'roster_sha256':sha(HERE/'confirm120.questions.frozen.jsonl'),'gate':'One selected candidate vs baseline once on120 balanced queries. Require higher total pass count AND higher Mu pass count; otherwise no promotion. Report dataset regressions and uncertainty.'},
      'review':'Unchanged F/S/C/U, current Codex nonindependent query/output review; lock baseline judgments before reading candidate outputs. Identical arrays inherit judgments. All failures/uncertainty remain in denominator.',
      'no_runtime_semantic_gate_or_answer_feedback':True,'api_calls':0,'new_training':False,
      'record_every_payload':'Full actual prompt, selected example IDs, scores, raw output, native tokens, runtime, checkpoint identity.',
      'test_uses':'Old formal300 and prior confirmation90 are diagnostic history only; neither selects these candidates. New confirmation outputs cannot drive further edits in this iteration.',
    })
    write(OUT/'DECISIONS.json',{'decisions':[
      {'id':'B00','change':'Freeze same baseline weights, training3500, original prompts/reviews.'},
      {'id':'D01','change':'Explain development/test denominators; Mu contributes80% of6.67pp gap, not more four-hop share.'},
      {'id':'D02','change':'Audit60 stratified random +20 targeted training labels; quarantine4 fail4 uncertain from example bank only; no original label changes.'},
      {'id':'M01','change':'Replace two generic examples with two real reviewed Mu examples; rules unchanged.'},
      {'id':'M02','change':'Replace only same two slots with query-selected reviewed Mu examples; no new rules, model, training or answer evidence.'},
      {'id':'E01','change':'Freeze new screen90 and confirmation120 before inference, full ID/text/primitive exclusion. New native-test Mu source explicitly recorded.'},
    ],'api_calls':0})
    print(json.dumps({'bank_n':len(bank),'audit':collections.Counter(r['verdict'] for r in judged),'variants':list(specs),'protocol_sha256':sha(HERE/'PROTOCOL_FROZEN.json')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
