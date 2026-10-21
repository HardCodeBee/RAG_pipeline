"""Freeze an unseen, query-only300 and stage the unchanged base4B/v12 protocol."""
from collections import Counter
import hashlib,json,re,shutil,sys,unicodedata,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREVIOUS=HERE.parent/'multigrain_4b_prompt_train3500_v1_20261017'
OUT=ROOT/'outputs/query_decomposition'/HERE.name
PACKAGE=HERE/'remote'
SEED='20261017-frozen-before-after-eval300-v12-01'
QUOTAS={'hotpotqa':{'bridge':70,'comparison':30},'2wiki':{'comparison':25,'inference':25,'compositional':25,'bridge_comparison':25},'musique':{'2-hop':35,'3-hop':35,'4-hop':30}}
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(s):return hashlib.sha256((SEED+s).encode()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip(' \t\n?!？！.')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_rows(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8')

def freeze_queries():
    if (HERE/'questions300.frozen.jsonl').exists():return rows(HERE/'questions300.frozen.jsonl')
    meta=json.loads((PREVIOUS/'SELECTION_FROZEN.json').read_text(encoding='utf-8'))
    paths={ROOT/r['path'] for r in meta['excluded_files']}
    for pattern in ['*questions*.jsonl','cases.frozen.jsonl','*queries*.jsonl']:
        paths.update((ROOT/'experiments/query_decomposition').rglob(pattern))
    train=ROOT/'outputs/query_decomposition'/PREVIOUS.name/'training/MASTER_3500.jsonl'
    paths.update([train,PREVIOUS/'EXCLUDE_FROM_TRAINING.jsonl',PREVIOUS.parent/'multigrain_decomposer_fresh_eval_v1_20261017/hard_cases_for_4b/CANDIDATES.jsonl'])
    ids=set();texts=set();exclusions=[]
    for p in sorted(paths):
        if not p.exists():continue
        count=0
        for r in rows(p):
            if not isinstance(r,dict):continue
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q));count+=1
            for key in ['qid','id']:
                if isinstance(r.get(key),str):ids.add(r[key]);ids.add(r[key].split(':',1)[-1])
            src=r.get('source')
            if isinstance(src,dict):
                for key in ['id','native_id']:
                    if isinstance(src.get(key),str):ids.add(src[key]);ids.add(src[key].split(':',1)[-1])
        if count:exclusions.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'query_records':count})
    # The original project split/family assignments remain protected, even if a
    # prior question file omitted the actual source IDs.
    old=ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data'
    splits=json.loads((old/'split_manifest.json').read_text(encoding='utf-8'))
    for r in splits['records']:
        if r['project_assignment'] in {'project_dev','project_test','reserved_test_family','reserved_dev_family'}:
            ids.add(r.get('source',{}).get('id',''))
    screening=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_FROZEN.json'
    ids.update(json.loads(screening.read_text(encoding='utf-8'))['ids'])
    # Where MuSiQue exposes compositional IDs, also protect shared primitive
    # question components with the3500 actual training examples.
    training=rows(train);mu_parts=set()
    for r in training:
        if 'musique' in r['source']['dataset'].lower():
            rid=r['source']['id'].split(':')[-1]
            if '__' in rid:mu_parts.update(rid.split('__',1)[1].split('_'))
    native=ROOT/'data/multihop_comprehensive_v1_20261015'
    files={'hotpotqa':ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl','2wiki':native/'2wiki/native/dev.json','musique':native/'musique/native/musique_ans_v1.0_dev.jsonl'}
    pool=[];seen=set();reasons=Counter()
    for ds,p in files.items():
        source=json.loads(p.read_text(encoding='utf-8')) if ds=='2wiki' else rows(p)
        for r in source:
            qid=r.get('_id',r.get('id'));q=r['question'];key=norm(q)
            if qid in ids or ds+':'+qid in ids or key in texts or key in seen:reasons['known_id_or_text']+=1;continue
            if ds=='musique' and '__' in qid and set(qid.split('__',1)[1].split('_'))&mu_parts:reasons['shared_musique_training_component']+=1;continue
            kind=str(len(r['question_decomposition']))+'-hop' if ds=='musique' else r['type']
            pool.append({'id':ds+':'+qid,'qid':qid,'dataset':ds,'query':q,'type':kind,'source_split':'dev'});seen.add(key)
    selected=[];availability={}
    for ds,kinds in QUOTAS.items():
        for kind,n in kinds.items():
            candidates=sorted([r for r in pool if r['dataset']==ds and r['type']==kind],key=lambda r:rank(r['id']))
            availability[ds+'/'+kind]=len(candidates)
            assert len(candidates)>=n,(ds,kind,len(candidates),n)
            selected.extend(candidates[:n])
    selected.sort(key=lambda r:rank('order:'+r['id']))
    for i,r in enumerate(selected,1):r['index']=i
    assert len(selected)==len({r['id'] for r in selected})==len({norm(r['query']) for r in selected})==300
    assert not {norm(r['query']) for r in selected}&{norm(r['query']) for r in training}
    write_rows(HERE/'questions300.frozen.jsonl',selected)
    write(HERE/'EVALUATION_FROZEN.json',{'seed':SEED,'quotas':QUOTAS,'role':'fixed-before-after-evaluation300; not training, prompt selection or checkpoint tuning','original_model':'Qwen/Qwen3-4B-Instruct-2507 no adapter','prompt':'v12 exact frozen','prompt_sha256':sha(PREVIOUS/'FINAL_PROMPT.txt'),'training_sha256':sha(train),'selection_uses_answers_or_predictions':False,'exclusion_sources':exclusions,'excluded_ids':len(ids),'excluded_texts':len(texts),'excluded_musique_training_components':len(mu_parts),'candidate_exclusions':dict(reasons),'remaining_availability':availability,'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files.values()],'limitations':'Known normalized text/ID and MuSiQue component checks; not complete paraphrase or pretraining decontamination. Evaluation reviewed by current Codex, not independent human gold.'})
    write_rows(HERE/'EXCLUDE_FROM_TRAINING.jsonl',[{'qid':r['qid'],'dataset':r['dataset'],'query':r['query'],'role':'frozen_eval300_do_not_train_or_tune'} for r in selected])
    return selected

def stage():
    selected=freeze_queries();PACKAGE.mkdir(exist_ok=True)
    replacements={'queries100.query_only.jsonl':'queries300.query_only.jsonl','base4b_prompt100_return.zip':'base4b_eval300_return.zip','base4b-prompt100':'base4b-eval300'}
    for name in ['code/contracts.py','code/run_remote.py','prepare_remote.py','runtime/model_runtime.py','runtime/adapter_runtime.py','runtime/run_stage.slurm']:
        text=(PREVIOUS/'remote'/name).read_text(encoding='utf-8')
        if name in ['code/contracts.py','code/run_remote.py','prepare_remote.py']:
            for a,b in replacements.items():text=text.replace(a,b)
            if name=='code/contracts.py':
                text=text.replace('ARMS = ("v6", "v8", "v11", "v12")','ARMS = ("v12",)').replace('== 100','== 300').replace('"planned": 100','"planned": 300').replace('== 400','== 300').replace('"planned": 400','"planned": 300')
            else:
                text=text.replace('400','300')
                if name=='code/run_remote.py':text=text.replace('index % 4','index % len(ARMS)')
                else:text=text.replace('100 shared queries,4 prompts,300 attempts','300 queries,1 frozen prompt,300 attempts')
        dest=PACKAGE/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8',newline='\n')
    (PACKAGE/'prompts').mkdir(exist_ok=True);shutil.copyfile(PREVIOUS/'FINAL_PROMPT.txt',PACKAGE/'prompts/v12.txt')
    shutil.copyfile(PREVIOUS/'remote/BASE_MODEL_IDENTITY.expected.json',PACKAGE/'BASE_MODEL_IDENTITY.expected.json')
    config=json.loads((PREVIOUS/'remote/base_runtime.json').read_text(encoding='utf-8'));config['arms']=['v12'];write(PACKAGE/'base_runtime.json',config)
    write_rows(PACKAGE/'queries300.query_only.jsonl',[{'id':r['id'],'query':r['query']} for r in selected])
    write(PACKAGE/'PROTOCOL_FROZEN.json',{'task':'base4b_frozen_eval300','queries':300,'planned_outputs':300,'arms':['v12'],'adapter_loaded':False,'training_started':False,'api_calls':0,'query_sha256':sha(PACKAGE/'queries300.query_only.jsonl'),'evaluation_metadata_sha256':sha(HERE/'EVALUATION_FROZEN.json'),'prompt_sha256':sha(PACKAGE/'prompts/v12.txt'),'review':'Original query and literal output; F/S/C/U with uncertainty/errors/truncation in full denominator. No answer retrieval or final RAG QA.','same_generation_settings_as_prompt100':True})
    files={str(p.relative_to(PACKAGE)).replace('\\','/'):sha(p) for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json'}
    write(PACKAGE/'PACKAGE_MANIFEST.json',{'files':files,'expected_outputs':300,'api_calls':0,'adapter_loaded':False})
    OUT.mkdir(parents=True,exist_ok=True);archive=OUT/'base4b_eval300_remote_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'PACKAGE_MANIFEST.json']:z.write(PACKAGE/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    report={'queries':len(selected),'datasets':dict(Counter(r['dataset'] for r in selected)),'archive':str(archive),'sha256':sha(archive),'api_calls':0,'training_started':False,'status':'built_not_submitted'}
    write(OUT/'PACKAGE_BUILD.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':stage()
