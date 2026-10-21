"""1500 formal questions only; remote scope is frozen-checkpoint decomposition."""
import json
import zipfile
from .common import ROOT, HERE, RUN, OUTPUT, read, rows, sha, save, log

def main():
    parent=RUN/'remote1500';package=parent/'remote_package'
    if (parent/'PACKAGE_AUDIT.json').exists():
        print(json.dumps(read(parent/'PACKAGE_AUDIT.json')));return
    questions=[];mapping={}
    for dataset in ['hotpotqa','2wiki','musique']:
        folder=HERE/'inputs_frozen'/dataset;frozen=read(folder/'SAMPLES_FROZEN.json')
        path=folder/'formal.questions.jsonl'
        if sha(path)!=frozen['files_sha256'][path.name]:raise ValueError('Frozen questions changed')
        cases=rows(path)
        if len(cases)!=500:raise ValueError('Expected 500 per dataset')
        for case in cases:
            remote_id=dataset+':'+case['qid']
            questions.append({'qid':remote_id,'question':case['question']})
            mapping[remote_id]={'dataset':dataset,'qid':case['qid']}
    if len(questions)!=1500 or len(mapping)!=1500:raise ValueError('Duplicate remote IDs')
    template=OUTPUT/'dev200_remote_20261014/remote_package';previous=read(template/'PACKAGE_MANIFEST.json')
    for name,expected in previous['files'].items():
        if sha(template/name)!=expected:raise ValueError('Frozen remote template differs')
    run_name='comprehensive_formal1500_20261015'
    cfg=read(template/'predict200.proposed.json')
    for key in ['input_file','output_dir']:cfg[key]=cfg[key].replace(previous['run_name'],run_name)
    cfg['scope']='three_dataset_formal1500_query_only_decomposition'
    package.mkdir(parents=True,exist_ok=True)
    combined=parent/'questions1500.query_only.jsonl'
    combined.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in questions),encoding='utf8')
    # Names are kept so the already-audited receiver/inference interface is compatible.
    (package/'queries200.query_only.jsonl').write_text(''.join(json.dumps({'id':r['qid'],'query':r['question']},ensure_ascii=False)+'\n' for r in questions),encoding='utf8')
    save(parent/'LOCAL_ID_MAPPING.json',mapping)
    save(package/'predict200.proposed.json',cfg)
    script=(template/'prepare_remote.py').read_text(encoding='utf8').replace('len(rows) != 200','len(rows) != 1500').replace('}) != 200','}) != 1500').replace('exactly200','exactly1500').replace('"inputs": 200','"inputs": 1500').replace('qdmr4b-dev200','qdmr4b-multidata1500')
    compile(script,'prepare_remote.py','exec')
    (package/'prepare_remote.py').write_text(script,encoding='utf8')
    task=f'''Only execute query decomposition for all1500 query-only records: HotpotQA500, 2Wiki500, MuSiQue500.
Use the existing Qwen3-4B checkpoint-000148 and unchanged instruction/template/decoding.
Dataset-prefixed IDs are opaque IDs: preserve each ID and query verbatim. Never remove the prefix.
Do not train, retrieve, answer questions, look up reference answers, or score semantics.
The historical filenames contain200; the audited manifest and records specify1500.

1. Unzip in a separate directory and verify PACKAGE_MANIFEST.json file hashes.
2. Run on an allocated GPU: python prepare_remote.py --submit
   If deployment moved, supply --route-root /absolute/route2_question_decomposer.
   Only deployment paths may change. BF16/SDPA, native nonthinking chat template,
   greedy decoding, max_new_tokens1024, num_beams1 and seed1729 are fixed.
3. Run directory: runs/{run_name}. Preserve all raw predictions, invalid JSON,
   generation errors, diagnostics, actual config/input and Slurm logs.
4. Before any resume reconcile scheduler/job state; do not launch duplicate jobs.
5. Return {run_name}_returned.zip beside this upload with the accepted archive layout:
   execution/package/*, runs/{run_name}/*, checkpoint_identity/checkpoint-000148/manifest.json,
   model_identity/BASE_MODEL_IDENTITY.json and RETURN_FILE_MANIFEST.json.
   The return manifest must cover all other files with path,bytes,sha256.
   Include predictions/predictions.jsonl, diagnostics.json, source and model identities.
   Return no weights or credentials. Local work owns all RAG and evaluation.
'''
    (package/'TASK_FOR_REMOTE_CODEX.txt').write_text(task,encoding='utf8')
    manifest={k:v for k,v in previous.items() if k!='files'}
    manifest.update(inputs=1500,queries=1500,scope=cfg['scope'],run_name=run_name,
      task='three_dataset_formal1500_frozen_checkpoint000148',local_source=str(combined.relative_to(ROOT)).replace('\\','/'),
      local_source_sha256=sha(combined),files={p.name:sha(p) for p in sorted(package.iterdir())})
    save(package/'PACKAGE_MANIFEST.json',manifest)
    archive=parent/'qdmr4b_multidata_formal1500_remote_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.iterdir()):z.write(path,path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==5
        for name,expected in manifest['files'].items():assert __import__('hashlib').sha256(z.read(name)).hexdigest()==expected
        assert [json.loads(s) for s in z.read('queries200.query_only.jsonl').decode('utf8').splitlines()]==[{'id':r['qid'],'query':r['question']} for r in questions]
    upload="$ErrorActionPreference = 'Stop'\n$multidataPackage = Join-Path $PSScriptRoot 'qdmr4b_multidata_formal1500_remote_package.zip'\n& scp $multidataPackage 'u7830488@cluster1.comp.anu.edu.au:/home/remote/u7830488/densex_workspace/'\nif ($LASTEXITCODE -ne 0) { throw \"scp failed: $LASTEXITCODE\" }\n"
    (parent/'UPLOAD.ps1').write_text(upload,encoding='utf8')
    result={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'questions':1500,
       'query_only':True,'remote_scope':'decomposition_only','model_instruction_decoding_unchanged':True,'remote_executed':False,'passed':True}
    save(parent/'PACKAGE_AUDIT.json',result);log('remote1500_package_ready',**result);print(json.dumps(result))

if __name__=='__main__':main()
