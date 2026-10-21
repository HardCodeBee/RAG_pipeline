"""Package the reviewed training data and necessary audit trail, without weights."""
import hashlib,json,zipfile
from training_sources import HERE,OUT,rows,sha,write

def main():
    validation=json.loads((OUT/'FINAL_VALIDATION.json').read_text(encoding='utf-8'))
    report=json.loads((OUT/'DATA_REPORT.json').read_text(encoding='utf-8'))
    assert report['status']=='ready_for_training_setup_not_trained' and report['accepted_count']==3500
    assert not report['mechanical_errors'] and validation['truncated_records']==0
    assert len(rows(OUT/'train3500.messages.jsonl'))==3500
    files={}
    for name in ['train3500.messages.jsonl','MASTER_3500.jsonl','DATA_REPORT.json','FINAL_VALIDATION.json','EXCLUDE_FUTURE_EVALUATION.jsonl','TOKEN_LENGTHS.jsonl']:
        files[name]=OUT/name
    for name in ['REVIEWED_ALL.jsonl','OLD_NEW_DIFF.jsonl','QUARANTINE.jsonl','INPUTS_FROZEN.json','LABELING_INPUTS.frozen.jsonl','OLD_DRAFTS.unreviewed.jsonl','NEW_2WIKI_DRAFTS.unreviewed.jsonl','QUALITY_SAMPLE300.frozen.jsonl','QUALITY_SAMPLE300.review.jsonl','NEW_SELECTION.json']:
        files['audit/'+name]=OUT/name
    for name in ['FINAL_PROMPT.txt','DATASET_README.txt','BASELINE_REPORT.txt','FINAL_PROMPT_SELECTION.json','DECISIONS.txt','EXCLUDE_FROM_TRAINING.jsonl']:
        files[name]=HERE/name
    for name in ['BASELINE_COMPARISON.json','BASELINE_SEMANTIC_REVIEW.jsonl']:
        files[name]=OUT.parent/name
    files['audit/baseline_questions100.frozen.jsonl']=HERE/'questions100.frozen.jsonl'
    files['audit/baseline_predictions400.raw.jsonl']=OUT.parent/'returned_20261017T094753Z/predictions.jsonl'
    for p in (HERE/'training_reviews').glob('*.json'):files['audit/training_reviews/'+p.name]=p
    for name in ['training_sources.py','training_drafts.py','new_training_labels.py','author_bridge_review.py','audit_training_progress.py','assemble_training.py','validate_training.py','review_baseline.py','BASELINE_AUTHORED.json']:
        files['audit/code/'+name]=HERE/name
    for name in ['tokenizer.json','tokenizer_config.json']:files['base_tokenizer/'+name]=OUT/'base_tokenizer'/name
    entries=[{'path':name,'size':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(files.items())]
    manifest=OUT/'ARTIFACT_MANIFEST.json';write(manifest,{'format':'training-only message export plus separated audit; no model weights, training launch, answers or passages','accepted_records':3500,'files':entries})
    files['ARTIFACT_MANIFEST.json']=manifest
    package=OUT.parent/'train3500_v12_reviewed_20261017.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,name)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(files)
        for name,path in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha(path),name
        assert len(z.read('train3500.messages.jsonl').decode().splitlines())==3500
    result={'package':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'entries':len(files),'zip_crc_and_all_content_hashes_verified':True,'train_records':3500,'api_calls':0,'training_started':False}
    write(OUT/'PACKAGE_BUILD_TRAINING.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
