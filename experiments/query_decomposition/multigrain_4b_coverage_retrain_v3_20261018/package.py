"""Create a byte-preserving POSIX-manifest bundle after data/plan validation."""
import datetime,json,zipfile
from prepare import HERE,ROOT,OUT,rows,sha,write,freeze

def main():
    p=json.loads((HERE/'PLAN_FROZEN.json').read_text(encoding='utf-8'));assert sha(HERE/'DATA3632.frozen.jsonl')==p['data_sha256'];assert sha(HERE/'v12.txt')==p['prompt_sha256']
    files=[f for f in HERE.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='MANIFEST.json']
    manifest={'files':[{'path':f.relative_to(HERE).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(files)]};freeze(HERE/'MANIFEST.json',manifest)
    z=OUT/'coverage_retrain_v3_package_v001.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
        for f in files+[HERE/'MANIFEST.json']:archive.write(f,f.relative_to(HERE).as_posix())
    write(OUT/'PACKAGE_BUILD.json',{'zip':str(z.relative_to(ROOT)),'sha256':sha(z),'files':len(files)+1,'frozen_prompt_exact_bytes':True,'api_calls':0})
    write(OUT/'CHANGELOG.json',{'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decisions':[
        {'id':'V3-01','change':'Retain original165 and prior t02 checkpoints unchanged. Warm-start each new arm from same t02; native template,v12 and greedy unchanged.'},
        {'id':'V3-02','change':'Quarantine4 previously uncertain existing labels from both arms. Keep8 prior repaired labels. Base3496.'},
        {'id':'V3-03','change':'Reserve dev60/confirm120 before labeling/training; Mu from unused primitive-disjoint native train, not official test. New official-test branch2 families exhausted by historical exposure.'},
        {'id':'V3-04','change':'Initial attempt restricted new training to prior180 candidates and could not fill isolation quotas; stopped before any output or model inference. Expand candidate search to remaining native train under same isolation and cap3; original180 retained.'},
        {'id':'V3-05','change':'Author/review120 real inputs, quarantine24 unresolved queries, accept96. No new native4hop2 survives protected validation primitive isolation; report this deficit explicitly instead of adding overlap.'},
        {'id':'V3-06','change':'Add40 clearly marked fictional training-only examples across10 skills, including minimal direction/time/owner pairs. Not represented as real Mu or40 independent skills.'},
        {'id':'V3-07','change':'Second-pass label corrections and every quarantine reason in LABEL_REVIEW_FROZEN and label batch files; single Codex self-review, not independent gold.'},
        {'id':'V3-08','change':'Two fixed40-step controls, freshAdamW,lr2e-5,warmup4,seed2718,batch32,maxsequence1536,no truncation. Same640 nonMu+320 oldMu; remaining320 are oldMu vs240 new-native+80 synthetic. Only end checkpoint is considered.'},
        {'id':'V3-09','change':'One dev60 selection under predeclared Mu/overall/control/F-S gates; only an eligible nominee gets fresh120 paired confirmation. No repeated tuning on validation or old scores.'}
      ],'api_calls':0,'original_training_and_frozen_scores_changed':False})
    print(json.dumps({'zip':str(z.resolve()),'sha256':sha(z),'files':len(files)+1}))

if __name__=='__main__':main()
