"""Freeze a full new-LoRA SFT independently of the warm-start confirmation."""
import collections, hashlib, json, random, re, shutil, unicodedata
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/query_decomposition' / HERE.name
PRIOR = HERE.parent / 'multigrain_4b_coverage_retrain_v3_20261018'
ORIGINAL = HERE.parent / 'multigrain_4b_sft3500_v1_20261017/remote_v003'
REMOTE = '/home/projects/u7830488/' + HERE.name

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def norm(q): return ' '.join(unicodedata.normalize('NFKC', q).casefold().split()).strip(' ?!.')
def freeze(p, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    if p.exists(): assert p.read_bytes() == data, str(p)
    else: p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
def copy_frozen(source, destination):
    if destination.exists(): assert destination.read_bytes() == source.read_bytes()
    else: destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, destination)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['DATA3632.frozen.jsonl', 'dev60.questions.frozen.jsonl', 'confirm120.questions.frozen.jsonl', 'v12.txt']:
        copy_frozen(PRIOR/name, HERE/name)
    for name in ['dev150.master.jsonl', 'train_probe150.master.jsonl']:
        copy_frozen(ORIGINAL/'data'/name, HERE/name)
    train = rows(HERE/'DATA3632.frozen.jsonl'); dev = rows(HERE/'dev60.questions.frozen.jsonl'); test = rows(HERE/'confirm120.questions.frozen.jsonl')
    assert len(train) == 3632 and len({r['id'] for r in train}) == 3632
    for r in train:
        assert isinstance(r['target'], list) and all(isinstance(q, str) and q.strip() and not re.search(r'#\d+', q) for q in r['target'])
    assert not ({norm(r['query']) for r in train} & {norm(r['query']) for r in dev+test})
    assert not ({r['id'] for r in train} & {r['id'] for r in dev+test})
    assert not ({norm(r['query']) for r in train} & {norm(r['query']) for r in rows(HERE/'dev150.master.jsonl')})
    plan=[]
    for epoch in range(2):
        order=list(range(3632)); random.Random(1729+epoch).shuffle(order)
        batches=[order[i:i+32] for i in range(0,len(order),32)]
        assert len(batches)==114 and len(batches[-1])==16
        assert sorted(i for b in batches for i in b)==list(range(3632))
        plan.extend({'epoch':epoch+1,'indices':b} for b in batches)
    freeze(HERE/'BATCHES_FROZEN.json', plan)
    c=json.loads((ORIGINAL/'config.json').read_text(encoding='utf-8'))
    c.update(adapter_path=None, data_dir=REMOTE, output_root=REMOTE+'/results', prompt_file=REMOTE+'/v12.txt',
             train_records=3632, dev_records=150, epochs_max=2, scheduler_horizon=228,
             scheduler='linear_warmup10_fixed228', tail_policy='16_actual_each_epoch_no_drop_no_replay',
             checkpoint_selection=['dev60_complete_pass','worst_dataset_pass','earliest_step'])
    c['optimizer']['lr']=5e-5
    c['data_sha256']={n:sha(HERE/n) for n in ['DATA3632.frozen.jsonl','dev150.master.jsonl','train_probe150.master.jsonl','dev60.questions.frozen.jsonl','confirm120.questions.frozen.jsonl']}
    freeze(HERE/'config.json', c)
    decision={
      'run':HERE.name, 'stage':'Full retraining; distinct from completed 40-update continuation trials.',
      'initialization':'Immutable original Qwen3-4B-Instruct-2507 base + fresh ordinary LoRA. No old adapter/optimizer initialization.',
      'data':{'total':3632,'reused_old':3496,'new_native_musique':96,'new_explicit_synthetic':40,
              'old_data_changes':'Prior eight documented label fixes retained; four uncertain old records quarantined, not declared factually false. Original3500 immutable.',
              'native_quarantine':24,'unchanged_known_issue':'One old DROP exact-goal repetition remains deferred, not silently changed after freezing.',
              'review':'Codex authoring plus same-reviewer second pass; not independent human gold.',
              'sha256':sha(HERE/'DATA3632.frozen.jsonl')},
      'training':{'epochs_cap':2,'updates_per_epoch':114,'updates_cap':228,'micro_batch':1,'effective_batch':32,
                  'epoch_tail':16,'examples_per_epoch':3632,'total_exposures':7264,'sampler':'Uniform shuffled without replacement within each epoch; no focused oversampling.',
                  'seed':1729,'lr':5e-5,'warmup_updates':10,'lr_schedule':'update1..10 = peak*(update/10); update11..228 = peak*(228-update)/218; fixed horizon.',
                  'optimizer':'Fresh AdamW beta .9/.999, eps1e-8, weight_decay.01; gradient clip1.',
                  'lora':c['lora'],'bf16':True,'max_length':1024,'truncate':False,
                  'reason':'All old/new records reused. Old peak near1.5 epochs and later overfit motivate cap2 and lower LR; not a controlled data-only causal comparison.',
                  'checkpoint_updates':[57,114,171,228],'generated_dev60_updates':[114,171,228],
                  'loss_monitor':'Same frozen old dev150 and train_probe150 at57/114/171/228; diagnostic only, no test-based scheduler changes.',
                  'smoke':'Two updates in separate fresh process; verify changed/reloaded adapter and unchanged base. Formal run starts fresh again.'},
      'selection':{'source':'Only fixed dev60, not confirm120; train all fixed228 before semantic choice.',
                   'eligibility':'Total>=47/60; Mu>=15/20; Hot+Wiki>=31/40; Mu F/S failure rows<=4. If no checkpoint qualifies retain frozen baseline, no forced promotion.',
                   'ranking':'Highest total pass, then highest worst-dataset pass, then earliest update.',
                   'rubric':'Original F/S/C/U; uncertainty is nonpass; exact arrays inherit locked parent. Same Codex, not independent.',
                   'frozen_parent':'T02:47/60 total, Mu15/20. Pilot R02:51/60 total, Mu17/20 reported alongside, never overwritten.'},
      'confirmation':{'roster_sha256':sha(HERE/'confirm120.questions.frozen.jsonl'),'n':120,
                      'already_queued_pilot_jobs':['154345','154347'],
                      'at_design_freeze':'No new confirmation model outputs viewed. Existing jobs pending at2026-10-18T04:35:37Z.',
                      'policy':'Keep pilot confirmation jobs. Freeze this recipe before viewing outputs; score one dev-selected full-SFT model once on the same fixed120, no reselection or subsequent changes from these scores.',
                      'mu_scope':'Unused native-train held-out with primitive exclusion, not an official MuSiQue dev/test estimate.',
                      'measure':'Decomposition pass, not downstream QA F1; report full denominators and paired uncertainty.'},
      'api_calls':0,'prompt':'Exact frozen v12 bytes, native nonthinking greedy evaluation; no postprocessing.',
      'preservation':'Original3500, checkpoint165, T02, pilotR01/R02 all read-only; new isolated directory.',
      'batches_sha256':sha(HERE/'BATCHES_FROZEN.json'),'config_sha256':sha(HERE/'config.json')}
    freeze(HERE/'PLAN_FROZEN.json',decision)
    freeze(OUT/'PLAN_FROZEN.json',decision)
    stamp=HERE/'FREEZE_TIME.json'
    if not stamp.exists(): freeze(stamp,{'utc':datetime.now(timezone.utc).isoformat(),'confirmation_outputs_viewed':0})
    freeze(OUT/'DATA_REUSE_AUDIT.json',{'count':len(train),'unique_ids':len({r['id'] for r in train}),
      'new_native':96,'new_synthetic':40,'old_reused':3496,'uniform_epoch_complete':True,
      'train_dev60_and_confirm120_query_disjoint':True,'train_old_dev150_query_disjoint':True,
      'dataset_counts':dict(collections.Counter(r['source']['dataset'] for r in train)),
      'file_sha256':c['data_sha256'],'old_files_modified':False})
    print(json.dumps({'training_records':len(train),'updates':len(plan),'data_sha256':sha(HERE/'DATA3632.frozen.jsonl'),'api_calls':0}))

if __name__=='__main__': main()
