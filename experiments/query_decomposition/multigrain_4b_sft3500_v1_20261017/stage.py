"""Stage the approved new-task training bundle without mutating prior runs."""
import json,shutil,zipfile
from pathlib import Path
from reaudit import HERE,ROOT,PREVIOUS,SOURCE,OUT,sha,write
REMOTE='/home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package'

def main():
    p=HERE/'remote';(p/'data').mkdir(exist_ok=True)
    for name in ['MASTER_3500.jsonl','train3500.messages.jsonl','dev150.master.jsonl','dev150.messages.jsonl','train_probe150.master.jsonl','train_probe150.messages.jsonl']:
        shutil.copyfile(OUT/'data_v002'/name,p/'data'/name)
    shutil.copyfile(PREVIOUS/'FINAL_PROMPT.txt',p/'v12.txt')
    shutil.copyfile(PREVIOUS/'remote/BASE_MODEL_IDENTITY.expected.json',p/'BASE_MODEL_IDENTITY.json')
    shutil.copyfile(PREVIOUS/'remote/runtime/run_stage.slurm',p/'runtime/run_stage.slurm')
    old=ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer'
    original=json.loads((old/'configs/train.json').read_text(encoding='utf-8'))
    c={k:original[k] for k in ['model_path','expected_base_parameters','expected_adapter_parameters','template_policy','lora','seed','allow_tf32','attn_implementation','runtime_device'] if k in original}
    c.update(base_identity_file=REMOTE+'/BASE_MODEL_IDENTITY.json',runtime_device='cuda',adapter_path=None,
             data_dir=REMOTE+'/data',output_root=REMOTE+'/results',prompt_file=REMOTE+'/v12.txt',
             prompt_sha256=sha(p/'v12.txt'),data_sha256={x.name:sha(x) for x in (p/'data').glob('*.jsonl')},
             train_records=3500,dev_records=150,micro_batch=1,max_length=1024,epochs_max=3,
             warmup_steps=10,scheduler_horizon=330,gradient_checkpointing=True,
             max_new_tokens=1024,num_beams=1,optimizer={'lr':.0001,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'clip':1.},
             scheduler='fixed330_linear_warmup10_decay',tail_policy='12_actual_no_replay_no_drop',
             checkpoint_selection=['dev_complete_pass','worst_dataset_pass','earliest_step'])
    write(p/'config.json',c)
    validation=json.loads((OUT/'data_v002/VALIDATION.json').read_text());checks=json.loads((OUT/'LOCAL_CONTRACT_CHECKS.json').read_text());baseline=json.loads((OUT/'baseline300/SUMMARY.json').read_text())
    assert validation['train_records']==3500 and validation['dev_records']==150 and checks['passed'] and baseline['complete']
    write(p/'EXECUTION_READY.json',{'local_data_validation_passed':True,'local_contract_checks_passed':True,'baseline300_semantic_review_completed':True,'data_validation_sha256':sha(OUT/'data_v002/VALIDATION.json'),'baseline_review_sha256':sha(OUT/'baseline300/REVIEW.progress.jsonl'),'train_revision':'v002','authorization':'User approved plan and execution; separate smoke2 followed by fresh original-base LoRA SFT','api_calls':0})
    write(p/'RUNTIME_PROVENANCE.json',{'model_runtime.py':{'copied_from':str((old/'work/model_runtime.py').relative_to(ROOT)),'sha256':sha(p/'runtime/model_runtime.py'),'unchanged':sha(p/'runtime/model_runtime.py')==sha(old/'work/model_runtime.py')},'adapter_runtime.py':{'copied_from':str((old/'work/adapter_runtime.py').relative_to(ROOT)),'sha256':sha(p/'runtime/adapter_runtime.py'),'unchanged':sha(p/'runtime/adapter_runtime.py')==sha(old/'work/adapter_runtime.py')},'new_code':'New array task loader, actual tail sampler, fixed horizon, minimum-two-epoch loss stop, dev prediction retention and new isolated output. Old trainer/model artifacts unchanged.'})
    files={x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.name!='PACKAGE_MANIFEST.json'}
    write(p/'PACKAGE_MANIFEST.json',{'files':files,'remote_root':REMOTE,'api_calls':0})
    archive=OUT/'sft3500_v002_v12_remote_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'PACKAGE_MANIFEST.json']:z.write(p/name,name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write(OUT/'PACKAGE_BUILD.json',{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'remote_root':REMOTE,'members':len(files)+1,'status':'built_not_submitted','api_calls':0})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2))
if __name__=='__main__':main()
