"""New-task LoRA SFT. Fixed full schedule, actual tail batches, resumable state."""
import argparse,copy,json,math,os,shutil,sys,time
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'runtime'))
from contract import batches,lr_factor,validate_config,sha,format_result
from model_runtime import (allocation_guard,load_libraries,load_model,seed_all,runtime_info,
    encode_supervised,encode_prompt,collate,StopRequest,get_rng,set_rng,json_dump,
    jsonl_append,object_hash,checkpoint_manifest,verify_checkpoint,read_jsonl,
    save_torch_state_with_sha256)
from adapter_runtime import (frozen_base_fingerprint,trainable_parameter_spec,adapter_state,
    validate_trainable_parameters,compare_adapter_state)

def encode_all(tokenizer,records,prompt):
    encoded=[encode_supervised(tokenizer,r,prompt,'qwen3') for r in records]
    assert max(r['total_tokens'] for r in encoded)<=1024,'Never truncate a training/dev row'
    return encoded

def loss_eval(model,tokenizer,data,torch):
    model.eval();total=0.;tokens=0
    with torch.inference_mode():
        for r in data:
            batch=collate([r],tokenizer.pad_token_id,torch)
            value=float(model(**batch,use_cache=False).loss)
            assert math.isfinite(value),(r['id'],value)
            total+=value*r['target_tokens'];tokens+=r['target_tokens']
    return {'target_token_loss':total/tokens,'target_tokens':tokens,'examples':len(data)}

def predict(model,tokenizer,records,prompt,out,torch):
    out.mkdir(exist_ok=True,parents=True);model.eval()
    for index,r in enumerate(records,1):
        p=out/f'{index:03d}.json'
        if p.exists():
            old=json.loads(p.read_text());assert old['id']==r['id'] and old['query']==r['query'];continue
        ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3')
        assert len(ids)+1024<=model.config.max_position_embeddings
        x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
        with torch.inference_mode():
            y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)
        generated=y[0,len(ids):].tolist();raw=tokenizer.decode(generated,skip_special_tokens=True)
        json_dump({'index':index,'id':r['id'],'query':r['query'],'dataset':r.get('dataset'),
                   'raw_output':raw,**format_result(raw),'input_token_ids':ids,'output_token_ids':generated,
                   'input_tokens':len(ids),'output_tokens':len(generated),'elapsed_seconds':time.monotonic()-started,
                   'truncated':len(generated)>=1024 and generated[-1]!=tokenizer.eos_token_id,'generation_error':None},p)

def save(model,tokenizer,opt,sched,state,c,out,torch,np,reason):
    destination=out/'checkpoints'/f'checkpoint-{state["global_step"]:06d}'
    if destination.exists():verify_checkpoint(destination);return destination
    assert frozen_base_fingerprint(model,torch)==state['frozen_base_fingerprint'],'Frozen base values changed'
    validate_trainable_parameters(model,torch)
    destination.parent.mkdir(exist_ok=True,parents=True)
    temp=destination.with_name('.'+destination.name+'.tmp-'+str(os.getpid()));temp.mkdir(exist_ok=False)
    model.save_pretrained(temp,safe_serialization=True,save_embedding_layers=False,selected_adapters=['default'])
    tokenizer.save_pretrained(temp)
    for name in ['LICENSE','LICENSE.txt','NOTICE','UPSTREAM_LICENSE.txt']:
        p=Path(c['model_path'])/name
        if p.exists():shutil.copyfile(p,temp/name)
    json_dump(state['base_model_identity'],temp/'BASE_MODEL_IDENTITY.json')
    json_dump(c,temp/'training_config.json');json_dump(state,temp/'trainer_state.json')
    (temp/'MODIFICATION_NOTICE.txt').write_text('Original Qwen3-4B-Instruct-2507 + newly initialized ordinary LoRA. v12 self-contained supporting-question arrays. Codex-reviewed supervision, no new human gold. Frozen base is required; old adapters were not initialization.\n')
    rng=get_rng(torch,np)
    written=save_torch_state_with_sha256({'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng,'trainer':state,'optimizer_parameter_spec':trainable_parameter_spec(model)},temp/'training_state.pt',torch)
    manifest=checkpoint_manifest(temp,write_time_hashes={'training_state.pt':written})
    json_dump(manifest,temp/'manifest.json');json_dump({'global_step':state['global_step'],'complete':True,'manifest_sha256':manifest['manifest_sha256'],'reason':reason},temp/'COMPLETE.json')
    os.rename(temp,destination);set_rng(rng,torch,np)
    json_dump({'global_step':state['global_step'],'checkpoint':str(destination)},out/'LATEST_CHECKPOINT.json')
    return destination

def run(c,mode,resume):
    validate_config(c);out=Path(c['output_root'])/mode;out.mkdir(exist_ok=True,parents=True)
    import fcntl
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not resume and (out/'metrics.jsonl').exists():raise RuntimeError('Existing training state; explicit verified resume required')
        if mode=='train':
            smoke=json.loads((Path(c['output_root'])/'smoke/SMOKE_VERIFIED.json').read_text())
            assert smoke['passed'] and smoke['config_sha256']==object_hash(c),'Smoke is missing or belongs to other configuration'
        allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        torch,np,transformers=load_libraries(allocation);seed_all(c['seed'],torch,np,c)
        stop=StopRequest();stop.install()
        prompt=Path(c['prompt_file']).read_text(encoding='utf-8').strip()
        master=read_jsonl(Path(c['data_dir'])/'MASTER_3500.jsonl');dev=read_jsonl(Path(c['data_dir'])/'dev150.master.jsonl');probe=read_jsonl(Path(c['data_dir'])/'train_probe150.master.jsonl')
        assert len(master)==3500 and len(dev)==len(probe)==150
        plan=batches();assert len(plan)==330
        fingerprint=object_hash({'config':c,'code_sha256':sha(__file__),'contract_sha256':sha(PACKAGE/'code/contract.py'),'plan':plan})
        mc=dict(c)
        if resume:verify_checkpoint(resume);mc['adapter_path']=str(resume)
        model,tokenizer,report=load_model(mc,torch,transformers,train=True)
        json_dump(report,out/'MODEL_REPORT.json');json_dump(runtime_info(torch,transformers,allocation,c),out/'RUNTIME.json')
        json_dump(c,out/'CONFIG.json')
        train_enc=encode_all(tokenizer,master,prompt);dev_enc=encode_all(tokenizer,dev,prompt);probe_enc=encode_all(tokenizer,probe,prompt)
        json_dump({'train':len(train_enc),'dev':len(dev_enc),'probe':len(probe_enc),'max_train_tokens':max(r['total_tokens'] for r in train_enc),'max_dev_tokens':max(r['total_tokens'] for r in dev_enc),'silent_truncation':0,'eos_supervised':True,'system_user_padding_masked':True},out/'ENCODING.json')
        params=[p for p in model.parameters() if p.requires_grad]
        o=c['optimizer'];opt=torch.optim.AdamW(params,lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
        state={'global_step':0,'consumed_examples':0,'fingerprint':fingerprint,'mode':mode,'best_dev_loss':None,'bad_checks':0,'base_model_identity':report['base_identity'],'frozen_base_fingerprint':frozen_base_fingerprint(model,torch),'loss_history':[]}
        initial_adapter={k:v.detach().cpu().clone() for k,v in adapter_state(model).items()} if mode=='smoke' else None
        if resume:
            payload=torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=False)
            state=payload['trainer'];assert state['fingerprint']==fingerprint and state['mode']==mode
            assert payload['optimizer_parameter_spec']==trainable_parameter_spec(model)
            assert state['frozen_base_fingerprint']==frozen_base_fingerprint(model,torch)
            opt.load_state_dict(payload['optimizer']);sched.load_state_dict(payload['scheduler']);set_rng(payload['rng'],torch,np)
        end=2 if mode=='smoke' else 330;early=False;checkpoint=None
        model.train()
        for position in range(state['global_step'],end):
            item=plan[position];batch_rows=[train_enc[i] for i in item['indices']]
            tokens=sum(r['target_tokens'] for r in batch_rows);opt.zero_grad(set_to_none=True);loss_sum=0.;started=time.monotonic();lr=opt.param_groups[0]['lr']
            for r in batch_rows:
                batch=collate([r],tokenizer.pad_token_id,torch);result=model(**batch,use_cache=False);loss=result.loss
                assert torch.isfinite(loss),('nonfinite_loss',r['id'])
                (loss*(r['target_tokens']/tokens)).backward();loss_sum+=float(loss.detach())*r['target_tokens']
                del batch,result,loss
            for name,p in model.named_parameters():
                if not p.requires_grad:assert p.grad is None,('base_gradient',name)
                elif p.grad is not None:assert torch.isfinite(p.grad).all(),('nonfinite_gradient',name)
            grad=float(torch.nn.utils.clip_grad_norm_(params,o['clip']));assert math.isfinite(grad)
            opt.step();sched.step();state['global_step']=position+1;state['consumed_examples']+=len(batch_rows)
            jsonl_append({'step':position+1,'epoch':item['epoch'],'actual_examples':len(batch_rows),'consumed_examples':state['consumed_examples'],'loss':loss_sum/tokens,'target_tokens':tokens,'gradient_norm':grad,'learning_rate':lr,'next_learning_rate':opt.param_groups[0]['lr'],'seconds':time.monotonic()-started,'ids':[master[i]['id'] for i in item['indices']]},out/'metrics.jsonl')
            print(json.dumps({'step':position+1,'epoch':item['epoch'],'examples':len(batch_rows),'loss':loss_sum/tokens}),flush=True)
            step=position+1
            if mode=='train' and step in [55,110,165,220,275,330]:
                rng=get_rng(torch,np);d=loss_eval(model,tokenizer,dev_enc,torch);t=loss_eval(model,tokenizer,probe_enc,torch)
                val=d['target_token_loss'];best=state['best_dev_loss']
                if best is None or val<best-.001:state['best_dev_loss']=val;state['bad_checks']=0
                else:state['bad_checks']+=1
                result={'step':step,'dev':d,'train_probe':t,'gap':val-t['target_token_loss'],'bad_checks':state['bad_checks']}
                state['loss_history'].append(result);jsonl_append(result,out/'validation.jsonl')
                if step in [110,165,220,275,330]:predict(model,tokenizer,dev,prompt,out/'dev_predictions'/f'checkpoint-{step:06d}',torch)
                set_rng(rng,torch,np);model.train();early=step>=220 and state['bad_checks']>=2
                checkpoint=save(model,tokenizer,opt,sched,state,c,out,torch,np,'scheduled')
            if stop.requested or early or (mode=='smoke' and step==2):
                checkpoint=save(model,tokenizer,opt,sched,state,c,out,torch,np,'signal' if stop.requested else 'early_stop' if early else 'smoke')
                break
        if mode=='smoke':
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            verify_checkpoint(checkpoint);saved=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
            assert any(not torch.equal(initial_adapter[k],v.detach().cpu()) for k,v in adapter_state(model).items()),'No adapter update'
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if p.requires_grad:p.zero_()
            set_peft_model_state_dict(model,saved,adapter_name='default');compare_adapter_state(adapter_state(model),saved,torch)
            assert frozen_base_fingerprint(model,torch)==state['frozen_base_fingerprint']
            json_dump({'passed':True,'steps':2,'examples':64,'config_sha256':object_hash(c),'checkpoint_verified':True,'adapter_changed_and_reloaded_exactly':True,'base_values_unchanged':True,'finite_loss_and_gradients':True,'formal_initialization':'separate fresh Python process, original base, new adapter and optimizer'},out/'SMOKE_VERIFIED.json')
        json_dump({'status':'smoke_passed' if mode=='smoke' else 'interrupted' if stop.requested else 'early_stopped' if early else 'completed','global_step':state['global_step'],'consumed_examples':state['consumed_examples'],'checkpoint':str(checkpoint) if checkpoint else None,'semantic_checkpoint_selection':'pending_local_dev_review','api_calls':0},out/'STATE.json')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--mode',choices=['smoke','train'],required=True);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    run(json.loads(a.config.read_text()),a.mode,a.resume)
