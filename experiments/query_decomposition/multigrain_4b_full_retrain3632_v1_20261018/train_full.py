"""Fresh-base full SFT, reusing the proven runtime/checkpoint implementation."""
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-package',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--mode',choices=['smoke','train'],required=True)
    ap.add_argument('--resume',type=Path)
    a=ap.parse_args()
    sys.path[:0]=[str(a.base_package/'code'),str(a.base_package/'runtime')]
    from contract import validate_config, sha
    from model_runtime import (allocation_guard,load_libraries,load_model,seed_all,runtime_info,
      encode_supervised,collate,StopRequest,get_rng,set_rng,json_dump,jsonl_append,
      object_hash,verify_checkpoint,read_jsonl)
    from adapter_runtime import frozen_base_fingerprint,trainable_parameter_spec,adapter_state,validate_trainable_parameters
    spec=importlib.util.spec_from_file_location('original_full_trainer',a.base_package/'code/train.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    original=json.loads((a.base_package/'config.json').read_text());validate_config(original)
    c=json.loads((a.experiment/'config.json').read_text())
    plan=json.loads((a.experiment/'PLAN_FROZEN.json').read_text())
    sampling=json.loads((a.experiment/'BATCHES_FROZEN.json').read_text())
    assert sha(a.experiment/'config.json')==plan['config_sha256']
    assert sha(a.experiment/'BATCHES_FROZEN.json')==plan['batches_sha256']
    assert c['adapter_path'] is None and c['train_records']==3632 and c['scheduler_horizon']==228
    assert c['optimizer']['lr']==5e-5 and len(sampling)==228
    assert sha(a.experiment/'v12.txt')==c['prompt_sha256']
    for name,h in c['data_sha256'].items(): assert sha(a.experiment/name)==h,name
    # All model/runtime dependencies are recorded; new code never edits the old package.
    dependency_names=['code/train.py','code/contract.py','runtime/model_runtime.py','runtime/adapter_runtime.py']
    fingerprint=object_hash({'config':c,'batches':sampling,'code_sha256':sha(__file__),
      'dependencies':{n:sha(a.base_package/n) for n in dependency_names}})
    out=Path(c['output_root'])/a.mode;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'metrics.jsonl').exists() and not a.resume: raise RuntimeError('Existing run; no implicit restart.')
        if a.mode=='train':
            smoke=json.loads((Path(c['output_root'])/'smoke/SMOKE_VERIFIED.json').read_text())
            assert smoke['passed'] and smoke['fingerprint']==fingerprint
        allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        torch,np,transformers=load_libraries(allocation);seed_all(c['seed'],torch,np,c)
        stop=StopRequest();stop.install()
        json_dump({'status':'initializing','mode':a.mode,'initial_adapter':None,'api_calls':0},out/'STATE.json')
        mc=dict(c)
        if a.resume: verify_checkpoint(a.resume);mc['adapter_path']=str(a.resume)
        model,tokenizer,report=load_model(mc,torch,transformers,train=True)
        if not a.resume: assert report['adapter']['loaded_adapter'] is None
        json_dump(report,out/'MODEL_REPORT.json');json_dump(c,out/'CONFIG.json')
        json_dump(runtime_info(torch,transformers,allocation,c),out/'RUNTIME.json')
        prompt=(a.experiment/'v12.txt').read_text(encoding='utf-8')
        master=read_jsonl(a.experiment/'DATA3632.frozen.jsonl')
        dev=read_jsonl(a.experiment/'dev150.master.jsonl');probe=read_jsonl(a.experiment/'train_probe150.master.jsonl')
        queries=read_jsonl(a.experiment/'dev60.questions.frozen.jsonl')
        assert len(master)==3632 and len(dev)==len(probe)==150 and len(queries)==60
        enc=[[encode_supervised(tokenizer,r,prompt,'qwen3') for r in rr] for rr in [master,dev,probe]]
        maximum=max(r['total_tokens'] for rr in enc for r in rr);assert maximum<=1024,maximum
        train_enc,dev_enc,probe_enc=enc
        json_dump({'train_examples':3632,'max_total_tokens':maximum,'max_train_tokens':max(r['total_tokens'] for r in train_enc),
          'limit':1024,'truncated_records':0,'prompt_padding_masked':True,'eos_supervised':True},out/'ENCODING.json')
        pars=[p for p in model.parameters() if p.requires_grad];o=c['optimizer']
        opt=torch.optim.AdamW(pars,lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])
        # LambdaLR initialization is update1; after each step it prepares the next update.
        def factor(position):
            update=position+1
            return update/10 if update<=10 else max(0.,(228-update)/218)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,factor)
        state={'global_step':0,'consumed_examples':0,'fingerprint':fingerprint,'mode':a.mode,
          'base_model_identity':report['base_identity'],'frozen_base_fingerprint':frozen_base_fingerprint(model,torch),'loss_history':[]}
        initial={k:v.detach().cpu().clone() for k,v in adapter_state(model).items()}
        if a.resume:
            payload=torch.load(a.resume/'training_state.pt',map_location='cpu',weights_only=False)
            state=payload['trainer'];assert state['fingerprint']==fingerprint and state['mode']==a.mode
            assert payload['optimizer_parameter_spec']==trainable_parameter_spec(model)
            assert state['frozen_base_fingerprint']==frozen_base_fingerprint(model,torch)
            opt.load_state_dict(payload['optimizer']);sched.load_state_dict(payload['scheduler']);set_rng(payload['rng'],torch,np)
            del payload
        checkpoint=None

        def scheduled_check(step):
            rng=get_rng(torch,np)
            if not any(r['step']==step for r in state['loss_history']):
                d=helper.loss_eval(model,tokenizer,dev_enc,torch);t=helper.loss_eval(model,tokenizer,probe_enc,torch)
                v={'step':step,'dev':d,'train_probe':t,'gap':d['target_token_loss']-t['target_token_loss']}
                state['loss_history'].append(v);jsonl_append(v,out/'validation.jsonl')
            cp=helper.save(model,tokenizer,opt,sched,state,c,out,torch,np,'scheduled')
            if step in [114,171,228]:
                p=out/'dev60_predictions'/f'checkpoint-{step:06d}'
                ident={'checkpoint_manifest_sha256':json.loads((cp/'COMPLETE.json').read_text())['manifest_sha256'],
                  'prompt_sha256':c['prompt_sha256'],'query_sha256':c['data_sha256']['dev60.questions.frozen.jsonl'],
                  'seed':1729,'do_sample':False,'num_beams':1,'max_new_tokens':1024,'api_calls':0}
                if (p/'IDENTITY.json').exists():assert json.loads((p/'IDENTITY.json').read_text())==ident
                else:json_dump(ident,p/'IDENTITY.json')
                model.gradient_checkpointing_disable();model.config.use_cache=True;seed_all(1729,torch,np,c)
                helper.predict(model,tokenizer,queries,prompt,p,torch)
            set_rng(rng,torch,np);model.config.use_cache=False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
            return cp

        if a.resume and state['global_step'] in [57,114,171,228]: checkpoint=scheduled_check(state['global_step'])
        model.train();end=2 if a.mode=='smoke' else 228
        for position in range(state['global_step'],end):
            item=sampling[position];indices=item['indices'];rr=[train_enc[i] for i in indices]
            assert len(rr) in [16,32] and len(set(indices))==len(indices)
            tokens=sum(r['target_tokens'] for r in rr);opt.zero_grad(set_to_none=True)
            started=time.monotonic();lr=opt.param_groups[0]['lr'];weighted=0.
            for r in rr:
                batch=collate([r],tokenizer.pad_token_id,torch);result=model(**batch,use_cache=False);loss=result.loss
                assert torch.isfinite(loss),r['id'];(loss*(r['target_tokens']/tokens)).backward()
                weighted+=float(loss.detach())*r['target_tokens'];del batch,result,loss
            for name,p in model.named_parameters():
                if not p.requires_grad: assert p.grad is None,name
                elif p.grad is not None: assert torch.isfinite(p.grad).all(),name
            grad=float(torch.nn.utils.clip_grad_norm_(pars,o['clip']));assert math.isfinite(grad)
            opt.step();sched.step();step=position+1;state['global_step']=step;state['consumed_examples']+=len(rr)
            jsonl_append({'step':step,'epoch':item['epoch'],'actual_examples':len(rr),'consumed_examples':state['consumed_examples'],
              'loss':weighted/tokens,'target_tokens':tokens,'gradient_norm':grad,'learning_rate':lr,
              'next_learning_rate':opt.param_groups[0]['lr'],'seconds':time.monotonic()-started,
              'ids':[master[i]['id'] for i in indices]},out/'metrics.jsonl')
            json_dump({'status':'training','mode':a.mode,'global_step':step,'planned_updates':end,
              'consumed_examples':state['consumed_examples'],'api_calls':0},out/'STATE.json')
            print(json.dumps({'mode':a.mode,'step':step,'loss':weighted/tokens,'learning_rate':lr}),flush=True)
            if a.mode=='train' and step in [57,114,171,228]: checkpoint=scheduled_check(step)
            if stop.requested or (a.mode=='smoke' and step==2):
                checkpoint=helper.save(model,tokenizer,opt,sched,state,c,out,torch,np,'signal' if stop.requested else 'smoke')
                break
        assert frozen_base_fingerprint(model,torch)==state['frozen_base_fingerprint']
        changed=any(not torch.equal(initial[k],v.detach().cpu()) for k,v in adapter_state(model).items())
        assert changed or a.resume,'No adapter update'
        if a.mode=='smoke' and not stop.requested:
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            from adapter_runtime import compare_adapter_state
            verify_checkpoint(checkpoint)
            saved=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
            with torch.no_grad():
                for p in pars:p.zero_()
            set_peft_model_state_dict(model,saved,adapter_name='default');compare_adapter_state(adapter_state(model),saved,torch)
            assert frozen_base_fingerprint(model,torch)==state['frozen_base_fingerprint']
            json_dump({'passed':True,'fingerprint':fingerprint,'steps':2,'examples':64,'adapter_changed_and_reloaded':True,
              'base_unchanged':True,'formal_process':'Independent new Python process from raw base and fresh LoRA'},out/'SMOKE_VERIFIED.json')
        json_dump({'status':'interrupted' if stop.requested else 'smoke_passed' if a.mode=='smoke' else 'completed',
          'global_step':state['global_step'],'consumed_examples':state['consumed_examples'],
          'planned_examples':64 if a.mode=='smoke' else 7264,'checkpoint':str(checkpoint),'base_unchanged':True,
          'initial_adapter':None,'semantic_selection':'pending_dev60_review','api_calls':0},out/'STATE.json')

if __name__=='__main__':main()
