"""Conditional, fixed20-update exposure control from a read-only parent adapter."""
import argparse,gc,json,math,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path[:0]=[str(a.base_package/'code'),str(a.base_package/'runtime')]
    from contract import validate_config,sha,format_result
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,json_dump,jsonl_append,object_hash,verify_checkpoint,checkpoint_manifest,encode_supervised,encode_prompt,collate,StopRequest
    from adapter_runtime import frozen_base_fingerprint,validate_trainable_parameters,adapter_state,compare_adapter_state
    import fcntl
    plan=json.loads((a.experiment/'continuation/PLAN_FROZEN.json').read_text());gate=json.loads((a.experiment/'TRAINING_LAUNCH_GATE.json').read_text())
    assert gate['all_three_prompt_candidates_failed_screen_gate'] is True
    c=json.loads((a.base_package/'config.json').read_text());validate_config(c)
    parent=a.base_package/'results/train/checkpoints/checkpoint-000165';parent_identity=verify_checkpoint(parent)
    baseline=json.loads((a.experiment/'BASELINE_FROZEN.json').read_text());assert parent_identity['manifest_sha256']==baseline['checkpoint_manifest_sha256']
    data_path=a.experiment/'continuation/MASTER3500_SCOPE8.candidate.jsonl';assert sha(data_path)==plan['train3500_candidate_sha256']
    train=[json.loads(s) for s in data_path.read_text().splitlines() if s.strip()];queries=[json.loads(s) for s in (a.experiment/'screen90.questions.frozen.jsonl').read_text().splitlines() if s.strip()];assert len(train)==3500 and len(queries)==90
    prompt=(a.experiment/'prompts/b00_v12.txt').read_text(encoding='utf-8');assert sha(a.experiment/'prompts/b00_v12.txt')==baseline['prompt_sha256']
    allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch,np,transformers=load_libraries(allocation)
    a.output.mkdir(parents=True,exist_ok=True);stop=StopRequest();stop.install()
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,sampling in plan['sampling_plans'].items():
            out=a.output/name
            if (out/'STATE.json').exists() and json.loads((out/'STATE.json').read_text())['status']=='completed':continue
            assert not out.exists(),'No implicit restart/resume; preserve partial trial and stage a new recorded directory.'
            out.mkdir();seed_all(plan['seed'],torch,np,c);mc=dict(c);mc['adapter_path']=str(parent)
            model,tokenizer,report=load_model(mc,torch,transformers,train=True);validate_trainable_parameters(model,torch)
            frozen=frozen_base_fingerprint(model,torch);initial={k:v.detach().cpu().clone() for k,v in adapter_state(model).items()}
            needed=sorted(set(i for b in sampling for i in b));encoded={i:encode_supervised(tokenizer,train[i],prompt,'qwen3') for i in needed}
            assert max(r['total_tokens'] for r in encoded.values())<=1024
            parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=plan['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
            json_dump({'parent_manifest':parent_identity['manifest_sha256'],'plan_sha256':sha(a.experiment/'continuation/PLAN_FROZEN.json'),'sampler':name,'sample_counts':plan['sampling_counts'][name],'optimizer_reset':True,'original_base_identity':report['base_identity']},out/'IDENTITY.json')
            model.train()
            for step,indices in enumerate(sampling,1):
                assert len(indices)==32 and len(set(indices))==32
                lr=plan['lr']*min(1.,step/plan['warmup_updates'])
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad(set_to_none=True);total=sum(encoded[i]['target_tokens'] for i in indices);weighted=0.;started=time.monotonic()
                for i in indices:
                    r=encoded[i];batch=collate([r],tokenizer.pad_token_id,torch);result=model(**batch,use_cache=False);loss=result.loss
                    assert torch.isfinite(loss),(train[i]['id'],'nonfinite_loss')
                    (loss*(r['target_tokens']/total)).backward();weighted+=float(loss.detach())*r['target_tokens'];del batch,result,loss
                for param_name,p in model.named_parameters():
                    if not p.requires_grad:assert p.grad is None,param_name
                    elif p.grad is not None:assert torch.isfinite(p.grad).all(),param_name
                grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.));assert math.isfinite(grad);opt.step()
                jsonl_append({'step':step,'learning_rate':lr,'examples':32,'target_tokens':total,'loss':weighted/total,'gradient_norm':grad,'seconds':time.monotonic()-started,'ids':[train[i]['id'] for i in indices]},out/'metrics.jsonl')
                json_dump({'status':'training','completed_updates':step,'planned_updates':20,'api_calls':0},out/'STATE.json');print(json.dumps({'trial':name,'step':step,'loss':weighted/total}),flush=True)
                if stop.requested:raise RuntimeError('Stop requested; retain partial trial, do not score unfinished20-update weights')
            assert frozen_base_fingerprint(model,torch)==frozen
            current=adapter_state(model);assert any(not torch.equal(initial[k],v.detach().cpu()) for k,v in current.items());del initial
            cp=out/'checkpoint-final';cp.mkdir();model.save_pretrained(cp,safe_serialization=True,save_embedding_layers=False,selected_adapters=['default']);tokenizer.save_pretrained(cp)
            config={**c,'parent_checkpoint':str(parent),'parent_manifest':parent_identity['manifest_sha256'],
                    'adapter_path':str(parent),'new_updates':20,'continuation_plan':plan,'sampler':name,
                    'data_dir':str(data_path.parent),'data_sha256':{data_path.name:sha(data_path)},
                    'output_root':str(out),'epochs_max':None,'warmup_steps':2,'scheduler_horizon':20,
                    'scheduler':'warmup2_then_constant','optimizer':{**c['optimizer'],'lr':plan['lr']}}
            json_dump(report['base_identity'],cp/'BASE_MODEL_IDENTITY.json');json_dump(config,cp/'training_config.json')
            (cp/'MODIFICATION_NOTICE.txt').write_text('Bounded20-update continuation of the frozen165 ordinary LoRA, using8 corrected candidate labels. Original base and parent adapter unchanged. Same sampling budget; no optimizer-resume or new-data claim.\n')
            manifest=checkpoint_manifest(cp);json_dump(manifest,cp/'manifest.json');json_dump({'complete':True,'manifest_sha256':manifest['manifest_sha256'],'new_updates':20},cp/'COMPLETE.json');verify_checkpoint(cp)
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            saved=load_file(str(cp/'adapter_model.safetensors'),device='cpu');compare_adapter_state(adapter_state(model),saved,torch);set_peft_model_state_dict(model,saved,adapter_name='default');compare_adapter_state(adapter_state(model),saved,torch)
            del saved,opt;model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
            pred=out/'predictions';pred.mkdir();identity={'checkpoint_manifest_sha256':manifest['manifest_sha256'],'prompt_sha256':baseline['prompt_sha256'],'query_sha256':sha(a.experiment/'screen90.questions.frozen.jsonl'),'do_sample':False,'num_beams':1,'max_new_tokens':1024,'seed':1729,'api_calls':0};json_dump(identity,pred/'IDENTITY.json');seed_all(1729,torch,np,c)
            for index,r in enumerate(queries,1):
                ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3');x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
                with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                ts=y[0,len(ids):].tolist();raw=tokenizer.decode(ts,skip_special_tokens=True)
                json_dump({'index':index,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':raw,**format_result(raw),'actual_prompt':prompt,'input_token_ids':ids,'input_tokens':len(ids),'output_token_ids':ts,'output_tokens':len(ts),'truncated':len(ts)>=1024 and ts[-1]!=tokenizer.eos_token_id,'generation_error':None,'elapsed_seconds':time.monotonic()-started,'identity_sha256':object_hash(identity)},pred/f'{index:03d}.json')
                json_dump({'status':'scoring','completed_cases':index,'planned_cases':90,'new_updates':20,'api_calls':0},out/'STATE.json')
            assert frozen_base_fingerprint(model,torch)==frozen;assert verify_checkpoint(parent)['manifest_sha256']==parent_identity['manifest_sha256']
            json_dump({'status':'completed','completed_updates':20,'completed_cases':90,'base_unchanged':True,'parent_unchanged':True,'checkpoint_manifest_sha256':manifest['manifest_sha256'],'api_calls':0},out/'STATE.json')
            del model,parameters,encoded,tokenizer,current;gc.collect();torch.cuda.empty_cache()
        json_dump({'status':'completed','trials':list(plan['sampling_plans']),'api_calls':0},a.output/'STATE.json')

if __name__=='__main__':main()
