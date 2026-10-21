"""Fixed-checkpoint example-only comparison, with complete per-query payloads."""
import argparse,collections,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--phase',choices=['screen90','confirm120'],required=True);ap.add_argument('--variants',nargs='+',required=True);a=ap.parse_args()
    sys.path[:0]=[str(a.package/'code'),str(a.package/'runtime')]
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,runtime_info,json_dump,object_hash,verify_checkpoint,StopRequest,encode_prompt,encode_supervised
    from contract import sha,format_result,validate_config,batches
    protocol=json.loads((a.experiment/'PROTOCOL_FROZEN.json').read_text());baseline=json.loads((a.experiment/'BASELINE_FROZEN.json').read_text())
    c=json.loads((a.package/'config.json').read_text());validate_config(c)
    cp=a.package/'results/train/checkpoints/checkpoint-000165';manifest=verify_checkpoint(cp);assert manifest['manifest_sha256']==baseline['checkpoint_manifest_sha256']
    qpath=a.experiment/(a.phase+'.questions.frozen.jsonl');role='screen' if a.phase=='screen90' else 'confirmation'
    assert sha(qpath)==protocol[role]['roster_sha256']
    queries=[json.loads(s) for s in qpath.read_text().splitlines() if s.strip()];planned=protocol['n_screen' if role=='screen' else 'n_confirmation'];assert len(queries)==planned
    payloads={r['id']:r for r in [json.loads(s) for s in (a.experiment/(a.phase+'.adaptive_prompts.frozen.jsonl')).read_text().splitlines() if s.strip()]}
    allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch,np,transformers=load_libraries(allocation);seed_all(1729,torch,np,c);c['adapter_path']=str(cp)
    model,tokenizer,report=load_model(c,torch,transformers,train=False);model.eval();a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        json_dump(report,a.output/'MODEL_REPORT.json');json_dump(runtime_info(torch,transformers,allocation,c),a.output/'RUNTIME.json')
        baseline_prompt=(a.experiment/'prompts/b00_v12.txt').read_text(encoding='utf-8')
        if not (a.output/'TRAIN_TARGET_TOKEN_EXPOSURE.json').exists():
            train=[json.loads(s) for s in (Path(c['data_dir'])/'MASTER_3500.jsonl').read_text().splitlines() if s.strip()]
            lengths=[encode_supervised(tokenizer,r,baseline_prompt,'qwen3')['target_tokens'] for r in train]
            for upto,label in [(110,'one_full_epoch'),(165,'actual_checkpoint165_prefix')]:
                indices=[i for batch in batches()[:upto] for i in batch['indices']];groups=collections.defaultdict(lambda:{'examples':0,'target_tokens':0})
                for i in indices:
                    ds=train[i]['source']['dataset'];ds='musique' if ds.lower()=='musique' else ds
                    groups[ds]['examples']+=1;groups[ds]['target_tokens']+=lengths[i]
                json_dump({'selection':label,'consumed_examples':len(indices),'total_target_tokens':sum(lengths[i] for i in indices),'by_source':dict(groups),'caution':'Supervised-token exposure is not a measured share of gradients or causal effect.'},a.output/('TRAIN_EXPOSURE_'+label+'.json'))
            json_dump({'n':3500,'total_supervised_tokens':sum(lengths),'api_calls':0},a.output/'TRAIN_TARGET_TOKEN_EXPOSURE.json')
        stop=StopRequest();stop.install()
        for name in a.variants:
            assert name in protocol['variants']
            if stop.requested:break
            seed_all(1729,torch,np,c);out=a.output/name;out.mkdir(exist_ok=True)
            spec=protocol['variants'][name]
            fixed=None if spec['mode']=='adaptive_examples' else (a.experiment/'prompts'/(name+'.txt')).read_text(encoding='utf-8')
            if fixed is not None:assert sha(a.experiment/'prompts'/(name+'.txt'))==spec['prompt_sha256']
            identity={'variant':name,'phase':a.phase,'queries_sha256':sha(qpath),'protocol_sha256':sha(a.experiment/'PROTOCOL_FROZEN.json'),'checkpoint_manifest_sha256':manifest['manifest_sha256'],'adaptive_payloads_sha256':sha(a.experiment/(a.phase+'.adaptive_prompts.frozen.jsonl')) if fixed is None else None,'prompt_sha256':spec.get('prompt_sha256'),'do_sample':False,'num_beams':1,'max_new_tokens':1024,'seed':1729,'api_calls':0}
            if (out/'IDENTITY.json').exists():assert json.loads((out/'IDENTITY.json').read_text())==identity
            else:json_dump(identity,out/'IDENTITY.json')
            for i,r in enumerate(queries,1):
                path=out/f'{i:03d}.json'
                if path.exists():
                    previous=json.loads(path.read_text());assert previous['id']==r['id'] and previous['identity_sha256']==object_hash(identity);continue
                if stop.requested:break
                prompt=fixed if fixed is not None else payloads[r['id']]['prompt'];prompt_hash=__import__('hashlib').sha256(prompt.encode()).hexdigest()
                if fixed is None:assert prompt_hash==payloads[r['id']]['prompt_sha256']
                ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3');assert len(ids)+1024<=model.config.max_position_embeddings
                x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
                try:
                    with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                    tokens=y[0,len(ids):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=True)
                    result={'raw_output':raw,**format_result(raw),'output_token_ids':tokens,'output_tokens':len(tokens),'truncated':len(tokens)>=1024 and tokens[-1]!=tokenizer.eos_token_id,'generation_error':None}
                except Exception as ex:result={'raw_output':None,**format_result(None),'output_token_ids':[],'output_tokens':0,'truncated':False,'generation_error':type(ex).__name__+': '+str(ex)}
                json_dump({'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'actual_prompt':prompt,'actual_prompt_sha256':prompt_hash,'selected_examples':payloads[r['id']]['examples'] if fixed is None else None,'input_token_ids':ids,'input_tokens':len(ids),'elapsed_seconds':time.monotonic()-started,'identity_sha256':object_hash(identity),**result},path)
                json_dump({'status':'running','completed':i,'planned':planned,'api_calls':0},out/'STATE.json')
                print(json.dumps({'variant':name,'index':i,'error':result['generation_error']}),flush=True)
                if result['generation_error']:stop.requested=True;break
            n=len(list(out.glob('[0-9][0-9][0-9].json')))
            json_dump({'status':'completed' if n==planned else 'interrupted_or_error','completed':n,'planned':planned,'api_calls':0},out/'STATE.json')
        json_dump({'status':'interrupted_or_error' if stop.requested else 'completed','variants':a.variants,'planned_each':planned,'api_calls':0},a.output/'STATE.json')

if __name__=='__main__':main()
