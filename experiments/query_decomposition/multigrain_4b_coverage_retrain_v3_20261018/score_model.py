"""Score one verified adapter with the byte-exact baseline prompt on frozen queries."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-package',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha',required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--queries-sha',required=True);p.add_argument('--prompt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.base_package/'code'),str(a.base_package/'runtime')]
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,json_dump,object_hash,verify_checkpoint,encode_prompt
    from contract import sha,validate_config,format_result
    c=json.loads((a.base_package/'config.json').read_text());validate_config(c);assert sha(a.queries)==a.queries_sha;assert sha(a.prompt)=='508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8'
    manifest=verify_checkpoint(a.checkpoint);assert manifest['manifest_sha256']==a.checkpoint_sha
    allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch,np,transformers=load_libraries(allocation);seed_all(1729,torch,np,c);c['adapter_path']=str(a.checkpoint)
    model,tokenizer,report=load_model(c,torch,transformers,train=False);model.eval();prompt=a.prompt.read_text(encoding='utf-8');qq=[json.loads(s) for s in a.queries.read_text().splitlines() if s.strip()]
    a.output.mkdir(exist_ok=True,parents=True)
    import fcntl
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={'checkpoint_manifest_sha256':a.checkpoint_sha,'query_sha256':sha(a.queries),'prompt_sha256':sha(a.prompt),'do_sample':False,'num_beams':1,'max_new_tokens':1024,'seed':1729,'api_calls':0}
        if (a.output/'IDENTITY.json').exists():assert json.loads((a.output/'IDENTITY.json').read_text())==identity
        else:json_dump(identity,a.output/'IDENTITY.json')
        for i,r in enumerate(qq,1):
            path=a.output/f'{i:03d}.json'
            if path.exists():
                previous=json.loads(path.read_text());assert previous['id']==r['id'] and previous['identity_sha256']==object_hash(identity);continue
            ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3');assert len(ids)+1024<=model.config.max_position_embeddings
            x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
            with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
            ts=y[0,len(ids):].tolist();raw=tokenizer.decode(ts,skip_special_tokens=True)
            json_dump({'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':raw,**format_result(raw),'actual_prompt':prompt,'input_token_ids':ids,'input_tokens':len(ids),'output_token_ids':ts,'output_tokens':len(ts),'truncated':len(ts)>=1024 and ts[-1]!=tokenizer.eos_token_id,'generation_error':None,'elapsed_seconds':time.monotonic()-started,'identity_sha256':object_hash(identity)},path)
            json_dump({'status':'running','completed':i,'planned':len(qq),'api_calls':0},a.output/'STATE.json');print(json.dumps({'variant':a.output.name,'index':i}),flush=True)
        json_dump({'status':'completed','completed':len(qq),'planned':len(qq),'api_calls':0},a.output/'STATE.json')

if __name__=='__main__':main()
