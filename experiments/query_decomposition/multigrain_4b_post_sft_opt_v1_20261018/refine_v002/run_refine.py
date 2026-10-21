"""One bounded revision of query-only drafts using the same frozen 4B adapter."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path[:0]=[str(a.package/'code'),str(a.package/'runtime')]
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,runtime_info,json_dump,object_hash,verify_checkpoint,StopRequest,encode_prompt
    from contract import sha,format_result,validate_config
    plan=json.loads((a.experiment/'PLAN_FROZEN.json').read_text());assert plan['n']==44
    qpath=a.experiment/'queries44.frozen.jsonl';assert sha(qpath)==plan['queries_sha256'] and sha(a.experiment/'refine.txt')==plan['prompt_sha256']
    queries=[json.loads(x) for x in qpath.read_text().splitlines() if x.strip()];assert len(queries)==44
    assert all(set(x)=={'index','id','dataset','query','draft_questions'} for x in queries)
    c=json.loads((a.package/'config.json').read_text());validate_config(c)
    cp=a.package/'results/train/checkpoints/checkpoint-000165';m=verify_checkpoint(cp);assert m['manifest_sha256']==plan['checkpoint_manifest_sha256']
    alloc=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch,np,transformers=load_libraries(alloc);seed_all(1729,torch,np,c);c['adapter_path']=str(cp)
    model,tokenizer,report=load_model(c,torch,transformers,train=False);model.eval();a.output.mkdir(exist_ok=True,parents=True)
    import fcntl
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        json_dump(report,a.output/'MODEL_REPORT.json');json_dump(runtime_info(torch,transformers,alloc,c),a.output/'RUNTIME.json')
        identity={'plan_sha256':sha(a.experiment/'PLAN_FROZEN.json'),'checkpoint_manifest_sha256':m['manifest_sha256'],'prompt_sha256':plan['prompt_sha256'],'queries_sha256':plan['queries_sha256'],'do_sample':False,'num_beams':1,'max_new_tokens':1024,'seed':1729,'additional_calls_per_query':1,'api_calls':0}
        if (a.output/'IDENTITY.json').exists():assert json.loads((a.output/'IDENTITY.json').read_text())==identity
        else:json_dump(identity,a.output/'IDENTITY.json')
        prompt=(a.experiment/'refine.txt').read_text(encoding='utf-8');stop=StopRequest();stop.install()
        for i,r in enumerate(queries,1):
            path=a.output/f'{i:03d}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['id']==r['id'] and old['identity_sha256']==object_hash(identity);continue
            if stop.requested:break
            user='Original question:\n'+r['query']+'\n\nDraft supporting questions:\n'+json.dumps(r['draft_questions'],ensure_ascii=False)
            ids=encode_prompt(tokenizer,user,prompt,'qwen3');assert len(ids)+1024<=model.config.max_position_embeddings
            x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
            try:
                with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                tokens=y[0,len(ids):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=True)
                result={'raw_output':raw,**format_result(raw),'output_token_ids':tokens,'output_tokens':len(tokens),'truncated':len(tokens)>=1024 and tokens[-1]!=tokenizer.eos_token_id,'generation_error':None}
            except Exception as ex:result={'raw_output':None,**format_result(None),'output_token_ids':[],'output_tokens':0,'truncated':False,'generation_error':type(ex).__name__+': '+str(ex)}
            json_dump({'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'draft_questions':r['draft_questions'],'input_token_ids':ids,'input_tokens':len(ids),'elapsed_seconds':time.monotonic()-started,'identity_sha256':object_hash(identity),**result},path)
            print(json.dumps({'index':i,'error':result['generation_error']}),flush=True)
            if result['generation_error']:break
        n=len([p for p in a.output.glob('*.json') if p.stem.isdigit()]);json_dump({'planned':44,'completed':n,'status':'completed' if n==44 else 'interrupted_or_error','api_calls':0},a.output/'STATE.json')

if __name__=='__main__':main()
