"""Allocated-GPU final evaluation after dev-only checkpoint selection."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--queries',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path[:0]=[str(a.package/'code'),str(a.package/'runtime')]
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,runtime_info,json_dump,object_hash,verify_checkpoint,StopRequest,encode_prompt
    from contract import sha,format_result,validate_config
    sel=json.loads(a.selection.read_text());assert sel['selection_uses_final_eval300'] is False
    c=json.loads((a.package/'config.json').read_text());validate_config(c)
    cp=a.package/'results/train/checkpoints'/f'checkpoint-{sel["selected"]["step"]:06d}'
    allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch,np,transformers=load_libraries(allocation);seed_all(1729,torch,np,c)
    manifest=verify_checkpoint(cp);assert manifest['manifest_sha256']==sel['selected']['checkpoint_manifest_sha256']
    expected=sel.get('final_queries_sha256');assert expected and sha(a.queries)==expected,'Must freeze test input identity before submission'
    queries=[json.loads(s) for s in a.queries.read_text().splitlines() if s.strip()];assert len(queries)==300 and len({r['id'] for r in queries})==300
    a.output.mkdir(exist_ok=True,parents=True)
    import fcntl
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c['adapter_path']=str(cp);model,tokenizer,report=load_model(c,torch,transformers,train=False);model.eval()
        json_dump(report,a.output/'MODEL_REPORT.json');json_dump(runtime_info(torch,transformers,allocation,c),a.output/'RUNTIME.json')
        identity={'selection_sha256':sha(a.selection),'queries_sha256':sha(a.queries),'checkpoint_manifest_sha256':manifest['manifest_sha256'],'prompt_sha256':c['prompt_sha256'],'seed':1729,'max_new_tokens':1024,'num_beams':1,'do_sample':False,'api_calls':0}
        if (a.output/'IDENTITY.json').exists():assert json.loads((a.output/'IDENTITY.json').read_text())==identity
        else:json_dump(identity,a.output/'IDENTITY.json')
        stop=StopRequest();stop.install();prompt=Path(c['prompt_file']).read_text(encoding='utf-8')
        for i,r in enumerate(queries,1):
            path=a.output/f'{i:03d}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['id']==r['id'] and old['identity_sha256']==object_hash(identity);continue
            if stop.requested:break
            ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3');assert len(ids)+1024<=model.config.max_position_embeddings
            x=torch.tensor([ids],device='cuda',dtype=torch.long);start=time.monotonic()
            try:
                with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                tokens=y[0,len(ids):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=True)
                result={'raw_output':raw,**format_result(raw),'output_token_ids':tokens,'output_tokens':len(tokens),'truncated':len(tokens)>=1024 and tokens[-1]!=tokenizer.eos_token_id,'generation_error':None}
            except Exception as e:result={'raw_output':None,**format_result(None),'output_token_ids':[],'output_tokens':0,'truncated':False,'generation_error':type(e).__name__+': '+str(e)}
            json_dump({'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'identity_sha256':object_hash(identity),'input_token_ids':ids,'input_tokens':len(ids),'elapsed_seconds':time.monotonic()-start,**result},path)
            if result['generation_error']:break
        completed=[p for p in a.output.glob('*.json') if p.stem.isdigit()]
        json_dump({'planned':300,'completed':len(completed),'status':'completed' if len(completed)==300 else 'interrupted_or_error','api_calls':0},a.output/'STATE.json')
if __name__=='__main__':main()
