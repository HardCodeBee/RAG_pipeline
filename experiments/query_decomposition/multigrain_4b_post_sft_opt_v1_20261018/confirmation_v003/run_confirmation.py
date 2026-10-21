"""One-time fresh90 confirmation; frozen baseline and development-selected prompt."""
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path[:0]=[str(a.package/'code'),str(a.package/'runtime')]
    from model_runtime import allocation_guard,load_libraries,load_model,seed_all,runtime_info,json_dump,object_hash,verify_checkpoint,StopRequest,encode_prompt
    from contract import sha,format_result,validate_config
    protocol=json.loads((a.experiment/'PROTOCOL_FROZEN.json').read_text())
    baseline=json.loads((a.experiment/'BASELINE_FROZEN.json').read_text())
    c=json.loads((a.package/'config.json').read_text());validate_config(c)
    cp=a.package/'results/train/checkpoints/checkpoint-000165'
    manifest=verify_checkpoint(cp);assert manifest['manifest_sha256']==baseline['checkpoint_manifest_sha256']
    queries_path=a.experiment/'confirmation90.questions.frozen.jsonl';assert sha(queries_path)==protocol['queries_sha256']
    queries=[json.loads(x) for x in queries_path.read_text().splitlines() if x.strip()];assert len(queries)==90
    for name,spec in protocol['variants'].items():assert sha(a.experiment/'prompts'/(name+'.txt'))==spec['prompt_sha256']
    allocation=allocation_guard();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch,np,transformers=load_libraries(allocation);seed_all(1729,torch,np,c)
    c['adapter_path']=str(cp);model,tokenizer,report=load_model(c,torch,transformers,train=False);model.eval()
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        json_dump(report,a.output/'MODEL_REPORT.json');json_dump(runtime_info(torch,transformers,allocation,c),a.output/'RUNTIME.json')
        stop=StopRequest();stop.install()
        for name,spec in protocol['variants'].items():
            if stop.requested:break
            seed_all(1729,torch,np,c)
            out=a.output/name;out.mkdir(exist_ok=True)
            prompt=(a.experiment/'prompts'/(name+'.txt')).read_text(encoding='utf-8')
            identity={'variant':name,'prompt_sha256':spec['prompt_sha256'],'protocol_sha256':sha(a.experiment/'PROTOCOL_FROZEN.json'),'queries_sha256':sha(queries_path),'checkpoint_manifest_sha256':manifest['manifest_sha256'],'do_sample':False,'num_beams':1,'max_new_tokens':1024,'seed':1729,'api_calls':0}
            if (out/'IDENTITY.json').exists():assert json.loads((out/'IDENTITY.json').read_text())==identity
            else:json_dump(identity,out/'IDENTITY.json')
            for i,r in enumerate(queries,1):
                path=out/f'{i:03d}.json'
                if path.exists():
                    old=json.loads(path.read_text());assert old['id']==r['id'] and old['identity_sha256']==object_hash(identity)
                    continue
                if stop.requested:break
                ids=encode_prompt(tokenizer,r['query'],prompt,'qwen3');assert len(ids)+1024<=model.config.max_position_embeddings
                x=torch.tensor([ids],device='cuda',dtype=torch.long);started=time.monotonic()
                try:
                    with torch.inference_mode():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                    tokens=y[0,len(ids):].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=True)
                    result={'raw_output':raw,**format_result(raw),'output_token_ids':tokens,'output_tokens':len(tokens),'truncated':len(tokens)>=1024 and tokens[-1]!=tokenizer.eos_token_id,'generation_error':None}
                except Exception as ex:result={'raw_output':None,**format_result(None),'output_token_ids':[],'output_tokens':0,'truncated':False,'generation_error':type(ex).__name__+': '+str(ex)}
                json_dump({'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'input_token_ids':ids,'input_tokens':len(ids),'elapsed_seconds':time.monotonic()-started,'identity_sha256':object_hash(identity),**result},path)
                print(json.dumps({'variant':name,'index':i,'error':result['generation_error']}),flush=True)
                if result['generation_error']:stop.requested=True;break
            done=[p for p in out.glob('*.json') if p.stem.isdigit()]
            json_dump({'status':'completed' if len(done)==90 else 'interrupted_or_error','completed':len(done),'planned':90,'api_calls':0},out/'STATE.json')
        json_dump({'status':'interrupted_or_error' if stop.requested else 'completed','api_calls':0},a.output/'STATE.json')

if __name__=='__main__':main()
