"""v003: retain the frozen v12 final newline and prove baseline input-token parity."""
from datetime import datetime,timezone
import json,math,shutil,sys,zipfile
from pathlib import Path
from jinja2 import Environment
from tokenizers import Tokenizer
from reaudit import HERE,ROOT,PREVIOUS,SOURCE,OUT,rows,sha,write,write_rows

def main():
    source=OUT/'data_v002';dest=OUT/'data_v003_exact_v12';assert not dest.exists(),'Never overwrite an existing revision'
    shutil.copytree(source,dest)
    prompt=(PREVIOUS/'FINAL_PROMPT.txt').read_text(encoding='utf-8')
    assert prompt.endswith('\n') and prompt[:-1]==prompt.strip()
    base=SOURCE/'base_tokenizer';settings=json.loads((base/'tokenizer_config.json').read_text(encoding='utf-8'))
    tok=Tokenizer.from_file(str(base/'tokenizer.json'));template=Environment().from_string(settings['chat_template'])
    class NativeTokenizer:
        eos_token_id=tok.token_to_id('<|im_end|>')
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            text=template.render(messages=messages,tools=None,add_generation_prompt=add_generation_prompt)
            return tok.encode(text,add_special_tokens=False).ids if tokenize else text
        def decode(self,ids,skip_special_tokens=False):return tok.decode(ids,skip_special_tokens=skip_special_tokens)
    sys.path.insert(0,str(HERE/'remote/runtime'));from model_runtime import encode_supervised,encode_prompt
    tokenizer=NativeTokenizer();lengths={}
    for master,messages in [('MASTER_3500.jsonl','train3500.messages.jsonl'),('dev150.master.jsonl','dev150.messages.jsonl'),('train_probe150.master.jsonl','train_probe150.messages.jsonl')]:
        data=rows(dest/master);old_messages=rows(source/messages);updated=[];values=[]
        for r,old in zip(data,old_messages):
            assert old['messages'][0]['content']==prompt.strip()
            item=json.loads(json.dumps(old));item['messages'][0]['content']=prompt;updated.append(item)
            encoded=encode_supervised(tokenizer,r,prompt,'qwen3');assert encoded['total_tokens']<=1024
            assert sum(t!=-100 for t in encoded['labels'])==encoded['target_tokens'] and encoded['labels'][-1]==tokenizer.eos_token_id
            assert json.loads(item['messages'][2]['content'])==r['target'];values.append(encoded['total_tokens'])
        write_rows(dest/messages,updated);values.sort();lengths[messages]={'records':len(values),'min':min(values),'median':values[len(values)//2],'p95':values[math.ceil(.95*len(values))-1],'max':max(values)}
    prediction_file=ROOT/'outputs/query_decomposition/multigrain_4b_frozen_eval300_v1_20261017/returned_20261017T113858Z/predictions.jsonl'
    matches=0;stripped_diff=0
    for r in rows(prediction_file):
        assert encode_prompt(tokenizer,r['query'],prompt,'qwen3')==r['input_token_ids']
        matches+=1;stripped_diff+=encode_prompt(tokenizer,r['query'],prompt.strip(),'qwen3')!=r['input_token_ids']
    assert matches==stripped_diff==300
    assert sha(dest/'MASTER_3500.jsonl')==sha(source/'MASTER_3500.jsonl')
    val=json.loads((source/'VALIDATION.json').read_text());val.update(revision='v003_exact_v12',token_lengths=lengths,prompt_payload_keeps_frozen_final_newline=True,baseline300_exact_input_token_matches=matches,old_stripped_prompt_differing_cases=stripped_diff,training_labels_changed=0)
    val['artifacts']={p.name:sha(p) for p in dest.glob('*.jsonl')};write(dest/'VALIDATION.json',val)
    frozen=json.loads((source/'TRAINING_DATA_FROZEN.json').read_text());frozen.update(revision='v003_exact_v12',semantic_label_source_revision='v002',messages_sha256=sha(dest/'train3500.messages.jsonl'),prompt_payload='exact file contents, including final newline');write(dest/'TRAINING_DATA_FROZEN.json',frozen)
    package=HERE/'remote_v003';assert not package.exists();shutil.copytree(HERE/'remote',package,ignore=shutil.ignore_patterns('__pycache__'))
    trainer=package/'code/train.py';text=trainer.read_text();old="prompt=Path(c['prompt_file']).read_text(encoding='utf-8').strip()";assert text.count(old)==1;trainer.write_text(text.replace(old,"prompt=Path(c['prompt_file']).read_text(encoding='utf-8')"),encoding='utf-8',newline='\n')
    for name in ['MASTER_3500.jsonl','train3500.messages.jsonl','dev150.master.jsonl','dev150.messages.jsonl','train_probe150.master.jsonl','train_probe150.messages.jsonl']:shutil.copyfile(dest/name,package/'data'/name)
    old_remote='/home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package';remote=old_remote+'_v003'
    c=json.loads((package/'config.json').read_text())
    for k in ['base_identity_file','data_dir','output_root','prompt_file']:c[k]=c[k].replace(old_remote,remote)
    c['data_sha256']={p.name:sha(p) for p in (package/'data').glob('*.jsonl')}
    c['prompt_payload_policy']='exact_frozen_v12_bytes_no_strip'
    write(package/'config.json',c)
    ready=json.loads((package/'EXECUTION_READY.json').read_text());ready.update(train_revision='v003_exact_v12',semantic_labels='same3500_v002',data_validation_sha256=sha(dest/'VALIDATION.json'),baseline300_exact_input_token_matches=300);write(package/'EXECUTION_READY.json',ready)
    files={p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json'}
    write(package/'PACKAGE_MANIFEST.json',{'files':files,'remote_root':remote,'api_calls':0})
    archive=OUT/'sft3500_v003_exact_v12_remote_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'PACKAGE_MANIFEST.json']:z.write(package/name,name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    report={'recorded_utc':datetime.now(timezone.utc).isoformat(),'reason':'Training draft stripped the final v12 newline; exact300 baseline token check detected a payload difference before any GPU training. Preserve the newline for exact before/after inputs.','superseded_job':'153977 held while still pending; no smoke or gradient updates','semantic_labels_changed':0,'old_package_retained':True,'old_data_v002_retained':True,'new_data':str(dest),'new_package':str(package),'remote_root':remote,'archive':str(archive),'archive_sha256':sha(archive),'baseline300_exact_input_token_matches':300,'native_encoding_and_mask_records':3800,'max_tokens':{k:v['max'] for k,v in lengths.items()},'api_calls':0}
    write(OUT/'EXACT_PROMPT_REVISION.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
