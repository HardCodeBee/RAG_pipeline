"""Export held-out references/probe and check the actual inherited loss encoder."""
import collections,json,math,random,sys
from pathlib import Path
from jinja2 import Environment
from tokenizers import Tokenizer
from reaudit import HERE,ROOT,PREVIOUS,SOURCE,OUT,rows,sha,norm,write,write_rows
from prepare_data import components,rank,EVAL

def main():
    train=rows(OUT/'data_v002/MASTER_3500.jsonl');dev=rows(HERE/'dev150.questions.frozen.jsonl');test=rows(EVAL/'questions300.frozen.jsonl')
    labels=json.loads((HERE/'dev150.labels.reviewed.json').read_text(encoding='utf-8'))
    assert set(labels['labels'])=={str(r['index']) for r in dev}
    for r in dev:r['target']=labels['labels'][str(r['index'])]
    for a,b in [(train,dev),(train,test),(dev,test)]:
        assert not {norm(r['query']) for r in a}&{norm(r['query']) for r in b}
        assert not set().union(*(components(r) for r in a))&set().union(*(components(r) for r in b))
    prompt=(PREVIOUS/'FINAL_PROMPT.txt').read_text(encoding='utf-8').strip()
    write_rows(OUT/'data_v002/dev150.master.jsonl',dev)
    def messages(r):return {'id':r['id'],'dataset':r.get('dataset',r.get('source',{}).get('dataset')),'messages':[{'role':'system','content':prompt},{'role':'user','content':r['query']},{'role':'assistant','content':json.dumps(r['target'],ensure_ascii=False,separators=(',',':'))}]}
    write_rows(OUT/'data_v002/dev150.messages.jsonl',[messages(r) for r in dev])
    probe=[]
    for ds in ['hotpot','2wiki','musique']:
        pool=[r for r in train if ds in r['source']['dataset'].lower()]
        probe.extend(sorted(pool,key=lambda r:rank('probe:'+r['id']))[:50])
    assert len(probe)==150
    write_rows(OUT/'data_v002/train_probe150.master.jsonl',probe)
    write_rows(OUT/'data_v002/train_probe150.messages.jsonl',[messages(r) for r in probe])
    sys.path.insert(0,str(HERE/'remote/runtime'))
    from model_runtime import encode_supervised
    base=SOURCE/'base_tokenizer';settings=json.loads((base/'tokenizer_config.json').read_text(encoding='utf-8'))
    tok=Tokenizer.from_file(str(base/'tokenizer.json'));template=Environment().from_string(settings['chat_template'])
    class NativeTokenizer:
        eos_token_id=tok.token_to_id('<|im_end|>')
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            text=template.render(messages=messages,tools=None,add_generation_prompt=add_generation_prompt)
            return tok.encode(text,add_special_tokens=False).ids if tokenize else text
        def decode(self,ids,skip_special_tokens=False):return tok.decode(ids,skip_special_tokens=skip_special_tokens)
    tokenizer=NativeTokenizer();lengths={};empty_checked=0
    for name,data in [('train3500',train),('dev150',dev),('train_probe150',probe)]:
        values=[]
        for r in data:
            assert isinstance(r['target'],list) and all(isinstance(q,str) and q.endswith('?') for q in r['target'])
            encoded=encode_supervised(tokenizer,r,prompt,'qwen3');values.append(encoded['total_tokens'])
            assert encoded['total_tokens']<=1024,(name,r['id'],encoded['total_tokens'])
            assert sum(v!=-100 for v in encoded['labels'])==encoded['target_tokens']
            assert encoded['labels'][-1]==tokenizer.eos_token_id
            if not r['target']:assert tokenizer.decode(encoded['labels'][encoded['prompt_tokens']:-1])=='[]';empty_checked+=1
        values.sort();lengths[name]={'records':len(values),'min':min(values),'median':values[len(values)//2],'p95':values[math.ceil(.95*len(values))-1],'max':max(values)}
    sample=random.Random(2026101703).sample(train,300)
    write_rows(OUT/'data_v002/CONSISTENCY_SAMPLE300.frozen.jsonl',[{'id':r['id'],'index':r['index'],'query':r['query'],'target':r['target']} for r in sample])
    # This is artifact/masking consistency, not a fabricated third semantic audit.
    report={'train_records':3500,'dev_records':150,'probe_records':150,'test_records':300,'prompt_sha256':sha(PREVIOUS/'FINAL_PROMPT.txt'),'all_pairs_text_disjoint':True,'all_pairs_musique_component_disjoint':True,'native_encoding_and_target_eos_masks_checked':3800,'empty_arrays_have_real_supervised_tokens':empty_checked,'token_lengths':lengths,'silent_truncation':0,'sample300_scope':'Frozen random artifact consistency sample after full second semantic pass; not independent semantic accuracy estimate.','reference_review':labels['reviewer'],'artifacts':{p.name:sha(p) for p in (OUT/'data_v002').glob('*.jsonl')},'api_calls':0,'training_started':False}
    write(OUT/'data_v002/VALIDATION.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
