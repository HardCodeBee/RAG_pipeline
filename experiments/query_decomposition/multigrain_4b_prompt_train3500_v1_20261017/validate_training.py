"""Check complete native chat lengths and artifact consistency; never truncates."""
from collections import Counter
import json,math
from jinja2 import Environment
from tokenizers import Tokenizer
from training_sources import HERE,OUT,rows,sha,write,write_rows

EXPECTED={'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4','tokenizer_config.json':'a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3'}

def main():
    master=rows(OUT/'MASTER_3500.jsonl');sft=rows(OUT/'train3500.messages.jsonl');assert len(master)==len(sft)==3500
    config=json.loads((OUT/'base_tokenizer/tokenizer_config.json').read_text(encoding='utf-8'))
    for name,h in EXPECTED.items():assert sha(OUT/'base_tokenizer'/name)==h,name
    tokenizer=Tokenizer.from_file(str(OUT/'base_tokenizer/tokenizer.json'))
    template=Environment().from_string(config['chat_template']);lengths=[]
    for r,item in zip(master,sft):
        messages=item['messages'];assert [m['role'] for m in messages]==['system','user','assistant']
        assert messages[0]['content']==(HERE/'FINAL_PROMPT.txt').read_text(encoding='utf-8').strip()
        assert messages[1]['content']==r['query'] and json.loads(messages[2]['content'])==r['target']
        full=template.render(messages=messages,tools=None,add_generation_prompt=False)
        prefix=template.render(messages=messages[:2],tools=None,add_generation_prompt=True)
        assert full.startswith(prefix) and full.endswith('<|im_end|>\n')
        encoded=tokenizer.encode(full,add_special_tokens=False)
        start=len(prefix);supervised=[k for k,(lo,hi) in enumerate(encoded.offsets) if hi>start]
        assert supervised and len(supervised)<len(encoded.ids)
        # Offsets check boundaries for future masked SFT; no trainer is executed.
        lengths.append({'index':r['index'],'id':r['id'],'input_tokens':len(tokenizer.encode(prefix,add_special_tokens=False).ids),'full_chat_tokens':len(encoded.ids),'assistant_and_end_tokens':len(supervised)})
    values=sorted(r['full_chat_tokens'] for r in lengths)
    review=rows(OUT/'QUALITY_SAMPLE300.review.jsonl');verdicts=Counter(r['verdict'] for r in review)
    assert len(review)==300 and len({r['id'] for r in review})==300
    fixes=json.loads((HERE/'training_reviews/SECOND_REVIEW_CORRECTIONS.json').read_text(encoding='utf-8'))['revisions']
    by_index={r['index']:r for r in master}
    for r in review:
        if r['verdict']!='pass':assert str(r['index']) in fixes
    for i,fix in fixes.items():assert by_index[int(i)]['target']==fix['target']
    summary={'records':3500,'tokenizer_source':'Authorized remote original Qwen/Qwen3-4B-Instruct-2507 checkout, read-only download; not old adapter tokenizer','model_revision':'cdbee75f17c01a7cc42f958dc650907174af0554','tokenizer_hashes':EXPECTED,'chat_template':'native tokenizer_config.json; no thinking or additional wrappers','full_chat_tokens':{'min':values[0],'median':values[len(values)//2],'p95':values[math.ceil(.95*len(values))-1],'p99':values[math.ceil(.99*len(values))-1],'max':values[-1]},'exceeds1024':sum(x>1024 for x in values),'exceeds1536':sum(x>1536 for x in values),'exceeds2048':sum(x>2048 for x in values),'truncated_records':0,'sample300_before_correction':dict(verdicts),'sample_scope':'Stratified89Hotpot/74DROP/32CWQ/44MuSiQue/61_2Wiki; frozen before second review; same Codex, not independent quality estimation','sample_problem_records_corrected':sum(r['verdict']!='pass' for r in review),'sample_meets_planned294_pass_threshold_before_correction':verdicts['pass']>=294,'correction_record_count':len(fixes),'known_protected_query_checks':'Prompt100 excluded; source-freezer protected evaluation text excluded except33 explicit user-designated hard cases with actual dev split retained. No claim of paraphrase or pretraining decontamination.','sft_contract':'3500 training examples, no held-out split removed from3500. Assistant output and end token only should be supervised by future trainer; actual training/loss masking still requires trainer validation.','api_calls':0,'training_started':False}
    write_rows(OUT/'TOKEN_LENGTHS.jsonl',lengths);write(OUT/'FINAL_VALIDATION.json',summary)
    report=json.loads((OUT/'DATA_REPORT.json').read_text(encoding='utf-8'));report['status']='ready_for_training_setup_not_trained';report['final_validation_sha256']=sha(OUT/'FINAL_VALIDATION.json');write(OUT/'DATA_REPORT.json',report)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
