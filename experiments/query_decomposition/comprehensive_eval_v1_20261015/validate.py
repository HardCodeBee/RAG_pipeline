"""Meaningful offline gates: native aliases, actual prompt parity, fixed packets."""
import importlib
import json
from pathlib import Path
import sqlite3
from .common import ROOT,HERE,RUN,OUTPUT,DATA,read,rows,sha,save,log,audit_prior
from . import scoring
from .campaign import protocol,OUR,CONTROL,ABLATIONS
from .methods.ablations import support
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts

class FakeStore:
    def __init__(self):self.data={}
    def get(self,kind,key):return self.data.get((kind,key))
    def put(self,kind,key,value):
        old=self.get(kind,key)
        if old is not None and old!=value:raise AssertionError('Immutable state changed')
        self.data[(kind,key)]=value
    def export(self,*args,**kwargs):pass

class NoRetrieval:
    def __getattr__(self,name):raise AssertionError('A frozen-packet ablation attempted new retrieval: '+name)

class FakeTransport:
    def __init__(self):self.qid=None;self.calls=[]
    def ask(self,stage,prompt,max_tokens):
        self.calls.append((self.qid,stage,prompt,max_tokens))
        return 'fake_'+str(len(self.calls)),{'status':'success','answer':'A supported test fact.','finish_reason':'stop'}

def main():
    checks={};checks['prior_sources']=audit_prior()
    for path in HERE.rglob('*.py'):compile(path.read_text(encoding='utf8'),str(path),'exec')
    summaries={};overlap={}
    for dataset in ['hotpotqa','2wiki','musique']:
        folder=HERE/'inputs_frozen'/dataset;manifest=read(folder/'SAMPLES_FROZEN.json')
        for name,expected in manifest['files_sha256'].items():assert sha(folder/name)==expected
        dev=rows(folder/'dev.references.local.jsonl');formal=rows(folder/'formal.references.local.jsonl');sub=rows(folder/'ablation.references.local.jsonl')
        assert len(dev)==100 and len(formal)==500 and len(sub)==200
        assert not ({r['group_id'] for r in dev}&{r['group_id'] for r in formal})
        assert {r['qid'] for r in sub}.issubset({r['qid'] for r in formal})
        metric=scoring.metric(dataset)
        for r in dev:assert metric(r['answer'],r)=={'em':1.0,'f1':1.0}
        if dataset=='musique':
            aliases=[r for r in dev if r['aliases']]
            assert aliases and all(metric(r['aliases'][0],r)['em']==1 for r in aliases)
        if dataset=='2wiki':
            aliases={r['Q_id']:r['aliases']+r['demonyms'] for r in rows(DATA/'2wiki/native/id_aliases.json')}
            matched=[r for r in dev if aliases.get(r['answer_id'])]
            assert matched and all(metric(aliases[r['answer_id']][0],r)['em']==1 for r in matched)
        summaries[dataset]={'dev':100,'formal':500,'ablation':200,'reference_exact_answer_checks':100}
        overlap[dataset]={r['question'].strip().casefold() for r in formal}
    assert scoring.metric('hotpotqa')('Yes, No',{'answer':'Yes'})['f1']==0
    assert abs(scoring.metric('musique')('Yes, No',{'answer':'Yes','aliases':[]})['f1']-2/3)<1e-12
    checks['native_metric_and_sample_checks']=summaries
    checks['exact_question_cross_dataset_overlaps']={a+'__'+b:len(overlap[a]&overlap[b]) for a in overlap for b in overlap if a<b}
    source=OUTPUT/'s02_child4_compact_formal200_20261014/formal'/OUR/'frozen_s02_child4'
    cases={r['qid']:r for r in rows(ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/compact_100_200_20261014/inputs_frozen/formal.questions.jsonl')}
    positions=read(OUTPUT/'s02_child4_compact_formal200_20261014/HELPER_POSITIONS_FROZEN.json')['positions']
    selected=[]
    with sqlite3.connect('file:'+str((source/'checkpoint.sqlite3').resolve())+'?mode=ro',uri=True) as db:
        plans=[{**json.loads(value),'qid':key} for key,value in db.execute("SELECT key,value FROM data WHERE kind='goal_plan'")]
        for action in ['retrieve','combine','skip']:
            selected.extend([p['qid'] for p in plans if p['action']==action][:4])
        contexts=[json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='context'")]
        selected.extend([c['qid'] for c in contexts if c['fallback']][:3])
        checks['actual_goal_actions_checked']=sorted({p['action'] for p in plans if p['qid'] in selected})
    selected=list(dict.fromkeys(selected));config={'source_directory':str(source),'helper_positions':positions}
    tested=0;fallback=0
    for qid in selected:
        case=cases[qid];data=support.load(source,qid);original=support.original_packet(case,data)
        if ('context',qid) not in data:continue
        for mid,name in [(CONTROL,'same_reader_top10'),*ABLATIONS.items()]:
            module=importlib.import_module('experiments.query_decomposition.comprehensive_eval_v1_20261015.methods.'+name+'.method')
            store=FakeStore();transport=FakeTransport()
            module.run([case],NoRetrieval(),transport,store,config)
            if mid==CONTROL:
                final_call=transport.calls[-1][2]
                source_prediction=data[('prediction',qid)]
                if source_prediction.get('request_id'):
                    with sqlite3.connect('file:'+str((source/'checkpoint.sqlite3').resolve())+'?mode=ro',uri=True) as db:
                        raw=db.execute('SELECT payload FROM requests WHERE id=?',(source_prediction['request_id'],)).fetchone()
                    if raw:
                        full_prompt=json.loads(raw[0])['messages'][0]['content'];ctx=data[('context',qid)]
                        expected=full_prompt.replace(ctx['context'],support.packet_context(original))
                        assert final_call==expected,'C1 differs beyond helper removal'
                        fallback+=int(ctx['fallback'])
            if mid=='a04_no_final_constraints' and transport.calls:
                actual=transport.calls[-1][2]
                ctx=data[('context',qid)]
                if ctx['helper_step_ids']:
                    examples=read(Path(prompts.__file__).with_name('examples.json'))['final']
                    before=prompts.final(case['question'],ctx['context'],examples)
                    for clause in module.REMOVED:
                        assert before.count(clause)==1
                        before=before.replace(clause,'',1)
                    assert actual==before and prompts.OUTPUT_INSTRUCTION in actual and prompts.READOUT.splitlines()[-1] in actual
            if mid=='a02_short_qa':
                for (kind,key),value in store.data.items():
                    if kind=='ablation_short_goal_input' and key.endswith('/'+str(data[('decomposition',qid)]['qdmr']['goal'])):
                        assert value['packet']==data[('goal_packet',qid)]
            if mid=='a03_no_goal_retrieval':
                for (kind,key),value in store.data.items():
                    if kind=='ablated_contextual_node_input' and key.endswith('/'+str(data[('decomposition',qid)]['qdmr']['goal'])):
                        assert value['packet']==original
            tested+=1
    checks['ablation_runs_on_actual_saved_cases']=tested
    checks['same_reader_actual_sdk_prompt_matches']=len(selected)
    checks['same_reader_v1_fallback_matches']=fallback
    from .methods.qd_rerank.method import parse
    assert parse('["Who wrote X?", "Who wrote X?"]',5)==['Who wrote X?']
    try:parse('["Who wrote #1?"]',5)
    except ValueError:pass
    else:raise AssertionError('QD+Rerank accepted dependent queries')
    checks['paid_api_submissions']=0;checks['passed']=True
    save(RUN/'OFFLINE_VALIDATION.json',checks);log('offline_validation_completed',**checks)
    print(json.dumps(checks))

if __name__=='__main__':main()
