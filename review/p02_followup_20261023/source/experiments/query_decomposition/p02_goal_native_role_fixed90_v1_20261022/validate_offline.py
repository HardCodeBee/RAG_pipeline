"""Isolated fake SQLite/runtime/formatter contracts; never contact a provider."""
from contextlib import contextmanager,closing
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3

from . import run
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022 import shared_runtime as shared
from p02_multigrain_timing_cachefix_v1.core.cache import Cache,run_lock


def receipt(answer='One complete factual sentence.',**changes):
    model=run.read(run.PACKAGE/'config.json')['generation']['model']
    return dict(status='success',requested_model=model,returned_model=model,finish_reason='stop',answer=answer,
        usage=dict(prompt_tokens=20,completion_tokens=5,total_tokens=25),**changes)


class Client:
    def __init__(self,responses):self.responses=list(responses);self.calls=0
    def request_once(self,payload):
        self.calls+=1
        assert self.responses,'Unexpected fake submission (possible retry leak)'
        return deepcopy(self.responses.pop(0))
    def close(self):pass


class Budget:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,actual REAL,status TEXT,reserved REAL)')
        self.db.commit()
    def reserve(self,rid,request):
        self.db.execute("INSERT INTO charges VALUES(?,NULL,'reserved',.001)",(rid,));self.db.commit()
    def finish(self,rid,result):
        if run.usage_ok(result):
            self.db.execute("UPDATE charges SET actual=?,status='returned_usage' WHERE id=?",(run.cost(result['usage']),rid));self.db.commit()
    def close(self):self.db.close()


class Ask(run.SubmissionAwareStrictAsk):
    def __init__(self,*args,client,**kw):
        super().__init__(*args,**kw);self.client=client
    def _client(self):return self.client


@contextmanager
def no_gate(*args,**kwargs):yield


def invoke(path,client,budget):
    config=run.read(run.PACKAGE/'config.json')
    with run_lock(path.parent),Cache(path) as cache:
        ask=Ask(cache,None,config,budget,lambda e:None,client=client,maximum_attempts=5)
        return ask('fake/fixed','Synthetic offline fixed payload',128)


def held(path,client,budget):
    try:invoke(path,client,budget)
    except RuntimeError:return
    raise AssertionError('Expected held receipt')


def validate():
    checks=[]
    old_gate,old_sleep=shared.provider_gate,shared.time.sleep
    shared.provider_gate=no_gate;shared.time.sleep=lambda n:None
    try:
        run.OUT.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='fake_runtime_',dir=run.OUT) as temporary:
            root=Path(temporary).resolve()
            assert root.is_relative_to(run.OUT.resolve())
            budget=Budget(root/'ledger.sqlite3')
            try:
                first=Client([receipt()]);invoke(root/'success/db.sqlite3',first,budget)
                resumed=Client([]);invoke(root/'success/db.sqlite3',resumed,budget)
                assert first.calls==1 and resumed.calls==0
                checks.append('same_checkpoint_complete_success_resume_zero_new_calls')
                unknown=dict(status='unknown',failure_class='APIConnectionError')
                five=Client([unknown]*5);held(root/'five/db.sqlite3',five,budget)
                again=Client([]);held(root/'five/db.sqlite3',again,budget)
                assert five.calls==5 and again.calls==0
                with closing(run.ro(root/'five/db.sqlite3')) as db:
                    assert db.execute('SELECT count(*) FROM retired_requests').fetchone()[0]==4
                    assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==1
                checks.append('five_attempt_bound_persistent_and_all_retired_receipts_preserved')
                config=run.read(run.PACKAGE/'config.json')
                payload=shared.build_payload(config['generation'],'Synthetic offline fixed payload',128)
                for name,finish_ledger in [('inflight',False),('ledger_window',True)]:
                    path=root/name/'db.sqlite3'
                    with run_lock(path.parent),Cache(path) as cache:
                        rid,_=cache.begin_request('fake/fixed',payload,before_start=lambda rid:budget.reserve(rid,payload))
                        if finish_ledger:budget.finish(rid,receipt())
                    client=Client([unknown]*4 if not finish_ledger else [])
                    held(path,client,budget)
                    assert client.calls==(0 if finish_ledger else 4)
                checks.extend(['prior_inflight_counts_as_one_attempt','ledger_returned_usage_cache_inflight_crash_window_holds'])
                for name,value in [('response_id','received'),('usage',{}),('raw_response',{}),('returned_model','model'),('answer','content'),('refusal','declined'),('finish_reason','length')]:
                    path=root/name/'db.sqlite3';c=Client([{**unknown,name:value}]);held(path,c,budget)
                    r=Client([]);held(path,r,budget);assert c.calls==1 and r.calls==0
                checks.append('seven_return_evidence_fields_hold_across_restarts_including_refusal_finish_reason')
                transient=dict(status='failed',http_status=429)
                known=Client([transient,receipt()]);invoke(root/'bare429/db.sqlite3',known,budget);assert known.calls==2
                tagged=Client([{**transient,'raw_response':{}}]);held(root/'tag429/db.sqlite3',tagged,budget);assert tagged.calls==1
                checks.append('bare429_recovers_but_known429_with_return_evidence_holds')
                # A separate old history file cannot seed seed=None in a new DB.
                historical=Client([receipt()]);invoke(root/'historical/db.sqlite3',historical,budget)
                fresh=Client([receipt('A different complete factual sentence.')]);invoke(root/'fresh/db.sqlite3',fresh,budget)
                assert historical.calls==fresh.calls==1
                checks.append('new_checkpoint_seed_None_does_not_import_other_history')
                data=run.rows(run.INPUTS);record=data[0]
                examples=run.read(run.PACKAGE/'prompts/final_examples.json')
                for name,responses,status,stages,fmt in [
                    ('normal',[receipt(),receipt('Paris')],'success',2,False),
                    ('short_formatter',[receipt('Paris'),receipt('Paris is a city in France.'),receipt('Paris')],'success',3,True),
                    ('goal_length',[{**receipt(), 'finish_reason':'length'}],'algorithm_failure',1,False),
                    ('formatter_refusal',[receipt('Paris'),{**receipt(), 'refusal':'declined'}],'algorithm_failure',2,True),
                    ('final_empty',[receipt(),receipt('')],'algorithm_failure',2,False),
                    ('final_length',[receipt(),{**receipt('partial'),'finish_reason':'length'}],'success_truncated',2,False),
                    ('final_length_empty',[receipt(),{**receipt(''),'finish_reason':'length'}],'algorithm_failure',2,False),
                    ('final_length_refusal',[receipt(),{**receipt('partial'),'finish_reason':'length','refusal':'declined'}],'algorithm_failure',2,False),
                ]:
                    path=root/name/'db.sqlite3';client=Client(responses)
                    task=dict(index=0,id=record['id'],dataset=record['dataset'],arm='L0',task_id=record['id']+'/L0')
                    with run_lock(path.parent),Cache(path) as cache:
                        ask=Ask(cache,None,config,budget,lambda e:None,client=client)
                        outcome=run.execute_condition(cache,ask,record,task,config,examples)
                        assert outcome['status']==status and len(outcome['stages'])==stages and outcome['formatter_triggered']==fmt
                        cache.put('outcome',task['task_id'],outcome)
                    assert client.calls==stages
                checks.append('native_goal_formatter_final_chain_and_terminal_suppression_use_actual_controller')
                # Isolate one final submission and its restart: length is scored
                # as native success_truncated, not zero or a second completion.
                path=root/'length_one_final/db.sqlite3';client=Client([{**receipt('partial native answer'),'finish_reason':'length'}])
                task=dict(index=0,id=record['id'],dataset=record['dataset'],arm='L0',task_id=record['id']+'/L0')
                final_prompt=run.templates.final_prompt(record,'One complete factual sentence.',config,examples)[0]
                with run_lock(path.parent),Cache(path) as cache:
                    ask=Ask(cache,None,config,budget,lambda e:None,client=client)
                    final,reason=run.obtain(cache,ask,record,task,'final',final_prompt,64,config)
                    assert reason is None and final['finish_reason']=='length' and client.calls==1
                restarted=Client([])
                with run_lock(path.parent),Cache(path) as cache:
                    ask=Ask(cache,None,config,budget,lambda e:None,client=restarted)
                    final,reason=run.obtain(cache,ask,record,task,'final',final_prompt,64,config)
                    assert reason is None and final['finish_reason']=='length' and restarted.calls==0
                checks.append('native_final_length_accepted_same_receipt_once_resume_zero_calls')
                for name,mutation in [('missing_usage',{'usage':None}),('mismatch',{'returned_model':'wrong'}),('bool_usage',{'usage':dict(prompt_tokens=True,completion_tokens=5,total_tokens=6)}),('bad_total',{'usage':dict(prompt_tokens=20,completion_tokens=5,total_tokens=24)})]:
                    client=Client([{**receipt(),**mutation}]);path=root/name/'db.sqlite3'
                    task=dict(index=0,id=record['id'],dataset=record['dataset'],arm='L0',task_id=record['id']+'/L0')
                    with run_lock(path.parent),Cache(path) as cache:
                        ask=Ask(cache,None,config,budget,lambda e:None,client=client)
                        caught=False
                        try:run.execute_condition(cache,ask,record,task,config,examples)
                        except RuntimeError:caught=True
                        assert caught and not cache.all('outcome') and not cache.all('stage')
                    assert client.calls==1
                    assert not run.normal({**receipt(),**mutation},config['generation']['model'])
                    assert run.algorithm_failure({**receipt(),**mutation,'finish_reason':'length'},config['generation']['model']) is None
                checks.append('missing_model_or_three_usage_integrity_is_held_never_terminal_zero')
            finally:budget.close()
    finally:shared.provider_gate,shared.time.sleep=old_gate,old_sleep
    value=dict(passed=True,checks=checks,new_provider_requests=0,new_global_charges=0,new_retrieval_GPU_SSH=0,
        fake_checkpoints_only_in_temporary_directory=True,production_checkpoint_created=False)
    run.atomic_json(run.OUT/'OFFLINE_RUNTIME_VALIDATION.json',value)
    return value


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=True))
