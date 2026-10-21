"""Authorized SSH lifecycle; password stays only in transient process memory."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko
from .prepare import HERE,OUT,REMOTE,sha,read


def status(name,value):
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
    locked_json(OUT/name,{'utc':datetime.now(timezone.utc).isoformat(),**value})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true')
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    password=os.environ.pop('RAG_EVAL_REMOTE_PASSWORD',None)
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect('cluster1.comp.anu.edu.au',username='u7830488',password=password,
                       look_for_keys=password is None,allow_agent=password is None,
                       timeout=15,auth_timeout=20,banner_timeout=15)
        client.get_transport().set_keepalive(30)
        if args.probe:
            status('REMOTE_CONNECT.json',{'connected':True,'remote_scope':'query_decomposition_only','submitted':False})
            print(json.dumps({'connected':True,'submitted':False}));return
        sftp=client.open_sftp()
        try:sftp.stat(REMOTE)
        except FileNotFoundError:sftp.mkdir(REMOTE)
        meta=read(HERE/'REMOTE_INPUTS.frozen.json')
        for name in [*meta['files'],'REMOTE_INPUTS.frozen.json']:
            source=HERE/name;target=REMOTE+'/'+name
            try:
                with sftp.open(target,'rb') as f:existing=f.read()
                assert hashlib.sha256(existing).hexdigest()==sha(source),'Existing remote input differs; do not overwrite'
            except FileNotFoundError:sftp.put(str(source),target)
        def call(action):
            _,stdout,stderr=client.exec_command('python3 '+shlex.quote(REMOTE+'/remote_worker.py')+' '+action,timeout=55)
            text=stdout.read().decode('utf8');error=stderr.read().decode('utf8')
            if stdout.channel.recv_exit_status():raise RuntimeError('Remote '+action+': '+error[-1800:])
            return json.loads(text)
        started=call('stage');status('REMOTE_START.json',started)
        print(json.dumps({k:started.get(k) for k in ['submission','job_id','completed','planned','queue']},ensure_ascii=False),flush=True)
        local=OUT/'remote_raw';local.mkdir(parents=True,exist_ok=True)
        last=None
        while True:
            value=call('progress');status('REMOTE_PROGRESS.json',value)
            identity=value.get('identity')
            if identity:
                destination=local/'IDENTITY.json'
                if destination.exists():assert read(destination)==identity,'Remote identity changed'
                else:destination.write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            for name in value['files']:
                assert name.endswith('.json') and name[:-5].isdigit()
                target=local/name
                if target.exists():continue
                with sftp.open(REMOTE+'/results/'+name,'rb') as f:raw=f.read()
                record=json.loads(raw)
                assert record['index']==int(name[:-5])
                temp=target.with_suffix('.part');temp.write_bytes(raw);temp.replace(target)
            key=(value['completed'],value['queue'],value['accounting'])
            if key!=last:
                print(json.dumps({'completed':value['completed'],'planned':1500,'queue':value['queue'],'accounting':value['accounting']},ensure_ascii=False),flush=True);last=key
            if value['state'] and value['state'].get('status')=='completed' and value['completed']==1500:
                status('REMOTE_COMPLETE.json',{'complete':True,'records':1500,'remote_scope':'query_decomposition_only','new_API_calls':0});break
            if any(w in value['accounting'] for w in ['FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL']):
                raise RuntimeError('Remote job ended before completion; partial output retained, no automatic resubmission')
            if not args.watch:break
            time.sleep(20)
        sftp.close()
    except BaseException as exc:
        # Avoid secret-bearing connection objects/tracebacks in public records.
        status('REMOTE_ERROR.json',{'error_type':type(exc).__name__,'reason':str(exc)[:1800],
                                  'remote_scope':'query_decomposition_only','new_API_calls':0})
        print(json.dumps({'remote_error_type':type(exc).__name__,'submitted_state_recorded':(OUT/'REMOTE_START.json').exists()}),flush=True)
        raise
    finally:
        client.close();password=None


if __name__=='__main__':main()
