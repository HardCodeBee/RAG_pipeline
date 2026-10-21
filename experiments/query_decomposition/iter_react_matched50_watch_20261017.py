"""Monitor the authorized repair; bounded recovery only for transport stops."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/comprehensive_v1_20261015/baseline_integrity_repair_v2_20261017'
MODULE='experiments.query_decomposition.iter_react_matched50_20261017'


def save(status,**extra):
    path=OUT/'SUPERVISION.json'
    value={'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'status':status,**extra}
    tmp=path.with_name('SUPERVISION.json.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)


def workers():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if MODULE in (p.info.get('cmdline') or []):result.append(p.info['pid'])
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    return result


def network_stop_only():
    """A generic transport message must not restart identity/auth/quota failures."""
    network_unknown=False
    for track in ['matched50','main']:
        for path in (OUT/track).glob('*/*/*/*/checkpoint.sqlite3'):
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                records=db.execute('select result from attempts order by id desc limit 20').fetchall()
            for raw, in records:
                if not raw:continue
                result=json.loads(raw)
                if result.get('status') in {'model_identity_mismatch','input_token_limit_violation'} or result.get('http_status') in {401,403,404,429}:
                    return False
                if result.get('status')=='unknown' and result.get('failure_class') in {
                    'APIConnectionError','APITimeoutError','ConnectError','ConnectTimeout',
                    'ReadError','ReadTimeout','WriteError','WriteTimeout','RemoteProtocolError'}:
                    network_unknown=True
    return network_unknown


def main():
    failures=0
    deadline=time.monotonic()+24*3600
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    while time.monotonic()<deadline:
        if (OUT/'STOP_REQUESTED').exists():
            save('user_stop_requested_no_auto_resume');return
        try:progress=json.loads((OUT/'progress.json').read_text(encoding='utf8'))
        except (FileNotFoundError,ValueError):time.sleep(20);continue
        live=workers()
        if progress.get('status')=='completed':
            save('completed',stage=progress.get('stage'),budget=progress.get('budget'));return
        if live:
            save('running',worker_pids=live,generation_progress=progress,recoveries=failures,source_sha256=source_sha)
            time.sleep(20);continue
        reason=progress.get('reason','')
        if progress.get('status')=='stopped_for_audit' and reason.startswith('Transport stopped;') and failures<2 and network_stop_only():
            failures+=1
            with (OUT/'recovered.stdout.log').open('a',encoding='utf8') as stdout,(OUT/'recovered.stderr.log').open('a',encoding='utf8') as stderr:
                p=subprocess.Popen([sys.executable,'-X','utf8','-B','-m',MODULE],
                    cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save('recovered_known_transport_stop',worker_pid=p.pid,recoveries=failures,
                 unknown_requests_not_replayed=True,generation_config_changed=False)
            with (OUT/'changes.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':'bounded_transport_recovery',
                    'recovery':failures,'source_sha256':source_sha,'unknown_requests_not_replayed':True})+'\n')
            time.sleep(20);continue
        save('needs_attention_no_auto_restart',generation_progress=progress,recoveries=failures);return
    save('watch_deadline_reached_no_generation_changes')


if __name__=='__main__':main()
