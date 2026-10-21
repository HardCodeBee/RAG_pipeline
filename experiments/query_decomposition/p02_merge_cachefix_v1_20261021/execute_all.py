"""Missing-only execution after cache-only r000 materialization.

Each dataset has a separate dispatch and unknown-result circuit. No source
method/prompt is changed across recovery rounds. Successes are never reopened.
"""
from collections import Counter
import argparse
import os

from . import run

DATASETS = ('2wiki','musique','hotpotqa')


def inspect():
    counts, pending, blocked = {}, [], []
    for ds in DATASETS:
        state = Counter()
        for case in run.rows(run.HERE / 'runtime300.frozen.jsonl'):
            if case['dataset'] != ds:
                continue
            for arm in run.ARMS:
                value = run.selected(case, arm)
                status = value['status'] if value else 'unstarted'
                state[status] += 1
                if status.startswith('success') or status == 'algorithm_failure':
                    continue
                item = {'id':case['id'],'dataset':ds,'arm':arm,'status':status,
                        'reason':(value or {}).get('reason'),
                        'pending_new_input':(value or {}).get('pending_new_input',False)}
                if value is None or value.get('retryable_infrastructure'):
                    pending.append(item)
                else:
                    blocked.append(item)
        counts[ds] = dict(state)
    scorable = sum(sum(n for k,n in states.items() if k.startswith('success') or k=='algorithm_failure') for states in counts.values())
    return {'utc':run.utc(),'planned':600,'scorable':scorable,'missing':600-scorable,
            'by_dataset':counts,'pending':pending,'blocked':blocked,'budget':run.budget_snapshot()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-round',type=int,default=1)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    plan=run.prepare()
    if args.inspect:
        print(run.json.dumps(inspect(),ensure_ascii=False),flush=True)
        return
    assert run.read(run.OUT/'OFFLINE_AUDIT.json')['passed']
    lock=run.OUT/'ACTIVE.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(str(os.getpid()))
        for round_id in range(args.start_round,plan['max_rounds']):
            before=inspect()
            if before['scorable']==600:
                run.save(run.OUT/'EXECUTION_COMPLETE.json',before)
                break
            if not before['pending']:
                run.save(run.OUT/'WAITING_NONRETRYABLE.json',before)
                break
            run.freeze(run.OUT/f'rounds/r{round_id:03d}/MISSING.frozen.json',before)
            for ds in DATASETS:
                if any(v['dataset']==ds for v in before['pending']):
                    run.run(plan,sample='300',round_id=round_id,dataset=ds)
                run.save(run.OUT/'LATEST_EXECUTION_STATUS.json',inspect())
            after=inspect()
            run.log('missing_only_round_completed',round=round_id,scorable=after['scorable'],budget=after['budget'])
            if after['scorable']==600:
                run.save(run.OUT/'EXECUTION_COMPLETE.json',after)
                break
            if after['scorable']==before['scorable'] and round_id>args.start_round:
                run.save(run.OUT/'WAITING_ZERO_PROGRESS.json',after)
                break
        print(run.json.dumps(inspect(),ensure_ascii=False),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
