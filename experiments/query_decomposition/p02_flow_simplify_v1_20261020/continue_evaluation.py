"""Finish the authorized 60->300 evaluation without overlapping dispatchers.

Waits for the pilot terminal file, then extends only the unattempted240,
recovers incomplete infrastructure cases once in r001 and exports reports.
Does not tune prompts, select by partial F1, or rerun successful cases.
"""
import argparse
import json
import subprocess
import sys
import time

from .run import HERE, OUT, ARMS, read, save, selected, rows, utc, budget_snapshot


def command(module, *args):
    cmd = [sys.executable, '-X', 'utf8', '-m', 'experiments.query_decomposition.p02_flow_simplify_v1_20261020.' + module, *args]
    record = {'utc': utc(), 'command': cmd}
    with (OUT / 'CONTINUATION.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        save(OUT / 'CONTINUATION_FAILURE.json', {**record, 'returncode': result.returncode, 'budget': budget_snapshot()})
        raise RuntimeError('Continuation step failed; no blind restart: ' + module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pilot', action='store_true')
    args = parser.parse_args()
    pilot_terminal = OUT / 'phases/pilot60_r000_all_union20_discount20_single_final/TERMINAL.json'
    if args.wait_for_pilot:
        begun = time.monotonic()
        while not pilot_terminal.exists():
            if time.monotonic() - begun > 7200:
                raise RuntimeError('Pilot has not reached a terminal state; inspect rather than starting another dispatcher')
            time.sleep(5)
    assert pilot_terminal.exists(), 'Pilot must finish before extension opens method owners'
    command('score', '--sample', 'pilot60', '--arms', *ARMS)
    command('report', '--sample', 'pilot60')
    command('cache_catalog_v002', '--audit')
    command('runner_v002', '--prepare')
    command('runner_v002', '--sample', 'extension240', '--arms', *ARMS)
    command('score', '--sample', '300', '--arms', *ARMS)
    # At most one immutable recovery for each incomplete infrastructure case.
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    recovery = [(c['id'], arm) for c in cases for arm in ARMS
                if (v := selected(c, arm)) is not None and v['status'] == 'incomplete'
                and v.get('retryable_infrastructure', False)]
    save(OUT / 'RECOVERY_SELECTION.frozen.json', {'utc': utc(), 'method_cases': recovery,
                                                'successful_cases_excluded': True, 'budget': budget_snapshot()})
    if recovery:
        command('runner_v002', '--sample', '300', '--round', '1', '--arms', *ARMS)
    command('score', '--sample', 'pilot60', '--arms', *ARMS)
    command('report', '--sample', 'pilot60')
    command('score', '--sample', '300', '--arms', *ARMS)
    command('report', '--sample', '300')
    save(OUT / 'CONTINUATION_COMPLETE.json', {'utc': utc(), 'all_stages_executed': True,
                                            'recovery_method_cases': len(recovery), 'budget': budget_snapshot(),
                                            'note': 'Final completeness and missingness are in SCORES_300_LATEST.json; do not infer all900 successful from this marker.'})


if __name__ == '__main__':
    main()
