"""Explicit execution-only continuation after a zero-request DNS preflight failure.

Keeps the failed empty round intact and uses a new recorded round namespace.
All rendering, budgets, samples, generation and scoring stay in experiment.py.
"""
from pathlib import Path
import json
import sqlite3
from . import experiment as e


def main():
    plan = e.prepare()
    if (e.OUT / 'TERMINAL.json').exists():
        print(json.dumps(e.read(e.OUT / 'TERMINAL.json'), ensure_ascii=False))
        return
    previous = e.OUT / 'rounds/pilot100_g01_goal_consistency_r001'
    assert previous.exists()
    databases = list((previous / 'methods').glob('*/*/checkpoint.sqlite3'))
    assert len(databases) == 3
    for p in databases:
        with sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True) as db:
            assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
    marker = previous / 'TERMINAL.json'
    if not marker.exists():
        e.save(marker, {'status': 'preflight_dns_failed', 'new_api_calls': 0,
                       'inference_started': False, 'namespace_preserved': True})
    cases = e.rows(e.HERE / 'runtime300.frozen.jsonl')
    manifests = {r['id']: r for r in e.rows(e.HERE / 'SOURCES.frozen.jsonl')}
    pilot = {r['id'] for r in e.rows(e.HERE / 'pilot100.ids.frozen.jsonl')}
    e.log('explicit_network_continuation', previous_namespace=str(previous.relative_to(e.ROOT)),
          paid_requests_in_previous_round=0, execution_only=True,
          helper_sha256=e.sha(Path(__file__)), source_and_protocol_unchanged=True)
    for arm in e.ARMS:
        e.run_phase(plan, [c for c in cases if c['id'] in pilot], arm, 'pilot100_network_recovery1', manifests)
        current = e.summarize(plan)
        pilot_path = e.OUT / (arm + '.PILOT.frozen.json')
        e.save(pilot_path, current['arms'][arm])
        if current['arms'][arm]['expand']:
            e.log('pilot_gate', arm=arm, expand=True, pilot_delta=current['arms'][arm]['pilot_delta'])
            e.run_phase(plan, [c for c in cases if c['id'] not in pilot], arm, 'extension200', manifests)
        else:
            e.log('pilot_gate', arm=arm, expand=False, pilot_delta=current['arms'][arm]['pilot_delta'],
                  reason='No positive complete100 delta; retain parent')
    current = e.summarize(plan)
    e.verify(plan)
    for row in manifests.values():
        assert e.sha(e.ROOT / row['base']) == row['base_sha256']
        assert e.sha(e.ROOT / row['control']) == row['control_sha256']
    e.save(e.OUT / 'FINAL_RESULTS.frozen.json', current)
    eligible = [(current['parent300_F1'], 'B0_P1_parent')]
    eligible += [(a['F1'], arm) for arm, a in current['arms'].items() if a['scorable'] == 300]
    score, winner = max(eligible, key=lambda x: x[0])
    e.save(e.OUT / 'SELECTION.frozen.json', {'selected': winner, 'F1': score,
           'scope': 'Exposed300 development selection', 'B0_and_S02_unchanged': True,
           'not_independent_confirmation': True})
    e.save(e.OUT / 'TERMINAL.json', {'utc': e.utc(), 'selected': winner,
           'budget': current['budget'], 'original_sources_unchanged': True})
    e.log('complete', selected=winner, final_F1=score, budget=current['budget'])
    print(json.dumps({'terminal': True, 'selected': winner, 'F1': score,
                     'budget': current['budget']}), flush=True)


if __name__ == '__main__':
    main()
