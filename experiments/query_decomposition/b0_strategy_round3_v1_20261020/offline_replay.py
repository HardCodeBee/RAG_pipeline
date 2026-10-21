"""Verify private transport binding reuses an unchanged actual parent payload."""
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from .experiment import OUT, HERE, ROOT, prepare, backend, read, rows, save, budget_snapshot, strategies


def main():
    plan = prepare()
    manifest = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    case = next(c for c in rows(HERE / 'runtime300.frozen.jsonl')
                if strategies().goal_extra_document(read(ROOT / manifest[c['id']]['control'])['nodes'],
                                                    read(ROOT / manifest[c['id']]['base'])['original_retrieval']) is None)
    before = budget_snapshot()
    engine = backend()
    class ForbiddenTransport:
        def __init__(self, *args, **kwargs):
            pass
        def ask(self, *args, **kwargs):
            raise AssertionError('Offline replay may not submit or reserve a new request')
    engine.OneShot = ForbiddenTransport
    (OUT / 'rounds').mkdir(parents=True, exist_ok=True)
    owner = Store(OUT / 'offline_replay/r000/methods' / case['dataset'] / 'e03_goal_evidence', {'offline': True})
    try:
        result = engine.run_case(case, 'e03_goal_evidence', owner, None, plan, manifest[case['id']])
        assert result['status'].startswith('success')
        parent = read(ROOT / manifest[case['id']]['control'])
        assert result['prediction'] == parent['prediction']
        assert owner.db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
        assert budget_snapshot()['new_submissions'] == before['new_submissions']
        save(OUT / 'OFFLINE_REPLAY.frozen.json', {'passed': True, 'id': case['id'],
              'actual_parent_prediction_reused': True, 'new_API_calls': 0, 'new_retrieval_calls': 0})
    finally:
        owner.db.close()
    print('Private transport exact-parent reuse passed; 0 API/retrieval calls')


if __name__ == '__main__':
    main()
