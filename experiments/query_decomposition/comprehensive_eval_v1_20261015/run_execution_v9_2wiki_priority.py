"""Run frozen 2Wiki S02 formal500 next, then resume the existing campaign."""
from .common import HERE, RUN, read, rows, sha, save, log, utc
from . import campaign, run_execution_v3, run_execution_v7, run_execution_v8_hotpot_priority, accept_return


def verify_epoch():
    epoch = read(RUN / 'TWOWIKI_PRIORITY_EXECUTION_FROZEN.json')
    for name, expected in epoch['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('2Wiki priority execution source changed: ' + name)
    if sha(RUN / 'HOTPOT_PRIORITY_EXECUTION_FROZEN.json') != epoch['parent_priority_sha256']:
        raise ValueError('Previous priority execution epoch changed')
    for name, expected in epoch['dataset_state_sha256'].items():
        if sha(campaign.TARGET / '2wiki' / name) != expected:
            raise ValueError('2Wiki frozen resource or baseline configuration changed: ' + name)
    run_execution_v8_hotpot_priority.verify_epoch()
    return epoch


def completed_baselines(dataset):
    selected = read(campaign.TARGET / dataset / 'BASELINE_CONFIGS_FROZEN.json')['selected']
    expected = {c['qid'] for c in campaign.questions(dataset, 'formal')[1]}
    for mid, cfg in selected.items():
        directory = campaign.TARGET / dataset / 'formal' / mid / cfg['id']
        score = directory / 'scores.json'
        if (not score.exists() or read(score)['overall']['terminal_records'] != 500
                or {r['qid'] for r in rows(directory / 'scores.jsonl')} != expected):
            raise ValueError('2Wiki priority requires completed baseline records: ' + dataset + '/' + mid)
    return selected


def prioritize_2wiki(instance):
    epoch = verify_epoch()
    completed_baselines('2wiki')
    remote = accept_return.main()
    if not remote.get('passed'):
        raise ValueError('Audited manual decompositions not available')
    question_path, cases = campaign.questions('2wiki', 'formal')
    if len(cases) != 500 or sha(question_path) != epoch['formal_question_sha256']:
        raise ValueError('2Wiki frozen formal500 questions changed')
    if sha(RUN / 'remote1500/2wiki.decompositions.jsonl') != epoch['decomposition_sha256']:
        raise ValueError('2Wiki accepted decomposition file changed')
    cfg = instance.receipt['protocol']['methods'][campaign.OUR]['default']
    if cfg != epoch['unchanged_method_config']:
        raise ValueError('2Wiki priority changed frozen S02 configuration')
    save(RUN / 'TWOWIKI_PRIORITY_STATUS.json', {'utc': utc(), 'status': 'running',
         'dataset': '2wiki', 'method': campaign.OUR, 'planned': 500,
         'unchanged_config': cfg, 'execution_epoch': epoch['id']})
    log('2wiki500_s02_prioritized', config=cfg, sample_sha256=sha(question_path),
        execution_order_only=True, other_checkpoints_preserved=True, api_cap_cny=200)
    for start in range(0, 500, 20):
        instance.execute('2wiki', 'formal', campaign.OUR, cfg, cases[start:start+20])
    scores = read(campaign.TARGET / '2wiki/formal' / campaign.OUR / cfg['id'] / 'scores.json')
    if scores['overall']['terminal_records'] != 500:
        raise ValueError('2Wiki priority did not retain all planned terminal records')
    save(RUN / 'TWOWIKI_PRIORITY_STATUS.json', {'utc': utc(), 'status': 'completed',
         'dataset': '2wiki', 'method': campaign.OUR, 'scores': scores,
         'next': 'resume existing baseline/control/ablation campaign', 'execution_epoch': epoch['id']})
    from .report import main as report
    report()
    log('2wiki500_s02_priority_completed', overall=scores['overall'],
        remaining_campaign_order='original', config_changed=False)


def install_priority(callback=None, owner=None):
    owner = campaign.Campaign if owner is None else owner
    callback = prioritize_2wiki if callback is None else callback
    previous = owner.baseline_main

    def baseline_main(self, dataset):
        if not getattr(self, '_2wiki_s02_priority_completed', False):
            callback(self)
            self._2wiki_s02_priority_completed = True
        if dataset in {'hotpotqa', '2wiki'}:
            return completed_baselines(dataset)
        return previous(self, dataset)

    owner.baseline_main = baseline_main


def main():
    verify_epoch()
    original_install = run_execution_v3.install

    def install():
        original_install()
        install_priority()

    run_execution_v3.install = install
    run_execution_v7.main()


if __name__ == '__main__':
    main()
