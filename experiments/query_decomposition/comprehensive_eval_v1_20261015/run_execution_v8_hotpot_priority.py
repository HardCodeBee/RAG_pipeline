"""Run frozen HotpotQA S02 formal500 first, then resume the original campaign."""
from .common import HERE, RUN, read, sha, save, once, log, utc
from . import campaign, run_execution_v3, run_execution_v7, accept_return


def verify_epoch():
    epoch = read(RUN / 'HOTPOT_PRIORITY_EXECUTION_FROZEN.json')
    for name, expected in epoch['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Hotpot priority execution source changed: ' + name)
    if sha(RUN / 'SCAN_API_ENTRYPOINT_FROZEN.json') != epoch['parent_entrypoint_sha256']:
        raise ValueError('Parent verified execution epoch changed')
    campaign.verify(read(campaign.TARGET / 'CAMPAIGN_FROZEN.json'))
    return epoch


def original_hotpot_baselines_complete():
    path = campaign.TARGET / 'hotpotqa/BASELINE_CONFIGS_FROZEN.json'
    selected = read(path)['selected']
    for mid, cfg in selected.items():
        score = campaign.TARGET / 'hotpotqa/formal' / mid / cfg['id'] / 'scores.json'
        if not score.exists() or read(score)['overall']['terminal_records'] != 500:
            raise ValueError('Priority requires the completed Hotpot baseline formal500')
    return selected


def prioritize_hotpot(instance):
    epoch = verify_epoch()
    original_hotpot_baselines_complete()
    # The same audited manual return and the same attach_inputs are used.
    remote = accept_return.main()
    if not remote.get('passed'):
        raise ValueError('Frozen manual query decompositions not available')
    question_path, cases = campaign.questions('hotpotqa', 'formal')
    if len(cases) != 500 or sha(question_path) != epoch['formal_question_sha256']:
        raise ValueError('Priority formal500 sample changed')
    cfg = instance.receipt['protocol']['methods'][campaign.OUR]['default']
    if cfg != epoch['unchanged_method_config']:
        raise ValueError('Priority changed the frozen S02 configuration')
    save(RUN / 'HOTPOT_PRIORITY_STATUS.json', {'utc': utc(), 'status': 'running',
        'dataset': 'hotpotqa', 'method': campaign.OUR, 'planned': 500,
        'unchanged_config': cfg, 'execution_epoch': epoch['id']})
    log('hotpot500_s02_prioritized', config=cfg, sample_sha256=sha(question_path),
        other_dataset_checkpoints_preserved=True, api_cap_cny=200)
    for start in range(0, 500, 20):
        instance.execute('hotpotqa', 'formal', campaign.OUR, cfg, cases[start:start+20])
    scores = read(campaign.TARGET / 'hotpotqa/formal' / campaign.OUR / cfg['id'] / 'scores.json')
    if scores['overall']['terminal_records'] != 500:
        raise ValueError('Priority Hotpot500 did not reach all planned terminal records')
    save(RUN / 'HOTPOT_PRIORITY_STATUS.json', {'utc': utc(), 'status': 'completed',
        'dataset': 'hotpotqa', 'method': campaign.OUR, 'scores': scores,
        'next': 'resume original baseline/control/ablation campaign', 'execution_epoch': epoch['id']})
    from .report import main as report
    report()
    log('hotpot500_s02_priority_completed', overall=scores['overall'],
        remaining_campaign_order='original', config_changed=False)


def install_priority(callback=None, owner=None):
    owner = campaign.Campaign if owner is None else owner
    callback = prioritize_hotpot if callback is None else callback
    previous = owner.baseline_main

    def baseline_main(self, dataset):
        if not getattr(self, '_hotpot_s02_priority_completed', False):
            callback(self)
            self._hotpot_s02_priority_completed = True
        if dataset == 'hotpotqa':
            return original_hotpot_baselines_complete()
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
