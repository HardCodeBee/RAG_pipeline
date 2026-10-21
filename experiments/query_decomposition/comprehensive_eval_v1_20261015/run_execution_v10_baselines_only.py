"""Execution scope only: defer S02, matched-reader controls, and ablations."""
import sys

from . import campaign, legacy_ablations, run_execution_v3, run_execution_v7
from .common import RUN, log, read


def require_policy():
    policy = read(RUN / 'OUR_METHOD_PAUSED_BY_USER.json')
    if policy.get('active') is not True or policy.get('resume_requires_user_instruction') is not True:
        raise ValueError('An explicit active user method-pause policy is required')
    return policy


def install_baseline_guard(owner=None):
    owner = campaign.Campaign if owner is None else owner
    previous = owner.execute

    def execute(self, dataset, split, mid, config, cases):
        require_policy()
        if not mid.startswith('b') or split not in {'dev', 'formal'}:
            raise RuntimeError('User-paused method/control/ablation cannot be submitted: ' + mid)
        return previous(self, dataset, split, mid, config, cases)

    owner.execute = execute


def main():
    require_policy()
    previous_install = run_execution_v3.install

    def install():
        previous_install()
        install_baseline_guard()

    def defer_legacy_ablations(instance):
        log('legacy_method_ablations_deferred_by_user', new_api_calls=0)

    run_execution_v3.install = install
    legacy_ablations.run = defer_legacy_ablations
    # Reuse every verified accounting, scan, rate, and transport overlay.
    # The previous priority entrypoints are deliberately not installed.
    sys.argv = [sys.argv[0], '--stage', 'baselines']
    log('baseline_only_scope_activated', paused_methods=[campaign.OUR, campaign.CONTROL, *campaign.ABLATIONS],
        frozen_algorithms_changed=False, existing_answers_preserved=True, api_cap_cny=200)
    run_execution_v7.main()


if __name__ == '__main__':
    main()
