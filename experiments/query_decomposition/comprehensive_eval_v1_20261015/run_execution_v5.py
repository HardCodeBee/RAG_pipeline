"""Verified execution overlays; every original frozen algorithm stays intact."""
import time
from .common import HERE, RUN, read, sha
from . import run_execution_v3, run_execution_v4, throughput_epoch


def main():
    epoch = read(RUN / 'THROUGHPUT_ENTRYPOINT_FROZEN.json')
    for name, expected in epoch['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Throughput entrypoint source changed: ' + name)
    from .offset_compatibility import install as install_offsets
    install_offsets()
    original_install = run_execution_v3.install

    def install():
        original_install()
        throughput_epoch.install()
        from . import campaign
        previous = campaign.Campaign.execute

        def execute(self, dataset, split, mid, config, cases):
            started = time.perf_counter()
            before = sum(s.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]
                         for s in self.stores.values())
            result = previous(self, dataset, split, mid, config, cases)
            after = sum(s.db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]
                        for s in self.stores.values())
            if after > before:
                from .common import log
                log('execution_epoch_block_timing', dataset=dataset, split=split, method=mid,
                    config=config['id'], new_terminal_records=after-before,
                    seconds=time.perf_counter()-started, epoch='three_slots_hybrid_prefetch_v1')
            return result

        campaign.Campaign.execute = execute

    run_execution_v3.install = install
    run_execution_v4.main()


if __name__ == '__main__': main()
