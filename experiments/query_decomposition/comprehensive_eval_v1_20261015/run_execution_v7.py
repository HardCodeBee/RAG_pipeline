"""Verified execution overlays; every original frozen algorithm stays intact."""
import time
import sqlite3
from .common import HERE, RUN, ROOT, read, sha
from . import run_execution_v3, run_execution_v4, throughput_epoch


def main():
    epoch = read(RUN / 'SCAN_API_ENTRYPOINT_FROZEN.json')
    for name, expected in epoch['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Throughput entrypoint source changed: ' + name)
    for name, expected in epoch.get('workspace_source_sha256', {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen scan dependency changed: ' + name)
    import faiss
    if faiss.__version__ != epoch['faiss_version']:
        raise ValueError('Validated FAISS version changed')
    from .offset_compatibility import install as install_offsets
    install_offsets()
    original_install = run_execution_v3.install

    def install():
        original_install()
        throughput_epoch.install()
        from . import scan_api_epoch
        from .exact_parallel_scan_v2 import ExactParallelScan
        scan_api_epoch.ParallelScan = ExactParallelScan
        scan_api_epoch.install()
        from . import campaign
        previous = campaign.Campaign.execute

        def execute(self, dataset, split, mid, config, cases):
            started = time.perf_counter()
            path = campaign.TARGET / dataset / split / mid / config['id'] / 'checkpoint.sqlite3'
            def count():
                if not path.exists(): return 0
                with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                    return db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]
            before = count()
            result = previous(self, dataset, split, mid, config, cases)
            after = count()
            if after > before:
                from .common import log
                log('execution_epoch_block_timing', dataset=dataset, split=split, method=mid,
                    config=config['id'], new_terminal_records=after-before,
                    seconds=time.perf_counter()-started, epoch='scan_api_capacity_v1')
            return result

        campaign.Campaign.execute = execute

    run_execution_v3.install = install
    run_execution_v4.main()


if __name__ == '__main__': main()
