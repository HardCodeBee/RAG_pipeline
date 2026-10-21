"""Replay saved method outputs with eight slots; audit drain/order invariants."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
import time
from .common import RUN, save, sha
from . import throughput_validate as replay
from .scan_api_epoch import CapacityAPI


def main():
    class WideReplay(replay.API):
        def __init__(self, requests):
            super().__init__(requests)
            self.pool.shutdown(wait=True)
            self.concurrency = 8
            self.pool = ThreadPoolExecutor(max_workers=8)
    class Probe(CapacityAPI):
        def __init__(self, fail=False):
            self.pool=ThreadPoolExecutor(max_workers=8)
            self.concurrency=8
            self.lock=Lock()
            self.active=self.maximum=self.completed=0
            self.fail=fail
        def _ask(self, qid, args, kwargs):
            with self.lock:
                self.active+=1
                self.maximum=max(self.maximum,self.active)
            try:
                time.sleep(.02)
                if self.fail and qid=='3':raise ValueError('Synthetic drain audit')
                return qid
            finally:
                with self.lock:
                    self.active-=1
                    self.completed+=1
    cases=[{'qid':str(i)} for i in range(24)]
    operations=[SimpleNamespace(index=i,args=(),kwargs={}) for i in range(24)]
    probes=[]
    for failure in [False,True]:
        api=Probe(failure)
        try:
            try:
                result=api.ask_group(operations,cases)
                assert not failure and result==[r['qid'] for r in cases]
            except ValueError:
                assert failure
        finally:api.close()
        assert api.maximum==8 and api.completed==24 and api.active==0
        probes.append({'failure_injected':failure,'max_concurrency':api.maximum,'all_calls_drained':True})
    replay.API=WideReplay
    def redirect(path,value):
        if path.name=='THROUGHPUT_OFFLINE_PARITY.json':
            path=RUN/'CAPACITY8_OFFLINE_PARITY.json'
            value.update(api_slots=8,production_slot_probes=probes,
                         wrapper_source_sha256=sha(__file__),paid_api_calls=0)
        save(path,value)
    replay.save=redirect
    replay.main()


if __name__=='__main__':main()
