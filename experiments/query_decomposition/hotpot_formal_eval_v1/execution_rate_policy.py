"""Serial infrastructure scheduling; no prompts, answers, or method decisions."""
import json
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Budget, Transport

RATE_CODES = {'Throttling.RateQuota', 'Throttling.RequestQuota', 'rate_limit_exceeded', 'Throttling'}


class RateBudget(Budget):
    def __init__(self, path, cap=200):
        super().__init__(path, cap)
        # Retain a previous slowdown on a restart; never reset it to the faster rate.
        self.db.execute("INSERT OR IGNORE INTO scheduler VALUES('adaptive_minimum_interval',2)")
        self.db.commit()

    def note_throttle(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            values = dict(self.db.execute('SELECT key,value FROM scheduler'))
            interval = max(6, values.get('adaptive_minimum_interval', 2))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('adaptive_minimum_interval',?)", (interval,))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('next_start',?)", (max(values.get('next_start', 0), time.time() + interval),))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return interval

    def wait_for_start(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            values = dict(self.db.execute('SELECT key,value FROM scheduler'))
            now = time.time()
            start = max(now, values.get('next_start', now))
            if start - now > 300:
                raise RuntimeError('Rate scheduler clock/backlog requires audit')
            interval = max(2, values.get('adaptive_minimum_interval', 2))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('next_start',?)", (start + interval,))
            self.db.execute("INSERT OR REPLACE INTO scheduler VALUES('attempt_count',?)", (int(values.get('attempt_count', 0)) + 1,))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        remaining = max(0, start - now)
        while remaining > 0:
            delay = min(60, remaining)
            time.sleep(delay)
            remaining -= delay


class RateTransport(Transport):
    def ask(self, *args, **kwargs):
        before = self.store.db.execute('SELECT coalesce(max(id),0) FROM attempts').fetchone()[0]
        try:
            # The original transport still owns payloads, request identities,
            # liability reservation, HTTP retry rules, and unknown-request stops.
            return super().ask(*args, **kwargs)
        finally:
            for (raw,) in self.store.db.execute('SELECT result FROM attempts WHERE id>?', (before,)):
                if not raw:
                    continue
                result = json.loads(raw)
                if result.get('http_status') == 429 and result.get('error_code') in RATE_CODES:
                    interval = self.budget.note_throttle()
                    self.store.event('infrastructure_rate_slowdown', minimum_interval_seconds=interval,
                                     reason='explicit provider rate-limit response', method_input_changed=False)
                    break
