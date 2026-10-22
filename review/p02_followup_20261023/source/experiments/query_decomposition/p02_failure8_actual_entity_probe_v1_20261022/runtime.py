"""Reuse I8 receipt/ledger protection; add return traces without editing I/shared."""
import json
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022.run import (
    receipt_integrity, algorithm_failure, normal_complete, SubmissionAwareStrictAsk,
)


class CandidateAsk(SubmissionAwareStrictAsk):
    def _retire(self, rid, stage, payload, attempts, reason):
        row = self.cache.db.execute("SELECT result FROM requests WHERE id=?", (rid,)).fetchone()
        result = json.loads(row[0]) if row and row[0] else {}
        # A provider can return either of these fields without content/id/usage.
        # Presence of a non-null field is sufficient to prevent blind resubmission.
        if any(k in result and result[k] is not None for k in ("refusal", "finish_reason")):
            raise RuntimeError("Held refusal/finish_reason returned evidence; preserve receipt/reservation")
        return super()._retire(rid, stage, payload, attempts, reason)
