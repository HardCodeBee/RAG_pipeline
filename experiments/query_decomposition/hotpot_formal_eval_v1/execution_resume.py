"""Continue a migrated known prefix without dispatching its RPC again."""
import json

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import canonical, digest
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_rate_policy import RateTransport


class ResumeTransport(RateTransport):
    def ask(self, stage, prompt, max_tokens):
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0.0, 'max_tokens': max_tokens,
                   'extra_body': {'enable_thinking': False}}
        key = digest([self.store.namespace, payload])
        alias = self.store.get('resume_request_alias', key)
        binding = {'qid': self.qid, 'stage': stage}
        if alias is None or binding not in alias['allowed_bindings']:
            return super().ask(stage, prompt, max_tokens)
        row = self.store.db.execute('SELECT payload,status,result FROM requests WHERE id=?',
                                    (alias['origin_request_id'],)).fetchone()
        if row is None or row[1] != 'success' or canonical(json.loads(row[0])) != canonical(payload):
            raise ValueError('Migrated prefix does not match this exact known-success payload')
        result = json.loads(row[2])
        if result.get('status') != 'success':
            raise ValueError('A failed or unknown result cannot become a resume alias')
        self.store.event('logical_api_call', qid=self.qid, stage=stage,
                         request_id=alias['origin_request_id'], cache_hit=True,
                         max_tokens=max_tokens, resumed_known_prefix=True,
                         new_namespace_request_key=key)
        # The original evidence row is already bound to this qid/stage/request.
        # No new result, attempt, billing liability, or evidence claim is invented.
        return alias['origin_request_id'], result
