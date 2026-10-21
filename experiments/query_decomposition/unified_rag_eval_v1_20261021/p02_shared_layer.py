"""P02 callbacks over the existing baseline-v2 layers; no client or paid runner.

The original P02 package and baseline modules remain byte-identical. Execution
requires a separately approved plan revision and explicit removal of user holds.
"""
import json
from pathlib import Path

from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.retrieval import Retrieval
from experiments.query_decomposition.baseline_origin_binding_fix_20261017 import OriginSafeTransport
from src.p02_multigrain_timing_cachefix_v1.core import pipeline

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
METHOD = 'p02_multigrain_timing_cachefix_v1'
EXECUTION_VARIANT = 'p02_shared_baseline_v2'
REQUEST_LAYER = OriginSafeTransport
RETRIEVAL_LAYER = Retrieval


class ExecutionNotApproved(RuntimeError):
    pass


class RecordedRequestFailure(RuntimeError):
    def __init__(self, stage, request_id, receipt):
        self.stage, self.request_id, self.receipt = stage, request_id, receipt
        super().__init__('Request failed at '+stage+': '+receipt.get('status','unknown'))


def require_execution_approval():
    plan = json.loads((HERE/'PLAN.json').read_text(encoding='utf8'))
    approval = plan.get('execution_approval', {})
    if not approval.get('approved') or approval.get('plan_revision') != plan['revision']:
        raise ExecutionNotApproved('Current unified plan has no user execution approval')
    holds = [ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
             common.OUT/'STOP_REQUESTED',
             ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED',
             HERE/'STOP_REQUESTED']
    if any(p.exists() for p in holds):
        raise ExecutionNotApproved('Explicit user pause remains active; do not clear it automatically')


class P02Callbacks:
    def __init__(self, case, request_layer, retrieval_layer):
        self.case = pipeline.safe_case(case)
        if not isinstance(request_layer, REQUEST_LAYER):
            raise TypeError('P02 must use the existing origin-safe baseline-v2 request layer')
        if not isinstance(retrieval_layer, RETRIEVAL_LAYER):
            raise TypeError('P02 must use the existing baseline-v2 complete-input retrieval layer')
        self.requests, self.retrieval = request_layer, retrieval_layer
        self.requests.qid = self.case['qid']

    def retrieve(self, query):
        require_execution_approval()
        return self.retrieval.retrieve_many([{'qid':self.case['qid'],'query':query}],10)[0]

    def ask(self, stage, prompt, native_maximum, allow_final_truncation=False):
        require_execution_approval()
        # The existing layer records native_maximum but requests common.MAX_OUTPUT
        # and rejects length/missing usage. P02's old final-length exception cannot
        # turn an incomplete response into a scorable unified-protocol result.
        rid, value = self.requests.ask(stage, prompt, native_maximum)
        if value.get('status') != 'success':
            raise RecordedRequestFailure(stage,rid,value)
        if value.get('finish_reason') != 'stop' or not value.get('usage'):
            raise common.IntegrityError('Incomplete response cannot enter P02 history or final scoring')
        return rid,value['answer'],value['finish_reason']

    def execute(self, record_node=None):
        require_execution_approval()
        result = pipeline.execute(self.case,self.retrieve,self.ask,record_node=record_node)
        return {**result,'method':METHOD,'execution_variant':EXECUTION_VARIANT,
                'protocol_plan_revision':json.loads((HERE/'PLAN.json').read_text(encoding='utf8'))['revision']}
