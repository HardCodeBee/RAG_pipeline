"""Approved compact metrics; no model invocation or LLM-call-count metric."""
import math


def complete_sum(values):
    """Unknown measurement stays unknown; an applicable empty set sums to zero."""
    values=list(values)
    if any(v is None for v in values):return None
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Measurement must be a nonnegative finite value or unknown')
    return sum(values)


def timing_for_query(components):
    local=[c.get('processing_seconds') for c in components if c['kind']=='local']
    api=[]
    for c in components:
        if c['kind']=='api':
            # Historical replay is not a new TTFT observation, even if its source
            # receipt contains an old first-chunk time or a previously measured TTFT.
            api.append(c.get('client_ttft_seconds') if c.get('observation')=='measured_live_content' else None)
    return {'local_processing_seconds':complete_sum(local),'client_ttft_seconds':complete_sum(api),
            'timing_definition':'separate local component sum and client TTFT sum; not total pipeline pure-compute latency'}


def quality_for_primary_table(scores, *, expected_qids, planned=500, confirmed_unrecoverable=None):
    if planned!=500:raise ValueError('Current formal primary table requires the frozen500 cohort')
    expected=set(expected_qids)
    scores=list(scores)
    if len(expected)!=500:raise ValueError('Expected IDs must be the frozen500 cohort')
    ids=[s['qid'] for s in scores]
    if len(ids)!=len(set(ids)) or not set(ids).issubset(expected):
        raise ValueError('Duplicate or out-of-cohort prediction')
    valid=[s for s in scores if s.get('f1') is not None and s.get('em') is not None]
    if any(not 0<=s[k]<=1 for s in valid for k in ['f1','em']):raise ValueError('Invalid answer score')
    if len(valid)>planned:raise ValueError('Too many scored questions')
    if len(valid)<planned and not confirmed_unrecoverable:
        return {'status':'not_ready_500','F1':None,'EM':None}
    if confirmed_unrecoverable and not all(confirmed_unrecoverable.get(k) for k in ['user_decision_id','evidence_receipt','reason']):
        raise ValueError('Irrecoverable partial result requires explicit user decision and evidence')
    n=len(valid)
    total=sum(s['f1'] for s in valid)
    value={'status':'complete500' if n==500 else 'confirmed_partial',
           'F1':total/n if n else None,'EM':sum(s['em'] for s in valid)/n if n else None}
    if n<500:
        value.update(scorable=n,planned=planned,exception=confirmed_unrecoverable,
                     F1_full_bounds=[total/planned,(total+planned-n)/planned])
    return value
