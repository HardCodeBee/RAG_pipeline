"""Verified QD batching plus offline accounting for unsubmitted trace events."""
from .common import HERE,RUN,read,save,sha,log
from . import campaign,scoring,scoring_safe_trace
from .qd_batch_epoch import install

def main():
    epoch={'id':'offline_unsubmitted_trace_v1','source_sha256':sha(HERE/'scoring_safe_trace.py'),
       'launcher_sha256':sha(__file__),'original_scoring_sha256':sha(HERE/'scoring.py'),
       'native_metric_functions':'imported unchanged from original scoring.metric',
       'change':'exclude planned-but-unsubmitted log entries from submitted-call cost statistics',
       'answer_generation_or_f1_rules_changed':False,'missing_answers_imputed':False}
    file=campaign.TARGET/'SCORING_COMPATIBILITY_EPOCH.json'
    if file.exists() and read(file)!=epoch:raise ValueError('Scoring compatibility epoch changed')
    if not file.exists():save(file,epoch);log('offline_accounting_compatibility_epoch_activated',**epoch)
    scoring.evaluate=scoring_safe_trace.evaluate
    install()
    campaign.main()

if __name__=='__main__':main()
