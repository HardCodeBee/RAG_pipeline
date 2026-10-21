"""Bounded restart; watch manual QDMR returns and index readiness, resume safely."""
import os
import subprocess
import sys
import time
from .common import ROOT,HERE,RUN,read,save,utc,log

def main():
    stop=RUN/'STOP_REQUESTED';status=RUN/'SUPERVISOR_STATUS.json';errors=0;passes=0
    from .campaign import TARGET
    with (RUN/'campaign.out.log').open('a',encoding='utf8') as out,(RUN/'campaign.err.log').open('a',encoding='utf8') as err:
        while not stop.exists():
            passes+=1
            save(status,{'utc':utc(),'status':'running_campaign_pass','pid':os.getpid(),'pass':passes,'consecutive_failures':errors})
            child=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','experiments.query_decomposition.comprehensive_eval_v1_20261015.campaign'],cwd=ROOT,stdout=out,stderr=err)
            save(status,{'utc':utc(),'status':'running_campaign_pass','pid':os.getpid(),'child_pid':child.pid,'pass':passes,'consecutive_failures':errors})
            code=child.wait()
            # Reporting can never submit API requests; preserve partial runs.
            reporter=subprocess.run([sys.executable,'-X','utf8','-B','-m','experiments.query_decomposition.comprehensive_eval_v1_20261015.report'],cwd=ROOT,stdout=out,stderr=err)
            complete=(TARGET/'RESULTS.json').exists() and read(TARGET/'RESULTS.json')['all_requested_evaluations_complete']
            if code==0 and reporter.returncode==0 and complete:
                save(status,{'utc':utc(),'status':'completed','pid':os.getpid(),'passes':passes});log('requested_api_evaluation_completed');return
            if code:
                errors+=1
                progress=read(TARGET/'progress.json') if (TARGET/'progress.json').exists() else {}
                reason=progress.get('reason','campaign_failed_before_progress')
                permanent=any(s in reason for s in ['Frozen campaign','Prior frozen','after freeze','Authorized API spending cap','Protected global budget','Less than90','baseline parameters'])
                if permanent or errors>=10:
                    save(status,{'utc':utc(),'status':'halted_for_audit','exit_code':code,'reason':reason,'consecutive_failures':errors});return
                save(status,{'utc':utc(),'status':'bounded_recovery_wait','exit_code':code,'reason':reason,'consecutive_failures':errors})
                time.sleep(min(60,5*errors))
            else:
                errors=0
                save(status,{'utc':utc(),'status':'waiting_for_index_or_manual_remote_return','pid':os.getpid(),'passes':passes,'next_check_seconds':60})
                time.sleep(60)
        save(status,{'utc':utc(),'status':'stopped_by_request','pid':os.getpid()})

if __name__=='__main__':main()
