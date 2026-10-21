"""Post-hoc legacy200 diagnostics, separate from independent new samples."""
import importlib
import json
from .common import ROOT,HERE,RUN,OUTPUT,read,rows,sha,save,once,utc,log
from .campaign import TARGET,OUR,CONTROL,ABLATIONS
from .execution import ExactTransport,run_parallel
from .retrieval import Retrieval
from .scoring import evaluate
from .methods.ablations.support import seed_exact_responses
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.compare_scores import paired,holm

def run(campaign):
    profile=ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/compact_100_200_20261014/inputs_frozen'
    source=OUTPUT/'s02_child4_compact_formal200_20261014/formal'/OUR/'frozen_s02_child4'
    cases=rows(profile/'formal.questions.jsonl');positions=read(OUTPUT/'s02_child4_compact_formal200_20261014/HELPER_POSITIONS_FROZEN.json')['positions']
    from .campaign import quarantine_unknown,verify
    root=TARGET/'legacy200_diagnostics';stores={}
    try:
        for block in range(0,200,20):
            for mid in [CONTROL,*ABLATIONS]:
                verify(campaign.receipt)
                cfg={'id':'fixed_legacy_packets','source_directory':str(source),'helper_positions':positions}
                directory=root/mid
                if mid not in stores:
                    identity={'method':mid,'scope':'post_hoc_legacy200_only','config':cfg,'questions_sha256':sha(profile/'formal.questions.jsonl'),
                      'source_config_sha256':sha(OUTPUT/'s02_child4_compact_formal200_20261014/CONFIG_FROZEN.json'),
                      'campaign_sha256':sha(TARGET/'CAMPAIGN_FROZEN.json'),'generator':campaign.options['model']}
                    stores[mid]=Store(directory,identity);quarantine_unknown(stores[mid])
                    seed_exact_responses(source,stores[mid],cases)
                store=stores[mid];remaining=[c for c in cases[block:block+20] if store.get('prediction',c['qid']) is None]
                if remaining:
                    liability=campaign.budget.summary()
                    if liability['returned_usage_list_price_cny']+liability['unreconciled_reserved_cny']>=195:raise RuntimeError('Protected global budget reserve reached')
                    retrieval=Retrieval(store,resources=campaign.resource('hotpotqa'))
                    if 'hotpotqa' in campaign.handles:retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse=campaign.handles['hotpotqa']
                    transport=ExactTransport(store,campaign.client,campaign.budget,campaign.options['model'])
                    module=importlib.import_module(campaign.receipt['protocol']['methods'][mid]['module'])
                    campaign.progress(status='running',dataset='legacy_hotpot200',split='post_hoc_diagnostic',method=mid,block_questions=len(remaining))
                    print(json.dumps({'start':'legacy200','method':mid,'block':block//20}),flush=True)
                    from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
                    from urllib.parse import urlsplit
                    with provider_dns(urlsplit(campaign.options['base_url']).hostname,store):run_parallel(module,remaining,retrieval,transport,store,cfg)
                store.export(full=True)
                evaluate('hotpotqa',directory,profile/'formal.references.local.jsonl')
        full={r['qid']:r for r in rows(source/'scored.local.jsonl')}
        results={}
        for mid in [CONTROL,*ABLATIONS]:
            values={r['qid']:r for r in rows(root/mid/'scores.jsonl')}
            results[mid]={'scores':read(root/mid/'scores.json'),**paired(full,values,seed=2026101501)}
        pvalues={k:v['two_sided_group_sign_flip_p'] for k,v in results.items() if 'two_sided_group_sign_flip_p' in v}
        for mid,p in holm(pvalues).items():results[mid]['holm_p_four_ablations']=p
        value={'utc':utc(),'scope':'post_hoc diagnostics on previously reviewed200; independent confirmation uses new formal subsets',
          'planned':200,'results':results,'completed':all(v['scores']['overall']['terminal_records']==200 for v in results.values())}
        save(root/'RESULTS.json',value)
        return value
    finally:
        for store in stores.values():store.export(full=True);store.db.close()
