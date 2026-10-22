"""Prospective frozen20 first, then optional historicalquality/expense references."""
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib

from p02_multigrain_timing_cachefix_v1.core.pipeline import safe_case
from experiments.query_decomposition.p02_ordinary_failure_challenge8_v1_20261022 import prepare as util
from .native_ask import usage_ok
from . import resource_contract

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FORMAL=util.FORMAL;FORMAL_OUT=util.FORMAL_OUT;PACKAGE=util.PACKAGE
FEASIBILITY=ROOT/"analysis/p02_m4_wiki20_feasibility_20261022"
CANDIDATES=FEASIBILITY/"CANDIDATES20.prospective.jsonl"
PREFLIGHT=ROOT/"outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
read,rows,sha,rel=util.read,util.rows,util.sha,util.rel
INPUT_RATE,OUTPUT_RATE=.8e-6,2e-6



def assemble():
    # The independent selector writes all20 before this function opens any
    # historicalcase/prediction, cost, native answer or score reference.
    selected=rows(CANDIDATES);feasibility=read(FEASIBILITY/"FEASIBILITY.json")
    assert len(selected)==len({r["id"]for r in selected})==20
    assert len({" ".join(r["question"].casefold().split())for r in selected})==20
    assert all(r["dataset"]=="2wiki"for r in selected)
    assert feasibility["old_quality_or_completion_or_score_availability_not_a_filter"]is True
    assert feasibility["no_reader_change_or_localtruth_filter"]is True
    assert feasibility["no_passive723_or270_exclusion"]is True and feasibility["valid_empty_D_allowed"]is True
    public,publicproof=util.public_index()
    qfile=FORMAL/"queries1500.id_query.frozen.jsonl"
    qindex={r["id"]:(i+1,r)for i,r in enumerate(rows(qfile))}
    assert sha(qfile)==public[rel(qfile)]["sha256"]and sha(FORMAL/"v12.txt")==public[rel(FORMAL/"v12.txt")]["sha256"]
    data=[];provenance=[]
    for item in selected:
        position,q=qindex[item["id"]];rawpath=ROOT/item["raw_file"];raw=read(rawpath)
        assert position==item["formal_position_one_based"]and sha(rawpath)==item["raw_sha256"]
        assert raw["index"]==position and raw["id"]==item["id"]==q["id"]
        assert raw["query"]==item["question"]==q["query"]and raw["parsed"]==item["supporting_questions"]
        assert raw["format_valid"]and not raw["truncated"]and not raw["generation_error"]
        assert raw["actual_prompt"].startswith((FORMAL/"v12.txt").read_text(encoding="utf8").rstrip())
        assert item["actual_R02_array_exact"]is True and item["semantic_repairs_applied"]is False
        assert item["source_R02_remote_inputs_sha256"]==sha(FORMAL/"REMOTE_INPUTS.frozen.json")
        expected=hashlib.sha256(("p02-m4-wiki20-v1|20261022|"+item["id"]).encode()).hexdigest()
        assert item["selection_rank_sha256"]==expected
        clean=safe_case({key:item[key]for key in("id","qid","dataset","question","supporting_questions")})
        runtime=FORMAL_OUT/"runtime/2wiki"/(clean["qid"]+".json")
        assert read(runtime)==clean and sha(runtime)==public[rel(runtime)]["sha256"]
        data.append(clean)
        provenance.append(dict(id=clean["id"],formal_position_one_based=position,raw_file=rel(rawpath),raw_sha256=sha(rawpath),
            public_runtime=rel(runtime),public_runtime_sha256=sha(runtime),public_container=public[rel(runtime)]["container"],
            actual_R02_array_exact=True,manual_or_semantic_repairs_applied=False,
            source_R02_remote_inputs_sha256=item["source_R02_remote_inputs_sha256"],
            source_R02_checkpoint_manifest_sha256=item["source_R02_checkpoint_manifest_sha256"],
            prior723_semantic_exposure=item["prior723_semantic_exposure"],
            passive270_review_not_an_exclusion=True,old_quality_or_availability_not_selection_inputs=True,
            raw_file_itself_prior_publication_not_claimed=True,all_inference_fields_exact_public_runtime=True))
    assert Counter(r["dataset"]for r in data)=={"2wiki":20}
    from .history_resolver import ROUND_ORDER,resolve,proxy
    policy=read(HERE/"HISTORY_RESOLVER_POLICY.frozen.json")
    assert policy["round_order"]==list(ROUND_ORDER)and policy["selection_by_score"]is False
    costs=[resolve(clean,public)for clean in data]
    proxies={}
    if any(not r["historical_quality_available"]for r in costs):
        value=proxy(public);proxies["2wiki"]=value
        costs=[r if r["historical_quality_available"]else{**r,"reference_CNY":value["mean_CNY"],
            "reference_kind":value["reference_kind"],"proxy_complete_queries":value["complete_queries"],
            "missing_reason":"neither_predeclared_round_passes_strict_native_same5field_receipt_retrieval_contract"}for r in costs]
    for r in costs:r["resolver_policy_sha256"]=sha(HERE/"HISTORY_RESOLVER_POLICY.frozen.json")
    wanted={r["id"]for r in data};refs=[]
    for row in rows(FORMAL/"references1500.local.frozen.jsonl"):
        if row["id"]in wanted:refs.append({key:deepcopy(row[key])for key in("id","qid","dataset","answer","answer_id")})
    assert len(refs)==20 and{r["id"]for r in refs}==wanted
    source=read(PREFLIGHT)["manifests"]["2wiki"]
    resources={"2wiki":dict(identity=resource_contract.identity(source["manifest"],source["sha256"]),contract=resource_contract.contract(source["manifest"],source["sha256"]))}
    publicproof.update(prospective_candidates=rel(CANDIDATES),prospective_candidates_sha256=sha(CANDIDATES),
        selection_feasibility=rel(FEASIBILITY/"FEASIBILITY.json"),selection_feasibility_sha256=sha(FEASIBILITY/"FEASIBILITY.json"),
        selected_inference_fields_exact_public_runtime=True,runtime_proofs=provenance,corpus_origin="Same frozen portable2Wiki publicbenchmark corpus/index",
        all_future_retrieved_text_previously_pushed_claimed=False,no_credentials_or_private_documents_in_inference=True,
        endpoint_configuration_sha256=sha(PACKAGE/"config.json"),old_preflight_not_a_new_live_readiness_assertion=True)
    return data,costs,provenance,refs,resources,proxies,publicproof
