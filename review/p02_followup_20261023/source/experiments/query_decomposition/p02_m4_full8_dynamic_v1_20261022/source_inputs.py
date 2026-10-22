"""Five inference fields from exactly the frozen M4 stage8 queries, never outputs."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from collections import Counter

from p02_multigrain_timing_cachefix_v1.core.pipeline import safe_case
from experiments.query_decomposition.p02_m4_reader_stage_probe8_v1_20261022 import prepare as stage
from .native_ask import usage_ok
from . import resource_contract

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
STAGE=stage.HERE;FORMAL=stage.source.FORMAL;FORMAL_OUT=stage.source.FORMAL_OUT
PACKAGE=stage.PACKAGE
PREFLIGHT=ROOT/"outputs/query_decomposition/p02_prompt_research_v1_20261022/REMOTE_RETRIEVAL_PREFLIGHT_PASSWORD.json"
read,rows,sha,rel=stage.read,stage.rows,stage.sha,stage.rel
INPUT_RATE,OUTPUT_RATE=.8e-6,2e-6


def assemble():
    # The selected8 were frozen before stage responses existed. No fresh stage
    # checkpoint, endpoint, blind key, semantic label or result is read here.
    stage.check()
    selected,selection,proofs,receipts,decomposition,stagepublic=stage.assemble()
    assert selected==rows(STAGE/"INPUTS8.frozen.jsonl")
    assert selection==read(STAGE/"SELECTION.frozen.json")
    assert len(selected)==len({r["id"]for r in selected})==8
    public,publicproof=stage.source.public_index()
    decomps={r["probe_id"]:r for r in decomposition}
    data=[];history=[];provenance=[]
    for item in selected:
        source=read(ROOT/item["source"])
        clean=safe_case({key:source[key]for key in("id","qid","dataset","question","supporting_questions")})
        assert clean["dataset"]=="2wiki"and clean["supporting_questions"]==item["supporting_questions"]
        runtime=FORMAL_OUT/"runtime/2wiki"/(clean["qid"]+".json")
        assert read(runtime)==clean and sha(runtime)==public[rel(runtime)]["sha256"]
        d=decomps[item["probe_id"]]
        assert sha(ROOT/d["raw_file"])==d["raw_sha256"] and d["array_exact"]is True
        assert sha(ROOT/item["source"])==item["source_sha256"]==public[item["source"]]["sha256"]
        calls=source["logical_calls"]
        assert source["status"]in("success","success_truncated")and isinstance(source["prediction"],str)and calls
        assert all(usage_ok(c)and c["status"]=="success"and c["full_payload_match_verified"]is True
            and c["finish_reason"]in(("stop","length")if c["stage"]=="legacy/final"else("stop",))for c in calls)
        value=sum(c["usage"]["prompt_tokens"]*INPUT_RATE+c["usage"]["completion_tokens"]*OUTPUT_RATE for c in calls)
        assert value>0
        data.append(clean)
        history.append(dict(id=clean["id"],dataset="2wiki",reference_CNY=value,reference_kind="same_query_complete_historical_flow",
            historical_quality_available=True,final_prediction_reference_only=source["prediction"],source=item["source"],source_sha256=item["source_sha256"]))
        provenance.append(dict(id=clean["id"],stage_probe_id=item["probe_id"],stage_selection_reason=next(r["selection_reason"]for r in selection["selected"]if r["id"]==clean["id"]),
            source=item["source"],source_sha256=item["source_sha256"],source_public_container=public[item["source"]]["container"],
            public_runtime=rel(runtime),public_runtime_sha256=sha(runtime),public_container=public[rel(runtime)]["container"],
            raw_file=d["raw_file"],raw_sha256=d["raw_sha256"],formal_position_one_based=d["formal1500_position_one_based"],
            actual_R02_array_exact=True,manual_or_semantic_repairs_applied=False,prior723_semantic_exposure=True,
            raw_file_itself_prior_publication_not_claimed=True,all_inference_fields_exact_public_runtime=True,
            source_R02_remote_inputs_sha256=sha(FORMAL/"REMOTE_INPUTS.frozen.json"),source_stage_protocol_sha256=sha(STAGE/"PROTOCOL.frozen.json")))
    refs=[];wanted={r["id"]for r in data}
    for row in rows(FORMAL/"references1500.local.frozen.jsonl"):
        if row["id"]in wanted:refs.append({k:deepcopy(row[k])for k in("id","qid","dataset","answer","answer_id")})
    assert len(refs)==8 and{r["id"]for r in refs}==wanted
    remote=read(PREFLIGHT);source=remote["manifests"]["2wiki"]
    resources={"2wiki":dict(identity=resource_contract.identity(source["manifest"],source["sha256"]),
        contract=resource_contract.contract(source["manifest"],source["sha256"]))}
    publicproof.update(source_stage_protocol_sha256=sha(STAGE/"PROTOCOL.frozen.json"),source_stage_inputs_sha256=sha(STAGE/"INPUTS8.frozen.jsonl"),
        selection="Exactly8 already frozenstagequeries;no stage response or dateprecision outcome used",runtime_proofs=provenance,
        selected_inference_fields_exact_public_runtime=True,corpus_origin="Same frozen portable2Wiki publicbenchmark corpus/index",
        all_future_retrieved_text_previously_pushed_claimed=False,no_credentials_or_private_documents_in_inference=True,
        endpoint_configuration_sha256=sha(PACKAGE/"config.json"),old_preflight_not_a_new_live_readiness_assertion=True)
    assert Counter(r["dataset"]for r in data)=={"2wiki":8}
    return data,history,provenance,refs,resources,{},publicproof
