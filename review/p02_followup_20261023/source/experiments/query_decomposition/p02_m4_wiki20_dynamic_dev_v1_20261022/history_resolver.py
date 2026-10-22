"""First valid source in fixed r000,r100 order; never select by final score."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sqlite3

from p02_multigrain_timing_cachefix_v1.core.cache import canonical
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline,safe_case,merge
from .native_ask import integrity

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/"src/p02_multigrain_timing_cachefix_v1"
FORMAL_OUT=ROOT/"outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021"
ROUND_ORDER=("r000","r100")


def strict_source(path,public,expected=None):
    relative=str(path.relative_to(ROOT)).replace("\\","/")
    raw=path.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    assert public[relative]["sha256"]==source_sha,"source_public_hash"
    case=json.loads(raw);clean=safe_case({k:case[k]for k in("id","qid","dataset","question","supporting_questions")})
    if expected is not None:assert clean==expected,"five_fields_or_actualD_mismatch"
    assert case["status"]in("success","success_truncated")and isinstance(case["prediction"],str),"incomplete_flow"
    calls=case["logical_calls"];index={c["stage"]:c for c in calls}
    assert calls and len(index)==len(calls),"missing_or_duplicate_logical_stages"
    byquery={}
    for n in case["nodes"]:
        assert n["original_retrieval"]["query"]==n["request"]
        if n["rewritten_retrieval"]is not None:assert n["rewritten_retrieval"]["query"]==n["rewritten_query"]
        assert merge(n["original_retrieval"],n["rewritten_retrieval"])==n["reader_packet"],"reader_merge"
        for field in("original_retrieval","rewritten_retrieval"):
            p=n[field]
            if p is None:continue
            assert p["qid"]==clean["qid"]and len(p["docs"])==len({d["doc_id"]for d in p["docs"]})==10,"incomplete_retrieval"
            if p["query"]in byquery:assert byquery[p["query"]]==p,"same_query_packet_mismatch"
            byquery[p["query"]]=p
    config=json.loads((PACKAGE/"config.json").read_text(encoding="utf8"));dbs={};seen=[]
    try:
        def ask(stage,prompt,maximum,allow_final_truncation=False):
            c=index[stage];dbpath=(ROOT/c["source"]).resolve()
            if dbpath not in dbs:dbs[dbpath]=sqlite3.connect(dbpath.as_uri()+"?mode=ro",uri=True)
            row=dbs[dbpath].execute("SELECT payload,status,result FROM requests WHERE id=?",(c["request_id"],)).fetchone()
            payload=build_payload(config["generation"],prompt,maximum)
            assert row and row[0]==canonical(payload)and row[1]=="success","native_wire_mismatch"
            r=json.loads(row[2]);assert r.get("requested_model")in(None,payload["model"])
            bound={**r,"requested_model":payload["model"]}
            assert integrity(bound,payload["model"]),"receipt_model_or_usage"
            assert r["finish_reason"]in(("stop","length")if allow_final_truncation else("stop",))and r.get("answer","").strip()and not r.get("refusal"),"incomplete_receipt"
            assert c["status"]=="success"and c["full_payload_match_verified"]is True and c["usage"]==r["usage"]
            assert c["payload_sha256"]==hashlib.sha256(canonical(payload).encode()).hexdigest()
            seen.append(stage);return c["request_id"],r["answer"].strip(),r["finish_reason"]
        result=Pipeline(config).execute(clean,lambda q:deepcopy(byquery[q]),ask)
        for key in("prediction","final_prompt","final_context","helper","history_all"):
            assert result[key]==case[key],"native_trajectory_"+key
        assert len(seen)==len(calls)and len(result["nodes"])==len(case["nodes"])
        for a,b in zip(result["nodes"],case["nodes"]):
            for key in("request","answer","initial_answer","child_prompt","rewrite_prompt","raw_rewrite","rewritten_query","reader_packet","prior_complete_answers"):
                assert a[key]==b[key],"native_node_"+key
    finally:
        for db in dbs.values():db.close()
    expense=sum(c["usage"]["prompt_tokens"]*.8e-6+c["usage"]["completion_tokens"]*2e-6 for c in calls)
    return dict(id=clean["id"],dataset=clean["dataset"],reference_CNY=expense,
        reference_kind="same_query_complete_historical_flow",historical_quality_available=True,
        final_prediction_reference_only=case["prediction"],source=relative,reference_source=relative,source_sha256=source_sha,
        strict_native_messages_receipts_and_retrieval_verified=True)


def resolve(clean,public):
    attempts=[]
    for roundname in ROUND_ORDER:
        path=FORMAL_OUT/"rounds"/roundname/"cases/2wiki/p02_multigrain_timing_cachefix_v1"/(clean["qid"]+".json")
        try:value=strict_source(path,public,expected=clean)
        except (AssertionError,KeyError,ValueError,RuntimeError,OSError,sqlite3.Error)as exc:
            attempts.append(dict(round=roundname,source=str(path.relative_to(ROOT)).replace("\\","/"),accepted=False,
                reason=type(exc).__name__+":"+str(exc)[:160]));continue
        attempts.append(dict(round=roundname,source=value["source"],accepted=True))
        return {**value,"reference_round":roundname,"reference_resolution_attempts":attempts}
    return dict(id=clean["id"],dataset="2wiki",historical_quality_available=False,final_prediction_reference_only=None,
        reference_source=None,reference_resolution_attempts=attempts)


def proxy(public):
    # Only needed when both selected-query rounds are unusable. The expense
    # proxy population is the strict native public r000 complete-flow snapshot.
    values=[]
    directory=FORMAL_OUT/"rounds/r000/cases/2wiki/p02_multigrain_timing_cachefix_v1"
    for path in sorted(directory.glob("*.json")):
        try:value=strict_source(path,public)
        except(AssertionError,KeyError,ValueError,RuntimeError,OSError,sqlite3.Error):continue
        values.append({k:value[k]for k in("id","dataset","source","source_sha256","reference_CNY")})
    assert values,"No legal complete-query expense proxy; report rather than inventzero"
    return dict(dataset="2wiki",reference_kind="same_method_same_dataset_complete_query_mean_proxy",complete_queries=len(values),
        mean_CNY=sum(v["reference_CNY"]for v in values)/len(values),sources=values,source_policy="strict public native r000 completeflows;no highestscoreselection")
