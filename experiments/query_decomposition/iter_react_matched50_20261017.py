"""Equal-maximum-budget reevaluation, preserving the unchanged ReAct baseline.

Iter-RetGen k10/T5 and ReAct k10/max_search5 each allow at most 50 document
slots. Actual searches, repeated documents, generation calls and tokens differ.
Old native and matched25 outputs are retained and are not primary comparisons.
"""
import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import httpx
from openai import OpenAI
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.common import *
from experiments.query_decomposition.baseline_origin_binding_fix_20261017 import install
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install as install_offsets
from experiments.query_decomposition.hotpot_formal_eval_v1.compare_scores import paired

JOB = OUT / "matched50"
ITER = "b05_iter_retgen_bge"
REACT = "b03_react_dense"
CONFIG = {"id": "k10_t5", "k": 10, "iterations": 5}


def validate():
    verify()
    install()
    for dataset in DATASETS:
        react = read(PARENT / dataset / "BASELINE_CONFIGS_FROZEN.json")["selected"][REACT]
        assert react == {"id": "k10_s5", "k": 10, "max_search": 5}
    # Exercise both real method controllers, including ReAct's sixth Finish turn.
    class Memory:
        def __init__(self): self.data = {}
        def get(self, kind, key): return self.data.get((kind, key))
        def put(self, kind, key, value):
            old = self.get(kind, key)
            assert old is None or old == value
            self.data[kind, key] = value
        def export(self): pass
    class Search:
        def __init__(self): self.calls = []
        def retrieve(self, qid, query, k):
            self.calls.append((qid, query, k))
            return {"docs": [{"doc_id": str(i), "block": "Passage " + str(i)} for i in range(k)]}
    class Calls:
        def __init__(self, react): self.n = 0; self.react = react
        def ask(self, stage, prompt, maximum):
            self.n += 1
            answer = ("Action: Search[query " + str(self.n) + "]" if self.n <= 5 else "Action: Finish[alpha]") if self.react else "Final answer: alpha"
            return str(self.n), {"status": "success", "answer": answer, "finish_reason": "stop"}
    counts = {}
    for name, is_react, cfg in [("iter_retgen_bge", False, CONFIG), ("react_dense", True, {"k":10,"max_search":5})]:
        store, retrieval, calls = Memory(), Search(), Calls(is_react)
        module = importlib.import_module("experiments.query_decomposition.baseline_integrity_repair_v2_20261017.methods." + name + ".method")
        module.run([{"qid":"offline-budget-case", "question":"Which object is alpha?"}], retrieval, calls, store, cfg)
        assert len(retrieval.calls) == 5 and all(c[2] == 10 for c in retrieval.calls)
        assert store.get("prediction", "offline-budget-case")["prediction"] == "alpha"
        counts[name] = {"retrieval_calls":len(retrieval.calls), "maximum_document_slots":50, "generation_calls":calls.n}
    value = {"utc":utc(), "passed":True, "source_sha256":sha(Path(__file__)),
             "checks":{"frozen_v2_unchanged":True,"origin_patch_validated":True,
                       "all_three_react_configs_k10_s5":True,"actual_controllers_equal_search_allowance":counts},
             "api_calls":0, "f1_used_for_selection":False}
    save(JOB / "VALIDATION.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def freeze():
    verify()
    validation = read(JOB / "VALIDATION.json")
    if not validation["passed"] or validation["source_sha256"] != sha(Path(__file__)):
        raise IntegrityError("Matching offline validation required")
    receipt = {"utc":utc(),"id":"iter_react_matched50_20261017", "source_sha256":sha(Path(__file__)),
        "authorization":"User requested equal Iter-RetGen/ReAct budgets and reevaluation of Iter-RetGen",
        "iter_config":CONFIG,"react_config":{"id":"k10_s5","k":10,"max_search":5},
        "definition":"Equal maximum5 retrieval calls,10 passages per call,50 document slots; actual use and generation tokens are reported separately. ReAct may Finish early.",
        "datasets":DATASETS,"formal_per_dataset":500,"dev_integrity_per_dataset":20,
        "execution_order":"2wiki then hotpotqa then musique; each paired evaluation first. Resume remaining non-Iter native baseline repairs afterwards, within the existing global budget.",
        "primary_iter_directory":"matched50/<dataset>/formal/b05_iter_retgen_bge/k10_t5",
        "primary_react_directory":"main/<dataset>/formal/b03_react_dense/k10_s5",
        "superseded_for_primary_comparison":["native Iter k5/T2 or T3","unstarted matched25 formal comparison"],
        "unchanged":["sample and references","model and endpoint","method prompts and demonstrations","retrieval backend and corpus","full-input/full-output safeguards","native ReAct parameters","all non-Iter native baseline parameters"],
        "old_results_deleted_or_overwritten":False,"new_model_training":False,
        "source_v2_config_sha256":sha(OUT/"CONFIG_FROZEN.json"),
        "origin_binding_validation_sha256":sha(OUT/"origin_binding_fix_20261017/VALIDATION.json"),
        "global_budget_cny":200,"protected_submission_cap_cny":PROTECTED_CAP,
        "selection":"T5/k10 fixed from unchanged ReAct allowance, not selected by F1; dev20 checks technical integrity only"}
    path = JOB / "CONFIG_FROZEN.json"
    if path.exists():
        existing = read(path)
        if {k:v for k,v in existing.items() if k!="utc"} != {k:v for k,v in receipt.items() if k!="utc"}:
            raise IntegrityError("Equal-budget protocol differs from its freeze")
        return existing
    once(path, receipt)
    journal("matched50_protocol_frozen", receipt=str(path), source_sha256=receipt["source_sha256"],
            original_react_unchanged=True, iter_k=10, iter_iterations=5, old_results_preserved=True)
    return receipt


def comparison(dataset):
    left = JOB / dataset / "formal" / ITER / CONFIG["id"]
    right = OUT / "main" / dataset / "formal" / REACT / "k10_s5"
    native_id = read(PARENT/dataset/"BASELINE_CONFIGS_FROZEN.json")["selected"][ITER]["id"]
    native = OUT / "main" / dataset / "formal" / ITER / native_id
    a = {r["qid"]:r for r in rows(left/"scores.jsonl")}
    b = {r["qid"]:r for r in rows(right/"scores.jsonl")}
    result = {"utc":utc(), "dataset":dataset,"planned":500,"maximum_retrieval_calls_each":5,
        "per_retrieval_k_each":10,"maximum_document_slots_each":50,
        "iter_overall":read(left/"scores.json")["overall"],"react_overall":read(right/"scores.json")["overall"],
        "iter_minus_react":paired(a,b,seed=2026101707),"same_scorable_cohort_required_for_pairing":True,
        "not_equal_realized_cost":True,"hallucination_judge_called":False}
    if (native/"scores.jsonl").exists():
        result["iter50_minus_old_native_iter"] = paired(a,{r["qid"]:r for r in rows(native/"scores.jsonl")},seed=2026101707)
    usage = {}
    for label, directory in [("iter", left),("react", right)]:
        with sqlite3.connect((directory/"checkpoint.sqlite3").resolve().as_uri()+"?mode=ro",uri=True) as db:
            preds=[json.loads(raw) for raw, in db.execute("select value from data where kind='prediction'")]
        calls=[p.get("iterations" if label=="iter" else "searches",0) for p in preds]
        if any(n>5 for n in calls):raise IntegrityError("Observed retrieval count exceeds matched budget")
        tokens={r["qid"]:r for r in rows(directory/"per_query_tokens.jsonl")}
        usage[label]={"mean_retrieval_calls_terminal":sum(calls)/len(calls),"max_retrieval_calls":max(calls),
            "mean_returned_document_slots_terminal":10*sum(calls)/len(calls),
            "logical_prompt_tokens":sum(r["logical_prompt_tokens"] for r in tokens.values()),
            "logical_completion_tokens":sum(r["logical_completion_tokens"] for r in tokens.values()),
            "new_returned_prompt_tokens":sum(r["new_returned_prompt_tokens"] for r in tokens.values()),
            "new_returned_completion_tokens":sum(r["new_returned_completion_tokens"] for r in tokens.values()),
            "per_query_tokens":str(directory/"per_query_tokens.jsonl")}
    result["actual_usage"] = usage
    save(JOB / dataset / "COMPARISON.json",result)
    c=result["iter_minus_react"]
    text=["相同最大检索预算：Iter k10/T5；ReAct k10/最多5次Search。",dataset,
          "固定500题；缺失保留，算法失败计0；共同可评分样本上计算同题差值。",
          "Iter: "+json.dumps(result["iter_overall"],ensure_ascii=False),
          "ReAct: "+json.dumps(result["react_overall"],ensure_ascii=False),
          "Iter-ReAct: "+json.dumps(c,ensure_ascii=False),
          "实际用量: "+json.dumps(usage,ensure_ascii=False),
          "原T2/T3及matched25结果保留，不能混称本次等预算结果。"]
    (JOB/dataset/"REPORT.txt").write_text("\n".join(text)+"\n",encoding="utf8")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    if args.validate_only:validate();return
    freeze()
    if (OUT/"STOP_REQUESTED").exists():raise IntegrityError("A stop request is still pending")
    runtime=install()
    options=get_api_option(2)
    receipt=read(OUT/"CONFIG_FROZEN.json")
    if options["model"]!=MODEL or options["base_url"]!=receipt["base_url"]:
        raise IntegrityError("Generator identity changed")
    install_offsets()
    budget=runtime.Budget(OUTPUT/"budget.sqlite3",200,2/3)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key("api_code_qwen.txt",allowed_root=ROOT,provider="dashscope"))
            network=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,
                limits=httpx.Limits(max_connections=8,max_keepalive_connections=8,keepalive_expiry=10)),timeout=60))
            client=stack.enter_context(OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],base_url=receipt["base_url"],
                max_retries=0,timeout=60,http_client=network))
            runner=runtime.Runner(receipt,StreamingClient(client),budget)
            for dataset in DATASETS:
                save(JOB/"STATUS.json",{"utc":utc(),"status":"running","dataset":dataset,"phase":"dev_integrity","budget":budget.summary()})
                scored,audit=runner.run_method("matched50",dataset,"dev",ITER,CONFIG,20)
                if scored["overall"]["terminal_records"]!=20 or not audit["passed"]:
                    raise IntegrityError("Matched50 technical gate failed")
                once(JOB/dataset/"DEV_GATE.json",{"passed":True,"planned":20,"f1_used_for_selection":False,"config":CONFIG})
                save(JOB/"STATUS.json",{"utc":utc(),"status":"running","dataset":dataset,"phase":"formal_iter","budget":budget.summary()})
                runner.run_method("matched50",dataset,"formal",ITER,CONFIG,500)
                react=receipt["native_selected_configs_preserved"][dataset][REACT]
                runner.run_method("main",dataset,"formal",REACT,react,500)
                result=comparison(dataset)
                journal("matched50_dataset_completed",dataset=dataset,iter_overall=result["iter_overall"],react_overall=result["react_overall"],budget=budget.summary())
            # Continue the already-authorized repairs without generating obsolete
            # native Iter or the superseded matched25 formal runs again.
            for dataset in DATASETS:
                for mid,cfg in receipt["native_selected_configs_preserved"][dataset].items():
                    if mid!=ITER:
                        runner.run_method("main",dataset,"formal",mid,cfg,500)
            save(JOB/"STATUS.json",{"utc":utc(),"status":"completed","all_three_pairs_completed":True,"remaining_non_iter_baselines_completed":True,"budget":budget.summary()})
            save(OUT/"progress.json",{"utc":utc(),"status":"completed","stage":"matched50_priority_and_remaining_baselines","budget":budget.summary()})
    except BaseException as error:
        record={"utc":utc(),"status":"stopped_for_audit","reason":str(error),"error":type(error).__name__,"budget":budget.summary()}
        save(JOB/"STATUS.json",record)
        save(OUT/"progress.json",record)
        journal("matched50_stopped",reason=str(error),error=type(error).__name__,budget=budget.summary())
        raise
    finally:
        budget.db.close()


if __name__=="__main__":main()
