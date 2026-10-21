"""Reconcile both cohorts, preserved versions, API attempts and training flags."""
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import hashlib
import unicodedata

from .review import HERE, ROOT, OUT, OLD, rows, sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    development = load(OUT / "dev.RESULTS.json")
    confirmation = load(OUT / "holdout.RESULTS.json")
    frozen = load(HERE / "FINAL_CANDIDATE_FROZEN.json")
    assert frozen["candidate"] == "v8" and not frozen["holdout_text_inspected_before_freeze"]
    assert sha(HERE / "TRAINING_PROMPT_v8.txt") == frozen["prompt_hashes"]["v8"]
    assert sha(HERE / "holdout.questions.frozen.jsonl") == frozen["holdout_sha256"]
    assert sha(HERE / "AUTHORED_dev.json") == frozen["review_sha256"]
    before = {r["index"]: r for r in load(HERE / "AUTHORED_dev_through_v8.frozen.json")["cases"]}
    now = {r["index"]: r for r in load(HERE / "AUTHORED_dev.json")["cases"]}
    for i in before:
        for v in ("v2", "v3", "v4", "v5", "v6", "v7", "v8"):
            assert before[i][v] == now[i][v]
    dev_cases = rows(HERE / "dev.questions.frozen.jsonl")
    hold_cases = rows(HERE / "holdout.questions.frozen.jsonl")
    manifest = ROOT / "outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data/split_manifest.json"
    protected = {r["query_sha256"] for r in load(manifest)["records"] if r["project_assignment"] in {"project_test", "project_dev", "reserved_test_family", "reserved_dev_family"}}
    assert not any(hashlib.sha256(" ".join(unicodedata.normalize("NFKC", r["question"]).casefold().split()).encode()).hexdigest() in protected for r in dev_cases + hold_cases)
    assert not ({(r["dataset"], r["qid"]) for r in dev_cases} & {(r["dataset"], r["qid"]) for r in hold_cases})
    flagged = rows(HERE / "hard_cases_for_4b/CANDIDATES.jsonl")
    hard_ids = {(r["source"]["dataset"], r["source"]["native_id"]) for r in flagged}
    assert len(hard_ids) == len(flagged) == 36
    assert not hard_ids & {(r["dataset"], r["qid"]) for r in hold_cases}
    for r in flagged:
        assert len(r["flag"]["failed_versions"]) >= 2
        assert r["training_preparation"]["target_output"] is None
        assert not r["training_preparation"]["admitted_to_training"]
        assert not r["training_preparation"]["protected_metadata_overlap"]
        for failure in r["failure_history"]:
            raw = load(ROOT / failure["raw_result_file"])
            assert raw["status"] == "success" and raw["finish_reason"] == "stop"
            assert raw["answer"] == failure["literal_output"]
    request_states, attempts = Counter(), 0
    for path in OUT.glob("*/api/*/*/*/checkpoint.sqlite3"):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            reqs = db.execute("SELECT id,status FROM requests").fetchall()
            assert len(reqs) == 1
            request_id, state = reqs[0]
            n = db.execute("SELECT count(*) FROM attempts WHERE request_id=?", (request_id,)).fetchone()[0]
            assert n == 1, path
            attempts += n
            request_states[state] += 1
    assert attempts == 600 and not request_states["in_flight"]
    probe = [load(p) for p in (OUT / "probe_probe/cases/probe").glob("*/*.json")]
    assert len(probe) == 2 and all(r["answer"] == '["CONTROL_ONLY"]' for r in probe)
    paired = confirmation["pairs"][0]
    audit = {"all_600_requests_have_one_attempt": True, "request_states": dict(request_states),
             "unknown_requests_replayed": False, "constant_system_probe_passed": True,
             "original_prompt_versions_unchanged": True, "all_single_pass_reviews_preserved_after_selection": True,
             "candidate_and_rubric_frozen_before_holdout": True,
             "holdout_tuning_performed": False, "hard_cases_include_no_holdout_ids": True,
             "hard_case_failure_outputs_match_raw_successful_responses": True,
             "candidate_gold_targets_written": 0, "actual_training_files_modified": False,
             "payload_checks": {"development": development["audit"], "holdout": confirmation["audit"]},
             "manual_reviews": "Current Codex same context, nonblind; no claim of independent human gold",
             "case_counts": {"development": 60, "holdout": 60, "old_exposed_cases_used_only_for_training_flags": 36},
             "selected120_protected_normalized_query_hash_overlaps": 0,
             "stage_request_counts": {"original_four_dev": 240, "three_new_single_pass_dev": 180, "editor_dev": 58, "system_probe": 2, "holdout": 120},
             "study_cost_basis": "returned token usage at existing pinned list-price estimate, not verified invoice",
             "study_estimated_cny": confirmation["study_usage_list_price_cny"],
             "study_unknown_reserved_cny": confirmation["study_unknown_reserved_cny"],
             "global_estimated_cny": confirmation["global_usage_list_price_cny"],
             "global_unknown_reserved_cny": confirmation["global_unknown_reserved_cny"]}
    (OUT / "EXECUTION_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conclusion = {"selected_candidate": "v8", "old_control": "v4", "holdout_planned": 60,
                  "candidate_pass": confirmation["stats"]["v8"]["verdicts"]["pass"],
                  "control_pass": confirmation["stats"]["v4"]["verdicts"]["pass"],
                  "paired_available": paired["n"], "paired_candidate_pass": paired["after_pass"], "paired_control_pass": paired["before_pass"],
                  "paired_gains": len(paired["gained"]), "paired_losses": len(paired["lost"]),
                  "ninety_percent_target_met": False, "statistical_superiority_claimed": False,
                  "mu_transfer_improved": False, "runtime_rag_configuration_changed": False,
                  "future_4b_training_hard_candidates": 36, "holdout_queries_reserved_for_training": 0}
    (OUT / "CONFIRMATION_CONCLUSION.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["自包含多粒度分解：新题四版对照、逐轮优化与保留集确认", "",
             "结论：v8是已测试候选中开发集通过最多的单次生成prompt。保留60题v8通过42题(70.00%)，v4通过35题(58.33%)；尚未接近整体90%目标。",
             "保留集共同完整55题：v8通过39，v4通过35，8题转为通过、4题失去通过；点估计增加7.27个百分点，未宣称统计显著。",
             "两组缺失数不同，完整60题的11.67个百分点差值包含可用性差异，不全部归因为语义提升。", "",
             "样本与口径：", "固定120道不同新题，开发60+保留60，三数据集每阶段各20。排除既有2368训练query、旧36/24例、列出的既有开发/正式样本及保护集已知ID。类型分层非自然总体比例；ID/规范化文本排除不证明语义或预训练去污染。",
             "Hotpot每阶段bridge14/comparison6；2Wiki四类各5；MuSiQue2-hop7/3-hop7/4-hop6。保留集在候选冻结后才打开，不据保留集修改prompt。",
             "语义通过要求F忠实与未知保留、S单问独立、C有用中间信息覆盖、U无冗余背景四项同时通过；JSON合法率另计。不确定与API缺失保留完整分母。",
             "所有评审为当前Codex同一上下文非盲初审，无人工金标/独立裁判保证；不读标准答案、语料、参考分解或下游F1。", "",
             "开发60题：", "版本\t通过\t失败\t不确定\t不可评\t通过率"]
    for version, s in development["stats"].items():
        c = s["verdicts"]
        lines.append(f"{version}\t{c.get('pass',0)}\t{c.get('fail',0)}\t{c.get('uncertain',0)}\t{c.get('unavailable',0)}\t{s['pass_rate_full']:.2%}")
    lines += ["", "v4/v5同为37通过，按预先确定的短prompt平分规则选择v4为旧版对照(5107字节 vs5139字节)。四版共同完整49题通过数为25/29/31/32，不能据完整分母平分就认定v4比v5语义更好。", "",
              "优化改动与去留：",
              "v6：在v5后增加三个针对性示例，处理末问代词、泛身份与未经限定的亲属路径。42/60，提升但仍有that singer/team及多余属性。",
              "v7：精简成5规则+8正例，覆盖漏桥接、比较属性、已知信息、性别与消歧。40/60；未采用，出现记忆影片年份1939及新指代错误。",
              "v8：用未知中间描述识别、运算所需属性两种构造操作生成辅助问题，保留多粒度及原题自然出现政策。43/60，作为单次候选冻结。",
              "v9：同一API额外审一次原query+v8初稿，58次编辑调用，两个上游缺失保留。42/60，未采用。共同完整55题只净增1题，修复2/33/60，损害13/45；不算单次prompt收益，也不把初稿计算当免费。",
              "固定qwen-plus-2025-12-01、temperature0、thinking关闭、max1024、并发3；系统/用户角色、精确prompt和输入逐请求审计。未知请求不重发。", "",
              "保留60题：", "方案\t通过\t失败\t不确定\t不可评\t通过率"]
    for v in ("v4", "v8"):
        s = confirmation["stats"][v]; c = s["verdicts"]
        lines.append(f"{v}\t{c.get('pass',0)}\t{c.get('fail',0)}\t{c.get('uncertain',0)}\t{c.get('unavailable',0)}\t{s['pass_rate_full']:.2%}")
    lines += ["", "数据集\tv4通过/20\tv8通过/20"]
    for ds in ("hotpotqa", "2wiki", "musique"):
        lines.append(ds + "\t" + str(confirmation["stats"]["v4"]["by_dataset"][ds].get("pass",0)) + "\t" + str(confirmation["stats"]["v8"]["by_dataset"][ds].get("pass",0)))
    lines += ["", "v8保留集明确失败16题，重叠检查计数F9/S6/C1/U3；不确定1题，API缺失1题。2Wiki小样本20/20不能外推成总体100%；MuSiQue8/20比v4的9/20低。核心残留为跨题指代、对象/关系或修饰语变化、冗余泛检索。",
              "真实失败：保留23把'nation called the nobilities commonwealth'变成'headquarters of the nobilities commonwealth'；保留30把American singer改为出生地；保留58把语言形式后来叫什么改成Holy Roman Empire后来叫什么。没有借读支持材料或答案修改这些判断。", "",
              "未来4B困难样本：",
              "按用户新要求，从旧已观察36题和本轮开发60题中标记至少两个单次版本明确语义失败的query，共36道：Hotpot14、2Wiki8、Mu14；旧36中8道、新开发60中28道。",
              "各队列最后测试版本状态：仍失败17，已修复16，不确定1，未返回2。旧队列最后版本为v5，新队列为v8；不是17道都已在v8上测试失败。",
              "30道需写正确分解金标，6道含不确定审阅需先复核原query/解释。错误输出只作反例；target_output为空，实际SFT文件未修改。保留60题未被用于困难样本挑选。",
              "候选保存完整query、来源ID/类型、各版失败原始输出、失败理由、最新参考输出与状态；后续4B评测排除ID/规范化query清单已保存。今后入训后不得当作未见测试题。", "",
              "费用与执行：",
              f"600次请求各一次尝试；状态{dict(request_states)}。按既有单价账本估算本轮已返回用量¥{confirmation['study_usage_list_price_cny']:.6f}，未核销未知预留¥{confirmation['study_unknown_reserved_cny']:.6f}；实际账单未核对。",
              f"全局估算已返回用量¥{confirmation['global_usage_list_price_cny']:.6f}＋未核销预留¥{confirmation['global_unknown_reserved_cny']:.6f}，低于用户¥200总上限。",
              "结论限于API分解质量初审及本分层样本；没有训练4B，没有运行新的下游RAG/F1，也没有改变既有冻结RAG配置。",
              "完整变更、diff与采样/候选冻结文件在experiments同名目录；逐题复核、原始请求/返回、统计和执行审计在本输出目录。"]
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "passed", "requests": attempts, "confirmation": conclusion, "study_estimated_cny": confirmation["study_usage_list_price_cny"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
