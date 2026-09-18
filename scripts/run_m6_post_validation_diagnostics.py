"""Freeze and run the specified descriptive diagnostics after full validation.

The lightweight completion gate never opens answer records or contributions.
The original completed_inputs performs the original scientific checks again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
ROOT = RESEARCH / "m6_beir_validation_v1"
OUT = PROJECT / "analysis" / "hotpotqa_router"
PLAN = OUT / "m6_post_validation_diagnostic_plan_20260915.md"
FREEZE = OUT / "m6_post_validation_diagnostics_implementation_20260915.json"
JSON_OUTPUT = OUT / "m6_post_validation_diagnostics_20260915.json"
MARKDOWN_OUTPUT = OUT / "m6_post_validation_diagnostics_20260915.md"
PROTOCOL = "5b4faa41dc4a3ae83400d31a796354cdafb62cb461bfc6ba1a6f594b2b7bc785"
ACTION_FREEZE = "3dc5799af16211ec3c1fe06ae967ce9f33e954b45790e9585429317708def999"
ACTION_ARRAYS = "586ad2d400e993ef6172a259e71ab1cef0855bf8fc3e7a5fbd93e3257c2ac2b1"


class NotReady(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completion_gate(root=ROOT):
    """Require full original validation without opening any outcome payload."""
    root = Path(root)
    answers = root / "answers_v1"
    manifest = read(answers / "outcomes_manifest.json")
    if not (manifest.get("status") == manifest.get("stop_reason") == "complete" and
            manifest.get("successful_generations") == manifest.get("required_successful_generations") == 31254):
        raise NotReady("All31254 outcomes and the complete export are required")
    required = [answers / "complete_ledger_checks.json", root / "evaluation.json",
                root / "evaluation_separate_checks.json"]
    if not all(path.is_file() for path in required):
        raise NotReady("Complete ledger audit, main evaluation and separate numerical check are required")
    audit = read(required[0])
    require(audit.get("status") == "passed_collection_integrity" and audit.get("complete") is True,
            "Complete ledger audit did not pass")
    require(audit.get("successful_generations") == audit.get("required_successful_generations") == 31254,
            "Complete audit coverage mismatch")
    require(audit.get("protocol_sha256") == PROTOCOL, "Audit protocol mismatch")
    require(audit.get("outcomes_manifest_sha256") == sha(answers / "outcomes_manifest.json"),
            "Audit is stale relative to export")
    checked = read(required[2])
    require(checked.get("status") == "passed_separate_complete_pair_effects_and_group_intervals",
            "Separate numerical check did not pass")
    require(checked.get("protocol_sha256") == PROTOCOL and checked.get("queries") == 5209 and
            checked.get("groups") == 5206 and checked.get("complete_outcomes") == 31254,
            "Separate check coverage or protocol mismatch")
    require(checked.get("evaluation_sha256") == sha(required[1]), "Separate check is stale relative to evaluation")
    evaluation = read(required[1])
    require(evaluation.get("status") == "complete_fixed_M6_BEIR_dev_evaluation_pending_separate_check",
            "Unexpected main evaluation identity")
    require(evaluation.get("protocol_sha256") == PROTOCOL and evaluation.get("queries") == 5209 and
            evaluation.get("groups") == 5206 and evaluation.get("successful_answers") == 31254,
            "Main evaluation coverage or protocol mismatch")
    require(evaluation.get("outcomes_sha256") == manifest["outcomes"]["sha256"], "Evaluation uses different outcomes")
    require(evaluation.get("complete_ledger_checks_sha256") == sha(required[0]), "Evaluation uses a different audit")
    require(evaluation.get("actions_freeze_sha256") == ACTION_FREEZE, "Evaluation uses different actions")
    # Deliberately no positive-effect/significance requirement here.
    return {"manifest": manifest, "audit": audit, "evaluation": evaluation, "separate_check": checked}


def implementation_sources():
    return [Path(__file__), PROJECT / "scripts" / "m6_post_validation_math.py",
            PROJECT / "scripts" / "check_m6_post_validation_math.py", PLAN]


def freeze():
    require(not FREEZE.exists(), "Preserve the frozen diagnostic implementation")
    require(not (ROOT / "evaluation.json").exists(), "Freeze before opening the complete primary result")
    manifest = read(ROOT / "answers_v1" / "outcomes_manifest.json")
    require(manifest["successful_generations"] < 31254 and manifest["status"] == "incomplete",
            "This freeze is restricted to the incomplete collection stage")
    require(sha(ROOT / "protocol.json") == PROTOCOL and sha(ROOT / "actions_freeze.json") == ACTION_FREEZE and
            sha(ROOT / "actions.npz") == ACTION_ARRAYS, "Frozen scientific input changed")
    value = {"status": "diagnostic_implementation_frozen_before_complete_primary_results",
        "frozen_at_unix": time.time(), "protocol_sha256": PROTOCOL,
        "actions_freeze_sha256": ACTION_FREEZE, "actions_npz_sha256": ACTION_ARRAYS,
        "source_sha256": {str(path): sha(path) for path in implementation_sources()},
        "collection_successes_at_freeze": manifest["successful_generations"],
        "outcome_payloads_read": 0, "new_model_fits": 0, "new_hypothesis_tests": 0,
        "scope": "Fixed descriptive diagnostics after full original evaluation, regardless of effect sign"}
    write_new(FREEZE, value)
    print(json.dumps(value), flush=True)


def run():
    require(not JSON_OUTPUT.exists() and not MARKDOWN_OUTPUT.exists(), "Preserve completed diagnostic outputs")
    frozen = read(FREEZE)
    require(frozen["source_sha256"] == {str(path): sha(path) for path in implementation_sources()},
            "Diagnostic implementation differs from its pre-result freeze")
    metadata = completion_gate()
    sys.path.insert(0, str(RESEARCH))
    import numpy as np
    import finish_m6_beir_validation as finish
    import m6_post_validation_math as mathematics
    import check_m6_post_validation_math as independent

    with finish.collection.common.exclusive_run(ROOT / "answers_v1"):
        finish.verify_analysis(PROTOCOL)
        require(metadata["separate_check"]["checker_sha256"] == sha(RESEARCH / "finish_m6_beir_validation.py"),
                "Separate checker source changed")
        require(metadata["separate_check"]["separate_numeric_core_sha256"] == sha(RESEARCH / "check_pool_evaluation.py"),
                "Separate numeric core changed")
        _protocol, manifest, actions, records = finish.completed_inputs(PROTOCOL)
        require(sha(ROOT / "actions.npz") == ACTION_ARRAYS, "Frozen score arrays changed")
        with np.load(ROOT / "actions.npz", allow_pickle=False) as archive:
            scores = archive["M6"].astype(np.float64)
            require(np.array_equal(archive["query_ids"], actions["query_ids"]), "Score query order mismatch")
            require(np.array_equal(archive["group_ids"], actions["group_ids"]), "Score group order mismatch")
        require(scores.shape == (5209,) and np.array_equal(scores > 0, actions["M6_switch"]), "Score actions mismatch")
        bins = (scores <= -.4296875, (scores > -.4296875) & (scores <= 0),
                (scores > 0) & (scores <= .1611328125), scores > .1611328125)
        require([int(mask.sum()) for mask in bins] == [2039, 2035, 568, 567], "Prespecified groups changed")
        lookup = {str(q): i for i, q in enumerate(actions["query_ids"])}
        utilities = np.full((5209, 2, 3), np.nan)
        seen = set()
        for row in records:
            key = (row["query_id"], row["action"], row["repeat_id"])
            require(key not in seen and key[0] in lookup and key[1] in ("bm25", "dense") and
                    type(key[2]) is int and key[2] in (0, 1, 2), "Unexpected or duplicate outcome")
            seen.add(key)
            i = lookup[key[0]]
            require(row["group_id"] == actions["group_ids"][i], "Outcome group mismatch")
            utilities[i, int(key[1] == "dense"), key[2]] = row["f1"]
        require(len(seen) == 31254 and np.isfinite(utilities).all(), "Incomplete paired utility array")
        result = mathematics.compute(scores, actions["group_ids"], utilities)
        checked = independent.verify(scores, actions["group_ids"], utilities, result)
        require(checked.get("status") == "passed_independent_fsum_diagnostic_reconstruction",
                "Independent diagnostic reconstruction did not pass")
        gap = (utilities[:, 0, :] - utilities[:, 1, :]).mean(axis=1)
        selected = (scores > 0).astype(float)
        direct = np.stack([selected * gap, -(1 - selected) * gap], axis=1)
        with np.load(ROOT / "contributions.npz", allow_pickle=False) as contribution:
            require(sha(ROOT / "contributions.npz") == metadata["evaluation"]["contributions_sha256"],
                    "Formal contributions changed")
            require(np.array_equal(contribution["query_ids"], actions["query_ids"]) and
                    np.array_equal(contribution["group_ids"], actions["group_ids"]), "Formal contribution order mismatch")
            require(contribution["F1_primary"].shape == (5209, 2), "Unexpected formal contribution shape")
            error = float(np.max(np.abs(contribution["F1_primary"] - direct)))
            require(error < 1e-12, "Diagnostics disagree with formal per-query contributions")
        require(np.max(np.abs(direct.mean(axis=0) - metadata["evaluation"]["primary_mean"])) < 1e-12,
                "Diagnostics disagree with formal main effects")
        value = {"status": "complete_prespecified_descriptive_diagnostics_separately_checked",
            "protocol_sha256": PROTOCOL, "implementation_freeze_sha256": sha(FREEZE),
            "evaluation_sha256": sha(ROOT / "evaluation.json"),
            "evaluation_separate_checks_sha256": sha(ROOT / "evaluation_separate_checks.json"),
            "outcomes_sha256": manifest["outcomes"]["sha256"], "diagnostics": result,
            "independent_numeric_check": checked, "formal_contributions_maximum_error": error,
            "formal_primary_mean": metadata["evaluation"]["primary_mean"],
            "formal_primary_intervals": metadata["evaluation"]["primary_intervals"],
            "formal_standard_met": metadata["evaluation"]["statistical_and_point_magnitude_standard_met"],
            "new_model_fits": 0, "new_hypothesis_tests": 0, "new_paid_calls": 0,
            "created_at_unix": time.time(), "scope": "One consumed conditional-source cohort; descriptive mechanism clues only"}
        markdown = render_markdown(value)
        write_new(JSON_OUTPUT, value)
        with MARKDOWN_OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown)
    print(json.dumps({"status": value["status"], "json": str(JSON_OUTPUT), "report": str(MARKDOWN_OUTPUT),
                      "new_model_fits": 0, "new_hypothesis_tests": 0, "new_paid_calls": 0}), flush=True)


def render_markdown(value):
    result = value["diagnostics"]
    fmt = lambda number: f"{number:+.9f}"
    lines = ["# M6 完整验证后的机制诊断", "",
        "本报告按效果揭晓前固定的预案生成；完整账本、正式主检验和另一路数值复核均已通过完整性检查。",
        "这些描述解释固定候选在一个条件来源中的表现，不生成新的策略或显著性结论。", "",
        "## 1. 原正式检验", "",
        "统计与点幅度标准：**" + ("通过" if value["formal_standard_met"] else "未通过") + "**。",
        "单位为原始F1（0–1）。两主比较各使用原双侧97.5%区间；预定点幅度要求同时至少+0.01。", "",
        "| 原主比较 | 平均F1差 | 原97.5%区间 |", "|---|---:|---|"
    ]
    for name, mean, interval in zip(("M6 − Dense", "M6 − BM25"), value["formal_primary_mean"], value["formal_primary_intervals"]):
        lines.append(f"| {name} | {fmt(mean)} | [{fmt(interval[0])}, {fmt(interval[1])}] |")
    lines += ["", f"覆盖 {result['queries']} 个query、{result['groups']} 个既定group，每个动作三个重复。",
        "一次条件来源的结果不能直接证明稳定部署收益或语义Answer Correctness。", "",
        "## 2. 固定分数组", "",
        "分组边界为−0.4296875、0、+0.1611328125，相同分数不拆组；s>0选BM25，否则Dense。",
        "near/far只表示该动作侧分数相对零的位置，不代表经过校准的正确概率。", "",
        "| 组 | query / group | 平均分数 | 组内平均gap | 正 / 负 / 零gap数 |", "|---|---:|---:|---:|---|"
    ]
    for row in result["bins"]:
        count = row["gap_counts"]
        lines.append(f"| {row['name']} | {row['query_count']} / {row['group_count']} | {fmt(row['mean_score'])} | "
                     f"{fmt(row['mean_gap'])} | {count['positive']} / {count['negative']} / {count['zero']} |")
    lines += ["", "gap为同一query的BM25−Dense三重复平均F1差。判零容差为1e−12，仅影响计数，不截断效用。",
        "组内均值除以该组query数；下表贡献除以全部query数，四组之和复现两个正式主比较。", "",
        "| 组 | 对M6−Dense的总体贡献 | 对M6−BM25的总体贡献 |", "|---|---:|---:|"
    ]
    for row in result["bins"]:
        contribution = row["contribution"]
        lines.append(f"| {row['name']} | {fmt(contribution['M6_minus_Dense'])} | {fmt(contribution['M6_minus_BM25'])} |")
    rank = result["within_action_rank_contrasts"]
    lines += ["", f"两个预定同侧描述差：D_near−D_far = {fmt(rank['D_near_minus_D_far'])}；"
        f"B_far−B_near = {fmt(rank['B_far_minus_B_near'])}。",
        "它们仅提供分数与观测效用关联的线索，不证明某个新阈值会改善策略。", "",
        "## 3. 固定动作效用分解", "", "| 量 | 数值 |", "|---|---:|"
    ]
    labels = {"bm25_fraction": "BM25选择比例p", "mean_gap": "总体平均gap μ",
        "beneficial_mass": "切换BM25的正效用贡献", "harmful_mass": "切换BM25的负效用贡献（正幅度）",
        "missed_bm25_positive_mass": "未选BM25时放弃的正gap贡献", "random_same_count_expected_gain": "同数量均匀随机选择的解析期望pμ",
        "selection_alignment": "选择关联alignment", "M6_minus_Dense": "M6−Dense", "M6_minus_BM25": "M6−BM25"}
    for key, label in labels.items():
        lines.append(f"| {label} | {fmt(result['global_decomposition'][key])} |")
    lines += ["", "核对的恒等式：`M6−Dense = pμ + alignment`；`M6−BM25 = alignment − (1−p)μ`。",
        "这是固定分数和观测gap的代数分解；随机项只是解析期望，不替代强固定基线，也不构成因果归因。", "",
        "## 4. 三重复稳定性", "", "| 量 | repeat0 | repeat1 | repeat2 | 均值 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    repeated = result["repeats"]
    series = [("M6−Dense", repeated["M6_minus_Dense"]), ("M6−BM25", repeated["M6_minus_BM25"])]
    series += [(name + "组内gap", item) for name, item in repeated["bin_mean_gaps"].items()]
    for label, item in series:
        numbers = item["values"] + [item["mean"], item["min"], item["max"]]
        lines.append("| " + label + " | " + " | ".join(fmt(number) for number in numbers) + " |")
    categories = {"all_zero": "三次均判零", "nonnegative_some_positive": "非负且至少一次为正",
        "nonpositive_some_negative": "非正且至少一次为负", "crosses_zero": "同时出现正与负"}
    lines += ["", "| 三重复符号类别 | 全体数量 / 比例 | D_far | D_near | B_near | B_far |",
              "|---|---:|---:|---:|---:|---:|"]
    stability = result["sign_stability"]
    for key, label in categories.items():
        total = stability["global"][key]
        entries = [f"{total['count']} / {total['fraction']:.4%}"]
        for name in ("D_far", "D_near", "B_near", "B_far"):
            part = stability["by_bin"][name][key]
            entries.append(f"{part['count']} / {part['fraction']:.4%}")
        lines.append("| " + label + " | " + " | ".join(entries) + " |")
    spread = stability["gap_range"]
    lines += ["", f"每条query三次gap极差的中位数为 {spread['median']:.9f}，p90为 {spread['p90']:.9f}。",
        "同名repeat没有共享随机种子的保证；三重复不能当作三份独立query样本，也不能单独确证长期噪声或服务端漂移。", "",
        "## 5. 核查与后续使用", "",
        f"另一求和实现已复核全部诊断量；与原正式逐query贡献的最大绝对差为 {value['formal_contributions_maximum_error']:.3g}。",
        "未新增模型拟合、收费调用、阈值搜索或假设检验。完整表格均予报告，不能只选有利分组。",
        "本批数据用于机制判断后属于已消费证据；任何新头、偏移或动作规则仍需另一份预先冻结的独立验证。", "",
        "依据：[诊断预案](m6_post_validation_diagnostic_plan_20260915.md)、[紧凑数值结果](m6_post_validation_diagnostics_20260915.json)、"
        "[原正式结果](../../../work/router_research/m6_beir_validation_v1/evaluation.json)、"
        "[原数值复核](../../../work/router_research/m6_beir_validation_v1/evaluation_separate_checks.json)。", ""
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "check-ready", "run"))
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "check-ready":
        try:
            completion_gate()
        except NotReady as error:
            print(json.dumps({"status": "waiting_for_complete_verified_main_evaluation", "reason": str(error), "outcome_payloads_read": 0}))
        else:
            print(json.dumps({"status": "ready_for_prespecified_descriptive_diagnostics", "outcome_payloads_read": 0}))
    else:
        run()
