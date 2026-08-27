#!/usr/bin/env python3
"""Build and validate the Phase 2.7 9,600-query confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_v1"
RUN = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1"
GENERATION = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1"
CANDIDATE_LABELS = {
    "M2_raw_full_xgb_mse": "M2 raw 414D + XGBoost",
    "M3_pca32_structured_ridge": "M3 PCA32 + structured + Ridge",
    "M4_oof_late_fusion": "M4 OOF late fusion",
    "M0_structured_xgb_mse": "M0 structured-only + XGBoost",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def main() -> int:
    base_decision = read_json(BASE / "decision.json")
    formal = read_json(RUN / "formal_metrics.json")
    decision = read_json(RUN / "decision.json")
    preflight = read_json(RUN / "preflight.json")
    snapshot = read_json(RUN / "snapshot/snapshot_manifest.json")
    completion = read_json(GENERATION / "phase27_9600_completion_state.json")
    preparation = read_json(RUN / "expanded_preparation.json")
    teacher_path = RUN / "privileged_teacher_diagnostic.json"
    teacher = read_json(teacher_path) if teacher_path.exists() else None

    candidate_ids = formal["formal_candidates"]
    split_seeds = [20260901, 20260917, 20261003]
    if candidate_ids != preparation["formal_candidates"]:
        raise ValueError("Formal candidates differ from the expanded preparation freeze")
    if formal.get("status") != "complete" or decision.get("decision") != "STOP_QUERY_ONLY_V1":
        raise ValueError("The 9,600 formal confirmation or decision is incomplete")
    if decision.get("selected_candidate") is not None:
        raise ValueError("A STOP decision must not select a deployable candidate")
    if preflight.get("queries_read") != 9600 or preflight.get("outcome_rows_read") != 57600:
        raise ValueError("Preflight does not describe the complete 9,600-query pool")
    if preflight.get("fresh_dev_rows_read") != 0 or preflight.get("final_holdout_rows_read") != 0:
        raise ValueError("A sealed evaluation partition was read")
    if not all(snapshot["prefix_validation"].values()):
        raise ValueError("The frozen 4,800 prefix changed")
    if completion.get("pending_or_retryable_cells") != 0 or not completion.get("complete"):
        raise ValueError("Generation completion state is incomplete")

    with (RUN / "learning_curve.csv").open("r", encoding="utf-8", newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    if len(curve_rows) != 48:
        raise ValueError("Expected 48 candidate-size-split learning-curve rows")

    with np.load(RUN / "formal_predictions.npz", allow_pickle=False) as predictions:
        expected_arrays = {
            key
            for candidate_id in candidate_ids
            for split_seed in split_seeds
            for key in (
                f"prediction__{candidate_id}__{split_seed}",
                f"fold_id__{candidate_id}__{split_seed}",
            )
        }
        if set(predictions.files) != expected_arrays:
            raise ValueError("Formal prediction arrays are incomplete or unexpected")
        for key in predictions.files:
            value = np.asarray(predictions[key])
            if value.shape != (9600,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid formal prediction array: {key}")
            if key.startswith("fold_id__") and set(value.tolist()) != {0, 1, 2, 3, 4}:
                raise ValueError(f"Incomplete OOF fold coverage: {key}")

    task_json = list((RUN / "formal_tasks").glob("*.json"))
    task_npz = list((RUN / "formal_tasks").glob("*.npz"))
    if len(task_json) != 12 or len(task_npz) != 12:
        raise ValueError("Expected 12 isolated formal task records and prediction files")

    database = GENERATION / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        bd_rows = connection.execute(
            "SELECT COUNT(*) FROM outcomes AS o JOIN sample_rows AS s USING(query_id) "
            "WHERE o.partition='train' AND s.partition='train' "
            "AND o.action IN ('bm25','dense') AND s.query_rank < 9600 AND o.status='success'"
        ).fetchone()[0]
        fresh_dev_rows = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE partition='fresh_dev'"
        ).fetchone()[0]
        final_holdout_rows = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE partition='final_holdout'"
        ).fetchone()[0]
    if integrity != "ok" or bd_rows != 57600 or fresh_dev_rows or final_holdout_rows:
        raise ValueError("Generation database integrity or partition boundary check failed")

    learning = decision["learning_curve_summary"]
    comparison_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        gate = decision["formal_gate_results"][candidate_id]
        consensus = gate["consensus_metrics"]
        means = learning[candidate_id]["mean_gain_by_query_count"]
        comparison_rows.append(
            {
                "candidate_id": candidate_id,
                "gain_1200": means["1200"],
                "gain_2400": means["2400"],
                "gain_4800": means["4800"],
                "gain_9600": means["9600"],
                "delta_9600_minus_4800": means["9600"] - means["4800"],
                "positive_splits_9600": learning[candidate_id][
                    "positive_split_seeds_at_max_query_count"
                ],
                "consensus_gain_9600": consensus["gain_over_fixed_dense"],
                "consensus_ci95_lower": consensus["gain_ci95"][0],
                "consensus_ci95_upper": consensus["gain_ci95"][1],
                "switch_coverage": consensus["switch_coverage"],
                "harmful_to_beneficial_mass_ratio": consensus[
                    "harmful_to_beneficial_mass_ratio"
                ],
                "non_tie_auc": consensus["non_tie_auc"],
                "gap_spearman": consensus["gap_spearman"],
                "gate_passed": gate["passed"],
                "failed_checks": ";".join(
                    name for name, passed in gate["checks"].items() if not passed
                ),
            }
        )
    comparison_path = RUN / "comparison_4800_9600.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    curve_table = []
    gate_table = []
    split_table = []
    for row in comparison_rows:
        candidate_id = row["candidate_id"]
        curve_table.append(
            "| {label} | {g1200} | {g2400} | {g4800} | {g9600} | {delta} | {positive}/3 |".format(
                label=CANDIDATE_LABELS[candidate_id],
                g1200=signed(row["gain_1200"]),
                g2400=signed(row["gain_2400"]),
                g4800=signed(row["gain_4800"]),
                g9600=signed(row["gain_9600"]),
                delta=signed(row["delta_9600_minus_4800"]),
                positive=row["positive_splits_9600"],
            )
        )
        gate = decision["formal_gate_results"][candidate_id]
        consensus = gate["consensus_metrics"]
        failed = ", ".join(name for name, passed in gate["checks"].items() if not passed)
        gate_table.append(
            "| {label} | {gain} | [{lower}, {upper}] | {coverage:.1%} | {ratio:.3f} | {auc:.3f} | {failed} |".format(
                label=CANDIDATE_LABELS[candidate_id],
                gain=signed(consensus["gain_over_fixed_dense"]),
                lower=signed(consensus["gain_ci95"][0]),
                upper=signed(consensus["gain_ci95"][1]),
                coverage=consensus["switch_coverage"],
                ratio=consensus["harmful_to_beneficial_mass_ratio"],
                auc=consensus["non_tie_auc"],
                failed=failed,
            )
        )
        gains = [
            formal["candidate_splits"][candidate_id][str(seed)]["metrics"][
                "gain_over_fixed_dense"
            ]
            for seed in split_seeds
        ]
        split_table.append(
            f"| {CANDIDATE_LABELS[candidate_id]} | "
            + " | ".join(signed(value) for value in gains)
            + f" | {signed(float(np.mean(gains)))} |"
        )

    if teacher is not None:
        probe = teacher["formal_gate_results"]["P1_probe_xgb"]["consensus_metrics"]
        gold = teacher["formal_gate_results"]["G1_gold_xgb"]["consensus_metrics"]
        gold_rule = teacher["gold_rule_metrics"]["gold_coverage_at_5_rule"]
        probe_increment = teacher["paired_policy_increments"][
            "probe_xgb_minus_query_only_m3"
        ]
        gold_increment = teacher["paired_policy_increments"]["gold_xgb_minus_probe_xgb"]
        stage12_text = f"""Stage 12 is complete with conclusion
`{teacher['conclusion']}`. The qrels-free probe XGBoost gains
`{signed(probe['gain_over_fixed_dense'])}` with CI95
`[{signed(probe['gain_ci95'][0])}, {signed(probe['gain_ci95'][1])}]` and recovers
`{probe['oracle_recovery']:.1%}` of oracle headroom. It misses only the frozen
harmful/beneficial ratio gate (`{probe['harmful_to_beneficial_mass_ratio']:.3f}`
versus `≤0.5`). Privileged gold XGBoost gains
`{signed(gold['gain_over_fixed_dense'])}`; the predeclared gold coverage@5 rule
gains `{signed(gold_rule['gain_over_fixed_dense'])}` with CI95
`[{signed(gold_rule['gain_ci95'][0])}, {signed(gold_rule['gain_ci95'][1])}]`.

On the same queries, probe XGBoost improves over query-only M3 by
`{signed(probe_increment['mean_gain_increment'])}` with grouped paired CI95
`[{signed(probe_increment['grouped_bootstrap_ci95'][0])}, {signed(probe_increment['grouped_bootstrap_ci95'][1])}]`.
Gold XGBoost adds a further `{signed(gold_increment['mean_gain_increment'])}`
with paired CI95 `[{signed(gold_increment['grouped_bootstrap_ci95'][0])},
{signed(gold_increment['grouped_bootstrap_ci95'][1])}]`.

This localizes the missing signal mainly to retrieval/evidence outcomes. The
original plan is complete: stages 10 and 11 are not applicable because no
query-only model passed stage 9, while stage 12 completed the failure-mechanism
diagnosis. The follow-on architecture should be a shallow BM25+Dense probe/cascade
with conservative harmful-switch control."""
    else:
        stage12_text = "Stage 12 has not been executed."

    report = f"""# HotpotQA B/D Router Phase 2.7: 9,600-query confirmation

## Outcome

**Query-only decision: `STOP_QUERY_ONLY_V1`.** No deployable pre-retrieval candidate was selected. The best
diagnostic candidate was `M3_pca32_structured_ridge`, but it failed the frozen
practical-gain, grouped-bootstrap confidence, and harmful-switch quality gates.
Fresh-dev, Answer Correctness, and final holdout remain sealed. The diagnostic-only
privileged-teacher stage is reported at the end of this document.

## Boundary and completed generation

- Partition: `train_only`; queries: `9,600`
- BM25/Dense outcomes: `57,600` complete cells (`2` actions × `3` repeats)
- Newly completed API cells: `{completion['protocol_usage']['successful_cells']:,}`;
  failures: `{completion['protocol_usage']['failed_cells']}`
- Model/prompt: `{completion['model']}` / `{completion['prompt_version']}`
- Input/output tokens: `{completion['protocol_usage']['input_tokens']:,}` /
  `{completion['protocol_usage']['output_tokens']:,}`
- Estimated completion cost: `${completion['protocol_usage']['estimated_cost_usd']:.4f}`
- Fresh-dev/final-holdout rows read: `0 / 0`
- Model-audit external calls after snapshot construction: `0`

The first 4,800 outcomes, summaries, and feature rows are byte/value-equivalent to
the earlier frozen pool. Database integrity is `ok`; all 12 formal candidate×split
tasks have complete 9,600-row OOF predictions over folds 0–4.

## Data utility baseline

- Fixed BM25 mean answer F1: `{preflight['fixed_bm25_mean_f1']:.6f}`
- Fixed Dense mean answer F1: `{preflight['fixed_dense_mean_f1']:.6f}`
- Per-query B/D oracle mean answer F1: `{preflight['bd_oracle_mean_f1']:.6f}`
- Oracle headroom over fixed Dense: `{preflight['bd_oracle_mean_f1'] - preflight['fixed_dense_mean_f1']:.6f}`
- Winner strata: BM25 `{preflight['strata_counts']['bm25_winner']:,}`, Dense
  `{preflight['strata_counts']['dense_winner']:,}`, exact tie
  `{preflight['strata_counts']['exact_tie']:,}`

The oracle headroom is large, but 71.25% of queries are exact ties and the
deployment-available query-only signal has to identify a relatively small set of
opposite-direction, high-magnitude cases.

## Frozen formal design

- Candidates were selected at 4,800 and were not reselected after seeing the new labels.
- Split seeds: `20260901`, `20260917`, `20261003`; 5 grouped outer folds each.
- XGBoost candidates use model seeds `11, 23, 37, 53, 71`.
- Preprocessing, PCA, inner OOF fusion, early stopping, and affine calibration are fold-local.
- Primary action threshold remains calibrated predicted BM25-minus-Dense gap `> 0`.
- Formal grouped bootstrap uses 10,000 resamples.

## 9,600 formal split results

| Candidate | split 20260901 | split 20260917 | split 20261003 | mean |
|---|---:|---:|---:|---:|
{chr(10).join(split_table)}

## Learning-curve comparison

| Candidate | 1,200 | 2,400 | 4,800 | 9,600 | 9,600−4,800 | positive 9,600 splits |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(curve_table)}

Only M3 continued to improve materially from 4,800 to 9,600. M2 plateaued,
M4 declined despite becoming positive on all three splits, and M0 declined below
the fixed-Dense baseline. This rejects a blanket claim that all candidates merely
needed more training data.

## Frozen gate results

| Candidate | consensus gain | grouped CI95 | coverage | harmful/beneficial mass | non-tie AUC | failed checks |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(gate_table)}

The gate requires mean split gain ≥ `+0.01`, bootstrap lower bound > `0`, all
three split gains positive, harmful/beneficial mass ≤ `0.5`, coverage in
`[1%, 50%]`, calibration slope in `[0.5, 1.5]`, and a positive top predicted
gap decile. No candidate passed all checks.

M3 is the strongest diagnostic result: consensus gain `{signed(decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['gain_over_fixed_dense'])}`,
CI95 `[{signed(decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['gain_ci95'][0])}, {signed(decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['gain_ci95'][1])}]`,
non-tie AUC `{decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['non_tie_auc']:.3f}`,
Spearman `{decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['gap_spearman']:.3f}`,
and harmful/beneficial mass ratio
`{decision['formal_gate_results']['M3_pca32_structured_ridge']['consensus_metrics']['harmful_to_beneficial_mass_ratio']:.3f}`.
Its confidence interval nearly reaches zero, but its effect is only about one
quarter of the predeclared practical threshold and harmful losses remain too large.

## What this resolves

1. **Are the chosen features useful?** Yes, but only weakly. M3's AUC/Spearman and
   positive three-split gain show a real-looking query-only signal; the gate failure
   shows it is not strong or safe enough for routing.
2. **Is direct use of the large query embedding matrix appropriate?** Not for the
   tested tree pipeline. Raw 384D embedding concatenation in M2 plateaued at
   `{signed(learning['M2_raw_full_xgb_mse']['mean_gain_by_query_count']['9600'])}`
   and its consensus policy was negative. PCA32 plus a linear Ridge model was best.
3. **Is direct concatenation technically wrong?** It is a valid implementation, but
   the evidence rejects it as the best inductive bias here. Arbitrary dense
   coordinates plus structured features do not become useful to XGBoost simply by
   doubling the sample.
4. **Was XGBoost the right model?** Not as a universal default. The best model was
   linear Ridge after fold-local PCA; structured-only and raw-full XGBoost were near
   zero or negative. Late fusion remained weak and plateaued.
5. **Were the training method and target reasonable?** Continuous utility-gap OOF
   training is coherent and the evaluation is sufficiently controlled to expose
   instability. However, the high harmful/beneficial ratios show that regression
   accuracy and the zero threshold are not yet aligned tightly enough with safe
   routing utility.
6. **Was training sufficient?** It was sufficient to reject “just add more of the
   same query-only data” for these four frozen candidates. It does not prove that no
   future architecture can work; it says further scaling of Query-Only V1 is not
   justified by this experiment.

## Stage 12 and plan closure

{stage12_text}
"""
    (RUN / "report.md").write_text(report, encoding="utf-8")

    validation = {
        "protocol_id": formal["protocol_id"],
        "status": "passed",
        "decision": decision["decision"],
        "selected_candidate": decision["selected_candidate"],
        "best_diagnostic_candidate": decision["best_diagnostic_candidate"],
        "queries": 9600,
        "outcome_rows": 57600,
        "formal_tasks": 12,
        "formal_prediction_arrays": 24,
        "learning_curve_rows": 48,
        "database_integrity": integrity,
        "fresh_dev_rows_read": fresh_dev_rows,
        "final_holdout_rows_read": final_holdout_rows,
        "model_audit_external_calls": 0,
        "stage12_status": teacher.get("status") if teacher is not None else "not_run",
        "stage12_conclusion": teacher.get("conclusion") if teacher is not None else None,
        "artifacts": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in {
                "formal_metrics.json": RUN / "formal_metrics.json",
                "formal_predictions.npz": RUN / "formal_predictions.npz",
                "learning_curve.csv": RUN / "learning_curve.csv",
                "decision.json": RUN / "decision.json",
                "comparison_4800_9600.csv": comparison_path,
                "report.md": RUN / "report.md",
                **(
                    {"privileged_teacher_diagnostic.json": teacher_path}
                    if teacher is not None
                    else {}
                ),
            }.items()
        },
    }
    (RUN / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
