#!/usr/bin/env python3
"""Read-only BM25/Dense and weighted-RRF analysis for the Selected-7 BEIR record.

This script deliberately ignores every reranked run.  It consumes the archived
BM25 and Dense Top-50 result rows, reuses their qrels, and performs only an
offline deterministic rank fusion.  No retrieval/model inference is executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


EPS = 1e-12
FAMILY_DISPLAY = {
    "arguana": "ArguAna",
    "fiqa": "FiQA-2018",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "webis-touche2020": "Touche-2020",
    "cqadupstack": "CQADupStack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("tmp/beir_retriever_dataset_interactions_v3/candidate_source_result_files.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/history_aware_retriever_policy/analysis"),
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    return parser.parse_args()


def score_query(doc_ids: list[str], qrels: dict[str, float], k: int) -> dict[str, float]:
    rel = {str(doc_id): float(value) for doc_id, value in qrels.items() if float(value) > 0}
    ranked = doc_ids[:k]
    dcg = sum(rel.get(doc_id, 0.0) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked, 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    hits = sum(doc_id in rel for doc_id in ranked)
    return {
        "ndcg": dcg / idcg if idcg else 0.0,
        "recall": hits / len(rel) if rel else 0.0,
        "hit": float(hits > 0),
    }


def read_source_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["semantic"] in {"bm25", "dense"}]
    if len(rows) != 36:
        raise RuntimeError(f"expected 36 BM25/Dense source files, found {len(rows)}")
    return rows


def load_run(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("status") != "success":
                raise RuntimeError(f"non-success row: {path}:{line_number}")
            qid = str(row["question_id"])
            if qid in result:
                raise RuntimeError(f"duplicate qid {qid}: {path}")
            results = row["retrieval"]["results"]
            docs = [str(item["doc_id"]) for item in results]
            if len(docs) != len(set(docs)):
                raise RuntimeError(f"duplicate doc id for {qid}: {path}")
            result[qid] = {
                "question": row.get("question", ""),
                "qrels": {str(k): float(v) for k, v in row["qrels"].items()},
                "docs": docs,
                "scores": [float(item["score"]) for item in results],
            }
    return result


def fuse_rrf(
    bm25_docs: list[str],
    dense_docs: list[str],
    *,
    alpha: float,
    rrf_k: int,
    bm25_depth: int,
    dense_depth: int,
    output_depth: int = 50,
) -> list[str]:
    """Symmetric weighted RRF with deterministic, method-neutral tie breaks."""
    b_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_docs[:bm25_depth], 1)}
    d_rank = {doc_id: rank for rank, doc_id in enumerate(dense_docs[:dense_depth], 1)}
    candidates = set(b_rank) | set(d_rank)
    scored: list[tuple[float, int, int, str]] = []
    missing = 10**9
    for doc_id in candidates:
        br = b_rank.get(doc_id)
        dr = d_rank.get(doc_id)
        value = (alpha / (rrf_k + br) if br is not None else 0.0) + (
            (1.0 - alpha) / (rrf_k + dr) if dr is not None else 0.0
        )
        best_rank = min(br if br is not None else missing, dr if dr is not None else missing)
        rank_sum = (br if br is not None else missing) + (dr if dr is not None else missing)
        scored.append((-value, best_rank, rank_sum, doc_id))
    scored.sort()
    return [doc_id for _, _, _, doc_id in scored[:output_depth]]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def family_aggregate(query_rows: list[dict], value_key: str) -> dict[str, float]:
    """BEIR family aggregation: CQADupStack is forum-equal, others query-macro."""
    by_family_unit: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in query_rows:
        by_family_unit[(row["family"], row["unit"])].append(float(row[value_key]))
    units_by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _unit), values in by_family_unit.items():
        units_by_family[family].append(mean(values))
    return {family: mean(unit_values) for family, unit_values in units_by_family.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.rrf_k < 0:
        raise ValueError("--rrf-k must be non-negative")
    steps = round(1.0 / args.alpha_step)
    alphas = [round(index * args.alpha_step, 10) for index in range(steps + 1)]
    if abs(alphas[-1] - 1.0) > EPS:
        raise ValueError("--alpha-step must divide 1.0")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    source_rows = read_source_map(args.source_map)
    grouped_sources: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for row in source_rows:
        grouped_sources[(row["family"], row["unit"])][row["semantic"]] = Path(row["source_path"])

    base_rows: list[dict] = []
    rrf_rows: list[dict] = []
    query_policy_rows: list[dict] = []
    for (family, unit), paths in sorted(grouped_sources.items()):
        if set(paths) != {"bm25", "dense"}:
            raise RuntimeError(f"incomplete paths for {(family, unit)}: {paths}")
        bm25 = load_run(paths["bm25"])
        dense = load_run(paths["dense"])
        if set(bm25) != set(dense):
            raise RuntimeError(f"query mismatch for {(family, unit)}")
        for qid in sorted(bm25):
            b = bm25[qid]
            d = dense[qid]
            if b["qrels"] != d["qrels"]:
                raise RuntimeError(f"qrels mismatch for {(family, unit, qid)}")
            b10 = score_query(b["docs"], b["qrels"], 10)
            d10 = score_query(d["docs"], d["qrels"], 10)
            b50 = score_query(b["docs"], b["qrels"], 50)
            d50 = score_query(d["docs"], d["qrels"], 50)
            relevant = {doc_id for doc_id, value in b["qrels"].items() if value > 0}
            b_pool = set(b["docs"][:50])
            d_pool = set(d["docs"][:50])
            shared_relevant = len(relevant & b_pool & d_pool)
            bm25_only_relevant = len((relevant & b_pool) - d_pool)
            dense_only_relevant = len((relevant & d_pool) - b_pool)
            neither_relevant = len(relevant - (b_pool | d_pool))
            relevant_count = len(relevant)
            base = {
                "family": family,
                "unit": unit,
                "query_id": qid,
                "bm25_ndcg_at_10": b10["ndcg"],
                "dense_ndcg_at_10": d10["ndcg"],
                "dense_minus_bm25_ndcg_at_10": d10["ndcg"] - b10["ndcg"],
                "bm25_recall_at_50": b50["recall"],
                "dense_recall_at_50": d50["recall"],
                "candidate_overlap": len(set(b["docs"][:50]) & set(d["docs"][:50])),
                "candidate_union": len(set(b["docs"][:50]) | set(d["docs"][:50])),
                "candidate_jaccard": len(b_pool & d_pool) / len(b_pool | d_pool),
                "union_recall_at_up_to_100": len(relevant & (b_pool | d_pool)) / relevant_count,
                "shared_relevant_fraction": shared_relevant / relevant_count,
                "bm25_only_relevant_fraction": bm25_only_relevant / relevant_count,
                "dense_only_relevant_fraction": dense_only_relevant / relevant_count,
                "neither_relevant_fraction": neither_relevant / relevant_count,
            }
            base_rows.append(base)

            policy_scores: dict[tuple[str, float], float] = {}
            # Protocol full100: both Top-50 inputs, fused output Top-50.
            # Protocol equal50: endpoints use one Top-50; hybrid uses 25+25.
            for protocol, b_depth, d_depth in (("full100", 50, 50), ("equal50", 25, 25)):
                for alpha in alphas:
                    # Endpoints are the exact single-retriever Top-50 lists.
                    # This matters for underfilled BM25 runs (notably NFCorpus):
                    # zero-weight documents from the other list must not leak in.
                    if alpha in {0.0, 1.0}:
                        docs = d["docs"][:50] if alpha == 0.0 else b["docs"][:50]
                    else:
                        docs = fuse_rrf(
                            b["docs"], d["docs"], alpha=alpha, rrf_k=args.rrf_k,
                            bm25_depth=b_depth, dense_depth=d_depth, output_depth=50,
                        )
                    m10 = score_query(docs, b["qrels"], 10)
                    m50 = score_query(docs, b["qrels"], 50)
                    row = {
                        "family": family,
                        "unit": unit,
                        "query_id": qid,
                        "protocol": protocol,
                        "alpha_bm25": alpha,
                        "bm25_input_depth": 50 if protocol == "equal50" and alpha == 1.0 else (0 if protocol == "equal50" and alpha == 0.0 else b_depth),
                        "dense_input_depth": 50 if protocol == "equal50" and alpha == 0.0 else (0 if protocol == "equal50" and alpha == 1.0 else d_depth),
                        "output_depth": 50,
                        "ndcg_at_10": m10["ndcg"],
                        "recall_at_50": m50["recall"],
                    }
                    rrf_rows.append(row)
                    policy_scores[(protocol, alpha)] = m10["ndcg"]

            for protocol in ("equal50", "full100"):
                endpoint_best = max(policy_scores[(protocol, 0.0)], policy_scores[(protocol, 1.0)])
                hybrid_values = {a: policy_scores[(protocol, a)] for a in alphas if 0.0 < a < 1.0}
                hybrid_best = max(hybrid_values.values())
                overall_best = max(endpoint_best, hybrid_best)
                optimal = [a for a in alphas if abs(policy_scores[(protocol, a)] - overall_best) <= EPS]
                if hybrid_best > endpoint_best + EPS:
                    cost_aware_action = "hybrid"
                elif policy_scores[(protocol, 1.0)] > policy_scores[(protocol, 0.0)] + EPS:
                    cost_aware_action = "bm25"
                elif policy_scores[(protocol, 0.0)] > policy_scores[(protocol, 1.0)] + EPS:
                    cost_aware_action = "dense"
                else:
                    cost_aware_action = "single_tie"
                query_policy_rows.append({
                    "family": family,
                    "unit": unit,
                    "query_id": qid,
                    "protocol": protocol,
                    "bm25_ndcg_at_10": policy_scores[(protocol, 1.0)],
                    "dense_ndcg_at_10": policy_scores[(protocol, 0.0)],
                    "best_single_ndcg_at_10": endpoint_best,
                    "best_hybrid_ndcg_at_10": hybrid_best,
                    "oracle_grid_ndcg_at_10": overall_best,
                    "hybrid_strictly_beats_both_singles": int(hybrid_best > endpoint_best + EPS),
                    "best_alpha_min": min(optimal),
                    "best_alpha_max": max(optimal),
                    "num_tied_best_alphas": len(optimal),
                    "bm25_is_optimal": int(1.0 in optimal),
                    "dense_is_optimal": int(0.0 in optimal),
                    "hybrid_is_optimal": int(any(0.0 < a < 1.0 for a in optimal)),
                    "cost_aware_oracle_action": cost_aware_action,
                })

    # Fixed BM25/Dense family comparison at k=10.
    base_summary: list[dict] = []
    for family in FAMILY_DISPLAY:
        rows = [row for row in base_rows if row["family"] == family]
        if family == "cqadupstack":
            unit_names = sorted({row["unit"] for row in rows})
            b_mean = mean([mean([r["bm25_ndcg_at_10"] for r in rows if r["unit"] == unit]) for unit in unit_names])
            d_mean = mean([mean([r["dense_ndcg_at_10"] for r in rows if r["unit"] == unit]) for unit in unit_names])
        else:
            b_mean = mean([row["bm25_ndcg_at_10"] for row in rows])
            d_mean = mean([row["dense_ndcg_at_10"] for row in rows])
        wins = sum(row["dense_ndcg_at_10"] > row["bm25_ndcg_at_10"] + EPS for row in rows)
        losses = sum(row["dense_ndcg_at_10"] < row["bm25_ndcg_at_10"] - EPS for row in rows)
        ties = len(rows) - wins - losses
        base_summary.append({
            "family": family,
            "family_display": FAMILY_DISPLAY[family],
            "aggregation": "forum_equal" if family == "cqadupstack" else "query_macro",
            "num_queries": len(rows),
            "bm25_ndcg_at_10": b_mean,
            "dense_ndcg_at_10": d_mean,
            "dense_minus_bm25": d_mean - b_mean,
            "dense_query_win_fraction": wins / len(rows),
            "bm25_query_win_fraction": losses / len(rows),
            "query_tie_fraction": ties / len(rows),
            "best_fixed_single": "dense" if d_mean > b_mean else "bm25",
            "single_oracle_ndcg_at_10": family_aggregate(rows, "bm25_ndcg_at_10")[family],
        })
        # Replace the placeholder above with the actual per-query single oracle aggregation.
        for row in rows:
            row["single_oracle"] = max(row["bm25_ndcg_at_10"], row["dense_ndcg_at_10"])
        base_summary[-1]["single_oracle_ndcg_at_10"] = family_aggregate(rows, "single_oracle")[family]
        base_summary[-1]["single_oracle_gain_over_best_fixed"] = (
            base_summary[-1]["single_oracle_ndcg_at_10"] - max(b_mean, d_mean)
        )

    # Top-50 pool composition.  This is a coverage diagnostic, not a same-budget hybrid.
    candidate_summary: list[dict] = []
    candidate_keys = [
        "candidate_jaccard",
        "bm25_recall_at_50",
        "dense_recall_at_50",
        "union_recall_at_up_to_100",
        "shared_relevant_fraction",
        "bm25_only_relevant_fraction",
        "dense_only_relevant_fraction",
        "neither_relevant_fraction",
    ]
    for family in FAMILY_DISPLAY:
        rows = [row for row in base_rows if row["family"] == family]
        values = {key: family_aggregate(rows, key)[family] for key in candidate_keys}
        candidate_summary.append({
            "family": family,
            "family_display": FAMILY_DISPLAY[family],
            "aggregation": "forum_equal" if family == "cqadupstack" else "query_macro",
            "num_queries": len(rows),
            **values,
            "union_headroom_over_better_fixed_pool": (
                values["union_recall_at_up_to_100"]
                - max(values["bm25_recall_at_50"], values["dense_recall_at_50"])
            ),
        })

    # Family and seven-family-equal alpha curves.
    alpha_summary: list[dict] = []
    unit_alpha_summary: list[dict] = []
    for protocol in ("equal50", "full100"):
        for alpha in alphas:
            selected = [row for row in rrf_rows if row["protocol"] == protocol and row["alpha_bm25"] == alpha]
            by_unit: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for row in selected:
                by_unit[(row["family"], row["unit"])].append(row)
            for (family, unit), unit_rows in sorted(by_unit.items()):
                unit_alpha_summary.append({
                    "family": family,
                    "family_display": FAMILY_DISPLAY[family],
                    "unit": unit,
                    "protocol": protocol,
                    "alpha_bm25": alpha,
                    "num_queries": len(unit_rows),
                    "ndcg_at_10": mean([row["ndcg_at_10"] for row in unit_rows]),
                    "recall_at_50": mean([row["recall_at_50"] for row in unit_rows]),
                })
            ndcg_families = family_aggregate(selected, "ndcg_at_10")
            recall_families = family_aggregate(selected, "recall_at_50")
            for family in FAMILY_DISPLAY:
                alpha_summary.append({
                    "scope": "family",
                    "family": family,
                    "family_display": FAMILY_DISPLAY[family],
                    "protocol": protocol,
                    "alpha_bm25": alpha,
                    "ndcg_at_10": ndcg_families[family],
                    "recall_at_50": recall_families[family],
                })
            alpha_summary.append({
                "scope": "selected7_equal",
                "family": "selected7_equal",
                "family_display": "Selected-7 equal",
                "protocol": protocol,
                "alpha_bm25": alpha,
                "ndcg_at_10": mean(list(ndcg_families.values())),
                "recall_at_50": mean(list(recall_families.values())),
            })

    # Oracle summaries, including how often hybrid is strictly necessary.
    oracle_summary: list[dict] = []
    for protocol in ("equal50", "full100"):
        selected = [row for row in query_policy_rows if row["protocol"] == protocol]
        for family in list(FAMILY_DISPLAY) + ["selected7_equal"]:
            rows = selected if family == "selected7_equal" else [row for row in selected if row["family"] == family]
            if family == "selected7_equal":
                single_by_family = family_aggregate(rows, "best_single_ndcg_at_10")
                hybrid_by_family = family_aggregate(rows, "best_hybrid_ndcg_at_10")
                oracle_by_family = family_aggregate(rows, "oracle_grid_ndcg_at_10")
                strict_by_family = family_aggregate(rows, "hybrid_strictly_beats_both_singles")
                single_value = mean(list(single_by_family.values()))
                hybrid_value = mean(list(hybrid_by_family.values()))
                oracle_value = mean(list(oracle_by_family.values()))
                strict_value = mean(list(strict_by_family.values()))
            else:
                single_value = family_aggregate(rows, "best_single_ndcg_at_10")[family]
                hybrid_value = family_aggregate(rows, "best_hybrid_ndcg_at_10")[family]
                oracle_value = family_aggregate(rows, "oracle_grid_ndcg_at_10")[family]
                strict_value = family_aggregate(rows, "hybrid_strictly_beats_both_singles")[family]
            oracle_summary.append({
                "family": family,
                "family_display": "Selected-7 equal" if family == "selected7_equal" else FAMILY_DISPLAY[family],
                "protocol": protocol,
                "num_queries": len(rows),
                "query_oracle_best_single_ndcg_at_10": single_value,
                "query_oracle_best_hybrid_ndcg_at_10": hybrid_value,
                "query_oracle_single_or_hybrid_ndcg_at_10": oracle_value,
                "hybrid_strict_gain_query_fraction": strict_value,
                "oracle_hybrid_gain_over_single_oracle": oracle_value - single_value,
                **{
                    f"cost_aware_{action}_fraction": (
                        mean([
                            family_aggregate(
                                [dict(row, action_indicator=float(row["cost_aware_oracle_action"] == action)) for row in rows],
                                "action_indicator",
                            )[family]
                        ])
                        if family != "selected7_equal"
                        else mean(list(family_aggregate(
                            [dict(row, action_indicator=float(row["cost_aware_oracle_action"] == action)) for row in rows],
                            "action_indicator",
                        ).values()))
                    )
                    for action in ("bm25", "dense", "hybrid", "single_tie")
                },
            })

    # Best family-level and macro alpha: descriptive test-set oracle, never a fair tuned baseline.
    best_alpha_rows: list[dict] = []
    for protocol in ("equal50", "full100"):
        scopes = [row for row in alpha_summary if row["protocol"] == protocol]
        for family in list(FAMILY_DISPLAY) + ["selected7_equal"]:
            rows = [row for row in scopes if row["family"] == family]
            max_ndcg = max(row["ndcg_at_10"] for row in rows)
            optimal = [row["alpha_bm25"] for row in rows if abs(row["ndcg_at_10"] - max_ndcg) <= EPS]
            endpoints = [row for row in rows if row["alpha_bm25"] in {0.0, 1.0}]
            best_single = max(row["ndcg_at_10"] for row in endpoints)
            half = next(row for row in rows if row["alpha_bm25"] == 0.5)
            best_alpha_rows.append({
                "family": family,
                "family_display": rows[0]["family_display"],
                "protocol": protocol,
                "best_alpha_min": min(optimal),
                "best_alpha_max": max(optimal),
                "best_test_oracle_ndcg_at_10": max_ndcg,
                "best_fixed_single_ndcg_at_10": best_single,
                "gain_over_best_fixed_single": max_ndcg - best_single,
                "alpha_0_5_ndcg_at_10": half["ndcg_at_10"],
                "alpha_0_5_gain_over_best_fixed_single": half["ndcg_at_10"] - best_single,
                "best_alpha_recall_at_50": next(row["recall_at_50"] for row in rows if row["alpha_bm25"] == min(optimal)),
            })

    unit_best_alpha_rows: list[dict] = []
    for protocol in ("equal50", "full100"):
        for family, unit in sorted({(row["family"], row["unit"]) for row in unit_alpha_summary}):
            rows = [
                row for row in unit_alpha_summary
                if row["protocol"] == protocol and row["family"] == family and row["unit"] == unit
            ]
            max_ndcg = max(row["ndcg_at_10"] for row in rows)
            optimal = [row["alpha_bm25"] for row in rows if abs(row["ndcg_at_10"] - max_ndcg) <= EPS]
            best_single = max(row["ndcg_at_10"] for row in rows if row["alpha_bm25"] in {0.0, 1.0})
            half = next(row for row in rows if row["alpha_bm25"] == 0.5)
            unit_best_alpha_rows.append({
                "family": family,
                "family_display": FAMILY_DISPLAY[family],
                "unit": unit,
                "protocol": protocol,
                "num_queries": rows[0]["num_queries"],
                "best_alpha_min": min(optimal),
                "best_alpha_max": max(optimal),
                "best_test_oracle_ndcg_at_10": max_ndcg,
                "best_fixed_single_ndcg_at_10": best_single,
                "gain_over_best_fixed_single": max_ndcg - best_single,
                "alpha_0_5_ndcg_at_10": half["ndcg_at_10"],
            })

    write_csv(out / "bm25_dense_k10_summary.csv", base_summary)
    write_csv(out / "candidate_pool_complementarity.csv", candidate_summary)
    write_csv(out / "query_level_bm25_dense.csv", base_rows)
    write_csv(out / "rrf_alpha_curves.csv", alpha_summary)
    write_csv(out / "rrf_best_alpha_test_oracle.csv", best_alpha_rows)
    write_csv(out / "rrf_unit_alpha_curves.csv", unit_alpha_summary)
    write_csv(out / "rrf_unit_best_alpha_test_oracle.csv", unit_best_alpha_rows)
    write_csv(out / "query_policy_oracle.csv", query_policy_rows)
    write_csv(out / "query_policy_oracle_summary.csv", oracle_summary)
    write_csv(out / "source_files.csv", source_rows)
    source_manifest = (out / "source_files.csv").resolve()
    metadata = {
        "scope": "BM25 and Dense first-stage only; reranked paths excluded",
        "source_map_input": str(args.source_map.resolve()),
        "replay_source_manifest": str(source_manifest),
        "replay_source_manifest_sha256": sha256_file(source_manifest),
        "source_file_count": len(source_rows),
        "unit_count": len(grouped_sources),
        "query_count": len(base_rows),
        "rrf_k": args.rrf_k,
        "alpha_semantics": "alpha_bm25; 0 is Dense-only and 1 is BM25-only",
        "alphas": alphas,
        "protocols": {
            "equal50": "single uses one Top-50; hybrid uses BM25 Top-25 + Dense Top-25; output <=50",
            "full100": "BM25 Top-50 + Dense Top-50 union; fused output Top-50; upstream scored-candidate depth up to100",
        },
        "tie_break": "fused score desc, best source rank asc, source-rank sum asc, doc_id asc",
        "retrieval_or_model_inference_executed": False,
        "warning": "best alpha and query oracle use test qrels and are descriptive upper bounds, not deployable results",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fraction_error = max(
        abs(
            row["shared_relevant_fraction"]
            + row["bm25_only_relevant_fraction"]
            + row["dense_only_relevant_fraction"]
            + row["neither_relevant_fraction"]
            - 1.0
        )
        for row in base_rows
    )
    validation = {
        "status": "passed" if fraction_error <= 1e-12 else "failed",
        "scope": "BM25/Dense first-stage plus deterministic offline weighted RRF",
        "retrieval_or_model_inference_executed": False,
        "family_count": len(FAMILY_DISPLAY),
        "unit_count": len(grouped_sources),
        "query_count": len(base_rows),
        "source_file_count": len(source_rows),
        "rrf_alpha_count": len(alphas),
        "max_relevant_fraction_sum_error": fraction_error,
        "evidence_boundaries": {
            "equal50": "single Top-50 versus hybrid BM25 Top-25 plus Dense Top-25; two retriever calls remain a cost difference",
            "full100": "up to 100 scored input candidates and therefore not equal upstream candidate budget",
            "fixed_best_alpha": "selected on test qrels; descriptive oracle, not a deployable tuned baseline",
            "query_oracle": "uses current-query qrels; upper bound, not policy performance",
            "beir_order": "no session or temporal semantics",
        },
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "query_count": len(base_rows), "alphas": len(alphas)}, indent=2))


if __name__ == "__main__":
    main()
