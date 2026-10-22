"""Build/check a metadata-only QA/raw plan. No model, retrieval, or API imports.

The committed metadata projection contains only qid/type/group_id. Its source
file hashes refer to the already-frozen local formal500 files; no question,
reference answer, passage, model output, score, or runtime status is committed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261023
N = 100
DATASETS = ("hotpotqa", "2wiki", "musique")
META = HERE / "SAMPLING_METADATA.qid_only.json"
OUT = HERE / "COHORT.qid_only.json"


def rank(dataset, purpose, qid):
    return hashlib.sha256(f"{SEED}:{dataset}:{purpose}:{qid}".encode("utf8")).hexdigest()


def build():
    payload = json.loads(META.read_text(encoding="utf8"))
    result = {
        "id": "p02_qa_raw_final_interface_300_revision007_20261023",
        "status": "planned_not_executed",
        "execution_authorized": False,
        "seed": SEED,
        "questions_per_dataset": N,
        "planned_queries": 3 * N,
        "new_final_arms": ["QA", "Raw"],
        "planned_new_final_results": 6 * N,
        "metadata_sha256": hashlib.sha256(META.read_bytes()).hexdigest(),
        "selection_fields": ["qid", "type", "group_id"],
        "selection": "largest-remainder proportional strata; lexicographic stratum ties; SHA256(seed:dataset:qa_raw_sample:qid) within stratum; separate qa_raw_order hash",
        "exposure": "subset of existing formal500; not claimed to be unseen or independent of historical research",
        "datasets": {},
    }
    for dataset in DATASETS:
        rows = payload["datasets"][dataset]["metadata"]
        if len(rows) != 500 or any(set(r) != {"qid", "type", "group_id"} for r in rows):
            raise ValueError("Expected 500 metadata-only rows per dataset")
        if any(not isinstance(r[k], str) or not r[k] for r in rows for k in r):
            raise ValueError("Metadata values must be nonempty strings")
        if len({r["qid"] for r in rows}) != 500 or len({r["group_id"] for r in rows}) != 500:
            raise ValueError("Duplicate qid/group in frozen formal population")
        count = Counter(r["type"] for r in rows)
        quota = {key: N * count[key] // len(rows) for key in sorted(count)}
        remainder_order = sorted(count, key=lambda key: (-(N * count[key] % len(rows)), key))
        for key in remainder_order[:N - sum(quota.values())]:
            quota[key] += 1
        selected = []
        for key in sorted(quota):
            candidates = sorted((r for r in rows if r["type"] == key),
                                key=lambda r: (rank(dataset, "qa_raw_sample", r["qid"]), r["qid"]))
            selected.extend(candidates[:quota[key]])
        selected.sort(key=lambda r: (rank(dataset, "qa_raw_order", r["qid"]), r["qid"]))
        assert len(selected) == N
        result["datasets"][dataset] = {
            "population": 500,
            "population_strata": dict(sorted(count.items())),
            "selected_strata": quota,
            "queries": [dict(r) for r in selected],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare against the saved plan without writing")
    args = parser.parse_args()
    content = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf8") != content:
            raise SystemExit("Saved cohort differs from deterministic metadata-only selection")
        print("PASS: 300 planned qids; no model, retrieval, or API calls")
    else:
        OUT.write_text(content, encoding="utf8")
        print("Wrote 300 planned qids and 600 planned final results; evaluation remains paused")


if __name__ == "__main__":
    main()
