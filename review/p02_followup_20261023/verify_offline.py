#!/usr/bin/env python3
"""Verify the portable P02 review package using Python's standard library only.

Usage: python verify_offline.py [--package DIRECTORY] [--json RESULT.json]
No archive extraction, repository imports, network, model calls or source writes.
Native per-query scores are re-aggregated, not re-scored from answer strings.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile


COT = "p02_local_cot_fixed90_v1_20261022"
CONTEXT = "p02_shared_context_fixed90_v1_20261022"
REFERENCE = "p02_reference_binding_priority_fixed90_v1_20261022"
DATASETS = {"hotpotqa", "2wiki", "musique"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_name(value):
    require(isinstance(value, str) and value, "Empty or non-string path")
    value = value.replace("\\", "/")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and ":" not in value,
            "Path must be safe and package-relative")
    require(value == p.as_posix(), "Path is not normalized")
    return value


def digest(stream):
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return size, h.hexdigest()
        size += len(chunk)
        h.update(chunk)


def mean(values):
    values = list(values)
    require(bool(values), "Cannot average an empty cohort")
    return math.fsum(values) / len(values)


def equal_number(actual, expected, label):
    require(isinstance(expected, (int, float)) and not isinstance(expected, bool),
            label + ": missing numeric reference")
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11),
            label + ": aggregate disagrees with packaged report")


class Package:
    def __init__(self, root, stack):
        self.root = root.resolve()
        self.stack = stack
        self.members = {}
        self.databases = {}
        self.delivery_paths = set()

    def path(self, relative):
        path = (self.root / relative_name(relative)).resolve()
        require(path.is_relative_to(self.root), "Path escapes the package")
        return path

    def local_json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8-sig"))

    def source_bytes(self, source):
        source = relative_name(source)
        require(source in self.members, "Required source missing: " + source)
        archive, member = self.members[source]
        return archive.read(member)

    def source_json(self, source):
        return json.loads(self.source_bytes(source).decode("utf-8-sig"))

    def source_rows(self, source):
        return [json.loads(line) for line in self.source_bytes(source).decode("utf-8-sig").splitlines()
                if line.strip()]

    def verify_delivery(self):
        manifest = self.local_json("DELIVERY_MANIFEST.json")
        seen = set()
        total = 0
        for row in manifest["files"]:
            name = relative_name(row["path"])
            require(name not in seen, "Duplicate delivery path: " + name)
            seen.add(name)
            with self.path(name).open("rb") as stream:
                size, sha = digest(stream)
            require(size == row["bytes"] and sha == row["sha256"], "Delivery hash/size mismatch: " + name)
            total += size
        require("ARCHIVE_MANIFEST.json" in seen and "SQLITE_EXPORTS.json" in seen,
                "Delivery manifest must cover both secondary manifests")
        self.delivery_paths = seen
        return {"files_verified": len(seen), "bytes_verified": total,
                "manifest_self_hash_excluded": "DELIVERY_MANIFEST.json" not in seen}

    def verify_archives(self):
        manifest = self.local_json("ARCHIVE_MANIFEST.json")
        archive_names = set()
        count = total = 0
        for row in manifest["archives"]:
            name = relative_name(row["path"])
            require(name not in archive_names, "Duplicate archive path")
            require(name in self.delivery_paths, "Archive missing from delivery manifest")
            archive_names.add(name)
            with self.path(name).open("rb") as stream:
                _, sha = digest(stream)
            require(sha == row["sha256"], "Archive hash mismatch: " + name)
            archive = self.stack.enter_context(zipfile.ZipFile(self.path(name)))
            members = archive.infolist()
            require(len({m.filename for m in members}) == len(members), "Duplicate ZIP member")
            require(not any(m.is_dir() for m in members), "ZIP must contain files only")
            expected = {}
            for item in row["members"]:
                source = relative_name(item["source"])
                require(source not in expected, "Duplicate archive manifest member")
                expected[source] = item
            require({m.filename for m in members} == set(expected), "ZIP member inventory mismatch: " + name)
            for member in members:
                source = relative_name(member.filename)
                require(source not in self.members, "Source duplicated between archives: " + source)
                expected_row = expected[source]
                with archive.open(member) as stream:
                    size, sha = digest(stream)
                require(size == expected_row["bytes"] and sha == expected_row["sha256"],
                        "ZIP member hash/size mismatch: " + source)
                self.members[source] = (archive, member)
                count += 1
                total += size
        return {"archives_verified": len(archive_names), "members_verified": count,
                "uncompressed_bytes_verified": total}

    def verify_exports(self):
        manifest = self.local_json("SQLITE_EXPORTS.json")
        tables = rows_total = 0
        paths = set()
        for database in manifest["databases"]:
            source = relative_name(database["source"])
            require(source not in self.databases, "Duplicate SQLite export source")
            summary = {"tables": {}, "outcomes": {}, "kinds": Counter(), "request_statuses": Counter()}
            selected = source in {"outputs/query_decomposition/" + e + "/checkpoint.sqlite3"
                                  for e in (COT, CONTEXT, REFERENCE)}
            for table in database["tables"]:
                name, path = table["name"], relative_name(table["path"])
                require(name not in summary["tables"] and path not in paths, "Duplicate exported table/path")
                require(path in self.delivery_paths, "SQLite export missing from delivery manifest")
                paths.add(path)
                columns = table["columns"]
                require(all(isinstance(c, str) for c in columns) and len(set(columns)) == len(columns),
                        "Invalid exported column schema")
                count = 0
                with gzip.open(self.path(path), "rt", encoding="utf-8-sig") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        require(isinstance(record, dict) and set(record) == set(columns),
                                "Exported table columns mismatch: " + path)
                        require(all(value is None or isinstance(value, (str, int, float, bool))
                                    for value in record.values()), "SQLite scalar/JSON-string contract violated")
                        count += 1
                        if name == "data" and "kind" in record:
                            summary["kinds"][record["kind"]] += 1
                            if selected and record["kind"] == "outcome":
                                require(isinstance(record["value"], str), "SQLite JSON column must remain a string")
                                body = json.loads(record["value"])
                                key = record["key"]
                                require(key not in summary["outcomes"], "Duplicate exported outcome key")
                                summary["outcomes"][key] = {field: body.get(field) for field in
                                                           ("task_id", "id", "arm", "dataset", "status", "prediction")}
                        if name == "requests" and "status" in record:
                            summary["request_statuses"][record["status"]] += 1
                require(count == table["rows"], "Exported row count mismatch: " + path)
                summary["tables"][name] = count
                tables += 1
                rows_total += count
            self.databases[source] = summary
        return {"databases_verified": len(self.databases), "tables_verified": tables,
                "rows_verified": rows_total, "json_columns_preserved_as_strings": True}


def quality_index(rows, arms):
    index = {}
    for row in rows:
        key = (row["id"], row["arm"])
        require(key not in index and row["arm"] in arms, "Duplicate or unexpected score row")
        require(row["dataset"] in DATASETS, "Unexpected dataset")
        require(row["task_id"] == row["id"] + "/" + row["arm"], "Task identity mismatch")
        f1, em = row.get("f1"), row.get("em")
        if f1 is None or em is None:
            require(f1 is None and em is None and row["status"] == "technical_held", "Invalid missing-score state")
        else:
            require(0 <= f1 <= 1 and em in (0, 1), "Invalid native score range")
        index[key] = row
    return index


def available(index, arm):
    return {qid for (qid, a), row in index.items() if a == arm and row.get("f1") is not None}


def cohort_summary(index, ids, arm):
    rows = [index[(qid, arm)] for qid in sorted(ids)]
    by_dataset = {dataset: [r for r in rows if r["dataset"] == dataset] for dataset in sorted(DATASETS)}
    result = {"n": len(rows), "dataset_counts": {d: len(rs) for d, rs in by_dataset.items()}}
    for metric in ("f1", "em"):
        result[metric.upper()] = mean(r[metric] for r in rows)
        result["macro_" + metric.upper()] = mean(mean(r[metric] for r in rs) for rs in by_dataset.values())
    usage = [r["full_flow_logical_usage"] for r in rows]
    require(all(u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"] for u in usage),
            "Logical token sum mismatch")
    result["mean_tokens"] = {key: mean(u[key] for u in usage) for key in
                              ("prompt_tokens", "completion_tokens", "total_tokens")}
    result["mean_logical_estimated_CNY"] = mean(r["full_flow_logical_estimated_CNY"] for r in rows)
    return result


def paired(index, ids, control, candidate):
    deltas = [index[(qid, candidate)]["f1"] - index[(qid, control)]["f1"] for qid in sorted(ids)]
    return {"n": len(deltas), "delta_F1": mean(deltas),
            "delta_EM": mean(index[(q, candidate)]["em"] - index[(q, control)]["em"] for q in ids),
            "gain": sum(d > 1e-12 for d in deltas), "harm": sum(d < -1e-12 for d in deltas),
            "tie": sum(abs(d) <= 1e-12 for d in deltas)}


def check_summary(actual, expected, label, macro=False):
    for metric in ("F1", "EM"):
        equal_number(actual[("macro_" if macro else "") + metric], expected[metric], label + "/" + metric)


def check_pair(actual, expected, label):
    for field in ("delta_F1", "delta_EM"):
        equal_number(actual[field], expected[field], label + "/" + field)
    for field, alternatives in (("n", ("complete_pairs", "n")), ("gain", ("improved", "gain")),
                                ("harm", ("harmed", "harm")), ("tie", ("tied", "tie"))):
        key = next((key for key in alternatives if key in expected), None)
        require(key is not None and actual[field] == expected[key], label + "/" + field + " mismatch")


def check_outcome_export(package, experiment, index, expected_count):
    source = "outputs/query_decomposition/" + experiment + "/checkpoint.sqlite3"
    require(source in package.databases, "Primary checkpoint export missing: " + experiment)
    export = package.databases[source]
    successful = {row["task_id"]: row for row in index.values() if row.get("f1") is not None}
    require(len(export["outcomes"]) == expected_count and set(export["outcomes"]) == set(successful),
            "Exported outcomes and native DETAILS identities differ")
    for task, row in successful.items():
        body = export["outcomes"][task]
        require(all(body[key] == row[key] for key in ("task_id", "id", "arm", "dataset", "status", "prediction")),
                "Exported outcome and DETAILS fields differ")
    require(export["request_statuses"].get("in_flight", 0) == 0, "Primary export retains an in-flight request")


def verify_cot(package):
    base = "outputs/query_decomposition/" + COT + "/"
    arms = ("N128", "N256", "F256", "R256")
    index = quality_index(package.source_rows(base + "DETAILS.jsonl"), arms)
    require(len(index) == 360, "CoT expected 360 declared score rows")
    require(all(sum(a == arm for _, a in index) == 90 for arm in arms), "CoT planned arm counts differ")
    check_outcome_export(package, COT, index, 359)
    report = package.source_json(base + "REPORT.json")
    common_report = package.source_json(base + "COMMON_COMPLETE_COMPARISON.json")
    primary_ids = available(index, "N128") & available(index, "R256")
    common_ids = set.intersection(*(available(index, arm) for arm in arms))
    require(len(primary_ids) == 90 and len(common_ids) == 89, "CoT matched denominators differ")
    require(common_ids == set(common_report["common_four_arm_ids"]), "CoT common89 IDs differ")
    primary = {}
    for arm in ("N128", "N256", "R256"):
        summary = cohort_summary(index, primary_ids, arm)
        require(set(summary["dataset_counts"].values()) == {30}, "CoT primary dataset balance differs")
        check_summary(summary, report["arms"][arm], "CoT primary " + arm)
        check_summary(summary, report["equal_weight_dataset_macro"][arm], "CoT primary macro " + arm, True)
        primary[arm] = summary
    common = {}
    for arm in arms:
        summary = cohort_summary(index, common_ids, arm)
        expected = common_report["common_four_arm"]
        check_summary(summary, expected["arms"][arm], "CoT common89 " + arm)
        check_summary(summary, expected["equal_weight_dataset_macro"][arm], "CoT common89 macro " + arm, True)
        for own, target in (("prompt_tokens", "mean_input_tokens"), ("completion_tokens", "mean_output_tokens"),
                            ("total_tokens", "mean_total_tokens")):
            equal_number(summary["mean_tokens"][own], expected["arms"][arm][target], "CoT common89 tokens " + arm)
        equal_number(summary["mean_logical_estimated_CNY"], expected["arms"][arm]["mean_logical_estimated_CNY"],
                     "CoT common89 cost " + arm)
        common[arm] = summary
    pairs = {}
    for control, candidate, ids in (("N128", "N256", primary_ids), ("N128", "R256", primary_ids),
                                    ("N128", "F256", common_ids), ("F256", "R256", common_ids)):
        key = control + "_to_" + candidate
        actual = paired(index, ids, control, candidate)
        expected = report["paired_predeclared"][key] if len(ids) == 90 else common_report["common_four_arm"]["paired_predeclared"][key]
        check_pair(actual, expected, "CoT pair " + key)
        pairs[key] = actual
    held = [r["task_id"] for r in index.values() if r.get("f1") is None]
    require(held == ["musique:3hop1__604797_443779_52195/F256"], "CoT held identity differs")
    return {"primary90": primary, "common89": common, "paired": pairs, "held": held,
            "native_scores_preserved": True, "semantic_correctness_not_inferred": True}


def verify_context(package):
    base = "outputs/query_decomposition/" + CONTEXT + "/"
    arms = ("reset", "flat", "chat")
    index = quality_index(package.source_rows(base + "DETAILS.jsonl"), arms)
    require(len(index) == 269, "Shared context expected 269 successful native rows")
    check_outcome_export(package, CONTEXT, index, 269)
    ids = set.intersection(*(available(index, arm) for arm in arms))
    require(len(ids) == 89, "Shared context matched denominator differs")
    report = package.source_json(base + "ROOT_FINAL_RESULTS.json")
    require(report["common_query_n"] == 89, "Shared context report denominator differs")
    summaries = {}
    for arm in arms:
        summary = cohort_summary(index, ids, arm)
        require(summary["dataset_counts"] == report["dataset_counts"], "Shared context dataset counts differ")
        expected = report["overall"]["arms"][arm]
        check_summary(summary, expected, "Shared context " + arm)
        for token in summary["mean_tokens"]:
            equal_number(summary["mean_tokens"][token], expected["mean_tokens"][token], "Shared context tokens " + arm)
        summaries[arm] = summary
    pairs = {}
    for control, candidate in (("reset", "flat"), ("reset", "chat"), ("flat", "chat")):
        key = candidate + "_minus_" + control
        value = paired(index, ids, control, candidate)
        check_pair(value, report["overall"]["paired_F1"][key], "Shared context pair " + key)
        pairs[key] = value
    return {"common89": summaries, "paired": pairs,
            "historical_nonconcurrent_arm_not_reaggregated": True}


def verify_reference(package):
    source = "outputs/query_decomposition/" + REFERENCE + "/checkpoint.sqlite3"
    require(source in package.databases, "Reference checkpoint export missing")
    database = package.databases[source]
    order = package.source_rows("experiments/query_decomposition/" + REFERENCE + "/ORDER360.jsonl")
    planned = {row["task_id"] for row in order}
    require(len(order) == len(planned) == 360, "Reference planned conditions differ")
    outcomes = database["outcomes"]
    require(len(outcomes) == 61 and set(outcomes) <= planned, "Reference partial outcome count/IDs differ")
    require(all(row["status"] == "success" for row in outcomes.values()), "Reference non-success outcome")
    require(database["request_statuses"].get("in_flight", 0) == 0, "Reference still has in-flight requests")
    return {"planned": 360, "completed": 61, "not_started": len(planned - set(outcomes)),
            "by_arm": dict(sorted(Counter(row["arm"] for row in outcomes.values()).items())),
            "status": "partial_paused_not_complete90", "full90_quality_not_inferred": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", type=Path, help="Optional output report; omit for read-only verification")
    args = parser.parse_args()
    results = {"schema": "p02_publication_offline_verification_v1", "scope":
               "Package integrity, scalar SQLite table exports, and independent aggregation of selected preserved native scores; "
               "not answer rescoring, confidence-interval resampling, semantic validation, full prompt replay or all-experiment numerical audit.",
               "network_or_models_used": False, "checks": []}
    with ExitStack() as stack:
        package = Package(args.package, stack)
        checks = [("delivery_manifest", package.verify_delivery), ("archive_members", package.verify_archives),
                  ("sqlite_exports", package.verify_exports), ("cot90_native_reaggregation", lambda: verify_cot(package)),
                  ("shared_context89_native_reaggregation", lambda: verify_context(package)),
                  ("reference_factor_partial61", lambda: verify_reference(package))]
        for name, operation in checks:
            try:
                detail = operation()
                results["checks"].append({"name": name, "passed": True, "detail": detail})
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
                results["checks"].append({"name": name, "passed": False,
                                          "error": type(exc).__name__ + ": " + str(exc)})
    results["passed"] = all(check["passed"] for check in results["checks"])
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
