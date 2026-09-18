"""Identity-only BEIR test audit; real metadata access requires a reviewed freeze.

No model, retrieval, answer-outcome, or network code is imported. The JSON byte
scanner decodes only ID/field names and, for train/test rows, the answer and
supporting-fact titles needed by the historical canonical group rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import string
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent / "work/router_research"
DATA = REPO / "data/beir/hotpotqa"
OUT = REPO / "outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1"
PLAN = REPO / "analysis/hotpotqa_router/m6_test_group_identity_plan_20260915.md"
SOURCES = {"script": Path(__file__), "plan": PLAN}
INPUTS = {
    "queries": DATA / "queries/queries.jsonl",
    "train_qrels": DATA / "qrels/train.tsv",
    "dev_qrels": DATA / "qrels/dev.tsv",
    "test_qrels": DATA / "qrels/test.tsv",
    "internal_split": REPO / "outputs/router/hotpotqa_bd_router_v1/split.json",
    "dev_identities": WORK / "beir_dev_group_identities.npz",
    "validated_identities": WORK / "m6_beir_validation_v1/identities.npz",
    "validated_protocol": WORK / "m6_beir_validation_v1/protocol.json",
    "eligibility": WORK / "beir_dev_eligibility.json",
    "eligibility_checks": WORK / "beir_dev_eligibility_checks.json",
    "source_supplement": WORK / "beir_dev_validation_supplement.json",
    "history_inventory": REPO / "analysis/hotpotqa_router/m6_confirmation_inventory_20260915.json",
    "history_sources": WORK / "m6_validation_sources.json",
    "history_checks": WORK / "m6_validation_sources_checks.json",
}
EXPECTED_DEV = {
    "old_group_equivalence_queries": 85000,
    "old_group_equivalence_groups": 83167,
    "dev_queries": 5447,
    "dev_groups": 5439,
    "dev_queries_sharing_old_group": 225,
    "dev_groups_sharing_old_group": 220,
    "dev_queries_sharing_BEIR_test_group": 21,
    "dev_groups_sharing_BEIR_test_group": 21,
    "dev_queries_sharing_both": 8,
    "dev_queries_without_either_group_overlap": 5209,
    "dev_groups_without_either_group_overlap": 5206,
}
CONFIG = {
    "schema_version": 1,
    "expected_queries": {"train": 85000, "dev": 5447, "test": 7405},
    "expected_query_archive_rows": 97852,
    "expected_reference_metadata_rows": 92405,
    "expected_old_groups": 83167,
    "expected_validated_queries": 5209,
    "expected_validated_groups": 5206,
    "expected_history_group_union": 88373,
    "minimum_test_old_shared_groups": 8,
    "canonical_rule": "sha256_utf8_json_compact_[lower_ascii_punctuation_remove_unicode_articles_remove_whitespace_collapse(answer),sorted_unique_exact_support_titles]",
    "id_order": "Python lexicographic ascending; preserve every test ID",
    "selection_history_status": "unresolved; group isolation is not confirmation eligibility",
    "new_validation_subset_selection": False,
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
JSON_INTEGER = re.compile(rb"-?(?:0|[1-9][0-9]*)")


class AuditError(Exception):
    """Only fixed, non-payload error codes may be passed here."""


def require(condition, code):
    if not condition:
        raise AuditError(code)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def set_sha(values):
    return json_sha(sorted(set(values)))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bindings(paths):
    return {key: {"path": str(path.resolve()), "sha256": sha(path)} for key, path in paths.items()}


def normalize_answer(answer):
    text = "".join(c for c in answer.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def group_key(answer, titles):
    return json_sha([normalize_answer(answer), sorted(set(titles))])


class Projection:
    """A byte cursor which never JSON-decodes the skipped query text."""

    def __init__(self, line, counters=None):
        require(isinstance(line, bytes), "projection_requires_bytes")
        self.line = line
        self.i = 0
        self.counters = counters if counters is not None else Counter()

    def ws(self):
        while self.i < len(self.line) and self.line[self.i] in b" \t\r\n":
            self.i += 1

    def token(self, byte):
        self.ws()
        require(self.line[self.i:self.i + 1] == byte, "unexpected_json_token")
        self.i += 1

    def string_end(self):
        self.ws()
        require(self.line[self.i:self.i + 1] == b'"', "expected_json_string")
        self.i += 1
        while self.i < len(self.line):
            char = self.line[self.i]
            if char == 34:
                self.i += 1
                return self.i
            if char == 92:
                self.i += 1
                require(self.i < len(self.line), "truncated_json_escape")
                esc = self.line[self.i]
                if esc == 117:
                    chunk = self.line[self.i + 1:self.i + 5]
                    require(len(chunk) == 4 and all(c in b"0123456789abcdefABCDEF" for c in chunk), "invalid_unicode_escape")
                    self.i += 5
                else:
                    require(esc in b'"\\/bfnrt', "invalid_json_escape")
                    self.i += 1
            else:
                require(char >= 32, "unescaped_json_control")
                self.i += 1
        raise AuditError("unterminated_json_string")

    def decoded_string(self, role):
        self.ws()
        start = self.i
        stop = self.string_end()
        try:
            value = json.loads(self.line[start:stop])
        except (ValueError, UnicodeError):
            raise AuditError("invalid_projected_string") from None
        require(isinstance(value, str), "projected_value_not_string")
        self.counters[role] += 1
        return value

    def skip_value(self):
        self.ws()
        char = self.line[self.i:self.i + 1]
        if char == b'"':
            self.string_end()
        elif char in (b"{", b"["):
            end = b"}" if char == b"{" else b"]"
            self.i += 1
            self.ws()
            if self.line[self.i:self.i + 1] == end:
                self.i += 1
                return
            while True:
                if char == b"{":
                    self.string_end()
                    self.token(b":")
                self.skip_value()
                self.ws()
                if self.line[self.i:self.i + 1] == end:
                    self.i += 1
                    return
                self.token(b",")
        else:
            match = re.match(rb"(?:null|true|false|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", self.line[self.i:])
            require(match is not None, "invalid_skipped_json_value")
            self.i += match.end()

    def titles(self):
        self.token(b"[")
        values = []
        while True:
            self.token(b"[")
            values.append(self.decoded_string("support_title_strings"))
            self.token(b",")
            self.ws()
            match = JSON_INTEGER.match(self.line, self.i)
            require(match is not None, "support_sentence_index_not_integer")
            self.i = match.end()  # Index syntax checked; its numerical value is unused.
            self.token(b"]")
            self.ws()
            if self.line[self.i:self.i + 1] == b"]":
                self.i += 1
                return values
            self.token(b",")

    def metadata_hash(self):
        self.token(b"{")
        seen = set()
        answer = titles = None
        while True:
            field = self.decoded_string("field_name_strings")
            require(field in {"answer", "supporting_facts"} and field not in seen, "unexpected_metadata_field")
            seen.add(field)
            self.token(b":")
            if field == "answer":
                answer = self.decoded_string("reference_answer_strings")
            else:
                titles = self.titles()
            self.ws()
            if self.line[self.i:self.i + 1] == b"}":
                self.i += 1
                break
            self.token(b",")
        require(seen == {"answer", "supporting_facts"}, "missing_metadata_field")
        key = group_key(answer, titles)
        self.counters["reference_metadata_rows"] += 1
        return key

    def project(self, selected_ids):
        self.token(b"{")
        require(self.decoded_string("field_name_strings") == "_id", "id_must_be_first_field")
        self.token(b":")
        qid = self.decoded_string("query_id_strings")
        require(bool(qid), "empty_query_id")
        selected = qid in selected_ids
        seen = {"_id"}
        key = None
        while True:
            self.ws()
            if self.line[self.i:self.i + 1] == b"}":
                self.i += 1
                break
            self.token(b",")
            field = self.decoded_string("field_name_strings")
            require(field in {"text", "metadata"} and field not in seen, "unexpected_top_level_field")
            seen.add(field)
            self.token(b":")
            if field == "text":
                self.string_end()
                self.counters["query_text_strings_skipped"] += 1
            elif selected:
                key = self.metadata_hash()
            else:
                self.skip_value()
                self.counters["metadata_rows_skipped"] += 1
        self.ws()
        require(self.i == len(self.line) and seen == {"_id", "text", "metadata"}, "incomplete_query_record")
        return qid, key


def add_bijection(group_to_hash, hash_to_group, gid, key):
    require(group_to_hash.setdefault(gid, key) == key, "one_old_group_has_multiple_hashes")
    require(hash_to_group.setdefault(key, gid) == gid, "one_hash_has_multiple_old_groups")


def overlap_counts(keys, reference):
    hits = [key for key in keys if key in reference]
    return {"queries": len(hits), "groups": len(set(hits)), "group_set_sha256": set_sha(hits)}


def self_test():
    """Synthetic values only: no workspace input file is opened here."""
    passed = []

    def check(name, condition):
        require(condition, "synthetic_check_failed_" + name)
        passed.append(name)

    def rejected(name, function):
        try:
            function()
        except AuditError:
            passed.append(name)
        else:
            raise AuditError("synthetic_rejection_failed_" + name)

    def independent_normalize(text):
        punctuation = {chr(c) for a, b in ((33, 47), (58, 64), (91, 96), (123, 126)) for c in range(a, b + 1)}
        cleaned = "".join(c for c in text.lower() if c not in punctuation)
        pieces, word = [], []
        for char in cleaned + " ":
            if char.isalnum() or char == "_":
                word.append(char)
            else:
                if word:
                    token = "".join(word)
                    pieces.append(" " if token in {"a", "an", "the"} else token)
                    word.clear()
                pieces.append(char)
        return " ".join("".join(pieces).split())

    examples = [" The, Cat! ", "THE—cat", "theory anaconda", "a_the", "Straße STRASSE", "İ A\u00a0THE\tCAT", "the猫 the猫a", "Ａ a ²the a²", "The\u0301 a\n an"]
    check("independent_unicode_normalization", all(normalize_answer(v) == independent_normalize(v) for v in examples))
    check("lower_is_not_casefold", normalize_answer("Straße") != normalize_answer("STRASSE"))
    check("ascii_punctuation_only", normalize_answer("The—Cat!") == "—cat")
    check("article_word_boundaries", normalize_answer("theory anaconda") == "theory anaconda")
    check("title_deduplication_and_order", group_key("The, Cat!", ["B", "A", "B"]) == group_key("cat", ["A", "B"]))
    check("exact_title_case", group_key("cat", ["A"]) != group_key("cat", ["a"]))
    metadata = {"answer": " The, Cat! ", "supporting_facts": [["B", 1], ["A", 2], ["B", 3]]}
    counters = Counter()
    raw = json.dumps({"_id": "q1", "text": 'escaped "metadata": { } \\ / \n 猫', "metadata": metadata}, ensure_ascii=False).encode()
    qid, key = Projection(raw, counters).project({"q1"})
    check("selected_projection", qid == "q1" and key == group_key("cat", ["A", "B"]))
    check("decode_boundary", counters["query_text_strings_skipped"] == 1 and counters["reference_answer_strings"] == 1 and counters["support_title_strings"] == 3 and counters["query_text_strings"] == 0)
    changed = json.dumps({"_id": "q2", "metadata": {"supporting_facts": [["A", 900], ["B", -1]], "answer": "cat"}, "text": "different question"}).encode()
    check("id_text_index_invariance", Projection(changed).project({"q2"})[1] == key)
    skipped = Counter()
    nonselected = b'{"_id":"skip","text":"hidden","metadata":{"not_the_expected_reference_schema":[false,null,{"answer":"hidden"}]}}'
    check("nonselected_metadata_not_decoded", Projection(nonselected, skipped).project({"q1"}) == ("skip", None) and skipped["reference_answer_strings"] == skipped["support_title_strings"] == 0 and skipped["metadata_rows_skipped"] == 1)
    rejected("reject_duplicate_metadata_field", lambda: Projection(b'{"_id":"q","text":"x","metadata":{"answer":"x","answer":"y"}}').project({"q"}))
    rejected("reject_index_float", lambda: Projection(b'{"_id":"q","text":"x","metadata":{"answer":"x","supporting_facts":[["T",1.2]]}}').project({"q"}))
    rejected("reject_index_bool", lambda: Projection(b'{"_id":"q","text":"x","metadata":{"answer":"x","supporting_facts":[["T",true]]}}').project({"q"}))
    rejected("reject_non_id_first", lambda: Projection(b'{"text":"x","_id":"q","metadata":{}}').project({"q"}))
    rejected("reject_trailing_payload", lambda: Projection(raw + b" null").project({"q1"}))
    rejected("reject_bad_query_escape_without_decode", lambda: Projection(b'{"_id":"q","text":"\\z","metadata":{}}').project(set()))
    g2h, h2g = {}, {}
    add_bijection(g2h, h2g, "g1", "h1")
    add_bijection(g2h, h2g, "g1", "h1")
    add_bijection(g2h, h2g, "g2", "h2")
    check("two_way_bijection", len(g2h) == len(h2g) == 2)
    rejected("reject_group_to_multiple_hashes", lambda: add_bijection(dict(g2h), dict(h2g), "g1", "h2"))
    rejected("reject_hash_to_multiple_groups", lambda: add_bijection(dict(g2h), dict(h2g), "g3", "h1"))
    counts = overlap_counts(["a", "a", "b", "c"], {"a", "c"})
    check("query_group_overlap_denominators", counts["queries"] == 3 and counts["groups"] == 2)
    return {"status": "passed_pure_synthetic", "count": len(passed), "checks": passed, "real_input_files_read": 0}


def freeze():
    require(not OUT.exists(), "output_directory_already_exists")
    synthetic = self_test()
    # This binds raw bytes but does not project any real reference metadata.
    source_bindings, input_bindings = bindings(SOURCES), bindings(INPUTS)
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {"id": "m6_test_group_identity_audit_v1", "status": "frozen_identity_only_before_reference_projection", "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG, "sources": source_bindings, "inputs": input_bindings, "synthetic": synthetic}
    write_json(OUT / "protocol.json", protocol)
    return {"status": protocol["status"], "protocol_sha256": sha(OUT / "protocol.json"), "reference_metadata_rows_projected": 0}


def historical_path_digest(mapping, path):
    hits = [digest for saved_path, digest in mapping.items() if Path(saved_path).resolve() == path.resolve()]
    require(len(hits) == 1, "historical_path_binding_not_unique")
    return hits[0]


def verify_provenance(protocol):
    hashes = {key: record["sha256"] for key, record in protocol["inputs"].items()}
    old = read_json(INPUTS["eligibility"])
    checked = read_json(INPUTS["eligibility_checks"])
    require(checked["status"] == "passed_separate_group_projection_and_broader_identity_inventory", "old_group_check_not_passed")
    require(checked["results_sha256"] == hashes["eligibility"], "old_group_result_not_bound")
    require(old["group_counts"] == checked["group_counts"] == EXPECTED_DEV, "historical_group_counts_changed")
    require(old["identity_npz_sha256"] == hashes["dev_identities"], "old_dev_identities_not_bound")
    for key in ("queries", "train_qrels", "dev_qrels", "test_qrels", "internal_split"):
        require(historical_path_digest(old["source_sha256"], INPUTS[key]) == hashes[key], "old_group_input_binding_mismatch")
    supplement = read_json(INPUTS["source_supplement"])
    for key in ("eligibility", "eligibility_checks", "dev_identities"):
        require(historical_path_digest(supplement["inputs_sha256"], INPUTS[key]) == hashes[key], "supplement_input_binding_mismatch")
    for key in ("queries", "train_qrels", "dev_qrels", "test_qrels"):
        rows = [r for r in supplement["archive_members"] if Path(r["local_path"]).resolve() == INPUTS[key].resolve()]
        require(len(rows) == 1, "archive_member_binding_not_unique")
        row = rows[0]
        require(row["exact_bytes_equal"] and row["archive_digest"]["sha256"] == row["local_digest"]["sha256"] == hashes[key], "archive_member_binding_mismatch")
    validated = read_json(INPUTS["validated_protocol"])
    require(validated["sample"]["sha256"] == hashes["validated_identities"], "validated_sample_binding_mismatch")
    require(validated["sample"]["queries"] == 5209 and validated["sample"]["groups"] == 5206, "validated_sample_size_mismatch")
    require(validated["query_source"]["sha256"] == hashes["queries"], "validated_query_source_mismatch")
    history = read_json(INPUTS["history_checks"])
    require(history["status"] == "passed_separate_identity_union_and_qrel_first_column_counts", "history_check_not_passed")
    require(history["results_sha256"] == hashes["history_sources"], "history_source_binding_mismatch")
    require(history["all_internal_partitions_group_unexposed_queries"] == 0, "history_group_coverage_changed")
    return {"historical_group_audit_and_checker_bound": True, "official_archive_member_bindings_reused": True, "original_source_split_independently_regenerated": False, "validated_sample_protocol_bound": True, "prior_internal_group_exposure_check_bound": True, "history_inventory_bound_without_rerunning_search": True}


def qrel_ids(path):
    ids = set()
    with path.open("rb") as stream:
        require(next(stream).startswith(b"query-id\t"), "qrels_header_mismatch")
        for line in stream:
            first, sep, _ = line.partition(b"\t")
            require(bool(sep) and bool(first), "invalid_qrel_id_column")
            ids.add(first.decode("utf-8"))
    return ids


def identity_array(archive, field, length=None, is_hash=False):
    value = archive[field]
    require(value.ndim == 1 and value.dtype.kind == "U", "identity_array_not_unicode_vector")
    if length is not None:
        require(len(value) == length, "identity_array_length_mismatch")
    values = value.tolist()
    require(all(bool(v) for v in values), "empty_identity")
    if is_hash:
        require(all(HEX64.fullmatch(v) for v in values), "invalid_group_hash")
    return values


def save_npz(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def run(protocol_sha):
    protocol_path = OUT / "protocol.json"
    require(sha(protocol_path) == protocol_sha, "protocol_sha_mismatch")
    protocol = read_json(protocol_path)
    require(protocol["config"] == CONFIG and protocol["sources"] == bindings(SOURCES), "frozen_source_or_config_mismatch")
    require(protocol["inputs"] == bindings(INPUTS), "frozen_input_mismatch")
    require({p.name for p in OUT.iterdir()} == {"protocol.json"}, "audit_already_started_or_output_not_empty")
    provenance = verify_provenance(protocol)
    splits = {part: qrel_ids(INPUTS[part + "_qrels"]) for part in ("train", "dev", "test")}
    require({k: len(v) for k, v in splits.items()} == CONFIG["expected_queries"], "qrel_split_size_mismatch")
    require(not (splits["train"] & splits["dev"] or splits["train"] & splits["test"] or splits["dev"] & splits["test"]), "qrel_splits_overlap")
    rows = read_json(INPUTS["internal_split"])["assignments"]
    lookup = {row["query_id"]: row["group_id"] for row in rows}
    require(len(rows) == len(lookup) == 85000 and set(lookup) == splits["train"], "old_split_id_mismatch")
    require(all(isinstance(q, str) and isinstance(g, str) and g for q, g in lookup.items()), "old_split_identity_type_mismatch")
    with np.load(INPUTS["dev_identities"], allow_pickle=False) as archive:
        dev_q = identity_array(archive, "query_ids", 5447)
        dev_h = identity_array(archive, "group_key_sha256", 5447, True)
        saved_flags = {}
        for name in ("overlaps_old_group", "overlaps_BEIR_test_group", "no_either_group_overlap"):
            flag = archive[name]
            require(flag.shape == (5447,) and flag.dtype.kind == "b", "dev_flag_shape_or_type_mismatch")
            saved_flags[name] = flag.copy()
    require(dev_q == sorted(splits["dev"]), "dev_identity_order_or_set_mismatch")
    with np.load(INPUTS["validated_identities"], allow_pickle=False) as archive:
        val_q = identity_array(archive, "query_ids", 5209)
        val_h = identity_array(archive, "group_ids", 5209, True)
    keep = saved_flags["no_either_group_overlap"]
    require(val_q == [q for q, flag in zip(dev_q, keep) if flag] and val_h == [h for h, flag in zip(dev_h, keep) if flag], "validated_dev_bridge_mismatch")
    write_json(OUT / "run_started.json", {"status": "identity_projection_started", "protocol_sha256": protocol_sha, "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "92405 reference identities only; all test rows retained; no outcome or policy access"})
    selected = splits["train"] | splits["test"]
    all_ids = set().union(*splits.values())
    seen, old_hash, test_hash, group_to_hash, hash_to_group = set(), {}, {}, {}, {}
    counters = Counter()
    with INPUTS["queries"].open("rb") as stream:
        for line in stream:
            qid, key = Projection(line, counters).project(selected)
            require(qid in all_ids and qid not in seen, "unexpected_or_duplicate_query_id")
            seen.add(qid)
            if qid in splits["train"]:
                old_hash[qid] = key
                add_bijection(group_to_hash, hash_to_group, lookup[qid], key)
            elif qid in splits["test"]:
                test_hash[qid] = key
            else:
                require(key is None, "dev_metadata_was_projected")
    require(seen == all_ids and len(seen) == 97852, "archive_identity_coverage_mismatch")
    require(set(old_hash) == splits["train"] and set(test_hash) == splits["test"], "rebuilt_identity_coverage_mismatch")
    require(len(group_to_hash) == len(hash_to_group) == 83167, "old_group_bijection_size_mismatch")
    require(counters["reference_metadata_rows"] == counters["reference_answer_strings"] == 92405 and counters["metadata_rows_skipped"] == 5447 and counters["query_text_strings_skipped"] == 97852, "projection_boundary_counter_mismatch")
    old_keys, test_keys, dev_keys, val_keys = set(old_hash.values()), set(test_hash.values()), set(dev_h), set(val_h)
    require(len(val_keys) == 5206 and not (old_keys & val_keys), "validated_history_group_union_mismatch")
    history_keys = old_keys | val_keys
    require(len(history_keys) == 88373, "history_group_union_size_mismatch")
    old_hit = np.array([key in old_keys for key in dev_h])
    test_hit = np.array([key in test_keys for key in dev_h])
    for name, actual in (("overlaps_old_group", old_hit), ("overlaps_BEIR_test_group", test_hit), ("no_either_group_overlap", ~(old_hit | test_hit))):
        require(np.array_equal(saved_flags[name], actual), "dev_group_bridge_flag_mismatch")
    both = old_hit & test_hit
    bridge = {key for key, flag in zip(dev_h, both) if flag}
    counts_dev = {"old_group_equivalence_queries": len(old_hash), "old_group_equivalence_groups": len(old_keys), "dev_queries": len(dev_q), "dev_groups": len(dev_keys), "dev_queries_sharing_old_group": int(old_hit.sum()), "dev_groups_sharing_old_group": len(set(np.array(dev_h)[old_hit])), "dev_queries_sharing_BEIR_test_group": int(test_hit.sum()), "dev_groups_sharing_BEIR_test_group": len(set(np.array(dev_h)[test_hit])), "dev_queries_sharing_both": int(both.sum()), "dev_queries_without_either_group_overlap": int((~(old_hit | test_hit)).sum()), "dev_groups_without_either_group_overlap": len(set(np.array(dev_h)[~(old_hit | test_hit)]))}
    require(counts_dev == EXPECTED_DEV, "recomputed_dev_counts_mismatch")
    require(len(bridge) == 8 and bridge <= (old_keys & test_keys) and len(old_keys & test_keys) >= 8, "eight_group_overlap_lower_bound_failed")
    require(not (test_keys & val_keys), "test_validated5209_group_bridge_mismatch")
    test_q = sorted(test_hash)
    test_h = [test_hash[q] for q in test_q]
    flags = {
        "overlaps_internal85000_group": np.array([h in old_keys for h in test_h]),
        "overlaps_validated5209_group": np.array([h in val_keys for h in test_h]),
        "overlaps_dev5447_group": np.array([h in dev_keys for h in test_h]),
        "overlaps_other_consumed_router_group": np.array([h in history_keys for h in test_h]),
        "shares_train_dev_test_bridge_group": np.array([h in bridge for h in test_h]),
        "canonical_group_isolated_from_router_history": np.array([h not in history_keys for h in test_h]),
        "prior_retrieval_benchmark_exposure": np.ones(7405, dtype=bool),
    }
    # Verify identity-only inventory evidence for all-target retrieval exposure.
    inventory = read_json(INPUTS["history_inventory"])
    hits = [r for r in inventory["npz_identity_records"] if r["test_query_overlap"] > 0]
    require(len(hits) == 6 and all(r["test_query_overlap"] == r["query_count"] == 7405 and r["query_id_set_sha256"] == set_sha(test_q) for r in hits), "target_retrieval_inventory_binding_mismatch")
    require(protocol["inputs"] == bindings(INPUTS), "input_changed_during_projection")
    old_q = sorted(old_hash)
    save_npz(OUT / "old85000_group_identities.npz", query_ids=np.array(old_q), original_group_ids=np.array([lookup[q] for q in old_q]), group_key_sha256=np.array([old_hash[q] for q in old_q]))
    save_npz(OUT / "test7405_group_identities.npz", query_ids=np.array(test_q), group_key_sha256=np.array(test_h), **flags)
    intersections = {name: overlap_counts(test_h, ref) for name, ref in (("internal85000", old_keys), ("validated5209", val_keys), ("all_dev5447", dev_keys), ("other_consumed_router_history", history_keys), ("train_dev_test_bridge", bridge))}
    isolated = [h for h in test_h if h not in history_keys]
    results = {
        "status": "complete_identity_audit_selection_history_unresolved",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha,
        "provenance_checks": provenance,
        "query_counts": {"internal": len(old_q), "all_dev": len(dev_q), "validated": len(val_q), "test": len(test_q)},
        "group_counts": {"internal": len(old_keys), "all_dev": len(dev_keys), "validated": len(val_keys), "test": len(test_keys), "other_consumed_router_history": len(history_keys)},
        "old_bijection": {"query_pairs_checked": len(old_q), "original_group_to_canonical_hash_entries": len(group_to_hash), "canonical_hash_to_original_group_entries": len(hash_to_group), "passed": True},
        "historical_dev_bridge_counts": counts_dev,
        "eight_group_lower_bound": {"bridge_groups": len(bridge), "actual_test_internal_shared_groups": len(old_keys & test_keys), "passed": True},
        "test_overlap_with": intersections,
        "test_id_overlap_with": {"internal85000": len(set(test_q) & set(old_q)), "all_dev5447": len(set(test_q) & set(dev_q)), "validated5209": len(set(test_q) & set(val_q))},
        "canonical_group_isolated_from_router_history": {"queries": len(isolated), "groups": len(set(isolated)), "is_confirmation_eligibility": False},
        "test_query_id_set_sha256": set_sha(test_q),
        "test_canonical_group_set_sha256": set_sha(test_h),
        "projection_counters": dict(counters),
        "boundary": {"query_text_fields_deserialized": 0, "dev_reference_metadata_rows_projected": 0, "support_sentence_index_values_decoded": 0, "reference_texts_exported": 0, "new_outcome_payload_reads": 0, "new_effect_computations": 0, "new_policy_predictions": 0, "new_fits": 0, "new_network_calls": 0, "new_paid_calls": 0, "test_rows_saved": 7405, "new_validation_subset_selected": False, "independent_confirmation_eligibility_certified": False},
        "interpretation": "Exact historical canonical-key overlap only. All test IDs were previously retrieval-benchmarked. A missing overlap is not proof of semantic independence, no unrecorded prior use, or selection independence. The complete 7405-row inventory is not a selected confirmation cohort.",
        "artifact_sha256": {name: sha(OUT / name) for name in ("old85000_group_identities.npz", "test7405_group_identities.npz")},
    }
    write_json(OUT / "results.json", results)
    write_json(OUT / "completion.json", {"status": results["status"], "protocol_sha256": protocol_sha, "artifact_sha256": {name: sha(OUT / name) for name in ("run_started.json", "old85000_group_identities.npz", "test7405_group_identities.npz", "results.json")}})
    return {"status": results["status"], "test_queries": len(test_q), "test_groups": len(test_keys), "test_overlap_with": intersections, "boundary": results["boundary"], "results_sha256": sha(OUT / "results.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test", help="Synthetic checks only; no real inputs read")
    sub.add_parser("freeze", help="After review: bind source/input bytes, no metadata projection")
    run_parser = sub.add_parser("run", help="After review and freeze: project reference identities only")
    run_parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    try:
        result = self_test() if args.command == "self-test" else freeze() if args.command == "freeze" else run(args.protocol_sha256)
    except Exception as error:
        # A decoder exception can carry its document. Never print raw exceptions
        # or tracebacks that could surface a projected reference string.
        code = str(error) if isinstance(error, AuditError) else "unexpected_error_payload_suppressed"
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "code": code}))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
