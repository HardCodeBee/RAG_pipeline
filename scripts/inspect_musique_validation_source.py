"""Source/identity qualification only: no retrieval, fitting, generation or scoring.

Reuses the already established HotpotQA canonical hashes instead of reauditing
the old experiments. Reads public references solely to construct identity keys;
does not print questions, answers, or choose a confirmation subset.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/router/external_sources/musique_v1.0"
HISTORY = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1"


def normalize_question(text):
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def canonical_key(answer, titles):
    # Same lower/ASCII-punctuation/articles/whitespace rule as historical hashes.
    text = "".join(char for char in answer.lower() if char not in string.punctuation)
    text = " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())
    packed = json.dumps([text, sorted(set(titles))], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def read_queries(dataset):
    path = ROOT / "data/beir" / dataset / "queries/queries.jsonl"
    with path.open(encoding="utf-8") as stream:
        return {normalize_question(json.loads(line)["text"]) for line in stream if line.strip()}


def main():
    history = set()
    for path in (HISTORY / "old85000_group_identities.npz",
                 HISTORY / "test7405_group_identities.npz",
                 ROOT.parent / "work/router_research/beir_dev_group_identities.npz"):
        with np.load(path, allow_pickle=False) as archive:
            history.update(archive["group_key_sha256"].astype(str))
    hotpot_queries, nq_queries = read_queries("hotpotqa"), read_queries("nq")
    singlehop = json.loads((SOURCE / "dev_test_singlehop_questions_v1.0.json").read_text(encoding="utf-8"))
    seed_texts = {
        name: {normalize_question(row["question"]) for row in rows}
        for name, rows in singlehop.items()
    }
    with (SOURCE / "musique_ans_v1.0_dev.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(left)] = find(right)

    owners, paragraphs, ids, flags = {}, set(), set(), []
    paragraph_counts, hop_counts, canonical_counts = Counter(), Counter(), Counter()
    dev_seed_types = Counter()
    for index, row in enumerate(rows):
        assert row["answerable"] is True and row["id"] not in ids
        ids.add(row["id"])
        titles = [paragraph["title"] for paragraph in row["paragraphs"] if paragraph["is_supporting"]]
        key = canonical_key(row["answer"], titles)
        canonical_counts[key] += 1
        aliases = [row["answer"], *row["answer_aliases"]]
        keys = {canonical_key(answer, titles) for answer in aliases}
        question = normalize_question(row["question"])
        decomposition = {normalize_question(step["question"]) for step in row["question_decomposition"]}
        for name, texts in seed_texts.items():
            dev_seed_types[name] += int(bool(decomposition & texts))
        flags.append({
            "primary_canonical_overlap_hotpot_history": key in history,
            "any_alias_canonical_overlap_hotpot_history": bool(keys & history),
            "composed_question_overlap_hotpot": question in hotpot_queries,
            "composed_question_overlap_nq": question in nq_queries,
            "decomposition_question_overlap_hotpot": bool(decomposition & hotpot_queries),
            "decomposition_question_overlap_nq": bool(decomposition & nq_queries),
        })
        # Dependency components use shared constituent IDs OR shared reference
        # answer/title identities. These are stronger than unique composed IDs.
        tokens = [("seed", step["id"]) for step in row["question_decomposition"]]
        tokens.extend(("canonical", value) for value in keys)
        for token in tokens:
            if token in owners:
                union(index, owners[token])
            else:
                owners[token] = index
        paragraph_counts[len(row["paragraphs"])] += 1
        hop_counts[len(row["question_decomposition"])] += 1
        paragraphs.update((paragraph["title"], paragraph["paragraph_text"]) for paragraph in row["paragraphs"])
    components = Counter(find(index) for index in range(len(rows)))
    flagged_components = {find(index) for index, values in enumerate(flags) if any(values.values())}
    result = {
        "status": "identity_only_candidate_source_not_frozen_confirmation",
        "source": "MuSiQue-Ans v1.0 official dev; source cache provenance in source.json",
        "questions": len(rows), "unique_ids": len(ids),
        "paragraph_count_distribution": dict(sorted(paragraph_counts.items())),
        "hop_count_distribution": dict(sorted(hop_counts.items())),
        "unique_title_paragraph_pairs_in_dev": len(paragraphs),
        "primary_answer_support_title_groups": len(canonical_counts),
        "dependency_components": len(components),
        "component_size_distribution": dict(sorted(Counter(components.values()).items())),
        "largest_components": sorted(components.values(), reverse=True)[:10],
        "history_canonical_groups_reused": len(history),
        "normalized_history_query_counts": {"hotpotqa": len(hotpot_queries), "nq": len(nq_queries)},
        "overlap_query_counts": {key: sum(row[key] for row in flags) for key in flags[0]},
        "queries_with_any_identity_flag": sum(any(row.values()) for row in flags),
        "components_with_any_identity_flag": len(flagged_components),
        "queries_in_flagged_components": sum(components[group] for group in flagged_components),
        "dev_test_source_singlehop_counts": {key: len(value) for key, value in singlehop.items()},
        "dev_queries_matching_published_singlehop_text_by_source": dict(dev_seed_types),
        "published_dev_test_seed_text_overlap_local_nq": {
            name: len(texts & nq_queries) for name, texts in seed_texts.items()
        },
        "limitations": [
            "Exact normalized text and reference-key overlap only; not a semantic independence proof.",
            "Source single-hop list covers both dev and test; source attribution by text is not ID mapping.",
            "Pretraining exposure remains unknown; local history counts are not model training provenance.",
            "No frozen corpus, independent confirmation subset, paired outcomes, or effect estimates yet.",
        ],
    }
    with (SOURCE / "qualification.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
