"""One metadata-only screen of official 2Wiki dev in the existing RAG corpus.

No retrieval, model execution, outcome selection, new corpus or index. Support
sentences are the annotated evidence unit; whole-paragraph containment is also
reported. Neither condition proves freshness, answer validity or policy gain.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from inspect_musique_validation_source import canonical_key, normalize_question, read_queries


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/router/external_sources/2wiki_april7_2021"
BUILD = ROOT / "artifacts/build_65e9037e4c1eed7b"
HISTORY = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1"


def whitespace(text):
    return " ".join(text.split())


def main():
    started = time.monotonic()
    output = SOURCE / "current_corpus_compatibility.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    if previous and previous["status"] != "prepared_before_corpus_scan":
        raise FileExistsError("Reuse the existing screen; do not repeat the corpus scan")
    rows = json.loads((SOURCE / "dev.json").read_text(encoding="utf-8"))
    manifest = json.loads((BUILD / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    aliases = {}
    with (SOURCE / "id_aliases.json").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            aliases[row["Q_id"]] = set(row["aliases"] + row["demonyms"])
    history = set()
    for path in (HISTORY / "old85000_group_identities.npz", HISTORY / "test7405_group_identities.npz",
                 ROOT.parent / "work/router_research/beir_dev_group_identities.npz"):
        with np.load(path, allow_pickle=False) as archive:
            history.update(archive["group_key_sha256"].astype(str))
    hotpot, nq = read_queries("hotpotqa"), read_queries("nq")
    parent, owners = list(range(len(rows))), {}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    annotations, identity_flags, required, ids = [], [], defaultdict(dict), set()
    for index, row in enumerate(rows):
        assert row["_id"] not in ids
        ids.add(row["_id"])
        contexts = defaultdict(set)
        for title, sentences in row["context"]:
            contexts[title].add(tuple(sentences))
        support = defaultdict(list)
        for title, position in row["supporting_facts"]:
            support[title].append(position)
        assert support
        sentences_by_title, paragraphs_by_title, valid = {}, {}, True
        for title, positions in support.items():
            if len(contexts[title]) != 1:
                valid = False
                continue
            sentences = next(iter(contexts[title]))
            if any(not isinstance(position, int) or position < 0 or position >= len(sentences) for position in positions):
                valid = False
                continue
            texts = set(whitespace(sentences[position]) for position in positions)
            paragraph = whitespace(" ".join(sentences))
            if not paragraph or not all(texts):
                valid = False
                continue
            sentences_by_title[title], paragraphs_by_title[title] = texts, paragraph
            for text in texts | {paragraph}:
                required[title].setdefault(text, [])
        titles = sorted(support)
        answers = {row["answer"]} | aliases.get(row["answer_id"], set())
        keys = {canonical_key(answer, titles) for answer in answers}
        question = normalize_question(row["question"])
        identity_flags.append({"canonical_hotpot_history": bool(keys & history),
                               "question_hotpot": question in hotpot, "question_nq": question in nq})
        tokens = [("canonical", key) for key in keys] + [("question", question)]
        tokens.extend(("evidence_text", *(normalize_question(value) for value in triple)) for triple in row["evidences"])
        tokens.extend(("evidence_id", *triple) for triple in row["evidences_id"])
        for token in tokens:
            if token in owners:
                parent[find(index)] = find(owners[token])
            else:
                owners[token] = index
        annotations.append({"valid": valid, "titles": titles, "sentences": sentences_by_title,
                            "paragraphs": paragraphs_by_title})
    components = Counter(find(index) for index in range(len(rows)))
    flagged = {find(index) for index, flags in enumerate(identity_flags) if any(flags.values())}
    result = {
        "status": "prepared_before_corpus_scan", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": manifest["build_id"], "questions": len(rows), "dependency_components": len(components),
        "largest_dependency_components": sorted(components.values(), reverse=True)[:10],
        "history_canonical_groups_reused": len(history),
        "direct_identity_overlap_counts": {key: sum(flags[key] for flags in identity_flags) for key in identity_flags[0]},
        "queries_in_flagged_components": sum(components[group] for group in flagged),
        "valid_support_annotation_questions": sum(item["valid"] for item in annotations),
        "support_titles_required": len(required),
        "rules": {
            "title": "exact title against existing chunk source",
            "support_sentences": "all annotated sentences for a title must be wholly contained in one same-title document body; whitespace collapsed only",
            "support_paragraphs": "supplementary: whole source support paragraph contained in one same-title document body; whitespace collapsed only",
            "components": "union shared atomic evidence text or ID triple, any official alias/demonym answer-support-title canonical key, or normalized question",
            "history": "exclude entire component with canonical Hotpot history or composed question text overlap with known local Hotpot/NQ",
            "eligibility": "valid source annotations, all support sentences contained, no flagged component; provisional external conditional source only",
            "source_errors": "ambiguous same-title contexts or invalid sentence indexes are reported and excluded, never guessed",
        },
        "retrieval_calls": 0, "model_inference_calls": 0, "new_corpus_or_index": False,
    }
    if previous:
        assert previous["rules"] == result["rules"]
        result["implementation_recovery"] = "Initial scan stopped after 1M rows at title-prefix assertion; loader permits title-only documents. Restart completed scan with these treated as empty bodies, preserving eligibility rules."
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    title_docs = defaultdict(list)
    scanned, title_only_documents = 0, 0
    with (BUILD / "chunks.jsonl").open("rb") as stream:
        for scanned, line in enumerate(stream, 1):
            chunk = json.loads(line)
            title = chunk["source"]
            if title in required:
                prefix = title + "\n"
                if chunk["text"] == title:
                    body = ""
                    title_only_documents += 1
                else:
                    assert chunk["text"].startswith(prefix), (scanned, chunk["doc_id"], repr(title), repr(chunk["text"][:80]))
                    body = whitespace(chunk["text"][len(prefix):])
                title_docs[title].append(chunk["doc_id"])
                for text, matches in required[title].items():
                    if text in body:
                        matches.append(chunk["doc_id"])
            if scanned % 1000000 == 0:
                print(json.dumps({"scanned_chunks": scanned, "matching_titles": len(title_docs)}), flush=True)
    assert scanned == manifest["chunking"]["num_chunks"]
    counts, by_type, details = Counter(), defaultdict(Counter), []
    for index, (row, item) in enumerate(zip(rows, annotations)):
        documents = {}
        for title, sentences in item["sentences"].items():
            documents[title] = sorted(set.intersection(*(set(required[title][text]) for text in sentences)))
        title_ok = all(title_docs[title] for title in item["titles"])
        sentence_ok = item["valid"] and all(documents[title] for title in item["titles"])
        paragraph_ok = item["valid"] and all(required[title][text] for title, text in item["paragraphs"].items())
        unflagged = find(index) not in flagged
        flags = {"all_support_titles_found": title_ok, "all_support_sentences_contained": sentence_ok,
                 "all_support_paragraphs_contained": paragraph_ok, "unflagged_component": unflagged,
                 "unflagged_and_all_sentences": unflagged and sentence_ok,
                 "unflagged_and_all_paragraphs": unflagged and paragraph_ok}
        counts.update({key: int(value) for key, value in flags.items()})
        by_type[row["type"]].update({"questions": 1, **{key: int(value) for key, value in flags.items()}})
        details.append({"query_id": row["_id"], "component": find(index), "type": row["type"],
                        "valid_source_annotations": item["valid"], **flags, "support_document_ids": documents})
    eligible = [row for row in details if row["unflagged_and_all_sentences"]]
    result.update({"status": "complete_metadata_only_current_build_compatibility", "scanned_chunks": scanned,
                   "support_titles_found": sum(bool(value) for value in title_docs.values()),
                   "matched_title_documents_without_body": title_only_documents,
                   "question_counts": dict(counts), "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
                   "eligible_components": len({row["component"] for row in eligible}),
                   "eligible_component_size_distribution": dict(sorted(Counter(Counter(row["component"] for row in eligible).values()).items())),
                   "elapsed_seconds": time.monotonic() - started,
                   "limitations": ["Not original HotpotQA confirmation or the full 2Wiki benchmark.",
                                   "Exact-text availability is not retrieval success, generator correctness, or complete semantic independence.",
                                   "Atomic-triple grouping may miss shared templates or other dependencies; pretraining exposure unknown.",
                                   "No policy predictions, final-answer outcomes, effect or statistical power measured."]})
    with (SOURCE / "current_corpus_compatibility_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
