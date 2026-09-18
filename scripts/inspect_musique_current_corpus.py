"""Check public MuSiQue support against the unchanged HotpotQA build.

One streaming read of the existing chunks; no index build, retrieval, model
inference or answer scoring. Rules are defined before the corpus scan. Full
support-passage containment is a conservative textual availability condition,
not a claim about retrieval success or the generator's eventual answer.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from inspect_musique_validation_source import canonical_key, normalize_question, read_queries


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/router/external_sources/musique_v1.0"
BUILD = ROOT / "artifacts/build_65e9037e4c1eed7b"


def whitespace(text):
    return " ".join(text.split())


def main():
    started = time.monotonic()
    with (SOURCE / "musique_ans_v1.0_dev.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    previous = json.loads((SOURCE / "qualification.json").read_text(encoding="utf-8"))
    manifest = json.loads((BUILD / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and len(rows) == previous["questions"]
    nq = read_queries("nq")
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners, direct_flags, required = {}, [], defaultdict(dict)
    for index, row in enumerate(rows):
        supporting = [paragraph for paragraph in row["paragraphs"] if paragraph["is_supporting"]]
        titles = [paragraph["title"] for paragraph in supporting]
        tokens = [("seed", step["id"]) for step in row["question_decomposition"]]
        tokens.extend(("canonical", canonical_key(answer, titles)) for answer in [row["answer"], *row["answer_aliases"]])
        for token in tokens:
            if token in owners:
                parent[find(index)] = find(owners[token])
            else:
                owners[token] = index
        direct_flags.append(any(normalize_question(step["question"]) in nq for step in row["question_decomposition"]))
        for paragraph in supporting:
            text = whitespace(paragraph["paragraph_text"])
            assert text
            required[paragraph["title"]][text] = {"exact": [], "contained": []}
    flagged = {find(index) for index, value in enumerate(direct_flags) if value}
    components = {find(index) for index in range(len(rows))}
    assert len(components) == previous["dependency_components"]
    assert sum(direct_flags) == previous["overlap_query_counts"]["decomposition_question_overlap_nq"]
    assert sum(find(index) in flagged for index in range(len(rows))) == previous["queries_in_flagged_components"]

    result = {
        "status": "prepared_before_corpus_scan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": manifest["build_id"], "build_path": str(BUILD),
        "source_file": str(SOURCE / "musique_ans_v1.0_dev.jsonl"),
        "questions": len(rows), "support_titles_required": len(required),
        "support_title_passage_pairs_required": sum(len(value) for value in required.values()),
        "rules": {
            "title": "exact title match against chunk source; document IDs taken from existing build",
            "exact_passage": "equal full passage after whitespace collapse, with case and punctuation preserved",
            "contained_passage": "whole whitespace-normalized reference passage is a substring of the same-title document body",
            "step_answer_presence": "diagnostic only: lowercased ASCII-punctuation-stripped answer tokens appear contiguously in the same-title document body",
            "history": "retain original 544 dependency components; flag entire component containing a known local-NQ subquestion overlap",
            "conditional_eligibility": "all annotated support passages contained and component unflagged; not automatic confirmation qualification",
        },
        "retrieval_calls": 0, "model_inference_calls": 0, "new_corpus_or_index": False,
    }
    output = SOURCE / "current_corpus_compatibility.json"
    if output.exists():
        raise FileExistsError("Compatibility result already exists; inspect it rather than repeat the corpus scan")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    title_docs, bodies = defaultdict(list), defaultdict(list)
    scanned = 0
    with (BUILD / "chunks.jsonl").open("rb") as stream:
        for scanned, line in enumerate(stream, 1):
            chunk = json.loads(line)
            title = chunk["source"]
            if title in required:
                prefix = title + "\n"
                assert chunk["text"].startswith(prefix)
                body = chunk["text"][len(prefix):]
                title_docs[title].append(chunk["doc_id"])
                bodies[title].append(normalize_question(body))
                normalized = whitespace(body)
                for reference, matches in required[title].items():
                    if reference == normalized:
                        matches["exact"].append(chunk["doc_id"])
                    if reference in normalized:
                        matches["contained"].append(chunk["doc_id"])
            if scanned % 1000000 == 0:
                print(json.dumps({"scanned_chunks": scanned, "matching_titles": len(title_docs)}), flush=True)
    assert scanned == manifest["chunking"]["num_chunks"]
    counts, by_hop, detail = Counter(), defaultdict(Counter), []
    for index, row in enumerate(rows):
        support = {paragraph["idx"]: paragraph for paragraph in row["paragraphs"] if paragraph["is_supporting"]}
        assert support
        titles_found = all(title_docs[paragraph["title"]] for paragraph in support.values())
        exact = all(required[paragraph["title"]][whitespace(paragraph["paragraph_text"])]["exact"] for paragraph in support.values())
        contained = all(required[paragraph["title"]][whitespace(paragraph["paragraph_text"])]["contained"] for paragraph in support.values())
        step_answers = []
        for step in row["question_decomposition"]:
            paragraph = support[step["paragraph_support_idx"]]
            answer = normalize_question(step["answer"])
            step_answers.append(bool(answer) and any(" " + answer + " " in " " + body + " " for body in bodies[paragraph["title"]]))
        history_free = find(index) not in flagged
        flags = {
            "all_support_titles_found": titles_found,
            "all_support_passages_exact": exact,
            "all_support_passages_contained": contained,
            "all_step_answers_present_diagnostic": all(step_answers),
            "unflagged_component": history_free,
            "unflagged_and_all_titles": history_free and titles_found,
            "unflagged_and_all_passages_exact": history_free and exact,
            "unflagged_and_all_passages_contained": history_free and contained,
        }
        counts.update({key: int(value) for key, value in flags.items()})
        by_hop[len(row["question_decomposition"])].update({key: int(value) for key, value in flags.items()})
        detail.append({"query_id": row["id"], "component": find(index), "hops": len(row["question_decomposition"]),
                       **flags, "support_document_ids": {
                           str(position): required[paragraph["title"]][whitespace(paragraph["paragraph_text"])]["contained"]
                           for position, paragraph in support.items()}})
    contained_rows = [row for row in detail if row["unflagged_and_all_passages_contained"]]
    result.update({
        "status": "complete_metadata_only_current_build_compatibility",
        "scanned_chunks": scanned, "support_titles_found": len([key for key, value in title_docs.items() if value]),
        "question_counts": dict(counts), "by_hops": {str(key): dict(value) for key, value in sorted(by_hop.items())},
        "unflagged_contained_components": len({row["component"] for row in contained_rows}),
        "unflagged_contained_component_sizes": dict(sorted(Counter(Counter(row["component"] for row in contained_rows).values()).items())),
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "No semantic sufficiency judgment, independent policy validation, or final-answer gain measured.",
            "Text mismatch does not establish that the question is unanswerable in this corpus.",
            "Any retained questions define a corpus-compatible subset of an external query distribution, not natural HotpotQA confirmation.",
            "Exact local-history checks do not certify absence of pretraining exposure or semantic duplicates.",
        ],
    })
    with (SOURCE / "current_corpus_compatibility_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in detail:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
