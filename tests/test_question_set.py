from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.io_utils import read_jsonl, sha256_file
from src.provenance import json_sha256


QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "questions_v1.jsonl"
HELDOUT_PATH = Path(__file__).resolve().parents[1] / "data" / "questions_heldout_v1.jsonl"
CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus"
CORPUS_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus_manifest.json"


def test_question_set_schema_and_distribution() -> None:
    rows = list(read_jsonl(QUESTIONS_PATH))

    assert len(rows) == 24
    assert len({row["question_id"] for row in rows}) == 24
    assert Counter(row["answerable"] for row in rows) == {True: 22, False: 2}
    assert Counter(row["question_type"] for row in rows) == {
        "single_evidence": 19,
        "multi_evidence": 3,
        "unanswerable": 2,
    }

    for row in rows:
        assert row["question"]
        assert row["gold_answer"]
        if row["answerable"]:
            assert row["evidence"]
            assert row["expected_sources"]
        else:
            assert row["evidence"] == []
            assert row["expected_sources"] == []

        for item in row["evidence"]:
            alternatives = item.get("alternatives", [item])
            assert alternatives
            for alternative in alternatives:
                assert alternative["source"].endswith(".pdf")
                assert 1 <= alternative["page_start"] <= alternative["page_end"]

        if row["question_type"] == "multi_evidence":
            assert row["evidence_mode"] == "all"
            assert len(row["evidence"]) >= 2
            assert all(item.get("required") is True for item in row["evidence"])

    covered_sources = {source for row in rows for source in row["expected_sources"]}
    assert covered_sources == {
        "01_AquaPipe A Quality-Aware Pipeline for Knowledge Retrieval.pdf",
        "02_SAGE_A_Framework_of_Precise_Retrieval_for_RAG.pdf",
        "03_Gao_RAG_Survey.pdf",
        "04_Lewis_2020_RAG.pdf",
        "05_Karpukhin_2020_DPR.pdf",
    }


def test_heldout_set_is_small_unique_and_covers_each_paper() -> None:
    rows = list(read_jsonl(HELDOUT_PATH))
    assert len(rows) == 5
    assert len({row["question_id"] for row in rows}) == 5
    assert all(row["answerable"] is True and row["evidence"] for row in rows)
    assert len({row["expected_sources"][0] for row in rows}) == 5


def test_versioned_corpus_matches_public_manifest() -> None:
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = [
        {
            "source": path.name,
            "relative_path": path.relative_to(CORPUS_PATH).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(CORPUS_PATH.glob("*.pdf"))
    ]

    assert manifest["schema_version"] == 1
    assert manifest["documents"] == documents
    assert manifest["aggregate_sha256"] == json_sha256(documents)
