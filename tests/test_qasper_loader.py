from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts.run_qasper_eval import qasper_eval_inputs
from src.config import load_config
from src.loaders.qasper_loader import (
    QASPER_EVALUATION_SLICE,
    QasperCorpusLoader,
    is_qasper_evaluation_reference,
    qasper_evaluation_questions,
    qasper_evaluation_slice_stats,
    qasper_evidence_from_hits,
    qasper_questions,
)


ROOT = Path(__file__).resolve().parents[1]


def _article(paper_id: str = "paper-1") -> dict:
    return {
        "id": paper_id,
        "title": "A paper",
        "abstract": "Not added to the full-text context.",
        "full_text": {
            "section_name": ["Introduction", None],
            "paragraphs": [["First\nparagraph."], ["Second paragraph."]],
        },
        "figures_and_tables": {"caption": ["Table 1: Results."], "file": ["table.png"]},
        "qas": {
            "question_id": ["question-1"],
            "question": ["What was found?"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["a result"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["First paragraph."],
                            "highlighted_evidence": [],
                        },
                        {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": True,
                            "free_form_answer": "",
                            "evidence": ["Second paragraph."],
                            "highlighted_evidence": [],
                        },
                    ],
                    "annotation_id": ["annotation-1", "annotation-2"],
                    "worker_id": ["worker-1", "worker-2"],
                }
            ],
            "nlp_background": ["background"],
        },
    }


def test_qasper_loader_reads_local_split_and_maps_stable_units(tmp_path, monkeypatch) -> None:
    root = tmp_path / "hf_dataset"
    (root / "validation").mkdir(parents=True)
    (root / "validation" / "data.arrow").write_bytes(b"arrow")
    (root / "dataset_dict.json").write_text("{}", encoding="utf-8")
    dataset = {"validation": [_article(), _article("paper-2")]}
    calls = []

    fake_datasets = types.ModuleType("datasets")

    def load_from_disk(path: str):
        calls.append(Path(path))
        return dataset

    fake_datasets.load_from_disk = load_from_disk
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loader = QasperCorpusLoader(split="validation", max_documents=1)
    pages = loader.load(root)

    assert calls == [root.resolve()]
    assert [(page.doc_id, page.source, page.page, page.text) for page in pages] == [
        ("paper-1", "paper-1", 1, "A paper"),
        ("paper-1", "paper-1", 2, "Not added to the full-text context."),
        ("paper-1", "paper-1", 3, "Introduction"),
        ("paper-1", "paper-1", 4, "First paragraph."),
        ("paper-1", "paper-1", 5, "Second paragraph."),
        ("paper-1", "paper-1", 6, "Table 1: Results."),
    ]
    assert [path.relative_to(root).as_posix() for path in loader.discover(root)] == [
        "dataset_dict.json",
        "validation/data.arrow",
    ]


def test_qasper_questions_keep_references_but_drop_annotation_metadata() -> None:
    questions = qasper_questions(_article())

    assert len(questions) == 1
    assert set(questions[0]) == {"question_id", "question", "paper_id", "references"}
    assert len(questions[0]["references"]) == 2
    assert set(questions[0]["references"][0]) == {
        "unanswerable",
        "extractive_spans",
        "yes_no",
        "free_form_answer",
        "evidence",
    }
    assert "highlighted_evidence" not in questions[0]["references"][0]
    assert "worker_id" not in questions[0]
    assert "nlp_background" not in questions[0]


def test_qasper_evaluation_questions_keep_only_eligible_references() -> None:
    questions = qasper_evaluation_questions(_article())

    assert len(questions) == 1
    assert questions[0]["question_slice"] == QASPER_EVALUATION_SLICE
    assert questions[0]["question_type"] == "extractive"
    assert questions[0]["source_reference_count"] == 2
    assert len(questions[0]["references"]) == 1
    assert questions[0]["references"][0]["extractive_spans"] == ["a result"]


def test_qasper_evaluation_reference_requires_all_slice_conditions() -> None:
    eligible = {
        "unanswerable": False,
        "extractive_spans": ["answer"],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": ["Text paragraph."],
    }
    ineligible = [
        {**eligible, "unanswerable": True},
        {**eligible, "extractive_spans": [], "free_form_answer": "summary"},
        {**eligible, "extractive_spans": [], "yes_no": True},
        {**eligible, "evidence": []},
        {**eligible, "evidence": ["First paragraph.", "Second paragraph."]},
        {**eligible, "evidence": ["FLOAT SELECTED: Table 1"]},
    ]

    assert is_qasper_evaluation_reference(eligible) is True
    assert all(is_qasper_evaluation_reference(reference) is False for reference in ineligible)


def test_qasper_evaluation_slice_stats_count_questions_and_references() -> None:
    stats = qasper_evaluation_slice_stats([_article(), _article("paper-2")])

    assert stats == {
        "name": QASPER_EVALUATION_SLICE,
        "candidate_questions": 2,
        "selected_questions": 2,
        "excluded_questions": 0,
        "candidate_references": 4,
        "selected_references": 2,
        "excluded_references": 2,
    }


def test_qasper_hit_ranges_map_back_to_deduplicated_evidence() -> None:
    evidence = qasper_evidence_from_hits(
        _article(),
        [
            {"doc_id": "paper-1", "page_start": 4, "page_end": 5},
            {"doc_id": "paper-1", "page_start": 5, "page_end": 5},
            {"doc_id": "another-paper", "page_start": 1, "page_end": 6},
        ],
    )

    assert evidence == ["First paragraph.", "Second paragraph."]


def test_qasper_smoke_config_is_single_paper_and_offline() -> None:
    config = load_config(ROOT / "configs" / "qasper_smoke.yaml")

    assert config["loader"] == {
        "type": "qasper",
        "split": "validation",
        "max_documents": 1,
    }
    assert config["embedding"]["backend"] == "hashing"
    assert "local_files_only" not in config["embedding"]
    assert "local_files_only" not in config["chunking"]
    assert config["index"]["backend"] == "numpy"
    assert config["generation"]["provider"] == "extractive"


def test_qasper_all_split_loader_uses_fixed_order_and_global_limit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "hf_dataset"
    for split in ("train", "validation", "test"):
        (root / split).mkdir(parents=True)
        (root / split / "data.arrow").write_bytes(split.encode("utf-8"))
    dataset = {
        "train": [_article("train-paper")],
        "validation": [_article("validation-paper")],
        "test": [_article("test-paper")],
    }
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = lambda _: dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loader = QasperCorpusLoader(split="all", max_documents=2)

    assert [article["id"] for article in loader.articles(root)] == ["train-paper", "validation-paper"]
    assert [path.relative_to(root).as_posix() for path in loader.discover(root)] == [
        "test/data.arrow",
        "train/data.arrow",
        "validation/data.arrow",
    ]


def test_qasper_eval_inputs_keep_validation_questions_but_all_papers() -> None:
    dataset = {
        "train": [_article("train-paper")],
        "validation": [_article("validation-paper")],
        "test": [_article("test-paper")],
    }

    questions, papers = qasper_eval_inputs(dataset)

    assert len(questions) == 1
    assert questions[0]["paper_id"] == "validation-paper"
    assert questions[0]["expected_sources"] == ["validation-paper"]
    assert questions[0]["question_slice"] == QASPER_EVALUATION_SLICE
    assert len(questions[0]["references"]) == 1
    assert is_qasper_evaluation_reference(questions[0]["references"][0])
    assert set(papers) == {"train-paper", "validation-paper", "test-paper"}


def test_qasper_eval_inputs_exclude_questions_without_an_eligible_reference() -> None:
    validation = _article("validation-paper")
    validation["qas"]["answers"][0]["answer"] = [
        {
            "unanswerable": False,
            "extractive_spans": [],
            "yes_no": True,
            "free_form_answer": "",
            "evidence": ["First paragraph."],
        }
    ]
    dataset = {
        "train": [_article("train-paper")],
        "validation": [validation],
        "test": [_article("test-paper")],
    }

    questions, papers = qasper_eval_inputs(dataset)

    assert questions == []
    assert set(papers) == {"train-paper", "validation-paper", "test-paper"}
