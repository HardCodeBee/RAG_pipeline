"""Load the locally saved Hugging Face QASPER dataset into pipeline records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.records import PageRecord


QASPER_SPLITS = ("train", "validation", "test")
QASPER_EVALUATION_SLICE = "answerable_text_only_extractive_single_evidence_v1"
QASPER_FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED:"
_REFERENCE_FIELDS = (
    "unanswerable",
    "extractive_spans",
    "yes_no",
    "free_form_answer",
    "evidence",
)


@dataclass(frozen=True, slots=True)
class QasperUnit:
    """One indexable paper unit and its deterministic evidence representation."""

    text: str
    evidence: str


def _unit_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r", " ").replace("\n", " ").strip()


def qasper_unit_records(article: Mapping[str, Any]) -> list[QasperUnit]:
    """Return clean index text without reading any question or answer fields."""

    full_text = article.get("full_text") or {}
    units: list[QasperUnit] = []

    for field in ("title", "abstract"):
        text = _unit_text(article.get(field))
        if text:
            units.append(QasperUnit(text=text, evidence=text))

    if isinstance(full_text, Mapping):
        section_names = full_text.get("section_name") or []
        section_paragraphs = full_text.get("paragraphs") or []
        sections: Iterable[tuple[Any, Any]] = zip(section_names, section_paragraphs)
    else:
        sections = (
            (section.get("section_name"), section.get("paragraphs") or [])
            for section in full_text
            if isinstance(section, Mapping)
        )

    for section_name, paragraphs in sections:
        heading = _unit_text(section_name)
        if heading:
            units.append(QasperUnit(text=heading, evidence=heading))
        for paragraph in paragraphs:
            text = _unit_text(paragraph)
            if text:
                units.append(QasperUnit(text=text, evidence=text))

    figures_and_tables = article.get("figures_and_tables") or {}
    captions = figures_and_tables.get("caption") or [] if isinstance(figures_and_tables, Mapping) else []
    for caption in captions:
        text = _unit_text(caption)
        if text:
            # The annotation prefix is a label convention, not paper content.
            units.append(QasperUnit(text=text, evidence=f"{QASPER_FLOAT_EVIDENCE_PREFIX} {text}"))
    return units


def qasper_pages(article: Mapping[str, Any]) -> list[PageRecord]:
    paper_id = _unit_text(article.get("id"))
    if not paper_id:
        raise ValueError("QASPER paper is missing a non-empty id")
    return [
        PageRecord(doc_id=paper_id, source=paper_id, page=position, text=unit.text)
        for position, unit in enumerate(qasper_unit_records(article), start=1)
    ]


def _answer_references(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and "answer" in value:
        value = value["answer"]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("QASPER answers must contain a list of references")
    references = [
        {field: item.get(field) for field in _REFERENCE_FIELDS}
        for item in value
        if isinstance(item, Mapping)
    ]
    if not references:
        raise ValueError("QASPER question has no answer references")
    return references


def qasper_questions(article: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep only fields required for running and officially scoring QASPER."""

    paper_id = _unit_text(article.get("id"))
    qas = article.get("qas") or {}
    questions: list[dict[str, Any]] = []

    if isinstance(qas, Mapping):
        question_texts = qas.get("question") or []
        question_ids = qas.get("question_id") or []
        answer_groups = qas.get("answers") or []
        if not (len(question_texts) == len(question_ids) == len(answer_groups)):
            raise ValueError("QASPER question columns have inconsistent lengths")
        rows = zip(question_ids, question_texts, answer_groups)
    else:
        rows = (
            (item.get("question_id"), item.get("question"), item.get("answers"))
            for item in qas
            if isinstance(item, Mapping)
        )

    for question_id, question, answers in rows:
        normalized_id = _unit_text(question_id)
        normalized_question = _unit_text(question)
        if not normalized_id or not normalized_question:
            raise ValueError("QASPER question id and text must be non-empty")
        questions.append(
            {
                "question_id": normalized_id,
                "question": normalized_question,
                "paper_id": paper_id,
                "references": _answer_references(answers),
            }
        )
    return questions


def is_qasper_evaluation_reference(reference: Mapping[str, Any]) -> bool:
    """Return whether one annotation belongs to the retrieval-focused slice."""

    extractive_spans = [
        _unit_text(value)
        for value in (reference.get("extractive_spans") or [])
        if _unit_text(value)
    ]
    evidence = [
        _unit_text(value)
        for value in (reference.get("evidence") or [])
        if _unit_text(value)
    ]
    return (
        reference.get("unanswerable") is False
        and bool(extractive_spans)
        and reference.get("yes_no") is None
        and not _unit_text(reference.get("free_form_answer"))
        and len(evidence) == 1
        and not evidence[0].startswith(QASPER_FLOAT_EVIDENCE_PREFIX)
    )


def qasper_evaluation_questions(article: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select answerable, text-only, extractive questions with one evidence unit.

    QASPER may provide multiple independent annotations for one question. A
    question is selected when at least one annotation qualifies, and only its
    qualifying annotations are retained as scoring references.
    """

    selected = []
    for question in qasper_questions(article):
        references = [
            reference
            for reference in question["references"]
            if is_qasper_evaluation_reference(reference)
        ]
        if not references:
            continue
        selected.append(
            {
                **question,
                "references": references,
                "question_type": "extractive",
                "question_slice": QASPER_EVALUATION_SLICE,
                "source_reference_count": len(question["references"]),
            }
        )
    return selected


def qasper_evaluation_slice_stats(
    articles: Sequence[Mapping[str, Any]],
) -> dict[str, int | str]:
    """Count the effect of the fixed evaluation slice before running models."""

    candidate_questions = 0
    selected_questions = 0
    candidate_references = 0
    selected_references = 0
    for article in articles:
        questions = qasper_questions(article)
        selected = qasper_evaluation_questions(article)
        candidate_questions += len(questions)
        selected_questions += len(selected)
        candidate_references += sum(len(question["references"]) for question in questions)
        selected_references += sum(len(question["references"]) for question in selected)
    return {
        "name": QASPER_EVALUATION_SLICE,
        "candidate_questions": candidate_questions,
        "selected_questions": selected_questions,
        "excluded_questions": candidate_questions - selected_questions,
        "candidate_references": candidate_references,
        "selected_references": selected_references,
        "excluded_references": candidate_references - selected_references,
    }


def qasper_evidence_from_hits(
    article: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Map retrieved synthetic page ranges back to original QASPER evidence units."""

    paper_id = _unit_text(article.get("id"))
    units = qasper_unit_records(article)
    selected: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.get("doc_id") != paper_id:
            continue
        page_start = int(hit["page_start"])
        page_end = int(hit["page_end"])
        for page in range(page_start, page_end + 1):
            if not 1 <= page <= len(units):
                continue
            evidence = units[page - 1].evidence
            if evidence not in seen:
                selected.append(evidence)
                seen.add(evidence)
    return selected


def load_qasper_dataset(dataset_path: str | Path):
    """Load the existing save_to_disk directory without network access."""

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("QASPER loading requires requirements/experiment.txt") from error

    root = Path(dataset_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"QASPER dataset directory does not exist: {root}")
    return load_from_disk(str(root))


def selected_qasper_articles(dataset_path: str | Path, split: str, max_documents: int | None = None) -> list[dict]:
    dataset = load_qasper_dataset(dataset_path)
    selected_splits = QASPER_SPLITS if split == "all" else (split,)
    missing = [name for name in selected_splits if name not in dataset]
    if missing:
        raise ValueError(f"QASPER split is not available: {', '.join(missing)}")

    articles: list[dict] = []
    for name in selected_splits:
        split_dataset = dataset[name]
        for position in range(len(split_dataset)):
            articles.append(split_dataset[position])
            if max_documents is not None and len(articles) >= max_documents:
                return articles
    return articles


class QasperCorpusLoader:
    """Adapt QASPER papers to existing PageRecord-based indexing."""

    def __init__(self, split: str = "validation", max_documents: int | None = None):
        if split not in {*QASPER_SPLITS, "all"}:
            raise ValueError("QASPER split must be one of: train, validation, test, all")
        if max_documents is not None and (
            isinstance(max_documents, bool) or not isinstance(max_documents, int) or max_documents <= 0
        ):
            raise ValueError("max_documents must be a positive integer or None")
        self.split = split
        self.max_documents = max_documents

    def discover(self, corpus_path: str | Path) -> list[Path]:
        root = Path(corpus_path)
        if not root.is_dir():
            raise FileNotFoundError(f"QASPER dataset directory does not exist: {root}")
        selected_splits = QASPER_SPLITS if self.split == "all" else (self.split,)
        files = []
        for split in selected_splits:
            split_root = root / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"QASPER split directory does not exist: {split_root}")
            files.extend(path for path in split_root.rglob("*") if path.is_file())
        dataset_dict = root / "dataset_dict.json"
        if dataset_dict.is_file():
            files.append(dataset_dict)
        return sorted(files, key=lambda path: path.relative_to(root).as_posix())

    def articles(self, corpus_path: str | Path) -> list[dict]:
        articles = selected_qasper_articles(corpus_path, self.split, self.max_documents)
        paper_ids = [_unit_text(article.get("id")) for article in articles]
        if any(not paper_id for paper_id in paper_ids):
            raise ValueError("QASPER papers must have non-empty ids")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("QASPER paper ids must be unique across selected splits")
        return articles

    def load(self, corpus_path: str | Path) -> list[PageRecord]:
        pages = []
        for article in self.articles(corpus_path):
            pages.extend(qasper_pages(article))
        return pages
