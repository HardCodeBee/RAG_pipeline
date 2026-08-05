"""Prepare the fixed Natural Questions Open + DPR Wikipedia subset.

The canonical protocol selects 2,000 DPR NQ development questions, preserves
their annotated positive Wikipedia passages and pages, adds official hard
negatives, and fills the remaining corpus slots with a deterministic reservoir
sample.  Labels are written only to the question files and never into corpus
rows.
"""

from __future__ import annotations

# Dataset-specific preparation implementation.

import csv
import gzip
import hashlib
import heapq
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO


from src.persistence.artifact_io import describe_artifact, write_jsonl, write_manifest
from src.persistence.artifact_validation import validate_prepared_dataset
from src.provenance import canonical_json_bytes, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PROTOCOL = "nq_open_dpr_wiki_1m_gold_preserving_v1"
CORPUS_FORMAT = "dpr_psgs_w100_jsonl_v1"
TEXT_FORMAT = "title_newline_text_v1"
QUESTION_NORMALIZATION = "unicode_nfkc_whitespace_v1"
CONTEXT_MATCH_NORMALIZATION = "unicode_nfkc_whitespace_v1"
QUESTION_SELECTION = "sha256_seed_nul_normalized_question_v1"
BACKGROUND_SAMPLER = "algorithm_r_splitmix64_rejection_v1"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / PROTOCOL
DEFAULT_TARGET_PASSAGES = 1_000_000
DEFAULT_CALIBRATION_QUESTIONS = 500
DEFAULT_EVALUATION_QUESTIONS = 1_500
DEFAULT_HARD_NEGATIVES_PER_QUESTION = 50
DEFAULT_SEED = PROTOCOL

_UINT64_MASK = (1 << 64) - 1
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class _ContextRef:
    passage_id: str | None
    title: str
    text: str


@dataclass
class _QuestionCandidate:
    question_id: str
    question: str
    normalized_question: str
    answers: list[str]
    positives: list[_ContextRef]
    hard_negatives: list[_ContextRef]
    selection_hash: str
    source_position: int


class _SplitMix64:
    """Small versioned PRNG used only by the specified reservoir sampler."""

    def __init__(self, seed: int):
        self.state = seed & _UINT64_MASK

    def next_uint64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return (value ^ (value >> 31)) & _UINT64_MASK

    def randbelow(self, upper: int) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise ValueError("upper must be a positive integer")
        modulus = 1 << 64
        limit = modulus - (modulus % upper)
        while True:
            value = self.next_uint64()
            if value < limit:
                return value % upper


def _ids_sha256(values: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(values, key=_id_sort_key)
    for value in ordered:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _question_ids_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _id_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (int(text), text)


def _normalize_question(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _normalize_context(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def _selection_hash(seed: str, normalized_question: str) -> str:
    return hashlib.sha256(f"{seed}\0{normalized_question}".encode("utf-8")).hexdigest()


def _question_id(normalized_question: str) -> str:
    digest = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
    return f"nq_{digest}"


def _normalize_passage_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("DPR passage ids must be positive integers")
    text = str(value).strip()
    if not text or not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"Invalid DPR passage id: {value!r}")
    return str(int(text))


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raw = []
    answers: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        answer = item.strip()
        if answer and answer not in seen:
            answers.append(answer)
            seen.add(answer)
    if not answers:
        raise ValueError("A selected NQ question must have at least one non-empty answer")
    return answers


def _context_ref(value: Any, location: str) -> _ContextRef:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    title = value.get("title")
    text = value.get("text")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{location}.title must be non-empty")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{location}.text must be non-empty")
    passage_id = _normalize_passage_id(value.get("passage_id", value.get("id")))
    return _ContextRef(passage_id=passage_id, title=title, text=text)


def _candidate_from_record(
    value: Mapping[str, Any],
    *,
    normalized_question: str,
    selection_hash: str,
    source_position: int,
    hard_negatives_per_question: int,
) -> _QuestionCandidate:
    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"NQ question at source position {source_position} is empty")
    positives_value = value.get("positive_ctxs")
    if not isinstance(positives_value, Sequence) or isinstance(positives_value, (str, bytes)):
        raise TypeError("positive_ctxs must be a sequence")
    positives = [
        _context_ref(item, f"question[{source_position}].positive_ctxs[{position}]")
        for position, item in enumerate(positives_value)
    ]
    if not positives:
        raise ValueError("A selected NQ question must have at least one positive context")

    hard_value = value.get("hard_negative_ctxs") or []
    if not isinstance(hard_value, Sequence) or isinstance(hard_value, (str, bytes)):
        raise TypeError("hard_negative_ctxs must be a sequence")
    hard_negatives = [
        _context_ref(item, f"question[{source_position}].hard_negative_ctxs[{position}]")
        for position, item in enumerate(hard_value[:hard_negatives_per_question])
    ]
    return _QuestionCandidate(
        question_id=_question_id(normalized_question),
        question=question.strip(),
        normalized_question=normalized_question,
        answers=_answers(value.get("answers")),
        positives=positives,
        hard_negatives=hard_negatives,
        selection_hash=selection_hash,
        source_position=source_position,
    )


@contextmanager
def _open_text(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    if path.suffix.casefold() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8-sig", newline=newline)
    else:
        handle = path.open("r", encoding="utf-8-sig", newline=newline)
    try:
        yield handle
    finally:
        handle.close()


def _logical_suffix(path: Path) -> str:
    candidate = path.with_suffix("") if path.suffix.casefold() == ".gz" else path
    return candidate.suffix.casefold()


def _iter_json_array(handle: TextIO) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    first_value = True
    eof = False

    def fill() -> bool:
        nonlocal buffer, position, eof
        if position:
            buffer = buffer[position:]
            position = 0
        block = handle.read(1024 * 1024)
        if not block:
            eof = True
            return False
        buffer += block
        return True

    while True:
        while position >= len(buffer) and not eof:
            fill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if not started:
            if position >= len(buffer) and not eof:
                continue
            if position >= len(buffer) or buffer[position] != "[":
                raise ValueError("Expected a top-level JSON array")
            position += 1
            started = True
            continue

        while position >= len(buffer) and not eof:
            fill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer) and buffer[position] == "]":
            return
        if not first_value:
            if position >= len(buffer) and not eof:
                continue
            if position >= len(buffer) or buffer[position] != ",":
                raise ValueError("Expected a comma between JSON array values")
            position += 1
            while True:
                while position >= len(buffer) and not eof:
                    fill()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break

        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                position = end
                first_value = False
                yield value
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError("Invalid or truncated JSON array") from exc
                fill()


def _iter_question_records(path: Path) -> Iterator[Mapping[str, Any]]:
    suffix = _logical_suffix(path)
    if suffix in {".jsonl", ".ndjson"}:
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"Question JSONL row {line_number} must be an object")
                yield value
        return

    with _open_text(path) as handle:
        while True:
            character = handle.read(1)
            if not character:
                raise ValueError("Question file is empty")
            if not character.isspace():
                break
        handle.seek(0)
        if character == "[":
            for value in _iter_json_array(handle):
                if not isinstance(value, Mapping):
                    raise TypeError("Every DPR NQ array item must be an object")
                yield value
            return
        value = json.load(handle)
        rows = value.get("data") if isinstance(value, Mapping) else value
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("DPR NQ JSON must be an array or an object containing a data array")
        for item in rows:
            if not isinstance(item, Mapping):
                raise TypeError("Every DPR NQ item must be an object")
            yield item


def _select_questions(
    questions_path: Path,
    *,
    total_questions: int,
    hard_negatives_per_question: int,
    seed: str,
) -> tuple[list[_QuestionCandidate], int]:
    heap: list[tuple[int, int, _QuestionCandidate]] = []
    seen_question_ids: dict[str, str] = {}
    source_count = 0

    for source_position, record in enumerate(_iter_question_records(questions_path), start=1):
        source_count += 1
        normalized = _normalize_question(record.get("question"))
        if not normalized:
            raise ValueError(f"NQ question at source position {source_position} is empty")
        question_id = _question_id(normalized)
        previous = seen_question_ids.get(question_id)
        if previous is not None:
            if previous == normalized:
                raise ValueError(f"Duplicate normalized NQ question: {normalized}")
            raise RuntimeError("SHA-256 collision while generating NQ question ids")
        seen_question_ids[question_id] = normalized

        digest = _selection_hash(seed, normalized)
        digest_int = int(digest, 16)
        sort_key = (digest_int, source_position)
        if len(heap) >= total_questions:
            largest_key = (-heap[0][0], -heap[0][1])
            if sort_key >= largest_key:
                continue
        candidate = _candidate_from_record(
            record,
            normalized_question=normalized,
            selection_hash=digest,
            source_position=source_position,
            hard_negatives_per_question=hard_negatives_per_question,
        )
        entry = (-digest_int, -source_position, candidate)
        if len(heap) < total_questions:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)

    if len(heap) != total_questions:
        raise ValueError(
            f"Question source contains only {source_count} rows; "
            f"{total_questions} selected questions are required"
        )
    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda item: (item.selection_hash, item.source_position))
    return selected, source_count


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _iter_wikipedia_rows(path: Path) -> Iterator[tuple[str, str, str]]:
    _set_csv_field_size_limit()
    with _open_text(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != {"id", "text", "title"}:
            raise ValueError("DPR Wikipedia TSV header must contain exactly: id, text, title")
        previous_id = 0
        for row_number, row in enumerate(reader, start=2):
            passage_id = _normalize_passage_id(row.get("id"))
            assert passage_id is not None
            numeric_id = int(passage_id)
            if numeric_id <= previous_id:
                raise ValueError("DPR Wikipedia passage ids must be strictly increasing")
            previous_id = numeric_id
            title = row.get("title")
            text = row.get("text")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Wikipedia row {row_number} has an empty title")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Wikipedia row {row_number} has empty text")
            yield passage_id, title, text


def _resolve_contexts_and_titles(
    wikipedia_path: Path,
    questions: Sequence[_QuestionCandidate],
) -> tuple[set[str], set[str], int]:
    explicit: dict[str, list[_ContextRef]] = {}
    exact: dict[tuple[str, str], list[_ContextRef]] = {}
    positive_titles: set[str] = set()
    for question in questions:
        for reference in question.positives:
            normalized_title = _normalize_context(reference.title)
            positive_titles.add(normalized_title)
            if reference.passage_id is None:
                exact.setdefault(
                    (normalized_title, _normalize_context(reference.text)),
                    [],
                ).append(reference)
            else:
                explicit.setdefault(reference.passage_id, []).append(reference)
        for reference in question.hard_negatives:
            if reference.passage_id is None:
                exact.setdefault(
                    (
                        _normalize_context(reference.title),
                        _normalize_context(reference.text),
                    ),
                    [],
                ).append(reference)
            else:
                explicit.setdefault(reference.passage_id, []).append(reference)

    exact_matches: dict[tuple[str, str], list[str]] = {key: [] for key in exact}
    seen_explicit: set[str] = set()
    title_expansion_ids: set[str] = set()
    source_rows = 0
    for passage_id, title, text in _iter_wikipedia_rows(wikipedia_path):
        source_rows += 1
        normalized_title = _normalize_context(title)
        normalized_text = _normalize_context(text)
        if normalized_title in positive_titles:
            title_expansion_ids.add(passage_id)
        references = explicit.get(passage_id)
        if references:
            for reference in references:
                if (
                    _normalize_context(reference.title) != normalized_title
                    or _normalize_context(reference.text) != normalized_text
                ):
                    raise ValueError(
                        f"DPR context {passage_id} does not match normalized Wikipedia title/text"
                    )
            seen_explicit.add(passage_id)
        key = (normalized_title, normalized_text)
        if key in exact_matches and len(exact_matches[key]) < 2:
            exact_matches[key].append(passage_id)

    missing_explicit = sorted(set(explicit) - seen_explicit, key=_id_sort_key)
    if missing_explicit:
        raise ValueError(f"DPR context passage ids are absent from Wikipedia: {missing_explicit[:5]}")
    for key, references in exact.items():
        matches = exact_matches[key]
        if len(matches) != 1:
            raise ValueError(
                "A DPR context without passage_id must match exactly one Wikipedia row; "
                f"found {len(matches)} for title={key[0]!r}"
            )
        for reference in references:
            reference.passage_id = matches[0]

    positive_ids = {
        reference.passage_id
        for question in questions
        for reference in question.positives
        if reference.passage_id is not None
    }
    if len(positive_ids) == 0:
        raise RuntimeError("No annotated positive passages were resolved")
    title_expansion_ids.update(positive_ids)
    return title_expansion_ids, set(positive_ids), source_rows


def _select_hard_negative_ids(
    questions: Sequence[_QuestionCandidate],
    *,
    excluded_ids: set[str],
    maximum_count: int,
    per_question: int,
) -> set[str]:
    selected: set[str] = set()
    for round_index in range(per_question):
        for question in questions:
            if len(selected) >= maximum_count:
                return selected
            if round_index >= len(question.hard_negatives):
                continue
            passage_id = question.hard_negatives[round_index].passage_id
            if passage_id is None:
                raise RuntimeError("Hard negative passage id was not resolved")
            if passage_id not in excluded_ids:
                selected.add(passage_id)
    return selected


def _reservoir_background_ids(
    wikipedia_path: Path,
    *,
    excluded_ids: set[str],
    target_count: int,
    seed: str,
    expected_source_rows: int,
) -> set[str]:
    if target_count == 0:
        return set()
    seed_bytes = hashlib.sha256(f"{seed}\0background".encode("utf-8")).digest()[:8]
    generator = _SplitMix64(int.from_bytes(seed_bytes, "big"))
    reservoir: list[str] = []
    eligible_seen = 0
    source_rows = 0
    for passage_id, _, _ in _iter_wikipedia_rows(wikipedia_path):
        source_rows += 1
        if passage_id in excluded_ids:
            continue
        eligible_seen += 1
        if len(reservoir) < target_count:
            reservoir.append(passage_id)
            continue
        position = generator.randbelow(eligible_seen)
        if position < target_count:
            reservoir[position] = passage_id
    if source_rows != expected_source_rows:
        raise RuntimeError("Wikipedia row count changed between selection scans")
    if len(reservoir) != target_count:
        raise ValueError(
            f"Only {eligible_seen} background passages are available; {target_count} are required"
        )
    return set(reservoir)


def _write_selected_corpus(
    wikipedia_path: Path,
    *,
    selected_ids: set[str],
    corpus_dir: Path,
    expected_source_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    passage_ids_path = corpus_dir / "passage_ids.txt"
    passages_path = corpus_dir / "passages.jsonl"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    with passage_ids_path.open("w", encoding="utf-8", newline="\n") as handle:
        for passage_id in sorted(selected_ids, key=_id_sort_key):
            handle.write(passage_id)
            handle.write("\n")

    written = 0
    source_rows = 0
    with passages_path.open("w", encoding="utf-8", newline="\n") as handle:
        for passage_id, title, text in _iter_wikipedia_rows(wikipedia_path):
            source_rows += 1
            if passage_id not in selected_ids:
                continue
            row = {"id": passage_id, "title": title, "text": text}
            handle.write(canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")
            written += 1
    if source_rows != expected_source_rows:
        raise RuntimeError("Wikipedia row count changed while writing the selected corpus")
    if written != len(selected_ids):
        raise RuntimeError("Selected passage ids and written corpus rows differ")
    return (
        describe_artifact(passages_path, relative_to=corpus_dir, rows=written),
        describe_artifact(
            passage_ids_path,
            relative_to=corpus_dir,
            rows=len(selected_ids),
        ),
    )


def _question_row(question: _QuestionCandidate) -> dict[str, Any]:
    positive_ids: list[str] = []
    seen: set[str] = set()
    for reference in question.positives:
        if reference.passage_id is None:
            raise RuntimeError("Positive passage id was not resolved")
        chunk_id = f"dpr_wiki_passage:{reference.passage_id}"
        if chunk_id not in seen:
            positive_ids.append(chunk_id)
            seen.add(chunk_id)
    return {
        "question_id": question.question_id,
        "question": question.question,
        "answers": question.answers,
        "answerable": True,
        "question_type": "nq_open",
        "evidence": [{"alternatives": positive_ids}],
    }


def _write_questions(
    questions: Sequence[_QuestionCandidate],
    *,
    calibration_count: int,
    questions_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = questions_dir / "calibration.jsonl"
    evaluation_path = questions_dir / "evaluation.jsonl"
    calibration_rows = [_question_row(item) for item in questions[:calibration_count]]
    evaluation_rows = [_question_row(item) for item in questions[calibration_count:]]
    write_jsonl(calibration_path, calibration_rows)
    write_jsonl(evaluation_path, evaluation_rows)
    return (
        describe_artifact(
            calibration_path,
            relative_to=questions_dir,
            rows=len(calibration_rows),
        ),
        describe_artifact(
            evaluation_path,
            relative_to=questions_dir,
            rows=len(evaluation_rows),
        ),
    )


def prepare_nq_dpr_wiki(
    wikipedia_path: str | Path,
    questions_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    target_passages: int = DEFAULT_TARGET_PASSAGES,
    calibration_questions: int = DEFAULT_CALIBRATION_QUESTIONS,
    evaluation_questions: int = DEFAULT_EVALUATION_QUESTIONS,
    hard_negatives_per_question: int = DEFAULT_HARD_NEGATIVES_PER_QUESTION,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Prepare or validate one immutable local copy of the fixed protocol."""

    wikipedia = Path(wikipedia_path).resolve()
    questions_source = Path(questions_path).resolve()
    output = Path(output_dir).resolve()
    for path, name in ((wikipedia, "Wikipedia"), (questions_source, "questions")):
        if not path.is_file():
            raise FileNotFoundError(f"{name} source file does not exist: {path}")
    integer_values = {
        "target_passages": target_passages,
        "calibration_questions": calibration_questions,
        "evaluation_questions": evaluation_questions,
        "hard_negatives_per_question": hard_negatives_per_question,
    }
    for name, value in integer_values.items():
        minimum = 0 if name == "hard_negatives_per_question" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")

    wikipedia_sha = sha256_file(wikipedia)
    questions_sha = sha256_file(questions_source)
    request = {
        "wikipedia_sha256": wikipedia_sha,
        "questions_sha256": questions_sha,
        "target_passages": target_passages,
        "calibration_questions": calibration_questions,
        "evaluation_questions": evaluation_questions,
        "hard_negatives_per_question": hard_negatives_per_question,
        "seed": seed,
    }
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"Dataset output exists but is not a directory: {output}")
        return validate_prepared_dataset(
            output,
            expected_protocol=PROTOCOL,
            expected_request=request,
        )

    total_questions = calibration_questions + evaluation_questions
    selected_questions, source_question_count = _select_questions(
        questions_source,
        total_questions=total_questions,
        hard_negatives_per_question=hard_negatives_per_question,
        seed=seed,
    )
    mandatory_ids, positive_ids, source_passage_count = _resolve_contexts_and_titles(
        wikipedia,
        selected_questions,
    )
    if len(mandatory_ids) > target_passages:
        raise ValueError(
            f"Positive title expansion requires {len(mandatory_ids)} passages, "
            f"exceeding target {target_passages}"
        )
    hard_ids = _select_hard_negative_ids(
        selected_questions,
        excluded_ids=mandatory_ids,
        maximum_count=target_passages - len(mandatory_ids),
        per_question=hard_negatives_per_question,
    )
    excluded = mandatory_ids | hard_ids
    background_target = target_passages - len(excluded)
    background_ids = _reservoir_background_ids(
        wikipedia,
        excluded_ids=excluded,
        target_count=background_target,
        seed=seed,
        expected_source_rows=source_passage_count,
    )
    selected_ids = excluded | background_ids
    if len(selected_ids) != target_passages:
        raise RuntimeError("Final selected passage count differs from the requested target")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        corpus_dir = staging / "corpus"
        questions_dir = staging / "questions"
        passages_descriptor, passage_ids_descriptor = _write_selected_corpus(
            wikipedia,
            selected_ids=selected_ids,
            corpus_dir=corpus_dir,
            expected_source_rows=source_passage_count,
        )
        calibration_descriptor, evaluation_descriptor = _write_questions(
            selected_questions,
            calibration_count=calibration_questions,
            questions_dir=questions_dir,
        )
        canonical_counts = (
            target_passages == DEFAULT_TARGET_PASSAGES
            and calibration_questions == DEFAULT_CALIBRATION_QUESTIONS
            and evaluation_questions == DEFAULT_EVALUATION_QUESTIONS
            and hard_negatives_per_question == DEFAULT_HARD_NEGATIVES_PER_QUESTION
            and seed == DEFAULT_SEED
        )

        corpus_manifest = {
            "status": "complete",
            "protocol": PROTOCOL,
            "canonical_counts": canonical_counts,
            "format": CORPUS_FORMAT,
            "text_format": TEXT_FORMAT,
            "preparation_source_sha256": sha256_file(Path(__file__)),
            "sources": {
                "wikipedia": {
                    "resource": "DPR data.wikipedia_split.psgs_w100",
                    "file_name": wikipedia.name,
                    "sha256": wikipedia_sha,
                    "rows": source_passage_count,
                },
                "questions": {
                    "resource": "DPR data.retriever.nq-dev",
                    "file_name": questions_source.name,
                    "sha256": questions_sha,
                    "rows": source_question_count,
                },
            },
            "selection": {
                "seed": seed,
                "target_passages": target_passages,
                "question_normalization": QUESTION_NORMALIZATION,
                "context_match_normalization": CONTEXT_MATCH_NORMALIZATION,
                "question_selection": QUESTION_SELECTION,
                "positive_mapping": "passage_id_then_nfkc_title_text_v1",
                "positive_title_expansion": "nfkc_title_v1",
                "hard_negatives": {
                    "per_question": hard_negatives_per_question,
                    "selection": "source_order_question_round_robin_v1",
                },
                "background": {
                    "sampler": BACKGROUND_SAMPLER,
                    "seed_derivation": "sha256_seed_nul_background_first_u64_be_v1",
                },
            },
            "counts": {
                "selected_questions": total_questions,
                "annotated_positive_passages": len(positive_ids),
                "positive_title_expansion_passages": len(mandatory_ids - positive_ids),
                "mandatory_passages": len(mandatory_ids),
                "hard_negative_passages": len(hard_ids),
                "random_background_passages": len(background_ids),
                "final_passages": len(selected_ids),
            },
            "sets_sha256": {
                "selected_question_ids": _question_ids_sha256(
                    [item.question_id for item in selected_questions]
                ),
                "annotated_positive_passage_ids": _ids_sha256(positive_ids),
                "mandatory_passage_ids": _ids_sha256(mandatory_ids),
                "hard_negative_passage_ids": _ids_sha256(hard_ids),
                "random_background_passage_ids": _ids_sha256(background_ids),
                "selected_passage_ids": _ids_sha256(selected_ids),
            },
            "artifacts": {
                "passages": passages_descriptor,
                "passage_ids": passage_ids_descriptor,
            },
            "validation": {
                "unique_passage_ids": True,
                "strictly_increasing_output_ids": True,
                "selected_question_positive_coverage": 1.0,
                "positive_mapping_is_exact": True,
            },
        }
        corpus_manifest_path = corpus_dir / "manifest.json"
        write_manifest(corpus_manifest_path, corpus_manifest)

        question_ids = [item.question_id for item in selected_questions]
        questions_manifest = {
            "status": "complete",
            "protocol": PROTOCOL,
            "canonical_counts": canonical_counts,
            "source": {
                "resource": "DPR data.retriever.nq-dev",
                "file_name": questions_source.name,
                "sha256": questions_sha,
                "rows": source_question_count,
            },
            "selection": {
                "seed": seed,
                "normalization": QUESTION_NORMALIZATION,
                "question_selection": QUESTION_SELECTION,
                "split": "first_calibration_count_in_selection_hash_order_v1",
            },
            "counts": {
                "selected": total_questions,
                "calibration": calibration_questions,
                "evaluation": evaluation_questions,
            },
            "selected_question_ids_sha256": _question_ids_sha256(question_ids),
            "artifacts": {
                "calibration": calibration_descriptor,
                "evaluation": evaluation_descriptor,
            },
            "validation": {
                "unique_question_ids": len(question_ids) == len(set(question_ids)),
                "positive_evidence_coverage": 1.0,
            },
        }
        questions_manifest_path = questions_dir / "manifest.json"
        write_manifest(questions_manifest_path, questions_manifest)

        root_manifest = {
            "status": "complete",
            "protocol": PROTOCOL,
            "canonical_counts": canonical_counts,
            "request": request,
            "manifests": {
                "corpus": describe_artifact(
                    corpus_manifest_path,
                    relative_to=staging,
                ),
                "questions": describe_artifact(
                    questions_manifest_path,
                    relative_to=staging,
                ),
            },
        }
        write_manifest(staging / "manifest.json", root_manifest)
        os.replace(staging, output)
        return validate_prepared_dataset(
            output,
            expected_protocol=PROTOCOL,
            expected_request=request,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
