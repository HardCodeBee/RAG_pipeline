from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.io_utils import slugify


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[dict]:
    path = Path(path)
    doc_id = slugify(path.stem)
    reader = PdfReader(str(path))
    records: list[dict] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        records.append(
            {
                "doc_id": doc_id,
                "source": path.name,
                "page": page_index,
                "text": text,
            }
        )

    return records


def load_pdfs(corpus_path: str | Path) -> list[dict]:
    corpus_path = Path(corpus_path)
    pdfs = sorted(corpus_path.glob("*.pdf"))
    records: list[dict] = []
    for pdf in pdfs:
        records.extend(load_pdf(pdf))
    return records


def iter_supported_documents(corpus_path: str | Path, file_type: str = "pdf") -> Iterable[Path]:
    corpus_path = Path(corpus_path)
    if file_type.lower() != "pdf":
        raise ValueError(f"Unsupported file_type for v1: {file_type}")
    return sorted(corpus_path.glob("*.pdf"))

