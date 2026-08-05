"""Common query-time contract for dense and lexical retrievers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from src.records import RetrievalTrace


class Retriever(Protocol):
    """Return ranked chunks while preserving the pipeline trace contract."""

    def retrieve_trace(
        self,
        query: str,
        top_k: int | None = None,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> RetrievalTrace: ...
