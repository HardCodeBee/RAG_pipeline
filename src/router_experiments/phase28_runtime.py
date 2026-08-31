"""Public Phase 2.8 runtime functions reused by Phase 2.8b."""

from scripts.run_router_phase28_query_expansion import (
    _completed_labels,
    _prepare_bm25_incremental,
    _prepare_dense_incremental,
    _spent_generation_cost,
    generate,
)

completed_labels = _completed_labels
prepare_bm25_incremental = _prepare_bm25_incremental
prepare_dense_incremental = _prepare_dense_incremental
spent_generation_cost = _spent_generation_cost

__all__ = [
    "completed_labels",
    "generate",
    "prepare_bm25_incremental",
    "prepare_dense_incremental",
    "spent_generation_cost",
]

