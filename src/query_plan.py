"""Per-query retrieval decisions independent of pipeline wiring."""

from __future__ import annotations

# Query decisions are structural inputs to the pipeline.

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


SearchParamValue: TypeAlias = bool | int | float | str


def _validated_k(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validated_search_params(
    value: Mapping[str, SearchParamValue],
) -> Mapping[str, SearchParamValue]:
    if not isinstance(value, Mapping):
        raise TypeError("search_params must be a mapping")

    validated: dict[str, SearchParamValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("search_params keys must be non-empty strings without surrounding whitespace")
        if not isinstance(item, (bool, int, float, str)):
            raise TypeError("search_params values must be boolean, integer, finite float, or string")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("search_params float values must be finite")
        if isinstance(item, str) and not item:
            raise ValueError("search_params string values must be non-empty")
        validated[key] = item

    return MappingProxyType(dict(sorted(validated.items())))


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """A validated retrieval plan for one query.

    ``candidate_k`` is the number requested from the retriever, while
    ``final_k`` is the maximum number allowed to leave optional post-retrieval
    processing such as reranking. A disabled plan uses zero for both values.
    """

    retrieval_enabled: bool
    candidate_k: int
    final_k: int
    search_params: Mapping[str, SearchParamValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.retrieval_enabled, bool):
            raise TypeError("retrieval_enabled must be a boolean")

        candidate_k = _validated_k(self.candidate_k, "candidate_k")
        final_k = _validated_k(self.final_k, "final_k")
        search_params = _validated_search_params(self.search_params)

        if not self.retrieval_enabled:
            if candidate_k != 0 or final_k != 0:
                raise ValueError("A disabled retrieval plan must use candidate_k=0 and final_k=0")
            if search_params:
                raise ValueError("A disabled retrieval plan must not define search_params")
        else:
            if candidate_k == 0 or final_k == 0:
                raise ValueError("An enabled retrieval plan must use positive candidate_k and final_k")
            if candidate_k < final_k:
                raise ValueError("candidate_k must be greater than or equal to final_k")

        object.__setattr__(self, "search_params", search_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_enabled": self.retrieval_enabled,
            "candidate_k": self.candidate_k,
            "final_k": self.final_k,
            "search_params": dict(self.search_params),
        }


def fixed_query_plan(
    top_k: int,
    *,
    candidate_k: int | None = None,
    search_params: Mapping[str, SearchParamValue] | None = None,
) -> QueryPlan:
    """Construct a fixed plan, using ``top_k=0`` as retrieval gating."""

    validated_top_k = _validated_k(top_k, "top_k")
    if validated_top_k == 0:
        if candidate_k not in {None, 0}:
            raise ValueError("candidate_k must be zero or None when top_k=0")
        return QueryPlan(
            retrieval_enabled=False,
            candidate_k=0,
            final_k=0,
            search_params={} if search_params is None else search_params,
        )

    effective_candidate_k = validated_top_k if candidate_k is None else candidate_k
    return QueryPlan(
        retrieval_enabled=True,
        candidate_k=effective_candidate_k,
        final_k=validated_top_k,
        search_params={} if search_params is None else search_params,
    )
