"""Protocol-independent execution statistics for evaluation runs."""

from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence


def _numeric_values(
    rows: Sequence[Mapping[str, Any]],
    path: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        item: Any = row
        for key in path:
            if not isinstance(item, Mapping):
                item = None
                break
            item = item.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))
    return values


def _average(rows: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> float:
    values = _numeric_values(rows, path)
    return mean(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _provider_token_value(row: Mapping[str, Any], field: str) -> int | None:
    generation = row.get("generation")
    usage = generation.get("token_usage") if isinstance(generation, Mapping) else None
    reported = usage.get("provider_reported") if isinstance(usage, Mapping) else None
    value = reported.get(field) if isinstance(reported, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def summarize_execution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Summarize status, latency, and token usage without protocol scores."""

    retrieval_latencies = _numeric_values(rows, ("retrieval", "latency_ms"))
    generation_latencies = _numeric_values(rows, ("generation", "latency_ms"))
    total_latencies = _numeric_values(rows, ("total_latency_ms",))
    provider_input_tokens = [
        value
        for row in rows
        if (value := _provider_token_value(row, "input_tokens")) is not None
    ]
    provider_output_tokens = [
        value
        for row in rows
        if (value := _provider_token_value(row, "output_tokens")) is not None
    ]
    provider_total_tokens = [
        value
        for row in rows
        if (value := _provider_token_value(row, "total_tokens")) is not None
    ]
    summary: dict[str, int | float] = {
        "num_questions": len(rows),
        "num_successful_questions": sum(row.get("status") == "success" for row in rows),
        "num_failed_questions": sum(row.get("status") == "error" for row in rows),
        "avg_retrieval_latency_ms": _average(rows, ("retrieval", "latency_ms")),
        "avg_generation_latency_ms": _average(rows, ("generation", "latency_ms")),
        "avg_total_latency_ms": _average(rows, ("total_latency_ms",)),
        "p50_retrieval_latency_ms": _percentile(retrieval_latencies, 0.50),
        "p95_retrieval_latency_ms": _percentile(retrieval_latencies, 0.95),
        "p50_generation_latency_ms": _percentile(generation_latencies, 0.50),
        "p95_generation_latency_ms": _percentile(generation_latencies, 0.95),
        "p50_total_latency_ms": _percentile(total_latencies, 0.50),
        "p95_total_latency_ms": _percentile(total_latencies, 0.95),
        "avg_estimated_input_tokens": _average(
            rows,
            ("generation", "token_usage", "estimated", "input_tokens"),
        ),
        "avg_estimated_output_tokens": _average(
            rows,
            ("generation", "token_usage", "estimated", "output_tokens"),
        ),
        "avg_provider_input_tokens": (
            mean(provider_input_tokens) if provider_input_tokens else 0.0
        ),
        "avg_provider_output_tokens": (
            mean(provider_output_tokens) if provider_output_tokens else 0.0
        ),
        "total_provider_input_tokens": sum(provider_input_tokens),
        "total_provider_output_tokens": sum(provider_output_tokens),
        "total_provider_total_tokens": sum(provider_total_tokens),
    }
    for stage in (
        "query_embedding_ms",
        "index_search_ms",
        "chunk_mapping_ms",
        "rerank_ms",
    ):
        values = _numeric_values(rows, ("retrieval", "timings_ms", stage))
        label = stage.removesuffix("_ms")
        summary[f"avg_{label}_latency_ms"] = mean(values) if values else 0.0
        summary[f"p50_{label}_latency_ms"] = _percentile(values, 0.50)
        summary[f"p95_{label}_latency_ms"] = _percentile(values, 0.95)
    return summary
