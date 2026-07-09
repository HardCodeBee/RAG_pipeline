from __future__ import annotations

import re
from statistics import mean


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def expected_source_hit(results: list[dict], expected_sources: list[str] | None) -> bool | None:
    if not expected_sources:
        return None
    expected = {source.lower() for source in expected_sources}
    retrieved = {result.get("source", "").lower() for result in results}
    return bool(expected & retrieved)


def answer_contains_gold(answer: str, gold_answer: str | None) -> bool | None:
    if not gold_answer:
        return None
    answer_norm = normalize_text(answer)
    gold_norm = normalize_text(gold_answer)
    if not gold_norm:
        return None
    return gold_norm in answer_norm


def summarize_results(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_generation_latency_ms": 0.0,
            "avg_total_latency_ms": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "retrieval_expected_source_hit_rate": "",
            "answer_contains_gold_rate": "",
        }

    def avg(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            item = row
            for key in path:
                item = item.get(key, {})
            if isinstance(item, (int, float)):
                values.append(float(item))
        return mean(values) if values else 0.0

    hit_values = [
        row.get("metrics", {}).get("retrieval_expected_source_hit")
        for row in rows
        if row.get("metrics", {}).get("retrieval_expected_source_hit") is not None
    ]
    answer_values = [
        row.get("metrics", {}).get("answer_contains_gold")
        for row in rows
        if row.get("metrics", {}).get("answer_contains_gold") is not None
    ]

    return {
        "num_questions": len(rows),
        "avg_retrieval_latency_ms": avg(("retrieval", "latency_ms")),
        "avg_generation_latency_ms": avg(("generation", "latency_ms")),
        "avg_total_latency_ms": avg(("total_latency_ms",)),
        "avg_input_tokens": avg(("generation", "input_tokens")),
        "avg_output_tokens": avg(("generation", "output_tokens")),
        "retrieval_expected_source_hit_rate": mean(hit_values) if hit_values else "",
        "answer_contains_gold_rate": mean(answer_values) if answer_values else "",
    }

