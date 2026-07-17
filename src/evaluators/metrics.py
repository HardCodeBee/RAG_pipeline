"""基线 RAG 流水线的轻量评估指标。"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean


def normalize_text(text: str) -> str:
    """转小写并移除标点，用于简单 exact-match 风格检查。"""
    # 这里是轻量归一化，不做词干化或语义匹配，只用于 baseline 指标。
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _source_key(value) -> str:
    return str(value).strip().casefold()


def _source_keys(values) -> set[str]:
    """统一来源标识，避免不同检索指标各自实现大小写和空白处理。"""
    return {
        _source_key(value)
        for value in values
        if _source_key(value)
    }


def expected_source_hit(results: list[dict], expected_sources: list[str] | None) -> bool | None:
    """判断检索来源中是否有任意一个命中标注来源。"""
    # 没有标注 expected_sources 时返回 None，表示该指标不参与汇总。
    if not expected_sources:
        return None
    expected = _source_keys(expected_sources)
    if not expected:
        return None
    retrieved = _source_keys(result.get("source", "") for result in results)
    return bool(expected & retrieved)


def answer_contains_gold(answer: str, gold_answer: str | None) -> bool | None:
    """判断生成答案是否包含归一化后的标准答案。"""
    if not gold_answer:
        return None
    answer_norm = normalize_text(answer)
    gold_norm = normalize_text(gold_answer)
    if not gold_norm:
        return None
    return gold_norm in answer_norm


def answer_exact_match(answer: str, gold_answer: str | None) -> bool | None:
    # 精确匹配比“包含 gold”更严格：整段答案归一化后必须完全等于 gold。
    if gold_answer is None:
        return None
    gold_norm = normalize_text(gold_answer)
    if not gold_norm:
        return None
    return normalize_text(answer) == gold_norm


def answer_token_f1(answer: str, gold_answer: str | None) -> float | None:
    # 词元 F1 用词袋重叠衡量生成答案和标准答案的相似度。
    if gold_answer is None:
        return None
    gold_tokens = normalize_text(gold_answer).split()
    if not gold_tokens:
        return None
    answer_tokens = normalize_text(answer).split()
    if not answer_tokens:
        return 0.0
    overlap = sum((Counter(answer_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_is_refusal(answer: str) -> bool:
    # 判断模型是否按约定拒答，用于不可回答问题的评估。
    normalized = normalize_text(answer)
    refusal_phrases = (
        "i don t know based on the provided context",
        "i do not know based on the provided context",
        "insufficient information in the provided context",
    )
    return any(phrase in normalized for phrase in refusal_phrases)


def answerability_decision_accuracy(answer: str, answerable: bool | None) -> bool | None:
    # answerable=True 时希望不拒答；answerable=False 时希望拒答。
    if answerable is None:
        return None
    if not isinstance(answerable, bool):
        raise TypeError("answerable must be a boolean or None")
    return answer_is_refusal(answer) is (not answerable)


def source_recall_at_k(
    results: list[dict],
    expected_sources: list[str] | None,
    top_k: int | None = None,
) -> float | None:
    # 来源召回率关注“标注需要的来源文件，有多少出现在检索结果里”。
    if not expected_sources:
        return None
    expected = _source_keys(expected_sources)
    if not expected:
        return None
    selected = results[:top_k] if top_k is not None else results
    retrieved = _source_keys(result.get("source", "") for result in selected)
    return len(expected & retrieved) / len(expected)


def source_precision_at_k(
    results: list[dict],
    expected_sources: list[str] | None,
    top_k: int | None = None,
) -> float | None:
    # 来源精确率关注“检索出来的来源文件中，有多少属于标注来源”。
    if not expected_sources:
        return None
    expected = _source_keys(expected_sources)
    if not expected:
        return None
    selected = results[:top_k] if top_k is not None else results
    if not selected:
        return 0.0
    hits = sum(_source_key(result.get("source", "")) in expected for result in selected)
    return hits / len(selected)


def _required_evidence(expected_evidence: list[dict | str] | None) -> list[dict | str]:
    if not expected_evidence:
        return []
    # evidence 标注允许 required=False，表示辅助证据不强制命中。
    return [
        item
        for item in expected_evidence
        if not isinstance(item, dict) or item.get("required", True) is not False
    ]


def _evidence_matches(result: dict, evidence: dict | str) -> bool:
    # evidence 可以直接写 chunk_id 字符串，也可以写包含 source/page/chunk_id 的 dict。
    if isinstance(evidence, str):
        return str(result.get("chunk_id", "")) == evidence
    alternatives = evidence.get("alternatives")
    if alternatives is not None:
        # alternatives 表示任意一个证据命中即可。
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("evidence.alternatives must be a non-empty list")
        return any(_evidence_matches(result, alternative) for alternative in alternatives)
    compared = False
    chunk_id = evidence.get("chunk_id")
    if chunk_id is not None:
        compared = True
        # 如果显式标了 chunk_id，就必须精确匹配。
        if str(result.get("chunk_id", "")) != str(chunk_id):
            return False
    source = evidence.get("source")
    if source is not None:
        compared = True
        # source 比较忽略大小写和两侧空白。
        if str(result.get("source", "")).strip().casefold() != str(source).strip().casefold():
            return False
    page_start = evidence.get("page_start", evidence.get("page"))
    page_end = evidence.get("page_end", evidence.get("page"))
    if page_start is not None or page_end is not None:
        compared = True
        expected_start = int(page_start if page_start is not None else page_end)
        expected_end = int(page_end if page_end is not None else page_start)
        result_start = int(result.get("page_start", result.get("page", -1)))
        result_end = int(result.get("page_end", result.get("page", result_start)))
        # 页码证据用区间相交判断，允许 chunk 覆盖多页。
        if result_end < expected_start or result_start > expected_end:
            return False
    return compared


def evidence_recall_at_k(
    results: list[dict],
    expected_evidence: list[dict | str] | None,
    top_k: int | None = None,
) -> float | None:
    # 证据召回率关注“必需证据中有多少被 top_k 检索到”。
    evidence = _required_evidence(expected_evidence)
    if not evidence:
        return None
    selected = results[:top_k] if top_k is not None else results
    matched = sum(any(_evidence_matches(result, item) for result in selected) for item in evidence)
    return matched / len(evidence)


def evidence_all_hit(
    results: list[dict],
    expected_evidence: list[dict | str] | None,
    top_k: int | None = None,
) -> bool | None:
    # 全命中是证据召回率是否达到 100% 的布尔版本。
    recall = evidence_recall_at_k(results, expected_evidence, top_k=top_k)
    if recall is None:
        return None
    return recall == 1.0


def evidence_mrr(
    results: list[dict],
    expected_evidence: list[dict | str] | None,
    top_k: int | None = None,
) -> float | None:
    # MRR 关注第一个命中证据出现得有多靠前。
    evidence = _required_evidence(expected_evidence)
    if not evidence:
        return None
    selected = results[:top_k] if top_k is not None else results
    for position, result in enumerate(selected, start=1):
        if any(_evidence_matches(result, item) for item in evidence):
            return 1.0 / position
    return 0.0


def evaluate_result(
    answer: str,
    results: list[dict],
    gold_answer: str | None = None,
    expected_sources: list[str] | None = None,
    expected_evidence: list[dict | str] | None = None,
    answerable: bool | None = None,
) -> dict[str, bool | float | None]:
    # 不可回答问题不应用标准答案文本指标，否则会鼓励模型编答案。
    scored_gold_answer = None if answerable is False else gold_answer
    return {
        "retrieval_expected_source_hit": expected_source_hit(results, expected_sources),
        "retrieval_source_recall_at_k": source_recall_at_k(results, expected_sources),
        "retrieval_source_precision_at_k": source_precision_at_k(results, expected_sources),
        "retrieval_evidence_recall_at_k": evidence_recall_at_k(results, expected_evidence),
        "retrieval_evidence_all_hit": evidence_all_hit(results, expected_evidence),
        "retrieval_evidence_mrr": evidence_mrr(results, expected_evidence),
        "answer_contains_gold": answer_contains_gold(answer, scored_gold_answer),
        "answer_exact_match": answer_exact_match(answer, scored_gold_answer),
        "answer_token_f1": answer_token_f1(answer, scored_gold_answer),
        "answerability_decision_accuracy": answerability_decision_accuracy(answer, answerable),
    }


def summarize_results(rows: list[dict]) -> dict:
    """把逐问题结果聚合成一个评估汇总。"""
    def numeric_values(path: tuple[str, ...]) -> list[float]:
        # 提取数值列表给 percentile 使用；bool 不当作数值延迟统计。
        values = []
        for row in rows:
            item = row
            for key in path:
                if not isinstance(item, dict):
                    item = None
                    break
                item = item.get(key)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                values.append(float(item))
        return values

    def avg(path: tuple[str, ...]) -> float:
        """沿嵌套字典路径取数值并计算平均值。"""
        values = numeric_values(path)
        return mean(values) if values else 0.0

    def percentile(values: list[float], fraction: float) -> float:
        # 线性插值 percentile，避免依赖 numpy。
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    # 忽略指标为 None 的未标注行。
    def metric_values(name: str, *, answerable: bool | None = None) -> list[float]:
        # 布尔指标转成 1.0/0.0 后可直接求平均得到命中率。
        values = []
        for row in rows:
            if answerable is not None and row.get("answerable") is not answerable:
                continue
            value = row.get("metrics", {}).get(name)
            if isinstance(value, (bool, int, float)):
                values.append(float(value))
        return values

    hit_values = metric_values("retrieval_expected_source_hit")
    answer_values = metric_values("answer_contains_gold")
    source_recall_values = metric_values("retrieval_source_recall_at_k")
    source_precision_values = metric_values("retrieval_source_precision_at_k")
    evidence_recall_values = metric_values("retrieval_evidence_recall_at_k")
    evidence_all_values = metric_values("retrieval_evidence_all_hit")
    evidence_mrr_values = metric_values("retrieval_evidence_mrr")
    exact_match_values = metric_values("answer_exact_match")
    token_f1_values = metric_values("answer_token_f1")
    decision_values = metric_values("answerability_decision_accuracy")
    # 两个分组指标都是同一决策正确率在不同标签子集上的视图，不重复写入逐行 metrics。
    answerable_response_values = metric_values("answerability_decision_accuracy", answerable=True)
    unanswerable_refusal_values = metric_values("answerability_decision_accuracy", answerable=False)
    success_count = sum(row.get("status") == "success" for row in rows)
    fallback_count = sum(row.get("status") == "fallback" for row in rows)
    failed_count = sum(row.get("status") == "error" for row in rows)
    retrieval_latencies = numeric_values(("retrieval", "latency_ms"))
    generation_latencies = numeric_values(("generation", "latency_ms"))
    total_latencies = numeric_values(("total_latency_ms",))

    return {
        "num_questions": len(rows),
        "avg_retrieval_latency_ms": avg(("retrieval", "latency_ms")),
        "avg_generation_latency_ms": avg(("generation", "latency_ms")),
        "avg_total_latency_ms": avg(("total_latency_ms",)),
        "p50_retrieval_latency_ms": percentile(retrieval_latencies, 0.50),
        "p95_retrieval_latency_ms": percentile(retrieval_latencies, 0.95),
        "p50_generation_latency_ms": percentile(generation_latencies, 0.50),
        "p95_generation_latency_ms": percentile(generation_latencies, 0.95),
        "p50_total_latency_ms": percentile(total_latencies, 0.50),
        "p95_total_latency_ms": percentile(total_latencies, 0.95),
        "avg_input_tokens": avg(("generation", "input_tokens")),
        "avg_output_tokens": avg(("generation", "output_tokens")),
        "avg_estimated_input_tokens": avg(("generation", "token_usage", "estimated", "input_tokens")),
        "avg_estimated_output_tokens": avg(("generation", "token_usage", "estimated", "output_tokens")),
        "avg_provider_input_tokens": avg(("generation", "token_usage", "provider_reported", "input_tokens")),
        "avg_provider_output_tokens": avg(("generation", "token_usage", "provider_reported", "output_tokens")),
        "num_successful_questions": success_count,
        "num_fallback_questions": fallback_count,
        "num_failed_questions": failed_count,
        "num_answerable_questions": sum(row.get("answerable") is True for row in rows),
        "num_unanswerable_questions": sum(row.get("answerable") is False for row in rows),
        "retrieval_expected_source_hit_rate": mean(hit_values) if hit_values else "",
        "retrieval_expected_source_hit_valid_count": len(hit_values),
        "retrieval_source_recall_at_k": mean(source_recall_values) if source_recall_values else "",
        "retrieval_source_recall_at_k_valid_count": len(source_recall_values),
        "retrieval_source_precision_at_k": mean(source_precision_values) if source_precision_values else "",
        "retrieval_source_precision_at_k_valid_count": len(source_precision_values),
        "retrieval_evidence_recall_at_k": mean(evidence_recall_values) if evidence_recall_values else "",
        "retrieval_evidence_recall_at_k_valid_count": len(evidence_recall_values),
        "retrieval_evidence_all_hit_rate": mean(evidence_all_values) if evidence_all_values else "",
        "retrieval_evidence_all_hit_valid_count": len(evidence_all_values),
        "retrieval_evidence_mrr": mean(evidence_mrr_values) if evidence_mrr_values else "",
        "retrieval_evidence_mrr_valid_count": len(evidence_mrr_values),
        "answer_contains_gold_rate": mean(answer_values) if answer_values else "",
        "answer_contains_gold_valid_count": len(answer_values),
        "answer_exact_match_rate": mean(exact_match_values) if exact_match_values else "",
        "answer_exact_match_valid_count": len(exact_match_values),
        "answer_token_f1": mean(token_f1_values) if token_f1_values else "",
        "answer_token_f1_valid_count": len(token_f1_values),
        "answerability_decision_accuracy": mean(decision_values) if decision_values else "",
        "answerability_decision_accuracy_valid_count": len(decision_values),
        "answerable_non_refusal_rate": mean(answerable_response_values) if answerable_response_values else "",
        "answerable_non_refusal_valid_count": len(answerable_response_values),
        "unanswerable_refusal_accuracy": mean(unanswerable_refusal_values) if unanswerable_refusal_values else "",
        "unanswerable_refusal_valid_count": len(unanswerable_refusal_values),
    }
