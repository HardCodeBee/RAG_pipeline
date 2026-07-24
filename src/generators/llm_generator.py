"""prompt 构造和答案生成后端。"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.io_utils import approx_token_count


@dataclass
# 把一次生成结果包装成统一结构，方便后面的 pipeline.py 序列化、评估和记录日志
class GenerationResult:
    answer: str
    provider: str
    requested_model: str
    model: str
    response_id: str | None
    latency_ms: float # 生成阶段耗时
    token_usage: dict[str, Any] = field(default_factory=dict)

class LLMGenerator:
    """Explicit OpenAI or deterministic extractive generator."""

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ):
        # generator 是 query 阶段最后一步；这里集中校验生成参数。
        if not isinstance(provider, str) or provider.lower() not in {"openai", "extractive"}:
            raise ValueError("generator.provider must be one of: openai, extractive")
        if provider.lower() == "openai" and (not isinstance(model, str) or not model.strip()):
            raise ValueError("generator.model must be a non-empty string for OpenAI")
        if int(max_output_tokens) <= 0:
            raise ValueError("generator.max_output_tokens must be a positive integer")
        if float(timeout_seconds) <= 0:
            raise ValueError("generator.timeout_seconds must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("generator.max_retries must be a non-negative integer")
        # 保存配置
        self.provider = provider.lower()
        self.model = model.strip() if isinstance(model, str) and model.strip() else "extractive"
        self.temperature = temperature
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        environment_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = environment_key.strip() if environment_key and environment_key.strip() else None
        # OpenAI 客户端懒加载，只有真正调用 OpenAI 时才创建。
        self._client = None

    # 共同入口
    def generate_from_prompt(
        self,
        prompt: str,
        question: str,
        retrieved_chunks: list[dict],
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        started = time.perf_counter()

        if self.provider == "openai":
            answer, input_tokens, output_tokens, total_tokens, actual_model, response_id = (
                self._openai_generate(prompt)
            )
            latency_ms = (time.perf_counter() - started) * 1000
            return self._build_result(
                answer=answer,
                provider="openai",
                model=actual_model,
                response_id=response_id,
                latency_ms=latency_ms,
                prompt=prompt,
                provider_input_tokens=input_tokens,
                provider_output_tokens=output_tokens,
                provider_total_tokens=total_tokens,
            )

        answer = self._extractive_answer(question, retrieved_chunks)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._build_result(
            answer=answer,
            provider="extractive",
            model="extractive",
            latency_ms=latency_ms,
            prompt=prompt,
        )


    # 真正调用 OpenAI SDK 进行 generate
    def _openai_generate(
        self,
        prompt: str,
    ) -> tuple[str, int | None, int | None, int | None, str, str | None]:
        """调用 OpenAI 客户端库，并归一化不同接口的词元用量字段。"""
        from openai import OpenAI

        if self._client is None:
            # timeout/max_retries 交给 SDK 处理；pipeline 只记录总耗时。
            client_kwargs = {
                "timeout": self.timeout_seconds,
                "max_retries": self.max_retries,
            }
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self._client = OpenAI(**client_kwargs)
        client = self._client
        if hasattr(client, "responses"):
            # 较新的 SDK 暴露响应接口。
            # 响应接口返回 output_text 和输入/输出词元字段。
            response = client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            answer = getattr(response, "output_text", "") or str(response)
            usage = getattr(response, "usage", None)
            input_tokens = self._optional_usage_value(usage, "input_tokens")
            output_tokens = self._optional_usage_value(usage, "output_tokens")
            total_tokens = self._optional_usage_value(usage, "total_tokens")
            return (
                answer,
                input_tokens,
                output_tokens,
                total_tokens,
                str(getattr(response, "model", None) or self.model),
                str(response.id) if getattr(response, "id", None) else None,
            )

        # 兼容只暴露聊天补全接口的旧版 SDK。
        # 旧版 SDK 使用 chat.completions，usage 字段名也不同。
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = self._optional_usage_value(usage, "prompt_tokens")
        output_tokens = self._optional_usage_value(usage, "completion_tokens")
        total_tokens = self._optional_usage_value(usage, "total_tokens")
        return (
            answer,
            input_tokens,
            output_tokens,
            total_tokens,
            str(getattr(response, "model", None) or self.model),
            str(response.id) if getattr(response, "id", None) else None,
        )

    @staticmethod
    def _optional_usage_value(usage: Any, name: str) -> int | None:
        # 不同 SDK / provider 可能缺少某些 usage 字段，缺失时返回 None。
        value = getattr(usage, name, None)
        if value is None:
            return None
        return int(value)

    def _build_result(
        self,
        answer: str,
        provider: str,
        model: str,
        latency_ms: float,
        prompt: str,
        provider_input_tokens: int | None = None,
        provider_output_tokens: int | None = None,
        provider_total_tokens: int | None = None,
        response_id: str | None = None,
    ) -> GenerationResult:
        # 没有服务端用量时，使用正则词元估算值补齐日志字段。
        estimated_input_tokens = approx_token_count(prompt)
        estimated_output_tokens = approx_token_count(answer)
        input_source = "provider_reported" if provider_input_tokens is not None else "estimated"
        output_source = "provider_reported" if provider_output_tokens is not None else "estimated"
        if provider_total_tokens is None and provider_input_tokens is not None and provider_output_tokens is not None:
            # 有些 API 不返回 total，但返回 input/output 时可以安全相加。
            provider_total_tokens = provider_input_tokens + provider_output_tokens
        # 词元用量同时保存估算、服务端上报和计费视角，方便不同评估需求。
        token_usage = {
            "estimated": {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_input_tokens + estimated_output_tokens,
            },
            "provider_reported": {
                "input_tokens": provider_input_tokens,
                "output_tokens": provider_output_tokens,
                "total_tokens": provider_total_tokens,
            },
            "input_source": input_source,
            "output_source": output_source,
        }
        return GenerationResult(
            answer=answer,
            provider=provider,
            requested_model=self.model,
            model=model,
            response_id=response_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
    # 离线路径
    # Local extractive generation selects sentences from retrieved chunks.
    def _extractive_answer(self, question: str, retrieved_chunks: list[dict]) -> str:

        # 只取长度大于 3 的查询词，减少停用词造成的噪声匹配。
        question_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+", question)
            if len(token) > 3
        }
        scored_sentences: list[tuple[int, int, str, str]] = []
        for result in retrieved_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", result["text"])
            for sentence_index, sentence in enumerate(sentences):
                terms = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", sentence)}
                score = len(question_terms & terms)
                if score > 0:
                    citation = f"{result['source']} pp. {result['page_start']}-{result['page_end']}"
                    # 使用负 rank，让排序打平时较早的检索结果优先。
                    scored_sentences.append((score, -result["rank"], sentence.strip(), citation))

        if not scored_sentences and retrieved_chunks:
            # 如果没有句子共享查询词，就返回最高命中文本块的前几个句子。
            # 这样冒烟测试至少能生成基于检索文本的可检查答案。
            top = retrieved_chunks[0]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top["text"]) if s.strip()]
            citation = f"{top['source']} pp. {top['page_start']}-{top['page_end']}"
            scored_sentences = [(1, -top["rank"], sentence, citation) for sentence in sentences[:2]]

        if not scored_sentences:
            # 没有任何检索上下文时，按 prompt 约定拒答。
            return self._truncate_to_token_limit("I don't know based on the provided context.")

        scored_sentences.sort(key=lambda row: (-row[0], -row[1]))
        # Extractive answers stay short; this backend serves offline smoke tests.
        selected = scored_sentences[:3]
        lines = ["Extractive answer based on retrieved context:"]
        for _, _, sentence, citation in selected:
            lines.append(f"- {sentence} ({citation})")
        return self._truncate_to_token_limit("\n".join(lines))
    # 离线路径
    def _truncate_to_token_limit(self, text: str) -> str:
        # The extractive backend also obeys max_output_tokens.
        matches = list(re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))
        if len(matches) <= self.max_output_tokens:
            return text
        return text[: matches[self.max_output_tokens - 1].end()].rstrip()
