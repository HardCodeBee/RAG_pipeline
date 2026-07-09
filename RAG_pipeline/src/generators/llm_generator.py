from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from src.io_utils import approx_token_count


@dataclass
class GenerationResult:
    answer: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            "\n".join(
                [
                    f"[Chunk {item['rank']} | source={item['source']}, pages={item['page_start']}-{item['page_end']}]",
                    item["text"],
                ]
            )
        )
    context = "\n\n".join(context_parts)
    return "\n".join(
        [
            "You are a question answering assistant.",
            "",
            "Answer the question using only the provided context.",
            'If the context does not contain enough information, say: "I don\'t know based on the provided context."',
            "",
            "Question:",
            question,
            "",
            "Context:",
            context,
            "",
            "Answer:",
        ]
    )


class LLMGenerator:
    def __init__(
        self,
        provider: str = "auto",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_config(cls, config: dict) -> "LLMGenerator":
        generation = config.get("generation", {})
        return cls(
            provider=generation.get("provider", "auto"),
            model=generation.get("model", "gpt-4o-mini"),
            temperature=float(generation.get("temperature", 0.0)),
            max_output_tokens=int(generation.get("max_output_tokens", 512)),
        )

    def generate(self, question: str, retrieved_chunks: list[dict]) -> tuple[str, GenerationResult]:
        prompt = build_prompt(question, retrieved_chunks)
        started = time.perf_counter()

        if self._should_use_openai():
            try:
                answer, input_tokens, output_tokens = self._openai_generate(prompt)
                latency_ms = (time.perf_counter() - started) * 1000
                return prompt, GenerationResult(
                    answer=answer,
                    provider="openai",
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                answer = self._extractive_answer(question, retrieved_chunks)
                answer += f"\n\n[OpenAI generation failed; used extractive fallback: {exc.__class__.__name__}]"
        else:
            answer = self._extractive_answer(question, retrieved_chunks)

        latency_ms = (time.perf_counter() - started) * 1000
        return prompt, GenerationResult(
            answer=answer,
            provider="extractive",
            model="extractive-fallback",
            input_tokens=approx_token_count(prompt),
            output_tokens=approx_token_count(answer),
            latency_ms=latency_ms,
        )

    def _should_use_openai(self) -> bool:
        if self.provider == "extractive":
            return False
        if self.provider == "openai":
            return True
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _openai_generate(self, prompt: str) -> tuple[str, int, int]:
        from openai import OpenAI

        client = OpenAI()
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            answer = getattr(response, "output_text", "") or str(response)
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", approx_token_count(prompt)) or 0)
            output_tokens = int(getattr(usage, "output_tokens", approx_token_count(answer)) or 0)
            return answer, input_tokens, output_tokens

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", approx_token_count(prompt)) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", approx_token_count(answer)) or 0)
        return answer, input_tokens, output_tokens

    def _extractive_answer(self, question: str, retrieved_chunks: list[dict]) -> str:
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
                    scored_sentences.append((score, -result["rank"], sentence.strip(), citation))

        if not scored_sentences and retrieved_chunks:
            top = retrieved_chunks[0]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top["text"]) if s.strip()]
            citation = f"{top['source']} pp. {top['page_start']}-{top['page_end']}"
            scored_sentences = [(1, -top["rank"], sentence, citation) for sentence in sentences[:2]]

        if not scored_sentences:
            return "I don't know based on the provided context."

        scored_sentences.sort(key=lambda row: (-row[0], row[1]))
        selected = scored_sentences[:3]
        lines = ["Extractive fallback answer based on retrieved context:"]
        for _, _, sentence, citation in selected:
            lines.append(f"- {sentence} ({citation})")
        return "\n".join(lines)

