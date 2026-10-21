"""Construct the shared request payload and submit it once without retries."""

import os
import time


def build_payload(settings, prompt, max_tokens):
    return {
        "model": settings["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "extra_body": {"enable_thinking": False},
    }


class LLM:
    def __init__(self, settings, client=None):
        self.owns_client = client is None
        if client is None:
            from openai import OpenAI

            key = os.environ.get(settings["api_key_env"])
            if not key:
                raise ValueError(
                    "Set " + settings["api_key_env"] + " before generation"
                )
            client = OpenAI(
                api_key=key,
                base_url=settings["base_url"],
                timeout=settings["timeout_seconds"],
                max_retries=0,
            )
        self.client = client

    def close(self):
        if self.owns_client:
            self.client.close()

    def request_once(self, payload):
        """不自动重试；拿到响应后解析失败时也保留已有响应，便于人工核对。"""
        model = payload["model"]
        started = time.perf_counter()
        response = None
        raw = None
        try:
            response = self.client.chat.completions.create(**payload)
            raw = response.model_dump(mode="json")
            choice = response.choices[0]
            return {
                "status": (
                    "success" if response.model == model else "model_identity_mismatch"
                ),
                "answer": choice.message.content or "",
                "finish_reason": choice.finish_reason,
                "refusal": getattr(choice.message, "refusal", None),
                "response_id": response.id,
                "requested_model": model,
                "returned_model": response.model,
                "usage": raw.get("usage"),
                "raw_response": raw,
                "latency_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            # 有 HTTP 状态码是明确错误；无状态码时，无法判断服务端是否已经完成。
            result = {
                "status": "failed" if code is not None else "unknown",
                "http_status": code,
                "failure_class": type(exc).__name__,
                "error": str(exc),
                "requested_model": model,
                "latency_seconds": time.perf_counter() - started,
                "reconciliation": (
                    "HTTP response received"
                    if code is not None
                    else "submission outcome unknown"
                ),
            }
            if response is not None:
                result.update(
                    response_id=getattr(response, "id", None),
                    returned_model=getattr(response, "model", None),
                    reconciliation="Response received but could not be parsed; inspect receipt",
                )
            if raw is not None:
                result.update(
                    raw_response=raw,
                    usage=raw.get("usage") if isinstance(raw, dict) else None,
                )
            return result
