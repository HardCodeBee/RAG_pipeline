from __future__ import annotations

from typing import Any

from src.io_utils import TOKEN_RE, regex_token_sequence

# 专门检查“token 窗口大小”和“重叠 token 数”是否合法
def validate_token_window(
    size: int,
    overlap: int = 0,
    *,
    size_name: str = "size",
    overlap_name: str = "overlap",
) -> None:
    # bool 是 int 的子类，所以要显式排除 True/False。
    # size 必须是正整数
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{size_name} must be a positive integer")
    # overlap 必须是非负整数
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError(f"{overlap_name} must be a non-negative integer")
    # overlap 必须小于 size
    if overlap >= size:
        raise ValueError(f"{overlap_name} must be smaller than {size_name}")

# 给所有 token counter 统一提供 truncate() 方法
class _TruncationMixin:
    def truncate(self, text: str, max_tokens: int) -> str:
        parts = self.split(text, max_tokens)
        return parts[0] if parts else ""


class RegexTokenCounter(_TruncationMixin):
    # 使用统一 TOKEN_RE 做轻量 token 估算，适合无模型环境和快速测试。
    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(self.token_sequence(text))

    # 返回经过简单正则切分后的 token 序列
    def token_sequence(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # 返回完整 token 序列，供 overlap 统计等需要比较 token 边界的地方使用。
        return regex_token_sequence(text)

    # 把一段文本按最多 max_tokens 个 token 一组切成多个文本片段。
    def split(self, text: str, max_tokens: int) -> list[str]:
        validate_token_window(max_tokens) # 检查max_tokens 必须是正整数
        matches = list(TOKEN_RE.finditer(text))
        if not matches:
            return []
        parts = []
        # 每次取最多 max_tokens 个 token，然后根据第一个 token 的起始位置和最后一个 token 的结束位置，从原文中截取对应文本。
        for start in range(0, len(matches), max_tokens):
            selected = matches[start : start + max_tokens]
            parts.append(text[selected[0].start() : selected[-1].end()].strip())
        return [part for part in parts if part] # 过滤掉空字符串

class HuggingFaceTokenCounter(_TruncationMixin):
    # 使用真实 tokenizer 计数，适合需要更接近模型上下文窗口的实验。
    # tokenizer_kwargs：额外传给 AutoTokenizer.from_pretrained() 的参数
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        local_files_only: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        # model_name 必须是非空字符串
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        # 延迟导入 transformers
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face token counting requires transformers; install requirements/experiment.txt"
            ) from exc
        # 组织加载参数
        kwargs = dict(tokenizer_kwargs or {})
        kwargs["local_files_only"] = bool(local_files_only)
        # 处理版本固定：revision 固定，tokenizer 行为更容易复现。
        if revision is not None:
            kwargs["revision"] = revision
        self.model_name = model_name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    # 把文本编码成 Hugging Face tokenizer 的 token id
    def _encode(self, text: str) -> list[int]:
        # 不加特殊词元，因为 chunk 预算只针对原始文本内容。
        return list(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))
    # 把 token id 还原成文本
    def _decode(self, token_ids: list[int]) -> str:
        # 解码后去掉两侧空白，避免片段边界留下额外空白。
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    # 返回文本经过 Hugging Face tokenizer 后的 token 数量
    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(self.token_sequence(text))
    # 返回完整 token id 序列
    def token_sequence(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._encode(text)

        # 用真实 Hugging Face tokenizer，把长文本切成多个片段，并保证每个片段不超过 max_token
    def split(self, text: str, max_tokens: int) -> list[str]:

        validate_token_window(max_tokens)

        token_ids = self._encode(text)
        parts = []
        start = 0

        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids)) # 计算本次窗口终点
            decoded = self._decode(token_ids[start:end]) # 把当前 token id 窗口 decode 回文本
            # 有些 tokenizer 解码后再编码的 token 数可能变化，这里收缩窗口保证不超预算。
            while end > start + 1 and self.count(decoded) > max_tokens:
                end -= 1
                decoded = self._decode(token_ids[start:end])
            if decoded:
                parts.append(decoded)
            start = end
        return parts
