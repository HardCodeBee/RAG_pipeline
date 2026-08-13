from __future__ import annotations

import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_token_count(text: str) -> int:
    """Estimate token count without loading a model tokenizer."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(TOKEN_RE.findall(text))

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
        return approx_token_count(text)

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
