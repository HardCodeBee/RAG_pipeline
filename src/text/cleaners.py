from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """执行baseline loader使用的最小 PDF 文本清洗。"""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # PDF 抽取结果中偶尔会出现 NUL 字符，先替换成空格避免写入 JSONL 出问题。
    text = text.replace("\x00", " ")
    # 处理 PDF 换行造成的断词，例如 "retriev-\nal" -> "retrieval"。
    text = re.sub(r"-\s*\n\s*", "", text)
    # 其他换行、制表符和连续空格统一折叠成一个空格。
    return re.sub(r"\s+", " ", text).strip()
