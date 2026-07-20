from __future__ import annotations

import re

# 把 PageRecord.text 拆成句子
# 简单正则分句器：在英文/中文句末标点后按空白拆分
class RegexSentenceSplitter:
    def __init__(self, pattern: str = r"(?<=[.!?。！？])\s+"):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        # 预编译正则，避免每次 split 都重新编译。
        self._compiled = re.compile(pattern)

    # 定义切分方法
    def split(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # 去掉空片段，保证 chunker 不会收到空句子。
        return [part.strip()
                for part in self._compiled.split(text.strip())
                if part.strip()]
