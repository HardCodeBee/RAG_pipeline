"""RAG pipeline共享的辅助函数。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

"""token 辅助"""
#轻量 token 切分规则
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def approx_token_count(text: str) -> int:
    """在不依赖模型 tokenizer 的情况下返回简单 token 估算值。"""
    # 这是轻量估算，不依赖外部模型或 OpenAI 的 tokenizer。
    # 主要用于日志里的词元估计值，不用于严格切分文本块。
    return len(regex_token_sequence(text))


def regex_token_sequence(text: str) -> list[str]:
    """按轻量正则规则返回 token 序列。"""
    return TOKEN_RE.findall(text)


"""JSONL 文件 I/O辅助"""

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """把字典写入 UTF-8 JSONL 文件，每行一个 JSON 对象。"""
    path = Path(path)
    # 写产物或评估结果时，父目录可能还不存在，这里统一创建。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n") #要求中文不转义


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """从 JSONL 文件逐条读取字典，并跳过空行。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            # JSONL 中允许存在空行；读取时直接跳过。
            if line:
                yield json.loads(line) # 生成器函数：返回一个Iterator 可以逐个产出结果的对象


""" 身份与完整性"""

def slugify(value: str) -> str:
    """文件名或标题 -> 清洗后的英文数字标题_原始标题的12位hash。"""
    original = value.strip()
    value = unicodedata.normalize("NFKC", original)

    # 统一格式
    value = re.sub(r"^\d+[_\-\s]+", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    base = value or "document"
    # 同名标题可能来自不同文件；短 hash 可以避免 doc_id 碰撞。
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return f"{base}_{digest}"


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """文件内容 -> sha256_file() -> 唯一内容指纹"""
    digest = hashlib.sha256() # 创建一个 SHA256 计算器
    with Path(path).open("rb") as handle:
        while True:
            # 分块读取，避免大 PDF 或大 embedding 文件一次性占用太多内存。
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
