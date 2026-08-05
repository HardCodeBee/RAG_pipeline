"""把用户问题 + 检索得到的 context 拼成一个固定版本的问答 prompt，并给这个 prompt 生成 hash 方便复现"""

from __future__ import annotations

import hashlib

from src.records import ContextPackage, PromptPackage


PROMPT_VERSION = "fixed_qa_v1"
NQ_SHORT_PROMPT_VERSION = "nq_short_qa_v1"


def build_prompt(question: str, context: ContextPackage, version: str = PROMPT_VERSION) -> PromptPackage:
    # prompt 版本固定后，实验结果可以追溯到具体模板文本。
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if version not in {PROMPT_VERSION, NQ_SHORT_PROMPT_VERSION}:
        raise ValueError(f"Unsupported prompt version: {version}")
    if version == NQ_SHORT_PROMPT_VERSION:
        instructions = [
            "You are a short-answer question answering assistant.",
            "",
            "Answer using only the provided context.",
            "Return only the shortest answer span. Do not add an explanation or citation.",
            'If the context does not contain the answer, say exactly: "I don\'t know based on the provided context."',
        ]
    else:
        instructions = [
            "You are a question answering assistant.",
            "",
            "Answer the question using only the provided context.",
            "Cite supporting chunks with [Chunk N] after each factual claim.",
            'If the context does not contain enough information, say: "I don\'t know based on the provided context."',
        ]
    text = "\n".join(
        [
            *instructions,
            "",
            "Question:",
            question,
            "",
            "Context:",
            context.text,
            "",
            "Answer:",
        ]
    )
    # 提示词 hash 写入结果，便于确认两次运行是否使用了完全相同的提示词。
    return PromptPackage(
        text=text,
        template=version,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
