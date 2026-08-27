"""把用户问题 + 检索得到的 context 拼成一个固定版本的问答 prompt，并给这个 prompt 生成 hash 方便复现"""

from __future__ import annotations

import hashlib

from src.records import ContextPackage, PromptPackage


HOTPOT_SHORT_ANSWER_VERSION = "hotpot_short_answer_v1"
HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION = "hotpot_multihop_short_answer_v2"
SUPPORTED_PROMPT_VERSIONS = {
    HOTPOT_SHORT_ANSWER_VERSION,
    HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION,
}


def _package(text: str, version: str) -> PromptPackage:
    return PromptPackage(
        text=text,
        template=version,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def build_prompt(
    question: str,
    context: ContextPackage,
    version: str = HOTPOT_SHORT_ANSWER_VERSION,
) -> PromptPackage:
    # prompt 版本固定后，实验结果可以追溯到具体模板文本。
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported prompt version: {version}")
    if version == HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION:
        text = "\n".join(
            [
                "Context information is below.",
                "---------------------",
                context.text,
                "---------------------",
                "",
                "Given the context information and no prior knowledge, answer the question below.",
                "The answer may require combining evidence from multiple context passages.",
                "Resolve all parts of the question before answering.",
                'If the context does not contain enough information, answer exactly: "I don\'t know based on the provided context."',
                "Return only the minimal final answer span needed, such as an entity, date, number, short phrase, or yes/no.",
                "Do not add citations, explanation, or reasoning.",
                "",
                "Question:",
                question.strip(),
                "",
                "Answer:",
            ]
        )
        return _package(text, version)
    instructions = [
        "Answer the question using only the provided context.",
        "Return only the short answer. Do not add citations or explanation.",
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
    return _package(text, version)
