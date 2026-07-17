"""
retriever 和 prompt builder 之间的上下文组装层
构造基线使用的单一排序 context 格式。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src.core.records import ContextPackage, SearchHit

# 把检索出来的 chunks 拼成最终要放进 prompt 的 context
# 处理原理：
# 能完整放入的 chunk，依次放入
# 第一个放不下的 chunk 截断后放入 然后停止
def build_context(
    query: str,
    results: Sequence[SearchHit],
    token_counter,
    max_tokens: int | None,
) -> ContextPackage:
    # 检查 query 必须是非空字符串
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    # 要求 max_tokens 只能是 0 或 None
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer or None")

    # selected 保存实际进入 context 的检索命中；如果文本被截断，会同步替换文本块内容。
    selected = []
    # parts 是最终拼接进 prompt 的文本块，每个块都带来源头部。
    parts = []
    # 统计当前已经用了多少 token
    used = 0
    # 记录是否发生过截断
    truncated = False

    # 把检索结果一个个转成 prompt 里的 context 块，同时控制 token 预算。
    for result in results:
        # 构造 chunk 头部:头部明确标出排序、来源和页码，便于生成答案时引用来源。
        header = (
            f"[Chunk {result.rank} | source={result.chunk.source}, "
            f"pages={result.chunk.page_start}-{result.chunk.page_end}]"
        )
        #计算 header token 数
        header_tokens = token_counter.count(header)
        # 取出 chunk 正文 并 计算正文 token 数
        text = result.chunk.text
        text_tokens = token_counter.count(text)
        # 如果设置了 max_tokens，开始预算控制
        if max_tokens is not None:
            # 预算要同时覆盖头部和文本块正文。
            remaining = max_tokens - used - header_tokens
            # header 后的无正文空间
            if remaining <= 0:
                truncated = True
                break
            # 正文空间太长，超出预算
            if text_tokens > remaining:
                # 当前 chunk 放不下时，只截断当前 chunk，不再继续加入后面的 chunk。
                text = token_counter.truncate(text, remaining)
                text_tokens = token_counter.count(text)
                truncated = True
        # 如果 text 为空，停止
        if not text:
            break
        # dataclass.replace 保持其他字段不变，只替换进入 context 的文本和 token_count。
        # 创建一个新的 ChunkRecord 用于进入context， 替换其中text token_count
        chunk = replace(result.chunk, text=text, token_count=text_tokens)
        # 创建新的SearchHit: 实际进入 context 的检索命中结果
        selected.append(replace(result, chunk=chunk))
        # 保存最终 prompt 里的一段 context 文本
        parts.append(f"{header}\n{text}")
        # 更新已经使用的 token 数
        used += header_tokens + text_tokens
        if truncated:
            break

    # context 包是 prompt 构造器的唯一输入格式，避免 prompt 直接依赖 retriever 细节。
    return ContextPackage(
        text="\n\n".join(parts),
        results=tuple(selected),
        token_count=used,
        truncated=truncated,
        builder="ranked_concat_v1",
    )
