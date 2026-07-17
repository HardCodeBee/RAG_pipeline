from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from src.core.records import ChunkRecord, PageRecord
from src.text.token_counters import validate_token_window


# 一个 unit 是 chunker 内部使用的最小拼装单元：(文本片段, 所在页码, 片段 token 数)。
# 一般情况下一个 unit 是一句话；如果一句话超过 chunk_size_tokens，
# 它会先被 token_counter.split() 切成更小的片段。
Unit = tuple[str, int, int]


# 把输入统一成 PageRecord，方便后面的代码只处理一种结构。
# loader、测试或 JSONL 读取结果可能传入普通 dict，所以这里做一次兼容转换。
def _page_record(value: PageRecord | Mapping[str, Any]) -> PageRecord:
    return value if isinstance(value, PageRecord) else PageRecord.from_mapping(value)


# 把一堆 page 记录按 doc_id 分组：属于同一篇文档的页面放到同一个 list 里。
def _group_pages(records: Sequence[PageRecord | Mapping[str, Any]]) -> dict[str, list[PageRecord]]:
    grouped: dict[str, list[PageRecord]] = defaultdict(list)
    for value in records:
        record = _page_record(value)
        grouped[record.doc_id].append(record)
    return grouped


# 检查 chunk_size_tokens 和 chunk_overlap_tokens 是否合法。
def _validate_budgets(size: int, overlap: int) -> None:
    validate_token_window(
        size,
        overlap,
        size_name="chunk_size_tokens",
        overlap_name="chunk_overlap_tokens",
    )


# 把若干 unit 合成一个 ChunkRecord。
# 这个函数只做“记录构造”：计算页码范围、拼接文本、统计 token、生成 chunk_id。
def _record_chunk(
    doc_id: str,
    source: str,
    chunk_index: int,
    vector_id: int,
    units: list[Unit],
) -> ChunkRecord:
    page_start = min(unit[1] for unit in units)
    page_end = max(unit[1] for unit in units)
    # chunk_id 中带 page_start 和文档内序号，便于人工定位来源。
    chunk_id = f"{doc_id}_p{page_start}_c{chunk_index:04d}"
    return ChunkRecord(
        chunk_id=chunk_id,
        vector_id=vector_id,
        doc_id=doc_id,
        source=source,
        page_start=page_start,
        page_end=page_end,
        text=" ".join(unit[0] for unit in units).strip(),
        token_count=sum(unit[2] for unit in units),
    )


class FixedSentenceChunker:
    # chunking 策略：按句子为基本单位，在固定 token 预算内组装 chunk，
    # 相邻 chunk 之间保留一段尾部 overlap，降低边界切断语义的风险。
    name = "fixed_sentence"

    def __init__(
        self,
        sentence_splitter,
        token_counter,
        chunk_size_tokens: int = 300,
        chunk_overlap_tokens: int = 50,
    ):
        _validate_budgets(chunk_size_tokens, chunk_overlap_tokens)
        self.sentence_splitter = sentence_splitter
        self.token_counter = token_counter
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    # 把一页 PageRecord 拆成一组最小可拼装单元 unit。
    def _units(self, record: PageRecord) -> list[Unit]:
        units = []
        for sentence in self.sentence_splitter.split(record.text):
            sentence_tokens = self.token_counter.count(sentence)
            # 如果当前句子的 token 数没有超过 chunk 最大限制：直接保留整个句子。
            # 如果这个句子超过了 chunk_size_tokens：继续把句子拆成更小片段。
            fragments = (
                [sentence]
                if sentence_tokens <= self.chunk_size_tokens
                else self.token_counter.split(sentence, self.chunk_size_tokens)
            )
            # 对每个 fragment 重新计算 token，只保留 token 数大于 0 的片段。
            for fragment in fragments:
                count = self.token_counter.count(fragment)
                if count:
                    units.append((fragment, record.page, count))
        return units

    # 按页面顺序把一篇文档的所有页面展开成 unit 流。
    # 使用 Iterator 可以让主流程逐个消费 unit，不需要提前构造整篇文档的 unit 列表。
    def _document_units(self, pages: Sequence[PageRecord]) -> Iterator[Unit]:
        for page in pages:
            yield from self._units(page)

    # 从当前 chunk 的尾部挑出一部分 unit，作为下一个 chunk 的开头重叠内容。
    # 切块会制造边界，而语义经常跨边界
    def _overlap(self, units: list[Unit]) -> list[Unit]:
        selected = []
        total = 0
        # 从尾部开始遍历，优先保留最接近 chunk 结尾的上下文。
        for unit in reversed(units):
            # 把当前 unit 加进去之后超过 overlap 预算，就停止。
            if total + unit[2] > self.chunk_overlap_tokens:
                break
            selected.append(unit)
            total += unit[2]
        # reversed() 是从后往前选的，返回前要恢复原始阅读顺序。
        return list(reversed(selected))

    # 提交一个 chunk 之后，计算下一个 chunk 的初始 buffer。
    # emitted 是刚刚提交出去的 chunk；它的尾部 overlap 会成为下一个 chunk 的开头。
    def _next_buffer(self, emitted: list[Unit], incoming_tokens: int | None = None) -> tuple[list[Unit], int]:
        current = self._overlap(emitted)
        current_tokens = sum(unit[2] for unit in current)
        if incoming_tokens is not None:
            # 如果 overlap 加上即将进入的新 unit 会超过 chunk_size，
            # 就从 overlap 头部丢弃一些 unit，直到新 unit 放得下。
            while current and current_tokens + incoming_tokens > self.chunk_size_tokens:
                current_tokens -= current.pop(0)[2]
        return current, current_tokens

    @staticmethod
    # 调用 _record_chunk()
    # 创建一个 ChunkRecord，然后放入 chunks 列表
    def _append_chunk(
        chunks: list[ChunkRecord],
        doc_id: str,
        source: str,
        chunk_index: int,
        vector_id_start: int,
        units: list[Unit],
    ) -> None:
        # vector_id 是全局连续编号。
        # _chunk_document() 内部只知道本篇文档已经产生了多少 chunk，
        # 所以要用 vector_id_start 加上当前文档内的 len(chunks)。
        chunks.append(
            _record_chunk(
                doc_id,
                source,
                chunk_index,
                vector_id_start + len(chunks),
                units, )
        )

    # 顺序扫描一个document的所有 unit，然后把这些 unit 装进一个个 chunk，必要时带 overlap
    def _chunk_document(
        self,
        doc_id: str,
        pages: Sequence[PageRecord],
        vector_id_start: int,
    ) -> list[ChunkRecord]:
        # 保存该document最终切出来的所有 chunk
        chunks: list[ChunkRecord] = []

        # 同一个 doc_id 下的页面应该来自同一个 source；这里取第一页即可。
        source = pages[0].source

        # current 保存正在组装、尚未最终提交的 chunk。
        current: list[Unit] = []
        current_tokens = 0

        # 用来区分 current 里是否包含“新内容”，防止重复
        # 只有 overlap 但没有新 unit 时，不应该在文档末尾重复提交一次。
        current_contains_unemitted = False
        chunk_index = 1 #文档内部的chunk编号

        # 遍历文档里的所有 unit
        for unit in self._document_units(pages):
            # 新 unit 放不进当前 chunk 时，先提交当前 chunk。
            if current and current_tokens + unit[2] > self.chunk_size_tokens:
                self._append_chunk(chunks, doc_id, source, chunk_index, vector_id_start, current)
                chunk_index += 1

                # 提交后current保留尾部一部分作为 overlap，再准备接收新 unit。
                current, current_tokens = self._next_buffer(current, incoming_tokens=unit[2])
                current_contains_unemitted = False

            # 把当前 unit 放进 buffer，并更新 token 计数。
            current.append(unit)
            current_tokens += unit[2]
            current_contains_unemitted = True

            # 如果刚好达到预算上限，立即提交，避免继续等待下一个 unit。
            if current_tokens == self.chunk_size_tokens:
                self._append_chunk(chunks, doc_id, source, chunk_index, vector_id_start, current)
                chunk_index += 1
                current, current_tokens = self._next_buffer(current)
                current_contains_unemitted = False

        # 文档结束后，如果 buffer 中还有尚未提交的新内容，就补交最后一个 chunk。
        if current and current_contains_unemitted:
            self._append_chunk(chunks, doc_id, source, chunk_index, vector_id_start, current)
        return chunks

    # 主流程：把很多页 PageRecord 转成最终的 ChunkRecord 列表。
    def chunk(self, records: Sequence[PageRecord | Mapping[str, Any]]) -> list[ChunkRecord]:
        #把所有页面按 doc_id 分成不同组
        grouped = _group_pages(records)

        # 最终 chunks 列表
        chunks: list[ChunkRecord] = []

        # doc_id 排序保证输入顺序不同也能得到稳定输出，便于测试和复现实验。
        for doc_id in sorted(grouped):
            # 同一篇文档内部按页码排序，保证 chunk 文本保持原文阅读顺序。
            pages = sorted(grouped[doc_id], key=lambda item: item.page)
            # len(chunks) 是当前全局已经产生的 chunk 数，用作本篇文档的 vector_id 起点。
            chunks.extend(self._chunk_document(doc_id, pages, vector_id_start=len(chunks)))
        return chunks
