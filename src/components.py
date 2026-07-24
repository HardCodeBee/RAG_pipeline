"""基线研究组件的显式构造函数集合。"""

from __future__ import annotations

from typing import Any

from src.chunkers.modular_chunker import FixedSentenceChunker
from src.embedders.sbert_embedder import TextEmbedder
from src.generators.llm_generator import LLMGenerator
from src.indexes.faiss_index import FlatIPIndex
from src.loaders.corpus_loaders import PypdfCorpusLoader
from src.loaders.qasper_loader import QasperCorpusLoader
from src.text.splitters import RegexSentenceSplitter
from src.text.token_counters import HuggingFaceTokenCounter, RegexTokenCounter


def create_loader(config: dict[str, Any]):
    loader = config["loader"]
    if loader["type"] == "qasper":
        return QasperCorpusLoader(
            split=loader["split"],
            max_documents=loader["max_documents"],
        )
    # 组件工厂只接受 validate_config() 之后的配置，因此这里主要做后备保护。
    if loader["type"] != "pypdf":
        raise ValueError(f"Unsupported loader: {loader['type']}")
    # 当前基线只实现 pypdf 加载器；后续换加载器可以在这里扩展。
    return PypdfCorpusLoader(
        recursive=loader["recursive"],
        empty_page_policy=loader["empty_page_policy"],
    )


def create_token_counter(config: dict[str, Any]):
    chunking = config["chunking"]
    # 正则 tokenizer 是轻量默认值，适合冒烟测试和无模型环境。
    if chunking["tokenizer"] == "regex":
        return RegexTokenCounter()
    # 外部模型 tokenizer 用于更接近真实模型词元边界的实验。
    if chunking["tokenizer"] == "huggingface":
        return HuggingFaceTokenCounter(
            model_name=chunking["tokenizer_model"],
            revision=chunking["tokenizer_revision"],
            local_files_only=chunking["local_files_only"],
        )
    raise ValueError(f"Unsupported tokenizer: {chunking['tokenizer']}")


def create_chunker(config: dict[str, Any], token_counter):
    chunking = config["chunking"]
    # 分句器和词元计数器解耦，便于替换分句策略或 tokenizer。
    return FixedSentenceChunker(
        sentence_splitter=RegexSentenceSplitter(),
        token_counter=token_counter,
        chunk_size_tokens=chunking["chunk_size_tokens"],
        chunk_overlap_tokens=chunking["overlap_budget_tokens"],
    )


def create_embedder(config: dict[str, Any], *, override: dict[str, Any] | None = None):
    # 查询阶段会用 manifest 中的 embedding 信息覆盖配置，确保和构建阶段同空间。
    embedding = dict(config["embedding"])
    embedding.update(override or {})
    return TextEmbedder(
        backend=embedding["backend"],
        model_name=embedding.get("model_name"),
        revision=embedding.get("revision"),
        normalize=embedding["normalize"],
        batch_size=embedding.get("batch_size", 32),
        dimension=embedding.get("dimension", 384),
        query_prefix=embedding["query_prefix"],
        document_prefix=embedding["document_prefix"],
        max_sequence_length=embedding.get("max_sequence_length"),
        local_files_only=embedding.get("local_files_only", False),
    )


def create_index(
    config: dict[str, Any],
    *,
    backend: str | None = None,
):
    return FlatIPIndex(backend=backend or config["index"]["backend"])


def create_generator(config: dict[str, Any]):
    generation = config["generation"]
    return LLMGenerator(
        provider=generation["provider"],
        model=generation.get("model"),
        temperature=generation.get("temperature", 0.0),
        max_output_tokens=generation["max_output_tokens"],
        timeout_seconds=generation.get("timeout_seconds", 60.0),
        max_retries=generation.get("max_retries", 0),
    )
