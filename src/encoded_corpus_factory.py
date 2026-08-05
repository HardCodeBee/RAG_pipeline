"""Factories used to construct an encoded corpus."""

from __future__ import annotations

# Keep encoded-corpus assembly separate from query-only component assembly.

from pathlib import Path
from typing import Any

from src.embedders.text_embedder import TextEmbedder
from src.loaders.dpr_wikipedia_loader import DprWikipediaCorpusLoader
from src.loaders.qasper_loader import QasperCorpusLoader
from src.provenance import corpus_inventory
from src.text.token_counters import HuggingFaceTokenCounter, RegexTokenCounter


def create_loader(config: dict[str, Any]):
    loader = config["loader"]
    if loader["type"] == "dpr_wikipedia":
        return DprWikipediaCorpusLoader(
            expected_protocol=loader["expected_protocol"],
            text_format=loader["text_format"],
            require_canonical_counts=loader["require_canonical_counts"],
        )
    if loader["type"] == "qasper":
        return QasperCorpusLoader(
            split=loader["split"],
            max_documents=loader["max_documents"],
        )
    raise ValueError(f"Unsupported loader: {loader['type']}")


def discover_corpus(loader: Any, corpus_path: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Discover files and reuse a loader's verified identity inputs when available."""

    root = Path(corpus_path)
    documents = loader.discover(root)
    inventory_builder = getattr(loader, "corpus_inventory", None)
    inventory = (
        inventory_builder(root)
        if callable(inventory_builder)
        else corpus_inventory(documents, root)
    )
    return documents, inventory


def create_token_counter(config: dict[str, Any]):
    chunking = config["chunking"]
    if chunking["tokenizer"] == "regex":
        return RegexTokenCounter()
    if chunking["tokenizer"] == "huggingface":
        return HuggingFaceTokenCounter(
            model_name=chunking["tokenizer_model"],
            revision=chunking["tokenizer_revision"],
            local_files_only=chunking["local_files_only"],
        )
    raise ValueError(f"Unsupported tokenizer: {chunking['tokenizer']}")


def create_chunker(config: dict[str, Any], token_counter):
    # Chunking implementations are build-only dependencies. Import them here
    # so query-time factory imports do not load the chunking subsystem.
    from src.chunkers.fixed_sentence_chunker import FixedSentenceChunker
    from src.chunkers.presegmented_chunker import PresegmentedChunker
    from src.text.sentence_splitter import RegexSentenceSplitter

    chunking = config["chunking"]
    if chunking["strategy"] == "presegmented":
        return PresegmentedChunker(
            token_counter=token_counter,
            chunk_size_tokens=chunking["chunk_size_tokens"],
            chunk_overlap_tokens=chunking["overlap_budget_tokens"],
        )
    return FixedSentenceChunker(
        sentence_splitter=RegexSentenceSplitter(),
        token_counter=token_counter,
        chunk_size_tokens=chunking["chunk_size_tokens"],
        chunk_overlap_tokens=chunking["overlap_budget_tokens"],
    )


def create_embedder(
    config: dict[str, Any],
    *,
    override: dict[str, Any] | None = None,
):
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
        device=embedding.get("device", "auto"),
    )
