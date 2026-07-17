"""把文档文本切成检索单元的 chunking 策略。"""

from src.chunkers.modular_chunker import FixedSentenceChunker

__all__ = ["FixedSentenceChunker"]
