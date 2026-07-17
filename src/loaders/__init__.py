"""读取原始语料文件的文档 loader。"""

# 对外暴露当前 PDF 语料 loader。
from src.loaders.corpus_loaders import PypdfCorpusLoader

__all__ = ["PypdfCorpusLoader"]
