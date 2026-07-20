from __future__ import annotations

from pathlib import Path

from src.core.records import PageRecord
from src.io_utils import slugify
from src.text.cleaners import clean_text


class _PDFCorpusLoader:
    # PDF 加载器的公共基类：负责文件发现和文档身份校验。
    def __init__(
        self,
        recursive: bool = False,
        empty_page_policy: str = "skip",
    ):
        if empty_page_policy not in {"skip", "error"}:
            raise ValueError("empty_page_policy must be one of: error, skip")
        # recursive 控制是否递归扫描语料子目录。
        self.recursive = bool(recursive)
        # empty_page_policy 决定空白页是跳过还是直接报错。
        self.empty_page_policy = empty_page_policy

    # 按照文件的后缀 简单筛选pdf文件
    def discover(self, corpus_path: str | Path, file_type: str = "pdf") -> list[Path]:

        if str(file_type).casefold() != "pdf":
            raise ValueError("PDF loaders only support file_type=pdf")

        root = Path(corpus_path)
        if not root.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Corpus path is not a directory: {root}")
        candidates = root.rglob("*") if self.recursive else root.iterdir()
        #必须是文件；文件后缀必须是 .pdf
        documents = [path for path in candidates if path.is_file() and path.suffix.casefold() == ".pdf"]
        # 按相对路径排序，保证不同文件系统返回顺序不同也能稳定构建。
        return sorted(
            documents,
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )

    # 给每个 PDF 生成唯一的 doc_id，并检查语料库里不能有身份冲突的文件
    def _validate_identities(self, documents: list[Path]) -> dict[Path, str]:
        # doc_id 用于 chunk_id
        # source 用于展示和评估；
        # 两者都需要在 corpus 内唯一。
        doc_ids: dict[str, Path] = {}
        sources: dict[str, Path] = {}
        result = {}
        for path in documents:
            doc_id = slugify(path.stem) #对不带扩展名的文件名生成ID字符串后缀
            # 检查 doc_id 是否重复
            doc_key = doc_id.casefold()
            if doc_key in doc_ids:
                raise ValueError(f"Duplicate document id for {doc_ids[doc_key]} and {path}: {doc_id}")
            # 检查文件名 source 是否重复
            source_key = path.name.casefold()
            if source_key in sources:
                raise ValueError(f"Duplicate source name for {sources[source_key]} and {path}: {path.name}")

            doc_ids[doc_key] = path
            sources[source_key] = path
            result[path] = doc_id
        return result

# PypdfCorpusLoader 是真正执行 PDF 文本读取的类
class PypdfCorpusLoader(_PDFCorpusLoader):
    # 使用 pypdf 抽取每页文本，并输出 PageRecord。
    def load(self, corpus_path: str | Path, file_type: str = "pdf") -> list[PageRecord]:
        # 运行时才导入 pypdf
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF loading requires pypdf; install requirements/base.txt") from exc

        documents = self.discover(corpus_path, file_type) #找出所有 PDF 文件
        identities = self._validate_identities(documents)  #给每个 PDF 生成 doc_id，并检查重复
        records = [] # 准备存放每一页的文本记录

        for path in documents:
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                # 每页先做最小清洗，再进入 chunker。
                text = clean_text(page.extract_text() or "")
                # 清洗后没有某页文本
                if not text:
                    if self.empty_page_policy == "error":
                        raise ValueError(f"PDF page contains no extractable text: {path} page {page_number}")
                    continue
                records.append(
                    PageRecord(
                        doc_id=identities[path],
                        source=path.name,
                        page=page_number,
                        text=text,
                    )
                )
        return records
