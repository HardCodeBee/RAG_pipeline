"""构建、查询与评估的可复现规格及源码指纹。"""

from __future__ import annotations

# Reproducibility identities and source boundaries are structural concerns.

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

from src.config import resolve_path


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value canonically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally without materializing it in memory."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    # sort_keys + 紧凑 separators 保证同一个 JSON 值总是得到同一个 hash。
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def recorded_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回可写入运行 metadata 的配置，排除运行凭据。"""
    value = json.loads(json.dumps(config, ensure_ascii=False))
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _hash_files(root: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    # 先去重再排序，保证不同文件匹配顺序不会影响源码 hash。
    unique = sorted({path.resolve() for path in files if path.is_file()}, key=lambda path: path.as_posix())
    for path in unique:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        relative_bytes = relative.encode("utf-8")
        content = path.read_bytes()
        # 把路径长度、路径、内容长度、内容都写入 hash，避免简单拼接造成边界歧义。
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()

_ENCODED_CORPUS_SOURCE_PATTERNS = (
    "src/persistence/artifact_io.py",
    "src/persistence/artifact_validation.py",
    "src/config.py",
    "src/records.py",
    "src/embedders/*.py",
    "src/encoded_corpus_builder.py",
    "src/persistence/encoded_corpus_writer.py",
    "src/loaders/*.py",
    "src/model_backends/*.py",
    "src/provenance.py",
    "src/text/*.py",
)

_SPARSE_INDEX_SOURCE_PATTERNS = (
    "src/config.py",
    "src/records.py",
    "src/provenance.py",
    "src/persistence/artifact_io.py",
    "src/persistence/artifact_validation.py",
    "src/retrievers/bm25_index.py",
    "src/retrievers/bm25_retriever.py",
    "src/retrievers/chunk_store.py",
)


_SOURCE_GROUP_PATTERNS = {
    "encoded_corpus": _ENCODED_CORPUS_SOURCE_PATTERNS,
    "sparse_index": _SPARSE_INDEX_SOURCE_PATTERNS,
    # Every build materializes one encoded corpus, so encoded-corpus code is
    # part of the build dependency closure even when a cached copy is reused.
    "build": (
        *_ENCODED_CORPUS_SOURCE_PATTERNS,
        "src/index_builder.py",
        "src/vector_index_factory.py",
        "src/indexes/*.py",
    ),
    "run": (
        "src/persistence/artifact_io.py",
        "src/persistence/artifact_validation.py",
        "src/config.py",
        "src/records.py",
        "src/embedders/*.py",
        "src/generators/*.py",
        "src/vector_index_factory.py",
        "src/indexes/*.py",
        "src/loaders/*.py",
        "src/model_backends/*.py",
        "src/pipeline.py",
        "src/prompts/*.py",
        "src/provenance.py",
        "src/query_runtime_factory.py",
        "src/context_builders/*.py",
        "src/query_plan.py",
        "src/rerankers/*.py",
        "src/retrievers/*.py",
        "src/text/token_counters.py",
    ),
    "evaluation": (
        "scripts/cli_support.py",
        "scripts/run_beir_eval.py",
        "scripts/run_beir_suite.py",
        "src/persistence/artifact_io.py",
        "src/persistence/artifact_validation.py",
        "src/evaluation_runner.py",
        "src/evaluators/*.py",
        "src/loaders/beir_loader.py",
        "src/provenance.py",
        "src/persistence/run_output_writer.py",
    ),
}


def source_group_sha256(project_root: str | Path, group: str) -> str:
    """Hash the explicit code boundary for build, run, or evaluation."""

    if group not in _SOURCE_GROUP_PATTERNS:
        raise ValueError(f"Unknown source group: {group}")
    root = Path(project_root).resolve()
    files = [
        path
        for pattern in _SOURCE_GROUP_PATTERNS[group]
        for path in root.glob(pattern)
    ]
    return _hash_files(root, files)


def source_files_sha256(project_root: str | Path, relative_paths: Iterable[str]) -> str:
    root = Path(project_root).resolve()
    return _hash_files(root, [root / path for path in relative_paths])

# 把每个语料文件转换成一条身份记录，
# 再把所有记录组合成一个 corpus 级别的清单，
# 并为整个清单生成总 hash
def corpus_inventory(documents: list[Path], corpus_root: Path) -> dict[str, Any]:
    # 语料清单只记录文件级信息，不读取文件内容。
    # 文本抽取差异由 source/config/backend 信息共同控制。
    rows = [
        {
            "source": path.name,
            "relative_path": path.relative_to(corpus_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in documents
    ]
    return {"aggregate_sha256": json_sha256(rows)}


def _artifact_packages(
    identity: dict[str, Any],
    *,
    include_index: bool,
) -> tuple[str, ...]:
    packages = {"numpy"}
    if identity["embedding"]["backend"] == "sentence_transformers":
        packages.update({"sentence_transformers", "torch", "transformers"})
    elif identity["embedding"]["backend"] == "hf_dense":
        packages.update({"torch", "transformers"})
    if include_index and identity["index"]["backend"] == "faiss":
        packages.add("faiss")
    return tuple(sorted(packages))


def _document_embedding_config(config: dict[str, Any]) -> dict[str, Any]:
    embedding = dict(recorded_config(config)["embedding"])
    for key in (
        "query_prefix",
        "query_model_name",
        "query_revision",
        "local_files_only",
    ):
        embedding.pop(key, None)
    return embedding


def encoded_corpus_spec(
    config: dict[str, Any],
    corpus: dict[str, Any],
    encoded_corpus_source_sha256: str,
) -> dict[str, Any]:
    """Describe reusable chunks and document embeddings, excluding index choices."""

    identity = recorded_config(config)
    embedding = _document_embedding_config(config)
    return {
        "embedding": embedding,
        "corpus": corpus,
        "producer_environment": producer_environment(
            *_artifact_packages(identity, include_index=False)
        ),
        "encoded_corpus_source_sha256": encoded_corpus_source_sha256,
    }


def encoded_corpus_identity(
    config: dict[str, Any],
    corpus: dict[str, Any],
    encoded_corpus_source_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    spec = encoded_corpus_spec(config, corpus, encoded_corpus_source_sha256)
    digest = json_sha256(spec)
    return f"encoded_corpus_{digest[:16]}", digest, spec

# 建立 Build identity
# 对应：indexing 
# 记录：
#   loader
#   chunking
#   embedding
#   index
#   corpus 文件清单
#   source_sha256
# 用于判断：
#   当前索引是否需要重建？
#   已有 artifacts/build_xxx 能不能复用？
def build_spec(
    config: dict[str, Any],
    corpus: dict[str, Any],
    build_source_sha256: str,
) -> dict[str, Any]:
    # 构建规格只包含会影响构建产物的字段。
    # query_prefix、local_files_only 等运行时或环境字段不会改变已构建索引内容。
    identity = recorded_config(config)
    embedding = _document_embedding_config(config)
    return {
        "embedding": embedding,
        "index": identity["index"],
        "corpus": corpus,
        "producer_environment": producer_environment(
            *_artifact_packages(identity, include_index=True)
        ),
        "build_source_sha256": build_source_sha256,
    }


def build_identity(
    config: dict[str, Any],
    corpus: dict[str, Any],
    build_source_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    spec = build_spec(config, corpus, build_source_sha256)
    digest = json_sha256(spec)
    # build_id 是 build_spec 的短 hash，目录名稳定且可读。
    return f"build_{digest[:16]}", digest, spec

# 建立 Run identity
# 对应： retrieval + context + generation
# 记录：
#   build_id
#   query embedding 配置
#   retrieval 配置，如 top_k
#   context 配置
#   prompt 配置
#   generation 配置
#   source_sha256
# 用于判断：
# 两次 query/run 的条件是否相同？
# 结果是否可以直接比较？
def run_spec(
    config: dict[str, Any],
    build_id: str,
    run_source_sha256: str,
) -> dict[str, Any]:
    # run_spec 描述 query 阶段会影响答案的配置。
    # 它和 build_spec 分开，避免每次改 top_k 或 generation 参数都重建索引。
    value = recorded_config(config)
    retrieval = value["retrieval"]
    embedding = value["embedding"]
    if embedding["backend"] == "hf_dense":
        query_embedding = {
            "backend": embedding["backend"],
            "family": embedding["family"],
            "model_name": embedding.get("query_model_name", embedding["model_name"]),
            "revision": embedding.get("query_revision", embedding["revision"]),
            "pooling": embedding["pooling"],
            "normalize": embedding["normalize"],
            "query_prefix": embedding["query_prefix"],
            "max_sequence_length": embedding["max_sequence_length"],
        }
    else:
        query_embedding = {
            key: embedding.get(key)
            for key in (
                "backend",
                "model_name",
                "revision",
                "normalize",
                "query_prefix",
                "max_sequence_length",
            )
        }
    result = {
        "build_id": build_id,
        "query_embedding": (
            query_embedding
            if retrieval["method"] == "dense"
            else None
        ),
        "retrieval": retrieval,
        "context": value["context"],
        "prompt": value["prompt"],
        "generation": value["generation"],
        "run_source_sha256": run_source_sha256,
    }
    if retrieval["method"] == "bm25":
        result["bm25"] = value["bm25"]
    return result

# 建立 Evaluation identity
# 对应： evaluation 阶段
# 记录：
#   questions_sha256
#   source_sha256
#   metrics_version
# 用于判断：
# 是不是同一套题？
# 是不是同一版评估指标？
# 能不能 resume 或 recompute metrics？
def evaluation_spec(
    questions_sha256: str,
    evaluation_source_sha256: str,
    *,
    metrics_version: str,
) -> dict[str, Any]:
    # 评估规格用来标识一次评估的题集版本、源码版本和指标版本。
    return {
        "questions_sha256": questions_sha256,
        "evaluation_source_sha256": evaluation_source_sha256,
        "metrics_version": metrics_version,
    }


def producer_environment(*package_names: str) -> dict[str, Any]:
    """Return the environment subset that can change persisted numerical artifacts."""

    distributions = {
        "faiss": "faiss-cpu",
        "pyyaml": "PyYAML",
        "sentence_transformers": "sentence-transformers",
    }
    selected = {}
    for name in package_names:
        try:
            selected[name] = importlib.metadata.version(distributions.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            selected[name] = None
    torch_runtime: dict[str, Any] | None = None
    if "torch" in package_names and selected.get("torch") is not None:
        try:
            import torch

            torch_runtime = {
                "cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "cuda_capability": (
                    list(torch.cuda.get_device_capability(0))
                    if torch.cuda.is_available()
                    else None
                ),
            }
        except (ImportError, RuntimeError):
            torch_runtime = {"unavailable": True}
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": selected,
        "torch_runtime": torch_runtime,
    }


def resolved_roots(config: dict[str, Any]) -> dict[str, Path]:
    # 把配置里的相对路径统一解析成绝对 Path，后续阶段只处理 Path 对象。
    return {
        key: resolve_path(config, config["paths"][key])
        for key in ("corpus", "artifacts_root", "outputs_root")
    }
