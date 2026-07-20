"""为构建、查询运行和评估生成可复现、可比较的身份信息
    Build identity      : 修改是否需要重建索引？ 
    Run identity        : 运行是否可以直接比较？
    Evaluation identity : 实验是否可以安全恢复？
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from src.config import resolve_path
from src.io_utils import sha256_file


def json_sha256(value: Any) -> str:
    # sort_keys + 紧凑 separators 保证同一个 JSON 值总是得到同一个 hash。
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recorded_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回可写入运行 metadata 的完整配置。"""
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

#  对所有有效 Python 源码求 hash，避免维护分阶段文件清单。
def source_code_sha256(project_root: str | Path) -> str:

    root = Path(project_root).resolve()
    # src 和 scripts 都会影响构建或查询行为，所以一起纳入 source_sha256。
    files = [*root.glob("src/**/*.py"), *root.glob("scripts/*.py")]
    return _hash_files(root, files)

# 把每个语料文件转换成一条身份记录，
# 再把所有记录组合成一个 corpus 级别的清单，
# 并为整个清单生成总 hash
def corpus_inventory(documents: list[Path], corpus_root: Path) -> dict[str, Any]:
    # 语料清单只记录文件级信息，不读取 PDF 文本内容。
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
    return {"documents": rows, "aggregate_sha256": json_sha256(rows)}

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
def build_spec(config: dict[str, Any], corpus: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    # 构建规格只包含会影响构建产物的字段。
    # query_prefix、local_files_only 等运行时或环境字段不会改变已构建索引内容。
    identity = recorded_config(config)
    embedding = dict(identity["embedding"])
    embedding.pop("query_prefix", None)
    embedding.pop("local_files_only", None)
    chunking = dict(identity["chunking"])
    chunking.pop("local_files_only", None)
    return {
        "loader": identity["loader"],
        "chunking": chunking,
        "embedding": embedding,
        "index": identity["index"],
        "corpus": corpus,
        "source_sha256": source_sha256,
    }


def build_identity(config: dict[str, Any], corpus: dict[str, Any], source_sha256: str) -> tuple[str, str, dict[str, Any]]:
    spec = build_spec(config, corpus, source_sha256)
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
def run_spec(config: dict[str, Any], build_id: str, source_sha256: str) -> dict[str, Any]:
    # run_spec 描述 query 阶段会影响答案的配置。
    # 它和 build_spec 分开，避免每次改 top_k 或 generation 参数都重建索引。
    value = recorded_config(config)
    embedding = value["embedding"]
    return {
        "strict_backends": value["strict_backends"],
        "build_id": build_id,
        "query_embedding": {
            key: embedding.get(key)
            for key in ("backend", "model_name", "revision", "normalize", "query_prefix", "max_sequence_length")
        },
        "retrieval": value["retrieval"],
        "context": value["context"],
        "prompt": value["prompt"],
        "generation": {
            key: value["generation"][key]
            for key in (
                "provider",
                "model",
                "temperature",
                "max_output_tokens",
                "timeout_seconds",
                "max_retries",
            )
        },
        "source_sha256": source_sha256,
    }

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
def evaluation_spec(questions_sha256: str, source_sha256: str) -> dict[str, Any]:
    # 评估规格用来标识一次评估的题集版本、源码版本和指标版本。
    return {
        "questions_sha256": questions_sha256,
        "source_sha256": source_sha256,
        "metrics_version": "evidence_and_generation_v3",
    }


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def run(*args: str) -> str:
        # 通过 subprocess 调 git，失败时整体退化为 unknown，不阻塞 pipeline。
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        # zip 包、无 git 环境或 CI 限制下可能没有 git 信息。
        return {"commit": None, "dirty": None}


def environment_versions() -> dict[str, Any]:
    # 记录关键依赖版本，方便解释不同机器上的构建差异。
    packages = {
        "faiss": "faiss-cpu",
        "numpy": "numpy",
        "openai": "openai",
        "pypdf": "pypdf",
        "pyyaml": "PyYAML",
        "sentence_transformers": "sentence-transformers",
        "torch": "torch",
        "transformers": "transformers",
    }
    versions = {}
    for key, distribution in packages.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            # 可选依赖未安装是合法状态，例如使用哈希或 NumPy 回退实现。
            versions[key] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def resolved_roots(config: dict[str, Any]) -> dict[str, Path]:
    # 把配置里的相对路径统一解析成绝对 Path，后续阶段只处理 Path 对象。
    return {
        key: resolve_path(config, config["paths"][key])
        for key in ("corpus", "artifacts_root", "outputs_root")
    }
