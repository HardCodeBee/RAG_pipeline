"""加载、校验并解析紧凑实验配置。"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


_ROOT_KEYS = {
    "paths",
    "loader",
    "chunking",
    "embedding",
    "index",
    "retrieval",
    "context",
    "prompt",
    "generation",
    "logging",
    "_base_dir",
}


def _mapping(value: Any, location: str) -> dict[str, Any]:
    # 配置文件中的每个区块都应该是字典；location 用来生成可读错误信息。
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a mapping")
    return value


def _unknown(section: dict[str, Any], allowed: set[str], location: str) -> None:
    # 拒绝未知字段，避免拼写错误被静默忽略。
    extra = sorted(set(section) - allowed)
    if extra:
        raise ValueError(f"Unknown {location} config keys: {', '.join(extra)}")


def _text(value: Any, location: str) -> str:
    # 文本配置统一去掉两侧空白，避免路径或模型名受空白影响。
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{location} must be a boolean")
    return value


def _integer(value: Any, location: str, minimum: int = 1) -> int:
    # bool 是 int 的子类，所以需要显式排除 True/False。
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer")
    if value < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return value


def _number(value: Any, location: str, minimum: float, maximum: float) -> float:
    # 非数字值或无穷值会破坏 JSON 清单和指标统计，所以必须拒绝。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{location} must be between {minimum} and {maximum}")
    return result


def _choice(value: Any, choices: set[str], location: str) -> str:
    # 所有枚举配置都转成小写比较，减少 YAML 大小写差异。
    result = _text(value, location).casefold()
    if result not in choices:
        raise ValueError(f"{location} must be one of: {', '.join(sorted(choices))}")
    return result


def _reject_inline_secrets(value: Any, location: str = "config") -> None:
    # Credentials belong in the process environment, never in experiment config.
    secret_names = {"api_key", "authorization", "password", "secret", "token"}
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if (
                normalized in secret_names
                or normalized.endswith(("_api_key", "_password", "_secret"))
            ):
                raise ValueError(f"{location}.{key} must not contain an inline secret")
            _reject_inline_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _reject_inline_secrets(item, f"{location}[{position}]")

# 校验配置结构是否类型与范围合理 并对不同组件的配置进行区分
# 进行默认值补充
# 把外部传进来的配置变成 pipeline 可以安全使用的“标准配置对象”
def validate_config(config: dict[str, Any]) -> dict[str, Any]:

    # 深拷贝后再补默认值，避免调用者传入的原始 dict 被就地修改。
    value = copy.deepcopy(_mapping(config, "config"))
    _reject_inline_secrets(value)
    _unknown(value, _ROOT_KEYS, "root")

    # paths 是构建和查询都要用的三个根目录，保持为字符串，使用时再解析。
    paths = _mapping(value.get("paths"), "paths")
    _unknown(paths, {"corpus", "artifacts_root", "outputs_root"}, "paths")
    for key in ("corpus", "artifacts_root", "outputs_root"):
        paths[key] = _text(paths.get(key), f"paths.{key}")

    # loader 控制如何发现和读取 corpus 文件。
    loader = _mapping(value.setdefault("loader", {}), "loader")
    loader["type"] = _choice(loader.get("type", "pypdf"), {"pypdf", "qasper"}, "loader.type")
    if loader["type"] == "qasper":
        _unknown(loader, {"type", "split", "max_documents"}, "loader")
        loader["split"] = _choice(
            loader.get("split", "validation"),
            {"train", "validation", "test", "all"},
            "loader.split",
        )
        max_documents = loader.get("max_documents")
        loader["max_documents"] = (
            _integer(max_documents, "loader.max_documents")
            if max_documents is not None
            else None
        )
    else:
        _unknown(loader, {"type", "recursive", "empty_page_policy"}, "loader")
        loader["recursive"] = _boolean(loader.get("recursive", False), "loader.recursive")
        loader["empty_page_policy"] = _choice(
            loader.get("empty_page_policy", "skip"),
            {"error", "skip"},
            "loader.empty_page_policy",
        )
    # chunking 控制“页面文本 -> chunk”的策略和 token 预算。
    chunking = _mapping(value.get("chunking"), "chunking")
    chunking["tokenizer"] = _choice(
        chunking.get("tokenizer", "regex"),
        {"huggingface", "regex"},
        "chunking.tokenizer",
    )
    common_chunking_keys = {
        "chunk_size_tokens",
        "overlap_budget_tokens",
        "tokenizer",
    }
    if chunking["tokenizer"] == "huggingface":
        _unknown(
            chunking,
            common_chunking_keys | {"tokenizer_model", "tokenizer_revision", "local_files_only"},
            "chunking",
        )
    else:
        _unknown(chunking, common_chunking_keys, "chunking")
    chunking["chunk_size_tokens"] = _integer(
        chunking.get("chunk_size_tokens", 300),
        "chunking.chunk_size_tokens",
    )
    chunking["overlap_budget_tokens"] = _integer(
        chunking.get("overlap_budget_tokens", 50),
        "chunking.overlap_budget_tokens",
        minimum=0,
    )
    if chunking["overlap_budget_tokens"] >= chunking["chunk_size_tokens"]:
        raise ValueError("chunking.overlap_budget_tokens must be smaller than chunk_size_tokens")
    if chunking["tokenizer"] == "huggingface":
        # Model-backed tokenization must pin the exact tokenizer revision.
        chunking["tokenizer_model"] = _text(
            chunking.get("tokenizer_model"),
            "chunking.tokenizer_model",
        )
        chunking["tokenizer_revision"] = _text(
            chunking.get("tokenizer_revision"),
            "chunking.tokenizer_revision",
        )
        chunking["local_files_only"] = _boolean(
            chunking.get("local_files_only", False),
            "chunking.local_files_only",
        )

    # embedding 控制“chunk/query 文本 -> 向量”的 backend 和模型参数。
    embedding = _mapping(value.get("embedding"), "embedding")
    embedding["backend"] = _choice(
        embedding.get("backend"),
        {"hashing", "sentence_transformers"},
        "embedding.backend",
    )
    common_embedding_keys = {
        "backend",
        "normalize",
        "query_prefix",
        "document_prefix",
    }
    if embedding["backend"] == "hashing":
        _unknown(embedding, common_embedding_keys | {"dimension"}, "embedding")
        embedding["dimension"] = _integer(
            embedding.get("dimension", 384),
            "embedding.dimension",
        )
    else:
        _unknown(
            embedding,
            common_embedding_keys
            | {
                "model_name",
                "revision",
                "batch_size",
                "max_sequence_length",
                "local_files_only",
            },
            "embedding",
        )
        embedding["model_name"] = _text(embedding.get("model_name"), "embedding.model_name")
        embedding["revision"] = _text(embedding.get("revision"), "embedding.revision")
        embedding["batch_size"] = _integer(embedding.get("batch_size", 32), "embedding.batch_size")
        max_sequence_length = embedding.get("max_sequence_length")
        embedding["max_sequence_length"] = (
            _integer(max_sequence_length, "embedding.max_sequence_length")
            if max_sequence_length is not None
            else None
        )
        embedding["local_files_only"] = _boolean(
            embedding.get("local_files_only", False),
            "embedding.local_files_only",
        )
    embedding["normalize"] = _boolean(embedding.get("normalize", True), "embedding.normalize")
    for key in ("query_prefix", "document_prefix"):
        item = embedding.get(key, "")
        if not isinstance(item, str):
            raise TypeError(f"embedding.{key} must be a string")
        # prefix 允许保留空字符串；不同 embedding 模型可能需要 query/document 前缀。
        embedding[key] = item
    # index 控制向量索引后端；当前只支持平铺内积索引。
    index = _mapping(value.get("index"), "index")
    _unknown(index, {"backend"}, "index")
    index["backend"] = _choice(index.get("backend"), {"faiss", "numpy"}, "index.backend")

    # retrieval 控制查询时召回策略和默认 top_k。
    retrieval = _mapping(value.setdefault("retrieval", {}), "retrieval")
    _unknown(retrieval, {"top_k"}, "retrieval")
    retrieval["top_k"] = _integer(retrieval.get("top_k", 5), "retrieval.top_k")

    # context 控制把召回 chunk 拼进 prompt 时的 token 上限。
    context = _mapping(value.setdefault("context", {}), "context")
    _unknown(context, {"max_tokens"}, "context")
    context["max_tokens"] = (
        _integer(context["max_tokens"], "context.max_tokens")
        if context.get("max_tokens") is not None
        else None
    )

    # prompt 使用固定版本号，确保实验能追溯到具体 prompt 模板。
    prompt = _mapping(value.setdefault("prompt", {}), "prompt")
    _unknown(prompt, {"version"}, "prompt")
    prompt["version"] = _choice(prompt.get("version", "fixed_qa_v1"), {"fixed_qa_v1"}, "prompt.version")

    # generation explicitly selects either remote OpenAI or local extractive generation.
    generation = _mapping(value.get("generation"), "generation")
    generation["provider"] = _choice(
        generation.get("provider"),
        {"extractive", "openai"},
        "generation.provider",
    )
    if generation["provider"] == "extractive":
        _unknown(generation, {"provider", "max_output_tokens"}, "generation")
    else:
        _unknown(
            generation,
            {"provider", "model", "temperature", "max_output_tokens", "timeout_seconds", "max_retries"},
            "generation",
        )
        generation["model"] = _text(generation.get("model"), "generation.model")
        generation["temperature"] = _number(
            generation.get("temperature", 0.0),
            "generation.temperature",
            0.0,
            2.0,
        )
        generation["timeout_seconds"] = _number(
            generation.get("timeout_seconds", 60.0),
            "generation.timeout_seconds",
            0.001,
            3600.0,
        )
        generation["max_retries"] = _integer(
            generation.get("max_retries", 2),
            "generation.max_retries",
            minimum=0,
        )
    generation["max_output_tokens"] = _integer(
        generation.get("max_output_tokens", 512),
        "generation.max_output_tokens",
    )

    # logging 控制结果文件里保留哪些字段。
    logging = _mapping(value.setdefault("logging", {}), "logging")
    _unknown(logging, {"save_retrieved_text", "save_prompt"}, "logging")
    for key in ("save_retrieved_text", "save_prompt"):
        logging[key] = _boolean(logging.get(key, True), f"logging.{key}")

    # load_config() 会注入 _base_dir，后续 resolve_path() 依赖它。
    if "_base_dir" in value:
        value["_base_dir"] = _text(value["_base_dir"], "_base_dir")
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    # 从 YAML 读取配置，并记录配置文件所在目录，供相对路径解析使用。
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_base_dir"] = str(path.parent)
    return validate_config(config)


def _resolve_against(base: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base) / path).resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    # 配置里的相对路径都相对于配置文件目录，而不是当前 shell 工作目录。
    return _resolve_against(config["_base_dir"], value)


def resolve_cli_path(project_root: str | Path, value: str | Path) -> Path:
    # CLI 参数中的相对路径通常相对于项目根目录解析。
    return _resolve_against(project_root, value)


def apply_cli_overrides(config: dict[str, Any], *, top_k: int | None = None) -> dict[str, Any]:
    # 命令行覆盖项不修改原配置，而是返回一个重新校验过的副本。
    effective = copy.deepcopy(config)
    if top_k is not None:
        _integer(top_k, "retrieval.top_k")
        effective["retrieval"]["top_k"] = top_k
    return validate_config(effective)
