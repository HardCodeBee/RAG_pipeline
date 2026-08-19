"""配置加载、校验与路径解析的实现。"""

from __future__ import annotations

# Root-level configuration contract shared by all pipeline stages.

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml


_ROOT_KEYS = {
    "paths",
    "loader",
    "embedding",
    "index",
    "retrieval",
    "bm25",
    "context",
    "prompt",
    "generation",
    "logging",
    "_base_dir",
}

_HF_DENSE_FAMILIES = {
    "dpr": {
        "pooling": "pooler_output",
        "document_input_format": "title_text_pair",
        "max_sequence_length": 256,
    },
    "contriever": {
        "pooling": "masked_mean",
        "document_input_format": "title_space_text",
        "max_sequence_length": 512,
    },
}

_HF_COMMIT_REVISION = re.compile(r"[0-9a-fA-F]{40}")


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


def _optional_integer(value: Any, location: str) -> int | None:
    return None if value is None else _integer(value, location)


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


# 校验配置结构是否类型与范围合理 并对不同组件的配置进行区分
# 进行默认值补充
# 把外部传进来的配置变成 pipeline 可以安全使用的“标准配置对象”
def validate_config(config: dict[str, Any]) -> dict[str, Any]:

    # 深拷贝后再补默认值，避免调用者传入的原始 dict 被就地修改。
    value = copy.deepcopy(_mapping(config, "config"))
    _unknown(value, _ROOT_KEYS, "root")

    # paths 是构建和查询都要用的三个根目录，保持为字符串，使用时再解析。
    paths = _mapping(value.get("paths"), "paths")
    _unknown(paths, {"corpus", "artifacts_root", "outputs_root"}, "paths")
    for key in ("corpus", "artifacts_root", "outputs_root"):
        paths[key] = _text(paths.get(key), f"paths.{key}")

    # loader 控制如何发现和读取 corpus 文件。
    loader = _mapping(value.get("loader"), "loader")
    _unknown(loader, {"expected_dataset"}, "loader")
    loader["expected_dataset"] = _text(
        loader.get("expected_dataset"),
        "loader.expected_dataset",
    )
    # embedding 控制“chunk/query 文本 -> 向量”的 backend 和模型参数。
    embedding = _mapping(value.get("embedding"), "embedding")
    embedding["backend"] = _choice(
        embedding.get("backend"),
        {"hashing", "hf_dense", "sentence_transformers"},
        "embedding.backend",
    )
    common_embedding_keys = {
        "backend",
        "normalize",
        "query_prefix",
        "document_prefix",
        "device",
    }
    if embedding["backend"] == "hashing":
        _unknown(embedding, common_embedding_keys | {"dimension"}, "embedding")
        embedding["dimension"] = _integer(
            embedding.get("dimension", 384),
            "embedding.dimension",
        )
    elif embedding["backend"] == "sentence_transformers":
        _unknown(
            embedding,
            common_embedding_keys
            | {
                "model_name",
                "revision",
                "batch_size",
                "encode_call_rows",
                "shard_rows",
                "max_sequence_length",
                "local_files_only",
            },
            "embedding",
        )
        embedding["model_name"] = _text(embedding.get("model_name"), "embedding.model_name")
        embedding["revision"] = _text(embedding.get("revision"), "embedding.revision")
        embedding["batch_size"] = _integer(embedding.get("batch_size", 32), "embedding.batch_size")
        embedding["encode_call_rows"] = _integer(
            embedding.get("encode_call_rows", embedding["batch_size"]),
            "embedding.encode_call_rows",
        )
        embedding["shard_rows"] = _optional_integer(
            embedding.get("shard_rows"), "embedding.shard_rows"
        )
        embedding["max_sequence_length"] = _optional_integer(
            embedding.get("max_sequence_length"), "embedding.max_sequence_length"
        )
        embedding["local_files_only"] = _boolean(
            embedding.get("local_files_only", False),
            "embedding.local_files_only",
        )
    else:
        _unknown(
            embedding,
            common_embedding_keys
            | {
                "family",
                "model_name",
                "revision",
                "query_model_name",
                "query_revision",
                "pooling",
                "document_input_format",
                "batch_size",
                "encode_call_rows",
                "shard_rows",
                "max_sequence_length",
                "local_files_only",
            },
            "embedding",
        )
        embedding["family"] = _choice(
            embedding.get("family"),
            set(_HF_DENSE_FAMILIES),
            "embedding.family",
        )
        family = _HF_DENSE_FAMILIES[embedding["family"]]
        for key in ("model_name", "revision"):
            embedding[key] = _text(embedding.get(key), f"embedding.{key}")
        if _HF_COMMIT_REVISION.fullmatch(embedding["revision"]) is None:
            raise ValueError("embedding.revision must be a full 40-character commit SHA")
        if embedding["family"] == "dpr":
            for key in ("query_model_name", "query_revision"):
                embedding[key] = _text(embedding.get(key), f"embedding.{key}")
            if _HF_COMMIT_REVISION.fullmatch(embedding["query_revision"]) is None:
                raise ValueError(
                    "embedding.query_revision must be a full 40-character commit SHA"
                )
        elif "query_model_name" in embedding or "query_revision" in embedding:
            raise ValueError("contriever uses one shared encoder and does not accept query model overrides")
        embedding["pooling"] = _choice(
            embedding.get("pooling", family["pooling"]),
            {"masked_mean", "pooler_output"},
            "embedding.pooling",
        )
        if embedding["pooling"] != family["pooling"]:
            raise ValueError(
                f"embedding.pooling must be {family['pooling']!r} for {embedding['family']}"
            )
        embedding["document_input_format"] = _choice(
            embedding.get("document_input_format", family["document_input_format"]),
            {"title_space_text", "title_text_pair"},
            "embedding.document_input_format",
        )
        if embedding["document_input_format"] != family["document_input_format"]:
            raise ValueError(
                "embedding.document_input_format must be "
                f"{family['document_input_format']!r} for {embedding['family']}"
            )
        embedding["batch_size"] = _integer(
            embedding.get("batch_size", 32),
            "embedding.batch_size",
        )
        embedding["encode_call_rows"] = _integer(
            embedding.get("encode_call_rows", embedding["batch_size"]),
            "embedding.encode_call_rows",
        )
        embedding["shard_rows"] = _optional_integer(
            embedding.get("shard_rows"), "embedding.shard_rows"
        )
        embedding["max_sequence_length"] = _integer(
            embedding.get("max_sequence_length", family["max_sequence_length"]),
            "embedding.max_sequence_length",
        )
        if embedding["max_sequence_length"] != family["max_sequence_length"]:
            raise ValueError(
                "embedding.max_sequence_length must be "
                f"{family['max_sequence_length']} for {embedding['family']}"
            )
        embedding["local_files_only"] = _boolean(
            embedding.get("local_files_only", False),
            "embedding.local_files_only",
        )
    embedding["normalize"] = _boolean(
        embedding.get("normalize", embedding["backend"] != "hf_dense"),
        "embedding.normalize",
    )
    embedding["device"] = _choice(
        embedding.get("device", "auto"),
        {"auto", "cpu", "cuda"},
        "embedding.device",
    )
    if embedding["backend"] == "hashing" and embedding["device"] == "cuda":
        raise ValueError("hashing embedding does not support device=cuda")
    for key in ("query_prefix", "document_prefix"):
        item = embedding.get(key, "")
        if not isinstance(item, str):
            raise TypeError(f"embedding.{key} must be a string")
        # prefix 允许保留空字符串；不同 embedding 模型可能需要 query/document 前缀。
        embedding[key] = item
    if embedding["backend"] == "hf_dense":
        if embedding["normalize"]:
            raise ValueError("hf_dense DPR/Contriever baselines require normalize=false")
        if embedding["query_prefix"] or embedding["document_prefix"]:
            raise ValueError("hf_dense DPR/Contriever baselines do not accept text prefixes")
    # index 控制向量索引后端；当前只支持平铺内积索引。
    index = _mapping(value.get("index"), "index")
    index["backend"] = _choice(index.get("backend"), {"faiss", "numpy"}, "index.backend")
    index["type"] = _choice(
        index.get("type", "flat_ip"),
        {"flat_ip", "hnsw_flat", "ivf_flat", "ivf_pq", "streaming_flat_ip"},
        "index.type",
    )
    allowed_index_keys = {"backend", "type", "build_batch_size", "faiss_threads"}
    if index["type"] == "hnsw_flat":
        allowed_index_keys.update({"hnsw_m", "ef_construction"})
    if index["type"] in {"ivf_flat", "ivf_pq"}:
        allowed_index_keys.update({"nlist", "train_size", "train_seed"})
    if index["type"] == "ivf_pq":
        allowed_index_keys.update({"pq_m", "pq_nbits"})
    _unknown(index, allowed_index_keys, "index")
    if index["backend"] == "numpy" and index["type"] != "flat_ip":
        raise ValueError("index.backend=numpy only supports index.type=flat_ip")
    if index["type"] == "streaming_flat_ip" and index["backend"] != "faiss":
        raise ValueError("streaming_flat_ip requires index.backend=faiss")
    if embedding.get("shard_rows") is not None and index["type"] != "streaming_flat_ip":
        raise ValueError("embedding.shard_rows requires index.type=streaming_flat_ip")
    index["build_batch_size"] = _integer(
        index.get("build_batch_size", 65536),
        "index.build_batch_size",
    )
    index["faiss_threads"] = _integer(
        index.get("faiss_threads", 0),
        "index.faiss_threads",
        minimum=0,
    )
    if index["type"] == "hnsw_flat":
        index["hnsw_m"] = _integer(index.get("hnsw_m", 32), "index.hnsw_m")
        index["ef_construction"] = _integer(
            index.get("ef_construction", 200),
            "index.ef_construction",
        )
    if index["type"] in {"ivf_flat", "ivf_pq"}:
        index["nlist"] = _integer(index.get("nlist", 1024), "index.nlist")
        index["train_size"] = _integer(
            index.get("train_size", 100000),
            "index.train_size",
        )
        index["train_seed"] = _integer(
            index.get("train_seed", 20260731),
            "index.train_seed",
            minimum=0,
        )
    if index["type"] == "ivf_pq":
        index["pq_m"] = _integer(index.get("pq_m", 48), "index.pq_m")
        index["pq_nbits"] = _integer(index.get("pq_nbits", 8), "index.pq_nbits")
        if index["pq_nbits"] > 16:
            raise ValueError("index.pq_nbits must be <= 16")

    # retrieval 控制查询时召回策略和默认 top_k。
    retrieval = _mapping(value.setdefault("retrieval", {}), "retrieval")
    _unknown(
        retrieval,
        {
            "method",
            "candidate_k",
            "final_k",
            "nprobe",
            "ef_search",
            "max_codes",
            "search_threads",
            "corpus_chunk_size",
            "query_batch_size",
            "reranker",
        },
        "retrieval",
    )
    retrieval["method"] = _choice(
        retrieval.get("method", "dense"),
        {"dense", "bm25"},
        "retrieval.method",
    )
    retrieval["final_k"] = _integer(
        retrieval.get("final_k", 5),
        "retrieval.final_k",
        minimum=0,
    )
    retrieval["candidate_k"] = _integer(
        retrieval.get("candidate_k", retrieval["final_k"]),
        "retrieval.candidate_k",
        minimum=0,
    )
    if retrieval["final_k"] == 0 and retrieval["candidate_k"] != 0:
        raise ValueError("retrieval.candidate_k must be 0 when final_k=0")
    if retrieval["candidate_k"] < retrieval["final_k"]:
        raise ValueError("retrieval.candidate_k must be >= retrieval.final_k")
    retrieval["search_threads"] = _integer(
        retrieval.get("search_threads", 0),
        "retrieval.search_threads",
        minimum=0,
    )
    streaming_keys = ("corpus_chunk_size", "query_batch_size")
    if index["type"] == "streaming_flat_ip" and retrieval["method"] == "dense":
        retrieval["corpus_chunk_size"] = _integer(
            retrieval.get("corpus_chunk_size", 25_000),
            "retrieval.corpus_chunk_size",
        )
        retrieval["query_batch_size"] = _integer(
            retrieval.get("query_batch_size", 64),
            "retrieval.query_batch_size",
        )
    elif any(key in retrieval for key in streaming_keys):
        raise ValueError(
            "corpus_chunk_size/query_batch_size require dense streaming_flat_ip retrieval"
        )
    ann_parameter_names = ("nprobe", "ef_search", "max_codes")
    if retrieval["method"] == "bm25":
        if any(key in retrieval for key in ann_parameter_names):
            raise ValueError("retrieval.method=bm25 does not accept ANN search parameters")
    else:
        if index["type"] == "flat_ip" and any(
            key in retrieval for key in ann_parameter_names
        ):
            raise ValueError("flat_ip does not accept ANN search parameters")
        if index["type"] == "streaming_flat_ip" and any(
            key in retrieval for key in ann_parameter_names
        ):
            raise ValueError("streaming_flat_ip does not accept ANN search parameters")
        if index["type"] == "hnsw_flat":
            if any(key in retrieval for key in ("nprobe", "max_codes")):
                raise ValueError("hnsw_flat only accepts retrieval.ef_search")
            retrieval["ef_search"] = _integer(
                retrieval.get("ef_search", 64),
                "retrieval.ef_search",
            )
        if index["type"] in {"ivf_flat", "ivf_pq"}:
            if "ef_search" in retrieval:
                raise ValueError("IVF indexes do not accept retrieval.ef_search")
            retrieval["nprobe"] = _integer(
                retrieval.get("nprobe", min(16, index["nlist"])),
                "retrieval.nprobe",
            )
            retrieval["max_codes"] = _integer(
                retrieval.get("max_codes", 0),
                "retrieval.max_codes",
                minimum=0,
            )

    reranker = _mapping(retrieval.get("reranker", {"provider": "none"}), "retrieval.reranker")
    reranker["provider"] = _choice(
        reranker.get("provider", "none"),
        {"none", "cross_encoder"},
        "retrieval.reranker.provider",
    )
    if reranker["provider"] == "none":
        _unknown(reranker, {"provider"}, "retrieval.reranker")
    else:
        _unknown(
            reranker,
            {
                "provider",
                "model_name",
                "revision",
                "batch_size",
                "device",
                "local_files_only",
            },
            "retrieval.reranker",
        )
        reranker["model_name"] = _text(
            reranker.get("model_name"),
            "retrieval.reranker.model_name",
        )
        reranker["revision"] = _text(
            reranker.get("revision"),
            "retrieval.reranker.revision",
        )
        reranker["batch_size"] = _integer(
            reranker.get("batch_size", 32),
            "retrieval.reranker.batch_size",
        )
        reranker["device"] = _choice(
            reranker.get("device", "auto"),
            {"auto", "cpu", "cuda"},
            "retrieval.reranker.device",
        )
        reranker["local_files_only"] = _boolean(
            reranker.get("local_files_only", False),
            "retrieval.reranker.local_files_only",
        )
    retrieval["reranker"] = reranker

    bm25_value = value.get("bm25")
    if retrieval["method"] == "bm25":
        bm25 = _mapping({} if bm25_value is None else bm25_value, "bm25")
        _unknown(
            bm25,
            {
                "backend",
                "method",
                "k1",
                "b",
                "analyzer",
                "mmap",
                "transaction_documents",
            },
            "bm25",
        )
        bm25["backend"] = _choice(
            bm25.get("backend", "bm25s"),
            {"bm25s", "sqlite"},
            "bm25.backend",
        )
        bm25["method"] = _choice(
            bm25.get("method", "lucene"),
            {"lucene"},
            "bm25.method",
        )
        bm25["k1"] = _number(
            bm25.get("k1", 1.5),
            "bm25.k1",
            minimum=0.000001,
            maximum=100.0,
        )
        bm25["b"] = _number(
            bm25.get("b", 0.75),
            "bm25.b",
            minimum=0.0,
            maximum=1.0,
        )
        bm25["analyzer"] = _choice(
            bm25.get(
                "analyzer",
                (
                    "english_default_v1"
                    if bm25["backend"] == "bm25s"
                    else "english_regex_casefold_v1"
                ),
            ),
            (
                {"english_default_v1"}
                if bm25["backend"] == "bm25s"
                else {"english_regex_casefold_v1"}
            ),
            "bm25.analyzer",
        )
        if bm25["backend"] == "bm25s":
            if "transaction_documents" in bm25:
                raise ValueError("bm25.transaction_documents requires backend=sqlite")
            bm25["mmap"] = _boolean(bm25.get("mmap", True), "bm25.mmap")
        else:
            if "mmap" in bm25:
                raise ValueError("bm25.mmap applies only to backend=bm25s")
            bm25["transaction_documents"] = _integer(
                bm25.get("transaction_documents", 1000),
                "bm25.transaction_documents",
            )
        value["bm25"] = bm25
    elif bm25_value is not None:
        raise ValueError("bm25 config requires retrieval.method=bm25")

    # context 控制把召回 chunk 拼进 prompt 时的 token 上限。
    context = _mapping(value.setdefault("context", {}), "context")
    _unknown(context, {"max_tokens"}, "context")
    context["max_tokens"] = _optional_integer(
        context.get("max_tokens"), "context.max_tokens"
    )

    # prompt 使用固定版本号，确保实验能追溯到具体 prompt 模板。
    prompt = _mapping(value.setdefault("prompt", {}), "prompt")
    _unknown(prompt, {"version"}, "prompt")
    prompt["version"] = _choice(
        prompt.get("version", "fixed_qa_v1"),
        {"fixed_qa_v1"},
        "prompt.version",
    )

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
        _integer(top_k, "retrieval.top_k", minimum=0)
        effective["retrieval"]["final_k"] = top_k
        if top_k == 0:
            effective["retrieval"]["candidate_k"] = 0
        elif effective["retrieval"].get("candidate_k", 0) < top_k:
            effective["retrieval"]["candidate_k"] = top_k
    return validate_config(effective)
