"""只检查当前实验配置选中的依赖。"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_cli_path
from scripts.cli_support import temporary_openai_api_key


def _check_import(module_name: str, distribution_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None) or importlib.metadata.version(distribution_name)
        return {"status": "available", "version": str(version), "error": None}
    except Exception as exc:
        return {
            "status": "unavailable",
            "version": None,
            "error": {"type": exc.__class__.__name__, "message": str(exc)[:500]},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dependencies selected by one RAG config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict-credentials", action="store_true")
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="Local key file inside the project; its path and value are not printed",
    )
    args = parser.parse_args()
    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    selected = {
        "loader": config["loader"]["type"],
        "chunker": config["chunking"]["strategy"],
        "tokenizer": config["chunking"]["tokenizer"],
        "embedder": config["embedding"]["backend"],
        "index": f"{config['index']['backend']}:{config['index']['type']}",
        "retriever": config["retrieval"]["method"],
        "reranker": config["retrieval"]["reranker"]["provider"],
        "prompt": config["prompt"]["version"],
        "generator": config["generation"]["provider"],
    }
    dependencies = {
        "numpy": _check_import("numpy", "numpy"),
        "pyyaml": _check_import("yaml", "PyYAML"),
    }
    if config["loader"]["type"] == "qasper":
        dependencies["datasets"] = _check_import("datasets", "datasets")
    reranker = config["retrieval"]["reranker"]
    if (
        config["embedding"]["backend"] == "sentence_transformers"
        or reranker["provider"] == "cross_encoder"
    ):
        dependencies["sentence_transformers"] = _check_import("sentence_transformers", "sentence-transformers")
    if config["chunking"]["tokenizer"] == "huggingface":
        dependencies["transformers"] = _check_import("transformers", "transformers")
    if config["index"]["backend"] == "faiss":
        dependencies["faiss"] = _check_import("faiss", "faiss-cpu")
    if config["retrieval"]["method"] == "bm25":
        dependencies["bm25s"] = _check_import("bm25s", "bm25s")
    if config["generation"]["provider"] == "openai":
        dependencies["openai"] = _check_import("openai", "openai")

    environment_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    with temporary_openai_api_key(args.api_key_file, allowed_root=PROJECT_ROOT):
        effective_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    credentials = {
        "openai_api_key_present": effective_key_present,
        "openai_api_key_source": (
            "environment"
            if environment_key_present
            else ("local_file" if effective_key_present else None)
        ),
    }
    gpu_requested_by = []
    if config["embedding"]["device"] == "cuda":
        gpu_requested_by.append("embedding")
    if reranker["provider"] == "cross_encoder" and reranker["device"] == "cuda":
        gpu_requested_by.append("reranker")
    gpu = {"required": bool(gpu_requested_by), "requested_by": gpu_requested_by}
    if gpu["required"]:
        try:
            import torch

            gpu.update(
                {
                    "torch_version": torch.__version__,
                    "cuda_build": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "memory_mib": (
                        round(torch.cuda.get_device_properties(0).total_memory / 2**20)
                        if torch.cuda.is_available()
                        else None
                    ),
                }
            )
            if torch.cuda.is_available():
                probe = torch.ones(8, device="cuda")
                gpu["tensor_operation_ok"] = float(probe.sum().item()) == 8.0
                del probe
        except Exception as exc:
            gpu.update(
                {
                    "cuda_available": False,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc)[:500],
                    },
                }
            )
    dependency_failures = [name for name, value in dependencies.items() if value["status"] != "available"]
    credential_failure = (
        args.strict_credentials
        and config["generation"]["provider"] == "openai"
        and not credentials["openai_api_key_present"]
    )
    gpu_failure = gpu["required"] and (
        not gpu.get("cuda_available", False)
        or not gpu.get("tensor_operation_ok", False)
    )
    result = {
        "status": "failed"
        if dependency_failures or credential_failure or gpu_failure
        else "ready",
        "config": str(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "selected_components": selected,
        "pinned_revisions": {
            "embedding": config["embedding"].get("revision"),
            "tokenizer": config["chunking"].get("tokenizer_revision"),
            "reranker": reranker.get("revision"),
        },
        "reranker_model": {
            "model_name": reranker.get("model_name"),
            "device": reranker.get("device"),
            "local_files_only": reranker.get("local_files_only"),
            "download_on_first_use": (
                reranker["provider"] == "cross_encoder"
                and not reranker.get("local_files_only", False)
            ),
        },
        "dependencies": dependencies,
        "credentials": credentials,
        "gpu": gpu,
        "dependency_failures": dependency_failures,
        "credential_check": "presence_only_no_api_request",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
