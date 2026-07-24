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
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--strict-credentials", action="store_true")
    args = parser.parse_args()
    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    selected = {
        "loader": config["loader"]["type"],
        "chunker": "fixed_sentence",
        "tokenizer": config["chunking"]["tokenizer"],
        "embedder": config["embedding"]["backend"],
        "index": f"{config['index']['backend']}:flat_ip",
        "retriever": "dense",
        "prompt": config["prompt"]["version"],
        "generator": config["generation"]["provider"],
    }
    dependencies = {
        "numpy": _check_import("numpy", "numpy"),
        "pyyaml": _check_import("yaml", "PyYAML"),
    }
    if config["loader"]["type"] == "pypdf":
        dependencies["pypdf"] = _check_import("pypdf", "pypdf")
    if config["loader"]["type"] == "qasper":
        dependencies["datasets"] = _check_import("datasets", "datasets")
    if config["embedding"]["backend"] == "sentence_transformers":
        dependencies["sentence_transformers"] = _check_import("sentence_transformers", "sentence-transformers")
    if config["chunking"]["tokenizer"] == "huggingface":
        dependencies["transformers"] = _check_import("transformers", "transformers")
    if config["index"]["backend"] == "faiss":
        dependencies["faiss"] = _check_import("faiss", "faiss-cpu")
    if config["generation"]["provider"] == "openai":
        dependencies["openai"] = _check_import("openai", "openai")

    environment_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    credentials = {
        "openai_api_key_present": environment_key_present,
        "openai_api_key_source": "environment" if environment_key_present else None,
    }
    dependency_failures = [name for name, value in dependencies.items() if value["status"] != "available"]
    credential_failure = (
        args.strict_credentials
        and config["generation"]["provider"] == "openai"
        and not credentials["openai_api_key_present"]
    )
    result = {
        "status": "failed" if dependency_failures or credential_failure else "ready",
        "config": str(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "selected_components": selected,
        "pinned_revisions": {
            "embedding": config["embedding"].get("revision"),
            "tokenizer": config["chunking"].get("tokenizer_revision"),
        },
        "dependencies": dependencies,
        "credentials": credentials,
        "dependency_failures": dependency_failures,
        "credential_check": "presence_only_no_api_request",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
