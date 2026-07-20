"""读取、写入并校验不可变构建目录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.records import EmbeddingSpaceSpec
from src.io_utils import sha256_file


def read_manifest(build_dir: str | Path) -> dict[str, Any]:
    # 每个不可变构建目录都必须以 manifest.json 作为入口。
    path = Path(build_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Build manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Build manifest must be a JSON object: {path}")
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    # 关闭 allow_nan 可以避免 NaN/Infinity 写进 JSON，保证产物可移植。
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def artifact_descriptor(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    # 描述对象是 manifest 中记录单个产物的最小元数据。
    # 文件大小和 sha256 用于查询阶段检测文件是否被篡改或损坏。
    descriptor = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def validate_build_directory(build_dir: str | Path, expected_build_id: str | None = None) -> dict[str, Any]:
    directory = Path(build_dir)
    manifest = read_manifest(directory)
    # 只加载已经原子提交完成的构建，再检查其身份和具体产物文件。
    if manifest.get("status") != "complete":
        raise ValueError("Build manifest must have status=complete")
    if expected_build_id is not None and manifest.get("build_id") != expected_build_id:
        raise ValueError("Build directory identity does not match the expected build id")
    embedding = manifest.get("embedding")
    if not isinstance(embedding, dict) or set(embedding) != {"space"}:
        raise ValueError("Build manifest must contain one canonical embedding space")
    if not isinstance(embedding["space"], dict) or "query_prefix" in embedding["space"]:
        raise ValueError("Build manifest embedding space is invalid")
    try:
        EmbeddingSpaceSpec.from_mapping(embedding["space"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Build manifest embedding space is invalid") from exc
    for name in ("chunks", "embeddings", "index"):
        descriptor = manifest.get("artifacts", {}).get(name)
        if not isinstance(descriptor, dict) or not descriptor.get("file"):
            raise ValueError(f"Manifest is missing the {name} artifact descriptor")
        path = directory / descriptor["file"]
        # 每个产物都按 manifest 中的文件名、大小和 sha256 精确校验。
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} artifact is missing: {path}")
        if path.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"{name} artifact size does not match the manifest")
        if sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"{name} artifact hash does not match the manifest")
    return manifest
