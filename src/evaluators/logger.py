"""详细 JSONL 结果和紧凑 CSV 汇总的写入器。"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from pathlib import Path


def _atomic_write_text(path: Path, text: str, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件，再原子替换目标文件，避免中途失败留下半截结果。
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            # fsync 尽量确保内容落盘，再执行 replace/link。
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # overwrite=False 时用硬链接创建目标；如果目标已存在，os.link 会失败。
            os.link(temporary_path, path)
            temporary_path.unlink()
    finally:
        # 无论成功失败，都清理临时文件。
        if temporary_path.exists():
            temporary_path.unlink()


def write_results(path: str | Path, rows: list[dict], overwrite: bool = True) -> None:
    """把每个问题的完整评估轨迹写成 JSONL。"""
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Every JSONL row must be a dictionary")
        # 紧凑 JSONL 便于追加和机器读取；allow_nan=False 保证输出是标准 JSON。
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    text = "".join(f"{line}\n" for line in lines)
    _atomic_write_text(Path(path), text, overwrite=overwrite)


def write_summary_csv(path: str | Path, summary: dict) -> None:
    """把一次完成评估的汇总指标写成单行 CSV。"""
    path = Path(path)
    # 用 StringIO 先生成完整 CSV 文本，再走统一的原子写入。
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)
    _atomic_write_text(path, handle.getvalue())


def write_metadata_json(path: str | Path, metadata: dict, overwrite: bool = True) -> None:
    # metadata 通常较小，使用缩进格式方便人工查看。
    text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_write_text(Path(path), text, overwrite=overwrite)
