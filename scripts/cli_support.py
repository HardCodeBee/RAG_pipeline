"""命令行入口共享的参数校验和终端配置。"""

from __future__ import annotations

# Shared command-line boundary helpers.

import argparse
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise argparse.ArgumentTypeError("run id must use 1-128 ASCII letters, digits, '.', '_' or '-'")
    return value


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


@contextmanager
def temporary_openai_api_key(
    key_file: str | Path | None,
    *,
    allowed_root: str | Path,
) -> Iterator[None]:
    """Temporarily load one local key without adding it or its path to metadata."""

    if os.environ.get("OPENAI_API_KEY"):
        yield
        return
    if key_file is None:
        yield
        return

    root = Path(allowed_root).resolve()
    path = Path(key_file)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("The API key file must resolve inside the project directory") from exc
    if not path.is_file():
        raise FileNotFoundError("The API key file is missing or is not a regular file")
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except UnicodeError as exc:
        raise ValueError("The API key file must be valid UTF-8 text") from exc
    if len(lines) != 1 or "=" in lines[0] or any(character.isspace() for character in lines[0]):
        raise ValueError("The API key file must contain exactly one key value")

    os.environ["OPENAI_API_KEY"] = lines[0]
    try:
        yield
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
