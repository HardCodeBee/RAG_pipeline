"""只负责 CLI 参数、读取 config、调用 build_index、打印 manifest"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接从项目目录运行脚本，而不需要先把 src 安装成包。
PROJECT_ROOT = Path(__file__).resolve().parents[1] #找到“项目根目录”的绝对路径
sys.path.insert(0, str(PROJECT_ROOT)) #把项目根目录临时加入 Python 的模块搜索路径最前面， 优先使用该目录

from src.cli_utils import configure_utf8_output
from src.config import load_config, resolve_cli_path
from src.index_builder import build_index


def main() -> None:
    """解析命令行参数，构建索引，并打印构建 manifest。"""
    #创建一个 命令行参数解析器， 同时定义接受规则
    parser = argparse.ArgumentParser(description="Build the Naive RAG v1 vector index.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to a YAML config")
    # 根据前面定义的参数规则，解析终端输入，并把结果保存到 args 里。
    args = parser.parse_args()
    configure_utf8_output()

    # CLI 路径相对于项目根目录解析，因此从其他工作目录调用也得到同一个配置。
    config = load_config(resolve_cli_path(PROJECT_ROOT, args.config))
    manifest = build_index(config)  # 把刚才读到的配置 config 传进去，开始真正构建 RAG 的向量索引。

    # 打印 manifest，方便快速查看和记录复现实验信息。
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
