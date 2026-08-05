"""Build one immutable BM25S index over a config's verified chunk artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output
from src.config import load_config, resolve_cli_path
from src.index_builder import build_index
from src.retrievers.bm25_index import build_bm25_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a BM25S index over the active immutable chunk artifact."
    )
    parser.add_argument("--config", required=True, help="Path to a BM25 YAML config")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    configure_utf8_output()

    config = load_config(resolve_cli_path(PROJECT_ROOT, args.config))
    verified_build = build_index(config)
    verified_sparse_index = build_bm25_index(
        config,
        verified_build,
        show_progress=not args.no_progress,
    )
    print(json.dumps(verified_sparse_index.manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
