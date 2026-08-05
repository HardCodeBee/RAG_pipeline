"""Resolve pinned Hugging Face repositories to local snapshots."""

from __future__ import annotations

# Pinned model resource resolution for component implementations.

from pathlib import Path


def resolve_hf_snapshot(
    model_name: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    """Download or resolve once, then make model libraries use a local path."""

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()
