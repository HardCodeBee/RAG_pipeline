from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts.prepare_qasper import DATASET_ID, PARQUET_REVISION, REQUIRED_COLUMNS, prepare_qasper


class _FakeSplit:
    num_rows = 1
    column_names = sorted(REQUIRED_COLUMNS)


class _FakeDataset(dict):
    def save_to_disk(self, path: str) -> None:
        Path(path).mkdir(parents=True)


def test_prepare_qasper_downloads_once_then_uses_local_disk(tmp_path: Path, monkeypatch) -> None:
    dataset = _FakeDataset({name: _FakeSplit() for name in ("train", "validation", "test")})
    calls = {"remote": 0, "local": 0}

    def load_dataset(dataset_id: str, revision: str):
        calls["remote"] += 1
        assert dataset_id == DATASET_ID
        assert revision == PARQUET_REVISION
        return dataset

    def load_from_disk(path: str):
        calls["local"] += 1
        assert Path(path).is_dir()
        return dataset

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = load_dataset
    fake_datasets.load_from_disk = load_from_disk
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = tmp_path / "qasper"
    first = prepare_qasper(output)
    second = prepare_qasper(output)

    assert first["loaded_from"] == "huggingface"
    assert second["loaded_from"] == "local_disk"
    assert first["split_rows"] == second["split_rows"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert calls == {"remote": 1, "local": 1}
