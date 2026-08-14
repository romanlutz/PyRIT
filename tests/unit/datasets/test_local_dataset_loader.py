# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from pyrit.datasets.seed_datasets.local.local_dataset_loader import _LocalDatasetLoader
from pyrit.models import SeedDataset


class TestLocalDatasetLoader:
    @pytest.fixture
    def valid_yaml_content(self):
        return """
dataset_name: test_dataset
source: http://example.com
description: Test description
seeds:
  - value: test prompt
    data_type: text
"""

    def test_init(self, tmp_path, valid_yaml_content):
        file_path = tmp_path / "test.yaml"
        file_path.write_text(valid_yaml_content, encoding="utf-8")

        loader = _LocalDatasetLoader(file_path=file_path)
        assert loader.dataset_name == "test_dataset"
        assert loader.file_path == file_path

    def test_init_invalid_yaml(self, tmp_path):
        file_path = tmp_path / "test.yaml"
        file_path.write_text("invalid: yaml: content: :", encoding="utf-8")

        loader = _LocalDatasetLoader(file_path=file_path)
        # Should fallback to filename stem
        assert loader.dataset_name == "test"

    async def test_fetch_dataset(self, tmp_path, valid_yaml_content):
        file_path = tmp_path / "test.yaml"
        file_path.write_text(valid_yaml_content, encoding="utf-8")

        loader = _LocalDatasetLoader(file_path=file_path)
        dataset = await loader.fetch_dataset_async()

        assert isinstance(dataset, SeedDataset)
        assert dataset.dataset_name == "test_dataset"
        assert len(dataset.prompts) == 1
        assert dataset.prompts[0].value == "test prompt"

    async def test_async_loaders_offload_blocking_file_io(self, tmp_path: Path, valid_yaml_content: str) -> None:
        file_path = tmp_path / "test.yaml"
        file_path.write_text(valid_yaml_content, encoding="utf-8")
        loader = _LocalDatasetLoader(file_path=file_path)
        event_loop_thread_id = threading.get_ident()
        worker_thread_ids: list[int] = []
        original_from_yaml_file = SeedDataset.from_yaml_file
        original_read_yaml_file = loader._read_yaml_file

        def load_dataset(path: Path) -> SeedDataset:
            worker_thread_ids.append(threading.get_ident())
            return original_from_yaml_file(path)

        def read_yaml_file() -> object:
            worker_thread_ids.append(threading.get_ident())
            return original_read_yaml_file()

        with (
            patch.object(SeedDataset, "from_yaml_file", side_effect=load_dataset),
            patch.object(loader, "_read_yaml_file", side_effect=read_yaml_file),
        ):
            dataset = await loader.fetch_dataset_async()
            metadata = await loader._parse_metadata_async()

        assert dataset.dataset_name == "test_dataset"
        assert metadata is None
        assert len(worker_thread_ids) == 2
        assert all(thread_id != event_loop_thread_id for thread_id in worker_thread_ids)

    async def test_fetch_dataset_file_not_found(self):
        loader = _LocalDatasetLoader(file_path=Path("non_existent.yaml"))
        with pytest.raises(FileNotFoundError):
            await loader.fetch_dataset_async()
