# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

    async def test_async_loaders_dispatch_blocking_file_io(self, tmp_path: Path, valid_yaml_content: str) -> None:
        file_path = tmp_path / "test.yaml"
        file_path.write_text(valid_yaml_content, encoding="utf-8")
        loader = _LocalDatasetLoader(file_path=file_path)

        with patch(
            "pyrit.datasets.seed_datasets.local.local_dataset_loader.asyncio.to_thread",
            new_callable=AsyncMock,
            wraps=asyncio.to_thread,
        ) as to_thread:
            dataset = await loader.fetch_dataset_async()
            metadata = await loader._parse_metadata_async()

        assert dataset.dataset_name == "test_dataset"
        assert metadata is None
        assert to_thread.await_count == 2
        assert to_thread.await_args_list[0].args == (SeedDataset.from_yaml_file, file_path)
        assert to_thread.await_args_list[1].args == (loader._read_yaml_file,)

    async def test_fetch_dataset_file_not_found(self):
        loader = _LocalDatasetLoader(file_path=Path("non_existent.yaml"))
        with pytest.raises(Exception):
            await loader.fetch_dataset_async()
