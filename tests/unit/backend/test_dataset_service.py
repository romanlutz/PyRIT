# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend dataset service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.services.dataset_service import DatasetService, get_dataset_service
from pyrit.memory.memory_interface import SeedDatasetSummary


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    memory = MagicMock()
    memory.get_seed_dataset_names.return_value = []
    memory.get_seed_dataset_summaries.return_value = []
    return memory


@pytest.fixture
def dataset_service(mock_memory):
    """Create a dataset service with mocked memory."""
    with patch("pyrit.backend.services.dataset_service.CentralMemory") as mock_central:
        mock_central.get_memory_instance.return_value = mock_memory
        yield DatasetService()


@pytest.mark.usefixtures("patch_central_database")
class TestListDatasets:
    """Tests for DatasetService.list_datasets_async."""

    async def test_list_datasets(self, dataset_service):
        with patch(
            "pyrit.backend.services.dataset_service.SeedDatasetProvider.get_all_dataset_names_async",
            new_callable=AsyncMock,
            return_value=["airt_hate", "harmbench"],
        ):
            result = await dataset_service.list_datasets_async()

        assert [item.name for item in result.items] == ["airt_hate", "harmbench"]
        assert all(not item.loaded for item in result.items)
        assert all(item.provider_available for item in result.items)


    async def test_list_datasets_includes_memory_summaries(self, dataset_service, mock_memory):
        mock_memory.get_seed_dataset_summaries.return_value = [
            SeedDatasetSummary(
                dataset_name="harmbench",
                logical_examples=3,
                seed_pieces=4,
                objectives=1,
                modalities=("text", "image_path"),
                harm_categories=("hate", "violence"),
                has_unlabeled_harm_categories=True,
            ),
            SeedDatasetSummary(
                dataset_name=None,
                logical_examples=1,
                seed_pieces=2,
                objectives=0,
                modalities=("text",),
                harm_categories=(),
                has_unlabeled_harm_categories=True,
            ),
        ]
        with patch(
            "pyrit.backend.services.dataset_service.SeedDatasetProvider.get_all_dataset_names_async",
            new_callable=AsyncMock,
            return_value=["harmbench", "provider_only"],
        ):
            result = await dataset_service.list_datasets_async()

        loaded = next(item for item in result.items if item.name == "harmbench")
        assert loaded.selection_key == "dataset:harmbench"
        assert loaded.loaded is True
        assert loaded.provider_available is True
        assert loaded.logical_examples == 3
        assert loaded.seed_pieces == 4
        assert loaded.objectives == 1
        assert loaded.modalities == ["image_path", "text"]
        assert loaded.harm_categories == ["hate", "violence"]
        assert loaded.has_unlabeled_harm_categories is True

        provider_only = next(item for item in result.items if item.name == "provider_only")
        assert provider_only.loaded is False
        assert provider_only.provider_available is True
        assert provider_only.logical_examples is None

        unnamed = next(item for item in result.items if item.name == "(unnamed)")
        assert unnamed.selection_key == "dataset:__unnamed__"
        assert unnamed.loaded is True
        assert unnamed.provider_available is False
        assert unnamed.logical_examples == 1

    async def test_list_datasets_empty(self, dataset_service):
        with patch(
            "pyrit.backend.services.dataset_service.SeedDatasetProvider.get_all_dataset_names_async",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await dataset_service.list_datasets_async()

        assert result.items == []


@pytest.mark.usefixtures("patch_central_database")
def test_get_dataset_service_is_singleton():
    get_dataset_service.cache_clear()
    with patch("pyrit.backend.services.dataset_service.CentralMemory"):
        assert get_dataset_service() is get_dataset_service()
