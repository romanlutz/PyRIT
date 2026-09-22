# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset service for listing seed datasets.

Wraps ``SeedDatasetProvider`` discovery and memory to list available datasets.
"""

import logging
from functools import lru_cache

from pyrit.backend.models.datasets import (
    DatasetInfo,
    DatasetListResponse,
)
from pyrit.datasets import SeedDatasetProvider
from pyrit.memory import CentralMemory

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for listing seed datasets."""

    def __init__(self) -> None:
        """Initialize the dataset service."""
        self._memory = CentralMemory.get_memory_instance()

    async def list_datasets_async(self) -> DatasetListResponse:
        """
        List all available datasets.

        Combines datasets discoverable via registered providers with those
        already loaded into memory, since both are available for use.

        Returns:
            DatasetListResponse: Available datasets.
        """
        provider_names = await SeedDatasetProvider.get_all_dataset_names_async()
        summaries = self._memory.get_seed_dataset_summaries()
        summary_by_name = {summary.dataset_name: summary for summary in summaries}
        provider_name_set = set(provider_names)

        dataset_names = provider_name_set | {
            summary.dataset_name for summary in summaries if summary.dataset_name is not None
        }
        items: list[DatasetInfo] = []
        for dataset_name in sorted(dataset_names):
            summary = summary_by_name.get(dataset_name)
            items.append(
                DatasetInfo(
                    name=dataset_name,
                    selection_key=f"dataset:{dataset_name}",
                    loaded=summary is not None,
                    provider_available=dataset_name in provider_name_set,
                    logical_examples=summary.logical_examples if summary else None,
                    seed_pieces=summary.seed_pieces if summary else None,
                    objectives=summary.objectives if summary else None,
                    modalities=list(summary.modalities) if summary else [],
                    harm_categories=list(summary.harm_categories) if summary else [],
                    has_unlabeled_harm_categories=summary.has_unlabeled_harm_categories if summary else False,
                )
            )

        unnamed_summary = summary_by_name.get(None)
        if unnamed_summary is not None:
            items.append(
                DatasetInfo(
                    name="(unnamed)",
                    selection_key="dataset:__unnamed__",
                    loaded=True,
                    provider_available=False,
                    logical_examples=unnamed_summary.logical_examples,
                    seed_pieces=unnamed_summary.seed_pieces,
                    objectives=unnamed_summary.objectives,
                    modalities=list(unnamed_summary.modalities),
                    harm_categories=list(unnamed_summary.harm_categories),
                    has_unlabeled_harm_categories=unnamed_summary.has_unlabeled_harm_categories,
                )
            )

        return DatasetListResponse(items=items)


@lru_cache(maxsize=1)
def get_dataset_service() -> DatasetService:
    """
    Get the global dataset service instance.

    Returns:
        The singleton DatasetService instance.
    """
    return DatasetService()
