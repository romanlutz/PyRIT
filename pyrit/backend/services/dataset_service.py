# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset service for listing seed datasets.

Wraps ``SeedDatasetProvider`` discovery and memory to list available datasets.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from re import match
from urllib.parse import urlparse

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.datasets import (
    DatasetInfo,
    DatasetListResponse,
    SeedExampleDetailResponse,
    SeedExampleListResponse,
    SeedExampleMemberView,
    SeedExampleSummary,
)
from pyrit.datasets import SeedDatasetProvider
from pyrit.memory import CentralMemory, SeedExampleDatasetScope
from pyrit.models import SeedDatasetSummary

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for listing seed datasets."""

    def __init__(self) -> None:
        """Initialize the dataset service."""
        self._memory = CentralMemory.get_memory_instance()

    async def list_datasets_async(self, *, loaded_only: bool = False) -> DatasetListResponse:
        """
        List all available datasets.

        By default the response combines datasets discoverable via registered providers
        with those already loaded into memory, which is what the legacy name-list endpoint
        returned. Pass ``loaded_only=True`` to restrict the response to the memory-backed
        population; an empty memory then returns a valid empty response (#2746) whether or
        not providers are registered.

        Empty dataset names are never surfaced as a named choice: a seed with an empty
        name filters nothing in the seed queries, so selecting it would match seeds from
        every dataset. Seeds stored under an empty name are folded into the unnamed
        population instead. Named and unnamed selections live in distinct namespaces so a
        stored dataset named ``__unnamed__`` cannot collide with the unnamed population.

        Args:
            loaded_only (bool): When True, return only datasets that have seeds in memory.

        Returns:
            DatasetListResponse: Available datasets.
        """
        provider_names = {name for name in await SeedDatasetProvider.get_all_dataset_names_async() if name}
        summaries = self._memory.get_seed_dataset_summaries()
        named_summaries = {summary.dataset_name: summary for summary in summaries if summary.dataset_name}

        dataset_names = set(named_summaries) if loaded_only else provider_names | set(named_summaries)

        items: list[DatasetInfo] = []
        for dataset_name in sorted(dataset_names):
            summary = named_summaries.get(dataset_name)
            items.append(
                DatasetInfo(
                    name=dataset_name,
                    selection_key=f"dataset:named:{dataset_name}",
                    loaded=summary is not None,
                    provider_available=dataset_name in provider_names,
                    logical_examples=summary.logical_examples if summary else None,
                    seed_pieces=summary.seed_pieces if summary else None,
                    objectives=summary.objectives if summary else None,
                    modalities=list(summary.modalities) if summary else [],
                    harm_categories=list(summary.harm_categories) if summary else [],
                    has_unlabeled_harm_categories=summary.has_unlabeled_harm_categories if summary else False,
                )
            )

        unnamed_summary = self._merge_unnamed_summaries(summaries)
        if unnamed_summary is not None:
            items.append(
                DatasetInfo(
                    name="(unnamed)",
                    selection_key="dataset:unnamed",
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

    async def list_seed_examples_async(
        self,
        *,
        selection_key: str,
        limit: int = 20,
        cursor: str | None = None,
        search: str | None = None,
        data_types: Sequence[str] | None = None,
        harm_categories: Sequence[str] | None = None,
        seed_types: Sequence[str] | None = None,
    ) -> SeedExampleListResponse:
        """
        List logical seed examples using Memory's database-backed page query.

        Returns:
            SeedExampleListResponse: The selected logical examples and pagination metadata.
        """
        scope = self._selection_scope(selection_key)
        page = self._memory.get_seed_example_page(
            dataset_scope=scope,
            limit=limit,
            cursor=cursor,
            data_types=data_types,
            harm_categories=harm_categories,
            seed_types=seed_types,
            value_search=search,
        )
        items = [self._summary(item) for item in page.items]
        return SeedExampleListResponse(
            items=items,
            pagination=PaginationInfo(
                limit=limit,
                has_more=page.next_cursor is not None,
                next_cursor=page.next_cursor,
                prev_cursor=cursor,
            ),
        )

    async def get_seed_example_async(self, *, selection_key: str, example_id: str) -> SeedExampleDetailResponse:
        """Return one complete logical seed example without materializing seed models."""
        scope = self._selection_scope(selection_key)
        page = self._memory.get_seed_example_page(dataset_scope=scope, limit=100)
        item = next((candidate for candidate in page.items if str(candidate.example_id) == example_id), None)
        if item is None:
            raise ValueError(f"Seed example not found: {example_id}")
        return SeedExampleDetailResponse(
            example_id=item.example_id,
            dataset_name=item.dataset_name,
            seed_ids=item.seed_ids,
            piece_count=item.piece_count,
            objective_count=item.objective_count,
            modalities=item.modalities,
            seed_types=item.seed_types,
            harm_categories=item.harm_categories,
            has_unlabeled_harm=item.has_unlabeled_harm,
            members=[self._member(member) for member in item.members],
        )

    @staticmethod
    def _selection_scope(selection_key: str) -> SeedExampleDatasetScope:
        """
        Resolve the stable dataset selection namespace.

        Returns:
            SeedExampleDatasetScope: The named or unnamed memory query scope.
        """
        if selection_key == "dataset:unnamed":
            return SeedExampleDatasetScope.unnamed()
        prefix = "dataset:named:"
        if selection_key.startswith(prefix) and selection_key[len(prefix) :]:
            return SeedExampleDatasetScope.named(selection_key[len(prefix) :])
        raise ValueError(f"Invalid dataset selection key: {selection_key}")

    @classmethod
    def _summary(cls, item: object) -> SeedExampleSummary:
        members = item.members  # type: ignore[attr-defined]
        preview, truncated = cls._preview(members)
        return SeedExampleSummary(
            example_id=item.example_id,  # type: ignore[attr-defined]
            dataset_name=item.dataset_name,  # type: ignore[attr-defined]
            name=next((member.name for member in members if member.name), None),
            preview=preview,
            preview_truncated=truncated,
            seed_ids=item.seed_ids,  # type: ignore[attr-defined]
            modalities=item.modalities,  # type: ignore[attr-defined]
            seed_types=item.seed_types,  # type: ignore[attr-defined]
            piece_count=item.piece_count,  # type: ignore[attr-defined]
            objective_count=item.objective_count,  # type: ignore[attr-defined]
            harm_categories=item.harm_categories,  # type: ignore[attr-defined]
            has_unlabeled_harm=item.has_unlabeled_harm,  # type: ignore[attr-defined]
        )

    @staticmethod
    def _preview(members: Sequence[object]) -> tuple[str, bool]:
        safe_text = [
            member.value
            for member in members
            if member.data_type == "text"
            and not urlparse(member.value).scheme
            and not match(r"^(?:[A-Za-z]:[\\/]|/)", member.value)
        ]
        if safe_text:
            value = max(safe_text, key=len)
            return (value[:100] + "...", len(value) > 100) if len(value) > 100 else (value, False)
        data_type = members[0].data_type if members else "seed"
        return f"{data_type} seed", False

    @staticmethod
    def _member(member: object) -> SeedExampleMemberView:
        return SeedExampleMemberView(
            id=member.id,
            prompt_group_id=member.prompt_group_id,
            seed_type=member.seed_type,
            data_type=member.data_type,
            value=member.value,
            value_sha256=member.value_sha256,
            role=member.role,
            sequence=member.sequence,
            name=member.name,
            dataset_name=member.dataset_name,
            harm_categories=member.harm_categories,
            description=member.description,
            source=member.source,
            authors=member.authors,
            groups=member.groups,
            date_added=member.date_added,
            added_by=member.added_by,
            metadata=member.metadata,
            parameters=member.parameters,
            is_jinja_template=member.is_jinja_template,
        )

    @staticmethod
    def _merge_unnamed_summaries(summaries: Sequence[SeedDatasetSummary]) -> SeedDatasetSummary | None:
        """
        Fold the memory groups that filter nothing (``None`` and empty names) into one entry.

        Such seeds behave identically in the seed queries, so the summary represents them
        as a single unnamed population and never as a named choice.

        Args:
            summaries (Sequence[SeedDatasetSummary]): The memory-backed dataset summaries.

        Returns:
            SeedDatasetSummary | None: One merged summary, or None when no unnamed seeds
                are stored.
        """
        unnamed = [summary for summary in summaries if not summary.dataset_name]
        if not unnamed:
            return None
        return SeedDatasetSummary(
            dataset_name=None,
            logical_examples=sum(summary.logical_examples for summary in unnamed),
            seed_pieces=sum(summary.seed_pieces for summary in unnamed),
            objectives=sum(summary.objectives for summary in unnamed),
            modalities=tuple(sorted({modality for summary in unnamed for modality in summary.modalities})),
            harm_categories=tuple(sorted({category for summary in unnamed for category in summary.harm_categories})),
            has_unlabeled_harm_categories=any(summary.has_unlabeled_harm_categories for summary in unnamed),
        )


@lru_cache(maxsize=1)
def get_dataset_service() -> DatasetService:
    """
    Get the global dataset service instance.

    Returns:
        The singleton DatasetService instance.
    """
    return DatasetService()
