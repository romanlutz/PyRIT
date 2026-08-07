# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario service for listing available scenarios.

Provides read-only access to the ScenarioRegistry, exposing scenario metadata
through the REST API.
"""

import asyncio
import logging
from functools import lru_cache, partial

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.scenarios import ListRegisteredScenariosResponse
from pyrit.models.catalog.scenario import (
    RegisteredScenario,
    ScenarioDatasetSummary,
    ScenarioRunSizeEstimate,
)
from pyrit.registry import ScenarioMetadata, ScenarioRegistry

logger = logging.getLogger(__name__)


def _metadata_to_registered_scenario(
    *,
    metadata: ScenarioMetadata,
    dataset_summaries: list[ScenarioDatasetSummary],
    run_size: ScenarioRunSizeEstimate,
) -> RegisteredScenario:
    """
    Convert a ScenarioMetadata dataclass to a ScenarioSummary Pydantic model.

    Args:
        metadata: The registry metadata for a scenario.
        dataset_summaries: Resolved default dataset counts.
        run_size: Default outer-unit estimate.

    Returns:
        ScenarioSummary Pydantic model.
    """
    return RegisteredScenario(
        scenario_name=metadata.registry_name,
        scenario_type=metadata.class_name,
        description=metadata.class_description,
        description_markdown=metadata.description_markdown,
        default_technique=metadata.default_technique,
        default_techniques=list(metadata.default_techniques),
        aggregate_techniques=list(metadata.aggregate_techniques),
        all_techniques=list(metadata.all_techniques),
        default_datasets=list(metadata.default_datasets),
        default_dataset_summaries=dataset_summaries,
        default_run_size=run_size,
        supported_parameters=list(metadata.supported_parameters),
        baseline_policy=metadata.baseline_policy,
        include_baseline_by_default=metadata.include_baseline_by_default,
    )


class ScenarioService:
    """
    Service for listing available scenarios.

    Uses ScenarioRegistry as the source of truth for scenario metadata.
    """

    def __init__(self) -> None:
        """Initialize the scenario service."""
        self._registry = ScenarioRegistry.get_registry_singleton()
        self._default_details_cache: dict[
            str,
            tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate],
        ] = {}
        self._default_details_tasks: dict[
            str,
            asyncio.Task[tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]],
        ] = {}
        self._default_details_lock = asyncio.Lock()

    async def list_scenarios_async(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ListRegisteredScenariosResponse:
        """
        List all available scenarios with pagination.

        Args:
            limit: Maximum items to return per page.
            cursor: Pagination cursor (scenario_name to start after).

        Returns:
            ScenarioListResponse with paginated scenario summaries.
        """
        all_metadata = self._registry.get_all_registered_class_metadata()
        details = [await self._get_default_details_async(metadata=metadata) for metadata in all_metadata]
        all_summaries = [
            _metadata_to_registered_scenario(
                metadata=metadata,
                dataset_summaries=dataset_summaries,
                run_size=run_size,
            )
            for metadata, (dataset_summaries, run_size) in zip(all_metadata, details, strict=True)
        ]

        page, has_more = self._paginate(items=all_summaries, cursor=cursor, limit=limit)
        next_cursor = page[-1].scenario_name if has_more and page else None

        return ListRegisteredScenariosResponse(
            items=page,
            pagination=PaginationInfo(
                limit=limit,
                has_more=has_more,
                next_cursor=next_cursor,
                prev_cursor=cursor,
            ),
        )

    async def get_scenario_async(self, *, scenario_name: str) -> RegisteredScenario | None:
        """
        Get a single scenario by registry name.

        Args:
            scenario_name: The registry key of the scenario (e.g., 'foundry.red_team_agent').

        Returns:
            ScenarioSummary if found, None otherwise.
        """
        metadata = self._registry.get_registered_class_metadata(scenario_name)
        if metadata is not None:
            dataset_summaries, run_size = await self._get_default_details_async(metadata=metadata)
            return _metadata_to_registered_scenario(
                metadata=metadata,
                dataset_summaries=dataset_summaries,
                run_size=run_size,
            )
        return None

    async def _get_default_details_async(
        self,
        *,
        metadata: ScenarioMetadata,
    ) -> tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]:
        """
        Resolve and cache one scenario's default dataset and sizing details.

        Args:
            metadata: Static registry metadata for the scenario.

        Returns:
            tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]: Cached
                catalog details. Failures degrade only this scenario to unavailable.
        """
        cache = getattr(self, "_default_details_cache", None)
        if cache is None:
            cache = {}
            self._default_details_cache = cache
        cached = cache.get(metadata.registry_name)
        if cached is not None:
            return cached

        tasks = getattr(self, "_default_details_tasks", None)
        if tasks is None:
            tasks = {}
            self._default_details_tasks = tasks
        task = tasks.get(metadata.registry_name)
        if task is None:
            task = asyncio.create_task(self._resolve_default_details_serialized_async(metadata=metadata))
            tasks[metadata.registry_name] = task
            task.add_done_callback(
                partial(
                    self._clear_default_details_task,
                    scenario_name=metadata.registry_name,
                )
            )
        return await asyncio.shield(task)

    async def _resolve_default_details_serialized_async(
        self,
        *,
        metadata: ScenarioMetadata,
    ) -> tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]:
        """
        Resolve one estimate while owning the global dataset-resolution lock.

        Args:
            metadata: Static registry metadata for the scenario.

        Returns:
            tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]: Successful
                cached details or a retryable unavailable result.
        """
        cache = self._default_details_cache
        lock = getattr(self, "_default_details_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._default_details_lock = lock
        async with lock:
            cached = cache.get(metadata.registry_name)
            if cached is not None:
                return cached
            try:
                scenario_class = self._registry.get_class(metadata.registry_name)
                scenario = scenario_class()  # type: ignore[ty:missing-argument]
                details = await scenario.get_default_catalog_details_async()
            except Exception as exc:
                logger.warning(
                    "Default catalog estimate failed for scenario '%s': %s",
                    metadata.registry_name,
                    exc,
                    exc_info=True,
                )
                return (
                    [ScenarioDatasetSummary(name=name) for name in metadata.default_datasets],
                    ScenarioRunSizeEstimate(
                        status="unavailable",
                        total=None,
                        components=[],
                        caveat=f"{type(exc).__name__}: {exc}",
                    ),
                )
            cache[metadata.registry_name] = details
            return details

    def _clear_default_details_task(
        self,
        task: asyncio.Task[tuple[list[ScenarioDatasetSummary], ScenarioRunSizeEstimate]],
        *,
        scenario_name: str,
    ) -> None:
        """
        Remove a completed single-flight task without disturbing a replacement.

        Args:
            task: The completed resolution task.
            scenario_name: Registry name whose in-flight task completed.
        """
        tasks = self._default_details_tasks
        if tasks.get(scenario_name) is task:
            tasks.pop(scenario_name)

    @staticmethod
    def _paginate(
        *,
        items: list[RegisteredScenario],
        cursor: str | None,
        limit: int,
    ) -> tuple[list[RegisteredScenario], bool]:
        """
        Apply cursor-based pagination.

        Args:
            items: Full list of items.
            cursor: Scenario name to start after.
            limit: Maximum items per page.

        Returns:
            Tuple of (paginated items, has_more flag).
        """
        start_idx = 0
        if cursor:
            for i, item in enumerate(items):
                if item.scenario_name == cursor:
                    start_idx = i + 1
                    break

        page = items[start_idx : start_idx + limit]
        has_more = len(items) > start_idx + limit
        return page, has_more


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    """
    Get the global scenario service instance.

    Returns:
        The singleton ScenarioService instance.
    """
    return ScenarioService()
