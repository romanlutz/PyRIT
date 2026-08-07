# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario service for listing available scenarios.

Provides read-only access to the ScenarioRegistry, exposing scenario metadata
through the REST API.
"""

import asyncio
import logging
from collections import OrderedDict
from functools import lru_cache
from time import monotonic

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.scenarios import ListRegisteredScenariosResponse
from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.models.catalog.scenario import (
    RegisteredScenario,
    ScenarioDefaultRunSizeEstimate,
    ScenarioRunSizeEstimateRequest,
    ScenarioRunSizeEstimateStatus,
)
from pyrit.registry import ScenarioMetadata, ScenarioRegistry

logger = logging.getLogger(__name__)
_ESTIMATE_CACHE_SIZE = 128
_ESTIMATE_CONCURRENCY = 1
_UNAVAILABLE_CACHE_TTL_SECONDS = 30.0
_EstimateCacheKey = tuple[str, int]
_EstimateCacheValue = tuple[ScenarioDefaultRunSizeEstimate, float | None]


def _metadata_to_registered_scenario(
    metadata: ScenarioMetadata,
    *,
    default_run_size: ScenarioDefaultRunSizeEstimate | None = None,
) -> RegisteredScenario:
    """
    Convert a ScenarioMetadata dataclass to a ScenarioSummary Pydantic model.

    Args:
        metadata: The registry metadata for a scenario.
        default_run_size: Scenario-owned default-run estimate.

    Returns:
        ScenarioSummary Pydantic model.
    """
    return RegisteredScenario(
        scenario_name=metadata.registry_name,
        scenario_type=metadata.class_name,
        scenario_version=metadata.scenario_version,
        description=metadata.class_description,
        default_technique=metadata.default_technique,
        default_techniques=list(metadata.default_techniques),
        aggregate_techniques=list(metadata.aggregate_techniques),
        all_techniques=list(metadata.all_techniques),
        default_datasets=list(metadata.default_datasets),
        supported_parameters=list(metadata.supported_parameters),
        baseline_policy=metadata.baseline_policy,
        include_baseline_by_default=metadata.include_baseline_by_default,
        default_run_size=default_run_size or ScenarioDefaultRunSizeEstimate.unavailable(),
    )


class ScenarioService:
    """
    Service for listing available scenarios.

    Uses ScenarioRegistry as the source of truth for scenario metadata.
    """

    def __init__(self) -> None:
        """Initialize the scenario service."""
        self._registry = ScenarioRegistry.get_registry_singleton()
        self._estimate_cache: OrderedDict[_EstimateCacheKey, _EstimateCacheValue] = OrderedDict()
        self._estimate_semaphore = asyncio.Semaphore(_ESTIMATE_CONCURRENCY)

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
        all_summaries = [_metadata_to_registered_scenario(m) for m in all_metadata]

        page, has_more = self._paginate(items=all_summaries, cursor=cursor, limit=limit)
        metadata_by_name = {metadata.registry_name: metadata for metadata in all_metadata}
        estimates = await asyncio.gather(
            *(self._get_default_run_size_estimate_async(metadata=metadata_by_name[item.scenario_name]) for item in page)
        )
        page = [
            item.model_copy(update={"default_run_size": estimate})
            for item, estimate in zip(page, estimates, strict=True)
        ]
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
            estimate = await self._get_default_run_size_estimate_async(metadata=metadata)
            return _metadata_to_registered_scenario(metadata, default_run_size=estimate)
        return None

    async def estimate_scenario_run_size_async(
        self,
        *,
        scenario_name: str,
        request: ScenarioRunSizeEstimateRequest,
    ) -> ScenarioDefaultRunSizeEstimate | None:
        """
        Estimate one configured scenario without creating a run.

        Args:
            scenario_name: Registered scenario name.
            request: Request-specific techniques, datasets, baseline, and parameters.

        Returns:
            ScenarioDefaultRunSizeEstimate | None: Estimate, or ``None`` when the scenario is unknown.
        """
        metadata = self._registry.get_registered_class_metadata(scenario_name)
        if metadata is None:
            return None

        semaphore = getattr(self, "_estimate_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_ESTIMATE_CONCURRENCY)
            self._estimate_semaphore = semaphore
        async with semaphore:
            return await asyncio.to_thread(
                self._estimate_configured_run_size,
                scenario_name=scenario_name,
                request=request,
            )

    async def _get_default_run_size_estimate_async(
        self, *, metadata: ScenarioMetadata
    ) -> ScenarioDefaultRunSizeEstimate:
        """Return a cached scenario-owned estimate without blocking the event loop."""
        cache_key = (metadata.registry_name, metadata.scenario_version)
        cache = getattr(self, "_estimate_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._estimate_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            estimate, expires_at = cached
            if expires_at is None or monotonic() < expires_at:
                cache.move_to_end(cache_key)
                return estimate
            del cache[cache_key]

        semaphore = getattr(self, "_estimate_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_ESTIMATE_CONCURRENCY)
            self._estimate_semaphore = semaphore
        async with semaphore:
            cached = cache.get(cache_key)
            if cached is not None:
                estimate, expires_at = cached
                if expires_at is None or monotonic() < expires_at:
                    cache.move_to_end(cache_key)
                    return estimate
                del cache[cache_key]
            try:
                estimate = await asyncio.to_thread(
                    self._estimate_default_run_size,
                    scenario_name=metadata.registry_name,
                )
            except Exception as exc:
                logger.warning("Default-run estimate failed for scenario '%s': %s", metadata.registry_name, exc)
                estimate = ScenarioDefaultRunSizeEstimate.unavailable(
                    note=(f"The scenario could not resolve its default inputs for estimation ({type(exc).__name__}).")
                )

        expires_at = (
            monotonic() + _UNAVAILABLE_CACHE_TTL_SECONDS
            if estimate.status is ScenarioRunSizeEstimateStatus.Unavailable
            else None
        )
        cache[cache_key] = (estimate, expires_at)
        cache.move_to_end(cache_key)
        while len(cache) > _ESTIMATE_CACHE_SIZE:
            cache.popitem(last=False)
        return estimate

    def _estimate_default_run_size(self, *, scenario_name: str) -> ScenarioDefaultRunSizeEstimate:
        """
        Construct and estimate one scenario in a worker thread.

        Returns:
            ScenarioDefaultRunSizeEstimate: Scenario-owned estimate.
        """
        scenario = self._registry.create_instance(scenario_name)
        return asyncio.run(scenario.get_default_run_size_estimate_async())

    def _estimate_configured_run_size(
        self,
        *,
        scenario_name: str,
        request: ScenarioRunSizeEstimateRequest,
    ) -> ScenarioDefaultRunSizeEstimate:
        """
        Resolve and estimate one request in a worker thread.

        Returns:
            ScenarioDefaultRunSizeEstimate: Request-specific scenario estimate.
        """
        scenario_class = self._registry.get_class(scenario_name)
        objective_target = (
            ScenarioRunService.resolve_target_name(target_name=request.target_name) if request.target_name else None
        )
        estimate_kwargs = ScenarioRunService.resolve_scenario_configuration(
            scenario_name=scenario_name,
            scenario_class=scenario_class,
            objective_target=objective_target,
            techniques=request.techniques,
            dataset_names=request.dataset_names,
            max_dataset_size=request.max_dataset_size,
            dataset_filters=request.dataset_filters,
            include_baseline=request.include_baseline,
        )
        return asyncio.run(
            self._registry.create_and_estimate_async(
                scenario_name,
                scenario_params=request.scenario_params or {},
                **estimate_kwargs,
            )
        )

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
