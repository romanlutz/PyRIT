# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scenario catalog and side-effect-free planning service."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.scenarios import ListRegisteredScenariosResponse
from pyrit.backend.services.scenario_configuration_service import ScenarioConfigurationService
from pyrit.models import Message, ScenarioEstimateRequest, ScenarioRunSizeEstimate, ScenarioRunSizeStatus
from pyrit.models.catalog.scenario import RegisteredScenario
from pyrit.prompt_target import PromptTarget
from pyrit.registry import ScenarioMetadata, ScenarioRegistry, TargetRegistry

logger = logging.getLogger(__name__)


class _EstimateOnlyPromptTarget(PromptTarget):
    """Non-sending target used when a catalog estimate has no target selection."""

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        raise RuntimeError("The estimate-only target cannot send prompts.")


def _unavailable_estimate(*, caveat: str) -> ScenarioRunSizeEstimate:
    """
    Build a stable unavailable estimate at the catalog isolation boundary.

    Returns:
        ScenarioRunSizeEstimate: An unavailable estimate carrying the failure caveat.
    """
    return ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.UNAVAILABLE,
        total_planned_executions=None,
        components=[],
        datasets=[],
        caveat=caveat,
    )


def _metadata_to_registered_scenario(
    *,
    metadata: ScenarioMetadata,
    default_estimate: ScenarioRunSizeEstimate,
) -> RegisteredScenario:
    """
    Project registry metadata and its cached default estimate onto the API model.

    Returns:
        RegisteredScenario: The public catalog entry.
    """
    return RegisteredScenario(
        scenario_name=metadata.registry_name,
        scenario_type=metadata.class_name,
        description=metadata.class_description,
        description_markdown=metadata.class_description_markdown,
        default_technique=metadata.default_technique,
        default_techniques=list(metadata.default_techniques),
        aggregate_techniques=list(metadata.aggregate_techniques),
        aggregate_technique_expansions={
            aggregate: list(expansion) for aggregate, expansion in metadata.aggregate_technique_expansions
        },
        all_techniques=list(metadata.all_techniques),
        default_datasets=list(metadata.default_datasets),
        default_dataset_summaries=list(default_estimate.datasets),
        default_estimate=default_estimate,
        supported_parameters=list(metadata.supported_parameters),
        baseline_policy=metadata.baseline_policy,
        include_baseline_by_default=metadata.include_baseline_by_default,
    )


class ScenarioService:
    """Expose Scenario metadata and scenario-owned run-size planning."""

    def __init__(self) -> None:
        """Initialize registry access and the per-scenario default-estimate cache."""
        self._registry = ScenarioRegistry.get_registry_singleton()
        self._configuration_service = ScenarioConfigurationService()
        self._default_estimate_cache: dict[str, ScenarioRunSizeEstimate] = {}
        self._default_estimate_locks: dict[str, asyncio.Lock] = {}

    async def list_scenarios_async(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ListRegisteredScenariosResponse:
        """
        List scenarios with cached default estimates and cursor pagination.

        Returns:
            ListRegisteredScenariosResponse: The requested catalog page.
        """
        all_metadata = self._registry.get_all_registered_class_metadata()
        page_metadata, has_more = self._paginate_metadata(items=all_metadata, cursor=cursor, limit=limit)
        default_estimates = await asyncio.gather(
            *(self._get_default_estimate_async(metadata=metadata) for metadata in page_metadata)
        )
        page = [
            _metadata_to_registered_scenario(metadata=metadata, default_estimate=estimate)
            for metadata, estimate in zip(page_metadata, default_estimates, strict=True)
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
        Get one scenario and its cached default estimate.

        Returns:
            RegisteredScenario | None: The catalog entry, or None when it is not registered.
        """
        metadata = self._registry.get_registered_class_metadata(scenario_name)
        if metadata is None:
            return None
        estimate = await self._get_default_estimate_async(metadata=metadata)
        return _metadata_to_registered_scenario(metadata=metadata, default_estimate=estimate)

    async def estimate_scenario_async(
        self,
        *,
        scenario_name: str,
        request: ScenarioEstimateRequest,
    ) -> ScenarioRunSizeEstimate | None:
        """
        Estimate launch-aligned selections without creating a ScenarioResult or AttackResult.

        Returns:
            ScenarioRunSizeEstimate | None: The estimate, or None when the scenario is not registered.
        """
        if self._registry.get_registered_class_metadata(scenario_name) is None:
            return None
        return await asyncio.to_thread(
            self._estimate_scenario_in_worker,
            scenario_name,
            request,
        )

    async def _get_default_estimate_async(self, *, metadata: ScenarioMetadata) -> ScenarioRunSizeEstimate:
        """Return one cached default estimate, isolating failures to its scenario."""
        if not hasattr(self, "_default_estimate_cache"):
            self._default_estimate_cache = {}
            self._default_estimate_locks = {}
        cached = self._default_estimate_cache.get(metadata.registry_name)
        if cached is not None:
            return cached

        lock = self._default_estimate_locks.setdefault(metadata.registry_name, asyncio.Lock())
        async with lock:
            cached = self._default_estimate_cache.get(metadata.registry_name)
            if cached is not None:
                return cached
            try:
                estimate = await asyncio.to_thread(
                    self._estimate_scenario_in_worker,
                    metadata.registry_name,
                    ScenarioEstimateRequest(),
                )
            except Exception as exc:
                logger.exception("Default run-size estimation failed for scenario '%s'.", metadata.registry_name)
                estimate = _unavailable_estimate(
                    caveat=f"Default run-size estimation failed for this scenario: {type(exc).__name__}: {exc}"
                )
            self._default_estimate_cache[metadata.registry_name] = estimate
            return estimate

    def _estimate_scenario_in_worker(
        self,
        scenario_name: str,
        request: ScenarioEstimateRequest,
    ) -> ScenarioRunSizeEstimate:
        """
        Run potentially expensive dataset planning on a worker-owned event loop.

        Returns:
            ScenarioRunSizeEstimate: The scenario-owned estimate.
        """
        return asyncio.run(self._estimate_scenario_core_async(scenario_name=scenario_name, request=request))

    async def _estimate_scenario_core_async(
        self,
        *,
        scenario_name: str,
        request: ScenarioEstimateRequest,
    ) -> ScenarioRunSizeEstimate:
        """
        Resolve launch fields and invoke the registry's no-persistence lifecycle.

        Returns:
            ScenarioRunSizeEstimate: The scenario-owned estimate.
        """
        scenario_class = self._registry.get_class(scenario_name)
        target_is_configured = request.target_name is not None
        objective_target: PromptTarget
        if request.target_name is None:
            objective_target = _EstimateOnlyPromptTarget()
        else:
            resolved_target = TargetRegistry.get_registry_singleton().instances.get(request.target_name)
            if resolved_target is None:
                available = TargetRegistry.get_registry_singleton().instances.get_names()
                available_text = ", ".join(available) if available else "(none registered)"
                raise ValueError(
                    f"Target '{request.target_name}' not found in registry. Available targets: {available_text}"
                )
            objective_target = resolved_target

        estimate_kwargs = self._configuration_service.build_initialization_kwargs(
            configuration=request,
            scenario_name=scenario_name,
            scenario_class=scenario_class,
            objective_target=objective_target,
        )
        return await self._registry.create_and_estimate_async(
            scenario_name,
            scenario_params=request.scenario_params,
            target_is_configured=target_is_configured,
            **estimate_kwargs,
        )

    @staticmethod
    def _paginate_metadata(
        *,
        items: list[ScenarioMetadata],
        cursor: str | None,
        limit: int,
    ) -> tuple[list[ScenarioMetadata], bool]:
        """
        Apply scenario-name cursor pagination to registry metadata.

        Returns:
            tuple[list[ScenarioMetadata], bool]: The page and whether another page exists.
        """
        start_idx = 0
        if cursor:
            for index, item in enumerate(items):
                if item.registry_name == cursor:
                    start_idx = index + 1
                    break
        page = items[start_idx : start_idx + limit]
        return page, len(items) > start_idx + limit


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    """
    Get the process-wide Scenario service.

    Returns:
        ScenarioService: The cached service instance.
    """
    return ScenarioService()
