# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Thin REST-facing delegation to SDK attack analytics."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pyrit.analytics import AttackResultAnalytics

if TYPE_CHECKING:
    from pyrit.memory.memory_interface import MemoryInterface
    from pyrit.models import (
        AttackAnalyticsFacetQuery,
        AttackAnalyticsFacets,
        AttackAnalyticsQuery,
        AttackAnalyticsReport,
        AttackAnalyticsResults,
        AttackAnalyticsResultsQuery,
    )


class AnalyticsService:
    """Expose SDK analytics without duplicating its metric or filtering rules."""

    def __init__(self, *, memory: MemoryInterface | None = None) -> None:
        """Initialize the reusable SDK entry point."""
        self._analytics = AttackResultAnalytics(memory=memory)

    async def query_async(self, *, query: AttackAnalyticsQuery, access_scope: str) -> AttackAnalyticsReport:
        """
        Delegate a report query.

        Returns:
            AttackAnalyticsReport: The SDK report unchanged.
        """
        return await self._analytics.query_async(query=query, access_scope=access_scope)

    async def results_async(self, *, query: AttackAnalyticsResultsQuery, access_scope: str) -> AttackAnalyticsResults:
        """
        Delegate a results-only page request.

        Returns:
            AttackAnalyticsResults: The SDK page unchanged.
        """
        return await self._analytics.results_async(query=query, access_scope=access_scope)

    async def facets_async(self, *, query: AttackAnalyticsFacetQuery, access_scope: str) -> AttackAnalyticsFacets:
        """
        Delegate an opened facet lookup.

        Returns:
            AttackAnalyticsFacets: The SDK facet options unchanged.
        """
        return await self._analytics.facets_async(query=query, access_scope=access_scope)

    def shutdown(self) -> None:
        """Stop the backend's analytics workers outside the event loop."""
        self._analytics.shutdown()


@lru_cache(maxsize=1)
def get_analytics_service() -> AnalyticsService:
    """
    Get the reusable analytics service after memory initialization.

    Returns:
        AnalyticsService: The process-local service instance.
    """
    return AnalyticsService()


def shutdown_analytics_service() -> None:
    """Release an initialized service without constructing an unused one."""
    if get_analytics_service.cache_info().currsize:
        get_analytics_service().shutdown()
        get_analytics_service.cache_clear()
