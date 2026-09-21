# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Read-only SDK analytics endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from pyrit.analytics import AttackResultAnalytics
from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.services.analytics_service import get_analytics_service
from pyrit.models import (
    AttackAnalyticsFacetQuery,
    AttackAnalyticsFacets,
    AttackAnalyticsQuery,
    AttackAnalyticsReport,
    AttackAnalyticsResults,
    AttackAnalyticsResultsQuery,
)

router = APIRouter(
    prefix="/analytics/attacks",
    tags=["analytics"],
    responses={
        400: {"model": ProblemDetail, "description": "Invalid or stale query cursor"},
        503: {"model": ProblemDetail, "description": "Analytics capacity is busy; retry"},
        504: {"model": ProblemDetail, "description": "The analytics query exceeded its deadline"},
    },
)


def _access_scope(request: Request) -> str:
    """
    Partition in-flight request sharing by the authenticated caller.

    Authentication is enforced by the application middleware. This key does not
    grant access or add row-level authorization; it only prevents cross-user reuse.

    Returns:
        str: A stable caller partition, or the unauthenticated local-development partition.
    """
    user = getattr(request.state, "user", None)
    return f"user:{user.oid}" if isinstance(user, AuthenticatedUser) else "local"


@router.post("/query", response_model=AttackAnalyticsReport)
async def query_analytics_async(
    *,
    query: AttackAnalyticsQuery,
    request: Request,
    service: Annotated[AttackResultAnalytics, Depends(get_analytics_service)],
) -> AttackAnalyticsReport:
    """
    Query saved outcome statistics and their first matching result page.

    Returns:
        AttackAnalyticsReport: The SDK-computed report.
    """
    return await service.query_async(query=query, access_scope=_access_scope(request))


@router.post("/results", response_model=AttackAnalyticsResults)
async def query_results_async(
    *,
    query: AttackAnalyticsResultsQuery,
    request: Request,
    service: Annotated[AttackResultAnalytics, Depends(get_analytics_service)],
) -> AttackAnalyticsResults:
    """
    Fetch a lightweight page without recalculating report statistics.

    Returns:
        AttackAnalyticsResults: Matching saved results and a filter-bound cursor.
    """
    return await service.results_async(query=query, access_scope=_access_scope(request))


@router.post("/facets", response_model=AttackAnalyticsFacets)
async def query_facets_async(
    *,
    query: AttackAnalyticsFacetQuery,
    request: Request,
    service: Annotated[AttackResultAnalytics, Depends(get_analytics_service)],
) -> AttackAnalyticsFacets:
    """
    Fetch bounded options for one opened filter.

    Returns:
        AttackAnalyticsFacets: The requested page of metadata values.
    """
    return await service.facets_async(query=query, access_scope=_access_scope(request))
