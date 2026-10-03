# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario API routes.

Provides endpoints for listing available scenarios, their metadata,
and managing scenario runs.

Route structure:
    /api/scenarios/catalog       — scenario catalog (list + detail)
    /api/scenarios/runs          — scenario execution lifecycle
"""

from fastapi import APIRouter, HTTPException, Query, Request, status
from starlette.concurrency import run_in_threadpool

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.scenarios import (
    ListRegisteredScenariosResponse,
    ScenarioRunListResponse,
)
from pyrit.backend.routes.common import parse_label_query_params
from pyrit.backend.services.original_run_admission import OriginalAdmissionError
from pyrit.backend.services.scenario_run_service import get_scenario_run_service
from pyrit.backend.services.scenario_service import get_scenario_service
from pyrit.models import ScenarioQueueSnapshot, ScenarioResult, ScenarioRunProgress, ScenarioRunState
from pyrit.models.catalog import (
    RegisteredScenario,
    RunScenarioRequest,
    ScenarioRunSizeEstimate,
    ScenarioRunSizeEstimateRequest,
    ScenarioRunSummary,
)

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


def _operator(*, request: Request) -> AuthenticatedUser | None:
    """
    Read an authenticated operator without treating disabled development auth as approval.

    Returns:
        AuthenticatedUser | None: The authenticated user, or none without Entra authentication.
    """
    user = getattr(request.state, "user", None)
    return user if isinstance(user, AuthenticatedUser) else None


def _admission_http_error(*, error: OriginalAdmissionError) -> HTTPException:
    """
    Expose only a finite, non-secret admission or source-verification reason.

    Returns:
        HTTPException: A safe HTTP status and finite reason code.
    """
    from pyrit.models.catalog.scenario import OriginalRunReason

    status_code = (
        status.HTTP_403_FORBIDDEN
        if error.reason is OriginalRunReason.OPERATOR_NOT_AUTHORIZED
        else status.HTTP_400_BAD_REQUEST
        if error.reason in (OriginalRunReason.ADMISSION_EXPIRED, OriginalRunReason.PROFILE_NOT_ADMITTED)
        else status.HTTP_503_SERVICE_UNAVAILABLE
        if error.reason is OriginalRunReason.RUNNER_NOT_CONFIGURED
        else status.HTTP_409_CONFLICT
    )
    return HTTPException(status_code=status_code, detail=error.reason.value)


# ============================================================================
# Scenario Catalog
# ============================================================================


@router.get(
    "/catalog",
    response_model=ListRegisteredScenariosResponse,
)
async def list_scenarios(  # pyrit-async-suffix-exempt
    http_request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum items per page"),
    cursor: str | None = Query(None, description="Pagination cursor (scenario_name to start after)"),
    include_estimates: bool = Query(True, description="Wait for default run-size estimates"),
) -> ListRegisteredScenariosResponse:
    """
    List all available scenarios.

    Returns scenario metadata including techniques, datasets, and defaults.
    Use GET /api/scenarios/catalog/{scenario_name} for full details on a specific scenario.

    Returns:
        ScenarioListResponse: Paginated list of scenario summaries.
    """
    service = get_scenario_service()
    return await service.list_scenarios_async(
        limit=limit,
        cursor=cursor,
        include_estimates=include_estimates,
        operator=_operator(request=http_request),
    )


@router.get(
    "/catalog/{scenario_name:path}",
    response_model=RegisteredScenario,
    responses={
        404: {"model": ProblemDetail, "description": "Scenario not found"},
    },
)
async def get_scenario(scenario_name: str, http_request: Request) -> RegisteredScenario:  # pyrit-async-suffix-exempt
    """
    Get details for a specific scenario.

    Args:
        scenario_name: Registry name of the scenario (e.g., 'foundry.red_team_agent').
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioSummary: Full scenario metadata.
    """
    service = get_scenario_service()

    scenario = await service.get_scenario_async(scenario_name=scenario_name, operator=_operator(request=http_request))
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario '{scenario_name}' not found",
        )

    return scenario


@router.post(
    "/catalog/{scenario_name}/estimate",
    response_model=ScenarioRunSizeEstimate,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid estimate configuration"},
        404: {"model": ProblemDetail, "description": "Scenario not found"},
    },
)
async def estimate_scenario_run_size(  # pyrit-async-suffix-exempt
    *,
    scenario_name: str,
    request: ScenarioRunSizeEstimateRequest,
    http_request: Request,
) -> ScenarioRunSizeEstimate:
    """
    Estimate a configured scenario without creating or persisting a run.

    Args:
        scenario_name: Registry name of the scenario.
        request: Techniques, datasets, baseline choice, and scenario parameters to preview.
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioRunSizeEstimate: Structured request-specific planned-unit estimate.
    """
    service = get_scenario_service()
    try:
        estimate = await service.estimate_scenario_run_size_async(
            scenario_name=scenario_name,
            request=request,
            operator=_operator(request=http_request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    if estimate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario '{scenario_name}' not found",
        )
    return estimate


# ============================================================================
# Scenario Runs
# ============================================================================


@router.post(
    "/runs",
    response_model=ScenarioRunSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid request (bad scenario/target/technique)"},
    },
)
async def start_scenario_run(  # pyrit-async-suffix-exempt
    request: RunScenarioRequest, http_request: Request
) -> ScenarioRunSummary:
    """
    Start a new scenario run as a background task.

    Initialization runs eagerly so configuration errors surface here, then the run
    itself continues in the background. Returns a scenario_result_id that can be
    polled for status.

    Args:
        request: Scenario run configuration.
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioRunSummary: Run metadata with PENDING status.
    """
    service = get_scenario_run_service()
    try:
        return await service.start_run_async(request=request, operator=_operator(request=http_request))
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from None


@router.get(
    "/runs",
    response_model=ScenarioRunListResponse,
)
async def list_scenario_runs(  # pyrit-async-suffix-exempt
    *,
    http_request: Request,
    scenario_names: list[str] | None = Query(
        None,
        description="Registered or persisted scenario names; repeated values are OR-matched.",
    ),
    run_statuses: list[ScenarioRunState] | None = Query(
        None,
        description="Run states; repeated values are OR-matched.",
    ),
    label: list[str] | None = Query(
        None,
        description="key:value labels; OR within a key and AND across keys.",
    ),
    limit: int = Query(100, ge=1, le=100, description="Maximum items per page"),
    cursor: str | None = Query(None, description="Opaque descending history cursor"),
) -> ScenarioRunListResponse:
    """
    List tracked scenario runs (most recent first).

    Args:
        http_request: Request with the authenticated operator, if present.
        scenario_names: Registered or persisted scenario names to match.
        run_statuses: Run states to match.
        label: Repeated key:value label filters.
        limit: Maximum number of runs to return.
        cursor: Opaque cursor from the previous page.

    Returns:
        ScenarioRunListResponse: Runs, most recent first.
    """
    service = get_scenario_run_service()
    try:
        return await run_in_threadpool(
            service.list_runs,
            scenario_names=scenario_names,
            statuses=run_statuses,
            labels=parse_label_query_params(label),
            limit=limit,
            cursor=cursor,
            operator=_operator(request=http_request),
        )
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None


@router.get(
    "/runs/queue",
    response_model=ScenarioQueueSnapshot,
)
async def get_scenario_run_queue(http_request: Request) -> ScenarioQueueSnapshot:  # pyrit-async-suffix-exempt
    """
    Get the active scenario and ordered FIFO waiting queue.

    Returns:
        ScenarioQueueSnapshot: Current in-process scheduler state.
    """
    return get_scenario_run_service().get_queue_snapshot(operator=_operator(request=http_request))


@router.get(
    "/runs/{scenario_result_id}",
    response_model=ScenarioRunSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Run not found"},
    },
)
async def get_scenario_run(  # pyrit-async-suffix-exempt
    scenario_result_id: str, http_request: Request
) -> ScenarioRunSummary:
    """
    Get the current status and result of a scenario run.

    Args:
        scenario_result_id: The scenario_result_id returned by POST /runs.
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioRunSummary: Current run status (and result if completed).
    """
    service = get_scenario_run_service()
    active_snapshot = service.snapshot_active_run(
        scenario_result_id=scenario_result_id, operator=_operator(request=http_request)
    )
    try:
        run = await run_in_threadpool(
            service.get_run_from_storage,
            scenario_result_id=scenario_result_id,
            active_error=active_snapshot.error,
            queue_position=active_snapshot.queue_position,
            active_scenario_result_id=active_snapshot.active_scenario_result_id,
            operator=_operator(request=http_request),
        )
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario run '{scenario_result_id}' not found",
        )
    return run


@router.get(
    "/runs/{scenario_result_id}/progress",
    response_model=ScenarioRunProgress,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid progress cursor"},
        404: {"model": ProblemDetail, "description": "Run not found"},
    },
)
async def get_scenario_run_progress(  # pyrit-async-suffix-exempt
    *,
    scenario_result_id: str,
    http_request: Request,
    since: str | None = Query(None, description="Opaque ascending progress cursor"),
    limit: int = Query(100, ge=1, le=500),
) -> ScenarioRunProgress:
    """
    Get canonical progress rollups and a refresh-safe page of result deltas.

    Returns:
        ScenarioRunProgress: Backend-owned rollups, the run plan, and ascending result deltas.
    """
    service = get_scenario_run_service()
    active_snapshot = service.snapshot_active_run(
        scenario_result_id=scenario_result_id, operator=_operator(request=http_request)
    )
    try:
        progress = await run_in_threadpool(
            service.get_run_progress_from_storage,
            scenario_result_id=scenario_result_id,
            since=since,
            limit=limit,
            active_group_ids=active_snapshot.active_group_ids,
            queue_position=active_snapshot.queue_position,
            active_scenario_result_id=active_snapshot.active_scenario_result_id,
            operator=_operator(request=http_request),
        )
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario run '{scenario_result_id}' not found",
        )
    return progress


@router.post(
    "/runs/{scenario_result_id}/cancel",
    response_model=ScenarioRunSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Run not found"},
        409: {"model": ProblemDetail, "description": "Run already in terminal state"},
    },
)
async def cancel_scenario_run(  # pyrit-async-suffix-exempt
    scenario_result_id: str, http_request: Request
) -> ScenarioRunSummary:
    """
    Cancel a running scenario.

    Args:
        scenario_result_id: The scenario_result_id to cancel.
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioRunSummary: Updated run with CANCELLED status.
    """
    service = get_scenario_run_service()
    try:
        result = await service.cancel_run_async(
            scenario_result_id=scenario_result_id, operator=_operator(request=http_request)
        )
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from None

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario run '{scenario_result_id}' not found",
        )
    return result


@router.get(
    "/runs/{scenario_result_id}/results",
    response_model=ScenarioResult,
    responses={
        404: {"model": ProblemDetail, "description": "Run not found"},
        409: {"model": ProblemDetail, "description": "Run not yet completed"},
    },
)
async def get_scenario_run_results(  # pyrit-async-suffix-exempt
    scenario_result_id: str, http_request: Request
) -> ScenarioResult:
    """
    Get detailed results for a completed scenario run.

    Args:
        scenario_result_id: The scenario_result_id.
        http_request: Request with the authenticated operator, if present.

    Returns:
        ScenarioResult: Detailed run results. FastAPI handles JSON serialization.
    """
    service = get_scenario_run_service()
    try:
        result = service.get_run_results(
            scenario_result_id=scenario_result_id, operator=_operator(request=http_request)
        )
    except OriginalAdmissionError as error:
        raise _admission_http_error(error=error) from None
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from None

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario run '{scenario_result_id}' not found",
        )
    return result
