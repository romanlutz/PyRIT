# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target API routes.

Provides endpoints for managing target instances.
Target types are set at app startup via initializers - you cannot add new types at runtime.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    TargetCatalogResponse,
    TargetListResponse,
)
from pyrit.backend.services.target_service import get_target_service
from pyrit.models.catalog.target import TargetInstance

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get(
    "",
    response_model=TargetListResponse,
    responses={
        500: {"model": ProblemDetail, "description": "Internal server error"},
    },
)
async def list_targets(  # pyrit-async-suffix-exempt
    limit: int = Query(50, ge=1, le=200, description="Maximum items per page"),
    cursor: str | None = Query(None, description="Pagination cursor (target_registry_name)"),
) -> TargetListResponse:
    """
    List target instances with pagination.

    Returns paginated target instances.

    Returns:
        TargetListResponse: Paginated list of target instances.
    """
    service = get_target_service()
    return await service.list_targets_async(limit=limit, cursor=cursor)


@router.get(
    "/catalog",
    response_model=TargetCatalogResponse,
    responses={
        500: {"model": ProblemDetail, "description": "Internal server error"},
    },
)
async def list_target_catalog() -> TargetCatalogResponse:  # pyrit-async-suffix-exempt
    """
    List all available target types from the backend target registry.

    Returns:
        TargetCatalogResponse: List of available target types.
    """
    service = get_target_service()
    return await service.list_target_catalog_async()


@router.post(
    "",
    response_model=TargetInstance,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {
            "model": ProblemDetail,
            "description": "Invalid target type or parameters",
        },
    },
)
async def create_target(
    request: CreateTargetRequest,
) -> TargetInstance:  # pyrit-async-suffix-exempt
    """
    Create a new target instance.

    Instantiates a target with the given type and parameters.
    The target becomes available for use in attacks.

    Note: Sensitive parameters (API keys, tokens) are filtered from the response.

    Returns:
        CreateTargetResponse: The created target instance details.
    """
    service = get_target_service()

    try:
        return await service.create_target_async(request=request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create target: {str(e)}",
        ) from e


@router.get(
    "/{target_registry_name}",
    response_model=TargetInstance,
    responses={
        404: {"model": ProblemDetail, "description": "Target not found"},
    },
)
async def get_target(
    target_registry_name: str,
) -> TargetInstance:  # pyrit-async-suffix-exempt
    """
    Get a target instance by registry name.

    Returns:
        TargetInstance: The target instance details.
    """
    service = get_target_service()

    target = await service.get_target_async(target_registry_name=target_registry_name)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target '{target_registry_name}' not found",
        )

    return target


@router.delete(
    "/{target_registry_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ProblemDetail, "description": "Target not found"},
        409: {
            "model": ProblemDetail,
            "description": "Target was registered by an initializer and cannot be deleted via the API",
        },
    },
)
async def delete_target(target_registry_name: str) -> Response:  # pyrit-async-suffix-exempt
    """
    Delete a runtime-created target instance.

    Only targets created via ``POST /targets`` are deletable. Targets registered
    by initializers at startup (e.g., via ``.pyrit_conf``) return 409 — to
    remove those, edit the configuration file and restart the backend.

    Returns:
        Response: 204 No Content on success.
    """
    service = get_target_service()
    try:
        await service.delete_target_async(target_registry_name=target_registry_name)
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)
