# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset API routes.

Provides an endpoint for listing available seed datasets. Datasets are
discovered from registered ``SeedDatasetProvider`` subclasses.
"""

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.datasets import (
    DatasetListResponse,
    SeedExampleDetailResponse,
    SeedExampleListResponse,
)
from pyrit.backend.services.dataset_service import (
    DatasetNotFoundError,
    InvalidDatasetSelectionError,
    get_dataset_service,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get(
    "",
    response_model=DatasetListResponse,
    responses={
        500: {"model": ProblemDetail, "description": "Internal server error"},
    },
)
async def list_datasets(loaded_only: bool = False) -> DatasetListResponse:  # pyrit-async-suffix-exempt
    """
    List all available datasets.

    Args:
        loaded_only (bool): When True, return only datasets that have seeds in memory.

    Returns:
        DatasetListResponse: Available datasets.
    """
    service = get_dataset_service()
    return await service.list_datasets_async(loaded_only=loaded_only)


@router.get(
    "/{selection_key}/seeds",
    response_model=SeedExampleListResponse,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid cursor or selection"},
        404: {"model": ProblemDetail, "description": "Dataset not found"},
        422: {"model": ProblemDetail, "description": "Invalid query parameters"},
    },
)
async def list_seed_examples(
    selection_key: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum examples per page"),
    cursor: str | None = Query(None, description="Opaque page cursor"),
    search: str | None = Query(None, description="Literal text search"),
    modality: list[str] | None = Query(None),
    harm_category: list[str] | None = Query(None),
    seed_type: list[str] | None = Query(None),
) -> SeedExampleListResponse:
    """
    List a page of logical seed examples.

    Returns:
        SeedExampleListResponse: The selected examples and pagination metadata.
    """
    service = get_dataset_service()
    try:
        return await service.list_seed_examples_async(
            selection_key=selection_key,
            limit=limit,
            cursor=cursor,
            search=search,
            data_types=modality,
            harm_categories=harm_category,
            seed_types=seed_type,
        )
    except (DatasetNotFoundError, InvalidDatasetSelectionError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/{selection_key}/seeds/{example_id}",
    response_model=SeedExampleDetailResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Dataset or example not found"},
        400: {"model": ProblemDetail, "description": "Invalid example identifier"},
    },
)
async def get_seed_example(selection_key: str, example_id: str) -> SeedExampleDetailResponse:
    """Return the complete persisted members of one logical seed example."""
    service = get_dataset_service()
    try:
        return await service.get_seed_example_async(selection_key=selection_key, example_id=example_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
