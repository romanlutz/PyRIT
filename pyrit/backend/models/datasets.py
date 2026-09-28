# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dataset models for the PyRIT API.

Datasets are seed prompt/objective collections provided by
``SeedDatasetProvider`` subclasses. These models describe the wire format for
listing available datasets.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from pyrit.backend.models.common import PaginationInfo


class DatasetInfo(BaseModel):
    """Metadata about a single available dataset."""

    name: str = Field(..., description="Dataset name (e.g., 'harmbench')")
    selection_key: str = Field(
        ...,
        description=(
            "Stable key used to select this dataset; named datasets use 'dataset:named:<name>' "
            "and the unnamed population uses 'dataset:unnamed'"
        ),
    )
    loaded: bool = Field(..., description="Whether this dataset currently has seeds in memory")
    provider_available: bool = Field(..., description="Whether a registered provider can load this dataset")
    logical_examples: int | None = Field(
        None, description="Number of logical examples represented by stored seed groups"
    )
    seed_pieces: int | None = Field(None, description="Number of stored seed/prompt pieces")
    objectives: int | None = Field(None, description="Number of stored objective seeds")
    modalities: list[str] = Field(default_factory=list, description="Distinct stored data types")
    harm_categories: list[str] = Field(default_factory=list, description="Distinct stored harm categories")
    has_unlabeled_harm_categories: bool = Field(
        False, description="Whether any stored seed is missing a harm-category label"
    )


class DatasetListResponse(BaseModel):
    """Response for listing available datasets."""

    items: list[DatasetInfo] = Field(..., description="List of available datasets")


class SeedExampleMemberView(BaseModel):
    """Persisted member of a logical seed example."""

    id: UUID
    prompt_group_id: UUID | None = None
    seed_type: str
    data_type: str
    value: str
    value_sha256: str | None = None
    role: str | None = None
    sequence: int | None = None
    name: str | None = None
    dataset_name: str | None = None
    harm_categories: list[str] | None = None
    description: str | None = None
    source: str | None = None
    authors: list[str] | None = None
    groups: list[str] | None = None
    date_added: datetime
    added_by: str
    metadata: dict[str, Any] | None = None
    parameters: list[str] | None = None
    is_jinja_template: bool | None = None


class SeedExampleSummary(BaseModel):
    """List representation of one complete logical seed example."""

    example_id: UUID
    dataset_name: str | None = None
    name: str | None = None
    preview: str
    preview_truncated: bool
    seed_ids: list[UUID]
    modalities: list[str]
    seed_types: list[str]
    piece_count: int
    objective_count: int
    harm_categories: list[str]
    has_unlabeled_harm: bool


class SeedExampleListResponse(BaseModel):
    """Paginated logical seed examples."""

    items: list[SeedExampleSummary]
    pagination: PaginationInfo


class SeedExampleDetailResponse(BaseModel):
    """Complete persisted logical seed example."""

    example_id: UUID
    dataset_name: str | None = None
    seed_ids: list[UUID]
    piece_count: int
    objective_count: int
    modalities: list[str]
    seed_types: list[str]
    harm_categories: list[str]
    has_unlabeled_harm: bool
    members: list[SeedExampleMemberView]
