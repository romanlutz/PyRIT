# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lightweight contracts for querying persisted attack outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from pyrit.models.results.attack_result import AttackOutcome


@dataclass
class AttackStats:
    """Statistics for attack analysis results."""

    success_rate: float | None
    total_decided: int
    successes: int
    failures: int
    undetermined: int
    errors: int


@dataclass
class AttackAnalyticsStatistics(AttackStats):
    """Outcome statistics with their population and display proportions."""

    total_results: int
    decided_share: float | None
    outcome_shares: dict[AttackOutcome, float]


class AttackResultSelection(str, Enum):
    """The identity used when selecting persisted results."""

    ALL_RESULTS = "all_results"
    LATEST_PER_CONVERSATION = "latest_per_conversation"


class AttackAnalyticsDimensionName(str, Enum):
    """Supported result metadata dimensions."""

    OPERATION = "operation"
    OPERATOR = "operator"
    TARGETED_HARM_CATEGORY = "targeted_harm_category"
    ATTACK_TYPE = "attack_type"
    CONVERTER_TYPE = "converter_type"
    OBJECTIVE_TARGET = "objective_target"
    MODEL = "model"
    SCENARIO = "scenario"
    LABEL = "label"


class AttackAnalyticsConverterDirection(str, Enum):
    """The converter pipeline recorded on an attack."""

    REQUEST = "request"
    RESPONSE = "response"


class AttackAnalyticsMatchMode(str, Enum):
    """How values within one dimension predicate are combined."""

    ANY = "any"
    ALL = "all"


class AttackAnalyticsValueKind(str, Enum):
    """Disambiguate real metadata from missing values and empty pipelines."""

    VALUE = "value"
    MISSING = "missing"
    NO_CONVERTERS = "no_converters"


class _AnalyticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttackAnalyticsDimension(_AnalyticsModel):
    """A metadata dimension, optionally selecting a label or converter direction."""

    name: AttackAnalyticsDimensionName
    label_key: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    converter_direction: AttackAnalyticsConverterDirection = AttackAnalyticsConverterDirection.REQUEST

    @model_validator(mode="after")
    def _validate_options(self) -> Self:
        if (self.name is AttackAnalyticsDimensionName.LABEL) != (self.label_key is not None):
            raise ValueError("label_key is required only for the label dimension")
        if (
            self.name is not AttackAnalyticsDimensionName.CONVERTER_TYPE
            and self.converter_direction is not AttackAnalyticsConverterDirection.REQUEST
        ):
            raise ValueError("converter_direction applies only to converter_type")
        if self.label_key in {"operator", "operation"}:
            raise ValueError("Use the dedicated operator or operation dimension")
        return self


class AttackAnalyticsValue(_AnalyticsModel):
    """An exact metadata value or an explicit absence bucket."""

    kind: AttackAnalyticsValueKind = AttackAnalyticsValueKind.VALUE
    value: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if (self.kind is AttackAnalyticsValueKind.VALUE) != (self.value is not None):
            raise ValueError("Only a value bucket has a string value")
        return self


class AttackAnalyticsFilter(_AnalyticsModel):
    """One predicate; separate predicates are always AND-combined."""

    dimension: AttackAnalyticsDimension
    values: list[AttackAnalyticsValue] = Field(min_length=1, max_length=100)
    match_mode: AttackAnalyticsMatchMode = AttackAnalyticsMatchMode.ANY

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        converter = self.dimension.name is AttackAnalyticsDimensionName.CONVERTER_TYPE
        if self.match_mode is AttackAnalyticsMatchMode.ALL and not converter:
            raise ValueError("ALL matching is supported only for converter_type")
        for value in self.values:
            if value.kind is AttackAnalyticsValueKind.NO_CONVERTERS and not converter:
                raise ValueError("The no_converters bucket applies only to converter_type")
            if (
                self.dimension.name
                in {
                    AttackAnalyticsDimensionName.OPERATION,
                    AttackAnalyticsDimensionName.OPERATOR,
                }
                and value.value is not None
                and len(value.value) > 128
            ):
                raise ValueError("Operator and operation values must not exceed 128 characters")
        return self


class AttackAnalyticsFilters(_AnalyticsModel):
    """The shared cohort selection for reports, facets, and result pages."""

    dimensions: list[AttackAnalyticsFilter] = Field(default_factory=list, max_length=16)
    outcomes: list[AttackOutcome] = Field(default_factory=list, max_length=4)
    updated_after: AwareDatetime | None = None
    updated_before: AwareDatetime | None = None

    @model_validator(mode="after")
    def _validate_filters(self) -> Self:
        if sum(len(predicate.values) for predicate in self.dimensions) > 500:
            raise ValueError("At most 500 dimension values may be selected")
        if (
            self.updated_after is not None
            and self.updated_before is not None
            and self.updated_after >= self.updated_before
        ):
            raise ValueError("updated_after must be before updated_before")
        self.outcomes = sorted(set(self.outcomes), key=lambda outcome: outcome.value)
        if set(self.outcomes) == set(AttackOutcome):
            self.outcomes = []
        return self


class AttackAnalyticsQuery(_AnalyticsModel):
    """A report request, including the initial lightweight result page."""

    filters: AttackAnalyticsFilters = Field(default_factory=AttackAnalyticsFilters)
    group_by: AttackAnalyticsDimension = Field(
        default_factory=lambda: AttackAnalyticsDimension(name=AttackAnalyticsDimensionName.OPERATION)
    )
    compare_by: AttackAnalyticsDimension | None = None
    group_limit: int = Field(default=15, ge=1, le=50)
    group_offset: int = Field(default=0, ge=0, le=100_000)
    axis_limit: int = Field(default=20, ge=1, le=20)
    result_limit: int = Field(default=25, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_dimensions(self) -> Self:
        if self.compare_by == self.group_by:
            raise ValueError("Heatmap dimensions must be different")
        return self


class AttackAnalyticsResultsQuery(_AnalyticsModel):
    """A results-only request that does not recalculate aggregates."""

    filters: AttackAnalyticsFilters = Field(default_factory=AttackAnalyticsFilters)
    cursor: str | None = Field(default=None, max_length=2048)
    limit: int = Field(default=25, ge=1, le=100)


class AttackAnalyticsFacetQuery(_AnalyticsModel):
    """A bounded lookup for one opened filter control."""

    filters: AttackAnalyticsFilters = Field(default_factory=AttackAnalyticsFilters)
    dimension: AttackAnalyticsDimension
    search: str = Field(default="", max_length=128)
    offset: int = Field(default=0, ge=0, le=100_000)
    limit: int = Field(default=50, ge=1, le=100)


class AttackAnalyticsOption(_AnalyticsModel):
    """A filter or axis option with a stable, typed key."""

    key: AttackAnalyticsValue
    label: str


class AttackAnalyticsGroup(AttackAnalyticsOption):
    """One grouped outcome aggregate and its additional drill-down predicate."""

    statistics: AttackAnalyticsStatistics
    drilldown_filters: list[AttackAnalyticsFilter]


class AttackAnalyticsCell(_AnalyticsModel):
    """A heatmap cell and the predicates that select its exact cohort."""

    row: AttackAnalyticsValue
    column: AttackAnalyticsValue
    statistics: AttackAnalyticsStatistics
    drilldown_filters: list[AttackAnalyticsFilter]


class AttackAnalyticsResultRow(_AnalyticsModel):
    """A result projection without conversation, score, or media hydration."""

    attack_result_id: str
    objective_preview: str
    outcome: AttackOutcome
    updated_at: AwareDatetime
    operation: str | None
    operator: str | None
    attack_type: str | None
    target_model: str | None
    target_identifier_hash: str | None
    scenario_result_id: str | None
    targeted_harm_categories: list[str]
    request_converters: list[str]
    response_converters: list[str]
    labels: dict[str, str]


class AttackAnalyticsResults(_AnalyticsModel):
    """One fresh result page, independent of the last report's refresh time."""

    items: list[AttackAnalyticsResultRow]
    has_more: bool
    next_cursor: str | None
    computed_at: AwareDatetime


class AttackAnalyticsFacets(_AnalyticsModel):
    """A single page of facet options."""

    items: list[AttackAnalyticsOption]
    has_more: bool
    next_offset: int | None
    computed_at: AwareDatetime


class AttackAnalyticsReport(_AnalyticsModel):
    """A coherent saved-result report; all numeric analytics come from the SDK."""

    filters: AttackAnalyticsFilters
    group_by: AttackAnalyticsDimension
    compare_by: AttackAnalyticsDimension | None
    summary: AttackAnalyticsStatistics
    outcome_filter_applied: bool
    groups_overlap: bool
    groups: list[AttackAnalyticsGroup]
    has_more_groups: bool
    next_group_offset: int | None
    rows: list[AttackAnalyticsOption]
    columns: list[AttackAnalyticsOption]
    cells: list[AttackAnalyticsCell]
    axes_truncated: bool
    results: AttackAnalyticsResults
    computed_at: AwareDatetime
    warnings: list[str] = Field(default_factory=list)
