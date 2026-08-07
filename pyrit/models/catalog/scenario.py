# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario catalog and run-summary models.

These describe canonical PyRIT entities exposed over the REST catalog and
scenario-run endpoints; both the backend and external REST clients (the CLI
today) consume them. REST envelopes (pagination, list wrappers) stay in
``pyrit.backend.models``.

Validators that affect runtime behavior (``ge``, ``le``) remain on the
canonical models.
"""

from datetime import datetime
from enum import Enum
from math import prod
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pyrit.models.parameter import Parameter
from pyrit.models.results.scenario_result import ScenarioRunState
from pyrit.models.retry_event import RetryEvent

# Authoritative set of dataset seed filters exposed over the run request surface. Each entry
# is used verbatim as a ``MemoryInterface.get_seeds`` keyword argument, so a filter key IS the
# get_seeds kwarg. Every exposed filter must be a list-valued (Sequence) get_seeds parameter.
# Adding a filterable field is a one-line change here; the CLI ``--dataset-filters`` help text
# describes these keys, and this request model validates them server-side (covering the GUI too).
#
# Comma-list semantics differ per key because ``get_seeds`` treats each field differently, and
# that behavior lives in ``pyrit.memory`` (this layer cannot import it). As of today
# (see ``MemoryInterface.get_seeds`` / ``_add_list_conditions``):
#   - harm_categories -> AND + substring: a seed must be tagged with EVERY value, and each value
#     is a substring match (``cyber`` matches ``cyber_harm``). So ``harm_categories=cyber,violence``
#     is an intersection, not a union.
#   - data_types -> OR + exact: a seed matches ANY value, compared for exact equality. So
#     ``data_types=text,image_path`` is a union.
DATASET_FILTERS: frozenset[str] = frozenset({"harm_categories", "data_types"})


class ScenarioRunSizeStatus(str, Enum):
    """Confidence status for a scenario run-size estimate."""

    EXACT = "exact"
    CONDITIONAL = "conditional"
    UNAVAILABLE = "unavailable"


class ScenarioRunSizeFactor(BaseModel):
    """One labeled multiplicative factor in a run-size component."""

    label: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class ScenarioRunSizeComponent(BaseModel):
    """One additive component in a scenario run-size calculation."""

    label: str = Field(..., min_length=1)
    planned_executions: int = Field(..., ge=0)
    factors: list[ScenarioRunSizeFactor] = Field(..., min_length=1)
    is_baseline: bool = False
    caveat: str | None = None

    @model_validator(mode="after")
    def _validate_factor_product(self) -> "ScenarioRunSizeComponent":
        """
        Require the component total to equal its ordered factor product.

        Returns:
            ScenarioRunSizeComponent: The validated component.

        Raises:
            ValueError: If the component total differs from the factor product.
        """
        factor_product = prod(factor.count for factor in self.factors)
        if self.planned_executions != factor_product:
            raise ValueError(
                f"Component '{self.label}' planned_executions ({self.planned_executions}) "
                f"must equal its factor product ({factor_product})."
            )
        return self


class ScenarioDatasetSizeCap(BaseModel):
    """One configured cap that affects a dataset summary."""

    label: str = Field(..., min_length=1)
    count: int = Field(..., ge=1)


class ScenarioDatasetSummary(BaseModel):
    """Logical and effectively selected attack-group counts for one source."""

    name: str = Field(..., min_length=1)
    logical_group_count: int = Field(..., ge=0)
    selected_group_count: int = Field(..., ge=0)
    configured_caps: list[ScenarioDatasetSizeCap] = Field(default_factory=list)


class ScenarioRunSizeEstimate(BaseModel):
    """Structured estimate using persisted ``ScenarioRunPlan`` outer-unit semantics."""

    status: ScenarioRunSizeStatus
    total_planned_executions: int | None = Field(None, ge=0)
    components: list[ScenarioRunSizeComponent] = Field(default_factory=list)
    datasets: list[ScenarioDatasetSummary] = Field(default_factory=list)
    caveat: str | None = None
    retries_included: bool = False

    @field_validator("retries_included")
    @classmethod
    def _reject_retry_inclusion(cls, value: bool) -> bool:
        """
        Keep retries outside authoritative planned execution counts.

        Returns:
            bool: The validated false value.

        Raises:
            ValueError: If retry inclusion is requested.
        """
        if value:
            raise ValueError("Scenario run-size estimates exclude retries.")
        return value

    @model_validator(mode="after")
    def _validate_authoritative_total(self) -> "ScenarioRunSizeEstimate":
        """
        Validate totals according to the estimate confidence status.

        Returns:
            ScenarioRunSizeEstimate: The validated estimate.

        Raises:
            ValueError: If the status and authoritative total are inconsistent.
        """
        if self.status is ScenarioRunSizeStatus.UNAVAILABLE and self.total_planned_executions is not None:
            raise ValueError("Unavailable estimates cannot declare a total_planned_executions value.")
        if self.status is ScenarioRunSizeStatus.EXACT:
            component_total = sum(component.planned_executions for component in self.components)
            if self.total_planned_executions is None:
                raise ValueError("Exact estimates require total_planned_executions.")
            if self.total_planned_executions != component_total:
                raise ValueError(
                    f"Exact estimate total ({self.total_planned_executions}) "
                    f"must equal its additive component total ({component_total})."
                )
        return self


class RegisteredScenario(BaseModel):
    """Summary of a registered scenario."""

    scenario_name: str = Field(..., description="Scenario name  (e.g., 'foundry.red_team_agent')")
    scenario_type: str = Field(..., description="Scenario type identifier (e.g., 'RedTeamAgentScenario')")
    description: str = Field(..., description="Human-readable description of the scenario")
    description_markdown: str = Field(
        "",
        description="Dedented Markdown/MyST scenario description with paragraphs and block structure preserved",
    )
    default_technique: str = Field(..., description="Default technique name used when none specified")
    default_techniques: list[str] = Field(
        default_factory=list,
        description="Expanded concrete techniques selected when the caller omits techniques",
    )
    aggregate_techniques: list[str] = Field(
        ..., description="Aggregate techniques that combine multiple attack approaches"
    )
    aggregate_technique_expansions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Concrete ordered technique expansion for every aggregate selector",
    )
    all_techniques: list[str] = Field(..., description="All available concrete technique names")
    default_datasets: list[str] = Field(..., description="Default dataset names used by the scenario")
    default_dataset_summaries: list[ScenarioDatasetSummary] = Field(
        default_factory=list,
        description="Logical and effectively selected attack-group counts for the default configuration",
    )
    default_estimate: ScenarioRunSizeEstimate | None = Field(
        None,
        description="Cached run-size estimate for the scenario's default configuration",
    )
    baseline_policy: Literal["enabled", "disabled", "forbidden"] = Field(
        "enabled", description="Whether baseline execution is enabled, disabled, or forbidden"
    )
    include_baseline_by_default: bool = Field(True, description="Whether an omitted baseline flag includes it")
    supported_parameters: list[Parameter] = Field(
        default_factory=list, description="Scenario-declared custom parameters"
    )


class ScenarioRunConfiguration(BaseModel):
    """Launch-aligned scenario selection fields shared by runs and estimates."""

    techniques: list[str] | None = Field(None, description="Technique names to use (uses scenario default if omitted)")
    dataset_names: list[str] | None = Field(None, description="Dataset names to use (uses scenario default if omitted)")
    max_dataset_size: int | None = Field(None, ge=1, description="Maximum items per dataset")
    dataset_filters: dict[str, list[str]] | None = Field(
        None,
        description=(
            "Dataset seed filters keyed by field, applied before sampling. Accepted keys: harm_categories, data_types."
        ),
    )
    include_baseline: bool | None = Field(
        None, description="Override the scenario baseline default; forbidden scenarios reject true"
    )
    scenario_params: dict[str, Any] | None = Field(
        None,
        description="Custom parameters for the scenario (passed to scenario.set_params_from_args). "
        "Keys are parameter names declared by the scenario's supported_parameters().",
    )

    @field_validator("dataset_filters")
    @classmethod
    def _validate_dataset_filters(cls, value: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        """
        Reject any dataset-filter key not in the exposed ``DATASET_FILTERS`` allow-list.

        Runs for every request source (CLI and GUI), so the allow-list is enforced server-side.

        Args:
            value (dict[str, list[str]] | None): The submitted dataset filters.

        Returns:
            dict[str, list[str]] | None: The validated filters, unchanged.

        Raises:
            ValueError: If any key is not present in ``DATASET_FILTERS``.
        """
        for key in value or {}:
            if key not in DATASET_FILTERS:
                raise ValueError(f"Unknown dataset filter '{key}'. Allowed: {', '.join(sorted(DATASET_FILTERS))}.")
        return value


class ScenarioEstimateRequest(ScenarioRunConfiguration):
    """Request body for a side-effect-free scenario run-size preview."""

    target_name: str | None = Field(
        None,
        description="Registered objective target name when target capabilities affect the estimate",
    )


class RunScenarioRequest(ScenarioRunConfiguration):
    """Request body for starting a scenario run."""

    scenario_name: str = Field(..., description="Scenario name (e.g., 'foundry.red_team_agent')")
    target_name: str = Field(..., description="Name of a registered target from the TargetRegistry")
    initializers: list[str] | None = Field(
        None, description="Initializer names to run before scenario (e.g., ['target', 'load_default_datasets'])"
    )
    max_concurrency: int = Field(10, ge=1, le=100, description="Maximum concurrent operations")
    max_retries: int = Field(0, ge=0, le=20, description="Maximum retry attempts on failure")
    labels: dict[str, str] | None = Field(None, description="Labels to attach to memory entries")
    initializer_args: dict[str, dict[str, Any]] | None = Field(
        None,
        description="Per-initializer arguments keyed by initializer name. "
        "Each value is a dict of args passed to that initializer's set_params_from_args(). "
        "Example: {'target': {'endpoint': 'https://...'}}.",
    )
    scenario_result_id: str | None = Field(
        None,
        description="Optional ID of an existing ScenarioResult to resume. "
        "If provided, the scenario will resume from prior progress instead of starting fresh.",
    )


class AttackErrorSummary(BaseModel):
    """A single errored attack result surfaced in a run summary."""

    atomic_attack_name: str = Field(..., description="Atomic-attack cell that errored")
    objective: str = Field("", description="Objective that was being attempted")
    error_type: str | None = Field(None, description="Exception class name")
    error_message: str | None = Field(None, description="Exception message")
    total_retries: int = Field(0, ge=0, description="Retry attempts recorded for this attack")


class AttackRetrySummary(BaseModel):
    """Retry events recorded for one attack result, for near-real-time CLI display."""

    attack_result_id: str = Field(..., description="Stable ID of the attack result (used to de-duplicate)")
    atomic_attack_name: str = Field(..., description="Atomic-attack cell that retried")
    retries: list[RetryEvent] = Field(
        default_factory=list, description="Retry attempts, each with component role/name, endpoint, and exception"
    )


class ScenarioRunSummary(BaseModel):
    """Response for a scenario run (status + result details)."""

    scenario_result_id: str = Field(..., description="UUID of the ScenarioResult in memory")
    scenario_name: str = Field(..., description="Registry key of the scenario being run")
    scenario_registry_name: str | None = Field(None, description="Requested scenario registry key when available")
    scenario_version: int = Field(0, ge=0, description="Version of the scenario")
    status: ScenarioRunState = Field(..., description="Current run status")
    created_at: datetime = Field(..., description="When the run was created")
    updated_at: datetime = Field(..., description="When the run status last changed")
    error: str | None = Field(None, description="Error message if status is FAILED")
    error_type: str | None = Field(None, description="Exception class name if status is FAILED")
    techniques_used: list[str] = Field(default_factory=list, description="Technique names that were executed")
    total_attacks: int = Field(0, ge=0, description="Total number of attack results persisted for this run")
    completed_attacks: int = Field(0, ge=0, description="Number of attacks that reached a terminal outcome")
    objective_achieved_rate: int = Field(0, ge=0, le=100, description="Success rate as percentage (0-100)")
    failed_attacks: list[AttackErrorSummary] = Field(
        default_factory=list,
        description="Individual attack results that errored, surfaced regardless of overall run status",
    )
    attack_retries: list[AttackRetrySummary] = Field(
        default_factory=list,
        description="Per-attack retry events, surfaced as each attack result lands so the CLI can stream warnings",
    )
    total_retries: int = Field(
        0, ge=0, description="Total retry attempts recorded across all attack results (endpoint-stress signal)"
    )
    labels: dict[str, str] = Field(default_factory=dict, description="Labels attached to this run")
    completed_at: datetime | None = Field(None, description="When the scenario finished")
