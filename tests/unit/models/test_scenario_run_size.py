# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the canonical Scenario run-size contract."""

import pytest
from pydantic import ValidationError

from pyrit.models import (
    ScenarioEstimateRequest,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimate,
    ScenarioRunSizeFactor,
    ScenarioRunSizeStatus,
)


def _component(*, total: int = 6) -> ScenarioRunSizeComponent:
    """Build a valid two-factor component."""
    return ScenarioRunSizeComponent(
        label="Technique population",
        planned_executions=total,
        factors=[
            ScenarioRunSizeFactor(label="techniques", count=2),
            ScenarioRunSizeFactor(label="logical groups", count=3),
        ],
    )


def test_component_requires_factor_product() -> None:
    """A component total is always explained by its ordered multiplicative factors."""
    with pytest.raises(ValidationError, match="must equal its factor product"):
        _component(total=5)


def test_exact_estimate_requires_additive_component_total() -> None:
    """Exact totals must equal the ordered additive component sum."""
    with pytest.raises(ValidationError, match="must equal its additive component total"):
        ScenarioRunSizeEstimate(
            status=ScenarioRunSizeStatus.EXACT,
            total_planned_executions=7,
            components=[_component()],
        )


def test_conditional_estimate_allows_nullable_authoritative_total() -> None:
    """Conditional estimates can explain candidates without claiming an authoritative total."""
    estimate = ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.CONDITIONAL,
        total_planned_executions=None,
        components=[_component()],
        caveat="A target selection is required.",
    )

    assert estimate.total_planned_executions is None
    assert estimate.components[0].planned_executions == 6
    assert estimate.retries_included is False


def test_estimate_request_carries_launch_aligned_fields() -> None:
    """The preview request accepts the launch selectors without execution-only controls."""
    request = ScenarioEstimateRequest(
        target_name="target",
        techniques=["prompt_sending"],
        dataset_names=["harmbench"],
        max_dataset_size=4,
        dataset_filters={"harm_categories": ["cyber"]},
        include_baseline=False,
        scenario_params={"num_jailbreaks": 2},
    )

    assert request.techniques == ["prompt_sending"]
    assert request.scenario_params == {"num_jailbreaks": 2}
    assert request.include_baseline is False
