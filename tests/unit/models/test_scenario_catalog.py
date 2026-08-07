# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for canonical scenario catalog models."""

import pytest
from pydantic import ValidationError

from pyrit.models import (
    ScenarioDefaultRunSizeEstimate,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimateStatus,
    ScenarioRunSizeFactor,
)


def test_exact_default_run_size_requires_component_total() -> None:
    """Exact estimates reject totals that do not match their additive components."""
    with pytest.raises(ValidationError, match="components total 6, not 7"):
        ScenarioDefaultRunSizeEstimate(
            status=ScenarioRunSizeEstimateStatus.Exact,
            total_attack_count=7,
            components=[
                ScenarioRunSizeComponent(
                    label="Techniques",
                    count=6,
                    factors=[
                        ScenarioRunSizeFactor(label="seed groups", count=3),
                        ScenarioRunSizeFactor(label="techniques", count=2),
                    ],
                )
            ],
        )


def test_default_run_size_serializes_versioned_api_shape() -> None:
    """The estimate exposes stable status, total, component, and factor fields."""
    estimate = ScenarioDefaultRunSizeEstimate(
        status=ScenarioRunSizeEstimateStatus.Exact,
        total_attack_count=6,
        components=[
            ScenarioRunSizeComponent(
                label="Techniques",
                count=6,
                factors=[
                    ScenarioRunSizeFactor(label="seed groups", count=3),
                    ScenarioRunSizeFactor(label="techniques", count=2),
                ],
            )
        ],
    )

    assert estimate.model_dump(mode="json") == {
        "version": 1,
        "status": "exact",
        "total_attack_count": 6,
        "components": [
            {
                "label": "Techniques",
                "count": 6,
                "factors": [
                    {"label": "seed groups", "count": 3},
                    {"label": "techniques", "count": 2},
                ],
                "note": None,
            }
        ],
        "datasets": [],
        "note": None,
    }
