# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for canonical scenario catalog models."""

import pytest
from pydantic import ValidationError

from pyrit.models import (
    ScenarioDatasetSummary,
    ScenarioDefaultRunSizeEstimate,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimateRequest,
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


def test_conditional_estimate_exposes_dataset_counts_structurally() -> None:
    """Conditionality and effective dataset selection are machine-readable."""
    estimate = ScenarioDefaultRunSizeEstimate(
        status=ScenarioRunSizeEstimateStatus.Conditional,
        datasets=[
            ScenarioDatasetSummary(
                name="harmbench",
                logical_seed_group_count=100,
                selected_seed_group_count=4,
                selection_note="The default selection uses 4 of 100 logical seed groups.",
            )
        ],
        note="The final total depends on target capabilities.",
    )

    payload = estimate.model_dump(mode="json")
    assert payload["status"] == "conditional"
    assert payload["total_attack_count"] is None
    assert payload["datasets"] == [
        {
            "name": "harmbench",
            "kind": "dataset",
            "logical_seed_group_count": 100,
            "selected_seed_group_count": 4,
            "selection_note": "The default selection uses 4 of 100 logical seed groups.",
        }
    ]


def test_estimate_request_reuses_dataset_filter_validation() -> None:
    """Configured estimates reject the same unsupported dataset filters as launches."""
    with pytest.raises(ValidationError, match="Unknown dataset filter 'unknown'"):
        ScenarioRunSizeEstimateRequest(dataset_filters={"unknown": ["value"]})
