# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario catalog models."""

from pyrit.models import (
    ScenarioDatasetSummary,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimate,
    ScenarioRunSizeFactor,
)


def test_scenario_catalog_models_serialize_wire_contract() -> None:
    """The catalog sizing models expose the stable frontend-facing shape."""
    estimate = ScenarioRunSizeEstimate(
        status="conditional",
        total=None,
        components=[
            ScenarioRunSizeComponent(
                label="technique matrix",
                count=None,
                factors=[
                    ScenarioRunSizeFactor(label="techniques", count=2),
                    ScenarioRunSizeFactor(label="selected seed groups", count=None),
                ],
            )
        ],
        caveat="Target capability determines the selected seed groups.",
    )
    dataset = ScenarioDatasetSummary(
        name="example_dataset",
        seed_group_count=4,
        selected_seed_group_count=2,
    )

    assert estimate.model_dump(mode="json") == {
        "status": "conditional",
        "total": None,
        "components": [
            {
                "label": "technique matrix",
                "count": None,
                "factors": [
                    {"label": "techniques", "count": 2},
                    {"label": "selected seed groups", "count": None},
                ],
            }
        ],
        "caveat": "Target capability determines the selected seed groups.",
    }
    assert dataset.model_dump(mode="json") == {
        "name": "example_dataset",
        "seed_group_count": 4,
        "selected_seed_group_count": 2,
    }
