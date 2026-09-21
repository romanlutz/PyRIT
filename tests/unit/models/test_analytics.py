# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from pydantic import ValidationError

from pyrit.analytics import AttackStats as AnalyticsAttackStats
from pyrit.models import (
    AttackAnalyticsDimension,
    AttackAnalyticsFilters,
    AttackAnalyticsQuery,
    AttackAnalyticsResultsQuery,
    AttackAnalyticsValue,
    AttackOutcome,
    AttackStats,
)


def test_attack_stats_preserves_existing_import():
    assert AnalyticsAttackStats is AttackStats
    assert AttackStats(0.5, 2, 1, 1, 0, 0).success_rate == 0.5


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "missing", "value": "Unknown"},
        {"kind": "value", "value": None},
        {"kind": "no_converters", "value": "none"},
    ],
)
def test_analytics_value_rejects_ambiguous_keys(data):
    with pytest.raises(ValidationError):
        AttackAnalyticsValue.model_validate(data)


def test_real_unknown_label_is_not_missing():
    value = AttackAnalyticsValue(value="Unknown")
    missing = AttackAnalyticsValue(kind="missing")
    assert value != missing
    assert value.model_dump(mode="json") == {"kind": "value", "value": "Unknown"}


@pytest.mark.parametrize(
    "data",
    [
        {"name": "label"},
        {"name": "label", "label_key": "operation"},
        {"name": "label", "label_key": 'unsafe"]'},
        {"name": "operation", "label_key": "team"},
        {"name": "operator", "converter_direction": "response"},
        {"name": "arbitrary_sql"},
    ],
)
def test_analytics_dimension_rejects_invalid_options(data):
    with pytest.raises(ValidationError):
        AttackAnalyticsDimension.model_validate(data)


def test_all_outcomes_normalizes_to_unrestricted():
    filters = AttackAnalyticsFilters(outcomes=list(AttackOutcome))
    assert filters.outcomes == []


def test_outcome_filter_normalizes_duplicates():
    filters = AttackAnalyticsFilters(outcomes=[AttackOutcome.SUCCESS, AttackOutcome.SUCCESS])
    assert filters.outcomes == [AttackOutcome.SUCCESS]


def test_analytics_filter_keeps_additional_membership_constraints():
    filters = AttackAnalyticsFilters.model_validate(
        {
            "dimensions": [
                {
                    "dimension": {"name": "converter_type"},
                    "values": [{"value": "A"}, {"value": "B"}],
                },
                {
                    "dimension": {"name": "converter_type"},
                    "values": [{"value": "C"}],
                },
            ]
        }
    )
    assert len(filters.dimensions) == 2


@pytest.mark.parametrize(
    "data",
    [
        {"updated_after": "2026-01-02T00:00:00Z", "updated_before": "2026-01-01T00:00:00Z"},
        {"updated_after": "2026-01-01"},
        {
            "dimensions": [
                {
                    "dimension": {"name": "operation"},
                    "values": [{"value": "x" * 129}],
                }
            ]
        },
        {
            "dimensions": [
                {
                    "dimension": {"name": "targeted_harm_category"},
                    "values": [{"value": "privacy"}],
                    "match_mode": "all",
                }
            ]
        },
        {
            "dimensions": [
                {
                    "dimension": {"name": "operation"},
                    "values": [{"kind": "no_converters"}],
                }
            ]
        },
    ],
)
def test_analytics_filters_reject_invalid_requests(data):
    with pytest.raises(ValidationError):
        AttackAnalyticsFilters.model_validate(data)


@pytest.mark.parametrize("sizes, limit", [([1] * 17, "16"), ([100] * 5 + [1], "500")])
def test_analytics_filters_reject_effective_query_over_budget(*, sizes: list[int], limit: str) -> None:
    with pytest.raises(ValidationError, match=limit):
        AttackAnalyticsFilters.model_validate(
            {
                "dimensions": [
                    {
                        "dimension": {"name": "label", "label_key": f"label-{index}"},
                        "values": [{"value": str(value)} for value in range(size)],
                    }
                    for index, size in enumerate(sizes)
                ]
            }
        )


@pytest.mark.parametrize("field", ["group_limit", "axis_limit", "result_limit"])
def test_report_limits_are_bounded(field):
    with pytest.raises(ValidationError):
        AttackAnalyticsQuery.model_validate({field: 101})


def test_results_request_does_not_accept_chart_configuration():
    with pytest.raises(ValidationError):
        AttackAnalyticsResultsQuery.model_validate({"group_by": {"name": "operation"}})


def test_heatmap_requires_distinct_dimensions():
    with pytest.raises(ValidationError, match="different"):
        AttackAnalyticsQuery.model_validate({"compare_by": {"name": "operation"}})
