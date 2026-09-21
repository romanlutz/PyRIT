# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from sqlalchemy import update

from pyrit.analytics._profile_aggregation import ProfileAggregation
from pyrit.exceptions.analytics_exception import AnalyticsDataException, AnalyticsTimeoutException
from pyrit.memory.attack_analytics import AttackAnalyticsReader, RawAnalyticsReport
from pyrit.memory.memory_models import AttackResultEntry
from pyrit.models import (
    AttackAnalyticsDimension,
    AttackAnalyticsQuery,
    AttackAnalyticsResults,
    AttackOutcome,
)
from unit.memory.test_attack_analytics import control, make_result

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory


@pytest.fixture
def mixed_reader(sqlite_instance: SQLiteMemory) -> AttackAnalyticsReader:
    missing = make_result(index=5, outcome=AttackOutcome.ERROR, categories=["Unknown", ""])
    missing.atomic_attack_identifier = None
    results = [
        make_result(
            index=index,
            categories=["Privacy", "privacy", "", "Unknown"],
            converters=["Alpha", "ALPHA", ""],
            response_converters=["Beta"],
        )
        for index in (1, 2)
    ]
    results.extend(
        [
            make_result(
                index=3,
                outcome=AttackOutcome.FAILURE,
                categories=["privacy"],
                converters=["Alpha", "Gamma"],
                response_converters=["Beta", "BETA"],
            ),
            make_result(index=4, outcome=AttackOutcome.UNDETERMINED),
            missing,
            make_result(
                index=6,
                categories=["Éthique", "éthique"],
                converters=["Ünicode", "ünicode"],
                response_converters=["Other"],
            ),
            make_result(index=7, categories=["privacy"], converters=["Beta"], response_converters=["Gamma"]),
            make_result(index=8, categories=["legacy"]),
        ]
    )
    sqlite_instance.add_attack_results_to_memory(attack_results=results)
    legacy_identifier = {
        "children": {
            "attack": {
                "class_name": "LegacyAttack",
                "children": {
                    "request_converters": [{"class_name": "Alpha"}, {"class_name": "ALPHA"}, {}],
                    "response_converters": [],
                },
            }
        }
    }
    with sqlite_instance.engine.begin() as connection:
        connection.execute(
            update(AttackResultEntry)
            .where(AttackResultEntry.id == results[-2].attack_result_id)
            .values(atomic_attack_identifier_hash=None)
        )
        connection.execute(
            update(AttackResultEntry)
            .where(AttackResultEntry.id == results[-1].attack_result_id)
            .values(atomic_attack_identifier_hash=None, atomic_attack_identifier=legacy_identifier)
        )
    return AttackAnalyticsReader(memory=sqlite_instance)


@pytest.mark.parametrize(
    "options",
    [
        {"group_by": {"name": "targeted_harm_category"}, "group_limit": 3, "group_offset": 1},
        {"group_by": {"name": "converter_type"}, "group_limit": 3, "group_offset": 1},
        {"group_by": {"name": "attack_type"}, "group_limit": 1, "group_offset": 1},
        {
            "group_by": {"name": "targeted_harm_category"},
            "compare_by": {"name": "converter_type"},
            "axis_limit": 2,
        },
        {
            "group_by": {"name": "targeted_harm_category"},
            "compare_by": {"name": "attack_type"},
            "axis_limit": 2,
        },
        {"group_by": {"name": "attack_type"}, "compare_by": {"name": "converter_type"}},
        {
            "group_by": {"name": "converter_type"},
            "compare_by": {"name": "converter_type", "converter_direction": "response"},
            "axis_limit": 2,
        },
        {
            "group_by": {"name": "targeted_harm_category"},
            "compare_by": {"name": "converter_type"},
            "filters": {
                "dimensions": [
                    {"dimension": {"name": "converter_type"}, "values": [{"value": "Alpha"}, {"value": "Other"}]},
                    {"dimension": {"name": "converter_type"}, "values": [{"value": "Gamma"}]},
                ]
            },
        },
    ],
)
def test_profiles_match_sql_for_weighted_legacy_and_typed_memberships(
    mixed_reader: AttackAnalyticsReader, options: dict[str, Any]
) -> None:
    query = AttackAnalyticsQuery.model_validate(options)
    sql_report = mixed_reader.report(query=query, control=control())
    profile_report = mixed_reader.report(query=query, control=control(), use_compact_profiles=True)
    assert profile_report.profiles is not None
    ProfileAggregation.populate(report=profile_report, query=query, control=control())
    assert profile_report.counts == sql_report.counts
    assert profile_report.groups == sql_report.groups
    assert profile_report.rows == sql_report.rows
    assert profile_report.columns == sql_report.columns
    assert profile_report.cells == sql_report.cells
    assert profile_report.has_more_groups == sql_report.has_more_groups
    assert profile_report.axes_truncated == sql_report.axes_truncated
    assert profile_report.results.items == sql_report.results.items


@pytest.mark.parametrize(
    ("dimension", "raw", "expected"),
    [
        ("converter_type", None, {("missing", ""): None}),
        ("converter_type", "null", {("missing", ""): None}),
        ("converter_type", " \t[\r\n ] ", {("no_converters", ""): None}),
        ("targeted_harm_category", "[]", {("missing", ""): None}),
        ("converter_type", '[null, {}, {"class_name": null}]', {("missing", ""): None}),
        ("converter_type", '["", "Unknown", "unknown"]', {("value", ""): "", ("value", "unknown"): "Unknown"}),
        ("attack_type", "", {("value", ""): ""}),
        ("attack_type", "null", {("value", "null"): "null"}),
        (
            "targeted_harm_category",
            '["Éthique", "éthique", "ALPHA", "Alpha", "alpha"]',
            {("value", "Éthique"): "Éthique", ("value", "éthique"): "éthique", ("value", "alpha"): "ALPHA"},
        ),
    ],
)
def test_profile_keys_preserve_absence_blank_and_sqlite_case_rules(
    dimension: str, raw: str | None, expected: dict[tuple[str, str], str | None]
) -> None:
    values = ProfileAggregation._values(raw=raw, dimension=AttackAnalyticsDimension(name=dimension))
    assert {key: option.label for key, option in values.items()} == expected


@pytest.mark.parametrize("raw", ['{"class_name": "Alpha"}', '"Alpha"', "true", "[1]", "[true]", "not json"])
def test_profiles_reject_invalid_array_shapes_and_members(raw: str) -> None:
    with pytest.raises(AnalyticsDataException):
        ProfileAggregation._values(raw=raw, dimension=AttackAnalyticsDimension(name="converter_type"))


def _report(profiles: list[dict[str, Any]]) -> RawAnalyticsReport:
    return RawAnalyticsReport(
        counts={},
        groups=[],
        rows=[],
        columns=[],
        cells=[],
        has_more_groups=False,
        axes_truncated=False,
        results=AttackAnalyticsResults(items=[], has_more=False, next_cursor=None, computed_at=datetime.now(tz=UTC)),
        warnings=[],
        profiles=profiles,
    )


@pytest.mark.parametrize(
    "record",
    [
        {"source0": "[]", "weight": True, "outcome": "success"},
        {"source0": "[]", "weight": -1, "outcome": "success"},
        {"source0": "[]", "weight": "2", "outcome": "success"},
        {"source0": "[]", "weight": 1, "outcome": None},
        {"source0": 42, "weight": 1, "outcome": "success"},
    ],
)
def test_profiles_reject_invalid_weights_and_source_types(record: dict[str, Any]) -> None:
    with pytest.raises(AnalyticsDataException):
        ProfileAggregation.populate(
            report=_report([record]),
            query=AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name="converter_type")),
            control=control(),
        )


def test_profile_decoding_cache_is_request_local_not_result_caching() -> None:
    records = [
        {"source0": '["Alpha", "Alpha"]', "weight": 5, "outcome": "success"},
        {"source0": '["Alpha", "Alpha"]', "weight": 2, "outcome": "failure"},
        {"source0": '["ALPHA"]', "weight": 3, "outcome": "success"},
    ]
    query = AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name="converter_type"))
    report = _report(records)
    with patch.object(ProfileAggregation, "_values", wraps=ProfileAggregation._values) as decode:
        ProfileAggregation.populate(report=report, query=query, control=control())
        assert decode.call_count == 2
        assert report.groups[0].counts == {"success": 8, "failure": 2}
        assert report.groups[0].option.label == "ALPHA"
        records[0]["weight"] = 9
        ProfileAggregation.populate(report=report, query=query, control=control())
        assert decode.call_count == 4
        assert report.groups[0].counts == {"success": 12, "failure": 2}


def test_cancelled_profiles_stop_before_decoding() -> None:
    query_control = control()
    query_control.cancel()
    with patch.object(ProfileAggregation, "_profile", side_effect=AssertionError("Must not decode")):
        with pytest.raises(AnalyticsTimeoutException):
            ProfileAggregation.populate(
                report=_report([{"source0": "[]", "weight": 1, "outcome": "success"}]),
                query=AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name="converter_type")),
                control=query_control,
            )
