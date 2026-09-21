# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pyrit.analytics import AttackResultAnalytics
from pyrit.exceptions.analytics_exception import AnalyticsDataException
from pyrit.memory import SQLiteMemory
from pyrit.memory.attack_analytics import AttackAnalyticsReader
from pyrit.models import (
    AttackAnalyticsDimension,
    AttackAnalyticsFilters,
    AttackAnalyticsQuery,
    AttackAnalyticsResultsQuery,
    AttackOutcome,
)
from unit.memory.test_attack_analytics import make_result, predicate


@pytest.fixture
def analytics(sqlite_instance):
    service = AttackResultAnalytics(memory=sqlite_instance)
    yield service
    service.shutdown()


@pytest.fixture
def mixed_results(sqlite_instance):
    outcomes = [AttackOutcome.SUCCESS] * 4 + [AttackOutcome.FAILURE] * 2
    outcomes += [AttackOutcome.UNDETERMINED] * 3 + [AttackOutcome.ERROR]
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[make_result(index=index, outcome=outcome) for index, outcome in enumerate(outcomes, 1)]
    )
    return sqlite_instance


def test_mixed_outcomes_use_decided_denominator(analytics, mixed_results):
    report = analytics.query()
    assert report.summary.total_results == 10
    assert report.summary.total_decided == 6
    assert report.summary.success_rate == pytest.approx(4 / 6)
    assert report.summary.decided_share == 0.6
    assert report.summary.outcome_shares[AttackOutcome.ERROR] == 0.1
    assert not report.outcome_filter_applied
    assert report.groups[0].statistics == report.summary
    assert len(report.results.items) == 10


def test_empty_report_is_valid_but_rate_unavailable(analytics):
    report = analytics.query()
    assert report.summary.total_results == 0
    assert report.summary.success_rate is None
    assert report.groups == []
    assert report.results.items == []
    assert report.summary.decided_share is None


@pytest.mark.parametrize(
    "outcome, count, rate",
    [
        (AttackOutcome.SUCCESS, 4, 1.0),
        (AttackOutcome.FAILURE, 2, 0.0),
        (AttackOutcome.ERROR, 1, None),
        (AttackOutcome.UNDETERMINED, 3, None),
    ],
)
def test_outcome_filtered_rate_is_annotated_everywhere(analytics, mixed_results, outcome, count, rate):
    report = analytics.query(query=AttackAnalyticsQuery(filters=AttackAnalyticsFilters(outcomes=[outcome])))
    assert report.outcome_filter_applied
    assert report.summary.total_results == count
    assert report.summary.success_rate == rate
    assert report.groups[0].statistics.success_rate == rate
    assert all(row.outcome == outcome for row in report.results.items)


def test_selecting_every_outcome_removes_annotation(analytics, mixed_results):
    report = analytics.query(query=AttackAnalyticsQuery(filters=AttackAnalyticsFilters(outcomes=list(AttackOutcome))))
    assert not report.outcome_filter_applied
    assert report.summary.total_results == 10


def test_drilldown_retains_existing_converter_constraint(analytics, sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(converters=["Alpha", "Gamma"]),
            make_result(index=2, converters=["Gamma"]),
            make_result(index=3, converters=["Beta"]),
        ]
    )
    filters = AttackAnalyticsFilters.model_validate({"dimensions": [predicate("converter_type", ["Alpha", "Beta"])]})
    report = analytics.query(
        query=AttackAnalyticsQuery(filters=filters, group_by=AttackAnalyticsDimension(name="converter_type"))
    )
    gamma = next(group for group in report.groups if group.key.value == "gamma")
    narrowed = filters.model_copy(update={"dimensions": [*filters.dimensions, *gamma.drilldown_filters]})
    results = analytics.results(query=AttackAnalyticsResultsQuery(filters=narrowed))
    assert len(results.items) == gamma.statistics.total_results == 1


@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.parametrize("budget", ["predicates", "values"])
@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_drilldown_budget_keeps_terminal_reports_and_results_usable(
    *, analytics: AttackResultAnalytics, sqlite_instance: SQLiteMemory, compare: bool, budget: str, remaining: int
) -> None:
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result(categories=["privacy"])])
    additional = 2 if compare else 1
    sizes = [1] * (16 - remaining) if budget == "predicates" else [100, 100, 100, 100, 100 - remaining]
    filters = AttackAnalyticsFilters.model_validate(
        {
            "dimensions": [
                {
                    "dimension": {"name": "label", "label_key": f"absent-{index}"},
                    "values": [{"kind": "missing"}, *({"value": f"value-{item}"} for item in range(size - 1))],
                }
                for index, size in enumerate(sizes)
            ]
        }
    )
    query = AttackAnalyticsQuery(
        filters=filters,
        compare_by=AttackAnalyticsDimension(name="targeted_harm_category") if compare else None,
    )
    report = analytics.query(query=query)
    assert report.summary.total_results == 1
    assert len(report.results.items) == 1
    assert report.filters == filters
    extra = report.cells[0].drilldown_filters if compare else report.groups[0].drilldown_filters
    narrowed = filters.model_copy(update={"dimensions": [*filters.dimensions, *extra]})
    if remaining < additional:
        assert report.drilldown_unavailable_reason is not None
        assert f"maximum {16 if budget == 'predicates' else 500}" in report.drilldown_unavailable_reason
        with pytest.raises(ValidationError):
            analytics.results(query=AttackAnalyticsResultsQuery(filters=narrowed))
    else:
        assert report.drilldown_unavailable_reason is None
        terminal = analytics.query(query=query.model_copy(update={"filters": narrowed}))
        assert terminal.summary.total_results == 1
        assert (terminal.drilldown_unavailable_reason is not None) == (remaining < additional * 2)
        assert terminal.filters.dimensions[: len(filters.dimensions)] == filters.dimensions
        assert len(analytics.results(query=AttackAnalyticsResultsQuery(filters=narrowed)).items) == 1


def test_matrix_has_explicit_empty_cells_and_sdk_rates(analytics, sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(operation="one", categories=["privacy"]),
            make_result(index=2, operation="two", categories=["safety"], outcome=AttackOutcome.FAILURE),
        ]
    )
    report = analytics.query(
        query=AttackAnalyticsQuery(
            group_by=AttackAnalyticsDimension(name="operation"),
            compare_by=AttackAnalyticsDimension(name="targeted_harm_category"),
        )
    )
    assert len(report.cells) == 4
    assert report.groups_overlap
    assert sum(cell.statistics.total_results == 0 for cell in report.cells) == 2
    empty = next(cell for cell in report.cells if cell.statistics.total_results == 0)
    assert empty.statistics.success_rate is None
    failed = next(cell for cell in report.cells if cell.statistics.failures)
    assert failed.statistics.success_rate == 0.0


def test_statistics_reject_unknown_stored_outcomes():
    with pytest.raises(AnalyticsDataException, match="unsupported"):
        AttackResultAnalytics._statistics({"unexpected": 1})


async def test_async_matches_sync_contract_async(analytics, mixed_results):
    report = await analytics.query_async()
    assert report.summary.total_results == 10
    assert report.summary.success_rate == pytest.approx(4 / 6)


async def test_coalesced_results_are_copied_for_each_caller_async(analytics, mixed_results):
    entered = threading.Event()
    release = threading.Event()
    read = analytics._reader.report

    def blocked_read(**kwargs):
        entered.set()
        assert release.wait(5)
        return read(**kwargs)

    with patch.object(analytics._reader, "report", side_effect=blocked_read) as reader:
        first = asyncio.create_task(analytics.query_async(access_scope="same-user"))
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(analytics.query_async(access_scope="same-user"))
        await asyncio.sleep(0)
        release.set()
        one, two = await asyncio.gather(first, second)
    assert reader.call_count == 1
    one.summary.successes = 999
    one.results.items.clear()
    assert two.summary.successes == 4
    assert len(two.results.items) == 10


def test_access_scope_and_operation_partition_coalescing():
    query = AttackAnalyticsQuery()
    first = AttackResultAnalytics._key(operation="report", query=query, access_scope="one")
    second = AttackResultAnalytics._key(operation="report", query=query, access_scope="two")
    third = AttackResultAnalytics._key(operation="results", query=query, access_scope="one")
    assert len(first) == 64
    assert len({first, second, third}) == 3


def test_shutdown_keeps_backend_controller_registered_until_workers_exit(analytics):
    controller = analytics._execution()
    entered = threading.Event()
    release = threading.Event()
    shutdown = controller.shutdown

    def paused_shutdown():
        entered.set()
        assert release.wait(5)
        shutdown()

    with patch.object(controller, "shutdown", side_effect=paused_shutdown):
        with ThreadPoolExecutor(max_workers=1) as pool:
            closing = pool.submit(analytics.shutdown)
            try:
                assert entered.wait(5)
                assert analytics._execution() is controller
            finally:
                release.set()
            closing.result(timeout=5)
    assert analytics._execution() is not controller


@pytest.mark.parametrize("compare", [None, "converter_type", "attack_type"])
@pytest.mark.parametrize("limit_name", ["MAX_COMPACT_PROFILES", "MAX_COMPACT_VALUE_LENGTH", "MAX_COMPACT_TOTAL_LENGTH"])
def test_compact_profiles_match_sql_fallback(analytics, sqlite_instance, compare, limit_name):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(categories=["Privacy", "privacy", "\u00dcnicode"], converters=["Alpha", "ALPHA"]),
            make_result(
                index=2, categories=["\u00dcnicode", "safety"], converters=["Beta"], outcome=AttackOutcome.FAILURE
            ),
            make_result(index=3, outcome=AttackOutcome.ERROR),
            make_result(index=4, categories=["safety"], outcome=AttackOutcome.UNDETERMINED),
        ]
    )
    query = AttackAnalyticsQuery(
        group_by=AttackAnalyticsDimension(name="targeted_harm_category"),
        compare_by=AttackAnalyticsDimension(name=compare) if compare else None,
        axis_limit=2,
        group_limit=2,
    )
    fast = analytics.query(query=query)
    with patch.object(AttackAnalyticsReader, limit_name, 0):
        sql = analytics.query(query=query)
    fields = {
        "summary",
        "groups",
        "rows",
        "columns",
        "cells",
        "groups_overlap",
        "axes_truncated",
        "has_more_groups",
        "next_group_offset",
        "outcome_filter_applied",
    }
    assert fast.model_dump(include=fields) == sql.model_dump(include=fields)
    for group in fast.groups:
        results = analytics.results(
            query=AttackAnalyticsResultsQuery(filters=AttackAnalyticsFilters(dimensions=group.drilldown_filters))
        )
        assert len(results.items) == group.statistics.total_results


def test_compact_profile_request_cache_does_not_stale_outcomes(analytics, sqlite_instance):
    result = make_result(categories=["privacy"], converters=["Alpha"])
    sqlite_instance.add_attack_results_to_memory(attack_results=[result])
    query = AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name="targeted_harm_category"))
    assert analytics.query(query=query).groups[0].statistics.successes == 1
    sqlite_instance.update_attack_result_by_id(
        attack_result_id=result.attack_result_id, update_fields={"outcome": AttackOutcome.FAILURE.value}
    )
    updated = analytics.query(query=query)
    assert updated.summary.total_results == 1
    assert updated.groups[0].statistics.failures == 1
    assert updated.groups[0].statistics.success_rate == 0.0
