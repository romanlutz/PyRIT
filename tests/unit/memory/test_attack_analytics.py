# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import event, text, update
from sqlalchemy.dialects import mssql, sqlite

from pyrit.exceptions.analytics_exception import AnalyticsTimeoutException
from pyrit.memory.attack_analytics import AttackAnalyticsReader
from pyrit.memory.attack_analytics_query import AttackAnalyticsQueryCompiler
from pyrit.memory.memory_models import AttackIdentifierEntry, TargetIdentifierEntry
from pyrit.memory.query_control import QueryControl
from pyrit.models import (
    AtomicAttackIdentifier,
    AttackAnalyticsDimension,
    AttackAnalyticsFacetQuery,
    AttackAnalyticsFilters,
    AttackAnalyticsQuery,
    AttackAnalyticsResultsQuery,
    AttackIdentifier,
    AttackOutcome,
    AttackResult,
    ConverterIdentifier,
    TargetIdentifier,
)


def make_result(
    *,
    index=1,
    outcome=AttackOutcome.SUCCESS,
    operation="operation-a",
    categories=None,
    converters=None,
    response_converters=None,
    labels=None,
):
    return AttackResult(
        attack_result_id=str(uuid.UUID(int=index)),
        conversation_id="shared-main-conversation",
        objective=f"Synthetic objective {index}",
        operation=operation,
        operator="operator-a",
        labels=labels or {},
        targeted_harm_categories=categories or [],
        outcome=outcome,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        atomic_attack_identifier=AtomicAttackIdentifier.build(
            attack_identifier=AttackIdentifier(
                class_name="ProbeAttack",
                class_module="tests",
                objective_target=TargetIdentifier(class_name="MockTarget", class_module="tests", model_name="model-a"),
                request_converters=[
                    ConverterIdentifier(class_name=name, class_module="tests", params={"variant": position})
                    for position, name in enumerate(converters or [])
                ],
                response_converters=[
                    ConverterIdentifier(class_name=name, class_module="tests") for name in response_converters or []
                ],
            )
        ),
    )


def control():
    return QueryControl(deadline=time.monotonic() + 10)


def predicate(name, values, **options):
    return {
        "dimension": {"name": name, **options.pop("dimension_options", {})},
        "values": [{"value": value} if value is not None else {"kind": "missing"} for value in values],
        **options,
    }


def test_report_counts_result_ids_not_conversations(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[make_result(), make_result(index=2, outcome=AttackOutcome.FAILURE)]
    )
    result = AttackAnalyticsReader(memory=sqlite_instance).report(query=AttackAnalyticsQuery(), control=control())
    assert result.counts == {"failure": 1, "success": 1}
    assert len(result.results.items) == 2
    assert result.groups[0].counts == result.counts
    assert result.results.items[0].target_model == "model-a"
    assert result.results.items[0].attack_type == "ProbeAttack"


def test_multivalued_groups_and_cells_do_not_multiply_results(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(categories=["privacy", "privacy", "safety"], converters=["Alpha", "Alpha", "Beta"]),
            make_result(index=2, outcome=AttackOutcome.FAILURE, categories=["privacy"], converters=["Alpha"]),
        ]
    )
    query = AttackAnalyticsQuery(
        group_by=AttackAnalyticsDimension(name="targeted_harm_category"),
        compare_by=AttackAnalyticsDimension(name="converter_type"),
    )
    result = AttackAnalyticsReader(memory=sqlite_instance).report(query=query, control=control())
    cells = {(cell.option.key.value, cell.column.key.value): cell.counts for cell in result.cells}
    assert result.counts == {"failure": 1, "success": 1}
    assert cells[("privacy", "alpha")] == {"failure": 1, "success": 1}
    assert cells[("privacy", "beta")] == {"success": 1}
    assert cells[("safety", "alpha")] == {"success": 1}
    assert cells[("safety", "beta")] == {"success": 1}


def test_case_variants_within_one_result_have_one_membership(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[make_result(categories=["Privacy", "privacy"], converters=["Alpha", "ALPHA"])]
    )
    report = AttackAnalyticsReader(memory=sqlite_instance).report(
        query=AttackAnalyticsQuery(
            group_by=AttackAnalyticsDimension(name="targeted_harm_category"),
            compare_by=AttackAnalyticsDimension(name="converter_type"),
        ),
        control=control(),
    )
    assert len(report.cells) == 1
    assert report.cells[0].counts == {"success": 1}
    assert report.cells[0].option.key.value == "privacy"
    assert report.cells[0].column.key.value == "alpha"


def test_whitespace_empty_category_array_matches_missing_filter(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result()])
    with sqlite_instance.engine.begin() as connection:
        connection.execute(text("UPDATE AttackResultEntries SET targeted_harm_categories = '[   ]'"))
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    filters = AttackAnalyticsFilters.model_validate({"dimensions": [predicate("targeted_harm_category", [None])]})
    report = reader.report(
        query=AttackAnalyticsQuery(filters=filters, group_by=AttackAnalyticsDimension(name="targeted_harm_category")),
        control=control(),
    )
    assert report.counts == {"success": 1}
    assert report.groups[0].option.key.kind.value == "missing"


def test_additional_converter_predicate_keeps_existing_any_constraint(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(converters=["Alpha", "Gamma"]),
            make_result(index=2, converters=["Gamma"]),
            make_result(index=3, converters=["Beta"]),
        ]
    )
    filters = AttackAnalyticsFilters.model_validate(
        {"dimensions": [predicate("converter_type", ["Alpha", "Beta"]), predicate("converter_type", ["Gamma"])]}
    )
    result = AttackAnalyticsReader(memory=sqlite_instance).report(
        query=AttackAnalyticsQuery(filters=filters), control=control()
    )
    assert result.counts == {"success": 1}
    assert [row.attack_result_id for row in result.results.items] == [str(uuid.UUID(int=1))]


def test_response_converter_direction_and_all_matching(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(converters=["Alpha"], response_converters=["Beta", "Gamma"]),
            make_result(index=2, response_converters=["Beta"]),
        ]
    )
    filters = AttackAnalyticsFilters.model_validate(
        {
            "dimensions": [
                predicate(
                    "converter_type",
                    ["Beta", "Gamma"],
                    dimension_options={"converter_direction": "response"},
                    match_mode="all",
                )
            ]
        }
    )
    report = AttackAnalyticsReader(memory=sqlite_instance).report(
        query=AttackAnalyticsQuery(filters=filters), control=control()
    )
    assert report.counts == {"success": 1}
    assert report.results.items[0].response_converters == ["Beta", "Gamma"]
    assert report.results.items[0].request_converters == ["Alpha"]


def test_missing_values_do_not_collide_with_literal_labels(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[make_result(operation=None), make_result(index=2, operation="Not recorded")]
    )
    report = AttackAnalyticsReader(memory=sqlite_instance).report(query=AttackAnalyticsQuery(), control=control())
    assert {(group.option.key.kind.value, group.option.key.value) for group in report.groups} == {
        ("missing", None),
        ("value", "Not recorded"),
    }


def test_known_empty_converters_differ_from_missing_identifier(sqlite_instance):
    missing = make_result(index=2)
    missing.atomic_attack_identifier = None
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result(), missing])
    report = AttackAnalyticsReader(memory=sqlite_instance).report(
        query=AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name="converter_type")), control=control()
    )
    assert {group.option.key.kind.value for group in report.groups} == {"missing", "no_converters"}


def test_literal_label_paths_and_facet_search(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(labels={"a.b": "50%_done"}),
            make_result(index=2, labels={"a.b": "50xxdone"}),
            make_result(index=3, operation="operation-b", labels={"a.b": "different"}),
        ]
    )
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    filters = AttackAnalyticsFilters.model_validate(
        {"dimensions": [predicate("label", ["50%_done"], dimension_options={"label_key": "a.b"})]}
    )
    assert reader.report(query=AttackAnalyticsQuery(filters=filters), control=control()).counts == {"success": 1}
    facet = reader.facets(
        query=AttackAnalyticsFacetQuery(
            filters=filters, dimension=AttackAnalyticsDimension(name="label", label_key="a.b"), search="%_"
        ),
        control=control(),
    )
    assert [item.key.value for item in facet.items] == ["50%_done"]


def test_facet_excludes_its_own_filters_but_keeps_other_dimensions(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(operation="one", converters=["Alpha"]),
            make_result(index=2, operation="two", converters=["Alpha"]),
            make_result(index=3, operation="three", converters=["Beta"]),
        ]
    )
    filters = AttackAnalyticsFilters.model_validate(
        {"dimensions": [predicate("operation", ["one"]), predicate("converter_type", ["Alpha"])]}
    )
    values = AttackAnalyticsReader(memory=sqlite_instance).facets(
        query=AttackAnalyticsFacetQuery(filters=filters, dimension=AttackAnalyticsDimension(name="operation")),
        control=control(),
    )
    assert [item.key.value for item in values.items] == ["one", "two"]


def test_results_pagination_never_queries_aggregates_or_scores(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result(index=index) for index in range(1, 6)])
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    statements = []

    def record_statement(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(sqlite_instance.engine, "before_cursor_execute", record_statement)
    try:
        with patch.object(reader, "report", side_effect=AssertionError("Report must not run")):
            first = reader.results(query=AttackAnalyticsResultsQuery(limit=2), control=control())
            second = reader.results(
                query=AttackAnalyticsResultsQuery(limit=2, cursor=first.next_cursor), control=control()
            )
        assert len(first.items) == len(second.items) == 2
        assert not {item.attack_result_id for item in first.items} & {item.attack_result_id for item in second.items}
        assert all(
            "ScoreEntries" not in statement and "PromptMemoryEntries" not in statement for statement in statements
        )
        assert all("count(" not in statement.lower() for statement in statements)
    finally:
        event.remove(sqlite_instance.engine, "before_cursor_execute", record_statement)


def test_changed_filters_reject_stale_result_cursor(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result(), make_result(index=2)])
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    page = reader.results(query=AttackAnalyticsResultsQuery(limit=1), control=control())
    with pytest.raises(ValueError, match="stale"):
        reader.results(
            query=AttackAnalyticsResultsQuery(
                cursor=page.next_cursor, filters=AttackAnalyticsFilters(outcomes=[AttackOutcome.FAILURE])
            ),
            control=control(),
        )


def test_expired_control_prevents_database_access(sqlite_instance):
    with patch.object(sqlite_instance, "get_session", side_effect=AssertionError("Must not acquire")):
        with pytest.raises(AnalyticsTimeoutException):
            AttackAnalyticsReader(memory=sqlite_instance).report(
                query=AttackAnalyticsQuery(), control=QueryControl(deadline=0)
            )


def test_sqlite_execution_deadline_cleans_up_the_connection(sqlite_instance):
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    heavy = text(
        "WITH RECURSIVE work(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM work WHERE x < 10000) "
        "SELECT 'success', COUNT(*) FROM work a CROSS JOIN work b"
    )
    with patch.object(AttackAnalyticsQueryCompiler, "totals", return_value=heavy):
        with pytest.raises(AnalyticsTimeoutException):
            reader.report(
                query=AttackAnalyticsQuery(),
                control=QueryControl(deadline=time.monotonic() + 0.05),
            )
    assert reader.report(query=AttackAnalyticsQuery(), control=control()).counts == {}


def test_compacted_identifiers_retain_incomplete_projection_fallback(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(attack_results=[make_result()])
    with sqlite_instance.engine.begin() as connection:
        connection.execute(update(AttackIdentifierEntry).values(class_name=None))
        connection.execute(update(TargetIdentifierEntry).values(model_name=None, underlying_model_name=None))
    reader = AttackAnalyticsReader(memory=sqlite_instance)
    for name, expected in (("attack_type", "probeattack"), ("model", "model-a")):
        report = reader.report(
            query=AttackAnalyticsQuery(group_by=AttackAnalyticsDimension(name=name)), control=control()
        )
        assert report.groups[0].option.key.value == expected
        assert report.groups[0].counts == {"success": 1}


def test_label_and_converter_matrix_preserves_only_the_selected_label(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[
            make_result(labels={"team": "one", "case": "unique-a"}, converters=["Alpha"]),
            make_result(index=2, labels={"team": "one", "case": "unique-b"}, converters=["Alpha"]),
            make_result(index=3, labels={"team": "two"}, converters=["Beta"]),
        ]
    )
    report = AttackAnalyticsReader(memory=sqlite_instance).report(
        query=AttackAnalyticsQuery(
            group_by=AttackAnalyticsDimension(name="label", label_key="team"),
            compare_by=AttackAnalyticsDimension(name="converter_type"),
        ),
        control=control(),
    )
    cells = {(cell.option.key.value, cell.column.key.value): cell.counts for cell in report.cells}
    assert cells == {("one", "alpha"): {"success": 2}, ("two", "beta"): {"success": 1}}


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mssql.dialect()])
def test_queries_compile_with_backend_specific_json_and_no_score_joins(dialect):
    query = AttackAnalyticsQuery(
        group_by=AttackAnalyticsDimension(name="targeted_harm_category"),
        compare_by=AttackAnalyticsDimension(name="converter_type"),
    )
    compiler = AttackAnalyticsQueryCompiler(dialect=dialect.name, filters=query.filters)
    sql = str(compiler.matrix(query).compile(dialect=dialect))
    assert "ScoreEntries" not in sql and "PromptMemoryEntries" not in sql
    assert ("json_each" if dialect.name == "sqlite" else "OUTER APPLY OPENJSON") in sql
