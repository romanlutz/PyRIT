# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""SQL dialect compilation coverage for the #2748 seed browsing seam."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.dialects import mssql, sqlite

from pyrit.common.pagination import encode_keyset_cursor
from pyrit.memory.azure_sql_memory import AzureSQLMemory
from pyrit.memory.memory_interface import (
    SeedExampleDatasetScope,
    _build_seed_example_query,
    _query_seed_example_members,
    _query_seed_example_page,
)
from pyrit.memory.sqlite_memory import SQLiteMemory


class _Result:
    def all(self):
        return []

    def scalar_one(self):
        return 0

    def scalars(self):
        return []


class _StatementCapture:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return _Result()


def _query(*, scope: SeedExampleDatasetScope, cursor: str | None = None):
    return _build_seed_example_query(
        dataset_scope=scope,
        limit=100,
        cursor=cursor,
        data_types=["url", "image_path"],
        harm_categories=["violence", "self_harm_%"],
        seed_types=["prompt", "objective"],
        value_search=r"literal_%\value",
    )


def _capture_statements(*, scope: SeedExampleDatasetScope, memory_type):
    initial = _query(scope=scope)
    cursor = encode_keyset_cursor(
        timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        identifier=str(uuid4()),
        fingerprint=initial.fingerprint,
    )
    query = _query(scope=scope, cursor=cursor)
    capture = _StatementCapture()
    _query_seed_example_page(
        session=capture,
        query=query,
        harm_condition_builder=object.__new__(memory_type)._seed_example_harm_condition,
    )
    _query_seed_example_members(
        session=capture,
        query=query,
        example_ids=[uuid4() for _ in range(100)],
    )
    return capture.statements


def test_seed_browsing_statements_compile_for_sql_server_and_sqlite():
    for scope in (SeedExampleDatasetScope.named("dataset"), SeedExampleDatasetScope.unnamed()):
        sql_server_statements = _capture_statements(scope=scope, memory_type=AzureSQLMemory)
        sqlite_statements = _capture_statements(scope=scope, memory_type=SQLiteMemory)
        sql_server_compiled = [
            statement.compile(dialect=mssql.dialect(), compile_kwargs={"render_postcompile": True})
            for statement in sql_server_statements
        ]
        sql_server_sql = [str(statement).lower() for statement in sql_server_compiled]
        sqlite_sql = [str(statement.compile(dialect=sqlite.dialect())).lower() for statement in sqlite_statements]

        assert all(sql for sql in sql_server_sql)
        assert all(sql for sql in sqlite_sql)
        assert all(len(statement.params) < 2100 for statement in sql_server_compiled)
        assert any("openjson" in sql for sql in sql_server_sql)
        assert any("json_each" in sql for sql in sqlite_sql)
        assert any("group by" in sql and "min" in sql for sql in sql_server_sql)
        assert any("group by" in sql and "min" in sql for sql in sqlite_sql)
        assert any("order by" in sql and "offset" in sql or "top" in sql for sql in sql_server_sql)
        assert any("order by" in sql and "limit" in sql for sql in sqlite_sql)
        assert any("sequence" in sql and "case" in sql for sql in sql_server_sql)
        page_sql = sql_server_sql[0]
        assert "lower(cast(coalesce" in page_sql
        assert "example_id_key <" in page_sql
        assert "example_id_key desc" in page_sql
        assert "openjson(case when" in page_sql
        assert "isjson([seedpromptentries_" in page_sql
        assert "left(ltrim([seedpromptentries_" in page_sql
        assert "else :" in page_sql
        assert "openjson([seedpromptentries_" not in page_sql
        assert "openjson(json_query(" not in page_sql
