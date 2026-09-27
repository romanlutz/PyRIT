# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""SQLite migration round-trips and SQL Server index-key portability."""

from __future__ import annotations

from alembic import command
from sqlalchemy import ForeignKeyConstraint, Integer, String, Unicode, UniqueConstraint, create_engine, inspect
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateIndex, CreateTable

from pyrit.memory.memory_models import (
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    NativeCyberToolEventEntry,
    NativeCyberTurnEntry,
    NativeCyberTurnMessagePieceEntry,
)
from pyrit.memory.migration import _make_config, check_schema_migrations, run_schema_migrations

_NATIVE_TABLES = (
    NativeCyberEpisodeEntry.__table__,
    NativeCyberTurnEntry.__table__,
    NativeCyberTurnMessagePieceEntry.__table__,
    NativeCyberEventEntry.__table__,
    NativeCyberToolEventEntry.__table__,
    NativeCyberRawStreamEntry.__table__,
    NativeCyberRawChunkEntry.__table__,
)


def test_native_cyber_migration_up_down_preserves_prior_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            config = _make_config(connection=connection)
            command.upgrade(config, "7a9c1e3f5b2d")
            old_tables = set(inspect(connection).get_table_names())

            command.upgrade(config, "head")
            current_tables = set(inspect(connection).get_table_names())
            assert {table.name for table in _NATIVE_TABLES} <= current_tables
            assert old_tables <= current_tables
            event_indexes = {
                index["name"]: index for index in inspect(connection).get_indexes("NativeCyberEventEntries")
            }
            assert "ix_NativeCyberEventEntries_run_turn_sequence" in event_indexes
            assert event_indexes["ix_NativeCyberEventEntries_observed_id"]["unique"] == 0
            raw_indexes = {
                index["name"]: index for index in inspect(connection).get_indexes("NativeCyberRawStreamEntries")
            }
            assert raw_indexes["ix_NativeCyberRawStreamEntries_observed_source"]["unique"] == 0
            assert {
                tuple(foreign_key["constrained_columns"])
                for foreign_key in inspect(connection).get_foreign_keys("NativeCyberToolEventEntries")
            } == {("run_id", "event_sequence")}
            trigger_names = set(
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_native_cyber_%'"
                ).scalars()
            )
            assert len(trigger_names) == 6

            command.downgrade(config, "7a9c1e3f5b2d")
            assert set(inspect(connection).get_table_names()) == old_tables
            assert not set(
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_native_cyber_%'"
                ).scalars()
            )
        run_schema_migrations(engine=engine, silent=True)
        check_schema_migrations(engine=engine, silent=True)
    finally:
        engine.dispose()


def test_separate_tool_result_policy_upgrade_preserves_existing_episodes() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            config = _make_config(connection=connection)
            command.upgrade(config, "c7e4d9a1b2f0")
            connection.exec_driver_sql(
                'INSERT INTO "NativeCyberEpisodeEntries" '
                "(run_id, binding_name, binding_version, started_at, required_raw_streams, "
                "raw_byte_limit, stored_raw_bytes, capture_gaps, optional_gaps, coverage_complete) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("pre-policy-run", "synthetic", "1", "2026-01-01 00:00:00", "[]", 1024, 0, "[]", "[]", 0),
            )
            connection.exec_driver_sql(
                'INSERT INTO "NativeCyberTurnEntries" (run_id, turn_index, started_at, capture_gaps) '
                "VALUES (?, ?, ?, ?)",
                ("pre-policy-run", 1, "2026-01-01 00:00:00", "[]"),
            )

            command.upgrade(config, "head")
            columns = {
                column["name"]: column for column in inspect(connection).get_columns("NativeCyberEpisodeEntries")
            }
            assert not columns["require_separate_tool_results"]["nullable"]
            assert connection.exec_driver_sql(
                "SELECT require_separate_tool_results, response_policy_version, artifact_only_allowed "
                'FROM "NativeCyberEpisodeEntries" WHERE run_id = ?',
                ("pre-policy-run",),
            ).one() == (0, 1, 0)
            assert (
                connection.exec_driver_sql(
                    'SELECT response_mode FROM "NativeCyberTurnEntries" WHERE run_id = ?',
                    ("pre-policy-run",),
                ).scalar_one()
                == "message_required"
            )

            command.downgrade(config, "c7e4d9a1b2f0")
            assert {
                "require_separate_tool_results",
                "response_policy_version",
                "artifact_only_allowed",
            }.isdisjoint({column["name"] for column in inspect(connection).get_columns("NativeCyberEpisodeEntries")})
            assert "response_mode" not in {
                column["name"] for column in inspect(connection).get_columns("NativeCyberTurnEntries")
            }
            assert (
                connection.exec_driver_sql(
                    'SELECT run_id FROM "NativeCyberEpisodeEntries" WHERE run_id = ?',
                    ("pre-policy-run",),
                ).scalar_one()
                == "pre-policy-run"
            )
            assert (
                connection.exec_driver_sql(
                    'SELECT turn_index FROM "NativeCyberTurnEntries" WHERE run_id = ?',
                    ("pre-policy-run",),
                ).scalar_one()
                == 1
            )
            assert (
                len(
                    set(
                        connection.exec_driver_sql(
                            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_native_cyber_%'"
                        ).scalars()
                    )
                )
                == 6
            )
            command.upgrade(config, "head")
        check_schema_migrations(engine=engine, silent=True)
    finally:
        engine.dispose()


def test_native_cyber_sql_server_schema_uses_bounded_index_keys() -> None:
    dialect = mssql.dialect()
    for table in _NATIVE_TABLES:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert f"CREATE TABLE [{table.name}]" in ddl
        indexed_columns = set(table.primary_key.columns)
        for index in table.indexes:
            indexed_columns.update(index.columns)
        for constraint in table.constraints:
            if isinstance(constraint, (UniqueConstraint, ForeignKeyConstraint)):
                indexed_columns.update(constraint.columns)
        for column in indexed_columns:
            if isinstance(column.type, (String, Unicode)):
                assert column.type.length is not None
                assert column.type.length <= 256
        for index in (*table.indexes, *(c for c in table.constraints if isinstance(c, UniqueConstraint))):
            key_bytes = sum(
                column.type.length * (2 if isinstance(column.type, Unicode) else 1)
                if isinstance(column.type, String)
                else 4
                if isinstance(column.type, Integer)
                else 16
                for column in index.columns
            )
            assert key_bytes <= 900
    assert "VARBINARY(max)" in str(CreateTable(NativeCyberRawChunkEntry.__table__).compile(dialect=dialect))
    observed_index = next(
        index
        for index in NativeCyberEventEntry.__table__.indexes
        if index.name == "ix_NativeCyberEventEntries_observed_id"
    )
    observed_ddl = str(CreateIndex(observed_index).compile(dialect=dialect))
    assert not observed_index.unique
    assert "CREATE INDEX" in observed_ddl
    assert "(run_id, observed_event_id, sequence)" in observed_ddl
    raw_source_index = next(
        index
        for index in NativeCyberRawStreamEntry.__table__.indexes
        if index.name == "ix_NativeCyberRawStreamEntries_observed_source"
    )
    assert not raw_source_index.unique
    assert "CREATE INDEX" in str(CreateIndex(raw_source_index).compile(dialect=dialect))
    assert NativeCyberEpisodeEntry.__table__.c.require_separate_tool_results.type.compile(dialect=dialect) == "BIT"
    assert NativeCyberEpisodeEntry.__table__.c.artifact_only_allowed.type.compile(dialect=dialect) == "BIT"
    assert NativeCyberEpisodeEntry.__table__.c.response_policy_version.type.compile(dialect=dialect) == "INTEGER"
    assert NativeCyberTurnEntry.__table__.c.response_mode.type.compile(dialect=dialect) == "VARCHAR(24)"
    assert NativeCyberToolEventEntry.__table__.c.phase.type.compile(dialect=dialect) == "VARCHAR(16)"
