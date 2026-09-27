# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Portable CLI task identity and raw-range index on existing native episodes."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateIndex, CreateTable

from pyrit.memory.memory_models import NativeCyberEpisodeEntry, NativeCyberRawChunkEntry
from pyrit.memory.migration import _make_config, check_schema_migrations
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart


def test_cli_task_identity_migration_preserves_existing_ghcp_episodes() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            config = _make_config(connection=connection)
            command.upgrade(config, "e8d4b2a6c1f0")
            connection.exec_driver_sql(
                'INSERT INTO "NativeCyberEpisodeEntries" '
                "(run_id, binding_name, binding_version, started_at, required_raw_streams, "
                "raw_byte_limit, stored_raw_bytes, capture_gaps, optional_gaps, coverage_complete) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("ghcp-older-run", "synthetic-binding", "1", "2026-01-01 00:00:00", "[]", 1024, 0, "[]", "[]", 0),
            )
            command.upgrade(config, "head")
            columns = {
                column["name"]: column for column in inspect(connection).get_columns("NativeCyberEpisodeEntries")
            }
            assert columns["task_id"]["nullable"] and columns["task_version"]["nullable"]
            assert connection.exec_driver_sql(
                'SELECT task_id, task_version FROM "NativeCyberEpisodeEntries" WHERE run_id = ?',
                ("ghcp-older-run",),
            ).one() == (None, None)
            indexes = {item["name"] for item in inspect(connection).get_indexes("NativeCyberRawChunkEntries")}
            assert "ix_NativeCyberRawChunkEntries_stream_offset" in indexes

            command.downgrade(config, "e8d4b2a6c1f0")
            assert {"task_id", "task_version"}.isdisjoint(
                {column["name"] for column in inspect(connection).get_columns("NativeCyberEpisodeEntries")}
            )
            assert (
                connection.exec_driver_sql(
                    'SELECT run_id FROM "NativeCyberEpisodeEntries" WHERE run_id = ?',
                    ("ghcp-older-run",),
                ).scalar_one()
                == "ghcp-older-run"
            )
            assert "ix_NativeCyberRawChunkEntries_stream_offset" not in {
                item["name"] for item in inspect(connection).get_indexes("NativeCyberRawChunkEntries")
            }
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


def test_cli_task_identity_and_raw_offset_index_compile_for_sql_server() -> None:
    dialect = mssql.dialect()
    table = NativeCyberEpisodeEntry.__table__
    assert table.c.task_id.type.compile(dialect=dialect) == "NVARCHAR(128)"
    assert table.c.task_version.type.compile(dialect=dialect) == "NVARCHAR(128)"
    ddl = str(CreateTable(table).compile(dialect=dialect))
    assert "task_id NVARCHAR(128)" in ddl and "task_version NVARCHAR(128)" in ddl
    index = next(
        item
        for item in NativeCyberRawChunkEntry.__table__.indexes
        if item.name == "ix_NativeCyberRawChunkEntries_stream_offset"
    )
    assert not index.unique
    assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))
    assert "(stream_id, byte_offset)" in str(CreateIndex(index).compile(dialect=dialect))


@pytest.mark.parametrize(
    ("task_id", "task_version"),
    [(None, "v1"), ("task", None), (" ", "v1"), ("task", " "), ("x" * 129, "v1")],
)
def test_cli_task_identity_must_be_complete_and_bounded(task_id: str | None, task_version: str | None) -> None:
    with pytest.raises(ValueError):
        NativeCyberEpisodeStart(
            run_id="synthetic-run",
            binding_name="synthetic-binding",
            binding_version="1",
            task_id=task_id,
            task_version=task_version,
        )
