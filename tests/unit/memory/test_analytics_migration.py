# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import String, create_engine, inspect, text
from sqlalchemy.dialects import mssql, sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

from pyrit.memory.memory_models import AttackResultEntry

PREVIOUS_REVISION = "7a9c1e3f5b2d"


def test_analytics_migration_is_the_single_successor_of_main():
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[3] / "pyrit" / "memory" / "alembic"))
    assert scripts.get_heads() == ["b6d8f0a2c4e1"]
    assert scripts.get_revision("b6d8f0a2c4e1").down_revision == PREVIOUS_REVISION


def configuration(connection):
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[3] / "pyrit" / "memory" / "alembic"))
    config.attributes["connection"] = connection
    return config


def insert_result(connection, *, identifier, outcome="success"):
    result_id = str(uuid.uuid4())
    connection.execute(
        text(
            'INSERT INTO "AttackResultEntries" '
            "(id, conversation_id, objective, outcome, timestamp, executed_turns, "
            "execution_time_ms, atomic_attack_identifier) "
            "VALUES (:id, 'shared-conversation', 'Synthetic objective', :outcome, '2026-01-01', 0, 0, :identifier)"
        ),
        {"id": result_id, "outcome": outcome, "identifier": json.dumps(identifier)},
    )
    return result_id


def test_analytics_migration_roundtrip_preserves_result_ids_and_generated_lookup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.sqlite'}")
    try:
        with engine.begin() as connection:
            config = configuration(connection)
            command.upgrade(config, PREVIOUS_REVISION)
            first = insert_result(connection, identifier={"hash": "a" * 64})
            second = insert_result(connection, identifier={"hash": "b" * 64}, outcome="failure")
            command.upgrade(config, "head")
            command.check(config)
            rows = connection.execute(
                text(
                    "SELECT id, atomic_attack_identifier_hash, resolved_atomic_attack_identifier_hash "
                    'FROM "AttackResultEntries" ORDER BY outcome DESC'
                )
            ).all()
            assert {row.id for row in rows} == {first, second}
            assert all(row.atomic_attack_identifier_hash is None for row in rows)
            assert {row.resolved_atomic_attack_identifier_hash for row in rows} == {"a" * 64, "b" * 64}
            indexes = {entry["name"] for entry in inspect(connection).get_indexes("AttackResultEntries")}
            assert "ix_AttackResultEntries_analytics_facts_sqlite" in indexes
            assert "ix_AttackResultEntries_analytics_labels_sqlite" in indexes
            assert not any(name.endswith("_mssql") for name in indexes)
            connection.execute(
                text('UPDATE "AttackResultEntries" SET atomic_attack_identifier = :value WHERE id = :id'),
                {"value": json.dumps({"hash": "c" * 64}), "id": first},
            )
            assert (
                connection.execute(
                    text('SELECT resolved_atomic_attack_identifier_hash FROM "AttackResultEntries" WHERE id = :id'),
                    {"id": first},
                ).scalar_one()
                == "c" * 64
            )
            command.downgrade(config, PREVIOUS_REVISION)
            columns = {column["name"] for column in inspect(connection).get_columns("AttackResultEntries")}
            assert "resolved_atomic_attack_identifier_hash" not in columns
            assert connection.execute(text('SELECT COUNT(*) FROM "AttackResultEntries"')).scalar_one() == 2
            command.upgrade(config, "head")
            command.check(config)
    finally:
        engine.dispose()


def test_analytics_migration_rejects_oversized_outcomes_before_altering_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'invalid.sqlite'}")
    try:
        with engine.begin() as connection:
            config = configuration(connection)
            command.upgrade(config, PREVIOUS_REVISION)
            result_id = insert_result(connection, identifier={}, outcome="unexpected-" * 3)
            with pytest.raises(ValueError, match="16 characters"):
                command.upgrade(config, "head")
            assert (
                connection.execute(
                    text('SELECT outcome FROM "AttackResultEntries" WHERE id = :id'), {"id": result_id}
                ).scalar_one()
                == "unexpected-" * 3
            )
            columns = {column["name"] for column in inspect(connection).get_columns("AttackResultEntries")}
            assert "resolved_atomic_attack_identifier_hash" not in columns
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mssql.dialect()])
def test_generated_lookup_and_index_ddl_are_dialect_specific(dialect):
    sql = str(CreateTable(AttackResultEntry.__table__).compile(dialect=dialect))
    assert ("json_extract" if dialect.name == "sqlite" else "JSON_VALUE") in sql
    assert "resolved_atomic_attack_identifier_hash" in sql
    for index in AttackResultEntry.__table__.indexes:
        if index.info.get("dialect") != dialect.name:
            continue
        ddl = str(CreateIndex(index).compile(dialect=dialect))
        assert "CREATE INDEX" in ddl
        for column in index.columns:
            if isinstance(column.type, String):
                assert column.type.length is not None
        if dialect.name == "mssql":
            assert "INCLUDE" in ddl
            assert "json_extract" not in ddl
            assert "targeted_harm_categories" not in {column.name for column in index.columns}
            assert "labels" not in {column.name for column in index.columns}


def test_mssql_migration_uses_bounded_computed_key_and_included_json_columns():
    migration = importlib.import_module("pyrit.memory.alembic.versions.b6d8f0a2c4e1_index_attack_analytics")
    operations = MagicMock()
    operations.get_bind.return_value.dialect.name = "mssql"
    operations.get_bind.return_value.execute.return_value.scalar_one.return_value = 0
    with patch.object(migration, "op", operations):
        migration.upgrade()
    column = operations.add_column.call_args.args[1]
    assert column.type.length == 64
    assert "JSON_VALUE" in str(column.computed.sqltext)
    facts, labels = operations.create_index.call_args_list
    assert facts.kwargs["mssql_include"] == ["targeted_harm_categories"]
    assert labels.kwargs["mssql_include"] == ["labels"]
    assert "targeted_harm_categories" not in facts.args[2]
    assert "labels" not in labels.args[2]
