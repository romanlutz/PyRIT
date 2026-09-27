# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Persist native cyber episodes, source events, tool links and bounded raw bytes.

Revision ID: c7e4d9a1b2f0
Revises: 7a9c1e3f5b2d
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "c7e4d9a1b2f0"
down_revision: str | Sequence[str] | None = "7a9c1e3f5b2d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.Uuid().with_variant(sa.CHAR(36), "sqlite")


def upgrade() -> None:
    """Create the append-only native evidence tables."""
    op.create_table(
        "NativeCyberEpisodeEntries",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("binding_name", sa.Unicode(128), nullable=False),
        sa.Column("binding_version", sa.Unicode(128), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("source_session_id", sa.Unicode(128)),
        sa.Column("environment_id", sa.Unicode(128)),
        sa.Column("simulated", sa.Boolean()),
        sa.Column("required_raw_streams", sa.JSON(), nullable=False),
        sa.Column("raw_byte_limit", sa.BigInteger(), nullable=False),
        sa.Column("stored_raw_bytes", sa.BigInteger(), nullable=False),
        sa.Column("conversation_id", sa.String(128)),
        sa.Column("capture_gaps", sa.JSON(), nullable=False),
        sa.Column("optional_gaps", sa.JSON(), nullable=False),
        sa.Column("coverage_complete", sa.Boolean(), nullable=False),
        sa.Column("finalized_at", sa.DateTime()),
        sa.Column("report_content_id", _UUID, sa.ForeignKey("ScorableContentEntries.id")),
        sa.Column("report_sha256", sa.String(64)),
        sa.Column("score_id", _UUID, sa.ForeignKey("ScoreEntries.id")),
    )
    op.create_table(
        "NativeCyberTurnEntries",
        sa.Column("run_id", sa.String(128), sa.ForeignKey("NativeCyberEpisodeEntries.run_id"), primary_key=True),
        sa.Column("turn_index", sa.Integer(), primary_key=True),
        sa.Column("source_turn_id", sa.Unicode(128)),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime()),
        sa.Column("observed_event_count", sa.Integer()),
        sa.Column("source_complete", sa.Boolean()),
        sa.Column("capture_gaps", sa.JSON(), nullable=False),
    )
    op.create_table(
        "NativeCyberTurnMessagePieceEntries",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("turn_index", sa.Integer(), primary_key=True),
        sa.Column("direction", sa.String(16), primary_key=True),
        sa.Column("position", sa.Integer(), primary_key=True),
        sa.Column("message_piece_id", _UUID, sa.ForeignKey("PromptMemoryEntries.id"), nullable=False),
        sa.Column("piece_sha256", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "turn_index"], ["NativeCyberTurnEntries.run_id", "NativeCyberTurnEntries.turn_index"]
        ),
        sa.UniqueConstraint("run_id", "message_piece_id", name="uq_native_cyber_turn_piece"),
    )
    op.create_index(
        "ix_NativeCyberTurnMessagePieceEntries_piece_id", "NativeCyberTurnMessagePieceEntries", ["message_piece_id"]
    )
    op.create_table(
        "NativeCyberEventEntries",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("observed_event_id", sa.Unicode(256)),
        sa.Column("observed_session_id", sa.Unicode(128)),
        sa.Column("observed_stream_id", sa.Unicode(128)),
        sa.Column("stream_offset", sa.BigInteger()),
        sa.Column("tool_call_id", sa.Unicode(128)),
        sa.Column("tool_phase", sa.String(16)),
        sa.Column("event_type", sa.Unicode(128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "turn_index"], ["NativeCyberTurnEntries.run_id", "NativeCyberTurnEntries.turn_index"]
        ),
    )
    op.create_index(
        "ix_NativeCyberEventEntries_run_turn_sequence",
        "NativeCyberEventEntries",
        ["run_id", "turn_index", "sequence"],
    )
    op.create_index(
        "ix_NativeCyberEventEntries_stream_offset",
        "NativeCyberEventEntries",
        ["run_id", "observed_stream_id", "stream_offset"],
    )
    op.create_index(
        "ix_NativeCyberEventEntries_observed_id",
        "NativeCyberEventEntries",
        ["run_id", "observed_event_id", "sequence"],
    )
    op.create_table(
        "NativeCyberToolEventEntries",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("call_id", sa.Unicode(128), primary_key=True),
        sa.Column("phase", sa.String(16), primary_key=True),
        sa.Column("event_sequence", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "event_sequence"], ["NativeCyberEventEntries.run_id", "NativeCyberEventEntries.sequence"]
        ),
    )
    op.create_index("ix_NativeCyberToolEventEntries_event", "NativeCyberToolEventEntries", ["run_id", "event_sequence"])
    op.create_table(
        "NativeCyberRawStreamEntries",
        sa.Column("stream_id", _UUID, primary_key=True),
        sa.Column("run_id", sa.String(128), sa.ForeignKey("NativeCyberEpisodeEntries.run_id"), nullable=False),
        sa.Column("turn_index", sa.Integer()),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("observed_source_id", sa.Unicode(128), nullable=False),
        sa.Column("tool_call_id", sa.Unicode(128)),
        sa.Column("received_bytes", sa.BigInteger(), nullable=False),
        sa.Column("stored_bytes", sa.BigInteger(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("source_complete", sa.Boolean()),
        sa.Column("expected_bytes", sa.BigInteger()),
        sa.Column("stored_sha256", sa.String(64)),
        sa.Column("observed_sha256", sa.String(64)),
        sa.Column("capture_gaps", sa.JSON(), nullable=False),
        sa.Column("closed_at", sa.DateTime()),
        sa.ForeignKeyConstraint(
            ["run_id", "turn_index"], ["NativeCyberTurnEntries.run_id", "NativeCyberTurnEntries.turn_index"]
        ),
    )
    op.create_index("ix_NativeCyberRawStreamEntries_run_turn", "NativeCyberRawStreamEntries", ["run_id", "turn_index"])
    op.create_index(
        "ix_NativeCyberRawStreamEntries_observed_source",
        "NativeCyberRawStreamEntries",
        ["run_id", "source", "kind", "observed_source_id"],
    )
    op.create_table(
        "NativeCyberRawChunkEntries",
        sa.Column("stream_id", _UUID, sa.ForeignKey("NativeCyberRawStreamEntries.stream_id"), primary_key=True),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("byte_offset", sa.BigInteger(), nullable=False),
        sa.Column("byte_length", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("data", sa.LargeBinary().with_variant(mssql.VARBINARY(None), "mssql"), nullable=False),
    )
    if op.get_bind().dialect.name == "sqlite":
        _create_sqlite_native_evidence_triggers()


def downgrade() -> None:
    """Drop native evidence tables without changing existing scores or content."""
    if op.get_bind().dialect.name == "sqlite":
        _drop_sqlite_native_evidence_triggers()
    op.drop_table("NativeCyberRawChunkEntries")
    op.drop_table("NativeCyberRawStreamEntries")
    op.drop_table("NativeCyberToolEventEntries")
    op.drop_table("NativeCyberEventEntries")
    op.drop_table("NativeCyberTurnMessagePieceEntries")
    op.drop_table("NativeCyberTurnEntries")
    op.drop_table("NativeCyberEpisodeEntries")


def _create_sqlite_native_evidence_triggers() -> None:
    """Keep existing pieces and a finalized report/score immutable despite SQLite FK defaults."""
    guarded = (
        ("prompt", "PromptMemoryEntries", "NativeCyberTurnMessagePieceEntries", "message_piece_id"),
        ("score", "ScoreEntries", "NativeCyberEpisodeEntries", "score_id"),
        ("content", "ScorableContentEntries", "NativeCyberEpisodeEntries", "report_content_id"),
    )
    for label, table_name, link_table, link_column in guarded:
        for action in ("UPDATE", "DELETE"):
            op.execute(
                sa.text(
                    f'CREATE TRIGGER "trg_native_cyber_{label}_immutable_{action.lower()}" '
                    f'BEFORE {action} ON "{table_name}" '
                    f'WHEN EXISTS (SELECT 1 FROM "{link_table}" WHERE "{link_column}" = OLD.id) '
                    "BEGIN SELECT RAISE(ABORT, 'native cyber evidence is immutable'); END"
                )
            )


def _drop_sqlite_native_evidence_triggers() -> None:
    """Remove evidence guards before dropping their reference tables."""
    for label in ("prompt", "score", "content"):
        for action in ("update", "delete"):
            op.execute(sa.text(f'DROP TRIGGER IF EXISTS "trg_native_cyber_{label}_immutable_{action}"'))
