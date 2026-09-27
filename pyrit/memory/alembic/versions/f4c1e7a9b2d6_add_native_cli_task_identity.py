# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Keep CLI task identity separate from the GHCP binding identity.

Revision ID: f4c1e7a9b2d6
Revises: e8d4b2a6c1f0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from pyrit.memory.alembic.versions.c7e4d9a1b2f0_native_cyber_evidence import (
    _create_sqlite_native_evidence_triggers,
    _drop_sqlite_native_evidence_triggers,
)

revision: str = "f4c1e7a9b2d6"
down_revision: str | Sequence[str] | None = "e8d4b2a6c1f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable CLI task ID and version for existing GHCP episodes."""
    op.add_column("NativeCyberEpisodeEntries", sa.Column("task_id", sa.Unicode(128), nullable=True))
    op.add_column("NativeCyberEpisodeEntries", sa.Column("task_version", sa.Unicode(128), nullable=True))
    op.create_index(
        "ix_NativeCyberRawChunkEntries_stream_offset",
        "NativeCyberRawChunkEntries",
        ["stream_id", "byte_offset"],
    )


def downgrade() -> None:
    """Remove CLI-only task identity without altering GHCP run provenance."""
    op.drop_index("ix_NativeCyberRawChunkEntries_stream_offset", table_name="NativeCyberRawChunkEntries")
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    if is_sqlite:
        _drop_sqlite_native_evidence_triggers()
    with op.batch_alter_table("NativeCyberEpisodeEntries") as batch_op:
        batch_op.drop_column("task_version")
        batch_op.drop_column("task_id")
    if is_sqlite:
        _create_sqlite_native_evidence_triggers()
