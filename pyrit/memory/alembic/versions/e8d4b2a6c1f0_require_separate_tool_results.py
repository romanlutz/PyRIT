# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Declare native tool-result and artifact-only turn requirements.

Revision ID: e8d4b2a6c1f0
Revises: c7e4d9a1b2f0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from pyrit.memory.alembic.versions.c7e4d9a1b2f0_native_cyber_evidence import (
    _create_sqlite_native_evidence_triggers,
    _drop_sqlite_native_evidence_triggers,
)

revision: str = "e8d4b2a6c1f0"
down_revision: str | Sequence[str] | None = "c7e4d9a1b2f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Keep existing runs chat-required without a separate-result obligation."""
    op.add_column(
        "NativeCyberEpisodeEntries",
        sa.Column("require_separate_tool_results", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "NativeCyberEpisodeEntries",
        sa.Column("response_policy_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )
    op.add_column(
        "NativeCyberEpisodeEntries",
        sa.Column("artifact_only_allowed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "NativeCyberTurnEntries",
        sa.Column("response_mode", sa.String(24), nullable=False, server_default="message_required"),
    )


def downgrade() -> None:
    """Remove the versioned tool-result and response-mode requirements."""
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    if is_sqlite:
        _drop_sqlite_native_evidence_triggers()
    with op.batch_alter_table("NativeCyberTurnEntries") as batch_op:
        batch_op.drop_column("response_mode", mssql_drop_default=True)
    with op.batch_alter_table("NativeCyberEpisodeEntries") as batch_op:
        batch_op.drop_column("require_separate_tool_results", mssql_drop_default=True)
        batch_op.drop_column("response_policy_version", mssql_drop_default=True)
        batch_op.drop_column("artifact_only_allowed", mssql_drop_default=True)
    if is_sqlite:
        _create_sqlite_native_evidence_triggers()
