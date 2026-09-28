# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Persist the seed template marker for side-effect-free browsing."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a4c6e8b0d2"
down_revision: str | None = "7a9c1e3f5b2d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable persisted template marker."""
    with op.batch_alter_table("SeedPromptEntries") as batch_op:
        batch_op.add_column(sa.Column("is_jinja_template", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Remove the persisted template marker."""
    with op.batch_alter_table("SeedPromptEntries") as batch_op:
        batch_op.drop_column("is_jinja_template")
