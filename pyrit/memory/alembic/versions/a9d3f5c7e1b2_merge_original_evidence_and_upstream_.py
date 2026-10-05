# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Merge original evidence and upstream memory.

Revision ID: a9d3f5c7e1b2
Revises: 6ea3eb4b61c3, f4c1e7a9b2d6
Create Date: 2026-10-04 23:10:11.119854
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "a9d3f5c7e1b2"
down_revision: str | Sequence[str] | None = ("6ea3eb4b61c3", "f4c1e7a9b2d6")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this schema upgrade."""


def downgrade() -> None:
    """Revert this schema upgrade."""
