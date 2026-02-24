# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
One-time data migration: rename legacy label keys in the database.

Rewrites the JSON ``labels`` column in ``PromptMemoryEntries`` and
``ScenarioResultEntries`` so that legacy key names are replaced with
their canonical equivalents:

    username, user_name  →  operator
    op_name              →  operation

If both a legacy key and the canonical key exist in the same row, the
canonical value is preserved and the legacy key is simply removed.

Usage
-----
Run from the repo root while the backend is **stopped**::

    python build_scripts/migrate_legacy_label_keys.py          # local SQLite (default)
    python build_scripts/migrate_legacy_label_keys.py --dry-run # preview without writing

The script works directly on the SQLite file at ``dbdata/pyrit.db``
(or the path given by ``--db``).  For Azure SQL, use corresponding
T-SQL UPDATE statements or adapt the ``--connection-string`` flag.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

LEGACY_TO_CANONICAL: Dict[str, str] = {
    "username": "operator",
    "user_name": "operator",
    "op_name": "operation",
}

TABLES_WITH_LABELS: List[str] = [
    "PromptMemoryEntries",
    "ScenarioResultEntries",
]


def _normalize_label_dict(labels: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
    """
    Rewrite legacy keys in a labels dict to canonical keys.

    Returns the (possibly-modified) dict and whether any change was made.
    """
    changed = False
    normalized: Dict[str, str] = {}

    # First pass: copy non-legacy keys
    for key, value in labels.items():
        if key not in LEGACY_TO_CANONICAL:
            normalized[key] = value

    # Second pass: map legacy keys, but don't overwrite existing canonical
    for key, value in labels.items():
        canonical = LEGACY_TO_CANONICAL.get(key)
        if canonical is not None:
            changed = True  # legacy key present → always a change (key removed)
            if canonical not in normalized:
                normalized[canonical] = value

    return normalized, changed


def migrate_table(
    conn: sqlite3.Connection,
    table: str,
    *,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Migrate legacy label keys in a single table.

    Returns (rows_checked, rows_updated).
    """
    cur = conn.execute(f"SELECT id, labels FROM {table} WHERE labels IS NOT NULL")  # noqa: S608
    rows = cur.fetchall()

    checked = 0
    updated = 0

    for row_id, raw_labels in rows:
        checked += 1

        # labels column is JSON — SQLite stores it as text
        if isinstance(raw_labels, str):
            try:
                labels = json.loads(raw_labels)
            except (json.JSONDecodeError, TypeError):
                continue
        elif isinstance(raw_labels, dict):
            labels = raw_labels
        else:
            continue

        if not isinstance(labels, dict):
            continue

        normalized, changed = _normalize_label_dict(labels)
        if changed:
            updated += 1
            if not dry_run:
                conn.execute(
                    f"UPDATE {table} SET labels = ? WHERE id = ?",  # noqa: S608
                    (json.dumps(normalized), row_id),
                )

    return checked, updated


def main() -> None:
    default_db = Path(__file__).resolve().parent.parent / "dbdata" / "pyrit.db"

    parser = argparse.ArgumentParser(description="Migrate legacy label keys to canonical names")
    parser.add_argument(
        "--db",
        type=str,
        default=str(default_db),
        help=f"Path to SQLite database file (default: {default_db})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    conn = sqlite3.connect(str(db_path))
    try:
        for table in TABLES_WITH_LABELS:
            # Check table exists
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                print(f"  {table}: table not found, skipping")
                continue

            checked, updated = migrate_table(conn, table, dry_run=args.dry_run)
            print(f"  {table}: checked {checked} rows, {'would update' if args.dry_run else 'updated'} {updated} rows")

        if not args.dry_run:
            conn.commit()
            print("\nMigration committed.")
        else:
            print("\nDry run complete — no changes written.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
