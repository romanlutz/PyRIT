# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
One-time data migration for **Azure SQL**: rename legacy label keys.

Rewrites the JSON ``labels`` column (stored as NVARCHAR(MAX)) in
``PromptMemoryEntries`` and ``ScenarioResultEntries`` so that legacy
key names are replaced with their canonical equivalents:

    username, user_name  →  operator
    op_name              →  operation

If both a legacy key and the canonical key exist in the same row, the
canonical value is preserved and the legacy key is simply removed.

Prerequisites
-------------
* ``pyodbc`` with the ODBC Driver 18 for SQL Server installed.
* ``azure-identity`` for Entra ID (AAD) token authentication.
* Network access to the Azure SQL instance.

Usage
-----
Run from the repo root while the backend is **stopped**::

    # Entra ID (AAD) token auth — uses DefaultAzureCredential
    python build_scripts/migrate_legacy_label_keys_azure_sql.py \\
        --connection-string "Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=<db>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30"

    # Preview without writing
    python build_scripts/migrate_legacy_label_keys_azure_sql.py \\
        --connection-string "..." --dry-run

    # Batch size (default 500)
    python build_scripts/migrate_legacy_label_keys_azure_sql.py \\
        --connection-string "..." --batch-size 1000

Environment variables
~~~~~~~~~~~~~~~~~~~~~
Instead of ``--connection-string`` you can set
``AZURE_SQL_DB_CONNECTION_STRING`` (the same env var used by PyRIT).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

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


def _get_aad_token() -> bytes:
    """
    Acquire an Azure Entra ID access token for Azure SQL and pack it
    as a C struct compatible with pyodbc's SQL_COPT_SS_ACCESS_TOKEN.
    """
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print(
            "ERROR: azure-identity is required for Entra ID auth.\n"
            "       Install with: pip install azure-identity",
            file=sys.stderr,
        )
        sys.exit(1)

    credential = DefaultAzureCredential()
    token = credential.get_token("https://database.windows.net/.default")
    token_bytes = token.token.encode("utf-16-le")
    # Pack as: 4-byte length prefix + token bytes (what pyodbc expects)
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


def _connect(connection_string: str) -> Any:
    """
    Open a pyodbc connection to Azure SQL with AAD token auth.
    """
    try:
        import pyodbc
    except ImportError:
        print(
            "ERROR: pyodbc is required.\n"
            "       Install with: pip install pyodbc",
            file=sys.stderr,
        )
        sys.exit(1)

    SQL_COPT_SS_ACCESS_TOKEN = 1256  # noqa: N806
    token_struct = _get_aad_token()

    conn = pyodbc.connect(
        connection_string,
        attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
    )
    return conn


def _table_exists(cursor: Any, table: str) -> bool:
    """Check whether a table exists in the Azure SQL database."""
    cursor.execute(
        "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
        (table,),
    )
    return cursor.fetchone() is not None


def _build_legacy_key_filter(table: str) -> str:
    """
    Build a WHERE clause that matches rows containing any legacy key.

    Uses JSON_VALUE to check for the existence of legacy keys — a key
    exists if ``JSON_VALUE(labels, '$.key')`` returns non-NULL.
    """
    conditions = []
    for legacy_key in LEGACY_TO_CANONICAL:
        conditions.append(f"JSON_VALUE(labels, '$.{legacy_key}') IS NOT NULL")
    return (
        f"SELECT id, labels FROM {table} "
        f"WHERE ISJSON(labels) = 1 AND ({' OR '.join(conditions)})"
    )


def migrate_table(
    conn: Any,
    table: str,
    *,
    dry_run: bool = False,
    batch_size: int = 500,
) -> Tuple[int, int]:
    """
    Migrate legacy label keys in a single table.

    Reads matching rows in batches, normalizes the JSON in Python,
    and writes back the updated value.

    Returns (rows_checked, rows_updated).
    """
    cursor = conn.cursor()

    # Only fetch rows that actually have a legacy key — avoids scanning the
    # entire table when most rows are already canonical.
    query = _build_legacy_key_filter(table)
    cursor.execute(query)

    checked = 0
    updated = 0
    batch: List[Tuple[str, str]] = []  # (new_labels_json, row_id)

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for row_id, raw_labels in rows:
            checked += 1

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
                    batch.append((json.dumps(normalized), row_id))

        # Flush batch
        if batch and not dry_run:
            update_cursor = conn.cursor()
            update_cursor.executemany(
                f"UPDATE {table} SET labels = ? WHERE id = ?",  # noqa: S608
                batch,
            )
            update_cursor.close()
            batch = []

    # Final flush
    if batch and not dry_run:
        update_cursor = conn.cursor()
        update_cursor.executemany(
            f"UPDATE {table} SET labels = ? WHERE id = ?",  # noqa: S608
            batch,
        )
        update_cursor.close()

    cursor.close()
    return checked, updated


def main() -> None:
    import os

    default_conn = os.environ.get("AZURE_SQL_DB_CONNECTION_STRING", "")

    parser = argparse.ArgumentParser(
        description="Migrate legacy label keys to canonical names (Azure SQL)",
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        default=default_conn,
        help="pyodbc connection string for Azure SQL. "
        "Defaults to AZURE_SQL_DB_CONNECTION_STRING env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of rows per UPDATE batch (default: 500)",
    )
    args = parser.parse_args()

    if not args.connection_string:
        print(
            "ERROR: No connection string provided.\n"
            "       Use --connection-string or set AZURE_SQL_DB_CONNECTION_STRING.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Batch size: {args.batch_size}")
    print()

    conn = _connect(args.connection_string)
    try:
        cursor = conn.cursor()
        for table in TABLES_WITH_LABELS:
            if not _table_exists(cursor, table):
                print(f"  {table}: table not found, skipping")
                continue

            checked, updated = migrate_table(
                conn, table, dry_run=args.dry_run, batch_size=args.batch_size,
            )
            verb = "would update" if args.dry_run else "updated"
            print(f"  {table}: checked {checked} rows, {verb} {updated} rows")

        if not args.dry_run:
            conn.commit()
            print("\nMigration committed.")
        else:
            print("\nDry run complete — no changes written.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
