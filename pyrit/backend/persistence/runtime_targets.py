# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
On-disk persistence for API-created targets so they survive a backend restart.

Targets created at runtime via ``POST /api/targets`` are normally lost when
the backend process exits (the in-memory ``TargetRegistry`` is process-local).
This module persists their non-secret metadata to a JSON file and exposes
helpers to append, remove, and load entries.

Security policy: credential values are **never** written to disk. At restart
the target's environment-variable / identity resolution is used to recover
the credential. If neither path succeeds the restored target is flagged
``needs_reconfiguration`` in the UI instead of failing startup.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DEFAULT_FILE_NAME = "runtime_targets.json"
_PYRIT_HOME_DIR = ".pyrit"
_OVERRIDE_ENV_VAR = "PYRIT_RUNTIME_TARGETS_FILE"

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "proxy_authorization",
        "sas_token",
        "secret",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES: tuple[str, ...] = (
    "_api_key",
    "_credential",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)


@dataclass
class RuntimeTargetEntry:
    """
    A single persisted target entry.

    Attributes:
        target_registry_name: The registry name assigned when the target was created.
        type: The target class name (matches ``CreateTargetRequest.type``).
        auth_mode: ``"api_key"`` or ``"identity"``.
        params: Constructor parameters with credential values stripped.
        created_at: ISO-8601 UTC timestamp of when the entry was persisted.
    """

    target_registry_name: str
    type: str
    auth_mode: Literal["api_key", "identity"]
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        """Backfill ``created_at`` and strip any stray sensitive keys from params."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.params = sanitize_params(self.params)


def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``params`` with credential-bearing keys removed.

    Args:
        params (dict[str, Any]): The constructor parameters from a create request.

    Returns:
        dict[str, Any]: A recursively sanitized copy safe to persist to disk.
    """
    sanitized: dict[str, Any] = {}
    for key, value in params.items():
        if _is_sensitive_key(key):
            continue
        if isinstance(value, dict):
            sanitized[key] = sanitize_params(value)
        elif isinstance(value, list):
            sanitized[key] = [_sanitize_value(item) for item in value]
        else:
            sanitized[key] = value
    return sanitized


def contains_sensitive_params(params: dict[str, Any]) -> bool:
    """
    Return whether a parameter tree contains an inline credential.

    Args:
        params (dict[str, Any]): Constructor parameters to inspect.

    Returns:
        bool: True when a credential-bearing key occurs at any depth.
    """
    return any(
        (_is_sensitive_key(key) and value not in (None, "")) or _contains_sensitive_value(value)
        for key, value in params.items()
    )


def _is_sensitive_key(key: str) -> bool:
    """Return whether a parameter or header name conventionally carries a credential."""
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _contains_sensitive_value(value: Any) -> bool:
    """Return whether a nested JSON-compatible value contains a credential."""
    if isinstance(value, dict):
        return contains_sensitive_params(value)
    if isinstance(value, list):
        return any(_contains_sensitive_value(item) for item in value)
    return False


def _sanitize_value(value: Any) -> Any:
    """
    Recursively remove sensitive keys from a nested JSON-compatible value.

    Returns:
        Any: The sanitized value.
    """
    if isinstance(value, dict):
        return sanitize_params(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _resolve_default_path() -> Path:
    """
    Resolve the runtime-targets file path.

    Honours the ``PYRIT_RUNTIME_TARGETS_FILE`` environment variable when set,
    otherwise defaults to ``~/.pyrit/runtime_targets.json``.

    Returns:
        Path: The absolute path to use for persistence.
    """
    override = os.environ.get(_OVERRIDE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / _PYRIT_HOME_DIR / _DEFAULT_FILE_NAME


def _apply_secure_permissions(path: Path) -> None:
    """
    Tighten file permissions to owner-only on POSIX. No-op on Windows.

    Args:
        path (Path): The file whose permissions should be tightened.
    """
    if sys.platform.startswith("win"):
        return
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("Failed to set 0600 permissions on %s: %s", path, exc)


class RuntimeTargetStore:
    """
    Persists API-created target metadata to a JSON file.

    Writes are serialized via an ``asyncio.Lock`` and use a temp-file +
    ``os.replace`` swap so a crash mid-write cannot corrupt the file.
    The schema is wrapped in a top-level envelope (``{"version": ..., "targets": [...]}``)
    so future format changes can migrate cleanly.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        """
        Args:
            path (Path | None): Override the file location. Defaults to
                ``$PYRIT_RUNTIME_TARGETS_FILE`` or ``~/.pyrit/runtime_targets.json``.
        """
        self._path = (path or _resolve_default_path()).resolve()
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        """The resolved path the store writes to."""
        return self._path

    async def load_async(self) -> list[RuntimeTargetEntry]:
        """
        Load all persisted runtime target entries.

        Returns:
            list[RuntimeTargetEntry]: Persisted entries. Returns an empty list if
            the file does not exist, is empty, has the wrong schema version, or
            contains malformed JSON (a warning is logged in the malformed case).
        """
        async with self._lock:
            return await asyncio.to_thread(self._load_locked)

    async def append_async(self, entry: RuntimeTargetEntry) -> None:
        """
        Append a new entry. Replaces any existing entry with the same
        ``target_registry_name`` so callers can safely re-create a target.

        Args:
            entry (RuntimeTargetEntry): The entry to persist. Credential values
                are stripped before write as defense in depth (``__post_init__``).
        """
        async with self._lock:
            await asyncio.to_thread(self._append_locked, entry)

    async def remove_async(self, target_registry_name: str) -> bool:
        """
        Remove the entry with the given registry name.

        Args:
            target_registry_name (str): The registry name to remove.

        Returns:
            bool: ``True`` if an entry was removed, ``False`` if no matching entry was found.
        """
        async with self._lock:
            return await asyncio.to_thread(self._remove_locked, target_registry_name)

    def _append_locked(self, entry: RuntimeTargetEntry) -> None:
        """Append or replace an entry while the caller holds the async lock."""
        entries = self._load_locked()
        entries = [existing for existing in entries if existing.target_registry_name != entry.target_registry_name]
        entries.append(entry)
        self._write_locked(entries)

    def _remove_locked(self, target_registry_name: str) -> bool:
        """
        Remove an entry while the caller holds the async lock.

        Returns:
            bool: True if an entry was removed, or False if it did not exist.
        """
        entries = self._load_locked()
        filtered = [entry for entry in entries if entry.target_registry_name != target_registry_name]
        if len(filtered) == len(entries):
            return False
        self._write_locked(filtered)
        return True

    def _load_locked(self) -> list[RuntimeTargetEntry]:
        """
        Load entries assuming the lock is already held.

        Returns:
            list[RuntimeTargetEntry]: Persisted entries, or ``[]`` if the file
            is missing, empty, malformed, or has an unsupported schema.
        """
        if not self._path.exists():
            return []

        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read runtime targets file %s: %s", self._path, exc)
            return []

        if not raw.strip():
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Runtime targets file %s is malformed (%s); ignoring its contents.",
                self._path,
                exc,
            )
            return []

        if not isinstance(data, dict):
            logger.warning(
                "Runtime targets file %s has unexpected shape (not an object); ignoring.",
                self._path,
            )
            return []

        version = data.get("version")
        if version != _SCHEMA_VERSION:
            logger.warning(
                "Runtime targets file %s has unsupported schema version %r; ignoring.",
                self._path,
                version,
            )
            return []

        raw_entries = data.get("targets", [])
        if not isinstance(raw_entries, list):
            logger.warning(
                "Runtime targets file %s 'targets' field is not a list; ignoring.",
                self._path,
            )
            return []

        entries: list[RuntimeTargetEntry] = []
        for raw_entry in raw_entries:
            entry = _entry_from_dict(raw_entry)
            if entry is not None:
                entries.append(entry)
        return entries

    def _write_locked(self, entries: list[RuntimeTargetEntry]) -> None:
        """Write entries atomically assuming the lock is already held."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": _SCHEMA_VERSION,
            "targets": [asdict(entry) for entry in entries],
        }

        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _apply_secure_permissions(tmp_path)
        os.replace(tmp_path, self._path)
        _apply_secure_permissions(self._path)


def _entry_from_dict(raw_entry: Any) -> RuntimeTargetEntry | None:
    """
    Build a ``RuntimeTargetEntry`` from a loaded JSON object.

    Skips entries missing required keys with a warning so a single bad entry
    cannot brick startup.

    Args:
        raw_entry (Any): A single element from the persisted ``targets`` list.

    Returns:
        RuntimeTargetEntry | None: The parsed entry, or ``None`` if invalid.
    """
    if not isinstance(raw_entry, dict):
        logger.warning("Skipping non-object entry in runtime targets file: %r", raw_entry)
        return None

    name = raw_entry.get("target_registry_name")
    target_type = raw_entry.get("type")
    if not isinstance(name, str) or not isinstance(target_type, str):
        logger.warning("Skipping runtime targets entry without required string fields: %r", raw_entry)
        return None

    auth_mode_value = raw_entry.get("auth_mode", "api_key")
    if auth_mode_value == "entra":
        auth_mode_value = "identity"
    if auth_mode_value not in ("api_key", "identity"):
        logger.warning("Skipping runtime targets entry with unknown auth_mode %r", auth_mode_value)
        return None
    auth_mode = cast("Literal['api_key', 'identity']", auth_mode_value)

    params = raw_entry.get("params") or {}
    if not isinstance(params, dict):
        logger.warning("Skipping runtime targets entry whose params is not an object: %r", raw_entry)
        return None

    created_at = raw_entry.get("created_at") or ""
    if not isinstance(created_at, str):
        created_at = ""

    return RuntimeTargetEntry(
        target_registry_name=name,
        type=target_type,
        auth_mode=auth_mode,
        params=dict(params),
        created_at=created_at,
    )


@lru_cache(maxsize=1)
def get_runtime_target_store() -> RuntimeTargetStore:
    """
    Return the process-wide ``RuntimeTargetStore`` singleton.

    Tests that need an isolated store can either set ``PYRIT_RUNTIME_TARGETS_FILE``
    before calling this, or construct ``RuntimeTargetStore(path=...)`` directly.

    Returns:
        RuntimeTargetStore: The cached singleton instance.
    """
    return RuntimeTargetStore()
