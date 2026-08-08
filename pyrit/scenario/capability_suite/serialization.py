# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Canonical serialization, hashing, and versioned loading for capability-suite manifests.

Provides a stable byte representation (used for content-addressable manifest hashes),
strict JSON loading with unknown-field rejection (via the manifest models themselves),
and an explicit migration seam so old manifest JSON can be brought forward to the
schema version this build of PyRIT understands.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from pyrit.scenario.capability_suite.manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    CapabilitySuiteManifest,
)

if TYPE_CHECKING:
    from pathlib import Path

JSONDict = dict[str, Any]
ManifestMigration = Callable[[JSONDict], JSONDict]


class UnsupportedManifestVersionError(Exception):
    """Raised when a manifest's ``schema_version`` cannot be resolved to the current one."""


_MIGRATIONS: dict[int, ManifestMigration] = {}


def register_migration(*, from_version: int, migrate: ManifestMigration) -> None:
    """
    Register a migration transforming raw manifest JSON from ``from_version`` forward.

    ``migrate`` must return a new dict whose ``schema_version`` is greater than
    ``from_version`` (typically ``from_version + 1``); ``load_manifest_json`` applies
    registered migrations repeatedly until ``CURRENT_MANIFEST_SCHEMA_VERSION`` is reached.

    Raises:
        ValueError: If a migration is already registered for ``from_version``.
    """
    if from_version in _MIGRATIONS:
        raise ValueError(f"A migration from manifest schema_version {from_version} is already registered.")
    _MIGRATIONS[from_version] = migrate


def canonical_bytes(manifest: CapabilitySuiteManifest) -> bytes:
    """
    Return deterministic canonical JSON bytes for a manifest.

    Uses Pydantic's ``model_dump(mode="json")`` (deterministic enum/datetime/tuple
    conversion) followed by ``json.dumps`` with sorted keys and no incidental
    whitespace, so the same logical manifest always produces the same bytes
    regardless of field/dict insertion order.

    Returns:
        bytes: Canonical UTF-8 JSON bytes.
    """
    data = manifest.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_hash(manifest: CapabilitySuiteManifest) -> str:
    """
    Return the sha256 hex digest of a manifest's canonical bytes.

    Returns:
        str: A 64-character lowercase hex sha256 digest.
    """
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def dump_manifest_json(manifest: CapabilitySuiteManifest) -> JSONDict:
    """
    Return a plain JSON-serializable dict for a manifest, suitable for ``json.dump``.

    Returns:
        dict[str, Any]: A round-trip-safe JSON representation (field order follows
        model declaration order, not the sorted canonical order used for hashing).
    """
    return manifest.model_dump(mode="json")


def load_manifest_json(data: Mapping[str, Any] | str | bytes) -> CapabilitySuiteManifest:
    """
    Load, migrate, and strictly validate manifest JSON.

    Unknown fields anywhere in the manifest are rejected by the underlying Pydantic
    models (``extra="forbid"``); this function only handles the ``schema_version``
    migration seam before delegating to strict model validation.

    Returns:
        CapabilitySuiteManifest: The validated, immutable manifest.

    Raises:
        UnsupportedManifestVersionError: If ``schema_version`` is missing, newer than
            ``CURRENT_MANIFEST_SCHEMA_VERSION``, or has no registered migration path
            forward to it.
        ValueError: If parsed JSON does not contain a top-level object.
        pydantic.ValidationError: If the (migrated) JSON fails strict schema validation.
    """
    parsed: object = json.loads(data) if isinstance(data, (str, bytes)) else data
    if not isinstance(parsed, Mapping):
        raise ValueError("Capability-suite manifest JSON must contain an object at the top level.")
    raw = cast("JSONDict", dict(parsed))
    version = raw.get("schema_version")
    if not isinstance(version, int):
        raise UnsupportedManifestVersionError("Manifest JSON is missing an integer 'schema_version' field.")
    if version > CURRENT_MANIFEST_SCHEMA_VERSION:
        raise UnsupportedManifestVersionError(
            f"Manifest schema_version {version} is newer than the supported version {CURRENT_MANIFEST_SCHEMA_VERSION}."
        )
    while version < CURRENT_MANIFEST_SCHEMA_VERSION:
        migrate = _MIGRATIONS.get(version)
        if migrate is None:
            raise UnsupportedManifestVersionError(
                f"No migration is registered from manifest schema_version {version} to "
                f"{CURRENT_MANIFEST_SCHEMA_VERSION}."
            )
        raw = migrate(raw)
        migrated_version = raw.get("schema_version")
        if not isinstance(migrated_version, int) or migrated_version <= version:
            raise UnsupportedManifestVersionError(
                f"Migration from manifest schema_version {version} did not advance 'schema_version'."
            )
        version = migrated_version
    return CapabilitySuiteManifest.model_validate(raw)


def load_manifest_file(path: Path) -> CapabilitySuiteManifest:
    """
    Load and validate a UTF-8 JSON manifest file.

    Returns:
        CapabilitySuiteManifest: The validated, immutable manifest.
    """
    return load_manifest_json(path.read_text(encoding="utf-8"))
