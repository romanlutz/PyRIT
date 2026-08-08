# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

from pyrit.scenario.capability_suite.manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseMessageManifest,
    LocalSandboxProviderManifestConfig,
    SuiteProvenance,
)
from pyrit.scenario.capability_suite.serialization import (
    _MIGRATIONS,
    UnsupportedManifestVersionError,
    canonical_bytes,
    dump_manifest_json,
    load_manifest_file,
    load_manifest_json,
    manifest_hash,
    register_migration,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> CapabilitySuiteManifest:
    return CapabilitySuiteManifest(
        suite_id="suite-1",
        name="Example suite",
        provenance=SuiteProvenance(source="unit-test"),
        sandbox_provider=LocalSandboxProviderManifestConfig(),
        cases=(
            CapabilityCaseManifest(
                case_id="case-1",
                objective="finish",
                messages=(CaseMessageManifest(role="user", content="hi"),),
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _clear_migrations():
    saved = dict(_MIGRATIONS)
    _MIGRATIONS.clear()
    yield
    _MIGRATIONS.clear()
    _MIGRATIONS.update(saved)


def test_load_manifest_json_roundtrip() -> None:
    manifest = _manifest()
    data = dump_manifest_json(manifest)
    loaded = load_manifest_json(data)
    assert loaded == manifest


def test_load_manifest_json_migrates_frozen_v1_shape() -> None:
    data = dump_manifest_json(_manifest())
    data["schema_version"] = 1
    case = data["cases"][0]
    for field in (
        "sandbox_tools_default_environment",
        "sandbox_tools_allowed_environments",
        "sandbox_tools_default_user",
        "sandbox_tools_allow_user_override",
        "sandbox_tools_include_file_tools",
        "runnable",
        "unsupported_reason",
    ):
        case.pop(field)
    case["scorers"] = [
        {
            "kind": "sandbox_command",
            "config": {"environment": "victim", "argv": ["echo", "ready"]},
        }
    ]

    loaded = load_manifest_json(data)

    assert loaded.schema_version == CURRENT_MANIFEST_SCHEMA_VERSION
    assert loaded.cases[0].sandbox_tools_include_file_tools is True
    assert loaded.cases[0].runnable is True
    assert loaded.cases[0].scorers[0].required_environments == ("victim",)


def test_load_manifest_json_text_and_file_roundtrip(tmp_path: Path) -> None:
    manifest = _manifest()
    text = json.dumps(dump_manifest_json(manifest))
    path = tmp_path / "suite.json"
    path.write_text(text, encoding="utf-8")
    assert load_manifest_json(text) == manifest
    assert load_manifest_file(path) == manifest


def test_canonical_bytes_deterministic_regardless_of_key_order() -> None:
    manifest = _manifest()
    data = dump_manifest_json(manifest)
    reordered = dict(reversed(list(data.items())))
    reloaded = CapabilitySuiteManifest.model_validate(reordered)
    assert canonical_bytes(manifest) == canonical_bytes(reloaded)


def test_manifest_hash_changes_when_content_changes() -> None:
    manifest = _manifest()
    changed = manifest.model_copy(update={"name": "Different name"})
    assert manifest_hash(manifest) != manifest_hash(changed)
    assert len(manifest_hash(manifest)) == 64


def test_load_manifest_json_missing_schema_version_raises() -> None:
    data = dump_manifest_json(_manifest())
    del data["schema_version"]
    with pytest.raises(UnsupportedManifestVersionError):
        load_manifest_json(data)


def test_load_manifest_json_future_version_raises() -> None:
    data = dump_manifest_json(_manifest())
    data["schema_version"] = CURRENT_MANIFEST_SCHEMA_VERSION + 1
    with pytest.raises(UnsupportedManifestVersionError):
        load_manifest_json(data)


def test_load_manifest_json_old_version_without_migration_raises() -> None:
    data = dump_manifest_json(_manifest())
    data["schema_version"] = 0
    with pytest.raises(UnsupportedManifestVersionError):
        load_manifest_json(data)


def test_register_migration_and_load_old_version() -> None:
    data = dump_manifest_json(_manifest())
    data["schema_version"] = 0

    def _migrate(raw):
        migrated = copy.deepcopy(raw)
        migrated["schema_version"] = CURRENT_MANIFEST_SCHEMA_VERSION
        return migrated

    register_migration(from_version=0, migrate=_migrate)
    loaded = load_manifest_json(data)
    assert loaded.schema_version == CURRENT_MANIFEST_SCHEMA_VERSION


def test_register_migration_rejects_duplicate_registration() -> None:
    register_migration(from_version=0, migrate=lambda raw: {**raw, "schema_version": 1})
    with pytest.raises(ValueError, match="already registered"):
        register_migration(from_version=0, migrate=lambda raw: {**raw, "schema_version": 1})


def test_register_migration_that_does_not_advance_version_raises() -> None:
    data = dump_manifest_json(_manifest())
    data["schema_version"] = 0
    register_migration(from_version=0, migrate=lambda raw: dict(raw))
    with pytest.raises(UnsupportedManifestVersionError, match="did not advance"):
        load_manifest_json(data)
