# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from pyrit.compat.inspect_ai import source
from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE, InspectProfileMismatchError

if TYPE_CHECKING:
    from pathlib import Path


def test_validate_source_is_read_only_and_checks_pinned_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkout"
    (root / "src" / "inspect_evals").mkdir(parents=True)
    license_path = root / "LICENSE"
    license_path.write_text("fixture license", encoding="utf-8")
    license_digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    responses = {
        ("rev-parse", "HEAD"): PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision,
        ("rev-parse", "HEAD^{tree}"): "fixture-tree",
        ("status", "--porcelain", "--untracked-files=all", "--ignored"): "",
    }
    monkeypatch.setattr(source, "_TREE_HASH", "fixture-tree")
    monkeypatch.setattr(source, "_LICENSE_SHA256", license_digest)
    monkeypatch.setattr(source, "_run_git", lambda *, root, args, timeout_seconds: responses[args])
    before = tuple(sorted(path.relative_to(root) for path in root.rglob("*")))

    verification = source.validate_inspect_source(source_root=root)

    assert verification.clean is True
    assert verification.license_sha256 == license_digest
    assert tuple(sorted(path.relative_to(root) for path in root.rglob("*"))) == before


def test_validate_source_rejects_wrong_revision_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkout"
    (root / "src" / "inspect_evals").mkdir(parents=True)
    monkeypatch.setattr(source, "_run_git", lambda **kwargs: "wrong-revision")

    with pytest.raises(InspectProfileMismatchError, match="does not match pinned revision"):
        source.validate_inspect_source(source_root=root)


def test_prepare_source_offline_requires_content_addressed_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision):
        source.prepare_inspect_source(cache_dir=tmp_path, offline=True)


def test_source_lock_is_released_and_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "source.lock"
    first = source._acquire_lock(lock_path=lock_path, timeout_seconds=1)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        source._acquire_lock(lock_path=lock_path, timeout_seconds=0.05)
    source._release_lock(first)

    second = source._acquire_lock(lock_path=lock_path, timeout_seconds=1)
    source._release_lock(second)
