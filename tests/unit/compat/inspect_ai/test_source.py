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


def test_prepare_source_repairs_invalid_owned_checkout_online(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    checkout = tmp_path / revision / "checkout"
    checkout.mkdir(parents=True)
    dirty_file = checkout / "untrusted.py"
    dirty_file.write_text("modified", encoding="utf-8")
    verification = source.InspectSourceVerification(
        source_root=str(checkout),
        repository="fixture",
        revision=revision,
        tree_hash="tree",
        license="MIT",
        license_sha256="license",
        clean=True,
    )
    validation_count = 0

    def validate(*, source_root: Path, timeout_seconds: float) -> source.InspectSourceVerification:
        nonlocal validation_count
        del timeout_seconds
        assert source_root == checkout
        validation_count += 1
        if dirty_file.exists():
            raise InspectProfileMismatchError("dirty owned cache")
        return verification

    def fetch(*, checkout: Path, timeout_seconds: float) -> None:
        del timeout_seconds
        assert not checkout.exists()
        checkout.mkdir(parents=True)

    monkeypatch.setattr(source, "validate_inspect_source", validate)
    monkeypatch.setattr(source, "_fetch_checkout", fetch)

    assert source.prepare_inspect_source(cache_dir=tmp_path) == verification
    assert validation_count == 2
    assert not dirty_file.exists()


def test_prepare_source_offline_does_not_mutate_invalid_owned_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision / "checkout"
    checkout.mkdir(parents=True)
    dirty_file = checkout / "untrusted.py"
    dirty_file.write_text("modified", encoding="utf-8")

    def reject_invalid_checkout(**kwargs: object) -> None:
        del kwargs
        raise InspectProfileMismatchError("dirty owned cache")

    monkeypatch.setattr(source, "validate_inspect_source", reject_invalid_checkout)

    with pytest.raises(InspectProfileMismatchError, match="dirty owned cache"):
        source.prepare_inspect_source(cache_dir=tmp_path, offline=True)

    assert dirty_file.read_text(encoding="utf-8") == "modified"


def test_source_lock_is_released_and_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "source.lock"
    first = source._acquire_lock(lock_path=lock_path, timeout_seconds=1)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        source._acquire_lock(lock_path=lock_path, timeout_seconds=0.05)
    source._release_lock(first)

    second = source._acquire_lock(lock_path=lock_path, timeout_seconds=1)
    source._release_lock(second)
