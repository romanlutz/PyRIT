# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pinned Inspect-evals source acquisition and read-only verification."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from appdirs import user_cache_dir

from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE, InspectProfileMismatchError

_REPOSITORY = "https://github.com/UKGovernmentBEIS/inspect_evals"
_TREE_HASH = "4754468aa8ed464dd2f53df8341b9e0c91bea099"
_LICENSE_SHA256 = "c43db13e7dde0264140a5d8f65edeeaca9eadeae1c67c4a48b5a8a3eb5705701"


@dataclass(frozen=True)
class InspectSourceVerification:
    """Verified identities for one pinned source checkout."""

    source_root: str
    repository: str
    revision: str
    tree_hash: str
    license: str
    license_sha256: str
    clean: bool

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible verification data."""
        return asdict(self)


def default_inspect_source_cache_dir() -> Path:
    """Return the platform-specific PyRIT cache root for pinned Inspect source."""
    return Path(user_cache_dir("pyrit")) / "inspect_evals"


def validate_inspect_source(
    *,
    source_root: Path,
    require_clean: bool = True,
    timeout_seconds: float = 120.0,
) -> InspectSourceVerification:
    """
    Verify the pinned commit, tree, license, package layout, and optional cleanliness.

    Returns:
        InspectSourceVerification: Verified immutable source identities.

    Raises:
        InspectProfileMismatchError: If any pinned identity does not match.
        ValueError: If the timeout is not positive.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")
    root = source_root.resolve()
    package_root = root / "src" / "inspect_evals"
    if not package_root.is_dir():
        raise InspectProfileMismatchError(f"Source root '{root}' does not contain 'src/inspect_evals'.")
    revision = _run_git(root=root, args=("rev-parse", "HEAD"), timeout_seconds=timeout_seconds)
    expected_revision = PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    if revision != expected_revision:
        raise InspectProfileMismatchError(
            f"Source checkout revision '{revision}' does not match pinned revision '{expected_revision}'."
        )
    tree_hash = _run_git(root=root, args=("rev-parse", "HEAD^{tree}"), timeout_seconds=timeout_seconds)
    if tree_hash != _TREE_HASH:
        raise InspectProfileMismatchError(
            f"Source tree hash '{tree_hash}' does not match pinned tree hash '{_TREE_HASH}'."
        )
    license_path = root / "LICENSE"
    if not license_path.is_file():
        raise InspectProfileMismatchError(f"Pinned source checkout '{root}' has no LICENSE file.")
    license_sha256 = hashlib.sha256(license_path.read_bytes()).hexdigest()
    if license_sha256 != _LICENSE_SHA256:
        raise InspectProfileMismatchError(
            f"Source LICENSE sha256 '{license_sha256}' does not match pinned sha256 '{_LICENSE_SHA256}'."
        )
    clean = not _run_git(
        root=root,
        args=("status", "--porcelain", "--untracked-files=all", "--ignored"),
        timeout_seconds=timeout_seconds,
    )
    if require_clean and not clean:
        raise InspectProfileMismatchError(
            f"Source checkout '{root}' is not clean. PyRIT never modifies supplied source checkouts; "
            "use a clean pinned checkout or the content-addressed source cache."
        )
    return InspectSourceVerification(
        source_root=str(root),
        repository=_REPOSITORY,
        revision=revision,
        tree_hash=tree_hash,
        license="MIT",
        license_sha256=license_sha256,
        clean=clean,
    )


def prepare_inspect_source(
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    timeout_seconds: float = 300.0,
) -> InspectSourceVerification:
    """
    Return a verified pinned checkout from a content-addressed cache.

    The checkout is fetched only from the pinned repository and commit. The real
    Inspect package is never installed or executed.

    Returns:
        InspectSourceVerification: Verification data containing the cached source root.

    Raises:
        FileNotFoundError: If offline mode is requested before the source is cached.
        InspectProfileMismatchError: If a cached checkout fails pinned identity verification.
        TimeoutError: If cache locking or Git acquisition exceeds the timeout.
        ValueError: If the timeout is not positive.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")
    revision = PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    cache_root = (cache_dir or default_inspect_source_cache_dir()).resolve()
    checkout = cache_root / revision / "checkout"
    if offline:
        if checkout.is_symlink():
            raise InspectProfileMismatchError(f"Pinned source cache checkout '{checkout}' must not be a symbolic link.")
        if checkout.is_dir():
            return validate_inspect_source(source_root=checkout, timeout_seconds=timeout_seconds)
        raise FileNotFoundError(
            f"Pinned Inspect-evals source is not cached at '{checkout}'. "
            "Run 'inspect-evals source prepare' with network access or pass --source."
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f"{revision}.lock"
    lock_fd = _acquire_lock(lock_path=lock_path, timeout_seconds=timeout_seconds)
    try:
        if checkout.is_symlink() or (checkout.exists() and not checkout.is_dir()):
            _remove_owned_checkout(checkout=checkout)
        if checkout.is_dir():
            try:
                return validate_inspect_source(source_root=checkout, timeout_seconds=timeout_seconds)
            except InspectProfileMismatchError:
                _remove_owned_checkout(checkout=checkout)
        _fetch_checkout(checkout=checkout, timeout_seconds=timeout_seconds)
        return validate_inspect_source(source_root=checkout, timeout_seconds=timeout_seconds)
    finally:
        _release_lock(lock_fd)


def _fetch_checkout(*, checkout: Path, timeout_seconds: float) -> None:
    parent = checkout.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="checkout-", dir=parent))
    revision = PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    try:
        _run_command(("git", "init", "--quiet", str(temporary)), timeout_seconds=timeout_seconds)
        _run_git(root=temporary, args=("config", "core.longpaths", "true"), timeout_seconds=timeout_seconds)
        _run_git(root=temporary, args=("remote", "add", "origin", _REPOSITORY), timeout_seconds=timeout_seconds)
        _run_git(
            root=temporary,
            args=("fetch", "--quiet", "--depth=1", "origin", revision),
            timeout_seconds=timeout_seconds,
        )
        _run_git(
            root=temporary,
            args=("checkout", "--quiet", "--detach", "FETCH_HEAD"),
            timeout_seconds=timeout_seconds,
        )
        validate_inspect_source(source_root=temporary, timeout_seconds=timeout_seconds)
        try:
            temporary.replace(checkout)
        except FileExistsError:
            validate_inspect_source(source_root=checkout, timeout_seconds=timeout_seconds)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _remove_owned_checkout(*, checkout: Path) -> None:
    if checkout.is_symlink() or not checkout.is_dir():
        checkout.unlink(missing_ok=True)
    else:
        shutil.rmtree(checkout)


def _acquire_lock(*, lock_path: Path, timeout_seconds: float) -> int:
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    if os.fstat(lock_fd).st_size == 0:
        os.write(lock_fd, b"\0")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            _lock_file(lock_fd)
            return lock_fd
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                os.close(lock_fd)
                raise
            if time.monotonic() >= deadline:
                os.close(lock_fd)
                raise TimeoutError(f"Timed out waiting for Inspect source cache lock '{lock_path}'.") from None
            time.sleep(0.1)


def _lock_file(lock_fd: int) -> None:
    os.lseek(lock_fd, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock(lock_fd: int) -> None:
    try:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _run_git(*, root: Path, args: tuple[str, ...], timeout_seconds: float) -> str:
    return _run_command(("git", "-C", str(root), *args), timeout_seconds=timeout_seconds)


def _run_command(argv: tuple[str, ...], *, timeout_seconds: float) -> str:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to acquire or verify the pinned Inspect-evals source.") from error
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"Command '{' '.join(argv[:3])}' exceeded {timeout_seconds:g} seconds.") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Command '{' '.join(argv[:3])}' failed: {detail}")
    return completed.stdout.strip()
