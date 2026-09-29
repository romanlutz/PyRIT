# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-shot, owner-only bootstrap for the ephemeral guest model-gateway token."""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path


def _token_path(*, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", run_id):
        raise ValueError("Run token bootstrap requires a trusted canonical run UUID.")
    root = Path("/tmp")
    observed = root.lstat()
    if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != 0 or not observed.st_mode & stat.S_ISVTX:
        raise ValueError("Run token bootstrap requires the approved root-owned sticky /tmp tmpfs.")
    return root / f"pyrit-inspect-token-{run_id}"


def _require_nonroot_user() -> None:
    getuid = getattr(os, "getuid", None)
    if getuid is None or getuid() != 10001:
        raise RuntimeError("Run token bootstrap requires the nonroot agent/bridge UID10001.")


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if type(value) is not int:
        raise RuntimeError("Run token bootstrap requires Linux no-follow file opens.")
    return value


def _validate_token(*, data: bytes) -> str:
    if not re.fullmatch(rb"[A-Za-z0-9_-]{43}", data):
        raise ValueError("Run token source has an unexpected length or format.")
    return data.decode("ascii")


def _validate_private_file(*, fd: int) -> None:
    observed = os.fstat(fd)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_uid != 10001
        or observed.st_nlink != 1
    ):
        raise ValueError("Run token source is not an owner-only regular file.")


def write_scoped_token(*, run_id: str, data: bytes) -> None:
    """
    Create exactly one ephemeral token file from the trusted controller's stdin.

    Raises:
        OSError: If an atomic owner-only file cannot be fully written.
    """
    _require_nonroot_user()
    _validate_token(data=data)
    path = _token_path(run_id=run_id)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
    fd = os.open(path, flags, 0o600)
    complete = False
    try:
        _validate_private_file(fd=fd)
        if os.write(fd, data) != len(data):
            raise OSError("Run token write was incomplete.")
        os.fsync(fd)
        complete = True
    finally:
        os.close(fd)
        if not complete:
            path.unlink(missing_ok=True)


def read_scoped_token(*, run_id: str, token_file: str) -> str:
    """
    Read and remove the run's owner-only token before any GHCP model or tool call.

    Returns:
        str: The ephemeral token held only in the bounded guest worker process.

    Raises:
        ValueError: If the path, owner, mode, or token bytes differ from the trusted bootstrap.
        OSError: If the private file cannot be opened, read, closed or removed.
    """
    _require_nonroot_user()
    path = _token_path(run_id=run_id)
    if token_file != str(path):
        raise ValueError("Run token file differs from the trusted per-run path.")
    fd = os.open(path, os.O_RDONLY | _no_follow_flag())
    try:
        _validate_private_file(fd=fd)
    except (OSError, ValueError):
        os.close(fd)
        raise
    try:
        data = os.read(fd, 44)
        if os.read(fd, 1):
            raise ValueError("Run token file contains unexpected extra bytes.")
        return _validate_token(data=data)
    finally:
        os.close(fd)
        path.unlink()


def assert_scoped_token_absent(*, run_id: str) -> None:
    """
    Prove that the one-time file is absent, including broken symlinks.

    Raises:
        RuntimeError: If the run-scoped path still exists.
    """
    _require_nonroot_user()
    if os.path.lexists(_token_path(run_id=run_id)):
        raise RuntimeError("Run token file was not consumed before inference.")


def clear_scoped_token(*, run_id: str) -> None:
    """
    Remove an unconsumed private file after an aborted gateway or worker start.

    Raises:
        ValueError: If a preexisting file is not owned by the approved guest UID.
    """
    _require_nonroot_user()
    path = _token_path(run_id=run_id)
    try:
        fd = os.open(path, os.O_RDONLY | _no_follow_flag())
    except FileNotFoundError:
        return
    try:
        _validate_private_file(fd=fd)
        observed = path.lstat()
        opened = os.fstat(fd)
        if (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Run token path changed after its private file was opened.")
    finally:
        os.close(fd)
    path.unlink()
    assert_scoped_token_absent(run_id=run_id)


def main() -> None:
    """
    Perform only fixed, scoped token operations without printing its value.

    Raises:
        ValueError: If invoked outside the fixed bootstrap operations.
    """
    if len(sys.argv) != 3:
        raise ValueError("A scoped token operation and run UUID are required.")
    operation, run_id = sys.argv[1:]
    if operation == "write":
        write_scoped_token(run_id=run_id, data=sys.stdin.buffer.read(128))
        sys.stdout.write(f"{os.getpid()}\n")
    elif operation == "absent":
        assert_scoped_token_absent(run_id=run_id)
    elif operation == "clear":
        clear_scoped_token(run_id=run_id)
    else:
        raise ValueError("Unsupported scoped token operation.")


if __name__ == "__main__":
    main()
