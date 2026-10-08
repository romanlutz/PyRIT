# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Nonblocking ownership of one caller-resolved local file."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import BinaryIO


def lock_local_file(*, owner: BinaryIO, release: bool) -> None:
    """
    Lock one byte or release only the caller's held local process lock.

    Raises:
        OSError: If another owner holds the lock or the operation fails.
    """
    owner.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(owner.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(owner.fileno(), fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)
