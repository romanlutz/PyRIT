# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-time ephemeral token-file handoff never prints or persists the token."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.executor.benchmark import inspect_ghcp_token_file

if TYPE_CHECKING:
    from pathlib import Path


def test_run_scoped_token_file_is_one_time_and_unlinked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"
    token = b"x" * 43
    path = tmp_path / "scoped-token"
    with (
        patch.object(inspect_ghcp_token_file, "_require_nonroot_user"),
        patch.object(inspect_ghcp_token_file, "_token_path", return_value=path),
        patch.object(inspect_ghcp_token_file, "_no_follow_flag", return_value=0),
        patch.object(inspect_ghcp_token_file, "_validate_private_file"),
    ):
        inspect_ghcp_token_file.write_scoped_token(run_id=run_id, data=token)
        with pytest.raises(RuntimeError, match="not consumed"):
            inspect_ghcp_token_file.assert_scoped_token_absent(run_id=run_id)
        with pytest.raises(FileExistsError):
            inspect_ghcp_token_file.write_scoped_token(run_id=run_id, data=token)
        assert inspect_ghcp_token_file.read_scoped_token(run_id=run_id, token_file=str(path)) == token.decode()
        inspect_ghcp_token_file.assert_scoped_token_absent(run_id=run_id)
        with pytest.raises(FileNotFoundError):
            inspect_ghcp_token_file.read_scoped_token(run_id=run_id, token_file=str(path))
    output = capsys.readouterr()
    assert token.decode() not in output.out + output.err


def test_invalid_token_is_unlinked_and_aborted_start_removes_own_file(tmp_path: Path) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"
    path = tmp_path / "scoped-token"
    with (
        patch.object(inspect_ghcp_token_file, "_require_nonroot_user"),
        patch.object(inspect_ghcp_token_file, "_token_path", return_value=path),
        patch.object(inspect_ghcp_token_file, "_no_follow_flag", return_value=0),
        patch.object(inspect_ghcp_token_file, "_validate_private_file"),
    ):
        path.write_bytes(b"malformed")
        with pytest.raises(ValueError, match="unexpected length or format"):
            inspect_ghcp_token_file.read_scoped_token(run_id=run_id, token_file=str(path))
        inspect_ghcp_token_file.assert_scoped_token_absent(run_id=run_id)
        inspect_ghcp_token_file.write_scoped_token(run_id=run_id, data=b"x" * 43)
        inspect_ghcp_token_file.clear_scoped_token(run_id=run_id)
        inspect_ghcp_token_file.clear_scoped_token(run_id=run_id)
        inspect_ghcp_token_file.assert_scoped_token_absent(run_id=run_id)


def test_foreign_preexisting_file_is_not_overwritten_or_removed(tmp_path: Path) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"
    path = tmp_path / "scoped-token"
    path.write_bytes(b"foreign-content")
    with (
        patch.object(inspect_ghcp_token_file, "_require_nonroot_user"),
        patch.object(inspect_ghcp_token_file, "_token_path", return_value=path),
        patch.object(inspect_ghcp_token_file, "_no_follow_flag", return_value=0),
        patch.object(inspect_ghcp_token_file, "_validate_private_file", side_effect=ValueError("not owner-only")),
    ):
        with pytest.raises(FileExistsError):
            inspect_ghcp_token_file.write_scoped_token(run_id=run_id, data=b"x" * 43)
        with pytest.raises(ValueError, match="owner-only"):
            inspect_ghcp_token_file.clear_scoped_token(run_id=run_id)
    assert path.read_bytes() == b"foreign-content"


def test_partial_private_write_cannot_leave_token_file(tmp_path: Path) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"
    path = tmp_path / "scoped-token"
    with (
        patch.object(inspect_ghcp_token_file, "_require_nonroot_user"),
        patch.object(inspect_ghcp_token_file, "_token_path", return_value=path),
        patch.object(inspect_ghcp_token_file, "_no_follow_flag", return_value=0),
        patch.object(inspect_ghcp_token_file, "_validate_private_file"),
        patch.object(inspect_ghcp_token_file.os, "write", return_value=1),
    ):
        with pytest.raises(OSError, match="incomplete"):
            inspect_ghcp_token_file.write_scoped_token(run_id=run_id, data=b"x" * 43)
        inspect_ghcp_token_file.assert_scoped_token_absent(run_id=run_id)


def test_token_path_and_format_fail_closed_without_echoing_bad_input() -> None:
    with pytest.raises(ValueError, match="canonical run UUID"):
        inspect_ghcp_token_file._token_path(run_id="../foreign")
    with pytest.raises(ValueError, match="length or format"):
        inspect_ghcp_token_file._validate_token(data=b"bad-token-from-untrusted-source")
