# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pure checks for a private, non-disclosing Inspect token audit."""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import pytest

from build_scripts.inspect_ghcp_controller.audit_secret_retention import TokenAbsenceScanner, _fingerprint

if TYPE_CHECKING:
    from pathlib import Path


def test_token_audit_detects_plain_and_encoded_values_without_revealing_them() -> None:
    token = b"x" * 43
    scanner = TokenAbsenceScanner(token_sha256=hashlib.sha256(token).hexdigest())
    scanner.assert_absent(data=b"x" * 42 + b"y", source="synthetic safe source")
    encodings = (
        b"prefix" + token + b"suffix",
        base64.b64encode(token),
        base64.urlsafe_b64encode(token).rstrip(b"="),
        token.hex().encode("ascii"),
        token.decode("ascii").encode("utf-16-le"),
        token.decode("ascii").encode("utf-16-be"),
        b"".join(f"\\u{character:04x}".encode("ascii") for character in token),
        b"".join(f"%{character:02x}".encode("ascii") for character in token),
    )
    for content in encodings:
        with pytest.raises(RuntimeError, match="content withheld") as captured:
            scanner.assert_absent(data=content, source="synthetic retained source")
        assert token.decode("ascii") not in str(captured.value)


def test_token_audit_catches_split_file_window_and_compressed_attachment(tmp_path: Path) -> None:
    token = b"z" * 43
    scanner = TokenAbsenceScanner(token_sha256=hashlib.sha256(token).hexdigest())
    output = tmp_path / "controller.stderr"
    output.write_bytes(b"." * (65_536 - 10) + token)
    with pytest.raises(RuntimeError, match="content withheld"):
        scanner.scan_file(path=output, source="controller stderr")
    archive = tmp_path / "original.eval"
    with zipfile.ZipFile(archive, "w") as record:
        record.writestr("attachments/private.txt", base64.b64encode(token))
    with pytest.raises(RuntimeError, match="content withheld"):
        scanner.scan_archive(path=archive)


def test_pre_delivery_fingerprint_is_run_bound_without_persisting_token(tmp_path: Path) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"
    token = b"x" * 43
    digest = hashlib.sha256(token).hexdigest()
    stage = tmp_path / "controller-stage.jsonl"
    stage.write_text(json.dumps({"run_id": run_id, "stage": "episode_created", "control_token_sha256": digest}) + "\n")
    assert _fingerprint(run_dir=tmp_path, run_id=run_id) == digest
    assert token not in stage.read_bytes()
    with pytest.raises(ValueError, match="different|belong"):
        _fingerprint(run_dir=tmp_path, run_id="22222222-2222-2222-2222-222222222222")
