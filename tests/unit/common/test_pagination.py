# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from pyrit.common.pagination import decode_keyset_cursor, encode_keyset_cursor


@pytest.mark.parametrize("field", ["t", "i"])
@pytest.mark.parametrize("invalid", [None, 123, True, [], {}])
def test_decode_keyset_cursor_rejects_non_string_fields(*, field: str, invalid: object) -> None:
    payload = {
        "v": 1,
        "f": "cohort",
        "t": "2026-09-21T12:00:00Z",
        "i": "00000000-0000-0000-0000-000000000001",
        field: invalid,
    }
    cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    assert decode_keyset_cursor(cursor=cursor, fingerprint="cohort") is None


def test_decode_keyset_cursor_preserves_valid_identity_and_normalizes_utc() -> None:
    identifier = "00000000-0000-0000-0000-000000000001"
    cursor = encode_keyset_cursor(
        timestamp=datetime(2026, 9, 21, 5, tzinfo=timezone(timedelta(hours=-7))),
        identifier=identifier,
        fingerprint="cohort",
    )
    decoded = decode_keyset_cursor(cursor=cursor, fingerprint="cohort")
    assert decoded is not None
    assert decoded.identifier == identifier
    assert decoded.timestamp == datetime(2026, 9, 21, 12, tzinfo=UTC)


def test_decode_keyset_cursor_rejects_a_different_cohort() -> None:
    cursor = encode_keyset_cursor(
        timestamp=datetime(2026, 9, 21, tzinfo=UTC),
        identifier="00000000-0000-0000-0000-000000000001",
        fingerprint="original",
    )
    assert decode_keyset_cursor(cursor=cursor, fingerprint="changed") is None
