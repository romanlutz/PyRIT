# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from build_scripts.pyrit_wrapped.models import Period, Snapshot, WorkItem, parse_contributor


@pytest.mark.parametrize("value", ["owner", "@owner", " https://github.com/owner/ "])
def test_contributor_identity(value: str) -> None:
    assert parse_contributor(value) == "owner"


@pytest.mark.parametrize(
    "value",
    ["First Last", "a--b", "-owner", "owner-", "https://evil.example/owner", "https://github.com/owner/repo", "a" * 40],
)
def test_contributor_rejects_ambiguous_or_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError):
        parse_contributor(value)


def test_period_uses_half_open_window() -> None:
    now = datetime(2026, 10, 2, tzinfo=UTC)
    period = Period.for_year(year=2026, now=now)
    assert period.contains(datetime(2026, 1, 1, tzinfo=UTC))
    assert not period.contains(datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC))
    assert not period.contains(now)
    assert period.year_to_date
    full = Period.for_year(year=2025, now=now)
    assert full.end == datetime(2026, 1, 1, tzinfo=UTC)
    assert not full.year_to_date


@pytest.mark.parametrize("year", [2007, 2027])
def test_period_rejects_unsupported_year(year: int) -> None:
    with pytest.raises(ValueError):
        Period.for_year(year=year, now=datetime(2026, 10, 2, tzinfo=UTC))


def test_period_rejects_naive_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Period.for_year(year=2026, now=datetime(2026, 10, 2))  # noqa: DTZ001


def test_snapshot_rejects_incomplete_replay(snapshot: Snapshot) -> None:
    with pytest.raises(ValidationError, match="complete version-1"):
        Snapshot.model_validate({**snapshot.model_dump(), "complete": False})


def test_snapshot_round_trip(snapshot: Snapshot) -> None:
    assert Snapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_inconsistent_file_coverage_is_not_accepted(item: WorkItem) -> None:
    with pytest.raises(ValidationError, match="coverage"):
        WorkItem.model_validate({**item.model_dump(), "changed_files": 2})


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "https://evil.example/pull/1", "https://github.com/other/repo/pull/1"]
)
def test_source_links_are_validated(*, item: WorkItem, url: str) -> None:
    with pytest.raises(ValidationError, match="Source URLs"):
        WorkItem.model_validate({**item.model_dump(), "url": url})
