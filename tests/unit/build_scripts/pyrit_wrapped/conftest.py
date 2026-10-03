# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime
from pathlib import Path

import pytest

from build_scripts.pyrit_wrapped.models import (
    Actor,
    Comment,
    CommentKind,
    ItemKind,
    Period,
    Review,
    Snapshot,
    TaxonomyConfig,
    WorkItem,
)


@pytest.fixture
def contributor() -> Actor:
    return Actor(id="U_owner", login="owner")


@pytest.fixture
def other() -> Actor:
    return Actor(id="U_other", login="other")


@pytest.fixture
def period() -> Period:
    return Period.for_year(year=2026, now=datetime(2026, 10, 2, 12, tzinfo=UTC))


@pytest.fixture
def taxonomy_config() -> TaxonomyConfig:
    path = Path(__file__).resolve().parents[4] / "build_scripts" / "pyrit_wrapped" / "taxonomy.json"
    return TaxonomyConfig.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def item(contributor: Actor) -> WorkItem:
    return WorkItem(
        id="PR_1",
        number=1,
        kind=ItemKind.PR,
        title="FIX converter behavior",
        url="https://github.com/microsoft/PyRIT/pull/1",
        author=contributor,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
        merged_at=datetime(2026, 1, 3, tzinfo=UTC),
        merged_by=contributor,
        state="closed",
        labels=[],
        paths=["pyrit/converter/example.py"],
        changed_files=1,
        files_complete=True,
    )


@pytest.fixture
def review(contributor: Actor) -> Review:
    return Review(
        id=100,
        item_number=1,
        author=contributor,
        state="APPROVED",
        submitted_at=datetime(2026, 1, 4, tzinfo=UTC),
        has_body=False,
        url="https://github.com/microsoft/PyRIT/pull/1#pullrequestreview-100",
    )


@pytest.fixture
def comment(contributor: Actor) -> Comment:
    return Comment(
        id=200,
        item_number=1,
        author=contributor,
        kind=CommentKind.INLINE,
        created_at=datetime(2026, 1, 4, tzinfo=UTC),
        review_id=100,
        path="tests/unit/converter/test_example.py",
        url="https://github.com/microsoft/PyRIT/pull/1#discussion_r200",
    )


@pytest.fixture
def snapshot(*, contributor: Actor, period: Period, taxonomy_config: TaxonomyConfig) -> Snapshot:
    return Snapshot(
        contributor=contributor,
        period=period,
        collected_at=period.cutoff,
        earliest_response_at=period.cutoff,
        complete=True,
        taxonomy=taxonomy_config,
        items=[],
        reviews=[],
        comments=[],
    )
