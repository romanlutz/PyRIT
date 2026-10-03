# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import (
    Activity,
    Actor,
    Comment,
    CommentKind,
    ItemKind,
    Review,
    Snapshot,
    WorkItem,
)


def test_authored_landed_and_merged_are_distinct(*, snapshot: Snapshot, item: WorkItem, other: Actor) -> None:
    old = item.model_copy(update={"created_at": datetime(2025, 1, 1, tzinfo=UTC), "merged_by": other})
    stats = Metrics(snapshot.model_copy(update={"items": [old]})).calculate()
    assert stats.counts[Activity.AUTHORED] == 0
    assert stats.counts[Activity.LANDED] == 1
    assert stats.counts[Activity.MERGED] == 0


def test_same_merge_is_one_monthly_action(*, snapshot: Snapshot, item: WorkItem) -> None:
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.LANDED] == stats.counts[Activity.MERGED] == 1
    assert stats.distinct_monthly_events["2026-01"] == 2
    assert stats.own_prs_merged == 1
    assert stats.other_prs_merged == 0


def test_reviews_and_comment_types_do_not_double_count(
    *, snapshot: Snapshot, item: WorkItem, review: Review, comment: Comment, other: Actor
) -> None:
    item = item.model_copy(update={"author": other})
    repeat = review.model_copy(update={"id": 101, "has_body": True, "submitted_at": datetime(2026, 2, 2, tzinfo=UTC)})
    comments = [comment.model_copy(update={"id": number}) for number in range(200, 203)]
    comments.append(comment.model_copy(update={"kind": CommentKind.DISCUSSION, "id": 300, "review_id": None}))
    value = snapshot.model_copy(update={"items": [item], "reviews": [review, repeat], "comments": comments})
    stats = Metrics(value).calculate()
    assert stats.counts[Activity.REVIEWED] == 1
    assert stats.counts[Activity.REVIEWS] == 2
    assert stats.counts[Activity.REVIEW_BODIES] == 1
    assert stats.counts[Activity.INLINE] == 3
    assert stats.counts[Activity.PR_COMMENTS] == 1
    assert stats.other_pr_comments == 5
    assert stats.reviewed_authors == [other]
    assert stats.classifications["inline:200"].primary_artifact == "Tests"
    assert stats.activities[Activity.REVIEWED][0].event_at == review.submitted_at


def test_pending_reviews_and_comments_are_excluded(
    *, snapshot: Snapshot, item: WorkItem, review: Review, comment: Comment
) -> None:
    pending = review.model_copy(update={"state": "PENDING", "submitted_at": None, "has_body": True})
    stats = Metrics(
        snapshot.model_copy(update={"items": [item], "reviews": [pending], "comments": [comment]})
    ).calculate()
    assert stats.counts[Activity.REVIEWED] == stats.counts[Activity.INLINE] == stats.counts[Activity.REVIEW_BODIES] == 0


def test_dismissed_submitted_review_is_still_activity(*, snapshot: Snapshot, item: WorkItem, review: Review) -> None:
    review = review.model_copy(update={"state": "DISMISSED"})
    stats = Metrics(snapshot.model_copy(update={"items": [item], "reviews": [review]})).calculate()
    assert stats.counts[Activity.REVIEWS] == 1


def test_review_dates_are_independent_of_pr_dates(*, snapshot: Snapshot, item: WorkItem, review: Review) -> None:
    item = item.model_copy(
        update={"created_at": datetime(2024, 1, 1, tzinfo=UTC), "merged_at": datetime(2024, 2, 1, tzinfo=UTC)}
    )
    stats = Metrics(snapshot.model_copy(update={"items": [item], "reviews": [review]})).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.LANDED] == 0
    assert stats.counts[Activity.REVIEWED] == 1


def test_discussion_comments_are_not_formal_reviews(*, snapshot: Snapshot, item: WorkItem, comment: Comment) -> None:
    comment = comment.model_copy(update={"kind": CommentKind.DISCUSSION, "review_id": None})
    stats = Metrics(snapshot.model_copy(update={"items": [item], "comments": [comment]})).calculate()
    assert stats.counts[Activity.PR_COMMENTS] == 1
    assert stats.counts[Activity.REVIEWED] == 0


def test_issue_count_excludes_prs(*, snapshot: Snapshot, item: WorkItem) -> None:
    issue = item.model_copy(update={"id": "I_2", "number": 2, "kind": ItemKind.ISSUE})
    stats = Metrics(snapshot.model_copy(update={"items": [item, issue]})).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.ISSUES] == 1


def test_renamed_account_uses_stable_identity(*, snapshot: Snapshot, item: WorkItem) -> None:
    author = snapshot.contributor.model_copy(update={"login": "old-login"})
    stats = Metrics(snapshot.model_copy(update={"items": [item.model_copy(update={"author": author})]})).calculate()
    assert stats.counts[Activity.AUTHORED] == 1


def test_snapshot_verifies_pending_comment_parent(*, snapshot: Snapshot, item: WorkItem, comment: Comment) -> None:
    value = {**snapshot.model_dump(), "items": [item.model_dump()], "comments": [comment.model_dump()]}
    with pytest.raises(ValidationError, match="no review record"):
        Snapshot.model_validate(value)


def test_primary_breakdowns_reconcile_to_counts(*, snapshot: Snapshot, item: WorkItem) -> None:
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    for role, breakdown in stats.breakdowns.items():
        assert sum(breakdown.primary_topics.values()) == stats.counts[role]
        assert sum(breakdown.primary_artifacts.values()) == stats.counts[role]
        assert sum(breakdown.surfaces.values()) == stats.counts[role]


def test_timezone_boundary_is_grouped_in_utc(*, snapshot: Snapshot, item: WorkItem) -> None:
    created = datetime.fromisoformat("2025-12-31T20:00:00-08:00")
    item = item.model_copy(update={"created_at": created})
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    assert stats.counts[Activity.AUTHORED] == 1
    assert stats.monthly["2026-01"][Activity.AUTHORED] == 1


def test_deleted_and_bot_authors_are_not_counted_as_people(
    *, snapshot: Snapshot, item: WorkItem, review: Review
) -> None:
    for author in (None, Actor(id="B_1", login="example[bot]", type="Bot"), Actor(id="U_ghost", login="ghost")):
        item = item.model_copy(update={"author": author})
        stats = Metrics(snapshot.model_copy(update={"items": [item], "reviews": [review]})).calculate()
        assert stats.counts[Activity.REVIEWED] == 1
        assert not stats.reviewed_authors


def test_agent_owned_prs_are_not_human_authored(*, snapshot: Snapshot, item: WorkItem) -> None:
    agent = Actor(id="B_copilot", login="Copilot", type="Bot")
    item = item.model_copy(update={"author": agent})
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.LANDED] == 0
    assert stats.counts[Activity.MERGED] == 1


def test_comment_on_an_issue_has_its_own_count(*, snapshot: Snapshot, item: WorkItem, comment: Comment) -> None:
    item = item.model_copy(update={"kind": ItemKind.ISSUE})
    comment = comment.model_copy(update={"kind": CommentKind.DISCUSSION, "review_id": None})
    stats = Metrics(snapshot.model_copy(update={"items": [item], "comments": [comment]})).calculate()
    assert stats.counts[Activity.ISSUE_COMMENTS] == 1
    assert stats.counts[Activity.PR_COMMENTS] == stats.counts[Activity.REVIEWED] == 0


def test_comment_parent_must_belong_to_same_pr(
    *, snapshot: Snapshot, item: WorkItem, review: Review, comment: Comment
) -> None:
    second = item.model_copy(update={"number": 2, "id": "PR_2"})
    wrong_parent = review.model_copy(update={"item_number": 2})
    with pytest.raises(ValidationError, match="different PR"):
        Snapshot.model_validate(
            {**snapshot.model_dump(), "items": [item, second], "reviews": [wrong_parent], "comments": [comment]}
        )
