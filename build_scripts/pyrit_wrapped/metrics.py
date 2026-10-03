# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC
from typing import TYPE_CHECKING

from build_scripts.pyrit_wrapped.models import (
    Activity,
    Actor,
    Breakdown,
    CommentKind,
    Evidence,
    ItemKind,
    Snapshot,
    Stats,
    WorkItem,
    WrappedError,
)
from build_scripts.pyrit_wrapped.snapshot import same_actor
from build_scripts.pyrit_wrapped.taxonomy import Taxonomy

if TYPE_CHECKING:
    from datetime import datetime


class Metrics:
    _EVENT_ROLES = {
        Activity.AUTHORED,
        Activity.LANDED,
        Activity.MERGED,
        Activity.ISSUES,
        Activity.REVIEWS,
        Activity.INLINE,
        Activity.PR_COMMENTS,
        Activity.ISSUE_COMMENTS,
    }

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.items = {item.number: item for item in snapshot.items}
        self.activities: dict[Activity, list[Evidence]] = {activity: [] for activity in Activity}
        self.taxonomy = Taxonomy(snapshot.taxonomy)
        self.classifications = {item.ref: self.taxonomy.classify(item) for item in snapshot.items}

    def calculate(self) -> Stats:
        self._item_activity()
        self._review_activity()
        self._comment_activity()
        for activity in Activity:
            unique: dict[str, Evidence] = {}
            for record in sorted(self.activities[activity], key=lambda record: (record.event_at, record.ref)):
                unique.setdefault(record.ref, record)
            self.activities[activity] = list(unique.values())
        own_merges, other_merges, unknown_merges = self._ownership_counts([Activity.MERGED])
        own_comments, other_comments, unknown_comments = self._ownership_counts(
            [Activity.INLINE, Activity.REVIEW_BODIES, Activity.PR_COMMENTS]
        )
        monthly, distinct_events = self._monthly_activity()
        return Stats(
            repository=self.snapshot.repository,
            contributor=self.snapshot.contributor,
            period=self.snapshot.period,
            collected_at=self.snapshot.collected_at,
            earliest_response_at=self.snapshot.earliest_response_at,
            taxonomy=self.snapshot.taxonomy,
            counts={activity: len(records) for activity, records in self.activities.items()},
            activities=self.activities,
            classifications=self.classifications,
            breakdowns={activity: self._breakdown(records) for activity, records in self.activities.items()},
            monthly=monthly,
            distinct_monthly_events=distinct_events,
            reviewed_authors=self._reviewed_authors(),
            own_prs_merged=own_merges,
            other_prs_merged=other_merges,
            unknown_authors_merged=unknown_merges,
            own_pr_comments=own_comments,
            other_pr_comments=other_comments,
            unknown_author_pr_comments=unknown_comments,
            public_comment_proofs={
                comment.ref: timestamp
                for comment in self.snapshot.comments
                if (timestamp := comment.publicly_verified_at) is not None
            },
            warnings=self.snapshot.warnings,
        )

    def _item_activity(self) -> None:
        contributor, period = self.snapshot.contributor, self.snapshot.period
        for item in self.snapshot.items:
            if same_actor(item.author, contributor):
                if period.contains(item.created_at):
                    role = Activity.AUTHORED if item.kind == ItemKind.PR else Activity.ISSUES
                    self._record(activity=role, item=item, event_at=item.created_at)
                if item.kind == ItemKind.PR and period.contains(item.merged_at):
                    self._record(activity=Activity.LANDED, item=item, event_at=item.merged_at)
            if item.kind == ItemKind.PR and same_actor(item.merged_by, contributor) and period.contains(item.merged_at):
                self._record(activity=Activity.MERGED, item=item, event_at=item.merged_at)

    def _review_activity(self) -> None:
        for review in self.snapshot.reviews:
            if not same_actor(review.author, self.snapshot.contributor):
                continue
            if review.state == "PENDING" or not self.snapshot.period.contains(review.submitted_at):
                continue
            item = self.items[review.item_number]
            self._record(
                activity=Activity.REVIEWS, item=item, event_at=review.submitted_at, ref=review.ref, url=review.url
            )
            self._record(activity=Activity.REVIEWED, item=item, event_at=review.submitted_at)
            if review.has_body:
                self._record(
                    activity=Activity.REVIEW_BODIES,
                    item=item,
                    event_at=review.submitted_at,
                    ref=review.ref,
                    url=review.url,
                )

    def _comment_activity(self) -> None:
        reviews = {review.id: review for review in self.snapshot.reviews}
        for comment in self.snapshot.comments:
            if not same_actor(comment.author, self.snapshot.contributor) or not self.snapshot.period.contains(
                comment.created_at
            ):
                continue
            item = self.items[comment.item_number]
            if comment.kind == CommentKind.INLINE:
                review = reviews.get(comment.review_id) if comment.review_id is not None else None
                if review is not None and (review.state == "PENDING" or review.submitted_at is None):
                    continue
                activity = Activity.INLINE
                if comment.path:
                    path_item = item.model_copy(
                        update={"paths": [comment.path], "changed_files": 1, "files_complete": True}
                    )
                    self.classifications[comment.ref] = self.taxonomy.classify(path_item)
            else:
                activity = Activity.PR_COMMENTS if item.kind == ItemKind.PR else Activity.ISSUE_COMMENTS
            self._record(activity=activity, item=item, event_at=comment.created_at, ref=comment.ref, url=comment.url)

    def _record(
        self,
        *,
        activity: Activity,
        item: WorkItem,
        event_at: datetime | None,
        ref: str | None = None,
        url: str | None = None,
    ) -> None:
        if event_at is None:
            raise WrappedError(f"Missing event timestamp for {activity.value}: {item.ref}")
        self.activities[activity].append(
            Evidence(ref=ref or item.ref, item_ref=item.ref, title=item.title, url=url or item.url, event_at=event_at)
        )

    def _breakdown(self, records: list[Evidence]) -> Breakdown:
        topics: dict[str, list[str]] = defaultdict(list)
        classifications = [
            self.classifications.get(record.ref, self.classifications[record.item_ref]) for record in records
        ]
        for record, classification in zip(records, classifications, strict=True):
            for topic in classification.topics:
                topics[topic].append(record.ref)
        return Breakdown(
            count=len(records),
            topics=dict(sorted(topics.items())),
            primary_topics=dict(sorted(Counter(entry.primary_topic for entry in classifications).items())),
            surfaces=dict(sorted(Counter(entry.surface for entry in classifications).items())),
            primary_artifacts=dict(sorted(Counter(entry.primary_artifact for entry in classifications).items())),
            intents=dict(sorted(Counter(entry.intent for entry in classifications).items())),
        )

    def _monthly_activity(self) -> tuple[dict[str, dict[Activity, int]], dict[str, int]]:
        months = [
            f"{self.snapshot.period.year}-{month:02}"
            for month in range(1, 13)
            if month <= (self.snapshot.period.cutoff.month if self.snapshot.period.year_to_date else 12)
        ]
        monthly = {month: dict.fromkeys(Activity, 0) for month in months}
        unique_events: dict[str, set[str]] = {month: set() for month in months}
        for activity, records in self.activities.items():
            for record in records:
                month = record.event_at.astimezone(UTC).strftime("%Y-%m")
                monthly[month][activity] += 1
                if activity in self._EVENT_ROLES:
                    prefix = "merged" if activity in {Activity.LANDED, Activity.MERGED} else activity.value
                    unique_events[month].add(f"{prefix}:{record.ref}")
        return monthly, {month: len(events) for month, events in unique_events.items()}

    def _ownership_counts(self, roles: list[Activity]) -> tuple[int, int, int]:
        own, other, unknown = 0, 0, 0
        by_ref = {item.ref: item for item in self.snapshot.items}
        for role in roles:
            for record in self.activities[role]:
                author = by_ref[record.item_ref].author
                if author is None or author.is_deleted:
                    unknown += 1
                elif same_actor(author, self.snapshot.contributor):
                    own += 1
                else:
                    other += 1
        return own, other, unknown

    def _reviewed_authors(self) -> list[Actor]:
        authors: dict[str, Actor] = {}
        by_ref = {item.ref: item for item in self.snapshot.items}
        for record in self.activities[Activity.REVIEWED]:
            author = by_ref[record.item_ref].author
            if (
                author is not None
                and author.type == "User"
                and not author.is_deleted
                and not same_actor(author, self.snapshot.contributor)
            ):
                authors[author.id] = author
        return sorted(authors.values(), key=lambda author: author.login.lower())
