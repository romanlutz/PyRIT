# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from urllib.parse import urlparse

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class WrappedError(Exception):
    """A report cannot be generated reliably."""


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("url", check_fields=False)
    @classmethod
    def _validate_source_url(cls, value: str) -> str:
        if not re.fullmatch(
            r"https://github\.com/microsoft/PyRIT/(?:pull|issues)/[1-9]\d*(?:#[A-Za-z0-9_-]+)?",
            value,
            re.IGNORECASE,
        ):
            raise ValueError("Source URLs must reference a microsoft/PyRIT GitHub item.")
        return value


class ItemKind(str, Enum):
    PR = "pr"
    ISSUE = "issue"


class CommentKind(str, Enum):
    INLINE = "inline"
    DISCUSSION = "discussion"


class Activity(str, Enum):
    AUTHORED = "authored_prs"
    LANDED = "landed_prs"
    MERGED = "merged_prs"
    ISSUES = "opened_issues"
    REVIEWED = "reviewed_prs"
    REVIEWS = "submitted_reviews"
    INLINE = "inline_comments"
    REVIEW_BODIES = "review_summary_comments"
    PR_COMMENTS = "pr_discussion_comments"
    ISSUE_COMMENTS = "issue_discussion_comments"


class Actor(Model):
    id: str = Field(min_length=1)
    login: str = Field(min_length=1)
    type: str = "User"

    @property
    def is_deleted(self) -> bool:
        return self.login.lower() == "ghost"


class Period(Model):
    year: int = Field(ge=2008)
    start: AwareDatetime
    end: AwareDatetime
    cutoff: AwareDatetime

    @model_validator(mode="after")
    def _validate_bounds(self) -> Period:
        expected_start = datetime(self.year, 1, 1, tzinfo=UTC)
        expected_end = datetime(self.year + 1, 1, 1, tzinfo=UTC)
        if self.start != expected_start or self.end != expected_end or not self.start < self.cutoff <= self.end:
            raise ValueError("Period must use UTC calendar-year boundaries and a cutoff within that year.")
        return self

    def contains(self, value: datetime | None) -> bool:
        return value is not None and self.start <= value < self.cutoff

    @property
    def year_to_date(self) -> bool:
        return self.cutoff < self.end

    @classmethod
    def for_year(cls, *, year: int, now: datetime) -> Period:
        if now.tzinfo is None:
            raise ValueError("The collection clock must be timezone-aware.")
        if not 2008 <= year <= now.astimezone(UTC).year:
            raise ValueError(f"Year must be between 2008 and {now.astimezone(UTC).year}.")
        return cls(
            year=year,
            start=datetime(year, 1, 1, tzinfo=UTC),
            end=datetime(year + 1, 1, 1, tzinfo=UTC),
            cutoff=min(now, datetime(year + 1, 1, 1, tzinfo=UTC)),
        )


class WorkItem(Model):
    id: str | None = Field(min_length=1)
    kind: ItemKind
    number: int = Field(gt=0)
    title: str
    url: str
    author: Actor | None
    created_at: AwareDatetime | None
    updated_at: AwareDatetime | None
    state: str | None
    available: bool = True
    labels: list[str] = Field(default_factory=list)
    merged_at: AwareDatetime | None = None
    merged_by: Actor | None = None
    changed_files: int | None = Field(default=None, ge=0)
    paths: list[str] = Field(default_factory=list)
    files_complete: bool = False

    @model_validator(mode="after")
    def _validate_file_coverage(self) -> WorkItem:
        if self.available and (
            self.id is None or self.created_at is None or self.updated_at is None or self.state is None
        ):
            raise ValueError("Available work items require an ID, creation/update timestamps, and state.")
        if not self.available and any(
            value is not None
            for value in (
                self.id,
                self.author,
                self.created_at,
                self.updated_at,
                self.state,
                self.merged_at,
                self.merged_by,
            )
        ):
            raise ValueError("Unavailable item metadata must remain explicitly unknown.")
        if self.files_complete and (
            not self.available
            or self.kind != ItemKind.PR
            or self.changed_files != len(self.paths)
            or len(set(self.paths)) != len(self.paths)
        ):
            raise ValueError("Complete PR-file coverage must reconcile to the distinct changed-file count.")
        return self

    @property
    def ref(self) -> str:
        return f"{self.kind.value}:{self.number}"


class Review(Model):
    id: int
    item_number: int
    author: Actor | None
    state: str
    submitted_at: AwareDatetime | None
    has_body: bool
    url: str

    @property
    def ref(self) -> str:
        return f"review:{self.id}"


class Comment(Model):
    id: int
    item_number: int
    author: Actor | None
    kind: CommentKind
    created_at: AwareDatetime
    url: str
    review_id: int | None = None
    path: str | None = None
    publicly_verified_at: AwareDatetime | None = None

    @property
    def ref(self) -> str:
        return f"{self.kind.value}:{self.id}"


class PathRule(Model):
    prefix: str
    topic: str
    surface: str


class TaxonomyConfig(Model):
    version: int = Field(ge=1)
    path_rules: list[PathRule]
    topic_labels: dict[str, str]
    topic_keywords: dict[str, list[str]]
    intent_aliases: dict[str, str]
    intent_labels: dict[str, str]


class CollectionSession(Model):
    schema_version: int = 1
    identifier: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at: AwareDatetime
    period: Period
    taxonomy: TaxonomyConfig
    complete: bool = False

    @model_validator(mode="after")
    def _validate_version(self) -> CollectionSession:
        if self.schema_version != 1:
            raise ValueError("Unsupported collection-session version; use --refresh.")
        return self


class Snapshot(Model):
    schema_version: int = 1
    repository: str = "microsoft/PyRIT"
    contributor: Actor
    period: Period
    collected_at: AwareDatetime
    earliest_response_at: AwareDatetime
    complete: bool
    taxonomy: TaxonomyConfig
    items: list[WorkItem]
    reviews: list[Review]
    comments: list[Comment]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_integrity(self) -> Snapshot:
        if self.schema_version != 1 or self.repository != "microsoft/PyRIT" or not self.complete:
            raise ValueError("Only complete version-1 public microsoft/PyRIT snapshots can be replayed.")
        numbers = {item.number: item for item in self.items}
        available_ids = [item.id for item in self.items if item.available]
        if len(numbers) != len(self.items) or len(set(available_ids)) != len(available_ids):
            raise ValueError("Snapshot contains duplicate work items.")
        for records in (self.reviews, self.comments):
            if len({record.ref for record in records}) != len(records):
                raise ValueError("Snapshot contains duplicate activity records.")
            if any(record.item_number not in numbers for record in records):
                raise ValueError("Snapshot activity references a missing work item.")
        if any(numbers[review.item_number].kind != ItemKind.PR for review in self.reviews):
            raise ValueError("Reviews must reference pull requests.")
        reviews_by_id = {review.id: review for review in self.reviews}
        for comment in self.comments:
            if comment.kind == CommentKind.INLINE:
                if numbers[comment.item_number].kind != ItemKind.PR:
                    raise ValueError("Inline comments must reference pull requests.")
                if comment.review_id is not None:
                    parent = reviews_by_id.get(comment.review_id)
                    if parent is None and comment.publicly_verified_at is None:
                        raise ValueError("Inline comment has no review record to verify draft status.")
                    if parent is not None and parent.item_number != comment.item_number:
                        raise ValueError("Inline comment references a review on a different PR.")
        return self


class Classification(Model):
    topics: list[str]
    primary_topic: str
    surface: str
    artifacts: list[str]
    primary_artifact: str
    languages: list[str]
    intent: str
    evidence: list[str]
    inferred: bool = False


class Evidence(Model):
    ref: str
    item_ref: str
    title: str
    url: str
    event_at: AwareDatetime


class Breakdown(Model):
    count: int
    topics: dict[str, list[str]]
    primary_topics: dict[str, int]
    surfaces: dict[str, int]
    primary_artifacts: dict[str, int]
    intents: dict[str, int]


class Stats(Model):
    schema_version: int = 1
    repository: str
    contributor: Actor
    period: Period
    collected_at: AwareDatetime
    earliest_response_at: AwareDatetime
    taxonomy: TaxonomyConfig
    counts: dict[Activity, int]
    activities: dict[Activity, list[Evidence]]
    classifications: dict[str, Classification]
    breakdowns: dict[Activity, Breakdown]
    monthly: dict[str, dict[Activity, int]]
    distinct_monthly_events: dict[str, int]
    reviewed_authors: list[Actor]
    other_prs_merged: int
    own_prs_merged: int
    unknown_authors_merged: int
    own_pr_comments: int
    other_pr_comments: int
    unknown_author_pr_comments: int
    public_comment_proofs: dict[str, AwareDatetime]
    warnings: list[str]


class Slide(Model):
    type: str
    title: str
    summary: str
    facts: dict[str, int]
    evidence_refs: list[str]
    cue_id: str | None = None
    duration_ms: int | None = None


class OmittedSlide(Model):
    type: str
    reason: str


class Story(Model):
    schema_version: int = 1
    contributor: Actor
    period: Period
    slides: list[Slide]
    omitted: list[OmittedSlide]


def parse_contributor(value: str) -> str:
    name = value.strip()
    if "://" in name:
        parsed = urlparse(name)
        if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment:
            raise ValueError("Use a GitHub username or an https://github.com/username profile URL.")
        name = parsed.path.strip("/")
    name = name.removeprefix("@")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", name) or "--" in name:
        raise ValueError("Enter an unambiguous GitHub username, not a display name.")
    return name
