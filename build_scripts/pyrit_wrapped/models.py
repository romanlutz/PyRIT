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
    PR_CLOSED = "closed_prs"
    ISSUES_CLOSED = "closed_issues"
    SHIPPED = "shipped_prs"


class PeriodKind(str, Enum):
    YEAR = "year"
    RELEASE = "release"


class Capability(str, Enum):
    CLOSURES = "closures"
    LOC = "loc"


class LocScope(str, Enum):
    LANDED_PRS = "contributor_landed_pr_churn"
    RELEASE_DIFF = "release_net_tree_diff"


class Actor(Model):
    id: str = Field(min_length=1)
    login: str = Field(min_length=1)
    type: str = "User"
    database_id: int | None = None

    @property
    def is_deleted(self) -> bool:
        return self.login.lower() == "ghost"


class Period(Model):
    kind: PeriodKind = PeriodKind.YEAR
    year: int = Field(ge=2008)
    start: AwareDatetime
    end: AwareDatetime
    cutoff: AwareDatetime

    @model_validator(mode="after")
    def _validate_bounds(self) -> Period:
        if self.kind == PeriodKind.RELEASE:
            if self.year != self.start.astimezone(UTC).year or not self.start < self.cutoff <= self.end:
                raise ValueError("Release activity requires an ordered UTC window.")
            return self
        expected_start = datetime(self.year, 1, 1, tzinfo=UTC)
        expected_end = datetime(self.year + 1, 1, 1, tzinfo=UTC)
        if self.start != expected_start or self.end != expected_end or not self.start < self.cutoff <= self.end:
            raise ValueError("Period must use UTC calendar-year boundaries and a cutoff within that year.")
        return self

    def contains(self, value: datetime | None) -> bool:
        return value is not None and self.start <= value < self.cutoff

    @property
    def year_to_date(self) -> bool:
        return self.kind == PeriodKind.YEAR and self.cutoff < self.end

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

    @classmethod
    def for_release(cls, *, start: datetime, end: datetime) -> Period:
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("Release dates must be timezone-aware and ordered.")
        return cls(kind=PeriodKind.RELEASE, year=start.astimezone(UTC).year, start=start, end=end, cutoff=end)


class FileChange(Model):
    path: str
    previous_path: str | None = None
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    binary: bool | None = None


class ReleaseBoundary(Model):
    tag: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    published_at: AwareDatetime
    html_url: str


class ReleaseRange(Model):
    base: ReleaseBoundary
    head: ReleaseBoundary
    files: list[FileChange]
    commit_ids: list[str]
    first_parent_commits: list[str]
    base_is_ancestor: bool
    shipped_pr_numbers: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_release(self) -> ReleaseRange:
        if self.base.commit == self.head.commit or self.base.published_at >= self.head.published_at:
            raise ValueError("Choose different, chronologically ordered releases.")
        if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in self.commit_ids):
            raise ValueError("Release commits must be full immutable SHAs.")
        if len(set(self.commit_ids)) != len(self.commit_ids) or not set(self.first_parent_commits) <= set(
            self.commit_ids
        ):
            raise ValueError("Release commit membership is inconsistent.")
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("Release diff contains duplicate file records.")
        return self

    @property
    def label(self) -> str:
        return f"{self.base.tag} -> {self.head.tag}"


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
    closed_at: AwareDatetime | None = None
    merge_commit_sha: str | None = None
    changed_files: int | None = Field(default=None, ge=0)
    paths: list[str] = Field(default_factory=list)
    files_complete: bool = False
    file_changes: list[FileChange] = Field(default_factory=list)
    loc_complete: bool = False

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
                self.closed_at,
                self.merge_commit_sha,
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
        if self.loc_complete and (
            not self.files_complete
            or len(self.file_changes) != self.changed_files
            or {file.path for file in self.file_changes} != set(self.paths)
        ):
            raise ValueError("Complete LOC coverage must include every changed file.")
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
    schema_version: int = 2
    identifier: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at: AwareDatetime
    period: Period
    taxonomy: TaxonomyConfig
    complete: bool = False
    release: ReleaseRange | None = None

    @model_validator(mode="after")
    def _validate_version(self) -> CollectionSession:
        if self.schema_version not in {1, 2}:
            raise ValueError("Unsupported collection-session version; use --refresh.")
        return self


class Snapshot(Model):
    schema_version: int = 2
    repository: str = "microsoft/PyRIT"
    contributor: Actor | None
    period: Period
    collected_at: AwareDatetime
    earliest_response_at: AwareDatetime
    complete: bool
    taxonomy: TaxonomyConfig
    items: list[WorkItem]
    reviews: list[Review]
    comments: list[Comment]
    warnings: list[str] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    release: ReleaseRange | None = None

    @model_validator(mode="after")
    def _validate_integrity(self) -> Snapshot:
        if self.schema_version not in {1, 2} or self.repository != "microsoft/PyRIT" or not self.complete:
            raise ValueError("Only complete version-1 or version-2 public microsoft/PyRIT snapshots can be replayed.")
        if self.contributor is None and self.release is None:
            raise ValueError("A snapshot requires a contributor or release.")
        if self.contributor is not None and self.release is not None:
            raise ValueError("Contributor and release modes must remain separate.")
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
        if self.release is not None:
            if self.period.kind != PeriodKind.RELEASE:
                raise ValueError("Release snapshots require a release activity window.")
            if (
                self.period.start != self.release.base.published_at
                or self.period.cutoff != self.release.head.published_at
            ):
                raise ValueError("Release window must match publication dates.")
            for number in self.release.shipped_pr_numbers:
                if number not in numbers or numbers[number].merge_commit_sha not in self.release.commit_ids:
                    raise ValueError("Shipped PRs must have a merge commit in the pinned release range.")
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


class LineTotals(Model):
    additions: int = 0
    deletions: int = 0


class LocReport(Model):
    scope: LocScope
    complete: bool
    file_count: int
    binary_files: int | None
    totals: LineTotals | None
    by_language: dict[str, LineTotals] | None
    reason: str | None = None


class Peak(Model):
    buckets: list[str]
    count: int


class Stats(Model):
    schema_version: int = 2
    repository: str
    contributor: Actor | None
    release: ReleaseRange | None = None
    period: Period
    collected_at: AwareDatetime
    earliest_response_at: AwareDatetime
    taxonomy: TaxonomyConfig
    counts: dict[Activity, int | None]
    activities: dict[Activity, list[Evidence]]
    classifications: dict[str, Classification]
    breakdowns: dict[Activity, Breakdown]
    monthly: dict[str, dict[Activity, int]]
    distinct_monthly_events: dict[str, int]
    reviewed_authors: list[Actor]
    other_prs_merged: int | None
    own_prs_merged: int | None
    unknown_authors_merged: int | None
    own_pr_comments: int | None
    other_pr_comments: int | None
    unknown_author_pr_comments: int | None
    public_comment_proofs: dict[str, AwareDatetime]
    warnings: list[str]
    loc: LocReport
    peaks: dict[str, Peak]
    participants: dict[str, list[Actor]]
    release_file_topics: dict[str, int] = Field(default_factory=dict)


class SongCandidate(Model):
    title: str
    artist: str
    rationale: str
    selected: bool = False


class Slide(Model):
    type: str
    title: str
    summary: str
    facts: dict[str, int]
    evidence_refs: list[str]
    cue_id: str | None = None
    duration_ms: int | None = None
    song_candidates: list[SongCandidate] = Field(default_factory=list)


class OmittedSlide(Model):
    type: str
    reason: str


class Story(Model):
    schema_version: int = 2
    contributor: Actor | None
    release: ReleaseRange | None = None
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
