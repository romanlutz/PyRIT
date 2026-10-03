# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

import httpx
from pydantic import AwareDatetime, BaseModel, Field

from build_scripts.pyrit_wrapped.github_client import (
    GitHubClient,
    GitHubHttpError,
    iso_time,
    object_data,
    write_json_atomic,
)
from build_scripts.pyrit_wrapped.models import (
    Actor,
    Capability,
    Comment,
    CommentKind,
    FileChange,
    ItemKind,
    Period,
    ReleaseRange,
    Review,
    Snapshot,
    TaxonomyConfig,
    WorkItem,
    WrappedError,
)
from build_scripts.pyrit_wrapped.reviews import ReviewReader

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class ApiActor(BaseModel):
    node_id: str
    login: str
    type: str = "User"
    id: int | None = None

    def to_actor(self) -> Actor:
        return Actor(id=self.node_id, login=self.login, type=self.type, database_id=self.id)


class ApiLabel(BaseModel):
    name: str


class ApiItem(BaseModel):
    node_id: str
    number: int
    title: str
    html_url: str
    user: ApiActor | None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    state: str
    labels: list[ApiLabel]
    merged_at: AwareDatetime | None = None
    merged_by: ApiActor | None = None
    changed_files: int | None = None
    closed_at: AwareDatetime | None
    merge_commit_sha: str | None = None

    def to_item(self, kind: ItemKind) -> WorkItem:
        return WorkItem(
            id=self.node_id,
            kind=kind,
            number=self.number,
            title=self.title,
            url=self.html_url,
            author=self.user.to_actor() if self.user else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
            state=self.state,
            labels=[label.name for label in self.labels],
            merged_at=self.merged_at,
            merged_by=self.merged_by.to_actor() if self.merged_by else None,
            changed_files=self.changed_files,
            closed_at=self.closed_at,
            merge_commit_sha=self.merge_commit_sha,
        )


class ApiReview(BaseModel):
    id: int
    user: ApiActor | None
    state: str
    submitted_at: AwareDatetime | None = None
    html_url: str
    has_body: bool = Field(alias="_wrapped_has_body", default=False)

    def to_review(self, number: int) -> Review:
        return Review(
            id=self.id,
            item_number=number,
            author=self.user.to_actor() if self.user else None,
            state=self.state,
            submitted_at=self.submitted_at,
            has_body=self.has_body,
            url=self.html_url,
        )


class ApiComment(BaseModel):
    id: int
    user: ApiActor | None
    created_at: AwareDatetime
    html_url: str
    issue_url: str | None = None
    pull_request_url: str | None = None
    pull_request_review_id: int | None = None
    path: str | None = None

    def to_comment(self, kind: CommentKind) -> Comment:
        item_url = self.pull_request_url if kind == CommentKind.INLINE else self.issue_url
        match = re.fullmatch(r"https://api\.github\.com/repos/microsoft/PyRIT/(?:pulls|issues)/(\d+)", item_url or "")
        if match is None:
            raise WrappedError("Comment points to an unexpected repository or item.")
        return Comment(
            id=self.id,
            item_number=int(match[1]),
            author=self.user.to_actor() if self.user else None,
            kind=kind,
            created_at=self.created_at,
            url=self.html_url,
            review_id=self.pull_request_review_id,
            path=self.path,
        )


class ApiRepository(BaseModel):
    full_name: str
    private: bool
    created_at: AwareDatetime


class ApiFile(BaseModel):
    filename: str
    additions: int | None = None
    deletions: int | None = None
    previous_filename: str | None = None


T = TypeVar("T")


class Collector:
    def __init__(self, *, client: GitHubClient, progress: Callable[[str], None]) -> None:
        self.client = client
        self.progress = progress

    async def collect_async(
        self, *, login: str | None, period: Period, taxonomy: TaxonomyConfig, release: ReleaseRange | None = None
    ) -> Snapshot:
        if (login is None) == (release is None):
            raise WrappedError("Choose one contributor or release scope.")
        contributor = (
            ApiActor.model_validate((await self.client.get_async(path=f"users/{login}")).data).to_actor()
            if login is not None
            else None
        )
        if contributor is not None and contributor.is_deleted:
            raise WrappedError("The GitHub ghost account represents deleted accounts, not one contributor.")
        repository = ApiRepository.model_validate((await self.client.get_async(path="repos/microsoft/PyRIT")).data)
        if repository.private or repository.full_name.lower() != "microsoft/pyrit":
            raise WrappedError("Only the public microsoft/PyRIT repository is supported.")
        authored, issues, merged, reviewed = await self._candidate_numbers_async(
            contributor=contributor, period=period, repository=repository
        )
        pr_closed, issue_closed = await self._closed_numbers_async(contributor=contributor, period=period)
        associations = await self._release_associations_async(release) if release is not None else set()
        self.progress("Collecting comments in the activity window.")
        comments = await self._comments_async(contributor=contributor, period=period)
        reviews = await ReviewReader(self.client).read_async(sorted(reviewed)) if release is not None else []
        reviewed_active = {
            review.item_number
            for review in reviews
            if period.contains(review.submitted_at) and review.state != "PENDING"
        }
        initial = authored | merged | pr_closed | associations | (reviewed_active if release is not None else reviewed)
        pulls = await self._batch_async(numbers=sorted(initial), fetch=self._pull_async)
        items = {item.number: item for item in pulls}
        warnings = await self._load_missing_items_async(
            required_issues=issues | issue_closed, comments=comments, items=items
        )
        excluded = sorted(
            number
            for number in authored
            if contributor is not None and not same_actor(items[number].author, contributor)
        )
        if excluded:
            warnings.append(
                f"Author search returned {len(excluded)} PRs with a different or unavailable recorded author. "
                "These are not credited as contributor-authored PRs: "
                + ", ".join(f"pr:{number}" for number in excluded)
                + ". Agent-associated search matches do not establish human authorship."
            )
        review_numbers = {comment.item_number for comment in comments if comment.kind == CommentKind.INLINE}
        if release is None:
            review_numbers |= reviewed
        else:
            review_numbers -= reviewed
        review_lists = await self._batch_async(
            numbers=sorted(number for number in review_numbers if items[number].available), fetch=self._reviews_async
        )
        reviews.extend(review for batch in review_lists for review in batch)
        await self._verify_orphan_comments_async(comments=comments, reviews=reviews)
        relevant = self._relevant_numbers(
            contributor=contributor, period=period, items=items, reviews=reviews, comments=comments
        )
        if release is not None:
            release.shipped_pr_numbers = sorted(
                number
                for number in associations
                if items[number].merged_at is not None and items[number].merge_commit_sha in release.commit_ids
            )
            relevant.update(release.shipped_pr_numbers)
            if not release.base_is_ancestor:
                warnings.append(
                    "Release tags have divergent histories. LOC compares their trees; shipped PRs are associated with "
                    "newly reachable merge commits, not a guarantee every patch is first shipped."
                )
        items = {number: item for number, item in items.items() if number in relevant}
        warnings.extend(await self._load_files_async(items))
        required_review_ids = {comment.review_id for comment in comments if comment.review_id is not None}
        reviews = [
            review
            for review in reviews
            if review.item_number in relevant
            and (
                review.id in required_review_ids
                or (in_scope(review.author, contributor) and period.contains(review.submitted_at))
            )
        ]
        ordered_items: list[WorkItem] = sorted(items.values(), key=lambda item: item.number)
        ordered_reviews: list[Review] = sorted(
            {review.id: review for review in reviews}.values(), key=lambda review: review.id
        )
        ordered_comments: list[Comment] = sorted(
            {comment.ref: comment for comment in comments}.values(), key=lambda comment: comment.ref
        )
        return Snapshot(
            contributor=contributor,
            period=period,
            collected_at=datetime.now(UTC),
            earliest_response_at=min(self.client.response_times),
            complete=True,
            taxonomy=taxonomy,
            items=ordered_items,
            reviews=ordered_reviews,
            comments=ordered_comments,
            warnings=sorted(set(warnings)),
            release=release,
            capabilities=[Capability.CLOSURES, Capability.LOC],
        )

    async def _candidate_numbers_async(
        self, *, contributor: Actor | None, period: Period, repository: ApiRepository
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        self.progress(
            f"Searching public PyRIT activity for {'@' + contributor.login if contributor else 'all contributors'}."
        )
        author = f" author:{contributor.login}" if contributor is not None else ""
        authored = await self.client.search_async(
            query=f"is:pr{author}", date_field="created", start=period.start, end=period.cutoff
        )
        issues = await self.client.search_async(
            query=f"is:issue{author}", date_field="created", start=period.start, end=period.cutoff
        )
        merged = await self.client.search_async(
            query="is:pr is:merged", date_field="merged", start=period.start, end=period.cutoff
        )
        reviewed = await self.client.search_async(
            query=f"is:pr reviewed-by:{contributor.login}" if contributor is not None else "is:pr",
            date_field="created",
            start=repository.created_at,
            end=period.cutoff,
        )
        return authored, issues, merged, reviewed

    async def _closed_numbers_async(self, *, contributor: Actor | None, period: Period) -> tuple[set[int], set[int]]:
        records = await self.client.list_async(
            path="repos/microsoft/PyRIT/issues",
            params={
                "state": "all",
                "since": iso_time(period.start - timedelta(seconds=1)),
                "sort": "created",
                "direction": "asc",
            },
            stop_after=period.cutoff,
        )
        prs: set[int] = set()
        issues: set[int] = set()
        for record in records:
            item = ApiItem.model_validate(record)
            author = item.user.to_actor() if item.user is not None else None
            if item.state == "closed" and in_scope(author, contributor) and period.contains(item.closed_at):
                (prs if "pull_request" in record else issues).add(item.number)
        return prs, issues

    async def _release_associations_async(self, release: ReleaseRange) -> set[int]:
        async def lookup_async(sha: str) -> set[int]:
            values = await self.client.list_async(path=f"repos/microsoft/PyRIT/commits/{sha}/pulls")
            return {ApiItem.model_validate(value).number for value in values}

        numbers: set[int] = set()
        for offset in range(0, len(release.commit_ids), 30):
            batches = await asyncio.gather(*(lookup_async(sha) for sha in release.commit_ids[offset : offset + 30]))
            for batch in batches:
                numbers.update(batch)
        return numbers

    async def _comments_async(self, *, contributor: Actor | None, period: Period) -> list[Comment]:
        comments: list[Comment] = []
        for endpoint, kind in (("pulls/comments", CommentKind.INLINE), ("issues/comments", CommentKind.DISCUSSION)):
            raw = await self.client.list_async(
                path=f"repos/microsoft/PyRIT/{endpoint}",
                params={"since": iso_time(period.start), "sort": "created", "direction": "asc"},
                stop_after=period.cutoff,
            )
            for value in raw:
                comment = ApiComment.model_validate(value).to_comment(kind)
                if in_scope(comment.author, contributor) and period.contains(comment.created_at):
                    comments.append(comment)
        return comments

    async def _pull_async(self, number: int) -> WorkItem:
        response = await self.client.get_async(path=f"repos/microsoft/PyRIT/pulls/{number}")
        return ApiItem.model_validate(response.data).to_item(ItemKind.PR)

    async def _issue_async(self, number: int) -> WorkItem:
        response = await self.client.get_async(path=f"repos/microsoft/PyRIT/issues/{number}")
        value = object_data(response.data)
        if "pull_request" in value:
            return await self._pull_async(number)
        return ApiItem.model_validate(value).to_item(ItemKind.ISSUE)

    async def _reviews_async(self, number: int) -> list[Review]:
        records = await self.client.list_async(path=f"repos/microsoft/PyRIT/pulls/{number}/reviews")
        return [ApiReview.model_validate(record).to_review(number) for record in records]

    async def _load_missing_items_async(
        self, *, required_issues: set[int], comments: list[Comment], items: dict[int, WorkItem]
    ) -> list[str]:
        by_number = {comment.item_number: comment for comment in comments}
        missing = sorted((required_issues | by_number.keys()) - items.keys())
        warnings: list[str] = []

        async def fetch_async(number: int) -> WorkItem:
            try:
                return await self._issue_async(number)
            except GitHubHttpError as error:
                if error.status != 404 or number in required_issues or number not in by_number:
                    raise
                comment = by_number[number]
                kind = ItemKind.PR if "/pull/" in comment.url else ItemKind.ISSUE
                warnings.append(
                    f"{kind.value}:{number}: parent metadata is unavailable (HTTP 404). "
                    "Only observable comments are retained; unavailable reviews cannot be reconstructed."
                )
                return WorkItem(
                    id=None,
                    kind=kind,
                    number=number,
                    title=f"Unavailable {kind.value.upper()} #{number}",
                    url=comment.url.partition("#")[0],
                    author=None,
                    created_at=None,
                    updated_at=None,
                    state=None,
                    available=False,
                )

        for item in await self._batch_async(numbers=missing, fetch=fetch_async):
            items[item.number] = item
        return warnings

    async def _verify_orphan_comments_async(self, *, comments: list[Comment], reviews: list[Review]) -> None:
        review_ids = {review.id for review in reviews}
        orphaned = [
            comment
            for comment in comments
            if comment.kind == CommentKind.INLINE
            and comment.review_id is not None
            and comment.review_id not in review_ids
        ]
        if not orphaned:
            return
        self.progress(f"Verifying public publication for {len(orphaned)} comments with unavailable reviews.")
        async with httpx.AsyncClient(
            timeout=30, auth=None, headers={"Accept": "application/vnd.github+json"}
        ) as public_client:
            for comment in orphaned:
                await self._verify_public_comment_async(comment=comment, public_client=public_client)

    async def _verify_public_comment_async(self, *, comment: Comment, public_client: httpx.AsyncClient) -> None:
        cache_file = self.client.cache_dir / f"public-comment-{comment.id}.json"
        cached = await asyncio.to_thread(self._read_public_proof, cache_file)
        if cached is None:
            endpoint = f"https://api.github.com/repos/microsoft/PyRIT/pulls/comments/{comment.id}"
            for attempt in range(3):
                try:
                    response = await public_client.get(endpoint)
                except httpx.TransportError as error:
                    if attempt == 2:
                        raise WrappedError(f"Public comment verification failed for {comment.ref}: {error}") from error
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                if response.status_code != 200:
                    message = (
                        response.json().get("message", "")
                        if response.headers.get("content-type", "").startswith("application/json")
                        else ""
                    )
                    rate_limited = response.status_code in {403, 429} and (
                        "rate limit" in str(message).lower() or "retry-after" in response.headers
                    )
                    if rate_limited and attempt < 2:
                        delay = min(120, max(1, float(response.headers.get("retry-after", "10"))))
                        self.progress(f"Public comment proof is rate-limited; retrying after {delay:.0f} seconds.")
                        await asyncio.sleep(delay)
                        continue
                    raise WrappedError(
                        f"Cannot verify publication of {comment.ref} without its review: "
                        f"public HTTP {response.status_code}."
                    )
                cached = ApiComment.model_validate(response.json()).to_comment(CommentKind.INLINE)
                cached.publicly_verified_at = datetime.now(UTC)
                break
        same_author = cached is not None and (
            cached.author is comment.author is None or same_actor(cached.author, comment.author)
        )
        if cached is None or not same_author or self._comment_signature(cached) != self._comment_signature(comment):
            raise WrappedError(f"Public comment proof does not match {comment.ref}; no complete report was generated.")
        if cached.publicly_verified_at is None:
            raise WrappedError(f"Public comment proof lacks a verification timestamp: {comment.ref}")
        comment.publicly_verified_at = cached.publicly_verified_at
        await asyncio.to_thread(write_json_atomic, path=cache_file, content=cached.model_dump_json())
        self.client.response_times.append(cached.publicly_verified_at)

    @staticmethod
    def _read_public_proof(path: Path) -> Comment | None:
        return Comment.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    @staticmethod
    def _comment_signature(comment: Comment) -> tuple[int, int, datetime, int | None, str]:
        return (
            comment.id,
            comment.item_number,
            comment.created_at,
            comment.review_id,
            comment.url,
        )

    async def _load_files_async(self, items: dict[int, WorkItem]) -> list[str]:
        pulls = [item for item in items.values() if item.kind == ItemKind.PR and item.available]

        async def load_async(number: int) -> str | None:
            item = items[number]
            values = await self.client.list_async(path=f"repos/microsoft/PyRIT/pulls/{number}/files")
            files = [ApiFile.model_validate(value) for value in values]
            item.paths = [file.filename for file in files]
            item.files_complete = len(item.paths) == item.changed_files and len(set(item.paths)) == len(item.paths)
            item.loc_complete = item.files_complete and all(
                file.additions is not None and file.deletions is not None for file in files
            )
            item.file_changes = [
                FileChange(
                    path=file.filename,
                    previous_path=file.previous_filename,
                    additions=file.additions,
                    deletions=file.deletions,
                )
                for file in files
                if file.additions is not None and file.deletions is not None
            ]
            if not item.files_complete:
                return (
                    f"{item.ref}: file coverage incomplete "
                    f"({len(item.paths)} of {item.changed_files}); topics are inferred."
                )
            return None

        warnings = await self._batch_async(numbers=sorted(item.number for item in pulls), fetch=load_async)
        return [warning for warning in warnings if warning is not None]

    async def _batch_async(self, *, numbers: list[int], fetch: Callable[[int], Awaitable[T]]) -> list[T]:
        results: list[T] = []
        for offset in range(0, len(numbers), 50):
            results.extend(await asyncio.gather(*(fetch(number) for number in numbers[offset : offset + 50])))
            self.progress(f"Collected {min(offset + 50, len(numbers))}/{len(numbers)} records.")
        return results

    @staticmethod
    def _relevant_numbers(
        *,
        contributor: Actor | None,
        period: Period,
        items: dict[int, WorkItem],
        reviews: list[Review],
        comments: list[Comment],
    ) -> set[int]:
        relevant = {comment.item_number for comment in comments}
        relevant.update(
            review.item_number
            for review in reviews
            if in_scope(review.author, contributor)
            and review.state != "PENDING"
            and period.contains(review.submitted_at)
        )
        for item in items.values():
            if in_scope(item.author, contributor) and (
                period.contains(item.created_at) or period.contains(item.merged_at) or period.contains(item.closed_at)
            ):
                relevant.add(item.number)
            if in_scope(item.merged_by, contributor) and period.contains(item.merged_at):
                relevant.add(item.number)
        return relevant


def same_actor(left: Actor | None, right: Actor | None) -> bool:
    if left is None or right is None:
        return False
    if left.database_id is not None and right.database_id is not None:
        return left.database_id == right.database_id
    return left.id == right.id


def in_scope(actor: Actor | None, contributor: Actor | None) -> bool:
    return contributor is None or same_actor(actor, contributor)
