# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from build_scripts.pyrit_wrapped.github_client import GitHubClient, GitHubHttpError
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import Activity, Comment, Snapshot, WorkItem, WrappedError
from build_scripts.pyrit_wrapped.snapshot import Collector


@pytest.fixture
def collector(tmp_path: Path) -> Collector:
    client = GitHubClient(cache_dir=tmp_path / "cache", progress=lambda message: None)
    return Collector(client=client, progress=lambda message: None)


async def test_missing_comment_parent_stays_unknown(*, collector: Collector, comment: Comment) -> None:
    failure = GitHubHttpError(status=404, path="issues/1", message="Not Found")
    items: dict[int, WorkItem] = {}
    with patch.object(collector, "_issue_async", new_callable=AsyncMock, side_effect=failure):
        warnings = await collector._load_missing_items_async(required_issues=set(), comments=[comment], items=items)
    item = items[1]
    assert not item.available
    assert item.id is item.created_at is item.updated_at is item.state is item.author is None
    assert "HTTP 404" in warnings[0]
    assert item.title == "Unavailable PR #1"


async def test_required_issue_failure_does_not_become_unknown(*, collector: Collector, comment: Comment) -> None:
    failure = GitHubHttpError(status=404, path="issues/1", message="Not Found")
    with patch.object(collector, "_issue_async", new_callable=AsyncMock, side_effect=failure):
        with pytest.raises(GitHubHttpError):
            await collector._load_missing_items_async(required_issues={1}, comments=[comment], items={})


async def test_public_endpoint_proves_orphan_comment_publication(
    *, collector: Collector, comment: Comment, snapshot: Snapshot
) -> None:
    public_client = MagicMock(spec=httpx.AsyncClient)
    public_client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "id": comment.id,
                "user": {"node_id": snapshot.contributor.id, "login": snapshot.contributor.login},
                "created_at": comment.created_at.isoformat(),
                "html_url": comment.url,
                "pull_request_url": "https://api.github.com/repos/microsoft/PyRIT/pulls/1",
                "pull_request_review_id": comment.review_id,
                "path": comment.path,
                "body": "not stored in proof",
            },
        )
    )
    await collector._verify_public_comment_async(comment=comment, public_client=public_client)
    assert comment.publicly_verified_at is not None
    await collector._verify_public_comment_async(comment=comment, public_client=public_client)
    assert public_client.get.call_count == 1
    assert "not stored in proof" not in next(collector.client.cache_dir.glob("public-comment-*.json")).read_text()
    parent = WorkItem(
        id=None,
        number=1,
        kind="pr",
        title="Unavailable PR #1",
        url="https://github.com/microsoft/PyRIT/pull/1",
        author=None,
        created_at=None,
        updated_at=None,
        state=None,
        available=False,
    )
    value = Snapshot.model_validate({**snapshot.model_dump(), "items": [parent], "comments": [comment]})
    stats = Metrics(value).calculate()
    assert stats.counts[Activity.INLINE] == 1
    assert stats.counts[Activity.REVIEWS] == stats.counts[Activity.REVIEWED] == 0
    assert stats.unknown_author_pr_comments == 1
    assert comment.ref in stats.public_comment_proofs


async def test_orphan_comment_is_not_counted_without_public_proof(*, collector: Collector, comment: Comment) -> None:
    public_client = MagicMock(spec=httpx.AsyncClient)
    public_client.get = AsyncMock(return_value=httpx.Response(404))
    with pytest.raises(WrappedError, match="Cannot verify publication"):
        await collector._verify_public_comment_async(comment=comment, public_client=public_client)
    assert comment.publicly_verified_at is None
