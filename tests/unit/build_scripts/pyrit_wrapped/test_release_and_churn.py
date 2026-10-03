# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from build_scripts.pyrit_wrapped.churn import parse_numstat, summarize_churn
from build_scripts.pyrit_wrapped.cli import main
from build_scripts.pyrit_wrapped.github_client import GitHubClient, Response
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import (
    Activity,
    Actor,
    Capability,
    FileChange,
    LocScope,
    Period,
    ReleaseBoundary,
    ReleaseRange,
    Snapshot,
    WorkItem,
    WrappedError,
)
from build_scripts.pyrit_wrapped.release import ReleaseResolver
from build_scripts.pyrit_wrapped.reviews import ReviewReader
from build_scripts.pyrit_wrapped.snapshot import Collector
from build_scripts.pyrit_wrapped.story import StoryBuilder


@pytest.fixture
def release_range() -> ReleaseRange:
    return ReleaseRange(
        base=ReleaseBoundary(
            tag="v1.0.1",
            commit="a" * 40,
            published_at=datetime(2026, 7, 30, tzinfo=UTC),
            html_url="https://github.com/microsoft/PyRIT/releases/tag/v1.0.1",
        ),
        head=ReleaseBoundary(
            tag="v1.1.0",
            commit="b" * 40,
            published_at=datetime(2026, 9, 4, tzinfo=UTC),
            html_url="https://github.com/microsoft/PyRIT/releases/tag/v1.1.0",
        ),
        files=[
            FileChange(path="frontend/src/example.tsx", additions=10, deletions=3, binary=False),
            FileChange(path="pyrit/example.py", additions=20, deletions=8, binary=False),
            FileChange(path=".github/example.yml", additions=5, deletions=2, binary=False),
        ],
        commit_ids=["c" * 40],
        first_parent_commits=["c" * 40],
        base_is_ancestor=True,
        shipped_pr_numbers=[1],
    )


def test_git_numstat_handles_renames_binary_tabs_and_newlines() -> None:
    value = b"2\t1\tfoo\tbar.py\0-\t-\timage.png\0" + b"1\t1\t\0old.py\0new.ts\0" + b"3\t0\tline\nname.yml\0"
    files = parse_numstat(value)
    assert files[0].path == "foo\tbar.py"
    assert files[1].binary and files[1].additions == files[1].deletions == 0
    assert files[2].previous_path == "old.py"
    assert files[2].path == "new.ts"
    assert files[3].path == "line\nname.yml"


@pytest.mark.parametrize("value", [b"bad\0", b"x\t1\tfoo.py\0", b"1\t1\t\0old.py\0"])
def test_malformed_numstat_fails(value: bytes) -> None:
    with pytest.raises(WrappedError):
        parse_numstat(value)


def test_language_churn_uses_old_language_for_rename_deletions() -> None:
    loc = summarize_churn(
        files=[FileChange(path="new.ts", previous_path="old.py", additions=10, deletions=5, binary=False)],
        scope=LocScope.RELEASE_DIFF,
        complete=True,
    )
    assert loc.totals is not None and loc.by_language is not None
    assert loc.by_language["TypeScript"].additions == 10
    assert loc.by_language["Python"].deletions == 5
    assert loc.by_language["YAML"].additions == 0
    assert loc.totals.additions == 10 and loc.totals.deletions == 5


def test_incomplete_churn_is_not_a_zero_or_partial_total() -> None:
    loc = summarize_churn(
        files=[FileChange(path="x.py", additions=10, deletions=3)],
        scope=LocScope.LANDED_PRS,
        complete=False,
        reason="Missing files",
    )
    assert loc.totals is loc.by_language is None
    assert loc.reason == "Missing files"


def test_release_period_crosses_years() -> None:
    window = Period.for_release(start=datetime(2025, 12, 20, tzinfo=UTC), end=datetime(2026, 1, 10, tzinfo=UTC))
    assert window.contains(datetime(2026, 1, 1, tzinfo=UTC))
    assert not window.contains(window.cutoff)
    assert not window.year_to_date


def test_release_scope_includes_all_accounts_and_exact_shipped_membership(
    *, release_range: ReleaseRange, snapshot: Snapshot, item: WorkItem
) -> None:
    window = Period.for_release(start=release_range.base.published_at, end=release_range.head.published_at)
    bot = Actor(id="BOT_1", login="example[bot]", type="Bot")
    item = item.model_copy(
        update={
            "author": bot,
            "created_at": window.start,
            "merged_at": window.start + timedelta(days=1),
            "closed_at": window.start + timedelta(days=1),
            "merge_commit_sha": "c" * 40,
        }
    )
    value = Snapshot.model_validate(
        {
            **snapshot.model_dump(),
            "contributor": None,
            "period": window,
            "release": release_range,
            "items": [item],
            "capabilities": [Capability.CLOSURES, Capability.LOC],
        }
    )
    stats = Metrics(value).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.SHIPPED] == stats.counts[Activity.PR_CLOSED] == 1
    assert stats.own_prs_merged is None
    assert stats.participants["authors"] == [bot]
    assert stats.loc.totals is not None
    assert stats.loc.totals.additions == 35 and stats.loc.totals.deletions == 13
    assert 5 <= len(StoryBuilder(stats).build().slides) <= 10


def test_shipped_pr_can_be_outside_activity_dates(
    *, release_range: ReleaseRange, snapshot: Snapshot, item: WorkItem
) -> None:
    window = Period.for_release(start=release_range.base.published_at, end=release_range.head.published_at)
    item = item.model_copy(
        update={
            "created_at": window.start - timedelta(days=10),
            "merged_at": window.start - timedelta(days=1),
            "merge_commit_sha": "c" * 40,
        }
    )
    value = Snapshot.model_validate(
        {**snapshot.model_dump(), "contributor": None, "period": window, "release": release_range, "items": [item]}
    )
    stats = Metrics(value).calculate()
    assert stats.counts[Activity.SHIPPED] == 1
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.MERGED] == 0


def test_shipped_pr_must_have_a_pinned_merge_commit(
    *, release_range: ReleaseRange, snapshot: Snapshot, item: WorkItem
) -> None:
    window = Period.for_release(start=release_range.base.published_at, end=release_range.head.published_at)
    with pytest.raises(ValidationError, match="pinned release range"):
        Snapshot.model_validate(
            {**snapshot.model_dump(), "contributor": None, "period": window, "release": release_range, "items": [item]}
        )


def test_older_own_pr_closed_in_period_and_merge_is_one_event(*, snapshot: Snapshot, item: WorkItem) -> None:
    item = item.model_copy(update={"created_at": datetime(2024, 1, 1, tzinfo=UTC), "closed_at": item.merged_at})
    stats = Metrics(snapshot.model_copy(update={"items": [item], "capabilities": [Capability.CLOSURES]})).calculate()
    assert stats.counts[Activity.PR_CLOSED] == 1
    assert stats.counts[Activity.AUTHORED] == 0
    assert stats.peaks["day"].count == stats.peaks["week"].count == stats.peaks["month"].count == 1


def test_legacy_snapshot_exposes_missing_metrics(*, snapshot: Snapshot, item: WorkItem) -> None:
    value = Snapshot.model_validate({**snapshot.model_dump(), "schema_version": 1, "items": [item]})
    stats = Metrics(value).calculate()
    assert stats.counts[Activity.PR_CLOSED] is None
    assert stats.counts[Activity.ISSUES_CLOSED] is None
    assert not stats.loc.complete and stats.loc.totals is None
    assert stats.counts[Activity.AUTHORED] == 1


def test_contributor_loc_only_counts_landed_prs(*, snapshot: Snapshot, item: WorkItem) -> None:
    file = FileChange(path="x.py", additions=9, deletions=4)
    landed = item.model_copy(update={"file_changes": [file], "loc_complete": True})
    opened = item.model_copy(
        update={
            "id": "PR_2",
            "number": 2,
            "merged_at": None,
            "file_changes": [FileChange(path="x.py", additions=500, deletions=300)],
            "loc_complete": True,
        }
    )
    stats = Metrics(
        snapshot.model_copy(update={"items": [landed, opened], "capabilities": [Capability.LOC]})
    ).calculate()
    assert stats.loc.totals is not None and stats.loc.totals.additions == 9
    assert stats.loc.totals.deletions == 4
    assert stats.loc.binary_files is None


def test_iso_week_and_peak_ties_are_preserved(*, snapshot: Snapshot, item: WorkItem) -> None:
    first = item.model_copy(update={"created_at": datetime(2026, 1, 1, tzinfo=UTC), "merged_at": None})
    second = first.model_copy(update={"id": "PR_2", "number": 2, "created_at": datetime(2026, 1, 2, tzinfo=UTC)})
    stats = Metrics(snapshot.model_copy(update={"items": [first, second]})).calculate()
    assert stats.peaks["day"].buckets == ["2026-01-01", "2026-01-02"]
    assert stats.peaks["week"].buckets == ["2026-W01"] and stats.peaks["week"].count == 2


async def test_graphql_reader_paginates_each_pr_independently() -> None:
    client = MagicMock(spec=GitHubClient)
    client.progress = MagicMock()
    node = {
        "fullDatabaseId": "5000000000",
        "state": "APPROVED",
        "submittedAt": "2026-08-01T00:00:00Z",
        "url": "https://github.com/microsoft/PyRIT/pull/1#pullrequestreview-5000000000",
        "_wrapped_has_body": False,
        "author": {"id": "NEW_USER_ID", "login": "owner", "__typename": "User", "databaseId": 123},
    }

    def page(*, nodes: list[dict], next_page: bool, cursor: str) -> dict:
        return {"reviews": {"nodes": nodes, "pageInfo": {"hasNextPage": next_page, "endCursor": cursor}}}

    client.get_async = AsyncMock(
        side_effect=[
            Response(
                fetched_at=datetime.now(UTC),
                next_page=None,
                data={
                    "data": {
                        "repository": {
                            "p1": page(nodes=[node], next_page=True, cursor="cursor1"),
                            "p2": page(nodes=[], next_page=False, cursor="done"),
                        }
                    }
                },
            ),
            Response(
                fetched_at=datetime.now(UTC),
                next_page=None,
                data={
                    "data": {
                        "repository": {
                            "p1": page(nodes=[], next_page=False, cursor="cursor2"),
                        }
                    }
                },
            ),
        ]
    )
    reviews = await ReviewReader(client).read_async([1, 2])
    assert len(reviews) == 1 and reviews[0].id == 5000000000
    assert reviews[0].author is not None and reviews[0].author.database_id == 123
    assert "p2:" not in client.get_async.call_args_list[1].kwargs["params"]["query"]


async def test_graphql_mutations_are_rejected(tmp_path: Path) -> None:
    client = GitHubClient(cache_dir=tmp_path, progress=lambda message: None)
    with pytest.raises(WrappedError, match="read-only"):
        await client.get_async(path="graphql", params={"query": "query { viewer { login } } mutation { bad }"})


async def test_release_tag_validation_precedes_network(tmp_path: Path) -> None:
    client = GitHubClient(cache_dir=tmp_path, progress=lambda message: None)
    with pytest.raises(WrappedError, match="release tag"):
        await ReleaseResolver(client)._boundary_async("--upload-pack=bad")


def test_release_cli_rejects_year_without_collecting(tmp_path: Path) -> None:
    assert main(["summarize", "--release", "v1.1.0", "--year", "2026", "--output-dir", str(tmp_path / "out")]) == 1


def test_rest_and_graphql_account_ids_are_equivalent() -> None:
    from build_scripts.pyrit_wrapped.snapshot import same_actor

    assert same_actor(
        Actor(id="legacy-node", login="owner", database_id=123),
        Actor(id="new-node", login="owner", database_id=123),
    )
    assert not same_actor(
        Actor(id="legacy-node", login="owner", database_id=123),
        Actor(id="legacy-node", login="owner", database_id=124),
    )


def test_complete_loc_rejects_incomplete_file_records(item: WorkItem) -> None:
    with pytest.raises(ValidationError, match="every changed file"):
        WorkItem.model_validate({**item.model_dump(), "loc_complete": True, "file_changes": []})


async def test_previous_stable_release_is_selected(tmp_path: Path, release_range: ReleaseRange) -> None:
    client = GitHubClient(cache_dir=tmp_path, progress=lambda message: None)
    records = [
        {"tag_name": "v1.1.0", "published_at": "2026-09-04T00:00:00Z", "html_url": "", "draft": False},
        {
            "tag_name": "v1.1.0rc1",
            "published_at": "2026-09-01T00:00:00Z",
            "html_url": "",
            "draft": False,
            "prerelease": True,
        },
        {"tag_name": "v1.0.1", "published_at": "2026-07-30T00:00:00Z", "html_url": "", "draft": False},
    ]
    resolver = ReleaseResolver(client)
    with patch.object(client, "list_async", new_callable=AsyncMock, return_value=records):
        with patch.object(resolver, "_boundary_async", new_callable=AsyncMock) as boundary:
            boundary.side_effect = [release_range.base, release_range.head]
            with patch.object(resolver, "_ensure_commit_async", new_callable=AsyncMock):
                with patch.object(resolver, "_git_async", new_callable=AsyncMock) as git:
                    git.side_effect = [(0, b""), (0, b""), (0, b""), (0, b"")]
                    result = await resolver.resolve_async(base_tag=None, head_tag="v1.1.0")
    assert result.base.tag == "v1.0.1"
    assert boundary.call_args_list[0].args == ("v1.0.1",)


async def test_closure_census_includes_old_unmerged_prs_without_search(*, period: Period, contributor: Actor) -> None:
    record = {
        "node_id": "PR_old",
        "number": 7,
        "title": "Old unmerged PR",
        "html_url": "https://github.com/microsoft/PyRIT/pull/7",
        "user": {"node_id": contributor.id, "login": contributor.login, "type": "User"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
        "closed_at": "2026-08-01T00:00:00Z",
        "state": "closed",
        "labels": [],
        "pull_request": {"url": "https://api.github.com/repos/microsoft/PyRIT/pulls/7"},
    }
    client = MagicMock(spec=GitHubClient)
    client.list_async = AsyncMock(return_value=[record])
    client.search_async = AsyncMock()
    prs, issues = await Collector(client=client, progress=lambda message: None)._closed_numbers_async(
        contributor=contributor, period=period
    )
    assert prs == {7} and not issues
    assert client.search_async.call_count == 0
    assert client.list_async.call_args.kwargs["params"]["state"] == "all"
