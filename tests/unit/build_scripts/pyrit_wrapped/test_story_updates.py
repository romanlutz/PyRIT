# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from build_scripts.pyrit_wrapped.cli import main
from build_scripts.pyrit_wrapped.contributors import ContributorCredits
from build_scripts.pyrit_wrapped.github_client import GitHubClient, Response
from build_scripts.pyrit_wrapped.html_deck import HtmlDeck
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import (
    Activity,
    Actor,
    Capability,
    FileChange,
    LineTotals,
    LocScope,
    Period,
    Review,
    Snapshot,
    WorkItem,
    WrappedError,
)
from build_scripts.pyrit_wrapped.reviews import ReviewReader
from build_scripts.pyrit_wrapped.snapshot import Collector
from build_scripts.pyrit_wrapped.story import StoryBuilder


def repository_snapshot(*, snapshot: Snapshot, items: list[WorkItem]) -> Snapshot:
    return Snapshot.model_validate(
        {
            **snapshot.model_dump(),
            "contributor": None,
            "repository_year": True,
            "items": items,
            "capabilities": [Capability.CLOSURES, Capability.LOC],
        }
    )


def test_repository_year_is_explicit_and_separate(snapshot: Snapshot) -> None:
    with pytest.raises(ValidationError, match="explicit repository-year"):
        Snapshot.model_validate({**snapshot.model_dump(), "contributor": None})
    with pytest.raises(ValidationError, match="no contributor or release"):
        Snapshot.model_validate({**snapshot.model_dump(), "repository_year": True})
    whole = repository_snapshot(snapshot=snapshot, items=[])
    assert whole.repository_year and whole.contributor is None and whole.release is None


def test_cover_names_contributor_and_year(snapshot: Snapshot) -> None:
    stats = Metrics(snapshot).calculate()
    story = StoryBuilder(stats).build()
    assert story.slides[0].title == "PyRIT Wrapped"
    assert "@owner" in story.slides[0].summary and "2026" in story.slides[0].summary
    assert not story.slides[0].song_candidates
    assert "PyRIT Wrapped" in HtmlDeck(stats=stats, story=story).render()


def test_repository_cover_and_credit_slide(snapshot: Snapshot) -> None:
    stats = Metrics(repository_snapshot(snapshot=snapshot, items=[])).calculate()
    story = StoryBuilder(stats).build()
    assert story.slides[0].summary.startswith("PyRIT / 2026")
    assert "contributors" in [slide.type for slide in story.slides]
    assert len(story.slides) <= 10


def test_merged_intent_excludes_unmerged_work(*, snapshot: Snapshot, item: WorkItem) -> None:
    opened = item.model_copy(update={"id": "PR_2", "number": 2, "title": "FEAT unmerged", "merged_at": None})
    stats = Metrics(snapshot.model_copy(update={"items": [item, opened]})).calculate()
    story = StoryBuilder(stats).build()
    slide = next(slide for slide in story.slides if slide.type == "prs")
    assert stats.counts[Activity.AUTHORED] == 2 and stats.counts[Activity.LANDED] == 1
    assert "FIX: 1" in slide.summary and "FEAT" not in slide.summary
    presentation = HtmlDeck(stats=stats, story=story)._prs(slide)
    assert all("closed" not in metric.label.lower() for metric in presentation.metrics)
    assert [(bar.label, bar.value) for bar in presentation.bars] == [("FIX", 1)]


def test_closed_unmerged_pr_is_not_a_calendar_action(*, snapshot: Snapshot, item: WorkItem) -> None:
    value = item.model_copy(
        update={
            "created_at": datetime(2025, 1, 1, tzinfo=UTC),
            "merged_at": None,
            "closed_at": datetime(2026, 1, 3, tzinfo=UTC),
        }
    )
    stats = Metrics(snapshot.model_copy(update={"items": [value], "capabilities": [Capability.CLOSURES]})).calculate()
    assert stats.counts[Activity.PR_CLOSED] == 1
    assert stats.distinct_daily_events["2026-01-03"] == 0
    assert stats.peaks["day"].count == 0


def test_maintainers_others_and_bots_reconcile(*, snapshot: Snapshot, item: WorkItem, review: Review) -> None:
    actors = [
        Actor(id="maintainer", login="RoMaNLutz", database_id=10),
        Actor(id="other", login="new-person", database_id=11),
        Actor(id="copilot", login="Copilot", type="Bot", database_id=12),
        Actor(id="claude", login="claude[bot]", type="Bot", database_id=13),
    ]
    items = [
        item.model_copy(update={"number": n, "id": f"PR_{n}", "author": actor}) for n, actor in enumerate(actors, 1)
    ]
    value = repository_snapshot(snapshot=snapshot, items=items)
    value.reviews = [review.model_copy(update={"author": actors[2], "item_number": 2})]
    stats = Metrics(value).calculate()
    rows = {row.actor.login: row for row in stats.contributions}
    assert rows["RoMaNLutz"].group == "maintainers"
    assert rows["new-person"].group == "contributors"
    assert rows["Copilot"].group == rows["claude[bot]"].group == "bots"
    assert rows["Copilot"].submitted_reviews == 1
    assert sum(row.merged_prs for row in rows.values()) == stats.counts[Activity.LANDED] == 4
    assert sum(row.opened_prs for row in rows.values()) == stats.counts[Activity.AUTHORED] == 4
    assert stats.maintainer_logins == list(ContributorCredits.MAINTAINERS)
    assert len(stats.maintainer_logins) == 13
    output = HtmlDeck(stats=stats, story=StoryBuilder(stats).build()).render()
    assert "Bots and agents" in output and "claude[bot]" in output
    assert "Everyone's contributions" in output


def test_stable_identity_across_actor_encodings(*, snapshot: Snapshot, item: WorkItem, review: Review) -> None:
    author = Actor(id="rest", database_id=22, login="romanlutz")
    value = repository_snapshot(snapshot=snapshot, items=[item.model_copy(update={"author": author})])
    value.reviews = [
        review.model_copy(update={"id": 301, "author": author.model_copy(update={"id": "graphql"})}),
        review.model_copy(update={"id": 302, "author": author.model_copy(update={"database_id": None})}),
    ]
    rows = Metrics(value).calculate().contributions
    assert len(rows) == 1
    assert rows[0].merged_prs == 1 and rows[0].submitted_reviews == 2


@pytest.mark.parametrize(
    "author", [None, Actor(id="ghost", login="ghost"), Actor(id="unknown", login="imported", type="Mannequin")]
)
def test_unavailable_authors_remain_unattributed(*, snapshot: Snapshot, item: WorkItem, author: Actor | None) -> None:
    stats = Metrics(
        repository_snapshot(snapshot=snapshot, items=[item.model_copy(update={"author": author})])
    ).calculate()
    assert stats.counts[Activity.LANDED] == 1
    assert stats.unknown_contributions["merged_prs"] == 1
    assert not stats.contributions


def test_bot_with_reviews_but_no_merged_prs_is_still_visible(
    *, snapshot: Snapshot, item: WorkItem, review: Review
) -> None:
    value = repository_snapshot(snapshot=snapshot, items=[item])
    value.reviews = [review.model_copy(update={"author": Actor(id="claude", login="claude[bot]", type="Bot")})]
    stats = Metrics(value).calculate()
    story = StoryBuilder(stats).build()
    section = HtmlDeck(stats=stats, story=story)._contributors(
        next(slide for slide in story.slides if slide.type == "contributors")
    )
    bots = next(group for group in section.credit_groups if group.group == "bots")
    assert bots.rows[0].submitted_reviews == 1 and bots.rows[0].merged_prs == 0


def test_topic_loc_reconciles_and_rename_deletions_keep_origin(*, snapshot: Snapshot, item: WorkItem) -> None:
    changes = [
        FileChange(path="pyrit/converter/x.py", previous_path="frontend/x.ts", additions=20, deletions=7, binary=False),
        FileChange(path="pyrit/datasets/seed_datasets/image.png", additions=0, deletions=0, binary=True),
    ]
    item = item.model_copy(
        update={
            "paths": [file.path for file in changes],
            "file_changes": changes,
            "changed_files": 2,
            "loc_complete": True,
        }
    )
    stats = Metrics(repository_snapshot(snapshot=snapshot, items=[item])).calculate()
    assert stats.loc.scope == LocScope.REPOSITORY_PRS
    assert stats.loc.by_topic == {
        "Converters": LineTotals(additions=20),
        "Frontend": LineTotals(deletions=7),
    }
    assert sum(value.additions for value in stats.loc.by_topic.values()) == stats.loc.totals.additions == 20
    assert sum(value.deletions for value in stats.loc.by_topic.values()) == stats.loc.totals.deletions == 7
    assert sum(stats.code_file_topics.values()) == 2


def test_incomplete_topic_loc_is_not_plausible_partial_data(*, snapshot: Snapshot, item: WorkItem) -> None:
    stats = Metrics(repository_snapshot(snapshot=snapshot, items=[item])).calculate()
    assert not stats.loc.complete and stats.loc.by_topic is None
    assert stats.code_file_topics == {}


def test_compact_topic_chart_keeps_every_line_in_the_denominator() -> None:
    values = {f"Area {number}": LineTotals(additions=number, deletions=number * 2) for number in range(1, 15)}
    bars = HtmlDeck._line_bars(values=values, limit=8)
    assert len(bars) == 9
    assert bars[-1].label == "Other areas (6)"
    assert sum(row.additions for row in bars) == sum(row.additions for row in values.values())
    assert sum(row.deletions for row in bars) == sum(row.deletions for row in values.values())


@pytest.mark.parametrize(("days", "scale", "bars"), [(50, "day", 50), (51, "month", 2)])
def test_calendar_chart_threshold_and_chronological_order(
    *, snapshot: Snapshot, item: WorkItem, days: int, scale: str, bars: int
) -> None:
    period = Period.for_year(year=2026, now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=days))
    value = snapshot.model_copy(update={"period": period, "items": [item]})
    stats = Metrics(value).calculate()
    story = StoryBuilder(stats).build()
    section = HtmlDeck(stats=stats, story=story)._busiest(
        next(slide for slide in story.slides if slide.type == "busiest")
    )
    assert section.timeline_scale == scale and len(section.timeline) == bars
    assert [bar.label for bar in section.timeline] == sorted(bar.label for bar in section.timeline)
    assert sum(bar.value for bar in section.timeline) == sum(stats.distinct_monthly_events.values())
    if scale == "day":
        assert section.timeline[0].value == 0


def test_repository_cli_has_separate_cache_and_replays(*, snapshot: Snapshot, tmp_path: Path) -> None:
    async def collect_async(**kwargs: object) -> Snapshot:
        return Snapshot.model_validate(
            {
                **snapshot.model_dump(),
                "contributor": None,
                "repository_year": True,
                "period": kwargs["period"],
                "taxonomy": kwargs["taxonomy"],
            }
        )

    with patch("build_scripts.pyrit_wrapped.cli._collect_async", new_callable=AsyncMock) as collect:
        collect.side_effect = collect_async
        for index in (1, 2):
            assert (
                main(
                    [
                        "summarize",
                        "--repository",
                        "--year",
                        "2026",
                        "--cache-dir",
                        str(tmp_path / "cache"),
                        "--output-dir",
                        str(tmp_path / f"report-{index}"),
                    ]
                )
                == 0
            )
        assert collect.call_count == 1
        assert collect.call_args.kwargs["repository_year"]
    assert (tmp_path / "cache" / "repository-2026" / "session.json").is_file()
    assert (tmp_path / "report-1" / "stats.json").read_bytes() == (tmp_path / "report-2" / "stats.json").read_bytes()


async def test_repository_collector_uses_batched_reviews(*, snapshot: Snapshot, item: WorkItem, review: Review) -> None:
    client = MagicMock(spec=GitHubClient)
    client.response_times = [snapshot.period.cutoff]
    client.get_async = AsyncMock(
        return_value=Response(
            fetched_at=snapshot.period.cutoff,
            next_page=None,
            data={"full_name": "microsoft/PyRIT", "private": False, "created_at": "2024-01-01T00:00:00Z"},
        )
    )
    client.list_async = AsyncMock(return_value=[])
    client.progress = MagicMock()
    collector = Collector(client=client, progress=lambda message: None)
    with (
        patch.object(
            collector, "_candidate_numbers_async", new_callable=AsyncMock, return_value=({1}, set(), {1}, {1})
        ),
        patch.object(collector, "_closed_numbers_async", new_callable=AsyncMock, return_value=(set(), set())),
        patch.object(collector, "_comments_async", new_callable=AsyncMock, return_value=[]),
        patch.object(collector, "_pull_async", new_callable=AsyncMock, return_value=item),
        patch.object(collector, "_load_files_async", new_callable=AsyncMock, return_value=[]),
        patch.object(ReviewReader, "read_async", new_callable=AsyncMock, return_value=[review]) as reader,
    ):
        result = await collector.collect_async(
            login=None, period=snapshot.period, taxonomy=snapshot.taxonomy, repository_year=True
        )
    assert result.repository_year and result.contributor is None
    assert Metrics(result).calculate().counts[Activity.REVIEWS] == 1
    assert reader.call_count == 1
    assert not any(call.kwargs["path"].endswith("/reviews") for call in client.list_async.call_args_list)


async def test_repository_collection_requires_one_scope(snapshot: Snapshot) -> None:
    collector = Collector(client=MagicMock(spec=GitHubClient), progress=lambda message: None)
    with pytest.raises(WrappedError, match="Choose one"):
        await collector.collect_async(
            login="owner", period=snapshot.period, taxonomy=snapshot.taxonomy, repository_year=True
        )
