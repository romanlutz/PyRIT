# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from build_scripts.pyrit_wrapped.cli import main
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import Snapshot, WorkItem, WrappedError
from build_scripts.pyrit_wrapped.render import MarkdownReport, write_reports
from build_scripts.pyrit_wrapped.story import StoryBuilder


def test_empty_story_has_no_fake_achievements(snapshot: Snapshot) -> None:
    story = StoryBuilder(Metrics(snapshot).calculate()).build()
    assert [slide.type for slide in story.slides] == ["overview", "prs", "issues", "topics", "loc", "recap"]
    assert len(story.slides) + len(story.omitted) == 8
    assert 5 <= len(story.slides) <= 10
    assert all(slide.cue_id is None and slide.duration_ms is None for slide in story.slides)


def test_fix_and_artifact_stories_use_actual_evidence(*, snapshot: Snapshot, item: WorkItem) -> None:
    item = item.model_copy(update={"paths": ["tests/unit/converter/test_example.py"]})
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    story = StoryBuilder(stats).build()
    by_type = {slide.type: slide for slide in story.slides}
    assert by_type["prs"].facts["authored_prs"] == 1
    assert "FIX: 1" in by_type["prs"].summary
    assert by_type["topics"].evidence_refs == ["pr:1"]
    assert "Tests: 1" in by_type["topics"].summary
    assert "Primarily" not in by_type["topics"].summary


def test_dominance_requires_confirmed_majority(*, snapshot: Snapshot, item: WorkItem) -> None:
    items = [item.model_copy(update={"id": f"PR_{number}", "number": number}) for number in range(1, 6)]
    story = StoryBuilder(Metrics(snapshot.model_copy(update={"items": items})).calculate()).build()
    assert next(slide for slide in story.slides if slide.type == "topics").summary.startswith(
        "Primarily Python framework"
    )
    assert "FIX: 5" in next(slide for slide in story.slides if slide.type == "prs").summary
    inferred = [value.model_copy(update={"files_complete": False}) for value in items]
    story = StoryBuilder(Metrics(snapshot.model_copy(update={"items": inferred})).calculate()).build()
    assert not next(slide for slide in story.slides if slide.type == "topics").summary.startswith("Primarily")


def test_export_round_trip_and_complete_lists(*, snapshot: Snapshot, item: WorkItem, tmp_path: Path) -> None:
    snapshot = snapshot.model_copy(update={"items": [item]})
    stats = Metrics(snapshot).calculate()
    story = StoryBuilder(stats).build()
    destination = write_reports(snapshot=snapshot, stats=stats, story=story, output_dir=tmp_path / "report")
    assert Snapshot.model_validate_json((destination / "snapshot.json").read_text(encoding="utf-8")) == snapshot
    summary = (destination / "summary.md").read_text(encoding="utf-8")
    activity = (destination / "activity.md").read_text(encoding="utf-8")
    assert "FIX converter behavior" in activity
    assert "Converters (1)" in activity
    assert "[activity.md](activity.md)" in summary
    assert "recording supply and rights remain separate" in summary
    assert "Daft Punk" in (destination / "songs.md").read_text(encoding="utf-8")
    with pytest.raises(WrappedError, match="already exists"):
        write_reports(snapshot=snapshot, stats=stats, story=story, output_dir=destination)


def test_markdown_escapes_untrusted_titles(*, snapshot: Snapshot, item: WorkItem) -> None:
    item = item.model_copy(update={"title": "<script>alert(1)</script> [link] | surprise"})
    stats = Metrics(snapshot.model_copy(update={"items": [item]})).calculate()
    result = MarkdownReport(stats=stats, story=StoryBuilder(stats).build()).render_activity()
    assert "<script>" not in result
    assert "\\[link\\]" in result
    assert "\\|" in result


def test_offline_cli_replays_identically(*, snapshot: Snapshot, tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(snapshot.model_dump_json(), encoding="utf-8")
    for index in (1, 2):
        assert main(["summarize", "--snapshot", str(source), "--output-dir", str(tmp_path / f"report{index}")]) == 0
    for name in ("stats.json", "story.json", "summary.md", "activity.md", "songs.md"):
        assert (tmp_path / "report1" / name).read_bytes() == (tmp_path / "report2" / name).read_bytes()


def test_offline_cli_rejects_live_options(*, snapshot: Snapshot, tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(snapshot.model_dump_json(), encoding="utf-8")
    assert main(["summarize", "--snapshot", str(source), "--refresh"]) == 1


def test_dataset_story_separates_code_and_payloads(*, snapshot: Snapshot, item: WorkItem) -> None:
    provider = item.model_copy(update={"paths": ["pyrit/datasets/seed_datasets/provider.py"]})
    stats = Metrics(snapshot.model_copy(update={"items": [provider]})).calculate()
    story = StoryBuilder(stats).build()
    slide = next(slide for slide in story.slides if slide.type == "topics")
    assert "Datasets: 1" in slide.summary
    assert "Product code: 1" in slide.summary
    assert stats.classifications[provider.ref].primary_artifact == "Product code"


def test_live_cli_reuses_a_complete_snapshot_instead_of_mixing_windows(*, snapshot: Snapshot, tmp_path: Path) -> None:
    async def collect_async(**kwargs: object) -> Snapshot:
        return snapshot.model_copy(update={"period": kwargs["period"], "taxonomy": kwargs["taxonomy"]})

    with patch("build_scripts.pyrit_wrapped.cli._collect_async", new_callable=AsyncMock) as collect:
        collect.side_effect = collect_async
        for index in (1, 2):
            assert (
                main(
                    [
                        "summarize",
                        "--contributor",
                        "owner",
                        "--year",
                        "2026",
                        "--cache-dir",
                        str(tmp_path / "cache"),
                        "--output-dir",
                        str(tmp_path / f"report{index}"),
                    ]
                )
                == 0
            )
        assert collect.call_count == 1
    assert (tmp_path / "report1" / "stats.json").read_bytes() == (tmp_path / "report2" / "stats.json").read_bytes()
