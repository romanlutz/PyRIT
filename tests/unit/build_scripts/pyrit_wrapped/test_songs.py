# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import Snapshot, SongCandidate
from build_scripts.pyrit_wrapped.render import MarkdownReport
from build_scripts.pyrit_wrapped.songs import SongCatalog
from build_scripts.pyrit_wrapped.story import StoryBuilder


@pytest.mark.parametrize(
    ("slide_type", "title", "artist"),
    [
        ("overview", "Celebration", "Kool & the Gang"),
        ("prs", "Work", "Ava Max"),
        ("reviews_people", "With a Little Help from My Friends", "The Beatles"),
        ("issues", "Problem", "Ariana Grande feat. Iggy Azalea"),
        ("topics", "Purple Hat", "SOFI TUKKER"),
        ("busiest", "Don't Stop Me Now", "Queen"),
        ("loc", "Changes", "David Bowie"),
        ("recap", "Bottle Up", "Backstreet Boys"),
    ],
)
def test_confirmed_selection(*, slide_type: str, title: str, artist: str) -> None:
    selected = [candidate for candidate in SongCatalog.candidates(slide_type) if candidate.selected]
    assert len(selected) == 1
    assert (selected[0].title, selected[0].artist) == (title, artist)


def test_new_candidates_are_not_implicitly_selected() -> None:
    candidate = SongCandidate(title="Example", artist="Example artist", rationale="Future suggestion.")
    assert not candidate.selected


def test_purple_hat_is_selected_for_topic_focus() -> None:
    candidate = SongCatalog.candidates("topics")[0]
    assert candidate.title == "Purple Hat"
    assert candidate.artist == "SOFI TUKKER"
    assert candidate.selected


def test_song_report_separates_selection_from_supply(snapshot: Snapshot) -> None:
    stats = Metrics(snapshot).calculate()
    story = StoryBuilder(stats).build()
    markdown = MarkdownReport(stats=stats, story=story).render_songs()
    assert "| Selected |" in markdown
    assert "| Suggested |" not in markdown
    assert "does not mean a recording is supplied or licensed" in markdown
    assert all(slide.cue_id is None for slide in story.slides)
