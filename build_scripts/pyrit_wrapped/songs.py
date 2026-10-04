# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from build_scripts.pyrit_wrapped.models import SongCandidate


class SongCatalog:
    _VIDEOS = {
        "overview": "TBS6gAtj8gE",
        "prs": "cJRw7rYOOzA",
        "reviews_people": "0C58ttB2-Qg",
        "issues": "iS1g8G_njx8",
        "topics": "hOEMsqWx4nE",
        "busiest": "HgzGwKwLmgM",
        "loc": "4BgF7Y3q-as",
        "recap": "Tdxn1wc3As0",
    }
    _SELECTIONS = {
        "overview": ("Celebration", "Kool & the Gang"),
        "prs": ("Work", "Ava Max"),
        "reviews_people": ("With a Little Help from My Friends", "The Beatles"),
        "issues": ("Problem", "Ariana Grande feat. Iggy Azalea"),
        "topics": ("Purple Hat", "SOFI TUKKER"),
        "busiest": ("Don't Stop Me Now", "Queen"),
        "loc": ("Changes", "David Bowie"),
        "recap": ("Bottle Up", "Backstreet Boys"),
    }
    _TRACKS = {
        "overview": [
            ("Celebration", "Kool & the Gang", "Selected for the opening and collective milestone."),
        ],
        "prs": [
            ("Work", "Ava Max", "Selected for newly opened PRs and merged work."),
        ],
        "reviews_people": [
            ("With a Little Help from My Friends", "The Beatles", "Selected for the people helping work land."),
        ],
        "issues": [
            (
                "Problem",
                "Ariana Grande feat. Iggy Azalea",
                "Selected for questions, reported problems, and resolutions.",
            ),
        ],
        "topics": [
            (
                "Purple Hat",
                "SOFI TUKKER",
                "Selected for the colorful montage of topic areas and work focus.",
            ),
        ],
        "busiest": [
            ("Don't Stop Me Now", "Queen", "Selected for the peak month, week, and day."),
        ],
        "loc": [
            ("Changes", "David Bowie", "Selected for additions, deletions, and language changes."),
        ],
        "recap": [
            ("Bottle Up", "Backstreet Boys", "Selected for preserving the milestone feeling in the recap."),
        ],
    }

    @classmethod
    def candidates(cls, slide_type: str) -> list[SongCandidate]:
        if slide_type in {"cover", "contributors"}:
            return []
        return [
            SongCandidate(
                title=title,
                artist=artist,
                rationale=reason,
                selected=cls._SELECTIONS.get(slide_type) == (title, artist),
                youtube_id=cls._VIDEOS.get(slide_type),
            )
            for title, artist, reason in cls._TRACKS[slide_type]
        ]
