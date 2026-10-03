# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from build_scripts.pyrit_wrapped.models import SongCandidate


class SongCatalog:
    _TRACKS = {
        "overview": [
            ("The Final Countdown", "Europe", "A recognizable opening for a release or year reveal."),
            ("Celebration", "Kool & the Gang", "A warmer opening focused on the collective milestone."),
        ],
        "prs": [
            ("Workin' for the Weekend", "Loverboy", "Playful momentum for the opened/closed/landed pipeline."),
            ("Harder, Better, Faster, Stronger", "Daft Punk", "Fits iterative work and repeated improvements."),
        ],
        "reviews_people": [
            ("With a Little Help from My Friends", "The Beatles", "Reviews and merges are collaborative work."),
            ("Lean on Me", "Bill Withers", "A supportive alternative for the people behind the release."),
        ],
        "issues": [
            ("Fix You", "Coldplay", "A literal match for reported problems and fixes."),
            ("Under Pressure", "Queen & David Bowie", "A more energetic take on the issue backlog."),
        ],
        "topics": [
            ("Around the World", "Daft Punk", "A tour through the areas touched."),
            ("Come Together", "The Beatles", "Works for multiple components converging in one release."),
        ],
        "busiest": [
            ("Don't Stop Me Now", "Queen", "A burst of energy for the peak month, week, and day."),
            ("Pump Up the Jam", "Technotronic", "A rhythmic alternative for the activity peaks."),
        ],
        "loc": [
            ("Changes", "David Bowie", "Fits additions, deletions, and language changes without implying quality."),
            ("Technologic", "Daft Punk", "A technical, playful cue for the code and language breakdown."),
        ],
        "recap": [
            ("We Are the Champions", "Queen", "A recognizable closing cue, not a contributor ranking."),
            ("Celebration", "Kool & the Gang", "A collective thank-you rather than a competitive finale."),
        ],
    }

    @classmethod
    def candidates(cls, slide_type: str) -> list[SongCandidate]:
        return [
            SongCandidate(title=title, artist=artist, rationale=reason)
            for title, artist, reason in cls._TRACKS[slide_type]
        ]
