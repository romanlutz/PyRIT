# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import base64
import json
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from pydantic import Field

from build_scripts.pyrit_wrapped.models import Activity, Model, Slide, Stats, Story, WrappedError

if TYPE_CHECKING:
    from collections.abc import Callable


class Metric(Model):
    label: str
    value: str
    note: str = ""


class Bar(Model):
    label: str
    value: int
    width: float


class LanguageBar(Model):
    label: str
    additions: int
    deletions: int
    added_width: float
    deleted_width: float


class DeckSection(Model):
    type: str
    title: str
    description: str
    metrics: list[Metric]
    bars: list[Bar] = Field(default_factory=list)
    languages: list[LanguageBar] = Field(default_factory=list)
    chart_label: str = ""
    footnote: str = ""
    full_summary: str


class HtmlDeck:
    _WEB = Path(__file__).with_name("web")
    _MASCOT = Path(__file__).resolve().parents[2] / "doc" / "roakey.png"
    _HEADLINES = {
        "overview": "Look at that haul!",
        "prs": "All hands on deck.",
        "reviews_people": "One mighty crew.",
        "issues": "Questions aboard!",
        "topics": "X marks your focus.",
        "busiest": "Full sail. No brakes.",
        "loc": "Making waves in the code.",
        "recap": "That's a wrap, crew!",
    }

    def __init__(self, *, stats: Stats, story: Story) -> None:
        if stats.period != story.period or stats.contributor != story.contributor or stats.release != story.release:
            raise WrappedError("HTML facts and story must describe the same snapshot.")
        if not 5 <= len(story.slides) <= 10:
            raise WrappedError("The HTML viewer requires a compact 5-10-slide story.")
        self.stats = stats
        self.story = story

    def render(self) -> str:
        factories: dict[str, Callable[[Slide], DeckSection]] = {
            "overview": self._overview,
            "prs": self._prs,
            "reviews_people": self._people,
            "issues": self._issues,
            "topics": self._topics,
            "busiest": self._busiest,
            "loc": self._loc,
            "recap": self._recap,
        }
        if any(slide.type not in factories for slide in self.story.slides):
            raise WrappedError("Story contains a slide type unsupported by the HTML viewer.")
        sections = [factories[slide.type](slide) for slide in self.story.slides]
        title = (
            f"PyRIT release {self.stats.release.head.tag}"
            if self.stats.release
            else f"@{self.stats.contributor.login} / {self.stats.period.year}"
            if self.stats.contributor
            else "PyRIT Wrapped"
        )
        payload = {
            "title": title,
            "slides": [
                {
                    "type": slide.type,
                    "title": slide.title,
                    "track": next(
                        (
                            {"title": candidate.title, "artist": candidate.artist}
                            for candidate in slide.song_candidates
                            if candidate.selected
                        ),
                        None,
                    ),
                }
                for slide in self.story.slides
            ],
        }
        environment = Environment(loader=FileSystemLoader(self._WEB), autoescape=True)
        return environment.get_template("deck.html").render(
            title=title,
            sections=sections,
            css=(self._WEB / "deck.css").read_text(encoding="utf-8"),
            controller=(self._WEB / "recording.js").read_text(encoding="utf-8"),
            script=(self._WEB / "deck.js").read_text(encoding="utf-8"),
            payload=script_json(payload),
            tracks=[
                next((song for song in slide.song_candidates if song.selected), None) for slide in self.story.slides
            ],
            mascot="data:image/png;base64," + base64.b64encode(self._MASCOT.read_bytes()).decode("ascii"),
            period=self.stats.period,
            warnings=self.stats.warnings,
            source_time=self.stats.earliest_response_at.astimezone(UTC).isoformat(),
            collected=self.stats.collected_at.astimezone(UTC).isoformat(),
        )

    def _metric(self, *, label: str, role: Activity, note: str = "") -> Metric:
        count = self.stats.counts[role]
        return Metric(label=label, value=f"{count:,}" if count is not None else "Not collected", note=note)

    def _base(self, *, slide: Slide, description: str, metrics: list[Metric], **extras: object) -> DeckSection:
        return DeckSection.model_validate(
            {
                "type": slide.type,
                "title": self._HEADLINES[slide.type],
                "description": description,
                "metrics": metrics,
                "full_summary": slide.summary,
                **extras,
            }
        )

    @staticmethod
    def _bars(*, values: dict[str, int], limit: int = 6) -> list[Bar]:
        ordered = sorted(values.items(), key=lambda entry: (-entry[1], entry[0]))[:limit]
        maximum = max((value for _, value in ordered), default=0)
        return [
            Bar(label=label, value=value, width=100 * value / maximum if maximum else 0) for label, value in ordered
        ]

    def _overview(self, slide: Slide) -> DeckSection:
        release = self.stats.release
        description = (
            f"{release.base.tag} to {release.head.tag}: shipped code and the collaboration behind it."
            if release
            else "Your public PyRIT activity, one chapter at a time."
        )
        people = sum(actor.type == "User" for actor in self.stats.participants["authors"])
        return self._base(
            slide=slide,
            description=description,
            metrics=[
                self._metric(
                    label="PRs in the tag range" if release else "Your PRs landed",
                    role=Activity.SHIPPED if release else Activity.LANDED,
                ),
                Metric(label="Human PR authors represented", value=f"{people:,}"),
                self._metric(label="PRs reviewed", role=Activity.REVIEWED),
            ],
            footnote="Shipping uses merge-commit membership; collaboration uses the reporting window. Roles overlap.",
        )

    def _prs(self, slide: Slide) -> DeckSection:
        role = Activity.SHIPPED if self.stats.release else Activity.AUTHORED
        breakdown = self.stats.breakdowns.get(role)
        return self._base(
            slide=slide,
            description="Opening work and landing it are different parts of the story.",
            metrics=[
                self._metric(label="Opened", role=Activity.AUTHORED),
                self._metric(label="Closed", role=Activity.PR_CLOSED, note="Includes merges"),
                self._metric(label="Landed in the window", role=Activity.LANDED),
            ],
            bars=self._bars(values=breakdown.intents if breakdown else {}),
            chart_label="Change intent",
            footnote="Contributor mode counts your own opened/closed PRs. Release mode includes every author.",
        )

    def _people(self, slide: Slide) -> DeckSection:
        counts = {}
        for role in ("authors", "reviewers", "mergers"):
            actors = self.stats.participants[role]
            counts[role.title()] = sum(actor.type == "User" for actor in actors)
        return self._base(
            slide=slide,
            description="Reviews, comments, and merge credit make collaboration visible.",
            metrics=[
                self._metric(label="Distinct PRs reviewed", role=Activity.REVIEWED),
                self._metric(label="Submitted reviews", role=Activity.REVIEWS),
                self._metric(label="PRs merged", role=Activity.MERGED),
            ],
            bars=self._bars(values=counts),
            chart_label="Identifiable people by role",
            footnote="People can appear in multiple roles. "
            "Bots and unavailable identities stay separate in the evidence.",
        )

    def _issues(self, slide: Slide) -> DeckSection:
        groups = self.stats.breakdowns[Activity.ISSUES].topics
        return self._base(
            slide=slide,
            description="The questions raised, the threads continued, and the issues closed.",
            metrics=[
                self._metric(label="Opened", role=Activity.ISSUES),
                self._metric(label="Closed", role=Activity.ISSUES_CLOSED),
                self._metric(label="Discussion comments", role=Activity.ISSUE_COMMENTS),
            ],
            bars=self._bars(values={topic: len(refs) for topic, refs in groups.items()}),
            chart_label="Issue topics",
            footnote="Topics may overlap and can be inferred. Closure is the last recorded timestamp.",
        )

    def _topics(self, slide: Slide) -> DeckSection:
        role = Activity.SHIPPED if self.stats.release else Activity.AUTHORED
        breakdown = self.stats.breakdowns.get(role)
        values = self.stats.release_file_topics if self.stats.release else breakdown.primary_topics if breakdown else {}
        description = "A tour of the areas that carried the work."
        if breakdown:
            surfaces = sorted(breakdown.surfaces.items(), key=lambda entry: (-entry[1], entry[0]))[:3]
            description = "Primary PR surfaces: " + "; ".join(f"{name}: {count}" for name, count in surfaces) + "."
        touched: dict[str, int] = {}
        for record in self.stats.activities[role]:
            for artifact in self.stats.classifications[record.item_ref].artifacts:
                touched[artifact] = touched.get(artifact, 0) + 1
        artifact_note = "; ".join(
            f"{name}: {count}" for name, count in sorted(touched.items(), key=lambda entry: (-entry[1], entry[0]))[:3]
        )
        return self._base(
            slide=slide,
            description=description,
            metrics=[],
            bars=self._bars(values=values),
            chart_label="Changed files by area" if self.stats.release else "Opened PRs by primary area",
            footnote=f"PRs touching artifacts (overlapping): {artifact_note or 'No activity'}. "
            "Mixed and Unknown remain in the full evidence.",
        )

    def _busiest(self, slide: Slide) -> DeckSection:
        metrics = []
        for scale, peak in self.stats.peaks.items():
            names = ", ".join(peak.buckets[:2])
            suffix = f" +{len(peak.buckets) - 2} ties" if len(peak.buckets) > 2 else ""
            metrics.append(
                Metric(label=f"Peak {scale}: {names}{suffix}", value=f"{peak.count:,}", note="Distinct actions")
            )
        return self._base(
            slide=slide,
            description="The calendar moments when activity came together.",
            metrics=metrics,
            bars=self._bars(values=self.stats.distinct_monthly_events, limit=12),
            chart_label="Monthly actions",
            footnote="UTC dates and ISO weeks. The same merge is not counted again as its closure.",
        )

    def _loc(self, slide: Slide) -> DeckSection:
        loc = self.stats.loc
        if not loc.complete or loc.totals is None or loc.by_language is None:
            return self._base(
                slide=slide,
                description=loc.reason or "Complete LOC was not collected.",
                metrics=[Metric(label="Line totals", value="Not collected")],
            )
        maximum = max((max(row.additions, row.deletions) for row in loc.by_language.values()), default=1) or 1
        languages = [
            LanguageBar(
                label=name,
                additions=row.additions,
                deletions=row.deletions,
                added_width=100 * row.additions / maximum,
                deleted_width=100 * row.deletions / maximum,
            )
            for name, row in loc.by_language.items()
        ]
        return self._base(
            slide=slide,
            description="The net tag diff." if self.stats.release else "The sum of your landed PR diffs.",
            metrics=[
                Metric(label="Lines added", value=f"+{loc.totals.additions:,}"),
                Metric(label="Lines removed", value=f"-{loc.totals.deletions:,}"),
            ],
            languages=languages,
            chart_label="Text-line changes by language",
            footnote="Renames preserve their origin/destination language. Binary files have no text LOC. "
            "This is not a productivity score.",
        )

    def _recap(self, slide: Slide) -> DeckSection:
        return self._base(
            slide=slide,
            description="Keep the evidence. Celebrate the people. Revisit the work.",
            metrics=[
                self._metric(label="Opened PRs", role=Activity.AUTHORED),
                self._metric(label="Merged PRs", role=Activity.MERGED),
                self._metric(label="Opened issues", role=Activity.ISSUES),
            ],
            footnote="The transcript, facts, and full activity lists are available below. "
            "No contributor ranking is inferred.",
        )


def script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
