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

from build_scripts.pyrit_wrapped.models import (
    Activity,
    Contribution,
    LineTotals,
    Model,
    Slide,
    Stats,
    Story,
    WrappedError,
)

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
    axis_label: str = ""


class LanguageBar(Model):
    label: str
    additions: int
    deletions: int
    added_width: float
    deleted_width: float


class CreditGroup(Model):
    label: str
    group: str
    rows: list[Contribution]


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
    credit_groups: list[CreditGroup] = Field(default_factory=list)
    timeline: list[Bar] = Field(default_factory=list)
    timeline_scale: str = ""
    topic_lines: list[LanguageBar] = Field(default_factory=list)
    roster: bool = False


class HtmlDeck:
    _WEB = Path(__file__).with_name("web")
    _MASCOT = Path(__file__).resolve().parents[2] / "doc" / "roakey.png"
    _RUNNER = Path(__file__).resolve().parents[2] / "doc" / "sprites" / "roakey-run-and-flag.png"
    _EFFECTS = {
        "cover": "confetti",
        "overview": "disco",
        "contributors": "raccoon",
        "prs": "fireworks",
        "reviews_people": "confetti",
        "issues": "disco",
        "topics": "raccoon",
        "busiest": "fireworks",
        "loc": "disco",
        "recap": "confetti",
    }
    _HEADLINES = {
        "cover": "PyRIT Wrapped",
        "contributors": "Meet the crew.",
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
        if (
            stats.period != story.period
            or stats.contributor != story.contributor
            or stats.release != story.release
            or stats.repository_year != story.repository_year
        ):
            raise WrappedError("HTML facts and story must describe the same snapshot.")
        if not 5 <= len(story.slides) <= 10:
            raise WrappedError("The HTML viewer requires a compact 5-10-slide story.")
        self.stats = stats
        self.story = story

    def render(self) -> str:
        factories: dict[str, Callable[[Slide], DeckSection]] = {
            "cover": self._cover,
            "contributors": self._contributors,
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
        title = self.stats.identity
        payload = {
            "title": title,
            "slides": [
                {
                    "type": slide.type,
                    "title": slide.title,
                    "effect": self._EFFECTS[slide.type],
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
            runner="data:image/png;base64," + base64.b64encode(self._RUNNER.read_bytes()).decode("ascii"),
            contributor_groups=self._credit_groups() if self.stats.contributor is None else [],
            unknown_merged=self.stats.unknown_contributions.get("merged_prs", 0),
            maintainer_logins=self.stats.maintainer_logins,
            unmatched_maintainers=sorted(
                name
                for name in self.stats.maintainer_logins
                if name.casefold() not in {row.actor.login.casefold() for row in self.stats.contributions}
            ),
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
            f"Between {release.base.tag} and {release.head.tag} publication dates."
            if release
            else "Your authored work during the reporting year."
            if self.stats.contributor
            else "All PyRIT contributors during the reporting year."
        )
        people = sum(row.actor.type == "User" and row.merged_prs > 0 for row in self.stats.contributions)
        metrics = [
            self._metric(label="PRs merged", role=Activity.LANDED),
            Metric(label="Human authors of merged PRs", value=f"{people:,}"),
            self._metric(label="PRs newly opened", role=Activity.AUTHORED),
        ]
        if self.stats.contributor:
            metrics[1] = self._metric(label="Submitted reviews", role=Activity.REVIEWS)
        return self._base(
            slide=slide,
            description=description,
            metrics=metrics,
            credit_groups=self._credit_groups(merged_only=True, limit=5, include_bots=False)
            if not self.stats.contributor
            else [],
            footnote="Merged and newly opened are separate cohorts. An older PR can merge in this window. "
            + (
                f"{sum(row.merged_prs for row in self.stats.contributions if row.group == 'bots'):,} "
                "merged PRs have recorded bot authors. "
                if not self.stats.contributor
                else ""
            )
            + "No unmerged closures are included.",
        )

    def _cover(self, slide: Slide) -> DeckSection:
        return self._base(
            slide=slide,
            description=self.stats.identity,
            metrics=[],
            footnote="A celebration of the work, the people, and the wonderfully busy days.",
        )

    def _credit_groups(
        self, *, merged_only: bool = False, limit: int | None = None, include_bots: bool = True
    ) -> list[CreditGroup]:
        result = []
        for group, label in (
            ("maintainers", "Maintainers"),
            ("contributors", "Other contributors"),
            ("bots", "Bots and agents"),
        ):
            if group == "bots" and not include_bots:
                continue
            rows = [
                row
                for row in self.stats.contributions
                if row.group == group and (not merged_only or row.merged_prs or group == "bots")
            ]
            result.append(CreditGroup(label=label, group=group, rows=rows if limit is None else rows[:limit]))
        return result

    def _contributors(self, slide: Slide) -> DeckSection:
        return self._base(
            slide=slide,
            description="Authors of merged PRs, with bot activity called out separately.",
            metrics=[],
            credit_groups=self._credit_groups(merged_only=True),
            roster=True,
            footnote=f"{self.stats.unknown_contributions.get('merged_prs', 0)} merged PRs "
            "have unavailable/deleted authors. "
            "The complete list below also credits reviews, comments, newly opened PRs, and issues.",
        )

    def _prs(self, slide: Slide) -> DeckSection:
        role = Activity.LANDED
        breakdown = self.stats.breakdowns.get(role)
        return self._base(
            slide=slide,
            description="What the merged PRs brought aboard.",
            metrics=[
                self._metric(label="PRs newly opened", role=Activity.AUTHORED),
                self._metric(label="PRs merged", role=Activity.LANDED),
            ],
            bars=self._bars(values=breakdown.intents if breakdown else {}),
            chart_label="Change intent of merged PRs",
            footnote="Intent uses merged PRs only. Newly opened PRs may still be open; unmerged closures are omitted.",
        )

    def _people(self, slide: Slide) -> DeckSection:
        counts = {}
        for role in ("authors", "reviewers", "mergers"):
            actors = self.stats.participants[role]
            label = "Merged-PR authors" if role == "authors" else role.title()
            counts[label] = sum(actor.type == "User" for actor in actors)
        return self._base(
            slide=slide,
            description="Reviews, comments, and merge credit make collaboration visible.",
            metrics=[
                self._metric(label="Distinct PRs reviewed", role=Activity.REVIEWED),
                self._metric(label="Submitted reviews", role=Activity.REVIEWS),
                self._metric(label="Inline review comments", role=Activity.INLINE),
            ],
            bars=self._bars(values=counts),
            chart_label="Identifiable people by role",
            footnote="Submitted reviews count each submission; distinct PRs count each reviewed PR once. "
            + (
                f"Separately, {self.stats.counts[Activity.MERGED]:,} PRs record you as merger across all authors. "
                if self.stats.contributor
                else ""
            )
            + "People and bots can appear in multiple roles.",
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
        role = Activity.LANDED
        breakdown = self.stats.breakdowns.get(role)
        values = (
            self.stats.code_file_topics if self.stats.loc.complete else breakdown.primary_topics if breakdown else {}
        )
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
            chart_label="Changed files by area" if self.stats.loc.complete else "Merged PRs by primary area",
            topic_lines=self._line_bars(values=self.stats.loc.by_topic or {}, limit=8),
            footnote=f"PRs touching artifacts (overlapping): {artifact_note or 'No activity'}. "
            + (
                "Files and LOC use the net tag diff. "
                if self.stats.release
                else "Files and LOC sum merged PR diffs; files may appear in multiple PRs. "
            )
            + (
                self.stats.loc.reason
                or "Each line is assigned once; rename deletions keep the original topic. Unknown remains visible."
            ),
        )

    def _busiest(self, slide: Slide) -> DeckSection:
        metrics = []
        for scale, peak in self.stats.peaks.items():
            names = ", ".join(peak.buckets[:2])
            suffix = f" +{len(peak.buckets) - 2} ties" if len(peak.buckets) > 2 else ""
            metrics.append(
                Metric(label=f"Peak {scale}: {names}{suffix}", value=f"{peak.count:,}", note="Distinct actions")
            )
        daily = bool(self.stats.distinct_daily_events) and len(self.stats.distinct_daily_events) <= 50
        values = self.stats.distinct_daily_events if daily else self.stats.distinct_monthly_events
        maximum = max(values.values(), default=0)
        timeline = []
        for index, (label, value) in enumerate(values.items()):
            axis = label[5:] if daily else label
            show = not daily or index % max(1, len(values) // 6) == 0 or index == len(values) - 1
            timeline.append(
                Bar(
                    label=label,
                    value=value,
                    width=100 * value / maximum if maximum else 0,
                    axis_label=axis if show else "",
                )
            )
        return self._base(
            slide=slide,
            description="The calendar moments when activity came together.",
            metrics=metrics,
            timeline=timeline,
            timeline_scale="day" if daily else "month",
            chart_label="Daily actions" if daily else "Monthly actions",
            footnote="UTC dates and ISO weeks. Daily bars for windows of 50 days or fewer; longer windows use months. "
            "The same merge counts once. Unmerged PR closures are excluded.",
        )

    @staticmethod
    def _line_bars(*, values: dict[str, LineTotals], limit: int | None = None) -> list[LanguageBar]:
        ordered = sorted(values.items(), key=lambda entry: (-(entry[1].additions + entry[1].deletions), entry[0]))
        if limit is not None and len(ordered) > limit:
            remaining = ordered[limit:]
            grouped = LineTotals(
                additions=sum(row.additions for _, row in remaining),
                deletions=sum(row.deletions for _, row in remaining),
            )
            ordered = [*ordered[:limit], (f"Other areas ({len(remaining)})", grouped)]
        maximum = max((max(row.additions, row.deletions) for _, row in ordered), default=1) or 1
        return [
            LanguageBar(
                label=name,
                additions=row.additions,
                deletions=row.deletions,
                added_width=100 * row.additions / maximum,
                deleted_width=100 * row.deletions / maximum,
            )
            for name, row in ordered
        ]

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
            description="The net tag diff." if self.stats.release else "The sum of merged PR diffs.",
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
                self._metric(label="PRs merged", role=Activity.LANDED),
                self._metric(label="Opened issues", role=Activity.ISSUES),
            ],
            footnote="The transcript, facts, and full activity lists are available below. "
            "No contributor ranking is inferred.",
        )


def script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
