# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from build_scripts.pyrit_wrapped.models import Activity, OmittedSlide, Slide, Stats, Story
from build_scripts.pyrit_wrapped.songs import SongCatalog


class StoryBuilder:
    _ORDER = (
        "cover",
        "overview",
        "contributors",
        "prs",
        "reviews_people",
        "issues",
        "topics",
        "busiest",
        "loc",
        "recap",
    )

    def __init__(self, stats: Stats) -> None:
        self.stats = stats
        self.slides: dict[str, Slide] = {}
        self.omitted: dict[str, OmittedSlide] = {}

    def build(self) -> Story:
        self._add(key="cover", title="PyRIT Wrapped", summary=self.stats.identity, roles=[])
        self._overview()
        self._contributors()
        self._prs()
        self._reviews_people()
        self._issues()
        self._topics()
        self._busiest()
        self._loc()
        self._add(
            key="recap",
            title="The recap",
            summary=self._count_summary(),
            roles=[Activity.AUTHORED, Activity.LANDED, Activity.ISSUES, Activity.REVIEWS],
        )
        return Story(
            contributor=self.stats.contributor,
            release=self.stats.release,
            period=self.stats.period,
            repository_year=self.stats.repository_year,
            slides=[self.slides[key] for key in self._ORDER if key in self.slides],
            omitted=[self.omitted[key] for key in self._ORDER if key in self.omitted],
        )

    def _overview(self) -> None:
        people = [row for row in self.stats.contributions if row.actor.type == "User" and row.merged_prs]
        summary = self._count_summary()
        if self.stats.contributor is None:
            summary += f" {len(people)} human authors of merged PRs."
        if self.stats.release:
            summary += f" Collaboration window: {self.stats.release.label}, between publication dates."
        self._add(
            key="overview", title="The merged-PR haul", summary=summary, roles=[Activity.AUTHORED, Activity.LANDED]
        )

    def _contributors(self) -> None:
        if self.stats.contributor is not None:
            self.omitted["contributors"] = OmittedSlide(
                type="contributors", reason="People credit is a repository-wide slide."
            )
            return
        summaries = []
        for group, label in (("maintainers", "Maintainers"), ("contributors", "Other contributors"), ("bots", "Bots")):
            rows = [row for row in self.stats.contributions if row.group == group]
            summaries.append(
                label
                + ": "
                + (
                    "; ".join(
                        f"@{row.actor.login}: {row.merged_prs} merged PRs, {row.submitted_reviews} submitted reviews"
                        for row in rows
                    )
                    or "no recorded activity"
                )
            )
        self._add(
            key="contributors",
            title="The people and bots behind it",
            summary=". ".join(summaries) + ". Counts use recorded authors and actors, not inferred AI assistance.",
            roles=[Activity.LANDED, Activity.REVIEWS],
        )

    def _prs(self) -> None:
        summary = (
            f"{self._value(Activity.AUTHORED)} PRs newly opened; {self._value(Activity.LANDED)} PRs merged "
            "during the reporting window. "
        )
        if self.stats.contributor is not None:
            summary += "These are your authored PRs, not merge credit for other authors' work. "
        role = Activity.LANDED
        breakdown = self.stats.breakdowns.get(role)
        if breakdown:
            summary += "Change intent of merged PRs: " + self._distribution(breakdown.intents) + "."
        self._add(
            key="prs",
            title="The PR pipeline",
            summary=summary,
            roles=[Activity.AUTHORED, Activity.LANDED],
        )

    def _reviews_people(self) -> None:
        roles = [
            Activity.REVIEWED,
            Activity.REVIEWS,
            Activity.MERGED,
            Activity.INLINE,
            Activity.REVIEW_BODIES,
            Activity.PR_COMMENTS,
        ]
        if not any(self.stats.activities[role] for role in roles):
            self.omitted["reviews_people"] = OmittedSlide(
                type="reviews_people", reason="No review/merge/comment activity."
            )
            return
        summary = (
            f"{self._value(Activity.REVIEWED)} distinct PRs reviewed "
            f"in {self._value(Activity.REVIEWS)} submitted reviews; "
        )
        if self.stats.contributor is not None:
            summary += f"{self._value(Activity.MERGED)} PRs credited to you as the merger, across all authors. "
        summary += (
            f"{self._value(Activity.INLINE)} inline comments, {self._value(Activity.REVIEW_BODIES)} review summaries, "
            f"{self._value(Activity.PR_COMMENTS)} PR discussion comments. "
        )
        if self.stats.contributor is not None:
            summary += f"Work reviewed from {len(self.stats.reviewed_authors)} other identifiable human authors. "
        else:
            for role in ("authors", "reviewers", "mergers"):
                actors = self.stats.participants[role]
                people = sum(actor.type == "User" for actor in actors)
                bots = sum(actor.type == "Bot" for actor in actors)
                summary += f"{role.replace('_', ' ')}: {people} people, {bots} bots; "
            summary = summary.rstrip("; ") + ". Roles overlap."
        self._add(key="reviews_people", title="Reviews, merges, and people", summary=summary, roles=roles)

    def _issues(self) -> None:
        summary = (
            f"{self._value(Activity.ISSUES)} issues opened; {self._value(Activity.ISSUES_CLOSED)} closed; "
            f"{self._value(Activity.ISSUE_COMMENTS)} issue discussion comments. "
            "Closure uses the last recorded closure timestamp, not a reconstructed history of reopenings."
        )
        self._add(
            key="issues",
            title="Questions and resolutions",
            summary=summary,
            roles=[Activity.ISSUES, Activity.ISSUES_CLOSED, Activity.ISSUE_COMMENTS],
        )

    def _topics(self) -> None:
        role = Activity.LANDED
        breakdown = self.stats.breakdowns.get(role)
        if self.stats.release is not None:
            summary = "Top changed-file areas between tags: " + self._top(self.stats.release_file_topics) + ". "
        elif breakdown:
            total = breakdown.count
            leaders = [
                name
                for name, count in breakdown.surfaces.items()
                if name not in {"Mixed", "Unknown"} and total >= 5 and count > total / 2
            ]
            prefix = f"Primarily {leaders[0]}. " if leaders else ""
            summary = prefix + "Areas: " + self._top(breakdown.primary_topics) + ". "
            summary += "Surfaces: " + self._top(breakdown.surfaces) + ". "
        else:
            summary = "No source-supported topic distribution. "
        if breakdown:
            summary += "Primary artifacts: " + self._top(breakdown.primary_artifacts) + ". "
            tags: dict[str, int] = {}
            for record in self.stats.activities[role]:
                for artifact in self.stats.classifications[record.item_ref].artifacts:
                    tags[artifact] = tags.get(artifact, 0) + 1
            summary += (
                "PRs touching artifacts (overlapping): " + self._top(tags) + ". Full breakdowns are in the evidence."
            )
        if self.stats.loc.by_topic:
            lines = sorted(
                self.stats.loc.by_topic.items(), key=lambda row: (-(row[1].additions + row[1].deletions), row[0])
            )
            summary += (
                " LOC by area: "
                + "; ".join(f"{name}: +{value.additions:,} / -{value.deletions:,}" for name, value in lines)
                + "."
            )
        self._add(key="topics", title="Where the work focused", summary=summary, roles=[role])

    def _busiest(self) -> None:
        if not any(peak.count for peak in self.stats.peaks.values()):
            self.omitted["busiest"] = OmittedSlide(type="busiest", reason="No recorded actions in the window.")
            return
        descriptions = [
            f"{scale}: {', '.join(peak.buckets[:3])}"
            + (f" (+{len(peak.buckets) - 3} tied buckets)" if len(peak.buckets) > 3 else "")
            + f" ({peak.count} distinct actions)"
            for scale, peak in self.stats.peaks.items()
        ]
        self._add(
            key="busiest",
            title="The activity peaks",
            summary="; ".join(descriptions) + ". UTC calendar months/days and ISO weeks; ties retained.",
            roles=[Activity.AUTHORED, Activity.MERGED, Activity.REVIEWS],
        )

    def _loc(self) -> None:
        loc = self.stats.loc
        if not loc.complete or loc.totals is None or loc.by_language is None:
            summary = loc.reason or "Complete LOC data is unavailable; recollect this snapshot."
            facts = {}
        else:
            label = (
                "Net tag-to-tag tree diff"
                if self.stats.release
                else ("Sum of all merged PR diffs" if self.stats.repository_year else "Sum of your merged PR diffs")
            )
            summary = (
                f"{label}: +{loc.totals.additions:,} / -{loc.totals.deletions:,} text lines. "
                + "; ".join(
                    f"{name}: +{loc.by_language[name].additions:,} / -{loc.by_language[name].deletions:,}"
                    for name in ("TypeScript", "Python", "YAML")
                )
                + ". Other languages remain in the totals and evidence. LOC is not a productivity score."
            )
            facts = {"additions": loc.totals.additions, "deletions": loc.totals.deletions, "files": loc.file_count}
        self._add(
            key="loc",
            title="The code that changed",
            summary=summary,
            roles=[Activity.SHIPPED if self.stats.release else Activity.LANDED],
            facts=facts,
        )

    def _count_summary(self) -> str:
        return (
            f"{self._value(Activity.AUTHORED)} PRs newly opened, "
            f"{self._value(Activity.LANDED)} PRs merged; {self._value(Activity.REVIEWS)} submitted reviews; "
            f"{self._value(Activity.ISSUES)} issues opened, {self._value(Activity.ISSUES_CLOSED)} closed."
        )

    def _value(self, role: Activity) -> str:
        value = self.stats.counts[role]
        return f"{value:,}" if value is not None else "unavailable"

    def _add(
        self, *, key: str, title: str, summary: str, roles: list[Activity], facts: dict[str, int] | None = None
    ) -> None:
        known = {role.value: value for role in roles if (value := self.stats.counts[role]) is not None}
        self.slides[key] = Slide(
            type=key,
            title=title,
            summary=summary,
            facts={**known, **(facts or {})},
            evidence_refs=sorted({record.ref for role in roles for record in self.stats.activities[role]}),
            song_candidates=SongCatalog.candidates(key),
        )

    @staticmethod
    def _distribution(counts: dict[str, int]) -> str:
        return (
            "; ".join(f"{key}: {value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            or "no activity"
        )

    @staticmethod
    def _top(counts: dict[str, int]) -> str:
        return StoryBuilder._distribution(dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]))
