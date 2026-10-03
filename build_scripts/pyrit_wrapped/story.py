# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import Counter

from build_scripts.pyrit_wrapped.models import Activity, Evidence, OmittedSlide, Slide, Stats, Story


class StoryBuilder:
    _ORDER = (
        "welcome",
        "snapshot",
        "authored",
        "landed",
        "merged",
        "issues",
        "reviews",
        "comments",
        "collaborators",
        "timeline",
        "busiest_month",
        "home_territory",
        "surfaces",
        "artifacts",
        "intent_mix",
        "fixes",
        "features",
        "tests",
        "documentation",
        "datasets",
        "infrastructure",
        "review_topics",
        "issue_topics",
        "highlights",
        "recap",
    )

    def __init__(self, stats: Stats) -> None:
        self.stats = stats
        self.slides: dict[str, Slide] = {}
        self.omitted: dict[str, OmittedSlide] = {}

    def build(self) -> Story:
        self._identity_slides()
        self._activity_slides()
        self._timeline_slides()
        self._focus_slides()
        self._chapter_slides()
        self._closing_slides()
        if set(self.slides) | set(self.omitted) != set(self._ORDER):
            raise ValueError("Every catalog slide must be included or have an omission reason.")
        return Story(
            contributor=self.stats.contributor,
            period=self.stats.period,
            slides=[self.slides[key] for key in self._ORDER if key in self.slides],
            omitted=[self.omitted[key] for key in self._ORDER if key in self.omitted],
        )

    def _identity_slides(self) -> None:
        year = self.stats.period.year
        suffix = " (year to date)" if self.stats.period.year_to_date else ""
        self._add(
            key="welcome",
            title=f"@{self.stats.contributor.login}'s PyRIT Wrapped",
            summary=f"{year}{suffix}. Public GitHub records, with separate credit for each role.",
            facts={},
            records=[],
            required=True,
        )
        self._add(
            key="snapshot",
            title="Your contribution snapshot",
            summary=self._count_summary(),
            facts={role.value: count for role, count in self.stats.counts.items()},
            records=self._all_records(),
            required=True,
        )

    def _activity_slides(self) -> None:
        definitions = [
            ("authored", Activity.AUTHORED, "What you authored", "You opened {count} PRs in the selected period."),
            (
                "landed",
                Activity.LANDED,
                "Your work landed",
                "{count} PRs you authored were merged, including older PRs.",
            ),
            ("merged", Activity.MERGED, "Work you helped land", "GitHub records you as the merger of {count} PRs."),
            ("issues", Activity.ISSUES, "You raised the questions", "You opened {count} issues, separate from PRs."),
            ("reviews", Activity.REVIEWED, "Your review footprint", "You formally reviewed {count} distinct PRs."),
        ]
        for key, role, title, template in definitions:
            count = self.stats.counts[role]
            facts = {role.value: count}
            summary = template.format(count=count)
            if role == Activity.MERGED:
                facts.update(own_prs=self.stats.own_prs_merged, other_prs=self.stats.other_prs_merged)
                summary += (
                    f" {self.stats.own_prs_merged} were your own; {self.stats.other_prs_merged} were others' work."
                )
                if self.stats.unknown_authors_merged:
                    summary += f" {self.stats.unknown_authors_merged} have an unavailable author."
            if role == Activity.REVIEWED:
                facts[Activity.REVIEWS.value] = self.stats.counts[Activity.REVIEWS]
                summary += f" You submitted {self.stats.counts[Activity.REVIEWS]} reviews, including repeat reviews."
            self._add(key=key, title=title, summary=summary, facts=facts, records=self.stats.activities[role])
        self._comment_and_collaborator_slides()

    def _comment_and_collaborator_slides(self) -> None:
        roles = [Activity.INLINE, Activity.REVIEW_BODIES, Activity.PR_COMMENTS, Activity.ISSUE_COMMENTS]
        counts = self.stats.counts
        self._add(
            key="comments",
            title="You joined the conversation",
            summary=(
                f"{counts[Activity.INLINE]} inline comments, {counts[Activity.REVIEW_BODIES]} review summaries, "
                f"{counts[Activity.PR_COMMENTS]} PR discussion comments, "
                f"and {counts[Activity.ISSUE_COMMENTS]} issue comments. "
                f"PR comments: {self.stats.own_pr_comments} on your own work, "
                f"{self.stats.other_pr_comments} on others' work, "
                f"{self.stats.unknown_author_pr_comments} with an unavailable author."
            ),
            facts={role.value: counts[role] for role in roles},
            records=[record for role in roles for record in self.stats.activities[role]],
        )
        self._add(
            key="collaborators",
            title="People whose work you reviewed",
            summary=f"You reviewed PRs from {len(self.stats.reviewed_authors)} "
            "identifiable human accounts other than yourself.",
            facts={"reviewed_authors": len(self.stats.reviewed_authors)},
            records=self.stats.activities[Activity.REVIEWED] if self.stats.reviewed_authors else [],
        )

    def _timeline_slides(self) -> None:
        events = self.stats.distinct_monthly_events
        volume = sum(events.values())
        self._add(
            key="timeline",
            title="Your activity through the year",
            summary=f"{volume} distinct recorded actions across "
            f"{sum(count > 0 for count in events.values())} active UTC months. "
            "Opening, merging, reviewing, and commenting are different actions, not a productivity score.",
            facts=events,
            records=self._all_records(),
        )
        peak = max(events.values(), default=0)
        months = sorted(month for month, count in events.items() if count == peak)
        self._add(
            key="busiest_month",
            title="Your busiest month" if len(months) == 1 else "Your busiest months",
            summary=f"{', '.join(months)}: {peak} distinct recorded actions each. Ties are retained.",
            facts=dict.fromkeys(months, peak),
            records=self._all_records() if peak else [],
        )

    def _focus_slides(self) -> None:
        role = Activity.AUTHORED
        breakdown = self.stats.breakdowns[role]
        records = self.stats.activities[role]
        self._add(
            key="home_territory",
            title="Your home territory",
            summary=self._home_summary(),
            facts=breakdown.primary_topics,
            records=records,
        )
        self._add(
            key="surfaces",
            title="Where you built",
            summary=self._focus_summary(counts=breakdown.surfaces, label="Primary surfaces")
            + ". GUI backend work can also be Python; surface is not language.",
            facts=breakdown.surfaces,
            records=records,
        )
        self._add(
            key="artifacts",
            title="What kind of work you did",
            summary=self._focus_summary(counts=breakdown.primary_artifacts, label="Primary artifacts")
            + ". Mixed and Unknown remain in the denominator; generated files do not outweigh substantive paths.",
            facts=breakdown.primary_artifacts,
            records=records,
        )
        self._add(
            key="intent_mix",
            title="Your change mix",
            summary=self._focus_summary(counts=breakdown.intents, label="Recognized change intent")
            + ". Intent comes from prefixes or explicit labels, not a judgment of impact.",
            facts=breakdown.intents,
            records=records,
        )

    def _chapter_slides(self) -> None:
        definitions = [
            ("fixes", "The fixer chapter", "FIX", None),
            ("features", "The feature chapter", "FEAT", None),
            ("tests", "The testing chapter", None, "Tests"),
            ("documentation", "The documentation chapter", None, "Documentation/examples"),
            ("infrastructure", "Behind-the-scenes work", "MAINT", "Configuration/CI"),
        ]
        for key, title, intent, artifact in definitions:
            records = [
                record
                for record in self.stats.activities[Activity.AUTHORED]
                if self.stats.classifications[record.item_ref].intent == intent
                or artifact in self.stats.classifications[record.item_ref].artifacts
            ]
            description = f"{len(records)} PRs you opened"
            if intent and artifact:
                description += f" with {intent} intent or {artifact.lower()} changes"
            elif intent:
                description += f" classified as {intent}"
            else:
                description += f" touching {str(artifact).lower()}"
            self._add(key=key, title=title, summary=description + ".", facts={"prs": len(records)}, records=records)
        self._dataset_slide()
        for key, role, title in (
            ("review_topics", Activity.REVIEWED, "Areas you reviewed"),
            ("issue_topics", Activity.ISSUES, "Topics you surfaced in issues"),
        ):
            groups = self.stats.breakdowns[role].topics
            qualifier = (
                "These describe reviewed PR scope, not every line inspected."
                if role == Activity.REVIEWED
                else ("Issue topics use labels and conservative title inference; Unknown remains visible.")
            )
            self._add(
                key=key,
                title=title,
                summary=self._distribution({topic: len(refs) for topic, refs in groups.items()})
                + ". Topic groups overlap. "
                + qualifier,
                facts={topic: len(refs) for topic, refs in groups.items()},
                records=self.stats.activities[role],
            )

    def _closing_slides(self) -> None:
        selected: dict[str, Evidence] = {}
        for role in (Activity.AUTHORED, Activity.LANDED, Activity.MERGED, Activity.ISSUES, Activity.REVIEWED):
            for topic in sorted(self.stats.breakdowns[role].topics):
                refs = set(self.stats.breakdowns[role].topics[topic])
                record = next((record for record in self.stats.activities[role] if record.ref in refs), None)
                if record is not None:
                    selected[record.ref] = record
        self._add(
            key="highlights",
            title="Highlights worth revisiting",
            summary="The earliest recorded item for each topic and role, deduplicated. "
            "Representative, not ranked by importance.",
            facts={"representative_items": len(selected)},
            records=list(selected.values()),
        )
        self._add(
            key="recap",
            title="Your PyRIT recap",
            summary=self._count_summary(),
            facts={role.value: count for role, count in self.stats.counts.items()},
            records=self._all_records(),
            required=True,
        )

    def _home_summary(self) -> str:
        breakdown = self.stats.breakdowns[Activity.AUTHORED]
        known = {topic: count for topic, count in breakdown.primary_topics.items() if topic not in {"Unknown", "Mixed"}}
        if not known:
            return "No confirmed primary topic for the PRs you opened. Mixed/Unknown evidence is retained."
        maximum = max(known.values())
        leaders = sorted(topic for topic, count in known.items() if count == maximum)
        certain = Counter(
            self.stats.classifications[record.item_ref].primary_topic
            for record in self.stats.activities[Activity.AUTHORED]
            if not self.stats.classifications[record.item_ref].inferred
        )
        word = (
            "Primarily"
            if breakdown.count >= 5 and len(leaders) == 1 and certain[leaders[0]] > breakdown.count / 2
            else ("Most frequent primary topic(s):")
        )
        return f"{word} {', '.join(leaders)}: {maximum} of {breakdown.count} PRs you opened. " + (
            "Small samples are descriptive, not contributor personas."
            if breakdown.count < 5
            else "Mixed/Unknown are included."
        )

    def _count_summary(self) -> str:
        counts = self.stats.counts
        return (
            f"{counts[Activity.AUTHORED]} PRs opened; {counts[Activity.LANDED]} authored PRs landed; "
            f"{counts[Activity.MERGED]} PRs credited to you as merger; {counts[Activity.ISSUES]} issues opened; "
            f"{counts[Activity.REVIEWED]} distinct PRs reviewed in {counts[Activity.REVIEWS]} submitted reviews."
        )

    def _all_records(self) -> list[Evidence]:
        return [record for records in self.stats.activities.values() for record in records]

    def _focus_summary(self, *, counts: dict[str, int], label: str) -> str:
        total = self.stats.counts[Activity.AUTHORED]
        majority = [
            name
            for name, count in counts.items()
            if name not in {"Mixed", "Unknown", "Generated/lock files"} and total >= 5 and count > total / 2
        ]
        prefix = f"Primarily {majority[0]} ({counts[majority[0]]} of {total} opened PRs). " if majority else ""
        return prefix + f"{label} of PRs you opened: " + self._distribution(counts)

    def _dataset_slide(self) -> None:
        records = [
            record
            for record in self.stats.activities[Activity.AUTHORED]
            if "Datasets" in self.stats.classifications[record.item_ref].topics
        ]
        content = sum("Dataset content" in self.stats.classifications[record.item_ref].artifacts for record in records)
        implementation = sum(
            "Product code" in self.stats.classifications[record.item_ref].artifacts for record in records
        )
        self._add(
            key="datasets",
            title="The dataset chapter",
            summary=f"{len(records)} PRs you opened touched datasets: {content} included dataset content "
            f"and {implementation} included implementation code. These groups can overlap.",
            facts={"prs": len(records), "content_prs": content, "implementation_prs": implementation},
            records=records,
        )

    def _add(
        self,
        *,
        key: str,
        title: str,
        summary: str,
        facts: dict[str, int],
        records: list[Evidence],
        required: bool = False,
    ) -> None:
        if not records and not required:
            self.omitted[key] = OmittedSlide(
                type=key, reason="No supporting contributor activity in the selected period."
            )
            return
        self.slides[key] = Slide(
            type=key,
            title=title,
            summary=summary,
            facts=facts,
            evidence_refs=sorted({record.ref for record in records}),
        )

    @staticmethod
    def _distribution(counts: dict[str, int]) -> str:
        return "; ".join(
            f"{name}: {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ) or ("no classified activity")
