# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from build_scripts.pyrit_wrapped.html_deck import HtmlDeck
from build_scripts.pyrit_wrapped.models import Activity, Evidence, Snapshot, Stats, Story, WrappedError


class MarkdownReport:
    _LABELS = {
        Activity.AUTHORED: "PRs opened",
        Activity.LANDED: "Contributor-authored PRs merged",
        Activity.MERGED: "PRs credited to contributor as merger",
        Activity.ISSUES: "Issues opened",
        Activity.REVIEWED: "Distinct PRs formally reviewed",
        Activity.REVIEWS: "Submitted reviews",
        Activity.INLINE: "Inline review comments",
        Activity.REVIEW_BODIES: "Nonempty review summary comments",
        Activity.PR_COMMENTS: "PR discussion comments",
        Activity.ISSUE_COMMENTS: "Issue discussion comments",
        Activity.PR_CLOSED: "PRs closed (including merges)",
        Activity.ISSUES_CLOSED: "Issues closed",
        Activity.SHIPPED: "PR merge commits in the release range",
    }

    def __init__(self, *, stats: Stats, story: Story) -> None:
        self.stats = stats
        self.story = story

    def render(self) -> str:
        period = self.stats.period
        suffix = " (year to date)" if period.year_to_date else ""
        lines = [
            f"# {self._heading()}{suffix}",
            "",
            (
                f"Repository: {self.stats.repository}. UTC events from {period.start.isoformat()} "
                f"up to, but not including, {period.cutoff.isoformat()}."
            ),
            (
                f"Collection completed: {self.stats.collected_at.isoformat()}. "
                f"Earliest source response used: {self.stats.earliest_response_at.isoformat()}."
            ),
            "",
            "## Counts",
            "",
            "| Activity | Count |",
            "|---|---:|",
            *[self._count_row(role) for role in Activity],
            "",
            "Roles overlap. These counts must not be summed into a productivity score.",
            "Comment totals distinguish inline comments, nonempty review bodies, and discussion comments.",
            (
                "Titles, labels, file paths, and item states reflect the fetched snapshot, "
                "not reconstructed year-end metadata."
            ),
            "Only visible public GitHub records are available. Deleted/inaccessible activity cannot be reconstructed.",
            "Sources may have been fetched at different times; API collection is not an atomic GitHub snapshot.",
            "",
        ]
        lines.extend(self._slides())
        if self.stats.release is not None:
            release = self.stats.release
            lines.extend(
                [
                    "## Release provenance",
                    "",
                    f"Base: `{release.base.tag}` at `{release.base.commit}`.",
                    f"Head: `{release.head.tag}` at `{release.head.commit}`.",
                    (
                        "Collaboration counts use publication dates. Shipped PRs use reachable merge commits; "
                        "LOC and file topics use the complete tag-to-tag tree diff, including direct commits."
                    ),
                    (
                        "Contributor roles describe identified PR/review/issue accounts, "
                        "not a census of every commit author."
                    ),
                    "",
                ]
            )
        lines.extend(self._monthly())
        lines.extend(
            [
                "## Evidence",
                "",
                "See [activity.md](activity.md) for complete topic-grouped activity lists.",
                "See [songs.md](songs.md) for candidate tracks by slide type; no audio is included.",
                "",
            ]
        )
        lines.extend(self._caveats())
        return "\n".join(lines).rstrip() + "\n"

    def render_activity(self) -> str:
        heading = f"# {self._heading()}: activity evidence"
        return "\n".join([heading, "", *self._activity_lists()]).rstrip() + "\n"

    def _heading(self) -> str:
        if self.stats.release is not None:
            return f"PyRIT release wrapped: {escape_text(self.stats.release.label)}"
        return (
            f"@{escape_text(self.stats.contributor.login)}: PyRIT Wrapped {self.stats.period.year}"
            if self.stats.contributor
            else "PyRIT Wrapped"
        )

    def render_songs(self) -> str:
        lines = [
            "# Song selections and suggestions",
            "",
            (
                "Selected tracks reflect the user's choices; suggested tracks remain undecided. "
                "Selection does not mean a recording is supplied or licensed. No audio is downloaded or bundled."
            ),
            (
                "A roughly 10-second cue can match each slide. Supply and usage rights remain a user decision; "
                "short duration alone does not grant permission."
            ),
            "",
            "| Slide | Status | Track | Why it fits |",
            "|---|---|---|---|",
        ]
        for slide in self.story.slides:
            lines.extend(
                f"| {escape_text(slide.title)} | {'Selected' if candidate.selected else 'Suggested'} | "
                f"{escape_text(candidate.title)} - "
                f"{escape_text(candidate.artist)} | {escape_text(candidate.rationale)} |"
                for candidate in slide.song_candidates
            )
        return "\n".join(lines) + "\n"

    def _count_row(self, role: Activity) -> str:
        value = self.stats.counts[role]
        display = str(value) if value is not None else "Unavailable / not applicable"
        return f"| {self._LABELS[role]} | {display} |"

    def _slides(self) -> list[str]:
        lines = ["## Slide summaries", ""]
        for index, slide in enumerate(self.story.slides, 1):
            lines.extend(
                [
                    f"### {index}. {escape_text(slide.title)}",
                    "",
                    escape_text(slide.summary),
                    "",
                    f"Type: `{slide.type}`.",
                    "",
                ]
            )
        lines.extend(["## Omitted slide types", ""])
        lines.extend(f"- `{slide.type}`: {escape_text(slide.reason)}" for slide in self.story.omitted)
        lines.append("")
        return lines

    def _monthly(self) -> list[str]:
        lines = [
            "## Monthly activity",
            "",
            (
                "Distinct actions deduplicate the same PR merge credited under both author and merger roles. "
                "Review bodies are part of their submitted review, not additional actions. "
                "Months and timestamps use UTC."
            ),
            "",
            "| Month | Distinct recorded actions |",
            "|---|---:|",
        ]
        lines.extend(f"| {month} | {count} |" for month, count in self.stats.distinct_monthly_events.items())
        lines.append("")
        return lines

    def _activity_lists(self) -> list[str]:
        lines = [
            "## Activity grouped by topic",
            "",
            "Topic groups overlap; each group deduplicates source records.",
            "",
        ]
        for role in Activity:
            if role not in self.stats.breakdowns:
                continue
            records = {record.ref: record for record in self.stats.activities[role]}
            lines.extend([f"### {self._LABELS[role]} ({len(records)})", ""])
            for topic, refs in self.stats.breakdowns[role].topics.items():
                lines.extend([f"#### {escape_text(topic)} ({len(refs)})", ""])
                lines.extend(self._record_line(records[ref]) for ref in refs)
                lines.append("")
            if not records:
                lines.extend(["No activity recorded in the selected period.", ""])
        return lines

    def _caveats(self) -> list[str]:
        lines = [
            "## Classification and limitations",
            "",
            (
                "PR components/artifacts use changed paths where coverage is complete. "
                "Incomplete files fall back to explicitly inferred metadata. "
                "Inline-comment topics use comment paths where present; "
                "reviewed-PR topics describe PR scope, not inspected lines."
            ),
            "Issues use labels and conservative title inference. Unknown/Mixed remain in denominators.",
            (
                "Primary categories require a strict majority of substantive changed paths. "
                "Interpretive 'primarily' wording requires at least five opened PRs and a confirmed strict majority."
            ),
            "",
        ]
        inferred = sorted(ref for ref, classification in self.stats.classifications.items() if classification.inferred)
        if inferred:
            lines.extend(["Metadata-inferred classifications: " + ", ".join(f"`{ref}`" for ref in inferred) + ".", ""])
        if self.stats.warnings:
            lines.extend(
                ["### Collection warnings", "", *[f"- {escape_text(value)}" for value in self.stats.warnings], ""]
            )
        lines.extend(
            [
                "## Review checkpoint",
                "",
                (
                    "Review the attribution, topic groups, summaries, and omitted types before building HTML. "
                    "Track selections and undecided suggestions are in songs.md; "
                    "recording supply and rights remain separate."
                ),
            ]
        )
        return lines

    def _record_line(self, record: Evidence) -> str:
        parsed = urlparse(record.url)
        if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith("/microsoft/PyRIT/"):
            raise WrappedError(f"Refusing an unsafe source link for {record.ref}.")
        url = record.url.replace("(", "%28").replace(")", "%29")
        line = f"- [{escape_text(record.title)}]({url}) (`{record.ref}`, {record.event_at.isoformat()})"
        if record.ref in self.stats.public_comment_proofs:
            identifier = record.ref.split(":")[1]
            endpoint = f"https://api.github.com/repos/microsoft/PyRIT/pulls/comments/{identifier}"
            line += f" ([publication verified publicly]({endpoint}); review metadata unavailable)"
        return line


def escape_text(value: str) -> str:
    value = html.escape(value.replace("\r", " ").replace("\n", " "), quote=True)
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", value)


def write_reports(*, snapshot: Snapshot, stats: Stats, story: Story, output_dir: Path) -> Path:
    destination = output_dir.resolve()
    if destination.exists():
        raise WrappedError(f"Report directory already exists: {destination}. Choose a new --output-dir.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = MarkdownReport(stats=stats, story=story)
    summary = report.render()
    activity = report.render_activity()
    deck = HtmlDeck(stats=stats, story=story).render()
    with tempfile.TemporaryDirectory(prefix=".wrapped-report-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for filename, model in (("snapshot.json", snapshot), ("stats.json", stats), ("story.json", story)):
            (staging / filename).write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (staging / "summary.md").write_text(summary, encoding="utf-8")
        (staging / "activity.md").write_text(activity, encoding="utf-8")
        (staging / "songs.md").write_text(report.render_songs(), encoding="utf-8")
        (staging / "index.html").write_text(deck, encoding="utf-8")
        os.replace(staging, destination)
    return destination
