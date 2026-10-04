# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal

from build_scripts.pyrit_wrapped.models import Activity, Actor, Contribution
from build_scripts.pyrit_wrapped.snapshot import same_actor

if TYPE_CHECKING:
    from build_scripts.pyrit_wrapped.models import Evidence, Snapshot


class ContributorCredits:
    MAINTAINERS = (
        "romanlutz",
        "richlundeen",
        "hannahwestra25",
        "varunj-msft",
        "jsong468",
        "behnam-o",
        "adrian-gavrila",
        "jbolor21",
        "nina-msft",
        "bashirpartovi",
        "ValbuenaVC",
        "fdubut",
        "spencrr",
    )

    def __init__(self, *, snapshot: Snapshot, activities: dict[Activity, list[Evidence]]) -> None:
        self.snapshot = snapshot
        self.activities = activities
        self.rows: dict[str, Contribution] = {}
        self.aliases: dict[str, str] = {}
        self.unknown: Counter[str] = Counter()

    def calculate(self) -> tuple[list[Contribution], dict[str, int]]:
        items = {item.ref: item for item in self.snapshot.items}
        for role, field in (
            (Activity.AUTHORED, "opened_prs"),
            (Activity.LANDED, "merged_prs"),
            (Activity.ISSUES, "opened_issues"),
        ):
            for record in self.activities[role]:
                self._credit(actor=items[record.item_ref].author, field=field)
        reviews = {review.ref: review for review in self.snapshot.reviews}
        for record in self.activities[Activity.REVIEWS]:
            self._credit(actor=reviews[record.ref].author, field="submitted_reviews")
        comments = {comment.ref: comment for comment in self.snapshot.comments}
        for role in (Activity.INLINE, Activity.PR_COMMENTS, Activity.ISSUE_COMMENTS):
            for record in self.activities[role]:
                self._credit(actor=comments[record.ref].author, field="comments")
        return (
            sorted(
                self.rows.values(),
                key=lambda row: (-row.merged_prs, -row.submitted_reviews, row.actor.login.casefold()),
            ),
            dict(self.unknown),
        )

    def _credit(self, *, actor: Actor | None, field: str) -> None:
        if actor is None or actor.is_deleted or actor.type not in {"User", "Bot"}:
            self.unknown[field] += 1
            return
        key = f"db:{actor.database_id}" if actor.database_id is not None else f"node:{actor.id}"
        key = self.aliases.get(key, key)
        if key not in self.rows:
            existing = next((identity for identity, row in self.rows.items() if same_actor(row.actor, actor)), None)
            if existing is not None:
                self.aliases[key] = existing
                key = existing
        if key not in self.rows:
            group: Literal["maintainers", "contributors", "bots"] = (
                "bots"
                if actor.type == "Bot"
                else (
                    "maintainers"
                    if actor.login.casefold() in {name.casefold() for name in self.MAINTAINERS}
                    else "contributors"
                )
            )
            self.rows[key] = Contribution(actor=actor, group=group)
        row = self.rows[key]
        setattr(row, field, getattr(row, field) + 1)
