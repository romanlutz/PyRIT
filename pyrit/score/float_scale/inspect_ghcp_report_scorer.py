# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Replay an original Inspect cyber judgment; never execute its task or grader."""

from __future__ import annotations

import hashlib

from pyrit.models import (
    ComponentIdentifier,
    ContentEntryScorable,
    ContentScorable,
    Scorable,
    Score,
    ScoreStatus,
    ScoringExpectation,
)
from pyrit.models.inspect_ghcp import InspectGhcpReport, InspectGhcpStatus
from pyrit.score.float_scale.submission_report_scorer import SubmissionReportScorer


class InspectGhcpReportScorer(SubmissionReportScorer):
    """Map one retained original Inspect Score to an immutable PyRIT Score."""

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"report_sha256": self._report_sha256, "contract": "inspect-ghcp-v1"})

    def prepare_unpersisted_score(self, *, report: InspectGhcpReport) -> Score:
        """
        Prepare the one content-linked score for atomic report publication.

        Returns:
            Score: A complete or explicitly undetermined unpersisted Score.

        Raises:
            ValueError: If report content disagrees with the bound digest.
        """
        if report.sha256() != self._report_sha256:
            raise ValueError("Inspect GHCP report differs from its bound canonical digest.")
        return self._build_score(report=report, scorable=ContentScorable(value=report.canonical_json()))

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        if not isinstance(scorable, (ContentScorable, ContentEntryScorable)):
            raise TypeError("Inspect GHCP scores must name retained report content.")
        content = await self._resolve_content_async(scorable)
        if hashlib.sha256(content.value.encode("utf-8")).hexdigest() != self._report_sha256:
            raise ValueError("Inspect GHCP report content does not match its retained digest.")
        report = InspectGhcpReport.model_validate_json(content.value)
        if report.canonical_json() != content.value:
            raise ValueError("Inspect GHCP report content was not canonical JSON.")
        return [self._build_score(report=report, scorable=scorable, expectation=expectation)]

    def _build_score(
        self,
        *,
        report: InspectGhcpReport,
        scorable: ContentScorable | ContentEntryScorable,
        expectation: ScoringExpectation | None = None,
    ) -> Score:
        value = (
            report.judgment.numeric_value if report.status is InspectGhcpStatus.COMPLETED and report.judgment else None
        )
        return Score(
            score_type="float_scale",
            score_value=str(value) if value is not None else None,
            status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
            score_rationale=(
                report.judgment.explanation
                if value is not None and report.judgment and report.judgment.explanation
                else "Inspect original score is undetermined or required cyber evidence is incomplete."
            ),
            scorable=scorable,
            scorer_class_identifier=self.get_identifier(),
            scored_expectation=expectation,
            score_metadata={
                "run_id": report.run_id,
                "report_sha256": report.sha256(),
                "task_kind": report.task_kind.value,
                "inspect_task": report.task_name,
                "publication_role": "final_run_result",
            },
        )
