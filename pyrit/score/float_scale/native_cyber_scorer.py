# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from pyrit.models.native_cyber import NativeCyberReport, NativeCyberStatus
from pyrit.score.float_scale.submission_report_scorer import SubmissionReportScorer


class NativeCyberReportScorer(SubmissionReportScorer):
    """Read an acquired native report once; never execute a task, artifact, or grader."""

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"report_sha256": self._report_sha256, "contract": "native-cyber-v1"})

    def prepare_unpersisted_score(
        self, *, report: NativeCyberReport, expectation: ScoringExpectation | None = None
    ) -> Score:
        """
        Build a validated native score for a caller-owned atomic persistence operation.

        Args:
            report (NativeCyberReport): The canonical report to score.
            expectation (ScoringExpectation | None): The criterion recorded with the score.

        Returns:
            Score: An unpersisted score anchored to the report content.

        Raises:
            ValueError: If the report does not match this scorer's retained digest.
        """
        if report.sha256() != self._report_sha256:
            raise ValueError("Native report is not canonical retained content.")
        return self._build_score(
            report=report,
            scorable=ContentScorable(value=report.canonical_json()),
            expectation=expectation,
        )

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        if not isinstance(scorable, (ContentScorable, ContentEntryScorable)):
            raise TypeError("Native cyber results must name retained report content, not a final sentence.")
        content = await self._resolve_content_async(scorable)
        if hashlib.sha256(content.value.encode()).hexdigest() != self._report_sha256:
            raise ValueError("Native report content does not match its retained digest.")
        report = NativeCyberReport.model_validate_json(content.value)
        if report.sha256() != self._report_sha256:
            raise ValueError("Native report is not canonical retained content.")
        return [self._build_score(report=report, scorable=scorable, expectation=expectation)]

    def _build_score(
        self,
        *,
        report: NativeCyberReport,
        scorable: ContentScorable | ContentEntryScorable,
        expectation: ScoringExpectation | None,
    ) -> Score:
        """
        Construct the same verdict for both persistence paths.

        Returns:
            Score: The content-anchored verdict.
        """
        value = report.judgment.value if report.status is NativeCyberStatus.COMPLETED and report.judgment else None
        return Score(
            score_type="float_scale",
            score_value=str(value) if value is not None else None,
            status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
            score_rationale=(
                report.judgment.rationale if report.judgment else "No complete original judgment was acquired."
            ),
            scorable=scorable,
            scorer_class_identifier=self.get_identifier(),
            scored_expectation=expectation,
            score_metadata={
                "run_id": report.run_id,
                "report_sha256": report.sha256(),
                "binding": report.binding_name,
                "run_status": report.status.value,
                "simulated": int(report.simulated) if report.simulated is not None else "unknown",
                "publication_role": "final_run_result",
                "publication_boundary": "pyrit_score_commit",
            },
        )
