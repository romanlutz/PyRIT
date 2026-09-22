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
from pyrit.models.submission import SubmissionReportStatus
from pyrit.models.submission_v2 import RetainedSubmissionReportV2
from pyrit.score.float_scale.submission_report_scorer import SubmissionReportScorer


class SubmissionReportScorerV2(SubmissionReportScorer):
    """Read acquired v2 content without changing its provenance or invoking a service."""

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={"report_sha256": self._report_sha256, "contract": "strict-submission-v2"}
        )

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        if not isinstance(scorable, (ContentScorable, ContentEntryScorable)):
            raise TypeError("Submission reports must be content-anchored, never final-message scores.")
        content = await self._resolve_content_async(scorable)
        digest = hashlib.sha256(content.value.encode("utf-8")).hexdigest()
        if digest != self._report_sha256:
            raise ValueError("Retained v2 report content was modified or its identity does not match.")
        report = RetainedSubmissionReportV2.model_validate_json(content.value)
        if report.sha256() != digest:
            raise ValueError("Retained v2 report is not its canonical immutable representation.")
        binding = report.report
        value = binding.last_valid_grade if binding and report.status is SubmissionReportStatus.COMPLETED else None
        metadata: dict[str, str | int | float] = {
            "contract_version": "strict-submission-v2",
            "evidence_label": report.evidence_label,
            "mode": report.mode.value,
            "simulated": int(report.simulated),
            "report_sha256": digest,
            "run_id": report.run_id,
            "run_status": report.status.value,
            "selected_submission_id": (binding.selected_submission_id or "") if binding else "",
            "publication_role": "final_run_result",
            "publication_boundary": "pyrit_score_commit",
            "local_cleanup": report.local_cleanup.value,
        }
        if binding and binding.last_valid_grade is not None:
            metadata["retained_last_valid_grade"] = binding.last_valid_grade
        return [
            Score(
                score_type="float_scale",
                score_value=str(value) if value is not None else None,
                status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
                score_rationale="Latest valid acquired v2 submission; incomplete outcomes remain undetermined.",
                scorer_class_identifier=self.get_identifier(),
                scorable=scorable,
                score_metadata=metadata,
                scored_expectation=expectation,
            )
        ]
