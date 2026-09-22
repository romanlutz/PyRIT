# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
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
from pyrit.models.submission import RetainedSubmissionReport, SubmissionReportStatus
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer


class SubmissionReportScorer(FloatScaleScorer):
    """Replay an acquired offline report, without any evaluator, tool, or provider invocation."""

    def __init__(self, *, report_sha256: str) -> None:
        """
        Bind scoring to a retained report digest.

        Args:
            report_sha256 (str): SHA256 recorded when the immutable report was retained.

        Raises:
            ValueError: If the digest is not a lowercase SHA256.
        """
        if len(report_sha256) != 64 or any(character not in "0123456789abcdef" for character in report_sha256):
            raise ValueError("The scorer requires the retained report's lowercase SHA256.")
        self._report_sha256 = report_sha256
        super().__init__()

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={"report_sha256": self._report_sha256, "contract": "strict-submission-v1"}
        )

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        if not isinstance(scorable, (ContentScorable, ContentEntryScorable)):
            raise TypeError("Submission reports must be content-anchored, never final-message scores.")
        content = await self._resolve_content_async(scorable)
        digest = hashlib.sha256(content.value.encode("utf-8")).hexdigest()
        if digest != self._report_sha256:
            raise ValueError("Retained submission report content was modified or its identity does not match.")
        report = RetainedSubmissionReport.model_validate_json(content.value)
        if report.sha256() != digest:
            raise ValueError("Retained submission report is not its canonical immutable representation.")
        binding = report.report
        completed = report.status is SubmissionReportStatus.COMPLETED
        value = binding.last_valid_grade if completed and binding else None
        metadata: dict[str, str | int | float] = {
            "evidence_label": report.evidence_label,
            "report_sha256": digest,
            "run_id": report.run_id,
            "run_status": report.status.value,
            "selected_submission_id": binding.selected_submission_id or "" if binding else "",
            "publication_role": "final_run_result",
            "publication_boundary": "pyrit_score_commit",
        }
        if binding and binding.last_valid_grade is not None:
            metadata["retained_last_valid_grade"] = binding.last_valid_grade
        return [
            Score(
                score_type="float_scale",
                score_value=str(value) if value is not None else None,
                status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
                score_rationale="Latest valid acquired submission; overall incomplete states remain undetermined.",
                scorer_class_identifier=self.get_identifier(),
                scorable=scorable,
                score_metadata=metadata,
                scored_expectation=expectation,
            )
        ]

    async def _resolve_content_async(self, scorable: Scorable) -> ContentScorable:
        if isinstance(scorable, ContentScorable) and scorable.data_type == "text":
            return scorable
        if not isinstance(scorable, ContentEntryScorable) or scorable.data_type != "text":
            raise TypeError("SubmissionReportScorer requires retained JSON text, not a final assistant message.")
        contents = await asyncio.to_thread(self._memory.get_scorable_content, content_ids=[scorable.content_id])
        hashes = await asyncio.to_thread(self._memory.get_scorable_content_hashes, content_ids=[scorable.content_id])
        content = contents.get(scorable.content_id)
        if content is None or hashes.get(scorable.content_id) != self._report_sha256:
            raise ValueError("The stored report or immutable content identity is missing or changed.")
        return content
